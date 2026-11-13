"""pix2pix-turbo style one-step render->photo model (SD-Turbo + LoRA + skip-connected VAE).
Optional temporal mode: the UNet also sees the previous OUTPUT x0-latent (8-ch conv_in)."""
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import AutoTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

SD = 'stabilityai/sd-turbo'
PROMPT = 'a high quality photo of a real person talking to the camera, natural skin, photorealistic'


def enc_fwd(self, sample):
    sample = self.conv_in(sample)
    acts = []
    for b in self.down_blocks:
        acts.append(sample)
        sample = b(sample)
    sample = self.mid_block(sample)
    sample = self.conv_act(self.conv_norm_out(sample))
    sample = self.conv_out(sample)
    self.current_down_blocks = acts
    return sample


def dec_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    sample = self.mid_block(sample, latent_embeds)
    skips = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
    acts = self.incoming_skip_acts[::-1]
    for i, up in enumerate(self.up_blocks):
        sample = sample + skips[i](acts[i] * self.gamma)
        if self.training and torch.is_grad_enabled() and getattr(self, 'ckpt', False):
            sample = checkpoint(up, sample, latent_embeds, use_reentrant=False)
        else:
            sample = up(sample, latent_embeds)
    sample = self.conv_act(self.conv_norm_out(sample))
    return self.conv_out(sample)


UNET_T = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
          "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"]
VAE_T = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]


class Turbo(nn.Module):
    def __init__(self, temporal=False, r_unet=8, r_vae=4, dtype=torch.float32):
        super().__init__()
        self.temporal = temporal
        tok = AutoTokenizer.from_pretrained(SD, subfolder='tokenizer')
        te = CLIPTextModel.from_pretrained(SD, subfolder='text_encoder', variant='fp16', torch_dtype=torch.float32)
        with torch.no_grad():
            ids = tok(PROMPT, max_length=tok.model_max_length, padding='max_length', truncation=True, return_tensors='pt').input_ids
            self.register_buffer('emb', te(ids)[0].float(), persistent=False)
        del te
        sched = DDPMScheduler.from_pretrained(SD, subfolder='scheduler')
        self.pred_type = sched.config.prediction_type
        self.ab = float(sched.alphas_cumprod[999])
        vae = AutoencoderKL.from_pretrained(SD, subfolder='vae', variant='fp16', torch_dtype=torch.float32)
        unet = UNet2DConditionModel.from_pretrained(SD, subfolder='unet', variant='fp16', torch_dtype=torch.float32)
        vae.encoder.forward = enc_fwd.__get__(vae.encoder, type(vae.encoder))
        vae.decoder.forward = dec_fwd.__get__(vae.decoder, type(vae.decoder))
        d = vae.decoder
        d.skip_conv_1 = nn.Conv2d(512, 512, 1, bias=False)
        d.skip_conv_2 = nn.Conv2d(256, 512, 1, bias=False)
        d.skip_conv_3 = nn.Conv2d(128, 512, 1, bias=False)
        d.skip_conv_4 = nn.Conv2d(128, 256, 1, bias=False)
        for c in [d.skip_conv_1, d.skip_conv_2, d.skip_conv_3, d.skip_conv_4]:
            nn.init.constant_(c.weight, 1e-5)
        d.gamma = 1.0
        vae.requires_grad_(False); unet.requires_grad_(False)
        vae.add_adapter(LoraConfig(r=r_vae, init_lora_weights='gaussian', target_modules=VAE_T), adapter_name='vae_skip')
        unet.add_adapter(LoraConfig(r=r_unet, init_lora_weights='gaussian', target_modules=UNET_T))
        if temporal:  # 8-channel conv_in, new half zero-initialised
            old = unet.conv_in
            base = old.base_layer if hasattr(old, 'base_layer') else old
            new = nn.Conv2d(8, base.out_channels, 3, padding=1)
            with torch.no_grad():
                new.weight.zero_(); new.weight[:, :4] = base.weight; new.bias.copy_(base.bias)
            unet.conv_in = new
        else:
            base = unet.conv_in.base_layer if hasattr(unet.conv_in, 'base_layer') else unet.conv_in
        unet.conv_in.requires_grad_(True)
        for c in [d.skip_conv_1, d.skip_conv_2, d.skip_conv_3, d.skip_conv_4]:
            c.requires_grad_(True)
        self.vae, self.unet = vae, unet
        self.sf = vae.config.scaling_factor

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]

    def set_ckpt(self, on=True):
        self.vae.decoder.ckpt = on
        if on: self.unet.enable_gradient_checkpointing()

    def encode(self, x):
        return self.vae.encode(x).latent_dist.mode() * self.sf

    def forward(self, c, prev_lat=None, return_lat=False):
        B = c.shape[0]
        z = self.encode(c)
        skips = self.vae.encoder.current_down_blocks
        zin = z
        if self.temporal:
            if prev_lat is None: prev_lat = torch.zeros_like(z)
            zin = torch.cat([z, prev_lat], 1)
        t = torch.full((B,), 999, device=c.device, dtype=torch.long)
        pred = self.unet(zin, t, encoder_hidden_states=self.emb.expand(B, -1, -1).to(z.dtype)).sample
        a = self.ab
        if self.pred_type == 'v_prediction':
            x0 = a ** 0.5 * z - (1 - a) ** 0.5 * pred
        else:
            x0 = (z - (1 - a) ** 0.5 * pred) / a ** 0.5
        self.vae.decoder.incoming_skip_acts = skips
        out = self.vae.decode(x0 / self.sf).sample.clamp(-1, 1)
        return (out, x0) if return_lat else out

    def trainable_state(self):
        return {k: v.detach().cpu() for k, v in self.state_dict().items()
                if 'lora' in k or 'skip_conv' in k or 'conv_in' in k and 'unet' in k}

    def load_trainable(self, sd):
        missing = self.load_state_dict(sd, strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys[:5]
