"""Zero-shot SD-Turbo (no LoRA, no trained weights) 1-step img2img baselines.
BaseImg2Img(strength): VAE-encode (mode) -> add a FIXED noise tensor at t = round(strength*1000)-1 -> ONE UNet call -> x0 -> plain SD-VAE decode."""
import torch, torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import AutoTokenizer, CLIPTextModel
from turbo import SD, PROMPT


class BaseImg2Img(nn.Module):
    def __init__(self, strength=0.5, seed=0, size=384):
        super().__init__()
        tok = AutoTokenizer.from_pretrained(SD, subfolder='tokenizer')
        te = CLIPTextModel.from_pretrained(SD, subfolder='text_encoder', variant='fp16', torch_dtype=torch.float32)
        with torch.no_grad():
            ids = tok(PROMPT, max_length=tok.model_max_length, padding='max_length', truncation=True, return_tensors='pt').input_ids
            self.register_buffer('emb', te(ids)[0].float(), persistent=False)
        del te
        sched = DDPMScheduler.from_pretrained(SD, subfolder='scheduler')
        self.pred_type = sched.config.prediction_type
        self.acp = sched.alphas_cumprod.clone()
        self.vae = AutoencoderKL.from_pretrained(SD, subfolder='vae', variant='fp16', torch_dtype=torch.float32)
        self.unet = UNet2DConditionModel.from_pretrained(SD, subfolder='unet', variant='fp16', torch_dtype=torch.float32)
        self.sf = self.vae.config.scaling_factor
        g = torch.Generator().manual_seed(seed)
        self.register_buffer('noise', torch.randn(1, 4, size // 8, size // 8, generator=g), persistent=False)
        self.set_strength(strength)
        self.unet_calls = 0
        self.unet.register_forward_pre_hook(lambda m, a, k=None: setattr(self, 'unet_calls', self.unet_calls + 1))

    def set_strength(self, s):
        self.t = int(round(s * 1000)) - 1
        self.ab = float(self.acp[self.t])

    def forward(self, c):
        B = c.shape[0]
        z0 = self.vae.encode(c).latent_dist.mode() * self.sf
        a = self.ab
        n = self.noise.to(z0.dtype).expand(B, -1, -1, -1)
        zt = a ** 0.5 * z0 + (1 - a) ** 0.5 * n
        t = torch.full((B,), self.t, device=c.device, dtype=torch.long)
        pred = self.unet(zt, t, encoder_hidden_states=self.emb.expand(B, -1, -1).to(zt.dtype)).sample
        if self.pred_type == 'v_prediction':
            x0 = a ** 0.5 * zt - (1 - a) ** 0.5 * pred
        else:
            x0 = (zt - (1 - a) ** 0.5 * pred) / a ** 0.5
        return self.vae.decode(x0 / self.sf).sample.clamp(-1, 1)
