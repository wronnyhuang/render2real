import argparse, json, time, os, sys
import torch, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--turbo_ckpt', default='')
ap.add_argument('--turbo_t_ckpt', default='')
ap.add_argument('--unet_ckpt', default='')
ap.add_argument('--out', default='/root/latency.json')
ap.add_argument('--compile', type=int, default=1)
a = ap.parse_args()
dev = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
gpu = torch.cuda.get_device_name()
res = {'gpu': gpu, 'torch': torch.__version__, 'method': 'batch 1, steady state: 15 warmup + 60 timed iterations, '
       'time.perf_counter around uint8 HxWx3 host frame -> GPU -> model (incl. VAE/TAESD encode+decode) -> uint8 host frame, cuda synchronized', 'rows': []}


def bench(name, fn, size, note=''):
    x = torch.from_numpy(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))
    def step():
        t = x.to(dev, non_blocking=True).permute(2, 0, 1)[None].half().div(127.5).sub(1)
        y = fn(t)
        return ((y.clamp(-1, 1) + 1) * 127.5).byte()[0].permute(1, 2, 0).cpu()
    with torch.no_grad():
        for _ in range(15): step()
        torch.cuda.synchronize(); ts = []
        for _ in range(60):
            t0 = time.perf_counter(); step(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    ms = float(np.median(ts) * 1000)
    row = {'variant': name, 'res': size, 'ms_per_frame': round(ms, 2), 'fps': round(1000 / ms, 1), 'p90_ms': round(float(np.percentile(ts, 90) * 1000), 2),
           'measured': True, 'note': note}
    print(row, flush=True); res['rows'].append(row)
    json.dump(res, open(a.out, 'w'), indent=1)


def merge_lora(m):
    from peft.tuners.tuners_utils import BaseTunerLayer
    for mod in m.modules():
        if isinstance(mod, BaseTunerLayer): mod.merge()


# ---- unet_gan ----
if a.unet_ckpt:
    from unet import UNetG
    G = UNetG().to(dev).eval(); G.load_state_dict(torch.load(a.unet_ckpt))
    for s in (256, 384, 512):
        bench('unet_gan', lambda t: G(t.float()), s, 'fp32')
    Gh = G.half().to(memory_format=torch.channels_last)
    for s in (256, 384, 512):
        bench('unet_gan', lambda t: Gh(t), s, 'fp16 channels_last')
    if a.compile:
        Gc = torch.compile(Gh)
        bench('unet_gan', lambda t: Gc(t), 384, 'fp16 + torch.compile')
    del G, Gh
    torch.cuda.empty_cache()

# ---- turbo ----
for kind, ck in (('turbo_lora', a.turbo_ckpt), ('turbo_lora_temporal', a.turbo_t_ckpt)):
    if not ck: continue
    from turbo import Turbo
    M = Turbo(temporal=kind.endswith('temporal')).to(dev).eval(); M.load_trainable(torch.load(ck))
    merge_lora(M)
    prev = {}
    def full(t, M=M):
        if M.temporal:
            k = (t.shape[-1], t.dtype)
            out = M(t, prev.get(k)); prev[k] = M.encode(out); return out
        return M(t)
    for s in (256, 384, 512):
        Mf = M.float(); bench(kind, lambda t: full(t.float()), s, 'fp32 (tf32), full SD-VAE w/ skips, LoRA merged')
    M = M.half()
    for s in (256, 384, 512):
        with torch.autocast('cuda', dtype=torch.float16):
            bench(kind, full, s, 'fp16, full SD-VAE w/ skips, LoRA merged')
    if a.compile:
        M.unet = torch.compile(M.unet); M.vae.encoder = torch.compile(M.vae.encoder); M.vae.decoder = torch.compile(M.vae.decoder)
        t0 = time.time()
        with torch.autocast('cuda', dtype=torch.float16):
            bench(kind, full, 384, 'fp16 + torch.compile (unet, vae enc, vae dec), full SD-VAE w/ skips')
        print('compile+bench s', time.time() - t0)
    # TAESD swap (latency of the architecture only; this checkpoint was not trained with TAESD -> quality not valid)
    if kind == 'turbo_lora':
        from diffusers import AutoencoderTiny
        tae = AutoencoderTiny.from_pretrained('madebyollin/taesd', torch_dtype=torch.float16).to(dev).eval()
        unet = M.unet
        def tfwd(t):
            z = tae.encode(t).latents
            k = torch.full((1,), 999, device=dev, dtype=torch.long)
            pred = unet(z, k, encoder_hidden_states=M.emb.half()).sample
            x0 = (z - (1 - M.ab) ** 0.5 * pred) / M.ab ** 0.5
            return tae.decode(x0).sample
        for s in (256, 384, 512):
            with torch.autocast('cuda', dtype=torch.float16):
                bench('turbo_lora+TAESD', tfwd, s, 'fp16, TAESD enc+dec (no skips), unet compiled=%d; architecture latency only, NOT a trained variant' % a.compile)
        # UNet alone
        k = torch.full((1,), 999, device=dev, dtype=torch.long)
        for s in (384,):
            z = torch.randn(1, 4, s // 8, s // 8, device=dev, dtype=torch.float16)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                for _ in range(10): unet(z, k, encoder_hidden_states=M.emb.half())
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for _ in range(40): unet(z, k, encoder_hidden_states=M.emb.half())
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 40 * 1000
            res['rows'].append({'variant': 'sd-turbo UNet alone', 'res': s, 'ms_per_frame': round(ms, 2), 'fps': round(1000 / ms, 1), 'measured': True, 'note': 'fp16, compiled=%d' % a.compile})
            print(res['rows'][-1])
    del M; torch.cuda.empty_cache()
json.dump(res, open(a.out, 'w'), indent=1)
print('LAT_DONE')
