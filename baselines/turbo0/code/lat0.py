"""Same method as render2real fast/code/latency.py: batch 1, 15 warmup + 60 timed, uint8 host frame -> GPU -> model -> uint8 host, synchronized."""
import json, time, sys
import torch, numpy as np
dev = 'cuda'
OUT = sys.argv[1] if len(sys.argv) > 1 else '/root/latency.json'
CK = sys.argv[2] if len(sys.argv) > 2 else '/root/turbo_lora.pt'
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
res = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
       'method': 'same as render2real fast/code/latency.py: batch 1, 384x384, steady state 15 warmup + 60 timed iterations (median), time.perf_counter around '
                 'uint8 HxWx3 host frame -> GPU -> VAE encode -> 1 UNet eval -> VAE decode -> uint8 host frame, cuda synchronized; fp16 weights + autocast fp16', 'rows': []}


def bench(name, fn, note, size=384):
    x = torch.from_numpy(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))
    def step():
        t = x.to(dev, non_blocking=True).permute(2, 0, 1)[None].half().div(127.5).sub(1)
        y = fn(t)
        return ((y.clamp(-1, 1) + 1) * 127.5).byte()[0].permute(1, 2, 0).cpu()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        t0 = time.time()
        for _ in range(15): step()
        warm = time.time() - t0
        torch.cuda.synchronize(); ts = []
        for _ in range(60):
            t0 = time.perf_counter(); step(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    ms = float(np.median(ts) * 1000)
    row = {'variant': name, 'res': size, 'ms_per_frame': round(ms, 2), 'fps': round(1000 / ms, 1),
           'p90_ms': round(float(np.percentile(ts, 90) * 1000), 2), 'warmup_s': round(warm, 1), 'note': note}
    print(row, flush=True); res['rows'].append(row); json.dump(res, open(OUT, 'w'), indent=1)


def compile_(M):
    M.unet = torch.compile(M.unet); M.vae.encoder = torch.compile(M.vae.encoder); M.vae.decoder = torch.compile(M.vae.decoder)


from base0 import BaseImg2Img
B = BaseImg2Img(0.5).to(dev).eval().half()
bench('base_sdturbo_img2img_1step', B, 'fp16, eager; strength 0.5 (t=499); cost is identical for 0.25/0.75 (same graph)')
compile_(B)
bench('base_sdturbo_img2img_1step', B, 'fp16 + torch.compile (unet, vae enc, vae dec)')
del B; torch.cuda.empty_cache()

from turbo import Turbo
from peft.tuners.tuners_utils import BaseTunerLayer
M = Turbo().to(dev).eval(); M.load_trainable(torch.load(CK))
for mod in M.modules():
    if isinstance(mod, BaseTunerLayer): mod.merge()
M = M.half()
bench('turbo_lora', M, 'fp16, eager, full SD-VAE w/ skips, LoRA merged (render2real latency.py path)')
compile_(M)
bench('turbo_lora', M, 'fp16 + torch.compile (unet, vae enc, vae dec), full SD-VAE w/ skips, LoRA merged')
res['notes'] = ['base_asft has the same graph as turbo_lora (LoRA merged, skip convs) so its latency equals turbo_lora.']
json.dump(res, open(OUT, 'w'), indent=1)
print('LAT_DONE')
