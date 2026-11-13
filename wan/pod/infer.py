#!/usr/bin/env python3
"""Wan2.1-VACE-1.3B render->photoreal inference (runs on the pod).

Usage: python infer.py --variant NAME --control raw|depth|gray --clips id=path[,id=path...] [--steps 30] [--cfg 5]
       [--lora path] [--causvid path --causvid_alpha 1.0] [--denoise 1.0] [--window 0 --overlap 17]
Writes out/<variant>/<id>.mp4 (384x384, 16 fps, 128 frames = 8.0 s) and appends timing to out/latency.jsonl
"""
import argparse, json, os, time, pathlib
import numpy as np, torch, imageio.v3 as iio
from PIL import Image

os.environ.setdefault('DIFFSYNTH_DOWNLOAD_SOURCE', 'huggingface')
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

PROMPT = ("a real person talking to the camera, photorealistic webcam video, natural skin texture, real hair, "
          "real clothing, natural indoor lighting, head and shoulders, sharp focus")
NEG = ("cartoon, 3d render, cgi, plastic skin, flat shading, illustration, painting, anime, doll, blurry, "
       "low quality, deformed face, distorted mouth, extra teeth, static, subtitles, watermark")
FPS_OUT = 16
N_OUT = 128  # 8.0 s at 16 fps


def load_render(path, n=N_OUT + 1, size=384):
    fr = iio.imread(path)  # T,H,W,3 @30fps
    T = len(fr)
    idx = [min(T - 1, int(round(i * 30.0 / FPS_OUT))) for i in range(n)]
    return [Image.fromarray(fr[i]).resize((size, size), Image.BICUBIC) for i in idx]


_depth = None
def to_depth(frames):
    global _depth
    if _depth is None:
        from transformers import pipeline
        _depth = pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf', device=0)
    outs = [np.array(o['predicted_depth'].float().cpu()) if hasattr(o['predicted_depth'], 'cpu') else np.array(o['predicted_depth'])
            for o in _depth(frames, batch_size=16)]
    lo = min(np.percentile(o, 1) for o in outs); hi = max(np.percentile(o, 99) for o in outs)
    res = []
    for o in outs:
        d = np.clip((o - lo) / (hi - lo + 1e-6), 0, 1)
        d = (d * 255).astype(np.uint8)
        im = Image.fromarray(d).resize(frames[0].size, Image.BICUBIC).convert('RGB')
        res.append(im)
    return res


def to_gray(frames):
    return [f.convert('L').convert('RGB') for f in frames]


def build_pipe(args):
    M = '/w/models/Wan-AI/Wan2.1-VACE-1.3B/'
    mc = [ModelConfig(path=M + 'diffusion_pytorch_model.safetensors'),
          ModelConfig(path=M + 'models_t5_umt5-xxl-enc-bf16.pth'),
          ModelConfig(path=M + 'Wan2.1_VAE.pth')]
    pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cuda", model_configs=mc,
        tokenizer_config=ModelConfig(path='/w/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl'))
    if args.lora:
        pipe.load_lora(pipe.vace, args.lora, alpha=args.lora_alpha)
    if args.causvid:
        pipe.load_lora(pipe.dit, args.causvid, alpha=args.causvid_alpha)
    return pipe


def run_one(pipe, args, frames_render, prompt=PROMPT):
    if args.control == 'raw': ctrl = frames_render
    elif args.control == 'depth': ctrl = to_depth(frames_render)
    elif args.control == 'gray': ctrl = to_gray(frames_render)
    else: raise ValueError(args.control)
    kw = dict(prompt=prompt, negative_prompt=NEG, seed=args.seed, tiled=False, height=384, width=384,
              num_inference_steps=args.steps, cfg_scale=args.cfg, sigma_shift=args.shift, vace_scale=args.vace_scale)
    n = len(ctrl)
    if not args.window or args.window >= n:
        if args.denoise < 1.0: kw.update(input_video=frames_render, denoising_strength=args.denoise)
        return pipe(vace_video=ctrl, num_frames=n, **kw)
    # sliding windows with VACE extension: overlap frames come from previous output (mask 0 = keep)
    W, O = args.window, args.overlap
    out = []
    start = 0
    black = Image.new('L', (384, 384), 0); white = Image.new('L', (384, 384), 255)
    while True:
        end = min(start + W, n)
        seg_ctrl = list(ctrl[start:end]); mask = [white] * len(seg_ctrl)
        if out:
            k = len(out) - start  # frames already produced inside this window
            for j in range(k):
                seg_ctrl[j] = out[start + j]; mask[j] = black
        L = len(seg_ctrl)
        while L % 4 != 1: L -= 1
        seg_ctrl, mask = seg_ctrl[:L], [m.convert('RGB') for m in mask[:L]]
        seg = pipe(vace_video=seg_ctrl, vace_video_mask=mask, num_frames=L, **kw)
        out = out[:start] + list(seg)
        if start + L >= n: break
        start = start + L - O
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True); ap.add_argument('--control', default='raw')
    ap.add_argument('--clips', required=True); ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--cfg', type=float, default=5.0); ap.add_argument('--shift', type=float, default=5.0)
    ap.add_argument('--vace_scale', type=float, default=1.0); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lora'); ap.add_argument('--lora_alpha', type=float, default=1.0)
    ap.add_argument('--causvid'); ap.add_argument('--causvid_alpha', type=float, default=1.0)
    ap.add_argument('--denoise', type=float, default=1.0)
    ap.add_argument('--window', type=int, default=0); ap.add_argument('--overlap', type=int, default=17)
    ap.add_argument('--out', default='out'); ap.add_argument('--prompts')
    args = ap.parse_args()
    od = pathlib.Path(args.out) / args.variant; od.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); pipe = build_pipe(args); load_s = time.time() - t0
    gpu = torch.cuda.get_device_name(0)
    for item in args.clips.split(','):
        cid, path = item.split('=', 1)
        fr = load_render(path)
        torch.cuda.synchronize(); t = time.time()
        prompt = PROMPT
        if args.prompts:
            desc = json.load(open(args.prompts)).get(cid)
            if desc: prompt = desc[5:] if desc.startswith('FULL:') else desc + ', talking to the camera, ' + PROMPT.split('camera, ', 1)[1]
        vid = run_one(pipe, args, fr, prompt)
        torch.cuda.synchronize(); dt = time.time() - t
        vid = [np.array(v) for v in vid][:N_OUT]
        iio.imwrite(od / f'{cid}.mp4', np.stack(vid), fps=FPS_OUT, codec='libx264', pixelformat='yuv420p', macro_block_size=1,
                    output_params=['-crf', '18'])
        rec = dict(variant=args.variant, clip=cid, gpu=gpu, steps=args.steps, cfg=args.cfg, control=args.control,
                   lora=bool(args.lora), prompt=prompt[:200], causvid=bool(args.causvid), window=args.window, denoise=args.denoise,
                   frames_generated=len(fr), wall_s=round(dt, 2), s_per_frame=round(dt / len(fr), 4),
                   peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), pipe_load_s=round(load_s, 1))
        print(json.dumps(rec), flush=True)
        with open(pathlib.Path(args.out) / 'latency.jsonl', 'a') as f: f.write(json.dumps(rec) + '\n')


if __name__ == '__main__':
    main()
