#!/usr/bin/env python3
"""Build DiffSynth training set: 16 fps windows of (render -> real) from train pairs.
Out: wan/sync/train/{seg/*.mp4, metadata.csv}. Windows of 81 frames at offsets 0 and 48 (covering 129 frames @16fps)."""
import json, pathlib, subprocess, csv, sys
ROOT = pathlib.Path(__import__('os').path.expandvars('$WORK'))
OUT = ROOT / 'wan/sync/train'; SEG = OUT / 'seg'; SEG.mkdir(parents=True, exist_ok=True)
PROMPT = ("a real person talking to the camera, photorealistic webcam video, natural skin texture, real hair, "
          "real clothing, natural indoor lighting, head and shoulders, sharp focus")
W = int(sys.argv[1]) if len(sys.argv) > 1 else 81
offsets = [0, 128 - W] if W < 128 else [0]
man = json.loads((ROOT / 'data/manifest.json').read_text())
rows = []
import concurrent.futures as cf
JOBS=[]
for c in man['clips']:
    if c['split'] != 'train': continue
    d = ROOT / 'data/pairs' / c['id']
    if not (d / 'render.mp4').exists() or not (d / 'real.mp4').exists(): continue
    for off in offsets:
        names = {}
        for k in ('render', 'real'):
            o = SEG / f"{c['id']}_{off}_{k}.mp4"; names[k] = o
            if not o.exists():
                # 16 fps resample (nearest frame), then take W frames starting at off
                JOBS.append(['nice', 'ffmpeg', '-v', 'error', '-y', '-i', str(d / f'{k}.mp4'), '-vf',
                                f"fps=16,select='between(n\\,{off}\\,{off + W - 1})',setpts=N/16/TB", '-r', '16',
                                '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', str(o)])
        rows.append({'video': f"seg/{names['real'].name}", 'vace_video': f"seg/{names['render'].name}", 'prompt': PROMPT})
with cf.ThreadPoolExecutor(32) as ex: list(ex.map(lambda c: subprocess.run(c, check=True), JOBS))
with open(OUT / 'metadata.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['video', 'vace_video', 'prompt']); w.writeheader(); w.writerows(rows)
print(len(rows), 'samples')
