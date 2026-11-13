#!/usr/bin/env python3
"""Aggregate per-clip timing rows (latency_*.jsonl) into wan/latency.json."""
import json, glob, collections, statistics
D = __import__('os').path.expandvars('$WORK/wan/')
rows = []
for f in sorted(glob.glob(D + 'latency_*.jsonl')):
    seen = set()
    for l in open(f):
        r = json.loads(l); k = (r['variant'], r['clip'], r['gpu'])
        if k in seen: continue
        seen.add(k); rows.append(r)
g = collections.defaultdict(list)
for r in rows: g[(r['variant'], r['gpu'], r['steps'], r['cfg'])].append(r)
summary = []
for (v, gpu, steps, cfg), rs in sorted(g.items()):
    spf = statistics.median(r['s_per_frame'] for r in rs)
    nfe = steps * (2 if cfg != 1 else 1)
    summary.append(dict(variant=v, gpu=gpu, steps=steps, cfg=cfg, dit_evals_per_step=2 if cfg != 1 else 1, total_dit_evals=nfe,
                        clips=len(rs), frames_per_clip=rs[0]['frames_generated'], resolution='384x384', fps_generated=16,
                        median_wall_s_per_clip=round(statistics.median(r['wall_s'] for r in rs), 1),
                        median_s_per_generated_frame=round(spf, 4), generated_fps=round(1 / spf, 2),
                        realtime_factor_vs_16fps=round(16 * spf, 2), peak_mem_gb=max(r['peak_mem_gb'] for r in rs),
                        lora=rs[0]['lora'], causvid=rs[0]['causvid'], control=rs[0]['control']))
out = dict(note=('s_per_generated_frame = wall time of the whole pipe() call (text encode + VACE/VAE encode of the control video + '
                 'all denoising steps + VAE decode) / 129 frames generated at 16 fps; pipeline load time excluded; batch=1, bf16, '
                 'no torch.compile, no TeaCache, no FP8, single GPU. realtime_factor = seconds of compute per second of 16 fps video.'),
           summary=summary, rows=rows)
json.dump(out, open(D + 'latency.json', 'w'), indent=1)
for s in summary: print(s['variant'], s['gpu'], s['steps'], s['cfg'], s['median_s_per_generated_frame'], s['generated_fps'], s['clips'])
