"""Crude temporal stability: mean |frame_t - frame_{t-1}| (0-255 scale) per clip, plus a high-frequency version
(diff of 3x3-blur-removed residuals), for render, real and every output variant over the heldout clips."""
import json, glob, os, sys, numpy as np, cv2
from common import read_video
data, outroot = '/root/data', '/root/out'
m = json.load(open(data + '/manifest.json'))
held = [c['id'] for c in m['clips'] if c['split'] == 'heldout']
drive = sorted(os.path.basename(p)[:-4] for p in glob.glob('/root/drive/*.mp4'))


def stats(path):
    f, _ = read_video(path)
    a = np.stack(f).astype(np.float32)
    d = np.abs(np.diff(a, axis=0)).mean()
    hf = a - np.stack([cv2.GaussianBlur(x, (0, 0), 1.5) for x in a])
    dh = np.abs(np.diff(hf, axis=0)).mean()
    return float(d), float(dh)


res = {}
srcs = {'render': lambda c: f'{data}/pairs/{c}/render.mp4', 'real': lambda c: f'{data}/pairs/{c}/real.mp4'}
for v in sorted(os.listdir(outroot)):
    srcs[v] = (lambda v: lambda c: f'{outroot}/{v}/{c}.mp4')(v)
for name, fn in srcs.items():
    for group, ids in (('heldout', held), ('drive', drive)):
        vals = []
        for c in ids:
            p = fn(c) if not (group == 'drive' and name in ('render', 'real')) else (f'/root/drive/{c}.mp4' if name == 'render' else None)
            if p and os.path.exists(p): vals.append(stats(p))
        if vals:
            res[f'{name}/{group}'] = {'n': len(vals), 'mean_abs_frame_diff': round(float(np.mean([v[0] for v in vals])), 3),
                                      'mean_abs_hf_frame_diff': round(float(np.mean([v[1] for v in vals])), 3)}
            print(name, group, res[f'{name}/{group}'], flush=True)
json.dump(res, open('/root/flicker.json', 'w'), indent=1)
