"""Seed: convert earlier stage S01..S12 (clip + _cal render, 540x960) into 384x384 8 s pairs."""
import json, os, subprocess, numpy as np
T37 = __import__('os').path.expandvars('$WORK')
D = __import__('os').path.expandvars('$WORK/data')
clips = {c['id']: c for c in json.load(open(f'{T37}/clips.json'))}
man = []
def face_boxes(fl, W, H):
    x = fl[..., 0] * W; y = fl[..., 1] * H
    return np.stack([np.nanmin(x, 1), np.nanmin(y, 1), np.nanmax(x, 1), np.nanmax(y, 1)], 1)
for i in range(1, 13):
    cid = f'S{i:02d}'; c = clips[cid]
    tr = np.load(f'{T37}/tracks/{cid}.npz'); W, H = int(tr['W']), int(tr['H'])
    fb = face_boxes(tr['face_lm'][0], W, H)
    pid = c['file_id'].split('_')[-1]
    for seg, f0 in enumerate((0, 300)):
        b = fb[f0:f0 + 240]; ok = ~np.isnan(b[:, 0])
        if ok.mean() < 0.9: print(cid, seg, 'low coverage', ok.mean()); continue
        fw = np.median(b[ok, 2] - b[ok, 0]); cx = np.median((b[ok, 0] + b[ok, 2]) / 2); cy = np.median((b[ok, 1] + b[ok, 3]) / 2)
        S = int(round(min(fw / 0.5, W, H))) // 2 * 2
        x0 = int(np.clip(cx - S / 2, 0, W - S)); y0 = int(np.clip(cy - 0.42 * S, 0, H - S))
        inside = ((b[ok, 0] >= x0) & (b[ok, 2] <= x0 + S) & (b[ok, 1] >= y0) & (b[ok, 3] <= y0 + S)).mean()
        pid_ = f'{cid}{"ab"[seg]}'; od = f'{D}/pairs/{pid_}'; os.makedirs(od, exist_ok=True)
        vf = f'trim=start_frame={f0}:end_frame={f0+240},setpts=PTS-STARTPTS,crop={S}:{S}:{x0}:{y0},scale=384:384:flags=lanczos'
        for src, name in ((f'{T37}/clips/{cid}.mp4', 'real'), (f'{T37}/renders/{cid}_cal.mp4', 'render')):
            subprocess.run(['ffmpeg', '-v', 'error', '-i', src, '-vf', vf, '-an', '-c:v', 'libx264', '-crf', '16', '-preset', 'fast',
                            '-pix_fmt', 'yuv420p', '-r', '30', '-y', f'{od}/{name}.mp4'], check=True)
        meta = dict(id=pid_, identity=pid, source='seamless', file_id=c['file_id'], src_video=c['video'],
                    src_start_s=round(c['start'] + f0 / 30, 3), frames=240, fps=30, size=384,
                    crop_in_540x960=[x0, y0, S, S], face_frac=round(float(fw / S), 3), face_inside=round(float(inside), 3),
                    origin='earlier stage seed (render at 540x960, upscaled crop)')
        json.dump(meta, open(f'{od}/meta.json', 'w'), indent=1)
        man.append(dict(id=pid_, identity=pid, split='train', source='seamless', frames=240))
        print(pid_, S, x0, y0, round(float(fw/S),2), inside)
json.dump(dict(clips=man, status='seed'), open(f'{D}/manifest.json', 'w'), indent=1)
