import argparse, os, json, glob, time
import torch, numpy as np
from common import read_video, VideoWriter, to_t, t2im

ap = argparse.ArgumentParser()
ap.add_argument('--kind', required=True, choices=['turbo', 'turbo_t', 'unet'])
ap.add_argument('--ckpt', required=True)
ap.add_argument('--variant', required=True)
ap.add_argument('--data', default='/root/data')
ap.add_argument('--drive', default='/root/drive')
ap.add_argument('--outroot', default='/root/out')
ap.add_argument('--lat_ema', type=float, default=0.0, help='turbo: blend previous x0 latent (0=off)')
ap.add_argument('--train_samples', default='')
ap.add_argument('--only', default='', help='comma list: heldout,drive,train')
a = ap.parse_args()
dev = 'cuda'
if a.kind == 'unet':
    from unet import UNetG
    G = UNetG().to(dev).eval(); G.load_state_dict(torch.load(a.ckpt)); G = G.half()
else:
    from turbo import Turbo
    G = Turbo(temporal=(a.kind == 'turbo_t')).to(dev).eval(); G.load_trainable(torch.load(a.ckpt)); G = G.half()

m = json.load(open(a.data + '/manifest.json'))
jobs = []
only = set(a.only.split(',')) if a.only else {'heldout', 'drive', 'train'}
if 'heldout' in only:
    jobs += [(c['id'], f"{a.data}/pairs/{c['id']}/render.mp4") for c in m['clips'] if c['split'] == 'heldout']
if 'drive' in only:
    jobs += [(os.path.basename(p)[:-4], p) for p in sorted(glob.glob(a.drive + '/*.mp4'))]
if 'train' in only and a.train_samples:
    jobs += [(i + '_trainsample', f"{a.data}/pairs/{i}/render.mp4") for i in a.train_samples.split(',')]
od = f'{a.outroot}/{a.variant}'
os.makedirs(od, exist_ok=True)
t0 = time.time(); nf = 0
for cid, path in jobs:
    frames, fps = read_video(path)
    w = VideoWriter(f'{od}/{cid}.mp4', fps)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        if a.kind == 'unet' or (a.kind == 'turbo' and a.lat_ema == 0):
            for i in range(0, len(frames), 8):
                x = torch.stack([to_t(f) for f in frames[i:i + 8]]).to(dev).half()
                for im in t2im(G(x).float()): w.write(im)
        else:
            prev = None
            for f in frames:
                x = to_t(f)[None].to(dev).half()
                if a.kind == 'turbo_t':
                    out = G(x, prev); prev = G.encode(out)  # re-encode output: matches training (prev = VAE latent of a real frame)
                else:  # latent EMA on single-frame model: re-decode the blended latent
                    z = G.encode(x); skips = G.vae.encoder.current_down_blocks
                    tt = torch.full((1,), 999, device=dev, dtype=torch.long)
                    pred = G.unet(z, tt, encoder_hidden_states=G.emb.to(z.dtype)).sample
                    ab = G.ab
                    x0 = (a_ := ab ** 0.5) * z - (1 - ab) ** 0.5 * pred if G.pred_type == 'v_prediction' else (z - (1 - ab) ** 0.5 * pred) / ab ** 0.5
                    if prev is not None: x0 = (1 - a.lat_ema) * x0 + a.lat_ema * prev
                    prev = x0
                    G.vae.decoder.incoming_skip_acts = skips
                    out = G.vae.decode(x0 / G.sf).sample.clamp(-1, 1)
                w.write(t2im(out.float())[0])
    w.close(); nf += len(frames)
    print(cid, len(frames), f'{time.time() - t0:.0f}s', flush=True)
print('INFER_DONE', a.variant, nf, 'frames', f'{time.time() - t0:.0f}s')
