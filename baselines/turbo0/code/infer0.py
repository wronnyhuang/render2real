import argparse, os, json, time, sys
import torch
from common import read_video, VideoWriter, to_t, t2im

ap = argparse.ArgumentParser()
ap.add_argument('--variants', default='base_s025,base_s050,base_s075,base_asft')
ap.add_argument('--data', default='/root/data')
ap.add_argument('--drive', default='/root/drive')
ap.add_argument('--jobs', default='/root/jobs.json')
ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()
dev = 'cuda'
jobs = json.load(open(a.jobs))  # [[set, out_id, render_path], ...]
if a.limit: jobs = jobs[:a.limit]
log = {}
for var in a.variants.split(','):
    torch.manual_seed(0)
    if var == 'base_asft':
        from turbo import Turbo
        G = Turbo().to(dev).eval().half()  # NO load_trainable: untrained skip convs (1e-5) / LoRA at init (B=0)
        calls = [0]
        G.unet.register_forward_pre_hook(lambda m, args, c=calls: c.__setitem__(0, c[0] + 1))
        get_calls = lambda: calls[0]
    else:
        from base0 import BaseImg2Img
        G = BaseImg2Img(strength=int(var[-3:]) / 100).to(dev).eval().half()
        get_calls = lambda: G.unet_calls
        print(var, 't =', G.t, 'pred', G.pred_type, flush=True)
    t0 = time.time(); nf = 0; nsamp = 0
    for st, cid, path in jobs:
        frames, fps = read_video(path)
        od = f'/root/{"out" if st == "test" else "val_out"}/{var}'
        w = VideoWriter(f'{od}/{cid}.mp4', fps)
        c0 = get_calls()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            for i in range(0, len(frames), 8):
                x = torch.stack([to_t(f) for f in frames[i:i + 8]]).to(dev).half()
                nsamp += x.shape[0]
                for im in t2im(G(x).float()): w.write(im)
        w.close(); nf += len(frames)
        ncall = get_calls() - c0
        assert ncall == (len(frames) + 7) // 8, (ncall, len(frames))  # one batched UNet forward per 8 frames = 1 eval/frame
        print(var, st, cid, len(frames), 'frames', ncall, 'unet fwd (batch 8)', f'{time.time() - t0:.0f}s', flush=True)
    log[var] = {'frames': nf, 'unet_forward_calls': get_calls(), 'unet_samples': nsamp, 'sec': round(time.time() - t0, 1)}
    del G; torch.cuda.empty_cache()
json.dump(log, open('/root/infer_log.json', 'w'), indent=1)
print('INFER_DONE', json.dumps(log))
