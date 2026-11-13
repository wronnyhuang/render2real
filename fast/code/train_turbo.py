import argparse, os, time, json, random, math
import torch, torch.nn.functional as F, lpips, cv2, numpy as np
from common import PairDS, MultiD, d_hinge, g_hinge, prep_frames, t2im
from turbo import Turbo

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='/root/data')
ap.add_argument('--out', required=True)
ap.add_argument('--temporal', action='store_true')
ap.add_argument('--bs', type=int, default=2)
ap.add_argument('--lr', type=float, default=5e-5)
ap.add_argument('--minutes', type=float, default=60)
ap.add_argument('--max_steps', type=int, default=100000)
ap.add_argument('--gan_start', type=int, default=500)
ap.add_argument('--w_gan', type=float, default=0.5)
ap.add_argument('--w_lpips', type=float, default=5.0)
ap.add_argument('--resume', default='')
ap.add_argument('--no_ckpt', action='store_true')
ap.add_argument('--refresh_every', type=int, default=1000, help='re-scan frames dir for new clips')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True


def ids_now():
    return prep_frames(a.data + '/pairs', a.data + '/manifest.json')


ids = ids_now()
print('train clips', len(ids), flush=True)
model = Turbo(temporal=a.temporal).to(dev)
model.set_ckpt(not a.no_ckpt)
model.train()
if a.resume:
    model.load_trainable(torch.load(a.resume))
D = MultiD(6).to(dev)
lp = lpips.LPIPS(net='vgg').to(dev).eval().requires_grad_(False)
optG = torch.optim.AdamW(model.trainable(), lr=a.lr, weight_decay=1e-2)
optD = torch.optim.AdamW(D.parameters(), lr=a.lr, betas=(0.5, 0.999))
print('trainable M', sum(p.numel() for p in model.trainable()) / 1e6, flush=True)


def loader(ids):
    ds = PairDS(ids, prev=a.temporal)
    return iter(torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=10, drop_last=True, persistent_workers=False,
                                            prefetch_factor=4))


it = loader(ids)
t0 = time.time(); step = 0; log = open(a.out + '/log.txt', 'a'); acc = {}
while step < a.max_steps and time.time() - t0 < a.minutes * 60:
    if step and step % a.refresh_every == 0:
        n = ids_now()
        if len(n) != len(ids):
            ids = n; it = loader(ids); print('refreshed clips', len(ids), flush=True)
    b = next(it)
    x = b['render'].to(dev, non_blocking=True); y = b['real'].to(dev); xc = b['clean'].to(dev)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        prev = None
        if a.temporal:
            with torch.no_grad():
                prev = model.encode(b['prev'].to(dev)) * b['has_prev'].to(dev).view(-1, 1, 1, 1)
        out = model(x, prev)
        l2 = F.mse_loss(out.float(), y.float())
        llp = lp(out.float(), y.float()).mean()
        loss = l2 + a.w_lpips * llp
        lg = torch.zeros((), device=dev)
        if step >= a.gan_start:
            lg = g_hinge(D, torch.cat([xc, out], 1))
            loss = loss + a.w_gan * lg
    optG.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.trainable(), 1.0)
    optG.step()
    ld = torch.zeros(())
    if step >= a.gan_start - 200:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            ld = d_hinge(D, torch.cat([xc, y], 1), torch.cat([xc, out.detach()], 1))
        optD.zero_grad(set_to_none=True); ld.backward(); optD.step()
    for k, v in (('l2', l2), ('lpips', llp), ('g', lg), ('d', ld)):
        acc[k] = acc.get(k, 0) + float(v)
    step += 1
    if step % 50 == 0:
        msg = f'step {step} t {time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated()/2**30:.1f}G ' + ' '.join(f'{k} {v / 50:.4f}' for k, v in acc.items())
        print(msg, flush=True); log.write(msg + '\n'); log.flush(); acc = {}
    if step % 500 == 0 or step == 20:
        grid = np.concatenate([np.concatenate(list(t2im(z.float())), 0) for z in (x, out, y)], 1)
        cv2.imwrite(f'{a.out}/sample_{step:06d}.jpg', cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    if step % 1000 == 0:
        torch.save(model.trainable_state(), a.out + '/last.pt')
torch.save(model.trainable_state(), a.out + '/last.pt')
json.dump({'steps': step, 'seconds': time.time() - t0, 'bs': a.bs, 'clips': len(ids), 'lr': a.lr, 'temporal': a.temporal},
          open(a.out + '/train_meta.json', 'w'))
print('DONE', step, flush=True)
