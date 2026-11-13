import argparse, os, time, json
import torch, torch.nn.functional as F, lpips, cv2, numpy as np
from common import PairDS, MultiD, d_hinge, g_hinge, prep_frames, t2im
from unet import UNetG

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='/root/data')
ap.add_argument('--out', required=True)
ap.add_argument('--bs', type=int, default=8)
ap.add_argument('--lr', type=float, default=2e-4)
ap.add_argument('--minutes', type=float, default=60)
ap.add_argument('--gan_start', type=int, default=2000)
ap.add_argument('--w_gan', type=float, default=0.3)
ap.add_argument('--w_lpips', type=float, default=2.0)
ap.add_argument('--workers', type=int, default=6)
ap.add_argument('--resume', default='')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
ids_now = lambda: prep_frames(a.data + '/pairs', a.data + '/manifest.json')
ids = ids_now()
G = UNetG().to(dev); D = MultiD(6).to(dev)
if a.resume: G.load_state_dict(torch.load(a.resume))
print('G params M', sum(p.numel() for p in G.parameters()) / 1e6, 'clips', len(ids), flush=True)
lp = lpips.LPIPS(net='vgg').to(dev).eval().requires_grad_(False)
optG = torch.optim.AdamW(G.parameters(), lr=a.lr, betas=(0.5, 0.99))
optD = torch.optim.AdamW(D.parameters(), lr=a.lr, betas=(0.5, 0.99))
mk = lambda ids: iter(torch.utils.data.DataLoader(PairDS(ids), batch_size=a.bs, num_workers=a.workers, drop_last=True, prefetch_factor=4))
it = mk(ids)
t0 = time.time(); step = 0; acc = {}; log = open(a.out + '/log.txt', 'a')
while time.time() - t0 < a.minutes * 60:
    if step and step % 2000 == 0:
        n = ids_now()
        if len(n) != len(ids): ids = n; it = mk(ids); print('refreshed', len(ids), flush=True)
    b = next(it)
    x = b['render'].to(dev); y = b['real'].to(dev); xc = b['clean'].to(dev)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = G(x)
        l1 = F.l1_loss(out.float(), y)
        llp = lp(out.float(), y).mean()
        loss = l1 + a.w_lpips * llp
        lg = torch.zeros((), device=dev)
        if step >= a.gan_start:
            lg = g_hinge(D, torch.cat([xc, out], 1)); loss = loss + a.w_gan * lg
    optG.zero_grad(set_to_none=True); loss.backward(); optG.step()
    ld = torch.zeros(())
    if step >= a.gan_start - 500:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            ld = d_hinge(D, torch.cat([xc, y], 1), torch.cat([xc, out.detach()], 1))
        optD.zero_grad(set_to_none=True); ld.backward(); optD.step()
    for k, v in (('l1', l1), ('lpips', llp), ('g', lg), ('d', ld)): acc[k] = acc.get(k, 0) + float(v)
    step += 1
    if step % 100 == 0:
        msg = f'step {step} t {time.time() - t0:.0f}s ' + ' '.join(f'{k} {v / 100:.4f}' for k, v in acc.items())
        print(msg, flush=True); log.write(msg + '\n'); log.flush(); acc = {}
    if step % 1000 == 0:
        grid = np.concatenate([np.concatenate(list(t2im(z[:4].float())), 0) for z in (x, out, y)], 1)
        cv2.imwrite(f'{a.out}/sample_{step:06d}.jpg', cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        torch.save(G.state_dict(), a.out + '/last.pt')
torch.save(G.state_dict(), a.out + '/last.pt'); torch.save(D.state_dict(), a.out + '/D.pt')
json.dump({'steps': step, 'seconds': time.time() - t0, 'bs': a.bs, 'clips': len(ids)}, open(a.out + '/train_meta.json', 'w'))
print('DONE', step, flush=True)
