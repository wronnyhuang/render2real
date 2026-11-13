import os, json, random, glob, subprocess, math
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F

FR = os.environ.get('FRAMES', '/root/frames')


def prep_frames(pairs_dir, manifest, out=FR, splits=('train',)):
    """Extract jpg frames for every clip of the given splits (idempotent)."""
    m = json.load(open(manifest))
    ids = [c['id'] for c in m['clips'] if c['split'] in splits]
    jobs = []
    for cid in ids:
        d = os.path.join(out, cid)
        if os.path.exists(os.path.join(d, 'ok')):
            continue
        os.makedirs(d, exist_ok=True)
        cmd = (f"ffmpeg -v error -y -i {pairs_dir}/{cid}/render.mp4 -q:v 2 {d}/r_%04d.jpg && "
               f"ffmpeg -v error -y -i {pairs_dir}/{cid}/real.mp4 -q:v 2 {d}/p_%04d.jpg && touch {d}/ok")
        jobs.append(subprocess.Popen(cmd, shell=True))
        if len(jobs) >= 12:
            jobs.pop(0).wait()
    for j in jobs:
        j.wait()
    return [i for i in ids if os.path.exists(os.path.join(out, i, 'ok'))]


def rd(p):
    im = cv2.imread(p)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def to_t(im):
    return torch.from_numpy(im).permute(2, 0, 1).float().div(127.5).sub(1)


def appearance_aug(im, strong=True):
    """Degrade the RENDER input so the model is not brittle to another renderer. im uint8 HxWx3."""
    im = im.astype(np.float32)
    H, W = im.shape[:2]
    if random.random() < 0.5:  # resolution degradation
        f = random.uniform(0.35, 0.9)
        small = cv2.resize(im, (max(16, int(W * f)), max(16, int(H * f))), interpolation=cv2.INTER_AREA)
        im = cv2.resize(small, (W, H), interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]))
    if random.random() < 0.4:
        s = random.uniform(0.3, 1.6)
        im = cv2.GaussianBlur(im, (0, 0), s)
    if strong and random.random() < 0.25:  # flatter / cartoonier shading
        im = cv2.bilateralFilter(im, 9, random.uniform(20, 60), random.uniform(3, 9))
    if strong and random.random() < 0.15:  # posterize
        lv = random.choice([8, 12, 16, 24])
        im = np.round(im / (256 / lv)) * (256 / lv)
    if random.random() < 0.7:  # mild colour jitter (no hue: colours must carry through)
        b = random.uniform(-15, 15); c = random.uniform(0.85, 1.15); sat = random.uniform(0.8, 1.2)
        g = im.mean(2, keepdims=True)
        im = g + (im - g) * sat
        im = (im - 128) * c + 128 + b
    if random.random() < 0.4:
        im = im + np.random.randn(*im.shape).astype(np.float32) * random.uniform(1, 6)
    if strong and random.random() < 0.15:  # gamma
        im = 255 * (np.clip(im, 0, 255) / 255) ** random.uniform(0.8, 1.25)
    return np.clip(im, 0, 255).astype(np.uint8)


def geo(ims, size):
    """Same random affine (scale/translate/flip) for every image in the list."""
    H, W = ims[0].shape[:2]
    s = random.uniform(0.9, 1.12); tx = random.uniform(-0.06, 0.06) * W; ty = random.uniform(-0.06, 0.06) * H
    M = cv2.getRotationMatrix2D((W / 2, H / 2), random.uniform(-4, 4), s); M[0, 2] += tx; M[1, 2] += ty
    flip = random.random() < 0.5
    out = []
    for im in ims:
        im = cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if flip: im = im[:, ::-1]
        if size != W: im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
        out.append(np.ascontiguousarray(im))
    return out


class PairDS(torch.utils.data.Dataset):
    def __init__(self, ids, size=384, prev=False, length=10 ** 7, n=240):
        self.ids, self.size, self.prev, self.length, self.n = ids, size, prev, length, n

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        while True:
            cid = random.choice(self.ids)
            t = random.randint(4, self.n)
            d = os.path.join(FR, cid)
            fs = [f'{d}/r_{t:04d}.jpg', f'{d}/p_{t:04d}.jpg']
            if self.prev:
                fs.append(f'{d}/p_{t - random.choice([1, 1, 2, 3]):04d}.jpg')
            if all(os.path.exists(f) for f in fs):
                break
        ims = [rd(f) for f in fs]
        ims = geo(ims, self.size)
        clean = ims[0]
        ims[0] = appearance_aug(ims[0])
        out = {'render': to_t(ims[0]), 'clean': to_t(clean), 'real': to_t(ims[1])}
        if self.prev:
            q = ims[2].astype(np.float32)
            if random.random() < 0.5: q = cv2.GaussianBlur(q, (0, 0), random.uniform(0.3, 1.2))
            if random.random() < 0.5: q = (q - 128) * random.uniform(0.9, 1.1) + 128 + random.uniform(-10, 10)
            q = np.clip(q + np.random.randn(*q.shape).astype(np.float32) * random.uniform(0, 5), 0, 255).astype(np.uint8)
            out['prev'] = to_t(q)
            out['has_prev'] = torch.tensor(0.0 if random.random() < 0.15 else 1.0)
        return out


# ---------------- discriminator: multi-scale spectral-norm PatchGAN, conditional -------------
def SN(m): return nn.utils.spectral_norm(m)


class Patch(nn.Module):
    def __init__(self, cin=6, ch=64, n=4):
        super().__init__()
        L = [SN(nn.Conv2d(cin, ch, 4, 2, 1)), nn.LeakyReLU(0.2, True)]
        c = ch
        for i in range(1, n):
            c2 = min(ch * 2 ** i, 512)
            L += [SN(nn.Conv2d(c, c2, 4, 2 if i < n - 1 else 1, 1)), nn.LeakyReLU(0.2, True)]
            c = c2
        L += [SN(nn.Conv2d(c, 1, 4, 1, 1))]
        self.net = nn.Sequential(*L)

    def forward(self, x): return self.net(x)


class MultiD(nn.Module):
    def __init__(self, cin=6):
        super().__init__()
        self.ds = nn.ModuleList([Patch(cin), Patch(cin), Patch(cin, n=3)])

    def forward(self, x):
        outs = []
        for i, d in enumerate(self.ds):
            outs.append(d(x if i == 0 else F.avg_pool2d(x, 2 ** i)))
        return outs


def d_hinge(D, real, fake):
    l = 0
    for a, b in zip(D(real), D(fake)):
        l = l + F.relu(1 - a).mean() + F.relu(1 + b).mean()
    return l


def g_hinge(D, fake):
    return sum(-o.mean() for o in D(fake))


# ---------------- video io ----------------
def read_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fr = []
    while True:
        ok, im = cap.read()
        if not ok: break
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if im.shape[0] != 384 or im.shape[1] != 384:
            im = cv2.resize(im, (384, 384), interpolation=cv2.INTER_AREA)
        fr.append(im)
    return fr, fps


class VideoWriter:
    def __init__(self, path, fps, size=384):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{size}x{size}',
                                   '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '17',
                                   '-preset', 'fast', path], stdin=subprocess.PIPE)

    def write(self, im):
        self.p.stdin.write(np.ascontiguousarray(im).tobytes())

    def close(self):
        self.p.stdin.close(); self.p.wait()


def t2im(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
