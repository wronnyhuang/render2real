"""Small residual conv U-Net generator (speed anchor), ~20M params."""
import torch, torch.nn as nn, torch.nn.functional as F


class RB(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, c, 3, 1, 1), nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, c, 3, 1, 1))

    def forward(self, x): return x + self.b(x)


class UNetG(nn.Module):
    def __init__(self, ch=(48, 96, 192, 320, 448), nres=(1, 1, 2, 2, 3), cin=3):
        super().__init__()
        self.inc = nn.Conv2d(cin, ch[0], 3, 1, 1)
        self.down = nn.ModuleList(); self.enc = nn.ModuleList()
        for i, c in enumerate(ch):
            self.enc.append(nn.Sequential(*[RB(c) for _ in range(nres[i])]))
            if i < len(ch) - 1:
                self.down.append(nn.Conv2d(c, ch[i + 1], 4, 2, 1))
        self.up = nn.ModuleList(); self.dec = nn.ModuleList()
        for i in range(len(ch) - 1, 0, -1):
            self.up.append(nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(ch[i], ch[i - 1], 3, 1, 1)))
            self.dec.append(nn.Sequential(nn.Conv2d(2 * ch[i - 1], ch[i - 1], 1), *[RB(ch[i - 1]) for _ in range(nres[i - 1])]))
        self.out = nn.Sequential(nn.GroupNorm(8, ch[0]), nn.SiLU(), nn.Conv2d(ch[0], 3, 3, 1, 1))

    def forward(self, x):
        h = self.inc(x); hs = []
        for i, e in enumerate(self.enc):
            h = e(h)
            if i < len(self.down):
                hs.append(h); h = self.down[i](h)
        for u, d in zip(self.up, self.dec):
            h = u(h); h = d(torch.cat([h, hs.pop()], 1))
        return torch.tanh(self.out(h))
