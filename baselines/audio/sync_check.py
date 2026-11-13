#!/usr/bin/env python3
"""Audio/mouth sync check: RMS envelope of baselines/audio/<id>.m4a at 30 Hz vs mouth opening tracked on real.mp4.
Uses eval/cache tracking when present (read-only; never writes into ROOT), else tracks in-process.
  sync_check.py id [id ...]   -> prints r at lag 0, best lag (frames, + = audio lags the mouth) and r there."""
import os, sys, subprocess, hashlib, json
import numpy as np
E = __import__('os').path.expandvars('$WORK/eval')
sys.path.insert(0, E)
import metrics as M  # noqa
OUT = os.path.dirname(os.path.abspath(__file__))


def mouth(real_p):
    st = os.stat(real_p)
    key = hashlib.md5(f'{real_p}|{st.st_size}|{st.st_mtime_ns}|track'.encode()).hexdigest()
    p = os.path.join(M.CACHE, f'track_{key}.npz')
    if os.path.exists(p): return np.load(p)['open'], 'cache'
    return M.track(M.read_video(real_p)[0])['open'], 'tracked now'


def env30(m4a, n):
    r = subprocess.run(['ffmpeg', '-v', 'error', '-i', m4a, '-ac', '1', '-ar', '48000', '-f', 'f32le', '-'], capture_output=True)
    x = np.frombuffer(r.stdout, np.float32).astype(np.float64)
    hop = 1600; k = min(n, len(x) // hop)
    e = np.sqrt((x[:k * hop].reshape(k, hop) ** 2).mean(1) + 1e-10)
    return 20 * np.log10(e)


def lagcorr(a, b, maxlag=10):
    ok = ~np.isnan(a) & ~np.isnan(b); a = np.where(ok, a, np.nanmean(a)); b = np.where(ok, b, np.nanmean(b))
    a = (a - a.mean()) / (a.std() + 1e-9); b = (b - b.mean()) / (b.std() + 1e-9)
    cs = {}
    for L in range(-maxlag, maxlag + 1):  # L>0: audio[t+L] vs mouth[t]  => audio lags mouth by L frames
        x = a[max(0, L):len(a) + min(0, L)]; y = b[max(0, -L):len(b) + min(0, -L)]
        cs[L] = float(np.mean(x * y))
    best = max(cs, key=cs.get); return cs[0], best, cs[best], cs


res = {}
for cid in sys.argv[1:]:
    mo, how = mouth(f'{M.ROOT if hasattr(M, "ROOT") else os.path.dirname(E)}/data/pairs/{cid}/real.mp4')
    en = env30(os.path.join(OUT, cid + '.m4a'), len(mo)); mo = mo[:len(en)]
    r0, bl, rb, cs = lagcorr(en, mo)
    res[cid] = dict(r_lag0=round(r0, 3), best_lag_frames=bl, best_lag_ms=round(bl * 1000 / 30, 1), r_best=round(rb, 3),
                    curve={k: round(v, 3) for k, v in cs.items()}, mouth_from=how)
    print(f'{cid}: r(lag0)={r0:.3f} best lag={bl:+d} frames ({bl * 33.3:+.0f} ms) r={rb:.3f}  [mouth: {how}]')
    print('   curve', ' '.join(f'{k:+d}:{v:.2f}' for k, v in cs.items()))
json.dump(res, open(os.path.join(OUT, os.environ.get('SYNC_OUT', 'sync_check.json')), 'w'), indent=1)
