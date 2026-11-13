#!/usr/bin/env python3
"""render2real metrics.  Usage:
  metrics.py --track fast --variant v1 --clips S01a S02a muse_01 [--variant-dir DIR] [--workers 6]
  metrics.py --track fast --variant v1 --all          # every clip id found in the variant dir
Writes eval/results/<track>_<variant>.json  (per-clip + mean).
Clip id resolution: ROOT/data/pairs/<id>/{render,real}.mp4  or  ROOT/drive/<id>.mp4 (render only, no real).
"""
import argparse, json, os, sys, hashlib, time
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = __import__('os').path.expandvars('$WORK/models')
CACHE = os.path.join(HERE, 'cache'); os.makedirs(CACHE, exist_ok=True)
GRID_FPS = 30.0
N = 240


# ---------------------------------------------------------------- video io
def read_video(path, size=384):
    """-> frames (T,H,W,3 BGR uint8) resampled by timestamp onto a 30 fps grid covering the clip, plus native fps."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        if f.shape[0] != size or f.shape[1] != size: f = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
        fr.append(f)
    if not fr: return None, fps
    dur = len(fr) / fps
    n = int(round(dur * GRID_FPS))
    idx = np.clip(np.round((np.arange(n) / GRID_FPS) * fps).astype(int), 0, len(fr) - 1)
    return np.stack([fr[i] for i in idx]), fps


def fit_len(a, n):
    if a is None: return None
    if len(a) >= n: return a[:n]
    return np.concatenate([a, np.repeat(a[-1:], n - len(a), 0)])


# ---------------------------------------------------------------- face tracking (MediaPipe)
def _euler(M):
    from scipy.spatial.transform import Rotation as Rot
    M = M.copy(); bad = np.isnan(M[..., 0, 0]); M[bad] = np.eye(4)
    e = Rot.from_matrix(M[..., :3, :3].reshape(-1, 3, 3)).as_euler('xyz', degrees=True).reshape(M.shape[:-2] + (3,))
    out = np.stack([e[..., 1], -e[..., 0], e[..., 2]], -1)  # yaw, pitch, roll
    out[bad] = np.nan
    return out


def track(frames):
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision as V
    lmk = V.FaceLandmarker.create_from_options(V.FaceLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=os.path.join(MODELS, 'face_landmarker.task')),
        running_mode=V.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=False,
        output_facial_transformation_matrixes=True, min_face_detection_confidence=0.3, min_tracking_confidence=0.3))
    T = len(frames); H, W = frames.shape[1:3]
    open_ = np.full(T, np.nan); width = np.full(T, np.nan); eye = np.full(T, np.nan)
    M = np.full((T, 4, 4), np.nan)
    for t, f in enumerate(frames):
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        r = lmk.detect_for_video(img, int(round(t * 1000 / GRID_FPS)))
        if not r.face_landmarks: continue
        a = np.array([[p.x * W, p.y * H] for p in r.face_landmarks[0]])
        iod = np.linalg.norm(a[33] - a[263]) + 1e-6
        open_[t] = np.linalg.norm(a[13] - a[14]) / iod
        width[t] = np.linalg.norm(a[61] - a[291]) / iod
        eye[t] = (np.linalg.norm(a[159] - a[145]) + np.linalg.norm(a[386] - a[374])) / 2 / iod
        if r.facial_transformation_matrixes: M[t] = np.array(r.facial_transformation_matrixes[0])
    lmk.close()
    return dict(open=open_, width=width, eye=eye, pose=_euler(M))


def cached(path, tag, fn):
    st = os.stat(path)
    key = hashlib.md5(f'{path}|{st.st_size}|{st.st_mtime_ns}|{tag}'.encode()).hexdigest()
    p = os.path.join(CACHE, f'{tag}_{key}.npz')
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True); return {k: z[k] for k in z.files}
    out = fn()
    np.savez(p, **out)
    return out


# ---------------------------------------------------------------- signal helpers
def corr_lag(a, b, maxlag=10):
    """Pearson r at lag 0, best-lag r and best lag (frames; + means `a` lags `b`... see below)."""
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 30: return None, None, None
    a = np.where(ok, a, np.nanmean(a[ok])); b = np.where(ok, b, np.nanmean(b[ok]))
    a = (a - a.mean()) / (a.std() + 1e-9); b = (b - b.mean()) / (b.std() + 1e-9)
    cs = {}
    for L in range(-maxlag, maxlag + 1):
        x = a[max(0, L):len(a) + min(0, L)]; y = b[max(0, -L):len(b) + min(0, -L)]
        cs[L] = float(np.mean(x * y))
    L = max(cs, key=cs.get)  # L>0: a[t+L] matches b[t] -> output a is DELAYED by L frames relative to b
    return round(cs[0], 4), round(cs[L], 4), L


def blink_events(eye, drop=0.55, maxlen=12):
    ok = ~np.isnan(eye)
    if ok.sum() < 30: return None
    base = np.nanpercentile(eye, 80); b = np.where(ok, eye < drop * base, False)
    ev = []; t = 0
    while t < len(b):
        if b[t]:
            s = t
            while t < len(b) and b[t]: t += 1
            if t - s <= maxlen: ev.append((s + t) / 2)
        t += 1
    return ev


def blink_f1(ref, out, tol=4):
    if ref is None or out is None: return None
    if not ref and not out: return dict(f1=1.0, ref=0, out=0)
    hit = sum(any(abs(g - q) <= tol for q in out) for g in ref); prec = sum(any(abs(g - q) <= tol for g in ref) for q in out)
    rec = hit / len(ref) if ref else 0.0; pr = prec / len(out) if out else 0.0
    f1 = 2 * rec * pr / (rec + pr) if rec + pr else 0.0
    return dict(f1=round(f1, 3), recall=round(rec, 3), precision=round(pr, 3), ref=len(ref), out=len(out))


def motion_vs(ref, out, name):
    """compare face-signal dict `out` to reference `ref` (both from track())."""
    r = dict(name=name)
    for k in ('open', 'width'):
        c0, cb, L = corr_lag(out[k], ref[k])
        r[f'{k}_corr'] = c0; r[f'{k}_corr_bestlag'] = cb; r['lag_frames' if k == 'open' else 'lag_frames_width'] = L
        if k == 'open' and L is not None: r['lag_ms'] = round(L * 1000 / GRID_FPS, 1)
    pe = np.abs(out['pose'] - ref['pose'])
    ok = ~np.isnan(pe).any(1)
    if ok.sum() >= 10:
        # remove constant offset? no: report raw and mean-centred (rig offsets are arbitrary between render and real)
        d = out['pose'][ok] - ref['pose'][ok]
        for i, n in enumerate(('yaw', 'pitch', 'roll')):
            r[f'{n}_mae'] = round(float(np.abs(d[:, i]).mean()), 3)
            r[f'{n}_mae_centered'] = round(float(np.abs(d[:, i] - np.median(d[:, i])).mean()), 3)
        r['pose_mae'] = round(float(np.abs(d).mean()), 3)
        r['pose_mae_centered'] = round(float(np.mean([np.abs(d[:, i] - np.median(d[:, i])).mean() for i in range(3)])), 3)
        # pose correlation (shape of motion, offset/scale free)
        cs = [corr_lag(out['pose'][:, i], ref['pose'][:, i])[0] for i in range(3)]
        cs = [c for c in cs if c is not None]
        r['pose_corr_mean'] = round(float(np.mean(cs)), 4) if cs else None
    r['blink'] = blink_f1(blink_events(ref['eye']), blink_events(out['eye']))
    return r


# ---------------------------------------------------------------- identity (ArcFace)
_FA = None
def face_app():
    global _FA
    if _FA is None:
        from insightface.app import FaceAnalysis
        import onnxruntime as ort
        _FA = FaceAnalysis(name='buffalo_l', root=os.path.join(HERE, 'models'), providers=['CPUExecutionProvider'],
                           allowed_modules=['detection', 'recognition'])
        _FA.prepare(ctx_id=-1, det_size=(384, 384), det_thresh=0.3)
    return _FA


def embeddings(frames, step=4):
    app = face_app(); embs = []; idx = []
    for t in range(0, len(frames), step):
        fs = app.get(frames[t])
        if fs:
            f = max(fs, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            e = f.normed_embedding; embs.append(e / (np.linalg.norm(e) + 1e-9)); idx.append(t)
    n_try = len(range(0, len(frames), step))
    return dict(emb=np.array(embs) if embs else np.zeros((0, 512)), idx=np.array(idx), n_try=np.array(n_try))


def identity_stats(E):
    e = E['emb']; r = dict(det_rate=round(len(e) / max(1, int(E['n_try'])), 3))
    if len(e) >= 3:
        m = e.mean(0); m /= np.linalg.norm(m) + 1e-9; s = e @ m
        r['mean_cos'] = round(float(s.mean()), 4); r['min_cos'] = round(float(s.min()), 4)
    else:
        r['mean_cos'] = r['min_cos'] = None
    return r


def cross_identity(Eo, Er):
    if len(Eo['emb']) < 3 or len(Er['emb']) < 3: return None
    a = Eo['emb'].mean(0); b = Er['emb'].mean(0)
    return round(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))), 4)


# ---------------------------------------------------------------- flicker
def gray(frames): return np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]).astype(np.float32)


def tdiff(g):
    return float(np.abs(g[1:] - g[:-1]).mean())


def flows_backward(render_g, half=True):
    """flow[t] maps pixel in frame t to its position in frame t-1 (Farneback, computed at half res then upsampled),
    plus a forward-backward-consistency non-occlusion mask. Returns arrays fl (T,h,w,2) f16 and masks (T,h,w) bool at half res."""
    g = [cv2.resize(x, (x.shape[1] // 2, x.shape[0] // 2), interpolation=cv2.INTER_AREA) if half else x for x in render_g]
    h, w = g[0].shape; gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    fl = [np.zeros((h, w, 2), np.float16)]; masks = [np.zeros((h, w), bool)]
    for t in range(1, len(g)):
        a = g[t].astype(np.uint8); b = g[t - 1].astype(np.uint8)
        bw = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.1, 0)
        fw = cv2.calcOpticalFlowFarneback(b, a, None, 0.5, 3, 15, 3, 5, 1.1, 0)
        fw_s = cv2.remap(fw, gx + bw[..., 0], gy + bw[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        err = np.linalg.norm(bw + fw_s, axis=2); mag = np.linalg.norm(bw, axis=2) + np.linalg.norm(fw_s, axis=2)
        occ = err > (0.01 * mag + 0.5)
        inside = (gx + bw[..., 0] >= 0) & (gx + bw[..., 0] <= w - 1) & (gy + bw[..., 1] >= 0) & (gy + bw[..., 1] <= h - 1)
        fl.append(bw.astype(np.float16)); masks.append(~occ & inside)
    return dict(fl=np.stack(fl), masks=np.stack(masks))


def warp_error(g, F):
    """mean abs error of warp(frame t-1 -> t) using the INPUT render's flow, in non-occluded regions. g: (T,H,W) float."""
    fl, masks = F['fl'], F['masks']
    h, w = g.shape[1:]; gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    border = np.zeros((h, w), bool); border[4:-4, 4:-4] = True
    errs = []
    for t in range(1, len(g)):
        f = cv2.resize(fl[t].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) * (w / fl.shape[2])
        wp = cv2.remap(g[t - 1], gx + f[..., 0], gy + f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        m = cv2.resize(masks[t].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool) & border
        if m.sum() > 100: errs.append(float(np.abs(wp - g[t])[m].mean()))
    return float(np.mean(errs)) if errs else None


# ---------------------------------------------------------------- fidelity
_LP = None
def lpips_model():
    global _LP
    if _LP is None:
        import torch, lpips
        torch.set_num_threads(4)
        _LP = lpips.LPIPS(net='alex', verbose=False).eval()
    return _LP


def fidelity(out, real, step=6):
    from skimage.metrics import structural_similarity as ssim
    ps = []; ss = []
    for t in range(0, len(out), step):
        a = cv2.cvtColor(out[t], cv2.COLOR_BGR2RGB); b = cv2.cvtColor(real[t], cv2.COLOR_BGR2RGB)
        mse = float(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean()); ps.append(10 * np.log10(255 ** 2 / max(mse, 1e-9)))
        ss.append(float(ssim(a, b, channel_axis=2, data_range=255)))
    r = dict(psnr=round(float(np.mean(ps)), 3), ssim=round(float(np.mean(ss)), 4), lpips=None)
    try:
        import torch
        m = lpips_model(); ls = []
        with torch.no_grad():
            for t in range(0, len(out), step * 2):
                x = torch.from_numpy(cv2.cvtColor(out[t], cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float() / 127.5 - 1
                y = torch.from_numpy(cv2.cvtColor(real[t], cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float() / 127.5 - 1
                ls.append(float(m(x, y)))
        r['lpips'] = round(float(np.mean(ls)), 4)
    except Exception as e:
        r['lpips_error'] = f'{type(e).__name__}: {str(e)[:120]}'
    return r


# ---------------------------------------------------------------- per clip
def find_inputs(cid):
    p = os.path.join(ROOT, 'data', 'pairs', cid)
    if os.path.isdir(p):
        real = os.path.join(p, 'real.mp4')
        return os.path.join(p, 'render.mp4'), real if os.path.exists(real) else None, 'pair'
    d = os.path.join(ROOT, 'drive', cid + '.mp4')
    if os.path.exists(d): return d, None, 'drive'
    return None, None, None


def split_of(cid):
    try:
        for c in json.load(open(os.path.join(ROOT, 'data', 'manifest.json')))['clips']:
            if c['id'] == cid: return c['split']
    except Exception: pass
    return None


def eval_clip(args):
    cid, out_path, do_lpips = args
    cv2.setNumThreads(1)
    t0 = time.time()
    res = dict(clip=cid)
    try:
        ren_p, real_p, kind = find_inputs(cid)
        res['kind'] = kind; res['split'] = split_of(cid)
        if ren_p is None: return dict(clip=cid, error='no input render/drive clip found')
        if not os.path.exists(out_path): return dict(clip=cid, error='missing output', kind=kind, split=res['split'])
        ren, _ = read_video(ren_p); out, ofps = read_video(out_path); real, _ = read_video(real_p) if real_p else (None, 0)
        if out is None or ren is None: return dict(clip=cid, error='unreadable video')
        res['out_native_fps'] = round(ofps, 2); res['out_frames_grid'] = len(out)
        n = min(len(ren), len(out)); res['len_ratio'] = round(len(out) / len(ren), 3)
        ren = fit_len(ren, n); out = out[:n]
        if real is not None: real = fit_len(real, n)
        has_real = real is not None; res['has_real'] = has_real

        # ---- face / motion
        tr_ren = cached(ren_p, 'track', lambda: track(ren))
        tr_out = track(out)  # output may be rewritten between runs; no cache
        tr_real = cached(real_p, 'track', lambda: track(real)) if has_real else None
        res['face_det_rate'] = dict(render=round(float(np.mean(~np.isnan(tr_ren['open']))), 3),
                                    output=round(float(np.mean(~np.isnan(tr_out['open']))), 3),
                                    real=round(float(np.mean(~np.isnan(tr_real['open']))), 3) if has_real else None)
        # a "reference" of what a perfect model would score: real vs render
        res['motion_vs_input'] = motion_vs(tr_ren, tr_out, 'output_vs_render')
        if has_real:
            res['motion_vs_real'] = motion_vs(tr_real, tr_out, 'output_vs_real')
            res['motion_ref_real_vs_input'] = motion_vs(tr_ren, tr_real, 'real_vs_render')

        # ---- identity
        try:
            Eo = embeddings(out); res['identity'] = identity_stats(Eo)
            if has_real:
                Er = cached(real_p, 'emb', lambda: embeddings(real)); res['identity_real_ceiling'] = identity_stats(Er)
                res['identity']['cos_to_real_person'] = cross_identity(Eo, Er)
            Ei = cached(ren_p, 'emb', lambda: embeddings(ren)); res['identity_render'] = identity_stats(Ei)
        except Exception as e:
            res['identity_error'] = f'{type(e).__name__}: {str(e)[:150]}'

        # ---- flicker
        go = gray(out); gr = gray(ren)
        F = cached(ren_p, 'flow2', lambda: flows_backward(gr))
        res['flicker'] = dict(out_tdiff=round(tdiff(go), 4), render_tdiff=round(tdiff(gr), 4))
        res['flicker']['tdiff_ratio_vs_render'] = round(tdiff(go) / max(tdiff(gr), 1e-6), 4)
        res['flicker']['warp_err_out'] = round(warp_error(go, F), 4)
        res['flicker']['warp_err_render'] = round(warp_error(gr, F), 4)
        if has_real:
            gre = gray(real)
            res['flicker']['real_tdiff'] = round(tdiff(gre), 4)
            res['flicker']['tdiff_ratio_vs_real'] = round(tdiff(go) / max(tdiff(gre), 1e-6), 4)
            res['flicker']['warp_err_real'] = round(warp_error(gre, F), 4)
            res['flicker']['warp_err_ratio_vs_real'] = round(res['flicker']['warp_err_out'] / max(res['flicker']['warp_err_real'], 1e-6), 4)

        # ---- fidelity (context only)
        if has_real:
            res['fidelity'] = fidelity(out, real) if do_lpips else dict(skipped=True)
        res['secs'] = round(time.time() - t0, 1)
    except Exception as e:
        import traceback
        res['error'] = f'{type(e).__name__}: {e}'; res['trace'] = traceback.format_exc()[-800:]
    return res


# ---------------------------------------------------------------- aggregate
def flat(d, pre=''):
    for k, v in d.items():
        if isinstance(v, dict): yield from flat(v, pre + k + '.')
        elif isinstance(v, (int, float)) and not isinstance(v, bool): yield pre + k, float(v)


def aggregate(clips):
    ok = [c for c in clips if 'error' not in c]
    def mean_of(sub):
        acc = {}
        for c in sub:
            for k, v in flat({kk: vv for kk, vv in c.items() if kk not in ('clip', 'secs')}):
                if np.isfinite(v): acc.setdefault(k, []).append(v)
        return {k: round(float(np.mean(v)), 4) for k, v in acc.items()} | {'n': len(sub)}
    return dict(all=mean_of(ok), heldout=mean_of([c for c in ok if c.get('split') == 'heldout']),
                drive=mean_of([c for c in ok if c.get('kind') == 'drive']),
                pairs=mean_of([c for c in ok if c.get('kind') == 'pair']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--track', required=True); ap.add_argument('--variant', required=True)
    ap.add_argument('--clips', nargs='*', default=[]); ap.add_argument('--all', action='store_true')
    ap.add_argument('--variant-dir'); ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--no-lpips', action='store_true'); ap.add_argument('--results-dir', default=os.path.join(HERE, 'results'))
    a = ap.parse_args()
    vd = a.variant_dir or os.path.join(ROOT, a.track, 'out', a.variant)
    clips = list(a.clips)
    if a.all or not clips:
        clips = sorted(f[:-4] for f in os.listdir(vd) if f.endswith('.mp4')) if os.path.isdir(vd) else []
    os.makedirs(a.results_dir, exist_ok=True)
    jobs = [(c, os.path.join(vd, c + '.mp4'), not a.no_lpips) for c in clips]
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    with ctx.Pool(min(a.workers, max(1, len(jobs)))) as pool:
        for k in ('OMP_NUM_THREADS',): os.environ.setdefault(k, '2')
        results = pool.map(eval_clip, jobs, chunksize=1)
    doc = dict(track=a.track, variant=a.variant, variant_dir=vd, made=time.strftime('%F %T'), clips={c['clip']: c for c in results},
               mean=aggregate(results))
    p = os.path.join(a.results_dir, f'{a.track}_{a.variant}.json')
    json.dump(doc, open(p, 'w'), indent=1, default=float)
    print('wrote', p, 'clips', len(results), 'errors', sum('error' in c for c in results))


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    main()
