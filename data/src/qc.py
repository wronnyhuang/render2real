"""QC one pair: face-only MediaPipe on real & render (384) -> coverage, face-in-crop, mouth-open correlation, lag."""
import sys, json, numpy as np, cv2, mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision as V
D = __import__('os').path.expandvars('$WORK/data'); MOD = __import__('os').path.expandvars('$WORK/models/face_landmarker.task')
def run(path):
    fl = V.FaceLandmarker.create_from_options(V.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=MOD),
         running_mode=V.RunningMode.VIDEO, num_faces=2, min_face_detection_confidence=0.4, min_tracking_confidence=0.4))
    cap = cv2.VideoCapture(path); t = 0; mo = []; box = []; nf = []
    while True:
        ok, f = cap.read()
        if not ok: break
        r = fl.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), int(t * 1000 / 30)); t += 1
        nf.append(len(r.face_landmarks))
        if not r.face_landmarks: mo.append(np.nan); box.append([np.nan] * 4); continue
        a = max(r.face_landmarks, key=lambda lm: np.ptp([p.x for p in lm]))
        a = np.array([[p.x, p.y] for p in a])
        fw = np.linalg.norm(a[234] - a[454]); mo.append(np.linalg.norm(a[13] - a[14]) / max(fw, 1e-4))
        box.append([a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()])
    return np.array(mo), np.array(box), np.array(nf), t
out = {}
for cid in sys.argv[1:]:
    try:
        mr, br, nr, Tr = run(f'{D}/pairs/{cid}/real.mp4'); mx, bx, nx, Tx = run(f'{D}/pairs/{cid}/render.mp4')
    except Exception as e:
        out[cid] = dict(err=str(e)); continue
    ok = ~np.isnan(mr) & ~np.isnan(mx)
    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if len(a) > 20 and a.std() > 0 and b.std() > 0 else float('nan')
    lags = {}
    for L in range(-6, 7):
        if L >= 0: a, b = mr[L:], mx[:len(mx) - L]
        else: a, b = mr[:L], mx[-L:]
        m = ~np.isnan(a) & ~np.isnan(b); lags[L] = corr(a[m], b[m])
    best = max(lags, key=lambda k: -1 if np.isnan(lags[k]) else lags[k])
    vb = br[~np.isnan(br[:, 0])]
    inside = float(((vb[:, 0] > 0) & (vb[:, 1] > -0.02) & (vb[:, 2] < 1) & (vb[:, 3] < 1)).mean()) if len(vb) else 0.0
    fwid = float(np.median(vb[:, 2] - vb[:, 0])) if len(vb) else 0.0
    vx = bx[~np.isnan(bx[:, 0])]
    fwid_r = float(np.median(vx[:, 2] - vx[:, 0])) if len(vx) else 0.0
    cdx = float(np.nanmedian((br[:, 0] + br[:, 2]) / 2 - (bx[:, 0] + bx[:, 2]) / 2)); cdy = float(np.nanmedian((br[:, 1] + br[:, 3]) / 2 - (bx[:, 1] + bx[:, 3]) / 2))
    out[cid] = dict(frames_real=Tr, frames_render=Tx, cov_real=float((~np.isnan(mr)).mean()), cov_render=float((~np.isnan(mx)).mean()),
                    second_face_real=float((nr > 1).mean()), face_inside=inside, face_width_frac=fwid, render_face_width_frac=fwid_r, face_center_offset_real_minus_render=[cdx, cdy],
                    mouth_corr=lags[0], best_lag=int(best), best_corr=lags[best])
    print(cid, json.dumps(out[cid]), flush=True)
json.dump(out, open(f'{D}/qc/qc_{sys.argv[1]}.json', 'w'), indent=1)
