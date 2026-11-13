"""Contact sheets: per clip one row [render|real] at t=0.5,4,7.5 (128 px tiles) with id label; N rows per sheet."""
import sys, json, os, subprocess, numpy as np, cv2
D = __import__('os').path.expandvars('$WORK/data')
def frames(path, ts, s):
    cap = cv2.VideoCapture(path); out = []
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * 30)); ok, f = cap.read()
        out.append(cv2.resize(f, (s, s), interpolation=cv2.INTER_AREA) if ok else np.zeros((s, s, 3), np.uint8))
    return out
def row(cid, s=128, ts=(0.5, 4.0, 7.5)):
    a = frames(f'{D}/pairs/{cid}/render.mp4', ts, s); b = frames(f'{D}/pairs/{cid}/real.mp4', ts, s)
    r = np.hstack([np.hstack([x, y]) for x, y in zip(a, b)])
    cv2.putText(r, cid, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return r
if __name__ == '__main__':
    out, ids = sys.argv[1], sys.argv[2:]
    rows = [row(c) for c in ids]
    # two columns of rows
    half = (len(rows) + 1) // 2; L = rows[:half]; R = rows[half:] + [np.zeros_like(rows[0])] * (half - len(rows[half:]))
    img = np.hstack([np.vstack(L), np.full((128 * half, 6, 3), 255, np.uint8), np.vstack(R)])
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
