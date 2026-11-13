"""Pick 8 s windows from Seamless dev videos: one frontal talking adult, static head-and-shoulders square crop.
Writes work/cands.json (all per-video picks) and work/selected.json (identity-capped list)."""
import json, glob, os, numpy as np, collections
SEAM = __import__('os').path.expandvars('$SEAMLESS_ROOT')
D = __import__('os').path.expandvars('$WORK/data')
SEED_IDS = {'P0115', 'P0350', 'P0370', 'P0485', 'P1076', 'P0467', 'P0840', 'P0845', 'P0844A', 'P1020', 'P0731', 'P1281A'}
L = 240
cands = []
for vid in sorted(glob.glob(f'{SEAM}/*/dev/video/*.mp4')):
    fid = os.path.basename(vid)[:-4]; base = os.path.dirname(os.path.dirname(vid))
    try:
        kp = np.load(f'{base}/boxes_and_keypoints/keypoints/{fid}.npy'); vb = np.load(f'{base}/boxes_and_keypoints/is_valid_box/{fid}.npy').reshape(-1)
    except Exception as e: print('skip', fid, e); continue
    vad = [json.loads(l) for l in open(f'{base}/metadata/vad/{fid}.jsonl')] if os.path.exists(f'{base}/metadata/vad/{fid}.jsonl') else []
    T = len(kp); sp = np.zeros(T)
    for s in vad: sp[int(s['start'] * 30):int(s['end'] * 30)] = 1
    nose, le, re_ = kp[:, 0], kp[:, 3], kp[:, 4]
    ed = np.abs(le[:, 0] - re_[:, 0]); mid = (le[:, 0] + re_[:, 0]) / 2
    frontal = (le[:, 2] > 0.5) & (re_[:, 2] > 0.5) & (nose[:, 2] > 0.6) & (np.abs(nose[:, 0] - mid) / np.maximum(ed, 1) < 0.18) & (vb > 0)
    picks = []
    for st in range(90, T - L - 30, 15):
        w = slice(st, st + L)
        fr = frontal[w].mean(); s = sp[w].mean()
        if fr < 0.9 or s < 0.5: continue
        e = np.median(ed[w][frontal[w]]); S = int(round(2.0 * e / 0.98)) // 2 * 2   # ear-to-ear ~0.98 face width; face ~50% of crop
        nx, ny = np.median(nose[w][frontal[w], 0]), np.median(nose[w][frontal[w], 1])
        x0 = int(np.clip(nx - S / 2, 0, 1080 - S)); y0 = int(np.clip(ny - 0.47 * S, 0, 1920 - S))
        # face box approx per frame: nose +- 0.6 ed horiz, -0.85 ed .. +0.75 ed vertical
        fx0 = nose[w, 0] - 0.6 * ed[w]; fx1 = nose[w, 0] + 0.6 * ed[w]; fy0 = nose[w, 1] - 0.85 * ed[w]; fy1 = nose[w, 1] + 0.75 * ed[w]
        inside = ((fx0 > x0) & (fx1 < x0 + S) & (fy0 > y0) & (fy1 < y0 + S))[frontal[w]].mean()
        mot = np.std(nose[w][frontal[w], :2], 0).max() / e
        if inside < 0.98 or y0 == 0 and ny - 0.47 * S < -0.05 * S: continue
        picks.append(dict(score=fr + s - mot, st=st, S=S, x0=x0, y0=y0, speech=round(s, 2), frontal=round(fr, 2), mot=round(float(mot), 3), ed=float(e)))
    picks.sort(key=lambda p: -p['score']); chosen = []
    for p in picks:
        if all(abs(p['st'] - c['st']) > 900 for c in chosen): chosen.append(p)
        if len(chosen) == 3: break
    pid = fid.split('_')[-1]
    for k, c in enumerate(chosen):
        cands.append(dict(file_id=fid, video=vid, identity=pid, rank=k, **c))
    print(fid, len(picks), [(c['st'], round(c['score'], 2), c['S']) for c in chosen])
os.makedirs(f'{D}/work', exist_ok=True)
json.dump(cands, open(f'{D}/work/cands.json', 'w'), indent=1)
# identity cap: round-robin over files (rank 0 first), max CAP per identity, max 2 per file
CAP = 5
byid = collections.defaultdict(list)
for c in cands: byid[c['identity']].append(c)
sel = []
for pid, cs in sorted(byid.items()):
    cs.sort(key=lambda c: (c['rank'], -c['score'])); got = []; perfile = collections.Counter()
    for c in cs:
        if len(got) >= CAP: break
        if perfile[c['file_id']] >= 2: continue
        got.append(c); perfile[c['file_id']] += 1
    sel += got
for i, c in enumerate(sel): c['id'] = f"M{i:03d}_{c['identity']}"
json.dump(sel, open(f'{D}/work/selected.json', 'w'), indent=1)
print('cands', len(cands), 'selected', len(sel), 'identities', len({c['identity'] for c in sel}))
