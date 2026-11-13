"""QC gate + split + manifest + contact sheet. Usage: finalize.py [drop_ids...]"""
import json, glob, os, sys, collections, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sheet import row
D = __import__('os').path.expandvars('$WORK/data')
HELDOUT = ['P0115', 'P0485', 'P0845', 'P1067', 'P0989', 'P0217', 'P0370', 'P1174']
manual_drop = set(sys.argv[1:])
clips, dropped = [], []
for m in sorted(glob.glob(f'{D}/pairs/M*/meta.json')):
    meta = json.load(open(m)); cid = meta['id']; q = f'{D}/qc/qc_{cid}.json'
    if not os.path.exists(q): dropped.append((cid, 'no qc')); continue
    r = json.load(open(q)).get(cid, {})
    tr = np.load(f'{D}/work/{cid}/track.npz'); n2 = int(tr['n_second_face'])
    why = []
    if r.get('err'): why.append('qc error')
    else:
        if r['frames_real'] != 240 or r['frames_render'] != 240: why.append(f"frames {r['frames_real']}/{r['frames_render']}")
        if r['cov_real'] < 0.95: why.append(f"real face cov {r['cov_real']:.2f}")
        if r['cov_render'] < 0.9: why.append(f"render face cov {r['cov_render']:.2f}")
        if r['face_inside'] < 0.95: why.append(f"face leaves crop {r['face_inside']:.2f}")
        if r['second_face_real'] > 0.1 or n2 > 15: why.append(f'second face {n2}')
        if not (r['mouth_corr'] >= 0.5): why.append(f"mouth corr {r['mouth_corr']:.2f}")
        if abs(r['best_lag']) > 1: why.append(f"lag {r['best_lag']}")
    if cid in manual_drop: why.append('visual QC')
    if why: dropped.append((cid, '; '.join(why))); continue
    meta['qc'] = r; json.dump(meta, open(m, 'w'), indent=1)
    clips.append(dict(id=cid, identity=meta['identity'], split='heldout' if meta['identity'] in HELDOUT else 'train',
                      source='seamless', frames=240, mouth_corr=round(r['mouth_corr'], 3), face_width_frac=round(r['face_width_frac'], 3)))
json.dump(dict(clips=clips, format='384x384 H.264 yuv420p, 30 fps, 240 frames; render.mp4 and real.mp4 frame-aligned',
               heldout_identities=HELDOUT, dropped=[dict(id=a, why=b) for a, b in dropped]), open(f'{D}/manifest.json', 'w'), indent=1)
sp = collections.Counter(c['split'] for c in clips)
print('kept', len(clips), dict(sp), 'ids', len({c['identity'] for c in clips}),
      'heldout ids', len({c['identity'] for c in clips if c['split'] == 'heldout'}), 'dropped', len(dropped))
for a, b in dropped: print('DROP', a, b)
# contact sheet: all heldout + train spread, 24 rows
ho = [c['id'] for c in clips if c['split'] == 'heldout']; tr = [c['id'] for c in clips if c['split'] == 'train']
seen = set(); trs = []
for c in clips:
    if c['split'] == 'train' and c['identity'] not in seen: seen.add(c['identity']); trs.append(c['id'])
pick = ho[:12] + trs[:24 - len(ho[:12])]
rows = [row(c) for c in pick]; half = (len(rows) + 1) // 2
L = rows[:half]; R = rows[half:] + [np.zeros_like(rows[0])] * (half - len(rows[half:]))
cv2.imwrite(f'{D}/contact_sheet.jpg', np.hstack([np.vstack(L), np.full((128 * half, 6, 3), 255, np.uint8), np.vstack(R)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
