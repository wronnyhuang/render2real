#!/usr/bin/env python3
"""Blind judge (Claude subscription via `claude -p`, Read tool only).
  judge.py --track fast --variant v1 --clips S01a muse_01 ... [--variant-dir DIR] [--workers 6] [--shots 6]
  judge.py --track fast --variant v1 --all
Per clip: contact strips (6 frames x 384 px) of render / output / real, shown under neutral random filenames.
  * pairwise render-vs-output, BOTH orders: forced choice "more like real camera footage of a real person",
    1-10 realism for each, 1-10 "B has same pose/expression/mouth as A", one-line reason.
  * three-way absolute realism (render, output, real; random order) when a real video exists = ceiling.
Writes eval/results/judge_<track>_<variant>.json : per clip + win rate of output over render with Wilson CI
(over trials) and clip-level bootstrap CI.
"""
import argparse, json, os, random, re, subprocess, sys, time, hashlib, shutil, math
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CLAUDE = os.environ.get('JUDGE_CLAUDE', '/usr/local/bin/claude')
MODEL = os.environ.get('JUDGE_MODEL', 'sonnet')
WORK = os.path.join(HERE, 'judge_work')

SYSTEM = ("You are a careful visual quality rater. You compare strips of video frames. You judge only what is visible. "
          "You do not know how the images were made. Answer strictly in the requested JSON.")

PAIR_SCHEMA = {"type": "object", "properties": {
    "more_real": {"type": "string", "enum": ["A", "B"]},
    "realism_A": {"type": "integer", "minimum": 1, "maximum": 10},
    "realism_B": {"type": "integer", "minimum": 1, "maximum": 10},
    "same_pose_score": {"type": "integer", "minimum": 1, "maximum": 10},
    "reason": {"type": "string"}},
    "required": ["more_real", "realism_A", "realism_B", "same_pose_score", "reason"]}
TRI_SCHEMA = {"type": "object", "properties": {
    "realism_X": {"type": "integer", "minimum": 1, "maximum": 10},
    "realism_Y": {"type": "integer", "minimum": 1, "maximum": 10},
    "realism_Z": {"type": "integer", "minimum": 1, "maximum": 10},
    "reason": {"type": "string"}}, "required": ["realism_X", "realism_Y", "realism_Z", "reason"]}


def strip(path, out_png, shots=6, size=384):
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0: return False
    fr = []
    for k in range(shots):
        i = int((k + 0.5) * n / shots); cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
        if not ok: return False
        fr.append(cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA))
    cv2.imwrite(out_png, np.hstack(fr)); return True


def find_inputs(cid):
    p = os.path.join(ROOT, 'data', 'pairs', cid)
    if os.path.isdir(p):
        real = os.path.join(p, 'real.mp4'); return os.path.join(p, 'render.mp4'), real if os.path.exists(real) else None
    d = os.path.join(ROOT, 'drive', cid + '.mp4')
    return (d, None) if os.path.exists(d) else (None, None)


def call_claude(prompt, schema, cwd, tag):
    cmd = [CLAUDE, '-p', '--model', MODEL, '--tools', 'Read', '--permission-mode', 'dontAsk', '--no-session-persistence',
           '--setting-sources', '', '--system-prompt', SYSTEM, '--output-format', 'json', '--json-schema', json.dumps(schema)]
    last = None
    for attempt in (1, 2, 3):
        t0 = time.time()
        try:
            r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=cwd, timeout=600)
            d = json.loads(r.stdout)
            out = d.get('structured_output')
            if out is None: out = json.loads(re.search(r'\{.*\}', d.get('result') or '', re.S).group(0))
            return out
        except Exception as e:
            last = f'{type(e).__name__}: {str(e)[:200]}'
            time.sleep(2 * attempt)
    return {'error': last}


PAIR_PROMPT = """Two horizontal strips of video frames (6 frames each, left to right = time) are in your working directory:
{a} (call it strip A) and {b} (call it strip B). Use the Read tool on both files, then answer.
Each strip shows the same short moment of a person (or character) talking, from the same camera framing.
1. more_real: which strip looks more like REAL CAMERA FOOTAGE of a real human being (not a render, game, cartoon or CG)? Forced choice, A or B.
2. realism_A, realism_B: how real each looks, 1 (obviously CG/cartoon) to 10 (indistinguishable from a real camera).
3. same_pose_score: 1-10, how well strip B shows the same head pose, facial expression and mouth shape as strip A, frame by frame (10 = same, 1 = unrelated).
4. reason: one short line.
Return only the JSON."""
TRI_PROMPT = """Three horizontal strips of video frames (6 frames each) are in your working directory: {x} (strip X), {y} (strip Y), {z} (strip Z).
Use the Read tool on all three. Rate each independently for how much it looks like REAL CAMERA FOOTAGE of a real human being:
1 = obviously CG/cartoon/render, 10 = indistinguishable from a real camera recording. Use the full range. Add one short line reason. Return only the JSON."""


def judge_clip(job):
    cid, out_p, shots, seed = job
    ren_p, real_p = find_inputs(cid)
    res = dict(clip=cid)
    if ren_p is None or not os.path.exists(out_p): return dict(clip=cid, error='missing input or output')
    rnd = random.Random(f'{seed}-{cid}')
    d = os.path.join(WORK, f'{cid}_{hashlib.md5((out_p + str(os.path.getmtime(out_p))).encode()).hexdigest()[:6]}')
    os.makedirs(d, exist_ok=True)
    names = {}
    for role, p in (('render', ren_p), ('output', out_p), ('real', real_p)):
        if p is None: continue
        nm = f'img_{rnd.randrange(10**6):06d}.png'; names[role] = nm
        if not strip(p, os.path.join(d, nm), shots): return dict(clip=cid, error=f'cannot read {role}')
    trials = []
    for order in (('render', 'output'), ('output', 'render')):
        a, b = order
        r = call_claude(PAIR_PROMPT.format(a=names[a], b=names[b]), PAIR_SCHEMA, d, f'{cid} {a}-{b}')
        if 'error' in r: trials.append(dict(order=f'{a}/{b}', error=r['error'])); continue
        win = {'A': a, 'B': b}[r['more_real']]
        trials.append(dict(order=f'{a}/{b}', chose=win, output_wins=win == 'output',
                           realism_render=r['realism_A' if a == 'render' else 'realism_B'],
                           realism_output=r['realism_A' if a == 'output' else 'realism_B'],
                           same_pose=r['same_pose_score'] if b == 'output' else None,  # asked B-vs-A; keep both orders
                           same_pose_any=r['same_pose_score'], reason=r['reason']))
    res['pairwise'] = trials
    if real_p:
        roles = ['render', 'output', 'real']; rnd.shuffle(roles)
        r = call_claude(TRI_PROMPT.format(x=names[roles[0]], y=names[roles[1]], z=names[roles[2]]), TRI_SCHEMA, d, f'{cid} tri')
        if 'error' in r: res['three_way'] = dict(error=r['error'])
        else: res['three_way'] = dict(order=roles, **{roles[i]: r['realism_' + 'XYZ'[i]] for i in range(3)}, reason=r['reason'])
    return res


def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0, c - h), 3), round(min(1, c + h), 3)]


def summarize(clips):
    ok = [c for c in clips if 'pairwise' in c]
    tr = [t for c in ok for t in c['pairwise'] if 'error' not in t]
    k = sum(t['output_wins'] for t in tr); n = len(tr)
    s = dict(clips=len(ok), trials=n, output_win_rate=round(k / n, 3) if n else None, output_win_ci95_wilson=wilson(k, n))
    if ok:
        per = [np.mean([t['output_wins'] for t in c['pairwise'] if 'error' not in t]) for c in ok if any('error' not in t for t in c['pairwise'])]
        if per:
            rng = np.random.default_rng(0); bs = [np.mean(rng.choice(per, len(per))) for _ in range(2000)]
            s['output_win_rate_clipmean'] = round(float(np.mean(per)), 3)
            s['output_win_ci95_bootstrap_clips'] = [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
        # order consistency: both orders choose the same role
        cons = [c['pairwise'][0]['chose'] == c['pairwise'][1]['chose'] for c in ok if len(c['pairwise']) == 2 and all('error' not in t for t in c['pairwise'])]
        s['order_consistency'] = round(float(np.mean(cons)), 3) if cons else None
    for key in ('realism_render', 'realism_output', 'same_pose_any'):
        v = [t[key] for t in tr if t.get(key) is not None]
        if v: s['mean_' + key] = round(float(np.mean(v)), 2)
    tw = [c['three_way'] for c in clips if isinstance(c.get('three_way'), dict) and 'error' not in c['three_way']]
    if tw:
        for role in ('render', 'output', 'real'): s['three_way_mean_' + role] = round(float(np.mean([t[role] for t in tw])), 2)
        s['three_way_n'] = len(tw)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--track', required=True); ap.add_argument('--variant', required=True)
    ap.add_argument('--clips', nargs='*', default=[]); ap.add_argument('--all', action='store_true')
    ap.add_argument('--variant-dir'); ap.add_argument('--workers', type=int, default=4); ap.add_argument('--shots', type=int, default=6)
    ap.add_argument('--seed', default='1'); ap.add_argument('--results-dir', default=os.path.join(HERE, 'results'))
    a = ap.parse_args()
    vd = a.variant_dir or os.path.join(ROOT, a.track, 'out', a.variant)
    clips = list(a.clips)
    if a.all or not clips: clips = sorted(f[:-4] for f in os.listdir(vd) if f.endswith('.mp4')) if os.path.isdir(vd) else []
    os.makedirs(a.results_dir, exist_ok=True); os.makedirs(WORK, exist_ok=True)
    jobs = [(c, os.path.join(vd, c + '.mp4'), a.shots, a.seed) for c in clips]
    with ThreadPoolExecutor(min(a.workers, 6)) as ex: results = list(ex.map(judge_clip, jobs))
    split = {}
    try: split = {c['id']: c['split'] for c in json.load(open(os.path.join(ROOT, 'data', 'manifest.json')))['clips']}
    except Exception: pass
    for r in results: r['split'] = split.get(r['clip'])
    doc = dict(track=a.track, variant=a.variant, model=MODEL, made=time.strftime('%F %T'), clips={r['clip']: r for r in results},
               summary=summarize(results),
               summary_heldout=summarize([r for r in results if r.get('split') == 'heldout']),
               summary_drive=summarize([r for r in results if r['clip'] not in split]))
    p = os.path.join(a.results_dir, f'judge_{a.track}_{a.variant}.json')
    json.dump(doc, open(p, 'w'), indent=1, default=float); print('wrote', p, json.dumps(doc['summary']))


if __name__ == '__main__': main()
