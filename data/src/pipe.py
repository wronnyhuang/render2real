"""Driver: cut -> track -> rig -> Sonnet look -> Blender crop render, parallel pools. Usage: pipe.py [ids...] """
import json, os, sys, time, subprocess, threading
from concurrent.futures import ThreadPoolExecutor
D = __import__('os').path.expandvars('$WORK/data'); SRC = f'{D}/src'
PY = '$WORK/env/mp/bin/python'
sel = json.load(open(f'{D}/work/selected.json'))
if os.environ.get('ONLY_R'):   # render stage only, from saved per-clip state
    sel = [json.load(open(f"{D}/work/{c['id']}/c.json")) for c in sel if os.path.exists(f"{D}/work/{c['id']}/c.json")]
ids = set(sys.argv[1:])
if ids: sel = [c for c in sel if c['id'] in ids]
NA, NS, NR = int(os.environ.get('NA', 24)), int(os.environ.get('NS', 8)), int(os.environ.get('NR', 40))
lock = threading.Lock()
def log(**k):
    with lock:
        open(f'{D}/logs/pipe.jsonl', 'a').write(json.dumps(k) + '\n')
def sh(cmd, logf, env=None):
    with open(logf, 'a') as f:
        r = subprocess.run(['nice', '-n', '10'] + cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    if r.returncode: raise RuntimeError(f'{cmd[:3]} rc={r.returncode}')
def ctx_geom(c):
    S, x0, y0 = c['S'], c['x0'], c['y0']
    CW = min(1080, int(1.45 * S) // 2 * 2); CH = min(1920, int(CW * 16 / 9) // 2 * 2)
    cx0 = int(min(max(x0 + S / 2 - CW / 2, 0), 1080 - CW)); cy0 = int(min(max(y0 - 0.3 * S, 0), 1920 - CH))
    return CW, CH, cx0, cy0
def stageA(c):
    wd = f"{D}/work/{c['id']}"; os.makedirs(wd, exist_ok=True); pd = f"{D}/pairs/{c['id']}"; os.makedirs(pd, exist_ok=True)
    lg = f'{wd}/log.txt'; t0 = time.time()
    CW, CH, cx0, cy0 = ctx_geom(c); S, x0, y0 = c['S'], c['x0'], c['y0']
    fc = (f"[0:v]trim=start_frame=0:end_frame=240,setpts=PTS-STARTPTS,split[a][b];"
          f"[a]crop={CW}:{CH}:{cx0}:{cy0}[ctx];[b]crop={S}:{S}:{x0}:{y0},scale=384:384:flags=lanczos[real]")
    enc = ['-an', '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p', '-r', '30']
    sh(['ffmpeg', '-v', 'error', '-ss', f"{c['st']/30:.4f}", '-i', c['video'], '-filter_complex', fc,
        '-map', '[ctx]', *enc, '-crf', '12', '-y', f'{wd}/ctx.mp4', '-map', '[real]', *enc, '-crf', '16', '-y', f'{pd}/real.mp4'], lg)
    t1 = time.time()
    k = next(_slot) % 16; cores = f'{4*k}-{4*k+3}'
    sh(['taskset', '-c', cores, PY, '-I', f'{SRC}/track.py', c['id'], f'{wd}/ctx.mp4', f'{wd}/track.npz'], lg)
    t2 = time.time()
    sh([PY, '-I', f'{SRC}/rig.py', f'{wd}/track.npz', f'{wd}/rig.npz'], lg)
    t3 = time.time()
    c['ctx'] = [CW, CH, cx0, cy0]; c['crop_ctx'] = [x0 - cx0, y0 - cy0, S]
    log(id=c['id'], stage='A', cut=round(t1 - t0, 1), track=round(t2 - t1, 1), rig=round(t3 - t2, 1))
    return c
def stageS(c):
    wd = f"{D}/work/{c['id']}"; t0 = time.time()
    if not os.path.exists(f'{wd}/scene.json'):
        sh([PY, '-I', f'{SRC}/scene.py', c['id'], wd], f'{wd}/log.txt')
    log(id=c['id'], stage='S', sonnet=round(time.time() - t0, 1)); return c
def stageR(c):
    wd = f"{D}/work/{c['id']}"; t0 = time.time()
    open(f'{wd}/render_log.txt', 'w').close()
    cx, cy, S = c['crop_ctx']
    env = dict(os.environ, LP_NUM_THREADS='1')
    sh(['xvfb-run', '-a', '-s', '-screen 0 640x480x24', 'blender', '-b', '-t', '1', '--python', f'{SRC}/render_blender.py', '--',
        '--rig', f'{wd}/rig.npz', '--scene', f'{wd}/scene.json', '--out', f"{D}/pairs/{c['id']}/render.mp4", '--workers', '1',
        '--crop', f'{cx},{cy},{S}', '--size', '384', '--aa', '5'], f'{wd}/render_log.txt', env)
    log(id=c['id'], stage='R', render=round(time.time() - t0, 1))
    meta = dict(id=c['id'], identity=c['identity'], source='seamless', file_id=c['file_id'], src_video=c['video'],
                src_start_frame=c['st'], src_start_s=round(c['st'] / 30, 3), frames=240, fps=30, size=384,
                crop_src_1080x1920=[c['x0'], c['y0'], c['S'], c['S']], track_ctx_src=c['ctx'], crop_in_ctx=c['crop_ctx'],
                speech_frac=c['speech'], frontal_frac=c['frontal'],
                origin='render2real scale: tracked on native-res context crop, Blender renders only the crop window at 384')
    json.dump(meta, open(f"{D}/pairs/{c['id']}/meta.json", 'w'), indent=1)
    return c['id']
os.makedirs(f'{D}/logs', exist_ok=True)
pa, ps, pr = ThreadPoolExecutor(NA), ThreadPoolExecutor(NS), ThreadPoolExecutor(NR)
futs = []
import itertools
_slot = itertools.count()
def chainA(c):
    if os.path.exists(f"{D}/work/{c['id']}/c.json"): return
    try: c = stageA(c)
    except Exception as e: log(id=c['id'], stage='A', err=str(e)); return
    ps.submit(chainS, c)
def chainS(c):
    try: c = stageS(c)
    except Exception as e: log(id=c['id'], stage='S', err=str(e)); return
    json.dump(c, open(f"{D}/work/{c['id']}/c.json", 'w'))
    if not os.environ.get('NO_R'): futs.append(pr.submit(chainR, c))
def chainR(c):
    try: stageR(c)
    except Exception as e: log(id=c['id'], stage='R', err=str(e))
if os.environ.get('ONLY_R'):
    fa = [pr.submit(chainR, c) for c in sel if not os.path.exists(f"{D}/pairs/{c['id']}/render.mp4")]
else: fa = [pa.submit(chainA, c) for c in sel]
for f in fa: f.result()
pa.shutdown(wait=True); ps.shutdown(wait=True); pr.shutdown(wait=True)
print('done')
