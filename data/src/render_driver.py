"""Poll work/*/c.json (written after the Sonnet stage) and render each once, NR in parallel."""
import json, os, sys, time, glob, subprocess
from concurrent.futures import ThreadPoolExecutor
D = __import__('os').path.expandvars('$WORK/data')
NR = int(os.environ.get('NR', 32)); ASPID = int(open(f'{D}/logs/pipeAS.pid').read())
sys.argv = ['pipe.py', '__none__']; os.environ['NO_R'] = '1'
import importlib.util
spec = importlib.util.spec_from_file_location('pipe', f'{D}/src/pipe.py'); pipe = importlib.util.module_from_spec(spec); spec.loader.exec_module(pipe)
ex = ThreadPoolExecutor(NR); seen = set(); fs = []
def alive(pid):
    try: os.kill(pid, 0); return True
    except OSError: return False
while True:
    running = alive(int(open(f'{D}/logs/pipeAS.pid').read()))
    for f in sorted(glob.glob(f'{D}/work/M*/c.json')):
        c = json.load(open(f))
        if c['id'] in seen or os.path.exists(f"{D}/work/{c['id']}/render.started") or os.path.exists(f"{D}/pairs/{c['id']}/render.mp4"): continue
        seen.add(c['id']); open(f"{D}/work/{c['id']}/render.started", 'w').close(); fs.append(ex.submit(pipe.chainR, c))
    if not running: break
    time.sleep(10)
ex.shutdown(wait=True); print('render driver done', len(seen))
