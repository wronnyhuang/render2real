"""One-shot Sonnet call through the Claude subscription CLI (claude -p --model sonnet), images inline. Logs cost/time."""
import json, base64, subprocess, time, re, os, fcntl
LOG = __import__('os').path.expandvars('$WORK/data/logs/sonnet_calls.jsonl')
SYS = ("You are a careful visual analyst working on a motion-capture / 3D re-creation pipeline. "
       "Follow the output format exactly. When asked for JSON, reply with one JSON object only, no prose, no code fences.")
def call(stage, cid, text, images=(), model='sonnet', timeout=600, retries=2):
    content = []
    for im in images:
        if isinstance(im, tuple): label, path = im; content.append({"type": "text", "text": label})
        else: path = im
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                        "data": base64.b64encode(open(path, 'rb').read()).decode()}})
    content.append({"type": "text", "text": text})
    msg = {"type": "user", "message": {"role": "user", "content": content}}
    last = None
    for attempt in range(retries + 1):
        t0 = time.time()
        p = subprocess.run(['claude', '-p', '--model', model, '--input-format', 'stream-json', '--output-format', 'stream-json',
                            '--verbose', '--tools', '', '--max-turns', '1', '--system-prompt', SYS, '--setting-sources', ''],
                           input=json.dumps(msg) + '\n', capture_output=True, text=True, timeout=timeout, cwd='/tmp')
        res = None
        for l in p.stdout.splitlines():
            try: j = json.loads(l)
            except Exception: continue
            if j.get('type') == 'result': res = j
        dt = time.time() - t0
        if res is None or res.get('is_error'):
            last = (p.stderr[-400:], res and res.get('result')); continue
        txt = res.get('result', '')
        cost = res.get('total_cost_usd', 0)
        models = list((res.get('modelUsage') or {}).keys())
        with open(LOG, 'a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(dict(stage=stage, clip=cid, cost=cost, sec=round(dt, 1), models=models, n_img=len(images), attempt=attempt)) + '\n')
        m = re.search(r'\{.*\}', txt, re.S)
        if m:
            try: return json.loads(m.group(0)), txt
            except Exception as e: last = ('json', txt[:300])
        else: last = ('nojson', txt[:300])
    raise RuntimeError(f'sonnet call failed {stage} {cid}: {last}')
