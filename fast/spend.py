import json, glob, time, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
rows = []
for f in sorted(glob.glob('pod_state*.json')):
    s = json.load(open(f))
    if not s.get('pod_id'): continue
    end = s.get('terminated_at') or time.time()
    h = (end - s['created_at']) / 3600
    rows.append({'pod': s['pod_id'], 'gpu': s['gpu'], 'cloud': s['cloud'], 'rate': s['rate'], 'hours': round(h, 3),
                 'usd': round(h * s['rate'], 3), 'terminated': bool(s.get('terminated_at')),
                 'note': s.get('note', '')})
extra = 0.02  # 3x RTX 4090 SECURE pods auto-created at $0.89/h and terminated within seconds (over the $0.70 cap)
usd = round(sum(r['usd'] for r in rows) + extra, 3)
out = {'usd': usd, 'pod_hours': round(sum(r['hours'] for r in rows), 3),
       'rate': ' / '.join(f"{r['gpu'].replace('NVIDIA ', '')} ${r['rate']}/h" for r in rows),
       'pods': rows, 'extra_usd_instant_terminated_pods': extra, 'cap_usd': 4.0,
       'method': 'wall hours (create -> verified terminate) x $/h'}
json.dump(out, open('spend.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
