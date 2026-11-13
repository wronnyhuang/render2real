import json, time, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
s = json.load(open('pod_state.json'))
end = s.get('terminated_at') or time.time()
h = (end - s['created_at']) / 3600
pods = [{'pod': s['pod_id'], 'name': 'r2r-turbo0', 'gpu': s['gpu'], 'cloud': s['cloud'], 'rate': s['rate'], 'hours': round(h, 3),
         'usd': round(h * s['rate'], 3), 'terminated': bool(s.get('terminated_at'))}]
extra = []
if os.path.exists('extra_pods.log'):
    extra = [json.loads(l) for l in open('extra_pods.log') if l.strip()]
xusd = round(0.01 * len(extra), 3)  # over-cap auto-created pods terminated within seconds
out = {'usd': round(pods[0]['usd'] + xusd, 3), 'pod_hours': pods[0]['hours'], 'rate': s['rate'], 'pods': pods,
       'extra_instant_terminated_pods': extra, 'cap_usd': 2.5, 'method': 'wall hours (create -> verified terminate) x $/h'}
json.dump(out, open('spend.json', 'w'), indent=1)
print(json.dumps(out))
