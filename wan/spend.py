import json,time,os
D=__import__('os').path.expandvars('$WORK/wan/')
r=lambda f: float(open(D+f).read()) if os.path.exists(D+f) else None
now=time.time()
pods=[dict(id='56fyezppfmh8j6',gpu='NVIDIA L4',rate=0.59,start=r('pod_start_epoch'),end=r('pod_end_epoch') or now),
      dict(id='zfd5fmdgd4z7ij',gpu='NVIDIA RTX PRO 4500 Blackwell',rate=0.72,start=r('pod2_start_epoch'),end=r('pod2_end_epoch') or now)]
tot=0;hrs=0
for p in pods:
    p['hours']=round((p['end']-p['start'])/3600,4); p['usd']=round(p['hours']*p['rate'],4); tot+=p['usd']; hrs+=p['hours']
    p['terminated']=os.path.exists(D+('pod_end_epoch' if p['id'].startswith('56') else 'pod2_end_epoch'))
json.dump({'usd':round(tot,3),'pod_hours':round(hrs,3),'rate':'0.59 (L4) / 0.72 (RTX PRO 4500)','cap_usd':4.5,'pods':pods,'as_of':time.strftime('%Y-%m-%d %H:%M:%S %Z')},open(D+'spend.json','w'),indent=1)
print(round(tot,3), round(hrs,3))
