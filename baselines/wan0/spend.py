import json,time,os
D=__import__('os').path.expandvars('$WORK/wan0/')
r=lambda f: float(open(D+f).read()) if os.path.exists(D+f) else None
now=time.time()
p=dict(id='ib1g13mwi1dtsj',name='r2r-wan0',gpu='NVIDIA RTX PRO 4500 Blackwell',rate=0.72,start=r('pod_start_epoch'),end=r('pod_end_epoch') or now)
p['hours']=round((p['end']-p['start'])/3600,4); p['usd']=round(p['hours']*p['rate'],4); p['terminated']=os.path.exists(D+'pod_end_epoch')
json.dump({'usd':round(p['usd'],3),'pod_hours':round(p['hours'],3),'rate':0.72,'cap_usd':4.0,'pods':[p],'as_of':time.strftime('%Y-%m-%d %H:%M:%S %Z')},open(D+'spend.json','w'),indent=1)
print(p['usd'],p['hours'])
