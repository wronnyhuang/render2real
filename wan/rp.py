#!/usr/bin/env python3
import json,os,sys,urllib.request,urllib.error
def gql(q):
    r=urllib.request.Request('https://api.runpod.io/graphql',data=json.dumps({'query':q}).encode(),headers={'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY'],'Content-Type':'application/json','User-Agent':'r2r'})
    return json.loads(urllib.request.urlopen(r,timeout=30).read())
def rest(method,path,body=None):
    r=urllib.request.Request('https://rest.runpod.io/v1'+path,data=None if body is None else json.dumps(body).encode(),method=method,headers={'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY'],'Content-Type':'application/json','User-Agent':'r2r'})
    try:
        raw=urllib.request.urlopen(r,timeout=30).read(); return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e: return {'error':e.code,'body':e.read().decode()[:500]}
if __name__=='__main__':
    c=sys.argv[1]
    if c=='bal': print(json.dumps(gql('{ myself { clientBalance currentSpendPerHr } }')))
    elif c=='pods':
        for p in gql('{ myself { pods { id name desiredStatus costPerHr machine { gpuDisplayName } } } }')['data']['myself']['pods']: print(json.dumps(p))
    elif c=='gpus':
        d=gql('{ gpuTypes { id displayName memoryInGb communityPrice securePrice lowestPrice(input:{gpuCount:1}){ minimumBidPrice uninterruptablePrice stockStatus } } }')
        for g in d['data']['gpuTypes']:
            if 20<=g['memoryInGb']<=32: print(json.dumps(g))
    elif c=='get': print(json.dumps(rest('GET','/pods/'+sys.argv[2])))
    elif c=='create': print(json.dumps(rest('POST','/pods',json.loads(sys.argv[2]))))
    elif c=='delete': print(json.dumps(rest('DELETE','/pods/'+sys.argv[2])))

def ssh_info(pid):
    d=gql('{ pod(input:{podId:"%s"}) { id desiredStatus costPerHr runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } machine { gpuDisplayName } } }'%pid)
    return d['data']['pod']
if __name__=='__main__' and sys.argv[1]=='ssh':
    p=ssh_info(sys.argv[2]); print(json.dumps(p))
