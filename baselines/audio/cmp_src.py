import subprocess, numpy as np, sys
D=__import__('os').path.expandvars('$SEAMLESS_ROOT/naturalistic/dev')
fid=sys.argv[1]; other=sys.argv[2]; t0=float(sys.argv[3])
def rd(path, ch, extra=[]):
    r=subprocess.run(['ffmpeg','-v','error','-ss',str(t0),'-t','8','-i',path]+extra+['-ac',str(ch),'-ar','16000','-f','f32le','-'],capture_output=True)
    return np.frombuffer(r.stdout,np.float32).reshape(-1,ch)
m=rd(f'{D}/video/{fid}.mp4',2)
own=rd(f'{D}/audio/{fid}.wav',1)[:,0]
sess=fid.rsplit('_',1)[0]
oth=rd(f'{D}/audio/{sess}_{other}.wav',1)[:,0]
n=min(len(m),len(own),len(oth))
def c(a,b): a=a[:n];b=b[:n]; return float(np.corrcoef(a,b)[0,1])
db=lambda x:20*np.log10(np.sqrt(np.mean(x**2))+1e-12)
print('L-R corr',c(m[:,0],m[:,1]),'L dB',db(m[:,0]),'R dB',db(m[:,1]))
for nm,w in (('own',own),('other',oth)): print(nm,'dB',db(w),'corr L',c(m[:,0],w),'corr R',c(m[:,1],w))
def env(x,hop=160):
    k=len(x)//hop; return np.sqrt((x[:k*hop].reshape(k,hop)**2).mean(1))
eL=env(m[:,0]); eo=env(own); ex=env(oth)
def xc(a,b,L=50):
    best=None
    for l in range(-L,L+1):
        if l>=0: aa,bb=a[l:],b[:len(b)-l]
        else: aa,bb=a[:l],b[-l:]
        k=min(len(aa),len(bb)); r=np.corrcoef(np.log(aa[:k]+1e-6),np.log(bb[:k]+1e-6))[0,1]
        if best is None or r>best[1]: best=(l,r)
    return best
print('env xcorr L~own (lag 10ms, r)',xc(eL,eo),' L~other',xc(eL,ex))
# sample-level xcorr
from numpy.fft import rfft,irfft
a=m[:n,0]-m[:n,0].mean(); b=own[:n]-own[:n].mean()
N=1<<int(np.ceil(np.log2(2*n))); cc=irfft(rfft(a,N)*np.conj(rfft(b,N)),N)
cc=np.concatenate([cc[-n+1:],cc[:n]]); i=np.argmax(np.abs(cc)); print('sample xcorr lag',i-(n-1),'samples@16k norm',cc[i]/np.sqrt((a*a).sum()*(b*b).sum()))
