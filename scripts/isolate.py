"""Isolate keystrokes with the authors' own isolator (from the dataset README). Logs a timestamp on every line."""
import sys,time,json,numpy as np,librosa
def log(*a): print(time.strftime("%Y-%m-%dT%H:%M:%S"),*a,flush=True)
def isolator(signal,sr,size,scan,before,after,threshold):
    strokes=[]
    fft=librosa.stft(signal,n_fft=size,hop_length=scan)
    energy=np.abs(np.sum(fft,axis=0)).astype(float)
    peaks=np.where(energy>threshold)[0]
    prev_end=sr*0.1*(-1)
    for p in peaks:
        ts=(p*scan)+size//2
        if ts>prev_end+0.1*sr and ts-before>=0 and ts+after<=len(signal):
            strokes.append(signal[ts-before:ts+after]); prev_end=ts+after
    return strokes
KEYS="1234567890qwertyuiopasdfghjklzxcvbnm"
out={}
for name,folder in (("phone","orig/MBPWavs"),("zoom","orig/Zoom")):
    X,y,fid,info=[],[],[],{}
    for i,k in enumerate(KEYS):
        s,sr=librosa.load(f"../ksd/{folder}/{k.upper() if name=='phone' else k}.wav",sr=None)
        prom,step,st=0.06,0.005,[]
        s=s[int(1*sr):]
        for it in range(2000):
            st=isolator(s,sr,48,24,2400,12000,prom)
            if len(st)==25: break
            prom+= step if len(st)>25 else -step
            step*=0.99
            if prom<=0: break
        info[k]=len(st)
        for j,x in enumerate(st): X.append(x);y.append(i);fid.append(j)
        log(name,k,"found",len(st),"thr",round(prom,4),f"{i+1}/36")
    np.savez(f"data/{name}.npz",X=np.array([np.pad(x,(0,14400-len(x))) for x in X]),y=np.array(y),pos=np.array(fid),sr=sr)
    bad={k:v for k,v in info.items() if v!=25}
    log(name,"total strokes",len(X),"keys not at 25:",bad)
