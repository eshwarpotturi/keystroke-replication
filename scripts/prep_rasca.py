"""Pack two public recording sets (R-ASCA repo) in the same format as phone.npz/zoom.npz:
   'rpractical' = authors' original MacBook Pro 16 (M1 Pro) recordings; 'rmac' = a different, newer MacBook Pro (M3), different room and mic.
   Only keys a-z and 0-9 (36 classes), clips padded or cut to 14,400 samples."""
import glob,os,numpy as np,time
from scipy.io import wavfile
R="/home/claude/ksd/repo/R-ASCA"; KEYS="1234567890qwertyuiopasdfghjklzxcvbnm"
def log(*a): print(time.strftime("%Y-%m-%dT%H:%M:%S"),*a,flush=True)
for name,cond in (("rpractical","practical"),("rmac","custom_mac")):
    X,y,sr0=[],[],set()
    for sp in ("train","val","test"):
        for f in glob.glob(f"{R}/{sp}/{cond}/*/*.wav"):
            k=os.path.basename(os.path.dirname(f))
            if k not in KEYS: continue
            sr,d=wavfile.read(f); sr0.add(sr); d=d.astype(np.float32); d=d.mean(1) if d.ndim>1 else d
            d=d[:14400]; d=np.pad(d,(0,14400-len(d))); d/= (np.abs(d).max()+1e-9)
            X.append(d); y.append(KEYS.index(k))
    np.savez(f"data/{name}.npz",X=np.array(X),y=np.array(y),pos=np.zeros(len(y),int),sr=list(sr0)[0])
    log(name,"strokes",len(y),"sample rates",sr0,"keys",len(set(y)),"per key min/max",np.bincount(y).min(),np.bincount(y).max())
