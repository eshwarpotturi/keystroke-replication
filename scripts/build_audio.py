"""Write the real audio of every held-out press shown on the page (16 kHz, 16-bit, base64) to audio_<set>.json.
The split is rebuilt exactly as train.py does it; labels are checked against the saved predictions."""
import numpy as np,json,base64,os
from scipy.signal import resample_poly
from math import gcd
KEYS="1234567890qwertyuiopasdfghjklzxcvbnm"; OUT=16000
def split(pos,seed,mode):
    idx=np.arange(len(pos)); rng=np.random.RandomState(seed)
    if mode=="random":
        rng.shuffle(idx); a,b=int(.8*len(idx)),int(.9*len(idx)); return idx[b:]
    return idx[pos>=22]
os.makedirs("assets",exist_ok=True); total=0
for dev in ("phone","zoom"):
    d=np.load(f"data/{dev}.npz"); X,y,pos,sr=d["X"],d["y"],d["pos"],int(d["sr"])
    for sp in ("random","ordered"):
        pf=f"data/pred_{dev}_{dev}_{sp}_0.npz"
        if not os.path.exists(pf): continue
        te=split(pos,0,sp); p=np.load(pf)
        assert len(te)==len(p["y"]) and (y[te]==p["y"]).all(), f"split mismatch {dev} {sp}"
        g=gcd(sr,OUT); A=[]
        for i in te:
            w=resample_poly(X[i].astype(np.float32),OUT//g,sr//g); w=w/(np.abs(w).max()+1e-9)*0.9
            A.append(base64.b64encode((w*32767).astype(np.int16).tobytes()).decode())
        json.dump({"sr":OUT,"a":A},open(f"assets/audio_{dev}_{sp}.json","w"),separators=(",",":"))
        sz=os.path.getsize(f"assets/audio_{dev}_{sp}.json"); total+=sz; print(dev,sp,len(A),"presses",sz//1024,"KB, labels match")
print("total KB",total//1024)
