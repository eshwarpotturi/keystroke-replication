"""Pack held-out test strokes and the model's guesses into data/try.json for the in-page demo.
Only strokes that were never used for training or validation are included."""
import numpy as np,json,base64,os
KEYS="1234567890qwertyuiopasdfghjklzxcvbnm"
out={}
KEYS_=KEYS
def compact(name,f):
    d=np.load(f); L=[{"k":KEYS_[y],"p":KEYS_[p],"c":round(float(c),3)} for y,p,c in zip(d["y"],d["pred"],d["conf"])]
    out[name]={"n":len(L),"acc":float(np.mean([x["k"]==x["p"] for x in L])),"strokes":L,"nomel":True}
    print(name,len(L),f"acc {out[name]['acc']:.3f}")
for name,f in (("phone_zoom","data/pred_phone_zoom_random_0.npz"),("zoom_phone","data/pred_zoom_phone_random_0.npz"),
               ("rpractical_rmac","data/pred_rpractical_rmac_random_0.npz"),("rmac_rpractical","data/pred_rmac_rpractical_random_0.npz")):
    if os.path.exists(f): compact(name,f)
for dev in ("phone","zoom"):
    for sp in ("random","ordered"):
        f=f"data/pred_{dev}_{dev}_{sp}_0.npz"
        if not os.path.exists(f): continue
        d=np.load(f); L=[]
        for m,y,p,c,t3 in zip(d["mel"].astype(np.float32),d["y"],d["pred"],d["conf"],d["top3"]):
            m=m[:, :48]; lo,hi=np.percentile(m,2),np.percentile(m,99.5); q=np.clip((m-lo)/(hi-lo+1e-9),0,1)
            L.append({"k":KEYS[y],"p":KEYS[p],"c":round(float(c),3),"t":[KEYS[i] for i in t3],"m":base64.b64encode((q*255).astype(np.uint8).tobytes()).decode()})
        out[f"{dev}_{sp}"]={"n":len(L),"acc":float(np.mean([x["k"]==x["p"] for x in L])),"strokes":L}
        print(dev,sp,len(L),f"acc {out[f'{dev}_{sp}']['acc']:.3f}",len({x['k'] for x in L}),"keys covered")
json.dump(out,open("data/try.json","w"),separators=(",",":"))
print("bytes",os.path.getsize("data/try.json"))
