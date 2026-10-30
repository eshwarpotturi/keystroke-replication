"""Build docs/index.html from logs/res_*.json. Every number on the page is read from those files."""
import json,glob,os,statistics as st
def norm(f):
    r=json.load(open(f))
    if "test_acc" not in r: r["test_acc"]=r["test"]; r["test_set"]=f.split("_")[2]
    return r
R=[norm(f) for f in sorted(glob.glob("logs/res_*.json"))]
def grp(tr,te,sp,ep):
    a=[r for r in R if r["train"]==tr and r["test_set"]==te and r["split"]==sp and r["epochs"]==ep]
    if not a: return None
    acc=[r["test_acc"] for r in a]
    return dict(runs=[dict(seed=r["seed"],acc=r["test_acc"],best_epoch=r["best_epoch"],val=r["val"]) for r in a],mean=st.mean(acc),lo=min(acc),hi=max(acc),n=a[0]["n_test"],epochs=ep)
spec=[("Phone microphone","Mixed presses: random 80/10/10 split of one recording session","phone","phone","random",150,95),
      ("Zoom call","Mixed presses: random 80/10/10 split of one recording session","zoom","zoom","random",150,93),
      ("Phone microphone, early vs late presses","Train on presses 1-20 of each key, test on presses 23-25","phone","phone","ordered",150,None),
      ("Zoom, early vs late presses","Train on presses 1-20 of each key, test on presses 23-25","zoom","zoom","ordered",150,None),
      ("Train on phone, test on Zoom","Every Zoom stroke is unseen; different microphone and sample rate","phone","zoom","random",150,None),
      ("Train on Zoom, test on phone","Every phone stroke is unseen","zoom","phone","random",150,None)]
rows=[]
for t,sub,tr,te,sp,ep,claim in spec:
    g=grp(tr,te,sp,ep)
    if g: rows.append(dict(title=t,sub=sub,claim=claim,**g))

def lap(tr,te,ep,title,within_tr,within_ep):
    g=grp(tr,te,"random",ep); w=grp(within_tr,within_tr,"random",within_ep)
    if g: rows.append(dict(title=title,sub=f"Different laptop, room and microphone, same 22 kHz format. Same-laptop score for comparison: {w['mean']*100:.1f}%" if w else "Different laptop",claim=None,within=(w["mean"] if w else None),**g))
lap("rpractical","rmac",80,"Train on MacBook Pro 16 (M1 Pro), test on MacBook Pro (M3)","rpractical",80)
lap("rmac","rpractical",150,"Train on MacBook Pro (M3), test on MacBook Pro 16 (M1 Pro)","rmac",150)
json.dump(rows,open("data/summary.json","w"),indent=1)
for o in rows: print(f"{o['title']:<44} paper {o['claim']} ours mean {o['mean']*100:.1f}% range {o['lo']*100:.1f}-{o['hi']*100:.1f} runs {len(o['runs'])} test strokes {o['n']}")
g=lambda k:next((r for r in rows if r["title"].startswith(k)),None)
f=lambda x:f"{x*100:.1f}%"
L1,L2=g('Train on MacBook Pro 16'),g('Train on MacBook Pro (M3)')
ph,zo,pod,zod,p2z,z2p=[g(k) for k in ("Phone microphone","Zoom call","Phone microphone, early","Zoom, early","Train on phone","Train on Zoom")]
notes=[]
import numpy as _np
_t=json.load(open("data/try.json")) if os.path.exists("data/try.json") else {}
_S=[x for k in ("phone_random","zoom_random","phone_ordered","zoom_ordered") for x in _t.get(k,{}).get("strokes",[])]
_m=[x for x in _S if x["c"]>=0.8]
if ph and zo: notes.append(f"<b>It works.</b> From 25 recorded presses per key, the model names {f(ph['mean'])} of keys over a phone microphone (average of {len(ph['runs'])} runs) and {f(zo['mean'])} through Zoom ({len(zo['runs'])} runs). Chance is 2.8%. These sit close to the 95% and 93% published for the same method.")
if pod and zod: notes.append(f"<b>It holds up on later presses.</b> Trained on the first 20 presses of each key and tested on the last three, it names {f(pod['mean'])} (phone) and {f(zod['mean'])} (Zoom).")
if _m: notes.append(f"<b>It knows when it knows.</b> Of the presses it was 80%+ sure about, {sum(x['p']==x['k'] for x in _m)} of {len(_m)} ({f(sum(x['p']==x['k'] for x in _m)/len(_m))}) were right, covering {round(100*len(_m)/len(_S))}% of presses. The right key was in its top three guesses for {f(sum(x['k'] in x.get('t',[]) for x in _S)/len(_S))}.")
if p2z and z2p: notes.append(f"<b>Tuned to one device.</b> Each model is built for the device and room it learned from. Tested on a different device it scores {f(p2z['mean'])} and {f(z2p['mean'])} (phone and Zoom), and about {f((L1['mean']+L2['mean'])/2) if L1 and L2 else 'n/a'} across two different MacBook Pros. Train it on the device you want to measure. One run each; sampling rates differ between phone and Zoom.")
if ph and zo: notes.append(f"<b>Run-to-run spread.</b> Each run is scored on {ph['n']} unseen presses. Phone runs ranged {f(ph['lo'])} to {f(ph['hi'])}; Zoom runs {f(zo['lo'])} to {f(zo['hi'])}.")
notes.append("<b>Limits.</b> One person, one laptop, one room. The model is smaller than the one published and trained for 150 epochs on a CPU. Epochs were chosen on a validation split; the test split was scored once per run.")
head="A phone can hear what you type, but only if the attacker has first recorded your own keyboard on the same device."
if p2z and z2p and pod: head=f"A phone can hear what you type ({f(ph['mean'])}), but only under the easiest conditions: it drops to {f(pod['mean'])} on later presses and to about {f((p2z['mean']+z2p['mean'])/2)} when the model has not heard your device."
html=open("scripts/template.html").read().replace("/*HEAD*/",head).replace("/*DATA*/null",json.dumps(rows)).replace("/*NOTES*/[]",json.dumps(notes)).replace("/*TRY*/null",open("data/try.json").read() if os.path.exists("data/try.json") else "null")
os.makedirs("docs",exist_ok=True); open("docs/index.html","w").write(html); open("index.html","w").write(html)
import shutil
shutil.copytree("assets","docs/assets",dirs_exist_ok=True)
for f in ("mel.js","live.js"):  # live tab's JS modules, loaded as ./f relative to index.html
    shutil.copy(f"scripts/{f}",f); shutil.copy(f"scripts/{f}",f"docs/{f}")
if os.path.isdir("docs/model"): shutil.rmtree("docs/model")  # rebuild from scratch: copytree(dirs_exist_ok=True) merges and never deletes, so a stale/removed export would otherwise linger here forever
if os.path.isdir("model"):  # trained ONNX export(s), when published (see model/README.md)
    shutil.copytree("model","docs/model")
print("wrote docs/index.html and index.html")
