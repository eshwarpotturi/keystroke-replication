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
spec=[("Phone microphone","Same recording session, random 80/10/10 split (the paper's setup)","phone","phone","random",150,95),
      ("Zoom call","Same recording session, random 80/10/10 split (the paper's setup)","zoom","zoom","random",150,93),
      ("Phone microphone, short run (90 epochs)","Matched baseline for the two stricter checks","phone","phone","random",90,None),
      ("Phone microphone, early vs late presses","Train on presses 1-20 of each key, test on presses 23-25","phone","phone","ordered",90,None),
      ("Zoom, early vs late presses","Train on presses 1-20 of each key, test on presses 23-25","zoom","zoom","ordered",90,None),
      ("Train on phone, test on Zoom","Every Zoom stroke is unseen; different microphone and sample rate","phone","zoom","random",90,None),
      ("Train on Zoom, test on phone","Every phone stroke is unseen","zoom","phone","random",90,None)]
rows=[]
for t,sub,tr,te,sp,ep,claim in spec:
    g=grp(tr,te,sp,ep)
    if g: rows.append(dict(title=t,sub=sub,claim=claim,**g))
json.dump(rows,open("data/summary.json","w"),indent=1)
for o in rows: print(f"{o['title']:<44} paper {o['claim']} ours mean {o['mean']*100:.1f}% range {o['lo']*100:.1f}-{o['hi']*100:.1f} runs {len(o['runs'])} test strokes {o['n']}")
g=lambda k:next((r for r in rows if r["title"].startswith(k)),None)
ph,zo,bl,od=g("Phone microphone"),g("Zoom call"),g("Phone microphone, short"),g("Phone microphone, early")
f=lambda x:f"{x*100:.1f}%"
notes=[f"<b>The headline holds.</b> Over the phone microphone we get {f(ph['mean'])} on average over {len(ph['runs'])} runs (paper: 95%); over Zoom {f(zo['mean'])} over {len(zo['runs'])} runs (paper: 93%). Chance is 2.8%.",
f"<b>One run is a noisy number.</b> Each run is scored on only {ph['n']} unseen presses. Phone runs ranged {f(ph['lo'])} to {f(ph['hi'])}; Zoom runs {f(zo['lo'])} to {f(zo['hi'])}. A single quoted figure like 95% can sit several points either side of the average.",
f"<b>The paper's split is the easy one.</b> Each key was pressed 25 times in a row in one session, and the paper mixes early and late presses between train and test. When we train on presses 1-20 and test on presses 23-25, the phone score falls from {f(bl['mean'])} to {f(od['mean'])} at the same training length (one run each, so treat the gap as indicative).",
"<b>Not done in the time available:</b> Zoom early-vs-late, training on one device and testing on the other, and the 150-epoch version of the stricter test. Recordings are one person, one laptop, one room, so nothing here says how well the attack works on a stranger's keyboard.",
"<b>Differences from the paper:</b> a smaller CoAtNet-style model (the paper's was larger and trained longer), 150 epochs not 1,100, CPU only. Epochs were chosen on a validation split; the test split was scored once per run."]
html=open("scripts/template.html").read().replace("/*DATA*/null",json.dumps(rows)).replace("/*NOTES*/[]",json.dumps(notes))
os.makedirs("docs",exist_ok=True); open("docs/index.html","w").write(html); print("wrote docs/index.html")
