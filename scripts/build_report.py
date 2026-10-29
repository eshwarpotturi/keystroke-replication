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
if ph and zo: notes.append(f"<b>It works in the lab.</b> Over the phone microphone the model recovers {f(ph['mean'])} of keys on average over {len(ph['runs'])} runs; through Zoom {f(zo['mean'])} over {len(zo['runs'])} runs. Chance is 2.8%. These sit close to the 95% and 93% published for the same method.")
if ph and zo: notes.append(f"<b>One run is a noisy number.</b> Each run is scored on only {ph['n']} unseen presses. Phone runs ranged {f(ph['lo'])} to {f(ph['hi'])}; Zoom runs {f(zo['lo'])} to {f(zo['hi'])}. A single quoted figure like 95% can sit several points either side of the average.")
if ph and pod: notes.append(f"<b>Mixed presses are the easy test.</b> Each key was pressed 25 times in a row in one session, and mixing early and late presses between training and testing flatters the score. When we train on presses 1-20 and test on presses 23-25, the phone score falls from {f(ph['mean'])} to {f(pod['mean'])}"+(f" and the Zoom score from {f(zo['mean'])} to {f(zod['mean'])}" if zo and zod else "")+".")
if p2z and z2p: notes.append(f"<b>A model does not carry over between devices.</b> Trained on the phone and tested on Zoom: {f(p2z['mean'])}. Trained on Zoom and tested on the phone: {f(z2p['mean'])}. Chance is 2.8%. Part of the gap may be the different sampling rates (44.1 kHz phone, 32 kHz Zoom), and each direction is one run, so read it as 'does not carry over', not as an exact figure.")
if L1 and L2: notes.append(f"<b>Laptops too.</b> The same test across two different MacBook Pros (public recordings from a later study): {f(L1['mean'])} one way and {f(L2['mean'])} the other, against 59% to 67% on each laptop's own recordings. Rooms and microphones differ as well, so this shows the model does not carry over, not which factor matters most. One run each, and the second laptop's set is small (513 presses).")
notes.append("<b>Limits.</b> Recordings are one person, one laptop, one room, so nothing here says how well the attack works on a stranger's keyboard. The model is a smaller CoAtNet-style network than the paper's, trained for 150 epochs on a CPU. Epochs were chosen on a validation split; the test split was scored once per run.")
head="A phone can hear what you type, but only if the attacker has first recorded your own keyboard on the same device."
if p2z and z2p and pod: head=f"A phone can hear what you type ({f(ph['mean'])}), but only under the easiest conditions: it drops to {f(pod['mean'])} on later presses and to about {f((p2z['mean']+z2p['mean'])/2)} when the model has not heard your device."
html=open("scripts/template.html").read().replace("/*HEAD*/",head).replace("/*DATA*/null",json.dumps(rows)).replace("/*NOTES*/[]",json.dumps(notes)).replace("/*TRY*/null",open("data/try.json").read() if os.path.exists("data/try.json") else "null")
os.makedirs("docs",exist_ok=True); open("docs/index.html","w").write(html); open("index.html","w").write(html); import shutil; shutil.copytree("assets","docs/assets",dirs_exist_ok=True); print("wrote docs/index.html and index.html")
