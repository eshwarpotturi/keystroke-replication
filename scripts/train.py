"""Replicate the keystroke classifier: compact CoAtNet-style model, 36 keys, 80/10/10 split.
Epochs are chosen on the validation split only. The test split is scored once per run, after training stops.
usage: python train.py <train_set> <test_set> <seed> [epochs] [split]"""
import sys,time,json,math,random,numpy as np,torch,torch.nn as nn,librosa
import os; torch.set_num_threads(int(os.environ.get('THREADS','2')))
def log(*a): print(time.strftime("%Y-%m-%dT%H:%M:%S"),*a,flush=True)
tr_name,te_name,seed=sys.argv[1],sys.argv[2],int(sys.argv[3]); EPOCHS=int(sys.argv[4]) if len(sys.argv)>4 else 120
SPLIT=sys.argv[5] if len(sys.argv)>5 else "random"
import os as _os
_rf=f"logs/res_{tr_name}_{te_name}_{SPLIT}_{seed}{_os.environ.get('TAG','')}.json"
_pf=f"data/pred_{tr_name}_{te_name}_{SPLIT}_{seed}{_os.environ.get('TAG','')}.npz"
if _os.path.exists(_rf) and _os.path.exists(_pf) and json.load(open(_rf)).get("epochs")==EPOCHS:
    log("already done at this epoch count, skipping:",_rf); sys.exit(0)
random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
def mel_fn(sr):
    fb=torch.tensor(librosa.filters.mel(sr=sr,n_fft=1536,n_mels=64,fmin=100,fmax=2500),dtype=torch.float32)
    win=torch.hann_window(1024)
    def f(x):  # x: (B,T)
        S=torch.stft(x,1536,hop_length=300,win_length=1024,window=win,return_complex=True,center=True).abs()**2
        M=torch.matmul(fb,S)  # (B,64,frames)
        return torch.log2(M+1e-10)
    return f
def load(name):
    d=np.load(f"data/{name}.npz"); X=torch.tensor(d["X"],dtype=torch.float32); return X,torch.tensor(d["y"]),d["pos"],int(d["sr"])
def split(n_pos,pos,seed,mode):
    idx=np.arange(len(pos)); rng=np.random.RandomState(seed)
    if mode=="random":
        rng.shuffle(idx); a,b=int(.8*len(idx)),int(.9*len(idx)); return idx[:a],idx[a:b],idx[b:]
    # "ordered": first 20 presses of each key train, next 2 val, last 3 test (never mixes early and late presses)
    return idx[pos<20],idx[(pos>=20)&(pos<22)],idx[pos>=22]
class MB(nn.Module):
    def __init__(s,i,o,stride):
        super().__init__(); m=i*2
        s.f=nn.Sequential(nn.BatchNorm2d(i),nn.Conv2d(i,m,1),nn.GELU(),nn.Conv2d(m,m,3,stride,1,groups=m),nn.GELU(),nn.Conv2d(m,o,1))
        s.sc=nn.Sequential(nn.MaxPool2d(stride) if stride>1 else nn.Identity(),nn.Conv2d(i,o,1) if i!=o else nn.Identity())
    def forward(s,x): return s.sc(x)+s.f(x)
class Att(nn.Module):
    def __init__(s,d,heads=4):
        super().__init__(); s.n1=nn.LayerNorm(d); s.a=nn.MultiheadAttention(d,heads,batch_first=True); s.n2=nn.LayerNorm(d)
        s.m=nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Linear(d*2,d))
    def forward(s,x):
        h=s.n1(x); x=x+s.a(h,h,h,need_weights=False)[0]; return x+s.m(s.n2(x))
class CoAtLite(nn.Module):  # conv stem -> 2 MBConv stages -> 2 transformer stages (the CoAtNet recipe, narrower)
    def __init__(s,nc=36,w=(24,48,96,128)):
        super().__init__()
        s.stem=nn.Sequential(nn.Conv2d(1,w[0],3,2,1),nn.GELU(),nn.Conv2d(w[0],w[0],3,1,1))
        s.s1=nn.Sequential(MB(w[0],w[1],2),MB(w[1],w[1],1)); s.s2=nn.Sequential(MB(w[1],w[2],2),MB(w[2],w[2],1))
        s.proj=nn.Conv2d(w[2],w[3],1); s.pos=nn.Parameter(torch.zeros(1,48,w[3])); s.t=nn.Sequential(Att(w[3]),Att(w[3]))
        s.head=nn.Sequential(nn.LayerNorm(w[3]),nn.Dropout(0.3),nn.Linear(w[3],nc))
    def forward(s,x):
        x=s.proj(s.s2(s.s1(s.stem(x)))); B,C,H,W=x.shape; t=x.flatten(2).transpose(1,2)
        return s.head(s.t(t+s.pos).mean(1))
Xtr_all,ytr_all,ptr,sr_tr=load(tr_name); Xte_all,yte_all,pte,sr_te=load(te_name)
mtr,mte=mel_fn(sr_tr),mel_fn(sr_te)
def feats(X,mel): 
    M=mel(X); return M.unsqueeze(1)[...,:48]
if tr_name==te_name:
    a,b,c=split(len(ptr),ptr,seed,SPLIT); Xtr,ytr,Xva,yva,Xte,yte=Xtr_all[a],ytr_all[a],Xtr_all[b],ytr_all[b],Xtr_all[c],ytr_all[c]
else:  # transfer: train/val from the training medium, test = every stroke of the other medium
    a,b,_=split(len(ptr),ptr,seed,SPLIT); Xtr,ytr,Xva,yva=Xtr_all[a],ytr_all[a],Xtr_all[b],ytr_all[b]; Xte,yte=Xte_all,yte_all
log(f"train {tr_name} test {te_name} split {SPLIT} seed {seed}: train {len(ytr)} val {len(yva)} test {len(yte)}")
def aug(x,mel):
    sh=torch.randint(0,int(x.shape[1]*0.3)+1,(x.shape[0],)); x=torch.stack([torch.roll(r,int(s)) for r,s in zip(x,sh)])
    M=mel(x).unsqueeze(1)[...,:48]
    for _ in range(2):
        f=np.random.randint(0,7); f0=np.random.randint(0,64-f); M[:,:,f0:f0+f]=M.mean()
        t=np.random.randint(0,7); t0=np.random.randint(0,M.shape[3]-t); M[:,:,:,t0:t0+t]=M.mean()
    return M
mu,sd=None,None
Mva,Mte=feats(Xva,mtr),feats(Xte,mte)
with torch.no_grad(): mu,sd=feats(Xtr,mtr).mean(),feats(Xtr,mtr).std()
norm=lambda M:(M-mu)/sd
model=CoAtLite(); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.05)
sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=2e-3,total_steps=EPOCHS*math.ceil(len(ytr)/32))
best,best_state,hist=-1,None,[]
def evaluate(M,y):
    model.eval(); 
    with torch.no_grad(): return (model(norm(M)).argmax(1)==y).float().mean().item()
for ep in range(1,EPOCHS+1):
    model.train(); perm=torch.randperm(len(ytr)); tl=0
    for i in range(0,len(perm),32):
        j=perm[i:i+32]; loss=nn.functional.cross_entropy(model(norm(aug(Xtr[j],mtr))),ytr[j],label_smoothing=0.1)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl+=loss.item()*len(j)
    va=evaluate(Mva,yva); hist.append((ep,round(tl/len(ytr),4),round(va,4)))
    if va>best: best,best_ep,best_state=va,ep,{k:v.clone() for k,v in model.state_dict().items()}
    if ep%10==0 or ep==1: log(f"epoch {ep}/{EPOCHS} loss {tl/len(ytr):.3f} val {va:.3f} best_val {best:.3f}@{best_ep}")
model.load_state_dict(best_state); test=evaluate(Mte,yte)
model.eval()
with torch.no_grad(): P=torch.softmax(model(norm(Mte)),1)
np.savez_compressed(f"data/pred_{tr_name}_{te_name}_{SPLIT}_{seed}{os.environ.get('TAG','')}.npz",mel=Mte[:,0].numpy().astype(np.float16),y=yte.numpy(),pred=P.argmax(1).numpy(),conf=P.max(1).values.numpy(),top3=P.topk(3,1).indices.numpy())   # test scored ONCE, with the epoch chosen on validation
log(f"RESULT train={tr_name} test={te_name} split={SPLIT} seed={seed} best_epoch={best_ep} val={best:.4f} test={test:.4f}")
json.dump({"train":tr_name,"test_set":te_name,"split":SPLIT,"seed":seed,"epochs":EPOCHS,"best_epoch":best_ep,"val":best,"test_acc":test,"n_test":len(yte),"hist":hist},open(f"logs/res_{tr_name}_{te_name}_{SPLIT}_{seed}{os.environ.get('TAG','')}.json","w"))
