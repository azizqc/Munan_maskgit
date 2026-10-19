#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,math,random,time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.cluster import MiniBatchKMeans
import matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

SIDE=28; PATCH=4; GRID=7; L=49; K=64; MASK=64; IN_V=65; CLASSES=(0,1,2); NCLASS=3

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def load(path):
    d=np.load(path,allow_pickle=False)
    return d['x_train'].astype(np.float32)/255.,d['y_train'].astype(np.int64),d['x_test'].astype(np.float32)/255.,d['y_test'].astype(np.int64)

def subset012(x,y):
    m=np.isin(y,CLASSES); return x[m],y[m]

def split(x,y,vf,seed):
    rng=np.random.default_rng(seed);tr=[];va=[]
    for c in CLASSES:
        z=np.where(y==c)[0];rng.shuffle(z);nv=max(1,int(round(vf*len(z))));va.append(z[:nv]);tr.append(z[nv:])
    tr=np.concatenate(tr);va=np.concatenate(va);rng.shuffle(tr);rng.shuffle(va);return x[tr],y[tr],x[va],y[va]

def patches(x):
    return x.reshape(len(x),GRID,PATCH,GRID,PATCH).transpose(0,1,3,2,4).reshape(-1,PATCH*PATCH)

def fit_codebook(x,path,seed=7):
    km=MiniBatchKMeans(n_clusters=K,batch_size=8192,n_init=5,max_iter=200,random_state=seed,reassignment_ratio=.005)
    km.fit(patches(x));c=km.cluster_centers_.astype(np.float32);np.save(path,c);return c

def encode(x,c):
    p=patches(x);out=[];c2=(c*c).sum(1)
    for i in range(0,len(p),200000):
        q=p[i:i+200000];d=(q*q).sum(1,keepdims=True)-2*q@c.T+c2[None];out.append(d.argmin(1).astype(np.int64))
    return np.concatenate(out).reshape(len(x),L)

def decode(t,c):
    q=c[np.asarray(t,dtype=np.int64)].reshape(-1,GRID,GRID,PATCH,PATCH).transpose(0,1,3,2,4).reshape(-1,28,28)
    return np.clip(q,0,1).astype(np.float32)

class Block(nn.Module):
    def __init__(self,d,h,ff,drop=.05):
        super().__init__();self.h=h;self.dh=d//h;self.qkv=nn.Linear(d,3*d);self.proj=nn.Linear(d,d);self.n1=nn.LayerNorm(d);self.n2=nn.LayerNorm(d);self.f1=nn.Linear(d,ff);self.f2=nn.Linear(ff,d);self.drop=drop
    def forward(self,x):
        B,N,D=x.shape;qkv=self.qkv(x).reshape(B,N,3,self.h,self.dh).permute(2,0,3,1,4);q,k,v=qkv.unbind(0)
        a=F.scaled_dot_product_attention(q,k,v,dropout_p=self.drop if self.training else 0.0)
        x=self.n1(x+F.dropout(self.proj(a.transpose(1,2).reshape(B,N,D)),self.drop,self.training))
        return self.n2(x+F.dropout(self.f2(F.gelu(self.f1(x))),self.drop,self.training))

class Model(nn.Module):
    def __init__(self,d=128,layers=4,h=4,ff=512,drop=.05):
        super().__init__();self.tok=nn.Embedding(IN_V,d);self.pos=nn.Parameter(torch.randn(1,L,d)*.02);self.cls=nn.Embedding(NCLASS,d);self.blocks=nn.ModuleList([Block(d,h,ff,drop) for _ in range(layers)]);self.dense=nn.Linear(d,d);self.ln=nn.LayerNorm(d);self.bias=nn.Parameter(torch.zeros(K));self.reset()
    def reset(self):
        nn.init.trunc_normal_(self.tok.weight,std=.02,a=-.04,b=.04); nn.init.trunc_normal_(self.pos,std=.02,a=-.04,b=.04);nn.init.trunc_normal_(self.cls.weight,std=.02,a=-.04,b=.04)
        for m in self.modules():
            if isinstance(m,nn.Linear): nn.init.trunc_normal_(m.weight,std=.02,a=-.04,b=.04); nn.init.zeros_(m.bias)
    def forward(self,x,y):
        z=self.tok(x)+self.pos+self.cls(y)[:,None]
        for b in self.blocks:z=b(z)
        z=self.ln(F.gelu(self.dense(z)));return F.linear(z,self.tok.weight[:K],self.bias)

def mask_batch(t):
    B,N=t.shape;r=torch.rand(B,device=t.device);ratio=torch.cos(.5*math.pi*r);nmask=torch.round(ratio*N).long().clamp(1,N)
    score=torch.rand(B,N,device=t.device);rank=score.argsort(1).argsort(1);m=rank<nmask[:,None]
    inp=t.clone();inp[m]=MASK;return inp,m,ratio

def fixed_mask_batch(t,ratio,seed=1):
    B,N=t.shape;n=max(1,min(N,int(round(ratio*N))));g=torch.Generator(device=t.device);g.manual_seed(seed)
    score=torch.rand(B,N,device=t.device,generator=g);rank=score.argsort(1).argsort(1);m=rank<n;inp=t.clone();inp[m]=MASK;return inp,m

def lr_at(step,total,warmup,maxlr,minlr):
    if step<warmup:return maxlr*(step+1)/max(1,warmup)
    p=(step-warmup)/max(1,total-warmup);p=min(1,max(0,p));return minlr+.5*(maxlr-minlr)*(1+math.cos(math.pi*p))

def run_epoch(m,ld,dev,opt=None,global_step=0,total=1,warmup=500,maxlr=4e-4,minlr=2e-5):
    train=opt is not None;m.train(train);ls=ac=n=0.0
    for t,y in ld:
        t=t.to(dev);y=y.to(dev);inp,ma,r=mask_batch(t)
        if train:
            lr=lr_at(global_step,total,warmup,maxlr,minlr)
            for pg in opt.param_groups:pg['lr']=lr
            opt.zero_grad(set_to_none=True)
        z=m(inp,y);loss=F.cross_entropy(z[ma],t[ma]);acc=(z[ma].argmax(-1)==t[ma]).float().mean()
        if train:
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step();global_step+=1
        bs=len(t);ls+=loss.item()*bs;ac+=acc.item()*bs;n+=bs
    return ls/n,ac/n,global_step

@torch.no_grad()
def ratio_eval(m,tok,y,dev,ratios=(.25,.5,.75,.9,1.0),batch=512):
    m.eval();out={}
    T=torch.from_numpy(tok);Y=torch.from_numpy(y)
    for ratio in ratios:
        ls=ac=n=0.0
        for i in range(0,len(T),batch):
            t=T[i:i+batch].to(dev);yy=Y[i:i+batch].to(dev);inp,ma=fixed_mask_batch(t,ratio,seed=12345+i);z=m(inp,yy);loss=F.cross_entropy(z[ma],t[ma]);acc=(z[ma].argmax(-1)==t[ma]).float().mean();bs=len(t);ls+=loss.item()*bs;ac+=acc.item()*bs;n+=bs
        out[str(ratio)]={'ce':ls/n,'masked_token_acc':ac/n}
    return out

@torch.no_grad()
def generate(m,labels,dev,steps=16,temp0=.0,seed=1007,trajectory=False):
    m.eval();labels=labels.to(dev);B=len(labels);cur=torch.full((B,L),MASK,dtype=torch.long,device=dev);initial=torch.full((B,),L,dtype=torch.long,device=dev);g=torch.Generator(device=dev);g.manual_seed(seed)
    rec={'pre':[],'proposal':[],'conf':[],'post':[],'mask_before':[],'mask_after':[],'desired':[]} if trajectory else None
    for st in range(steps):
        unk=cur.eq(MASK)
        if trajectory:rec['pre'].append(cur.cpu().numpy().astype(np.int16));rec['mask_before'].append(unk.cpu().numpy())
        p=m(cur,labels).softmax(-1);s=torch.multinomial(p.reshape(-1,K),1,generator=g).reshape(B,L);prop=torch.where(unk,s,cur);sp=p.gather(-1,prop[...,None]).squeeze(-1);sp_for_rec=torch.where(unk,sp,torch.ones_like(sp))
        if trajectory:rec['proposal'].append(prop.cpu().numpy().astype(np.int16));rec['conf'].append(sp_for_rec.cpu().numpy().astype(np.float32))
        if st==steps-1:cur=prop;desired=torch.zeros(B,dtype=torch.long,device=dev)
        else:
            score_base=torch.where(unk,sp,torch.full_like(sp,float('inf')));frac=(st+1)/steps;desired=torch.floor(initial.float()*math.cos(.5*math.pi*frac)).long();desired=torch.minimum(desired,torch.clamp(unk.sum(1)-1,min=0));u=torch.rand(score_base.shape,device=dev,generator=g).clamp_(1e-6,1-1e-6);gum=-torch.log(-torch.log(u));score=torch.log(score_base.clamp_min(1e-12))+temp0*(1-frac)*gum;score=torch.where(torch.isinf(score_base),torch.full_like(score,float('inf')),score);new=prop.clone()
            for bi in range(B):
                k=int(desired[bi]);
                if k:new[bi,torch.topk(score[bi],k,largest=False).indices]=MASK
            cur=new
        if trajectory:rec['desired'].append(desired.cpu().numpy().astype(np.int16));rec['post'].append(cur.cpu().numpy().astype(np.int16));rec['mask_after'].append(cur.eq(MASK).cpu().numpy())
    if trajectory:rec={k:np.stack(v) for k,v in rec.items()}
    return cur.cpu().numpy().astype(np.int16),rec

class StrongRef(nn.Module):
    def __init__(self):
        super().__init__();self.c1=nn.Conv2d(1,32,3,padding=1);self.c2=nn.Conv2d(32,64,3,padding=1);self.fc1=nn.Linear(64*7*7,128);self.fc2=nn.Linear(128,3)
    def forward(self,x,feat=False):
        x=F.max_pool2d(F.relu(self.c1(x)),2);x=F.max_pool2d(F.relu(self.c2(x)),2);f=F.relu(self.fc1(x.flatten(1)));z=self.fc2(f);return (z,f) if feat else z

def train_ref(x,y,xte,yte,dev,epochs=6):
    yy=y.copy();yt=yte.copy() # labels already 0,1,2
    m=StrongRef().to(dev);o=torch.optim.Adam(m.parameters(),1e-3);ld=DataLoader(TensorDataset(torch.from_numpy(x[:,None]),torch.from_numpy(yy)),batch_size=256,shuffle=True)
    for ep in range(epochs):
        m.train()
        for xb,yb in ld:
            xb=xb.to(dev);yb=yb.to(dev);z=m(xb);loss=F.cross_entropy(z,yb);o.zero_grad();loss.backward();o.step()
    m.eval();xx=torch.from_numpy(xte[:,None]).to(dev);pred=[]
    with torch.no_grad():
        for i in range(0,len(xx),512):pred.append(m(xx[i:i+512]).argmax(1).cpu())
    acc=float((torch.cat(pred).numpy()==yt).mean());return m,acc

@torch.no_grad()
def classify(m,x,dev):
    m.eval();xx=torch.from_numpy(x[:,None].astype(np.float32));ps=[];fs=[]
    for i in range(0,len(xx),512):
        z,f=m(xx[i:i+512].to(dev),True);ps.append(z.softmax(-1).cpu());fs.append(f.cpu())
    return torch.cat(ps).numpy(),torch.cat(fs).numpy()

def save_grid(img,y,path,n=90):
    n=min(n,len(img));cols=15;rows=math.ceil(n/cols);fig,ax=plt.subplots(rows,cols,figsize=(cols,rows*1.15));ax=np.asarray(ax).reshape(-1)
    for i,a in enumerate(ax):
        a.axis('off')
        if i<n:a.imshow(img[i],cmap='gray',vmin=0,vmax=1,interpolation='nearest');a.set_title(str(int(y[i])),fontsize=7)
    plt.tight_layout();fig.savefig(path,dpi=180,bbox_inches='tight');plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--epochs',type=int,default=80);ap.add_argument('--batch',type=int,default=256);ap.add_argument('--d',type=int,default=128);ap.add_argument('--layers',type=int,default=4);ap.add_argument('--heads',type=int,default=4);ap.add_argument('--ff',type=int,default=512);ap.add_argument('--drop',type=float,default=.05);ap.add_argument('--threads',type=int,default=8);ap.add_argument('--seed',type=int,default=7);ap.add_argument('--steps',type=int,default=16);ap.add_argument('--temp',type=float,default=0.0);ap.add_argument('--lr',type=float,default=4e-4);ap.add_argument('--min-lr',type=float,default=2e-5);ap.add_argument('--warmup',type=int,default=500);ap.add_argument('--eval-every',type=int,default=10);ap.add_argument('--skip-final',action='store_true');a=ap.parse_args()
    seed_all(a.seed);torch.set_num_threads(a.threads);dev=torch.device('cpu');a.out.mkdir(parents=True,exist_ok=True)
    xall,yall,xte0,yte0=load(a.data);x012,y012=subset012(xall,yall);xte,yte=subset012(xte0,yte0);xtr,ytr,xva,yva=split(x012,y012,.1,a.seed)
    cp=a.out/'codebook.npy';cent=fit_codebook(xtr,cp,a.seed) if not cp.exists() else np.load(cp);tr=encode(xtr,cent);va=encode(xva,cent);te=encode(xte,cent)
    # tokenizer ceiling
    rte=decode(te,cent);mse=float(np.mean((rte-xte)**2));psnr=float(10*np.log10(1/max(mse,1e-12)))
    ld=DataLoader(TensorDataset(torch.from_numpy(tr),torch.from_numpy(ytr)),batch_size=a.batch,shuffle=True);vl=DataLoader(TensorDataset(torch.from_numpy(va),torch.from_numpy(yva)),batch_size=a.batch)
    m=Model(a.d,a.layers,a.heads,a.ff,a.drop).to(dev);opt=torch.optim.AdamW(m.parameters(),a.lr,betas=(.9,.96),weight_decay=.05);best=1e9;hist=[];global_step=0;total=a.epochs*len(ld);t0=time.time()
    print(json.dumps({'train':len(xtr),'val':len(xva),'test':len(xte),'params':sum(p.numel() for p in m.parameters()),'tok_mse':mse,'tok_psnr':psnr,'batches_per_epoch':len(ld)}),flush=True)
    for ep in range(1,a.epochs+1):
        tc,ta,global_step=run_epoch(m,ld,dev,opt,global_step,total,a.warmup,a.lr,a.min_lr);vc,vaa,_=run_epoch(m,vl,dev,None);rec={'epoch':ep,'train_ce':tc,'train_acc':ta,'val_ce':vc,'val_acc':vaa,'lr':opt.param_groups[0]['lr'],'elapsed_s':time.time()-t0};hist.append(rec)
        if vc<best:best=vc;torch.save({'model':m.state_dict(),'epoch':ep,'best':best},a.out/'best.pt')
        if ep%a.eval_every==0 or ep==1: print(json.dumps(rec),flush=True)
    with (a.out/'history.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=hist[0].keys());w.writeheader();w.writerows(hist)
    if a.skip_final:
        print('TRAINING_ONLY_DONE', flush=True); return
    ck=torch.load(a.out/'best.pt',map_location=dev,weights_only=False);m.load_state_dict(ck['model'])
    ratio=ratio_eval(m,te,yte,dev)
    # strong evaluator, only real data training
    ref,refacc=train_ref(xtr,ytr,xte,yte,dev,epochs=6);p_orig,f_orig=classify(ref,xte,dev);p_rec,f_rec=classify(ref,rte,dev)
    labs=torch.tensor(np.tile(np.array([0,1,2],dtype=np.int64),100),dtype=torch.long);tok,rec=generate(m,labs,dev,a.steps,a.temp,a.seed+1000,True);img=decode(tok,cent);pg,fg=classify(ref,img,dev);pred=pg.argmax(1)
    save_grid(img,labs.numpy(),a.out/'generation.png',n=90)
    # save original/recon comparison
    inds=np.concatenate([np.where(yte==c)[0][:10] for c in CLASSES]);comp=np.concatenate([xte[inds],rte[inds]],axis=0);clab=np.concatenate([yte[inds],yte[inds]]);save_grid(comp,clab,a.out/'tokenizer_original_then_recon.png',n=len(comp))
    np.savez_compressed(a.out/'trajectory.npz',labels=labs.numpy(),final_tokens=tok,images=img,pre_state=rec['pre'],proposal=rec['proposal'],proposal_confidence=rec['conf'],post_state=rec['post'],mask_before=rec['mask_before'],mask_after=rec['mask_after'],desired_mask_count=rec['desired'])
    met={'status':'B0_MNIST012_CONVENTIONAL','classes':[0,1,2],'train_samples':len(xtr),'val_samples':len(xva),'test_samples':len(xte),'params':sum(p.numel() for p in m.parameters()),'best_epoch':int(ck['epoch']),'best_val_ce':float(ck['best']),'tokenizer_mse':mse,'tokenizer_psnr_db':psnr,'ref_classifier_real_test_acc':refacc,'ref_classifier_tokenizer_recon_acc':float((p_rec.argmax(1)==yte).mean()),'ratio_eval':ratio,'generation_n':len(tok),'generated_condition_match_rate':float((pred==labs.numpy()).mean()),'generated_mean_confidence':float(pg.max(1).mean()),'generated_class_counts':np.bincount(pred,minlength=3).tolist(),'unique_token_sequences':int(np.unique(tok,axis=0).shape[0]),'steps':a.steps,'temperature':a.temp,'trajectory_mask_counts_first':[int(x) for x in rec['mask_before'].sum(2)[:,0]]}
    (a.out/'metrics.json').write_text(json.dumps(met,indent=2));print('FINAL',json.dumps(met),flush=True)
if __name__=='__main__':main()
