#!/usr/bin/env python3
"""MNIST digit-one 10x10 initial-state experiment using the original R6B.3 MSDL E/M readout dynamics.

This is a faithful *readout-level* experiment, not the full MaskGIT/P3 network:
the full route needs a frozen SNN-VQ tokenizer checkpoint absent from this repo.
Original dynamics/learning source:
SNN_TRANSFORM_R6B3_HG_DYNAMICS_STAGE4_DIRECT_SPIKE.py
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from urllib.request import urlretrieve

import numpy as np
from numba import njit

# Exact canonical E/M dynamics from R6B.3.
FORCE_DT_MS = 80.0
AF, AS = 0.62, 0.92
WF, WS = 0.70, 0.30
RF, RS = -math.log(AF)/FORCE_DT_MS, -math.log(AS)/FORCE_DT_MS
A_IZH = 1e-3
S_IZH = -(RF + RS)
P_IZH = RF*RS
Q_IZH = S_IZH + A_IZH
B_IZH = Q_IZH + P_IZH/A_IZH
V0 = (Q_IZH - 5.0)/0.08
U0 = B_IZH*V0
I0 = -0.04*V0*V0 - 5.0*V0 - 140.0 + U0
NUM_S = WF*RF + WS*RS
NUM_0 = RF*RS
G_E, G_M = 1e-5, 1e-6
N_SUB, FORCE_STEPS, LR = 80, 27, 0.6
K = 4
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_SHA = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"

def readout_coeff(gain):
    cv = NUM_S / gain
    cu = (NUM_0/gain - cv*A_IZH)/(A_IZH*B_IZH)
    return cv,cu
CV_E, CU_E = readout_coeff(G_E)
CV_M, CU_M = readout_coeff(G_M)

@njit(cache=True)
def fast_izh_step(v,u,target_input,pred_input,gain,cv,cu):
    spikes=0
    for _ in range(N_SUB):
        for j in range(v.shape[0]):
            I=I0+gain*target_input[j]-gain*pred_input[j]
            dv1=0.04*v[j]*v[j]+5.0*v[j]+140.0-u[j]+I
            du1=A_IZH*(B_IZH*v[j]-u[j])
            vm=v[j]+0.5*dv1
            um=u[j]+0.5*du1
            dv2=0.04*vm*vm+5.0*vm+140.0-um+I
            du2=A_IZH*(B_IZH*vm-um)
            v[j]+=dv2
            u[j]+=du2
            if v[j]>=30.0:
                v[j]=-65.0
                u[j]+=8.0
                spikes+=1
    out=np.empty(v.shape[0],np.float64)
    for j in range(v.shape[0]):
        out[j]=cv*(v[j]-V0)+cu*(u[j]-U0)
    return out,spikes

@njit(cache=True)
def softmax_numba(z):
    mx=np.max(z)
    q=np.exp(z-mx)
    return q/q.sum()

@njit(cache=True)
def train_chunk(W,ve,ue,vm,um,features,targets,n_paths,seed):
    np.random.seed(seed)
    nctx,nin=features.shape
    ze=np.zeros(nin);zm=np.zeros(K)
    se=0;sm=0
    for _p in range(n_paths):
        W0=W.copy()
        acc=np.zeros_like(W)
        for _k in range(FORCE_STEPS):
            ci=np.random.randint(0,nctx)
            x=features[ci]
            target=int(targets[ci])
            p=softmax_numba(W0@x)
            E,ne=fast_izh_step(ve,ue,x,ze,G_E,CV_E,CU_E)
            one=np.zeros(K)
            one[target]=1.0
            M,nm=fast_izh_step(vm,um,one,p,G_M,CV_M,CU_M)
            acc+=np.outer(M,E)
            se+=ne;sm+=nm
        W+=LR*acc/FORCE_STEPS
    return se,sm

def mnist_ones(path):
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        print("Downloading MNIST from",MNIST_URL,flush=True)
        urlretrieve(MNIST_URL,path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MNIST_SHA:
        raise ValueError("MNIST checksum mismatch: "+digest)
    with np.load(path,allow_pickle=False) as a:
        tr=a["x_train"][a["y_train"]==1].astype(np.float32)/255.0
        te=a["x_test"][a["y_test"]==1].astype(np.float32)/255.0
    return tr,te,{"source":MNIST_URL,"sha256":digest,"train_ones":len(tr),"test_ones":len(te)}

def make_contexts(images,nctx,seed):
    rng=np.random.default_rng(seed)
    # 28x28 -> fixed 7x7 patch means -> 4 occupancy bins (NOT SNN-VQ64).
    pooled=images.reshape(-1,7,4,7,4).mean(axis=(2,4))
    tokens=np.minimum((4.0*pooled).astype(np.int64),3).reshape(-1,49)
    X=np.empty((nctx,103),np.float64);y=np.empty(nctx,np.int64)
    for i in range(nctx):
        src=int(rng.integers(len(tokens)))
        t=int(rng.integers(49))
        mask=rng.choice(49,size=25,replace=False)
        if t not in mask:mask[0]=t
        arr=tokens[src].copy();arr[mask]=-1
        code=np.where(arr==-1,0.,(arr.astype(float)-1.5)/1.5)
        observed=(arr!=-1).astype(float)
        yy,xx=divmod(t,7)
        X[i]=np.r_[code,observed,math.sin(2*math.pi*xx/7),math.cos(2*math.pi*xx/7),
                     math.sin(2*math.pi*yy/7),math.cos(2*math.pi*yy/7),1.]
        y[i]=tokens[src,t]
    return X,y

def metrics(W,X,y):
    z=X@W.T
    z-=z.max(axis=1,keepdims=True)
    p=np.exp(z);p/=p.sum(axis=1,keepdims=True)
    loss=-np.log(np.maximum(p[np.arange(len(y)),y],1e-30))
    hit=(p.argmax(axis=1)==y)
    fg=(y!=0)
    return {"nll":float(loss.mean()),"accuracy":float(hit.mean()),
            "foreground_nll":float(loss[fg].mean()) if fg.any() else None,
            "foreground_accuracy":float(hit[fg].mean()) if fg.any() else None}

def plots(rows,levels,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    N=len(levels);a=np.full((N,N),np.nan);b=a.copy()
    for r in rows:
        a[r["u_index"],r["v_index"]]=r["minimum_validation_nll"]
        if r["paths_to_10pct_improvement"] is not None:
            b[r["u_index"],r["v_index"]]=r["paths_to_10pct_improvement"]
    for arr,title,file,label in [
        (a,"Minimum observed MNIST-1 validation error","min_error_heatmap.png","NLL"),
        (b,"Paths to 10% NLL reduction","training_speed_heatmap.png","Training paths")]:
        fig,ax=plt.subplots(figsize=(9,7))
        im=ax.imshow(arr,origin="lower",aspect="auto")
        ax.set_xticks(range(N),[f"{z:.2f}" for z in levels],rotation=45)
        ax.set_yticks(range(N),[f"{z:.2f}" for z in levels])
        ax.set_xlabel("Normalized v0");ax.set_ylabel("Normalized u0 (fixed across row)")
        ax.set_title(title);fig.colorbar(im,ax=ax,label=label)
        fig.tight_layout();fig.savefig(out/file,dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6))
    for ui,u in enumerate(levels):
        rr=sorted((r for r in rows if r["u_index"]==ui),key=lambda r:r["v_index"])
        ax.plot(levels,[r["minimum_validation_nll"] for r in rr],marker="o",ms=3,label=f"u={u:.2f}")
    ax.set_xlabel("Normalized v0");ax.set_ylabel("Minimum observed validation NLL")
    ax.legend(ncol=2,fontsize=8);ax.set_title("10 v initializations for each fixed u")
    fig.tight_layout();fig.savefig(out/"fixed_u_curves.png",dpi=180);plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mnist",type=Path,default=Path("data/mnist.npz"))
    ap.add_argument("--out",type=Path,default=Path("results/mnist_digit1_vu"))
    ap.add_argument("--levels",type=int,default=10)
    ap.add_argument("--paths",type=int,default=160)
    ap.add_argument("--checkpoint-every",type=int,default=10)
    ap.add_argument("--train-contexts",type=int,default=1024)
    ap.add_argument("--val-contexts",type=int,default=768)
    ap.add_argument("--seed",type=int,default=7311)
    ap.add_argument("--v-span",type=float,default=2e-4)
    ap.add_argument("--u-span",type=float,default=1e-7)
    a=ap.parse_args()
    if a.levels<2 or a.paths<1 or a.checkpoint_every<1:
        ap.error("levels>=2, paths>=1 and checkpoint-every>=1 required")
    a.out.mkdir(parents=True,exist_ok=True)
    xtr,xv,dataset=mnist_ones(a.mnist)
    Xt,yt=make_contexts(xtr,a.train_contexts,a.seed+1)
    Xv,yv=make_contexts(xv,a.val_contexts,a.seed+2)
    baseline=metrics(np.zeros((K,Xt.shape[1])),Xv,yv)
    print("DATA",json.dumps(dataset),"BASELINE",json.dumps(baseline),flush=True)
    # JIT warmup omitted from per-trial wall clock.
    train_chunk(np.zeros((K,Xt.shape[1])),np.full(Xt.shape[1],V0),np.full(Xt.shape[1],U0),
                np.full(K,V0),np.full(K,U0),Xt[:3],yt[:3],1,a.seed)
    levels=np.linspace(0.,1.,a.levels)
    rows=[]
    for ui,un in enumerate(levels):
        for vi,vn in enumerate(levels):
            vp=V0+(vn-0.5)*a.v_span
            up=U0+(un-0.5)*a.u_span
            W=np.zeros((K,Xt.shape[1]),np.float64)
            ve=np.full(Xt.shape[1],vp);ue=np.full(Xt.shape[1],up)
            vm=np.full(K,vp);um=np.full(K,up)
            best=baseline.copy();best_at=0;to_target=None;sec_target=None
            history=[{"paths":0,"elapsed_seconds":0.0,**baseline}]
            se=0;sm=0;elapsed=0.
            start=perf_counter()
            for first in range(0,a.paths,a.checkpoint_every):
                block=min(a.checkpoint_every,a.paths-first)
                nse,nsm=train_chunk(W,ve,ue,vm,um,Xt,yt,block,a.seed+7001+first)
                se+=int(nse);sm+=int(nsm)
                if not np.isfinite(W).all():break
                npath=first+block
                m=metrics(W,Xv,yv)
                elapsed=perf_counter()-start
                history.append({"paths":npath,"elapsed_seconds":elapsed,**m})
                if m["nll"]<best["nll"]:best=m.copy();best_at=npath
                if to_target is None and m["nll"]<=0.9*baseline["nll"]:
                    to_target=npath;sec_target=elapsed
            r={"u_index":ui,"v_index":vi,"u_normalized":float(un),"v_normalized":float(vn),
               "u_physical":float(up),"v_physical":float(vp),
               "initial_validation_nll":baseline["nll"],
               "minimum_validation_nll":best["nll"],"accuracy_at_minimum":best["accuracy"],
               "foreground_nll_at_minimum":best["foreground_nll"],
               "foreground_accuracy_at_minimum":best["foreground_accuracy"],
               "best_path":best_at,"paths_to_10pct_improvement":to_target,
               "seconds_to_10pct_improvement":sec_target,"total_seconds":elapsed,
               "spikes_e":se,"spikes_m":sm,"finite_weights":bool(np.isfinite(W).all()),
               "history":history}
            rows.append(r)
            print("RUN",json.dumps({k:v for k,v in r.items() if k!="history"}),flush=True)
    fields=[k for k in rows[0] if k!="history"]
    with (a.out/"results.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k:r[k] for k in fields} for r in rows])
    report={"experiment":"digit1 readout-level MSDL initial-state sweep",
            "not_full_p3_maskgit":True,"dataset":dataset,
            "baseline_validation":baseline,"canonical_V0":V0,"canonical_U0":U0,
            "normalization":"v=V0+(v_normalized-0.5)*v_span; u=U0+(u_normalized-0.5)*u_span",
            "protocol":"10 v at fixed u, repeated for 10 u; weights W start at zero; same seeded contexts and draws",
            "params":{**vars(a),"mnist":str(a.mnist),"out":str(a.out)},"results":rows}
    (a.out/"results.json").write_text(json.dumps(report,indent=2))
    plots(rows,levels,a.out)
    winner=min(rows,key=lambda x:x["minimum_validation_nll"])
    print("SUMMARY_JSON",json.dumps({"dataset":dataset,"run_count":len(rows),
        "best":{k:v for k,v in winner.items() if k!="history"},
        "target_reached_count":sum(r["paths_to_10pct_improvement"] is not None for r in rows)}),flush=True)

if __name__=="__main__":
    main()
