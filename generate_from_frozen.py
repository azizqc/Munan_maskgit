#!/usr/bin/env python3
"""Generate MNIST 0/1/2 samples from the exact frozen B0-HQ artifacts."""
from pathlib import Path
import argparse, json, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from maskgit_b0_mnist012 import Model, generate, decode


def save_grid(img, labels, path, cols=15):
    n=len(img); rows=math.ceil(n/cols)
    fig,ax=plt.subplots(rows,cols,figsize=(cols,rows*1.15))
    ax=np.asarray(ax).reshape(-1)
    for i,a in enumerate(ax):
        a.axis('off')
        if i<n:
            a.imshow(img[i],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
            a.set_title(str(int(labels[i])),fontsize=7)
    plt.tight_layout(); fig.savefig(path,dpi=180,bbox_inches='tight'); plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--n-per-class',type=int,default=100)
    ap.add_argument('--steps',type=int,default=10)
    ap.add_argument('--temp',type=float,default=0.25)
    ap.add_argument('--seed',type=int,default=20261005)
    ap.add_argument('--out',type=Path,default=Path('frozen_generation'))
    a=ap.parse_args()
    root=Path(__file__).resolve().parent
    a.out.mkdir(parents=True,exist_ok=True)
    dev=torch.device('cpu')
    ck=torch.load(root/'best.pt',map_location=dev,weights_only=False)
    cfg=ck.get('config',{'d':128,'layers':4,'heads':4,'ff':512})
    m=Model(cfg['d'],cfg['layers'],cfg['heads'],cfg['ff'],.05).to(dev)
    m.load_state_dict(ck['model']); m.eval()
    cent=np.load(root/'codebook.npy')
    labels=np.repeat(np.arange(3,dtype=np.int64),a.n_per_class)
    tok,traj=generate(m,torch.from_numpy(labels),dev,a.steps,a.temp,a.seed,True)
    img=decode(tok,cent)
    np.savez_compressed(a.out/'samples.npz',labels=labels,tokens=tok,images=img,**{f'trajectory_{k}':v for k,v in traj.items()})
    save_grid(img,labels,a.out/'samples.png')
    info={'checkpoint_epoch':int(ck['epoch']),'checkpoint_val_ce':float(ck['best']),
          'config':cfg,'codebook_shape':list(cent.shape),'n':len(labels),
          'steps':a.steps,'temperature':a.temp,'seed':a.seed,
          'unique_token_sequences':int(np.unique(tok,axis=0).shape[0])}
    (a.out/'run_info.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2))

if __name__=='__main__': main()
