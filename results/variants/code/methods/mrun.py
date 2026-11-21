import sys, os, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from joblib import Parallel, delayed
SP=__import__("os").environ.get("STCE_HARNESS_DIR", ".")  # folder with the r2-* helper harnesses (not in repo)
sys.path.insert(0,SP+r"\r2-causal")
import run as base   # r2-causal/run.py (load_nb, disc, FILES)
HERE=os.path.dirname(os.path.abspath(__file__))
base.CACHE=os.path.join(HERE,'cache')
NB=20

def pboot(g,Xtr,ds,seed):
    f=os.path.join(HERE,'cache',f"pb_{ds}_{seed}.npy")
    if os.path.exists(f): return np.load(f)
    import lingam, io, contextlib
    rs=np.random.RandomState(1000+seed); n=len(Xtr); acc=0; k=0
    for _ in range(NB):
        try:
            m=lingam.DirectLiNGAM(); m.fit(Xtr[rs.randint(n,size=n)])
            acc=acc+(np.abs(m.adjacency_matrix_)>0.05); k+=1
        except Exception: pass
    P=acc/max(k,1) if k else np.zeros((Xtr.shape[1],)*2)
    np.save(f,P); return P

def gen(X,y,rel,B,order,seed,mode,Pb=None,E5=False,rand_roots=False,resid_a=0.0,skip=0.05,cap=1.5):
    rng=np.random.RandomState(seed)
    Xm=X[y==1]; Xj=X[y==0]; nm,nj=len(Xm),len(Xj); nsyn=nj-nm; k1=min(3,nm-1); d=X.shape[1]
    Bm=np.where(np.abs(B)>0.05,B,0.0) if rel else np.zeros((d,d))
    if E5 and rel:
        Bm=Bm*Pb; Bm=np.where(np.abs(Bm)>0.05,Bm,0.0)   # stability-shrunk, re-thresholded
    # eqs 9-12 exactly as run.py 'cur'
    kmin=NearestNeighbors(n_neighbors=k1+1).fit(Xm); kmaj=NearestNeighbors(n_neighbors=4).fit(Xj); kall=NearestNeighbors(n_neighbors=6).fit(X)
    Dmin=kmin.kneighbors(Xm)[0][:,1:k1+1].mean(1); Dmaj=np.median(kmaj.kneighbors(Xm)[0][:,:3],1)
    ad=np.clip(1-Dmin/(Dmaj+1e-6),0,1)
    lab=y[kall.kneighbors(Xm)[1][:,1:6]]
    alpha_i=ad*(1-(lab==0).sum(1)/5)*np.log2(1+(lab==1).sum(1)/5)
    nb=kmin.kneighbors(Xm)[1][:,1:]
    ncand=int(nsyn*cap); okseed=alpha_i>=skip
    idx=rng.randint(nm,size=ncand); idx=idx[okseed[idx]]
    if len(idx)==0: return X,y,0,0
    a=alpha_i[idx]; jj=nb[idx,rng.randint(k1,size=len(idx))]; xi=Xm[idx]; xj=Xm[jj]
    if mode=='cur':
        xn=np.zeros_like(xi)
        for f in order:
            par=np.where(Bm[f]!=0)[0]
            if len(par)==0:
                ar=np.clip(a*rng.uniform(0,2,len(a)),0,1); xn[:,f]=xi[:,f]+ar*(xj[:,f]-xi[:,f])
            else:
                w=Bm[f,par]; ri=xi[:,f]-xi[:,par]@w; rj=xj[:,f]-xj[:,par]@w
                xn[:,f]=xn[:,par]@w+ri+a*(rj-ri)
    else:  # E2 family: interpolate root set, regenerate rest from seed's residual
        roots=np.array([f for f in order if not Bm[f].any()])
        if rand_roots:
            rr=np.random.RandomState(777+seed); roots=rr.choice(d,size=len(roots),replace=False)
        isr=np.zeros(d,bool); isr[roots]=True
        xn=np.zeros_like(xi)
        for f in order:
            par=np.where(Bm[f]!=0)[0]
            if isr[f]: xn[:,f]=xi[:,f]+a*(xj[:,f]-xi[:,f])
            elif len(par)==0: xn[:,f]=xi[:,f]
            else:
                w=Bm[f,par]; ri=xi[:,f]-xi[:,par]@w; rj=xj[:,f]-xj[:,par]@w
                xn[:,f]=xn[:,par]@w+ri+resid_a*a*(rj-ri)
    # eqs 17-18 consistency filter (range +-15%, threshold 0.55)
    lo,hi=Xm.min(0),Xm.max(0); mg=0.15*(hi-lo)
    ok=np.mean((xn>=lo-mg)&(xn<=hi+mg),1)>=0.55
    nrej=int((~ok).sum()); xn=xn[ok][:nsyn]
    if len(xn)==0: return X,y,0,nrej
    return np.vstack([X,xn]),np.concatenate([y,np.ones(len(xn),int)]),len(xn),nrej

V={'CDGS_current':dict(mode='cur'),'E2':dict(mode='e2'),'E2_ctrl_randroots':dict(mode='e2',rand_roots=True),
   'E2E5':dict(mode='e2',E5=True),
   'X_methods_E2res50':dict(mode='e2',resid_a=0.5)}   # extra: partial residual interpolation (alpha/2 on residuals)

def task(ds,seed):
    g=base.load_nb(); X,y=g['load_dataset'](ds); X=(X-X.mean(0))/(X.std(0)+1e-8)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=seed,stratify=y)
    rel,B,order,st=base.disc(g,Xtr,ytr,ds,seed)
    Pb=pboot(g,Xtr,ds,seed) if rel else None
    out=[]
    for n,kw in V.items():
        Xr,yr,ng,nr=gen(Xtr,ytr,rel,B,order,seed,Pb=Pb,**kw)
        clf=LogisticRegression(solver='liblinear',class_weight='balanced',max_iter=1000,random_state=42).fit(Xr,yr)
        p=clf.predict(Xte); pr=clf.predict_proba(Xte)[:,1]
        rc,pc,f1,gm,auc=g['calculate_metrics'](yte,p,pr)
        out.append(dict(variant=n,dataset=ds.split('.')[0],seed=seed,Recall=rc,Precision=pc,F1=f1,**{'G-Mean':gm},AUC=auc,AP=average_precision_score(yte,pr),MinorityShare=yr.mean(),n_generated=ng,n_rejected=nr))
    return out
if __name__=='__main__':
    res=Parallel(n_jobs=12)(delayed(task)(ds,s) for ds in base.FILES for s in range(20))
    df=pd.DataFrame([r for o in res for r in o]); df.to_csv(SP+r"\impl\results_methods.csv",index=False)
    d=df[(df.seed<10)]
    print(d.groupby(['dataset','variant'])[['F1','Recall','G-Mean','AP']].mean().round(3).to_string())
