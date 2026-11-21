import sys, os, pickle, io, contextlib, warnings
warnings.filterwarnings('ignore')
SP=__import__("os").environ.get("STCE_HARNESS_DIR", ".")  # folder with the r2-* helper harnesses (not in repo)
sys.path.insert(0,SP+r"\r2-causal")
import numpy as np, pandas as pd
import run as R   # r2-causal run.py (load_nb, FILES); its main is guarded
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from joblib import Parallel, delayed
HERE=SP+r"\impl\scope"; CACHE=HERE+r"\cache"; os.makedirs(CACHE,exist_ok=True)
FILES=R.FILES

def disc_x(g,Xtr,ytr,ds,seed):
    f=os.path.join(R.CACHE,f"{ds}_{seed}.pkl")
    try: return pickle.load(open(f,'rb'))
    except Exception: pass
    f=os.path.join(CACHE,f"x_{ds}_{seed}.pkl")
    if os.path.exists(f): return pickle.load(open(f,'rb'))
    np.random.seed(seed)
    d=g['FastCausalDiscovery']()
    with contextlib.redirect_stdout(io.StringIO()): d.fit(Xtr,ytr)
    r=(d.is_reliable,d.causal_matrix,d.causal_order,d.stability_score)
    pickle.dump(r,open(f,'wb')); return r

def disc_y(Xtr,ytr,ds,seed):
    """DirectLiNGAM on [X,y], y a sink (prior knowledge). Same bootstrap-stability rule (3 boots)."""
    f=os.path.join(CACHE,f"y_{ds}_{seed}.pkl")
    if os.path.exists(f): return pickle.load(open(f,'rb'))
    import lingam
    n,d=Xtr.shape; data=np.c_[Xtr,ytr.astype(float)]
    pk=-np.ones((d+1,d+1),int); pk[:d,d]=0      # pk[i,j]=0: no path j->i ; y (col d) causes nothing
    thr=0.70 if n<200 else (0.60 if n<500 else 0.55)
    np.random.seed(seed)
    out=(False,np.zeros((d,d)),np.arange(d),np.zeros(d,bool),0.0)
    try:
        m=lingam.DirectLiNGAM(prior_knowledge=pk).fit(data); A=m.adjacency_matrix_
        if not (np.abs(A[:d,:d])>0.05).any(): raise Exception('no edges')
        bs=[]
        for _ in range(3):
            try:
                idx=np.random.choice(n,size=n,replace=True)
                bs.append(lingam.DirectLiNGAM(prior_knowledge=pk).fit(data[idx]).adjacency_matrix_)
            except Exception: pass
        if len(bs)<2: raise Exception('boot')
        fr=np.mean([np.abs(a)>0.05 for a in bs],0); em=np.abs(A[:d,:d])>0.05
        st=np.mean(fr[:d,:d][em]>0.5)
        o=np.array(m.causal_order_).flatten(); o=o[o<d]
        mb=(np.abs(A[d,:d])>0.05)&(fr[d,:d]>0.5)   # y is a sink: MB(y)=parents(y) (no children => no co-parents)
        out=(st>=thr,A[:d,:d],o,mb,st)
    except Exception: pass
    pickle.dump(out,open(f,'wb')); return out

def gen(X,y,rel,B,order,seed,anchor=(),mode='cur',mult=0.5,cap=1.5,k1m=3,k2=3,kall=5,skip=0.05):
    rng=np.random.RandomState(seed)
    Xm=X[y==1]; Xj=X[y==0]; nm,nj=len(Xm),len(Xj); nsyn=nj-nm; k1=min(k1m,nm-1); d=X.shape[1]
    Bm=np.where(np.abs(B)>0.05,B,0.0) if rel else np.zeros((d,d))
    dm,_=NearestNeighbors(n_neighbors=k1+1).fit(Xm).kneighbors(Xm); Dmin=dm[:,1:k1+1].mean(1)
    dj,_=NearestNeighbors(n_neighbors=k2+1).fit(Xj).kneighbors(Xm); Dmaj=np.median(dj[:,:k2],1)
    ad=np.clip(1-Dmin/(Dmaj+1e-6),0,1)
    _,ind=NearestNeighbors(n_neighbors=kall+1).fit(X).kneighbors(Xm); lab=y[ind[:,1:kall+1]]
    alpha=ad*(1-(lab==0).sum(1)/kall)*np.log2(1+(lab==1).sum(1)/kall)
    _,nb=NearestNeighbors(n_neighbors=k1+1).fit(Xm).kneighbors(Xm); nb=nb[:,1:]
    idx=rng.randint(nm,size=int(nsyn*cap)); idx=idx[alpha[idx]>=skip]
    if len(idx)==0: return X,y,0,0
    a=alpha[idx]; jj=nb[idx,rng.randint(k1,size=len(idx))]; xi=Xm[idx]; xj=Xm[jj]
    anchor=set(anchor)
    if rel:
        xn=np.zeros_like(xi)
        for f in order:
            par=np.where(Bm[f]!=0)[0]
            if f in anchor:
                ar=np.clip(mult*a*rng.uniform(0,2,len(a)),0,1); xn[:,f]=xi[:,f]+ar*(xj[:,f]-xi[:,f])
            elif len(par)==0:
                ar=np.clip(a*rng.uniform(0,2,len(a)),0,1); xn[:,f]=xi[:,f]+ar*(xj[:,f]-xi[:,f])
            else:
                w=Bm[f,par]
                if mode=='resamp':   # E1: e ~ empirical minority residual pool (eq 14, all minority train pts)
                    pool=Xm[:,f]-Xm[:,par]@w
                    xn[:,f]=xn[:,par]@w+pool[rng.randint(len(pool),size=len(a))]
                else:
                    ri=xi[:,f]-xi[:,par]@w; rj=xj[:,f]-xj[:,par]@w
                    xn[:,f]=xn[:,par]@w+ri+a*(rj-ri)
    else: xn=xi+a[:,None]*(xj-xi)
    lo,hi=Xm.min(0),Xm.max(0); mg=0.15*(hi-lo)
    ok=np.mean((xn>=lo-mg)&(xn<=hi+mg),1)>=0.55      # eqs 17-18 range consistency check (as notebook)
    nrej=int((~ok).sum()); xn=xn[ok][:nsyn]
    if len(xn)==0: return X,y,0,nrej
    return np.vstack([X,xn]),np.concatenate([y,np.ones(len(xn),int)]),len(xn),nrej

def task(ds,seed):
    g=R.load_nb(); X,y=g['load_dataset'](ds); ds=os.path.basename(ds); X=(X-X.mean(0))/(X.std(0)+1e-8)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=seed,stratify=y)
    d=X.shape[1]; clf=lambda:LogisticRegression(solver='liblinear',class_weight='balanced',max_iter=1000,random_state=42)
    rel,B,order,_=disc_x(g,Xtr,ytr,ds,seed)
    relY,BY,oY,mb,stY=disc_y(Xtr,ytr,ds,seed); mbs=list(np.where(mb)[0]); k=len(mbs)
    rng=np.random.RandomState(1000+seed)
    rnd=list(rng.choice(d,k,replace=False)) if k else []
    coef=np.abs(clf().fit(Xtr,ytr).coef_[0]); lr=list(np.argsort(-coef)[:k])
    V={'CDGS_current':(rel,B,order,(),'cur'),
       'E3':(relY,BY,oY,mbs,'resamp'),'E3_ctrl_rand':(relY,BY,oY,rnd,'resamp'),'E3_ctrl_lr':(relY,BY,oY,lr,'resamp'),
       'X_scope_E3_mbonly':(relY,BY,oY,mbs,'cur')}
    rows=[]
    for n,(r,Bv,o,anc,mode) in V.items():
        Xr,yr,ng,nr=gen(Xtr,ytr,r,Bv,o,seed,anchor=anc,mode=mode)
        c=clf().fit(Xr,yr); p=c.predict(Xte); pr=c.predict_proba(Xte)[:,1]
        rc,pc,f1,gm,auc=g['calculate_metrics'](yte,p,pr)
        rows.append(dict(variant=n,dataset=ds.split('.')[0],seed=seed,Recall=rc,Precision=pc,F1=f1,**{'G-Mean':gm},AUC=auc,AP=average_precision_score(yte,pr),MinorityShare=yr.mean(),n_generated=ng,n_rejected=nr,mb_size=k,relY=relY,relX=rel))
    return rows
if __name__=='__main__':
    seeds=range(int(sys.argv[1]),int(sys.argv[2]))
    res=Parallel(n_jobs=22)(delayed(task)(ds,s) for ds in (sys.argv[3].split(',') if len(sys.argv)>3 else FILES) for s in seeds)
    pd.DataFrame([r for o in res for r in o]).to_csv(HERE+rf"\part_{sys.argv[1]}_{sys.argv[2]}{'_x' if len(sys.argv)>3 else ''}.csv",index=False)
