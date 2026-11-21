import sys, os, pickle, warnings
warnings.filterwarnings('ignore')
SP=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,os.path.join(SP,'r2-causal'))
import numpy as np, pandas as pd
from run import load_nb, disc, FILES
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from joblib import Parallel, delayed
HERE=os.path.dirname(os.path.abspath(__file__)); CACHE=os.path.join(HERE,'cache')

def edge_stab(Xtr,ds,seed):
    """P_boot(edge) from independent DirectLiNGAM bootstraps (20 if small, 8 if n>=1000)."""
    f=os.path.join(CACHE,f"{ds.replace('/','_')}_{seed}_pb.pkl")
    if os.path.exists(f): return pickle.load(open(f,'rb'))
    import lingam
    r=np.random.RandomState(1000+seed); n=len(Xtr); nb=20 if n<1000 else 8; A=[]
    for _ in range(nb):
        try:
            m=lingam.DirectLiNGAM().fit(Xtr[r.randint(n,size=n)]); A.append(np.abs(m.adjacency_matrix_)>0.05)
        except Exception: pass
    P=np.mean(A,0) if A else np.zeros((Xtr.shape[1],)*2)
    pickle.dump(P,open(f,'wb')); return P

def gen(X,y,rel,B,order,seed,mode='cur',P=None,local=False,skip=0.05,cap=1.5,k1m=3,k2=3,kall=5):
    rng=np.random.RandomState(seed)
    Xm=X[y==1]; Xj=X[y==0]; nm,nj=len(Xm),len(Xj); nsyn=nj-nm; k1=min(k1m,nm-1); d=X.shape[1]
    Bm=np.where(np.abs(B)>0.05,B,0.0) if rel else np.zeros((d,d))
    order=list(order)
    kmin=NearestNeighbors(n_neighbors=k1+1).fit(Xm); kmaj=NearestNeighbors(n_neighbors=k2+1).fit(Xj)
    dm,_=kmin.kneighbors(Xm); Dmin=dm[:,1:k1+1].mean(1)
    dj,_=kmaj.kneighbors(Xm); Dmaj=np.median(dj[:,:k2],1)
    ad=np.clip(1-Dmin/(Dmaj+1e-6),0,1)
    _,ind=NearestNeighbors(n_neighbors=kall+1).fit(X).kneighbors(Xm); lab=y[ind[:,1:kall+1]]
    alpha_i=ad*(1-(lab==0).sum(1)/kall)*np.log2(1+(lab==1).sum(1)/kall)
    _,nb=kmin.kneighbors(Xm); nb=nb[:,1:]
    idx=rng.randint(nm,size=int(nsyn*cap)); idx=idx[alpha_i[idx]>=skip]
    if len(idx)==0: return X,y,0,0
    a=alpha_i[idx]; jj=nb[idx,rng.randint(k1,size=len(idx))]; xi=Xm[idx]; xj=Xm[jj]; n=len(idx)
    rooted = rel and Bm.any()
    if mode=='off' or not rooted:
        xn=xi+a[:,None]*(xj-xi)
    elif mode=='cur':
        xn=np.zeros_like(xi)
        for f in order:
            par=np.where(Bm[f]!=0)[0]
            if len(par)==0:
                ar=np.clip(a*rng.uniform(0,2,n),0,1); xn[:,f]=xi[:,f]+ar*(xj[:,f]-xi[:,f])
            else:
                w=Bm[f,par]; ri=xi[:,f]-xi[:,par]@w; rj=xj[:,f]-xj[:,par]@w
                xn[:,f]=xn[:,par]@w+ri+a*(rj-ri)
    else:
        # E1-family: forward propagation with residuals drawn from minority pool
        Bu=Bm.copy(); ordu=order; nonroot_real=Bm.any(1); icpt=False
        if mode.startswith('E1E5') or mode=='E5_ctrl_shuffle':
            Pe=P.copy()
            if mode=='E5_ctrl_shuffle':
                e=Bm!=0; v=Pe[e].copy(); rng.shuffle(v); Pe=np.zeros_like(Pe); Pe[e]=v
            Bu=Bm*Pe
        elif mode=='E1_ctrl_randDAG':
            p=rng.permutation(d); rank=np.empty(d,int); rank[order]=np.arange(d)
            Bu=Bm[np.ix_(p,p)]; ordu=list(np.argsort([rank[p[i]] for i in range(d)]))
        elif mode=='E1_ctrl_B0':
            Bu=np.zeros((d,d))
        elif mode=='E1_ctrl_ols':
            ordu=list(rng.permutation(d)); Bu=np.zeros((d,d))
            for t,f in enumerate(ordu[1:],1):
                par=ordu[:t]; A=np.c_[Xm[:,par],np.ones(nm)]
                Bu[f,par]=np.linalg.lstsq(A,Xm[:,f],rcond=None)[0][:-1]
        Em=Xm-Xm@Bu.T  # minority residual pool (includes mean offset)
        xn=np.zeros_like(xi)
        for f in ordu:
            par=np.where(Bu[f]!=0)[0]
            if mode=='E1_ctrl_B0' and nonroot_real[f]:   # marginal bootstrap, no parents
                xn[:,f]=Xm[rng.randint(nm,size=n),f]
            elif len(par)==0:
                xn[:,f]=xi[:,f]+a*(xj[:,f]-xi[:,f])
            else:
                if local:
                    cand=np.c_[idx,nb[idx]]; pick=cand[np.arange(n),rng.randint(cand.shape[1],size=n)]
                else: pick=rng.randint(nm,size=n)
                xn[:,f]=xn[:,par]@Bu[f,par]+Em[pick,f]
    # eq 17-18 range-based consistency filter (as in notebook)
    lo,hi=Xm.min(0),Xm.max(0); mg=0.15*(hi-lo)
    ok=np.mean((xn>=lo-mg)&(xn<=hi+mg),1)>=0.55
    stop=np.searchsorted(np.cumsum(ok),nsyn)  # position of nsyn-th accepted
    stop=min(stop+1,n); nrej=int((~ok[:stop]).sum())
    xs=xn[:stop][ok[:stop]][:nsyn]
    if len(xs)==0: return X,y,0,nrej
    return np.vstack([X,xs]),np.concatenate([y,np.ones(len(xs),int)]),len(xs),nrej

V={'CDGS_current':dict(mode='cur'),'E1':dict(mode='E1'),'E1_ctrl_randDAG':dict(mode='E1_ctrl_randDAG'),
 'E1_ctrl_B0':dict(mode='E1_ctrl_B0'),'E1_ctrl_ols':dict(mode='E1_ctrl_ols'),'E1E5':dict(mode='E1E5'),
 'E5_ctrl_shuffle':dict(mode='E5_ctrl_shuffle'),'X_causal_E1local':dict(mode='E1',local=True)}
def task(ds,seed,names):
    g=load_nb(); X,y=g['load_dataset'](ds.replace('__','/')); X=(X-X.mean(0))/(X.std(0)+1e-8)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=seed,stratify=y)
    rel,B,order,st=disc(g,Xtr,ytr,ds.replace('/','_'),seed); P=edge_stab(Xtr,ds,seed); out=[]
    for n in names:
        Xr,yr,ng,nr=gen(Xtr,ytr,rel,B,order,seed,P=P,**V[n])
        clf=LogisticRegression(solver='liblinear',class_weight='balanced',max_iter=1000,random_state=42).fit(Xr,yr)
        p=clf.predict(Xte); pr=clf.predict_proba(Xte)[:,1]
        rc,pc,f1,gm,auc=g['calculate_metrics'](yte,p,pr)
        out.append(dict(variant=n,dataset=ds.split('/')[-1].split('.')[0],seed=seed,Recall=rc,Precision=pc,F1=f1,**{'G-Mean':gm},AUC=auc,AP=average_precision_score(yte,pr),MinorityShare=yr.mean(),n_generated=ng,n_rejected=nr))
    return out
if __name__=='__main__':
    if os.environ.get('EXTRA'): FILES[:]=['extra/'+f for f in ['wine_quality.csv','wilt.csv','shuttle-2_vs_5.dat']]
    names=sys.argv[1].split(',') if len(sys.argv)>1 and sys.argv[1]!='all' else list(V)
    seeds=range(int(sys.argv[3]),int(sys.argv[4])) if len(sys.argv)>4 else range(20)
    res=Parallel(n_jobs=22)(delayed(task)(ds,s,names) for ds in FILES for s in seeds)
    pd.DataFrame([r for o in res for r in o]).to_csv(sys.argv[2] if len(sys.argv)>2 else os.path.join(HERE,'res.csv'),index=False)
