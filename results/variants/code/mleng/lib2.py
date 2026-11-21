import sys, os, pickle, warnings
warnings.filterwarnings('ignore')
S0=__import__("os").environ.get("STCE_HARNESS_DIR", ".")  # folder with the r2-* helper harnesses (not in repo)
sys.path.insert(0,S0+r'\r2-causal')
import numpy as np, pandas as pd
import run as R
HERE=S0+r'\impl\mleng'
R.CACHE=HERE+r'\cache'
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
NB=5
FILES=R.FILES+['extra/wine_quality.csv','extra/wilt.csv','extra/shuttle-2_vs_5.dat']
def fit_y(Xtr,ytr,ds,seed):
    """DirectLiNGAM on [X,y], y sink (prior_knowledge); NB bootstraps. cached."""
    f=os.path.join(HERE,'cache',f'Y_{ds}_{seed}.pkl')
    if os.path.exists(f): return pickle.load(open(f,'rb'))
    import lingam
    d=Xtr.shape[1]; Z=np.column_stack([Xtr,ytr.astype(float)])
    pk=-np.ones((d+1,d+1),int); pk[:d,d]=0   # no path y -> any feature
    def fit(Zs):
        m=lingam.DirectLiNGAM(prior_knowledge=pk); m.fit(Zs); return m.adjacency_matrix_
    Bm=fit(Z); rs=np.random.RandomState(seed); bs=[]
    for _ in range(NB):
        try: bs.append(fit(Z[rs.randint(len(Z),size=len(Z))]))
        except Exception: pass
    r=(Bm,np.array(bs)); pickle.dump(r,open(f,'wb')); return r

def weights(Bm,bs,coef,mode,rng,thr=0.05):
    d=Bm.shape[0]-1
    Bx=Bm[:d,:d]; par=np.abs(Bm[d,:d])>thr; mb=par.astype(float)  # y sink: MB(y)=parents(y)
    stab=(np.abs(bs[:,:d,:d])>thr).mean(0) if len(bs) else np.ones((d,d))
    out=(np.abs(Bx)*stab).sum(0)          # column f = effect of f on others
    wc=1+mb+out
    if mode=='causal': w=wc
    elif mode=='rand': w=rng.permutation(wc)
    else:  # lr control
        a=np.abs(coef); k=int(mb.sum()); ind=np.zeros(d)
        if k: ind[np.argsort(-a)[:k]]=1
        t=a*(out.sum()/(a.sum()+1e-12))
        w=1+ind+t
    return w/w.mean()

def G(X,y,rel,B,order,seed,w=None,mode='cur',skip=0.05,cap=1.5,share=0.7):
    rng=np.random.RandomState(seed)
    Xm=X[y==1]; Xj=X[y==0]; nm,nj=len(Xm),len(Xj); nsyn=nj-nm; k1=min(3,nm-1); k2=3; kall=5; d=X.shape[1]
    Bm=np.where(np.abs(B)>0.05,B,0.0) if rel else np.zeros((d,d))
    sc=np.ones(d) if w is None else np.sqrt(w)
    S=X*sc; Sm=S[y==1]; Sj=S[y==0]
    dm,_=NearestNeighbors(n_neighbors=k1+1).fit(Sm).kneighbors(Sm); Dmin=dm[:,1:k1+1].mean(1)
    dj,_=NearestNeighbors(n_neighbors=k2+1).fit(Sj).kneighbors(Sm); Dmaj=np.median(dj[:,:k2],1)
    ad=np.clip(1-Dmin/(Dmaj+1e-6),0,1)
    _,ind=NearestNeighbors(n_neighbors=kall+1).fit(S).kneighbors(Sm); lab=y[ind[:,1:kall+1]]
    R_=(lab==0).sum(1)/kall; Gm=np.log2(1+(lab==1).sum(1)/kall)
    al=ad*(1-R_)*Gm
    _,nb=NearestNeighbors(n_neighbors=k1+1).fit(Sm).kneighbors(Sm); nb=nb[:,1:]
    ncand=int(nsyn*cap)
    if mode=='cur':
        idx=rng.randint(nm,size=ncand); idx=idx[al[idx]>=skip]; rej=ncand-len(idx)
    elif mode=='A':
        thr=max(np.percentile(al,25),1e-9); el=np.where(al>=thr)[0]
        if len(el)==0: return X,y,0,ncand
        nt=int(min(nsyn,cap*nsyn*len(el)/nm)); idx=rng.choice(el,nt); rej=ncand-int(ncand*len(el)/nm)
    elif mode=='B':
        nt=max(int(share/(1-share)*nj)-nm,0)
        if al.sum()<=0: return X,y,0,0
        idx=rng.choice(nm,nt,p=al/al.sum()); rej=0
    elif mode=='SW':
        i0=rng.randint(nm,size=ncand); nt=min(int((al[i0]>=skip).sum()),nsyn); rej=ncand-nt
        p=al/(al+0.05); 
        if p.sum()<=0: return X,y,0,ncand
        idx=rng.choice(nm,nt,p=p/p.sum())
    if len(idx)==0: return X,y,0,rej
    a=al[idx]; jj=nb[idx,rng.randint(k1,size=len(idx))]; xi=Xm[idx]; xj=Xm[jj]
    if rel:
        xn=np.zeros_like(xi)
        for f in order:
            par=np.where(Bm[f]!=0)[0]
            if len(par)==0:
                ar=np.clip(a*rng.uniform(0,2,len(a)),0,1); xn[:,f]=xi[:,f]+ar*(xj[:,f]-xi[:,f])
            else:
                wf=Bm[f,par]
                xn[:,f]=xn[:,par]@wf+(xi[:,f]-xi[:,par]@wf)+a*((xj[:,f]-xj[:,par]@wf)-(xi[:,f]-xi[:,par]@wf))
    else: xn=xi+a[:,None]*(xj-xi)
    if mode!="B": xn=xn[:nsyn]
    return np.vstack([X,xn]),np.concatenate([y,np.ones(len(xn),int)]),len(xn),rej

VAR={'CDGS_current':dict(mode='cur'),'A':dict(mode='A'),'B':dict(mode='B'),'SW':dict(mode='SW'),
     'E4':dict(w='causal'),'E4_ctrl_lr':dict(w='lr'),'E4_ctrl_rand':dict(w='rand'),
     'X_mleng_E4A':dict(w='causal',mode='A')}
def task(ds,seed,names=None):
    g=R.load_nb(); X,y=g['load_dataset'](ds); X=(X-X.mean(0))/(X.std(0)+1e-8)
    Xtr,Xte,ytr,yte=R.train_test_split(X,y,test_size=0.3,random_state=seed,stratify=y)
    rel,B,order,st=R.disc(g,Xtr,ytr,ds.replace('/','_'),seed)
    Bm,bs=fit_y(Xtr,ytr,ds.replace('/','_'),seed)
    coef=LogisticRegression(solver='liblinear',class_weight='balanced',max_iter=1000,random_state=42).fit(Xtr,ytr).coef_[0]
    out=[]
    for n,kw in VAR.items():
        if names and n not in names: continue
        kw=dict(kw); wm=kw.pop('w',None)
        w=weights(Bm,bs,coef,wm,np.random.RandomState(seed)) if wm else None
        Xr,yr,ng,nr=G(Xtr,ytr,rel,B,order,seed,w=w,**kw)
        clf=LogisticRegression(solver='liblinear',class_weight='balanced',max_iter=1000,random_state=42).fit(Xr,yr)
        p=clf.predict(Xte); pr=clf.predict_proba(Xte)[:,1]
        rc,pc,f1,gm,auc=g['calculate_metrics'](yte,p,pr)
        out.append(dict(variant=n,dataset=ds.split('/')[-1].split('.')[0],seed=seed,Recall=rc,Precision=pc,F1=f1,**{'G-Mean':gm},AUC=auc,AP=R.average_precision_score(yte,pr),MinorityShare=yr.mean(),n_generated=ng,n_rejected=nr))
    return out
if __name__=='__main__':
    from joblib import Parallel, delayed
    seeds=range(int(sys.argv[1]),int(sys.argv[2]))
    res=Parallel(n_jobs=22)(delayed(task)(f,s) for f in FILES for s in seeds)
    pd.DataFrame([r for o in res for r in o]).to_csv(os.path.join(HERE,f'res_{sys.argv[1]}_{sys.argv[2]}.csv'),index=False)
