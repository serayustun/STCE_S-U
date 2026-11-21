import sys,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np
from mrun import *
EX=['extra/wine_quality.csv','extra/wilt.csv','extra/shuttle-2_vs_5.dat']
def task2(ds,seed):
    g=base.load_nb(); X,y=g['load_dataset'](ds); X=(X-X.mean(0))/(X.std(0)+1e-8)
    key=ds.replace('/','_')
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=seed,stratify=y)
    rel,B,order,st=base.disc(g,Xtr,ytr,key,seed)
    Pb=pboot(g,Xtr,key,seed) if rel else None
    out=[]
    for n,kw in V.items():
        Xr,yr,ng,nr=gen(Xtr,ytr,rel,B,order,seed,Pb=Pb,**kw)
        clf=LogisticRegression(solver='liblinear',class_weight='balanced',max_iter=1000,random_state=42).fit(Xr,yr)
        p=clf.predict(Xte); pr=clf.predict_proba(Xte)[:,1]
        rc,pc,f1,gm,auc=g['calculate_metrics'](yte,p,pr)
        out.append(dict(variant=n,dataset=os.path.splitext(os.path.basename(ds))[0],seed=seed,Recall=rc,Precision=pc,F1=f1,**{'G-Mean':gm},AUC=auc,AP=average_precision_score(yte,pr),MinorityShare=yr.mean(),n_generated=ng,n_rejected=nr))
    return out
if __name__=='__main__':
    res=Parallel(n_jobs=12)(delayed(task2)(ds,s) for ds in EX for s in range(20))
    df=pd.DataFrame([r for o in res for r in o])
    f=SP+r"\impl\results_methods.csv"
    old=pd.read_csv(f); old=old[~old.dataset.isin(df.dataset.unique())]
    pd.concat([old,df]).to_csv(f,index=False)
    print(df.groupby(['dataset','variant'])[['F1','Recall','AP']].mean().round(3).to_string())
    print(df.groupby('dataset').MinorityShare.mean())
