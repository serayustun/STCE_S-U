import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'r2-stats'))
import numpy as np, pandas as pd
import harness as h
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from joblib import Parallel, delayed
NAMES=['ecoli2','mammography','page-blocks0','ecoli-0-1-4-6_vs_5','haberman','yeast-2_vs_4','led7digit-0-2-4-5-6-7-8-9_vs_1']

def task(fname, seed):
    X,y=h.load_dataset(fname); X=(X-X.mean(0))/(X.std(0)+1e-8)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.3,random_state=seed,stratify=y)
    d=h.get_disc(fname,seed,Xtr,ytr); ds=fname.split('.')[0]; rows=[]
    def add(v,Xr,yr):
        r=h.score(Xr,yr,Xte,yte)
        rows.append(dict(variant=v,dataset=ds,seed=seed,Recall=r['Recall'],Precision=r['Precision'],F1=r['F1'],**{'G-Mean':r['G-Mean']},
            AUC=r['AUC'],AP=r['AP'],MinorityShare=float(np.mean(yr==1)),n_generated=len(yr)-len(ytr),n_rejected=0))
    add('None',Xtr,ytr)
    for n,S in (('SMOTE',SMOTE(random_state=seed)),('BSM',BorderlineSMOTE(random_state=seed)),('ADASYN',ADASYN(random_state=seed))):
        try: add(n,*S.fit_resample(Xtr,ytr))
        except Exception: pass
    add('CDGS_current',*h.cdgs(Xtr,ytr,d,1.0,.05,True,seed))
    add('DGS_nocausal',*h.cdgs(Xtr,ytr,d,1.0,.05,False,seed))
    cfg,_=h.inner_select(Xtr,ytr,lambda a,b,c:h.smote_s(a,b,c,seed),h.SS,seed)
    add('SMOTE_tuned',*h.smote_s(Xtr,ytr,cfg,seed))
    return rows
if __name__=='__main__':
    t=time.time()
    res=Parallel(n_jobs=20)(delayed(task)(f,s) for f in h.FILES for s in range(20))
    df=pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results_stats.csv'),index=False); print('done',time.time()-t,len(df))
