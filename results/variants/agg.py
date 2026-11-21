import glob, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
D=os.path.dirname(os.path.abspath(__file__)); M=['F1','Recall','G-Mean','AP']; REF='CDGS_current'
fs=sorted(glob.glob(os.path.join(D,'results_*.csv')))
df=pd.concat([pd.read_csv(f,keep_default_na=False,na_values=['']).assign(src=os.path.basename(f)) for f in fs],ignore_index=True)
ref_src='results_stats.csv' if 'results_stats.csv' in set(df.src) else None
if ref_src:  # CDGS_current only from stats (reference)
    df=df[(df.variant!=REF)|(df.src==ref_src)]
df=df.drop_duplicates(['variant','dataset','seed'],keep='first')  # shared baselines (None, SMOTE...) keep first file
dsets=list(dict.fromkeys(df.dataset)); vars_=list(dict.fromkeys(df.variant))
out=[f"Files: {', '.join(os.path.basename(f) for f in fs)}\n"]

def md(d,floatfmt='.3f',index=False):
    f=lambda x:format(x,floatfmt) if isinstance(x,(float,np.floating)) else str(x)
    c=list(d.columns); L=['| '+' | '.join(c)+' |','|'+'---|'*len(c)]
    L+=['| '+' | '.join(f(x) for x in r)+' |' for r in d.itertuples(index=False)]
    return chr(10).join(L)
def holm(p):
    p=np.asarray(p); o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for r,i in enumerate(o): run=max(run,(m-r)*p[i]); adj[i]=min(1,run)
    return adj
def paired(a_v,b_v,met):
    """per-dataset (mean diff, raw p) of a minus b over common seeds"""
    res=[]
    for ds in dsets:
        a=df[(df.variant==a_v)&(df.dataset==ds)].set_index('seed')[met]; b=df[(df.variant==b_v)&(df.dataset==ds)].set_index('seed')[met]
        s=a.index.intersection(b.index)
        if len(s)<5: res.append((np.nan,1.0,len(s))); continue
        dd=(a[s]-b[s]).values
        p=1.0 if np.allclose(dd,0) else wilcoxon(dd).pvalue
        res.append((dd.mean(),p,len(s)))
    return res
def sig(a_v,b_v,met):
    r=paired(a_v,b_v,met); adj=holm([x[1] for x in r])
    return [(x[0],q) for x,q in zip(r,adj)]

# (a) per dataset
out.append("## (a) Per-dataset means (dev = seeds 0-9, conf = seeds 10-19)\n")
for ds in dsets:
    t=df[df.dataset==ds]; rows=[]
    for v in vars_:
        r={'variant':v}
        for nm,m in (('dev',t.seed<10),('conf',t.seed>=10)):
            g=t[(t.variant==v)&m]
            for k in M: r[f'{k}_{nm}']=g[k].mean() if len(g) else np.nan
        rows.append(r)
    out.append(f"### {ds}\n"+md(pd.DataFrame(rows))+"\n")

# (b) overall
out.append("## (b) Overall (mean over datasets, seeds 0-19); W/L = datasets significantly better/worse than CDGS_current (paired Wilcoxon, Holm over datasets, alpha .05)\n")
rows=[]
for v in vars_:
    r={'variant':v,'n_ds':df[df.variant==v].dataset.nunique(),'n_seeds':df[df.variant==v].groupby('dataset').seed.nunique().min()}
    pm=df[df.variant==v].groupby('dataset')[M].mean().mean()
    for k in M: r[k]=pm[k]
    for k in ('F1','Recall'):
        if v==REF or REF not in vars_: r[f'{k}_W/L']='-'; continue
        s=sig(v,REF,k); r[f'{k}_W/L']=f"{sum(q<.05 and d>0 for d,q in s)}/{sum(q<.05 and d<0 for d,q in s)}"
    rows.append(r)
out.append(md(pd.DataFrame(rows))+"\n")

# (c) controls
out.append("## (c) Variant vs its controls (X vs X_ctrl_*): mean diff (variant - control) per dataset, * = Holm-adjusted p<.05\n")
for v in vars_:
    ctrls=[c for c in vars_ if c.startswith(v+'_ctrl_')]
    for c in ctrls:
        rows=[]
        for k in ('F1','Recall','AP'):
            s=sig(v,c,k); rows.append({'metric':k,**{ds:(f"{d:+.3f}{'*' if q<.05 else ''}" if d==d else 'n/a') for ds,(d,q) in zip(dsets,s)},
                'W/L':f"{sum(q<.05 and d>0 for d,q in s)}/{sum(q<.05 and d<0 for d,q in s)}"})
        out.append(f"### {v} vs {c}\n"+md(pd.DataFrame(rows))+"\n")
txt='\n'.join(out); print(txt); open(os.path.join(D,'comparison.md'),'w',encoding='utf8').write(txt)
