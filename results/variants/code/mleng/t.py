import time,lib2
from lib2 import *
g=R.load_nb()
for ds in ['mammography.csv','ecoli2.dat']:
    X,y=g['load_dataset'](ds); X=(X-X.mean(0))/(X.std(0)+1e-8)
    Xtr,Xte,ytr,yte=R.train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
    t=time.time(); Bm,bs=fit_y(Xtr,ytr,ds,0); print(ds,time.time()-t,bs.shape); print(np.round(Bm[-1],2))
    rel,B,order,st=R.disc(g,Xtr,ytr,ds,0)
    a=R.gen(Xtr,ytr,rel,B,order,0); b=G(Xtr,ytr,rel,B,order,0)
    print(a[0].shape,b[0].shape,np.allclose(a[0],b[0]),b[2:])
