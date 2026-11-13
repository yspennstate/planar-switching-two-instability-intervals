import os
import json,time,warnings
import numpy as np
from numerics import ctmc,spectral,valley
from scipy.sparse.linalg import MatrixRankWarning
warnings.simplefilter('error', MatrixRankWarning)
rng=np.random.default_rng(180018)
rates=np.logspace(-3,3,49)
records=[];best=0.;start=time.time()
for trial in range(600):
    # General traceless pairs, plus normal forms. Common scalar shifts are chosen later.
    if trial%3==0:
        m0=np.array([[0.,-1.],[1.,0.]])
        m1=rng.integers(-15,16,size=(2,2)).astype(float)*10**rng.uniform(-1,1)
        m1-=np.eye(2)*np.trace(m1)/2
    elif trial%3==1:
        m0=np.array([[1.,0.],[0.,-1.]])
        m1=rng.integers(-15,16,size=(2,2)).astype(float)*10**rng.uniform(-1,1)
        m1-=np.eye(2)*np.trace(m1)/2
    else:
        mats=rng.integers(-10,11,size=(2,2,2)).astype(float)
        for m in mats: m-=np.eye(2)*np.trace(m)/2
        m0,m1=mats
    scale=max(np.linalg.norm(m0),np.linalg.norm(m1))
    mats=np.array([m0,m1])/scale
    try:
        values=np.array([ctmc(mats,r,n=256) for r in rates])
        if not np.all(np.isfinite(values)):continue
    except Exception:continue
    slow=(spectral(mats[0])+spectral(mats[1]))/2
    fast=spectral(mats.mean(axis=0))
    margin,i,j,k=valley(np.r_[slow,values,fast])
    rec=dict(trial=trial,matrices=mats.tolist(),slow=slow,fast=fast,values=values.tolist(),margin=margin,indices=[i,j,k])
    records.append(rec)
    if margin>best:
        best=margin
        print(json.dumps(dict(trial=trial,margin=margin,indices=[i,j,k],elapsed=time.time()-start)),flush=True)
    if trial%25==0:
        with open('data/search_ctmc.json','w') as f:json.dump(dict(seed=180018,n=256,rates=rates.tolist(),records=records),f)
with open('data/search_ctmc.json','w') as f:json.dump(dict(seed=180018,n=256,rates=rates.tolist(),records=records),f)
print('DONE',len(records),time.time()-start,flush=True)
