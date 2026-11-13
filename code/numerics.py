"""Two-state angle PDMP. All rates are actual transitions i -> 1-i."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

def fields(matrices,n):
    theta=(np.arange(n)+.5)*np.pi/n
    co,si=np.cos(theta),np.sin(theta)
    v=[];h=[]
    for a in matrices:
        v.append(a[1,0]*co**2-a[0,1]*si**2+(a[1,1]-a[0,0])*si*co)
        h.append(a[0,0]*co**2+(a[0,1]+a[1,0])*si*co+a[1,1]*si**2)
    return theta,np.array(v),np.array(h)

def ctmc(matrices,r,n=512,details=False):
    theta,v,h=fields(matrices,n)
    # Interleaving keeps the stationary linear system sparse and nearly banded.
    cells=np.arange(n)
    src=np.concatenate([2*cells+i for i in range(2)])
    dest=np.concatenate([2*((cells+np.where(v[i]>=0,1,-1))%n)+i for i in range(2)])
    adv=np.abs(v).ravel()*n/np.pi
    row=np.r_[dest,src,src^1,src]
    col=np.r_[src,src,src,src]
    val=np.r_[adv,-adv,np.full(2*n,r),np.full(2*n,-r)]
    qt=coo_matrix((val,(row,col)),shape=(2*n,2*n)).tocsc()
    p=np.r_[1.,spsolve(qt[1:,1:],-qt[1:,0].toarray().ravel())]
    p/=p.sum()
    lam=float(np.dot(p,h.T.ravel()))
    if details:
        return lam,p,float(np.max(np.abs(qt@p)))
    return lam

def spectral(a):
    return float(np.linalg.eigvals(a).real.max())

def valley(values):
    """Largest horizontal-level violation of quasiconcavity, with indices."""
    left=np.maximum.accumulate(values)
    right=np.maximum.accumulate(values[::-1])[::-1]
    margin=np.minimum(left,right)-values
    j=int(np.argmax(margin)); i=int(np.argmax(values[:j+1])); k=j+int(np.argmax(values[j:]))
    return float(margin[j]),i,j,k
