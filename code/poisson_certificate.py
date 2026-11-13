"""Construct exact rational global bounds using h_i + L f_i.
Fourier coordinate x=2 theta; each f_i is a trigonometric polynomial.
No floating-point quantity is used in the final bound.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
from fractions import Fraction as F
import json,sys
sys.set_int_max_str_digits(200000)
import numpy as np

def solve(r,K):
    # real basis [1,cos x,...cos Kx,sin x,...sin Kx]
    d=2*K+1
    D=np.zeros((d,d))
    for k in range(1,K+1):
        D[K+k,k]=-2*k
        D[k,K+k]=2*k
    # Multiplication by -2 sin x, projecting only at the last harmonic.
    V=np.zeros((d,d))
    V[K+1,0]=-2
    for k in range(1,K+1):
        if k+1<=K:V[K+k+1,k]-=1
        if k-1>=1:V[K+k-1,k]+=1
        # sin x sin kx = (cos(k-1)x-cos(k+1)x)/2
        V[k-1,K+k]-=1
        if k+1<=K:V[k+1,K+k]+=1
    L=np.block([[D-float(r)*np.eye(d),float(r)*np.eye(d)],
                [float(r)*np.eye(d),np.einsum("ij,jk->ik",V,D,optimize=False)-float(r)*np.eye(d)]])
    # h=[-3/4, -3/4 + 2 cos x]
    h=np.zeros(2*d);h[0]=h[d]=-.75;h[d+1]=2
    unit=np.zeros(2*d);unit[0]=unit[d]=1
    # Remove the common additive constant by fixing f0's constant to zero.
    mat=np.column_stack((L[:,1:],-unit))
    ans=np.linalg.solve(mat,-h)
    ff=np.r_[0,ans[:-1]].reshape(2,d)
    return ff,ans[-1]

def residual_exact(coeff,r):
    # Laurent coefficient arithmetic represented as pairs of rationals.
    def plus(a,b):return (a[0]+b[0],a[1]+b[1])
    def mul(a,b):return(a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0])
    K=(len(coeff[0])-1)//2
    la=[]
    for f in coeff:
        z={0:(f[0],F(0))}
        for k in range(1,K+1):
            z[k]=(f[k]/2,-f[K+k]/2)
            z[-k]=(f[k]/2,f[K+k]/2)
        la.append(z)
    out=[]
    for i in range(2):
        z={k:(F(0),F(0)) for k in range(-K-1,K+2)}
        z[0]=(F(-3,4),F(0))
        if i==1:z[1]=z[-1]=(F(1),F(0))
        for k in range(-K,K+1):
            diff=plus(la[1-i][k],(-la[i][k][0],-la[i][k][1]))
            z[k]=plus(z[k],(r*diff[0],r*diff[1]))
            der=mul((F(0),F(2*k)),la[i][k])
            if i==0:z[k]=plus(z[k],der)
            else:
                # -2 sin x = i exp(ix) - i exp(-ix)
                z[k+1]=plus(z[k+1],mul((F(0),F(1)),der))
                z[k-1]=plus(z[k-1],mul((F(0),F(-1)),der))
        center=z[0][0]
        assert z[0][1]==0
        # For positive k: 2 Re(z_k exp(ikx)); bound by 2(|Re|+|Im|).
        radius=sum((2*(abs(z[k][0])+abs(z[k][1])) for k in range(1,K+2)),F(0))
        out.append(dict(center=center,radius=radius,lower=center-radius,upper=center+radius))
    return out

def generate(r,K,denom=10**12):
    ff,lam=solve(r,K)
    coeff=[[F(round(float(x)*denom),denom) for x in f] for f in ff]
    bounds=residual_exact(coeff,r)
    lo=min(b['lower'] for b in bounds);hi=max(b['upper'] for b in bounds)
    return dict(r=str(r),K=K,coefficients=[[str(x) for x in f] for f in coeff],
                bounds=[{k:str(v) for k,v in b.items()} for b in bounds],
                lower=str(lo),upper=str(hi),decimal=[float(lo),float(hi)],estimate=float(lam))

if __name__=='__main__':
    results=[]
    for r in [F(5),F(20)]:
        for K in [16,32,64,128]:
            out=generate(r,K)
            print(r,K,out['decimal'],out['estimate'],flush=True)
            if out['decimal'][0]>0 or out['decimal'][1]<0:
                results.append(out);break
    with open('data/poisson_certificates.json','w') as f:json.dump(results,f,indent=2)
