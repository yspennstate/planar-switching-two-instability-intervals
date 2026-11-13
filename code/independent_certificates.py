"""Exact Laurent-polynomial check of data/poisson_certificates.json, written separately from verify_certificate.py."""
from fractions import Fraction as F
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ZERO=(F(0),F(0))
def add(a,b):return (a[0]+b[0],a[1]+b[1])
def mul(a,b):return (a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0])
def accum(p,k,z):p[k]=add(p.get(k,ZERO),z)
def load_poly(values,kmax):
    p={0:(F(values[0]),F(0))}
    for k in range(1,kmax+1):
        a,b=F(values[k])/2,F(values[kmax+k])/2
        p[k]=(a,-b);p[-k]=(a,b)
    return p
def audit(rec):
    r=F(rec['r']);K=rec['K']
    f=[load_poly(row,K) for row in rec['coefficients']]
    velocities=({0:(F(1),F(0))},{1:(F(0),F(1)),-1:(F(0),F(-1))})
    bounds=[]
    for i in (0,1):
        p={0:(F(-3,4),F(0))}
        if i==1:p.update({1:(F(1),F(0)),-1:(F(1),F(0))})
        for k in set(f[i])|set(f[1-i]):
            accum(p,k,mul((r,F(0)),f[1-i].get(k,ZERO)))
            accum(p,k,mul((-r,F(0)),f[i].get(k,ZERO)))
        for k,z in f[i].items():
            dz=mul((F(0),F(2*k)),z)
            for j,v in velocities[i].items():accum(p,k+j,mul(v,dz))
        assert p[0][1]==0
        for k in range(1,K+2):
            z=p.get(k,ZERO);other=p.get(-k,ZERO)
            assert other==(z[0],-z[1])
        radius=sum((2*(abs(z[0])+abs(z[1])) for k,z in p.items() if k>0),F(0))
        center=p[0][0]
        bounds.append((center-radius,center+radius))
    lo=min(b[0] for b in bounds);hi=max(b[1] for b in bounds)
    assert lo==F(rec['lower']) and hi==F(rec['upper'])
    if r==5:assert hi<0
    if r==20:assert lo>0
    # Sup norm of difference from the same exact Laurent coefficients.
    diff={}
    for k in set(f[0])|set(f[1]):
        a=f[0].get(k,ZERO);b=f[1].get(k,ZERO)
        diff[k]=(a[0]-b[0],a[1]-b[1])
    D=abs(diff[0][0])+sum(2*(abs(z[0])+abs(z[1])) for k,z in diff.items() if k>0)
    delta=F(3,1000) if r==5 else F(1,20)
    nlo,nhi=lo-delta*D,hi+delta*D
    assert nhi<0 if r==5 else nlo>0
    return dict(r=str(r),K=K,state_bounds=[[str(a),str(b)] for a,b in bounds],
                lower=str(lo),upper=str(hi),D=str(D),
                neighborhood=[str(r-delta),str(r+delta)],
                neighborhood_bound=[str(nlo),str(nhi)])
if __name__=='__main__':
    records=json.loads((ROOT/'data/poisson_certificates.json').read_text())
    output=[audit(rec) for rec in records]
    (ROOT/'data/independent/certificate_audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))
