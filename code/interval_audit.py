"""Outward interval audit of global Fourier generator bounds, mpmath.iv."""
from fractions import Fraction as F
import json
from mpmath import iv
iv.dps=50

def rat(s):
    q=F(s)
    return iv.mpf(str(q.numerator))/iv.mpf(str(q.denominator))

def audit(rec):
    r=rat(rec['r']);K=rec['K'];f=[[rat(x) for x in row] for row in rec['coefficients']]
    results=[]
    for i in range(2):
        c=[iv.mpf(0) for _ in range(K+2)];s=[iv.mpf(0) for _ in range(K+2)]
        c[0]=rat('-3/4')
        if i:c[1]=iv.mpf(2)
        for k in range(K+1):
            c[k]+=r*(f[1-i][k]-f[i][k])
            if k:s[k]+=r*(f[1-i][K+k]-f[i][K+k])
        for k in range(1,K+1):
            a,b=f[i][k],f[i][K+k]
            if not i:
                c[k]+=2*k*b;s[k]-=2*k*a
            else:
                c[k-1]+=2*k*a;c[k+1]-=2*k*a;s[k+1]-=2*k*b
                if k>1:s[k-1]+=2*k*b
        radius=sum([abs(q) for q in c[1:]+s[1:]],iv.mpf(0))
        lower=c[0]-radius;upper=c[0]+radius
        results.append(dict(lower_interval=str(lower),upper_interval=str(upper)))
        if rec['r']=='5':assert upper<rat('-9/1000')
        else:assert lower>rat('7/100')
    return dict(r=rec['r'],K=K,state_bounds=results,result='PASS')

out=[audit(rec) for rec in json.load(open('data/poisson_certificates.json'))]
r=rat('1/10')
low=rat('1/4')-r*r*iv.pi*iv.ln(2)/(2*(1-iv.exp(-iv.pi*r)))
assert low>rat('9/50')
out.insert(0,dict(r='1/10',lower_interval=str(low),result='PASS'))
with open('data/interval_audit.json','w') as f:json.dump(out,f,indent=2)
for z in out:print(z['r'],z['result'])
