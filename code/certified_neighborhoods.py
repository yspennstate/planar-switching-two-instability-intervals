from fractions import Fraction as F
import json
records=json.load(open('data/poisson_certificates.json'));out=[]
for z in records:
    f=[[F(q) for q in row] for row in z['coefficients']]
    D=sum(abs(a-b) for a,b in zip(*f))
    r=F(z['r']);delta=F(3,1000) if r==5 else F(1,20)
    lower=F(z['lower'])-delta*D
    upper=F(z['upper'])+delta*D
    assert upper<0 if r==5 else lower>0
    out.append(dict(r=str(r),delta=str(delta),difference_bound=str(D),rate_interval=[str(r-delta),str(r+delta)],exponent_bounds=[str(lower),str(upper)],decimal_bounds=[float(lower),float(upper)]))
with open('data/certified_neighborhoods.json','w') as f:json.dump(out,f,indent=2)
for z in out:print(z['rate_interval'],z['decimal_bounds'])
