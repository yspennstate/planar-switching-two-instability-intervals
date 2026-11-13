"""Independent verifier: Python standard library only, exact Fraction arithmetic.
Rebuilds generator action in a real trig basis (generation used complex Laurent).
"""
from fractions import Fraction as F
import json

def verify(record):
    r=F(record['r']);K=record['K'];d=2*K+1
    f=[[F(x) for x in row] for row in record['coefficients']]
    assert len(f)==2 and all(len(row)==d for row in f)
    bounds=[]
    for i in range(2):
        c=[F(0)]*(K+2);s=[F(0)]*(K+2)
        c[0]=F(-3,4)
        if i==1:c[1]=2
        for k in range(K+1):
            c[k]+=r*(f[1-i][k]-f[i][k])
            if k:
                s[k]+=r*(f[1-i][K+k]-f[i][K+k])
        if i==0:
            for k in range(1,K+1):
                c[k]+=2*k*f[0][K+k]
                s[k]-=2*k*f[0][k]
        else:
            for k in range(1,K+1):
                # -2 sin x * d_theta(a cos kx + b sin kx).
                a,b=f[1][k],f[1][K+k]
                c[k-1]+=2*k*a
                c[k+1]-=2*k*a
                s[k+1]-=2*k*b
                if k>1:s[k-1]+=2*k*b
        radius=sum(map(abs,c[1:]),F(0))+sum(map(abs,s[1:]),F(0))
        bounds.append((c[0]-radius,c[0]+radius))
    lo=min(t[0] for t in bounds);hi=max(t[1] for t in bounds)
    assert lo==F(record['lower']) and hi==F(record['upper'])
    if r==5:
        assert hi<F(-9,1000)
        print('r=5: lambda < -9/1000, EXACT PASS')
    elif r==20:
        assert lo>F(7,100)
        print('r=20: lambda > 7/100, EXACT PASS')
    else:raise ValueError('Unexpected rate')
    return dict(r=str(r),K=K,lower=str(lo),upper=str(hi),decimal=[float(lo),float(hi)])

if __name__=='__main__':
    data=json.load(open('data/poisson_certificates.json'))
    assert {x['r'] for x in data}=={'5','20'}
    result=[verify(x) for x in data]
    # Independent low-rate bound: pi<4, log 2<1 and 1-e^-x >= x/(1+x).
    low=F(1,4)-F(1,20)*(1+F(4,10))
    assert low==F(9,50)>0
    print('r=1/10: lambda > 9/50, analytic inequality EXACT PASS')
    result.insert(0,dict(r='1/10',lower=str(low),method='analytic renewal bound'))
    with open('data/verification.json','w') as out:json.dump(result,out,indent=2)
