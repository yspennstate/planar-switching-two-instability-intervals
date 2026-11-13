import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,csv
import numpy as np
from scipy.optimize import brentq
from stationary_density import density
from numerics import ctmc
mats=np.array([[[-.75,-1],[1,-.75]],[[1.25,0],[0,-2.75]]])
checks=[]
for r in [.1,5,20]:
    base=density(r)
    refined=density(r,tol=2e-13,tail=44.,edge=50.)
    markov=[]
    for n in [256,1024,4096,16384]:
        lam,p,res=ctmc(mats,r,n,True)
        markov.append(dict(n=n,exponent=lam,stationary_residual=res,min_mass=float(p.min())))
    mc=[]
    rows=list(csv.DictReader(open('data/monte_carlo_blocks.csv')))
    for seed in [1,2]:
        z=[q for q in rows if float(q['rate'])==r and int(q['seed'])==seed]
        g=np.array([float(q['log_growth']) for q in z]);t=np.array([float(q['time']) for q in z]);lam=g.sum()/t.sum()
        se=np.std(g-lam*t,ddof=1)/np.sqrt(len(g))/t.mean()
        mc.append(dict(seed=seed,cycles=2000000,time=float(t.sum()),exponent=float(lam),block_standard_error=float(se)))
    checks.append(dict(r=r,density=base,refined_density=refined,ctmc=markov,monte_carlo=mc))
    print(json.dumps(checks[-1]),flush=True)
with open('data/crosschecks.json','w') as f:json.dump(checks,f,indent=2)
rates=np.unique(np.r_[np.logspace(-3,2,401),.1,5,20])
grid=[]
with open('data/candidate_grid.csv','w') as f:
    out=csv.writer(f);out.writerow(['r','lambda','mass0','mass1','J'])
    for r in rates:
        d=density(float(r));out.writerow([r,d['exponent'],d['mass0'],d['mass1'],d['J']]);grid.append([r,d['exponent']])
        # Independent analytic upper bound and low-rate lower bound.
        assert d['exponent'] <= .25+1e-8
        if r<=.1:
            lower=.25-r*r*np.pi*np.log(2)/(2*(-np.expm1(-np.pi*r)))
            assert d['exponent']>=lower-1e-8
        assert abs(d['mass0']-.5)<1e-8
roots=[brentq(lambda r:density(r)['exponent'],1,2),brentq(lambda r:density(r)['exponent'],5,20)]
with open('data/numerical_roots.json','w') as f:json.dump(dict(roots=roots,warning='Numerical crossings only; no claim of exact count of zeros.'),f,indent=2)
print('numerical roots',roots,flush=True)

