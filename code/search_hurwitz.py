"""Reuse traceless search curves under exactly computable scalar shifts."""
import json
import numpy as np
p=json.load(open('data/search_ctmc.json'));out=[]
for z in p['records']:
    a=np.array(z['matrices'])
    spectral=[float(np.linalg.eigvals(m).real.max()) for m in a]
    for gamma in [.01,.05,.1]:
        shifts=np.array(spectral)+gamma
        b=a-shifts[:,None,None]*np.eye(2)
        values=np.array(z['values'])-shifts.mean()
        pos=values>0
        count=int(np.sum(pos & ~np.r_[False,pos[:-1]]))
        assert all(np.linalg.eigvals(m).real.max() < -gamma*.999999 for m in b)
        out.append(dict(trial=z['trial'],gamma=gamma,matrices=b.tolist(),eigenvalue_abscissas=[float(np.linalg.eigvals(m).real.max()) for m in b],
                        sampled_positive_components=count,max_sampled_exponent=float(values.max()),
                        shift_identity='lambda_new(r) = lambda_base(r) - mean(shifts)',shifts=shifts.tolist()))
with open('data/hurwitz_search.json','w') as f:
    json.dump(dict(n=p['n'],rates=p['rates'],pairs=len(out),two_component_candidates=sum(z['sampled_positive_components']>=2 for z in out),records=out),f)
print('Hurwitz pairs:',len(out),'sampled two-component candidates:',sum(z['sampled_positive_components']>=2 for z in out))
