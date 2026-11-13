import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import csv,json
import numpy as np
from stationary_density import density
rows=list(csv.DictReader(open('data/candidate_grid.csv')));errors=[]
with open('data/candidate_grid_refined.csv','w') as f:
    w=csv.writer(f);w.writerow(['r','lambda','mass0','mass1','J'])
    for z in rows:
        r=float(z['r']);d=density(r,tol=2e-13,tail=44,edge=50,max_step=.5)
        errors.append(abs(d['exponent']-float(z['lambda'])))
        assert abs(d['mass0']-.5)<1e-9
        assert d['exponent']<=.25+1e-9
        w.writerow([r,d['exponent'],d['mass0'],d['mass1'],d['J']])
result=dict(rows=len(rows),max_absolute_refinement_change=max(errors),at_rate=float(rows[int(np.argmax(errors))]['r']))
assert result['max_absolute_refinement_change']<1e-8
assert result['peak_rss_bytes']<4*1024**3
json.dump(result,open('data/grid_audit.json','w'),indent=2)
print(result)
