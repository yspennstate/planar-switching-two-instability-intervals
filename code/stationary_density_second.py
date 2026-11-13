"""Independent stationary-flux calculation, written separately from stationary_density.py.

Coordinates on the two arcs, each from repeller to attractor:
 theta_left=atan(exp(-t)), theta_right=pi/2+atan(exp(t)).
Both have cos(2 theta)=tanh(t), |dtheta/dt|=w=1/(2cosh(t)).
For q=p0/J, q'=r*(sigma*w*q-(q-1)/4), sigma=+1,-1.
Normalize using mass0=1/2; integrate (cos(2theta)-1)*p1 to
avoid integrating the slow, singular stationary-density tail.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import csv, json, resource, time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]

def flux_density(rate, edge=45., tol=2e-12, step=.25, save=False):
    arcs=[]
    for sigma in (1, -1):
        def rhs(t, state):
            w=np.exp(-abs(t))/(1+np.exp(-2*abs(t)))
            # Use y=q-1 to preserve the small perturbation in both tails.
            y=state[0]
            z=sigma*y/4
            return (rate*(sigma*w*(1+y)-y/4),
                    (1+y)*w, (np.tanh(t)-1)*z, z)
        sol=solve_ivp(rhs, (-edge,edge), (0.,0.,0.,0.),
                      method='DOP853', rtol=tol, atol=tol*.01,
                      max_step=step, dense_output=save)
        assert sol.success
        y,m0,correction,m1=sol.y[:,-1]
        # Outside the stable end w is exponentially small; solve y'=-r*y/4.
        m1+=sigma*y/rate
        arcs.append((m0,correction,m1,sol))
    s0=sum(a[0] for a in arcs)
    flux=1/(2*s0)
    answer=dict(r=float(rate), exponent=.25+sum(a[1] for a in arcs)/s0,
                mass0=.5, mass1=flux*sum(a[2] for a in arcs), J=flux,
                edge=edge,tol=tol,step=step)
    if save:
        rows=[]
        for sigma,arc in zip((1,-1),arcs):
            t=np.linspace(-25,25,2001)
            y=arc[3].sol(t)[0]
            w=np.exp(-abs(t))/(1+np.exp(-2*abs(t)))
            theta=np.arctan(np.exp(-t)) if sigma==1 else np.pi/2+np.arctan(np.exp(t))
            for a,b,c in zip(theta,flux*(1+y),flux*sigma*y/(4*w)):
                rows.append((a,b,c))
        np.savetxt(ROOT/'data'/'independent'/f'density_r{rate:g}.csv',
                   sorted(rows),delimiter=',',header='theta,p0,p1',comments='')
    return answer

def finite_volume(rate,n):
    """Conservative face-flux upwind density; independent discretization."""
    h=np.pi/n
    face=np.arange(n)*h
    rows=[];cols=[];vals=[]
    def moves(src,dst,a):
        rows.extend(dst);cols.extend(src);vals.extend(a)
        rows.extend(src);cols.extend(src);vals.extend(-a)
    cell=np.arange(n)
    for mode in (0,1):
        vel=np.ones(n) if mode==0 else -2*np.sin(2*face)
        left=2*((cell-1)%n)+mode;right=2*cell+mode
        pos=vel>0;neg=vel<0
        moves(left[pos],right[pos],vel[pos]/h)
        moves(right[neg],left[neg],-vel[neg]/h)
        moves(2*cell+mode,2*cell+(1-mode),np.full(n,rate))
    gen=coo_matrix((vals,(rows,cols)),shape=(2*n,2*n)).tocsc()
    # Pin mass in the first cell, solve remaining homogeneous equations, normalize.
    p=np.r_[1.,spsolve(gen[1:,1:],-gen[1:,0].toarray().ravel())]
    p/=p.sum()
    theta=(cell+.5)*h
    reward=np.column_stack((np.full(n,-.75),-.75+2*np.cos(2*theta))).ravel()
    return dict(r=rate,n=n,exponent=float(np.sum(reward*p)),
                mass0=float(p[::2].sum()),mass1=float(p[1::2].sum()),
                min_mass=float(p.min()),residual=float(abs(gen@p).max()))

if __name__=='__main__':
    start=time.time()
    refined=[];fv=[]
    for r in (.1,5.,20.):
        a=flux_density(r,save=True)
        b=flux_density(r,edge=55,tol=3e-14,step=.125)
        a['refined']=b;a['difference']=abs(a['exponent']-b['exponent'])
        refined.append(a)
        print(json.dumps(a),flush=True)
        for n in (1024,4096,16384,65536):
            z=finite_volume(r,n);fv.append(z);print(json.dumps(z),flush=True)
    grid=sorted(set(np.geomspace(.001,1000,141).tolist()+[.1,1.,1.2,1.3,2.,3.,5.,7.,8.,10.,20.]))
    with (ROOT/'data/independent/grid.csv').open('w') as f:
        out=csv.DictWriter(f,fieldnames=['r','exponent','mass0','mass1','J','edge','tol','step'])
        out.writeheader()
        for r in grid:out.writerow(flux_density(r,tol=5e-11,step=.5))
    result=dict(sign_points=refined,finite_volume=fv,grid_count=len(grid),
                seconds=time.time()-start,nice=os.getpriority(os.PRIO_PROCESS,0),
                maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                numpy=np.__version__)
    (ROOT/'data/independent/density.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Completed',result['seconds'],result['maxrss_bytes'],flush=True)
