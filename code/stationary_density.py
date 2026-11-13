"""Invariant-density ODE for the rotation/diagonal pair.
On 0<theta<pi/2 the coordinate is x=log|tan(theta)|; on pi/2<theta<pi it is
x=-log|tan(theta)|, i.e. theta=pi/2+atan(exp(x)), so that cos(2 theta)=-tanh(x) on the
first branch and +tanh(x) on the second.
This resolves the stable-direction power singularity without angular meshing.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import numpy as np
from scipy.integrate import solve_ivp

def density(r,tol=2e-11,tail=36.,edge=42.,max_step=1.):
    # J=1 before normalization, y=p0/J-1.
    total=np.zeros(3)
    stable=max(edge,4*tail/r)
    for branch in [0,1]:
        direction=-1 if branch==0 else 1
        ss=1 if branch==0 else -1 # sign of sin(2theta)
        def rhs(x,z):
            e=np.exp(-abs(x)); dx=e/(1+e*e)
            y=z[0]
            yp=(r/4-r*dx)*y-r*dx if branch==0 else -(r/4+r*dx)*y-r*dx
            cos2=-np.tanh(x) if branch==0 else np.tanh(x)
            return [yp,direction*(1+y)*dx,direction*ss*y/4,direction*2*cos2*ss*y/4]
        # Always resolve the active angular transition |x| <= edge.
        # One huge adaptive span can step over it when r is very small.
        span=(edge,-edge) if branch==0 else (-edge,edge)
        sol=solve_ivp(rhs,span,[0,0,0,0],method='LSODA',rtol=tol,atol=tol*.02,max_step=max_step)
        assert sol.success
        end=sol.y[:,-1]
        if stable>edge:
            span=(-edge,-stable) if branch==0 else (edge,stable)
            sol=solve_ivp(rhs,span,end,method='LSODA',rtol=tol,atol=tol*.02)
            assert sol.success
            end=sol.y[:,-1]
        total+=end[1:]
    norm=total[0]+total[1]
    return dict(exponent=-.75+total[2]/norm,mass0=total[0]/norm,
                mass1=total[1]/norm,J=1/norm,tail=tail,edge=edge,tol=tol)

if __name__=='__main__':
    import json
    for r in [.1,3,5,20]:print(r,json.dumps(density(r)),flush=True)
