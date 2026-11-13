import os
os.environ['MPLCONFIGDIR']=os.path.abspath('data/.matplotlib')
os.environ['XDG_CACHE_HOME']=os.path.abspath('data/.cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
p=np.genfromtxt('data/candidate_grid_refined.csv',delimiter=',',names=True)
fig,ax=plt.subplots(figsize=(7.2,4.5),layout='constrained')
ax.set_xscale('log')
ax.plot(p['r'],p['lambda'],color='#244D75',lw=2,label='stationary density')
ax.axhline(0,color='#777777',lw=.8)
ax.errorbar([5],[-.040550765604],yerr=[[.030804868347],[.030804868347]],fmt='o',color='#A33B36',capsize=4,label='bounds of Proposition 3.2')
ax.errorbar([20],[.07765474587],yerr=[[.00734492148],[.00734492148]],fmt='o',color='#A33B36',capsize=4)
ax.scatter([.1],[.18],marker='^',s=55,color='#27805A',zorder=5,label='lower bound of Proposition 4.1')
ax.set(xlabel='rate r',ylabel='λ(r)',xlim=(.001,100),ylim=(-.1,.29))
ax.grid(alpha=.16)
ax.legend(loc='lower left',fontsize=8,frameon=True)
Path('paper/figures').mkdir(parents=True,exist_ok=True)
fig.savefig('paper/figures/curve.pdf')
