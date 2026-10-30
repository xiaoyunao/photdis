#!/usr/bin/env python3
"""Small-manifest development visualization; no member table required."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from photdis.cluster_rgb_lower import slope

x=json.loads(Path('manifests/cluster_rgb_lower_model_v2.json').read_text())
fig,ax=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
grid=np.linspace(.5,3,251)
for kind,color in [('quadratic','tab:blue'),('monotone','tab:orange')]:
    r=x['results'][kind+'_grz']
    for j in range(2):
        curves=np.array([slope(m,grid)[:,j] for m in r['models']])
        ax[0,j].plot(grid,curves[0],color=color,label=kind)
        ax[0,j].fill_between(grid,*np.quantile(curves,[.16,.84],axis=0),color=color,alpha=.15)
        for cluster,marker in [('m3','o'),('m53','s')]:
            bins=r['development_residuals'][cluster]
            centers=[np.mean(list(map(float,b.split('_')))) for b in bins]
            ax[1,j].plot(centers,[v['median'][j] for v in bins.values()],marker=marker,
                         color=color,ls='-' if cluster=='m3' else '--',label=kind+' '+cluster)
for j,band in enumerate(['g-r','r-z']):
    ax[0,j].set(xlabel='Mg',ylabel='d('+band+')/dMg')
    ax[1,j].set(xlabel='Mg bin center',ylabel=band+' residual median')
for a in ax.flat:a.axhline(0,color='k',lw=.7);a.grid(alpha=.2);a.legend(fontsize=8)
fig.savefig('figures/cluster_rgb/07_lower_rgb_development_v2.png',dpi=160)
