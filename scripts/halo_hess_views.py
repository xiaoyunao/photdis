#!/usr/bin/env python3
"""Render the actual on/off Hess contrast without repeating identical on panels."""
from matplotlib.colors import SymLogNorm
from photdis.stage1 import require_compute
from halo_common import *


def main():
    require_compute();d=folder('hess_views');inputs={}
    p=OUT/'hess';r=json.loads((p/'complete.json').read_text());inputs['hess']=sha(p/'complete.json')
    with np.load(verified(p/'hess_contributions.npz',r['outputs']['hess_contributions.npz'])) as a:
        arrays={k:a[k] for k in a.files}
    with plt.rc_context(PAPER):
        for si in (1,2,5):
            f,axes=plt.subplots(4,2,figsize=(7.15,9.8),layout='constrained');key=f's{si}_r21_w0.06'
            stats=[]
            for row,stat in enumerate(('on','template','numerator','denominator')):
                if row==0:
                    prefix=key+'_peak_';pair=[arrays[prefix+'on'],arrays[prefix+'alpha']*arrays[prefix+'off']]
                    labels=['On field','Scaled control field']
                else:
                    pair=[arrays[key+'_'+label+'_'+stat] for label in ('peak','reference')]
                    labels=['Peak · '+stat,'Reference · '+stat]
                cap=max(float(np.max(abs(np.array(pair)))),1e-8)
                for col,(label,arr) in enumerate(zip(labels,pair)):
                    ax=axes[row,col];signed=stat=='numerator'
                    z=ax.pcolormesh(arrays['color'],arrays['apparent'],arr.T,cmap='RdBu_r' if signed else 'viridis',
                        norm=SymLogNorm(max(cap/100,1e-10),vmin=-cap if signed else 0,vmax=cap),rasterized=True)
                    ax.set(xlabel=r'$(g-r)_0$',ylabel=r'$r_0$ (mag)',ylim=(23,12),title=label)
                    f.colorbar(z,ax=ax,fraction=.035,extend='both' if signed else 'max',label='Counts / bin' if stat=='on' else 'Contribution / bin')
                    stats.append(dict(segment=si,view=label,total=arr.sum(),maximum_color=cap))
            figure(d,f'foreground_on_off_segment_{si}',f,stats)
    complete(d,inputs,scope='same numerical Hess, alpha, templates and variances; on/off fields shown explicitly; raw arrays in hess/hess_contributions.npz')


if __name__=='__main__':main()
