#!/usr/bin/env python3
"""Redraw final Aitoff plates from their compact physical display grids."""
import argparse
from halo_common import *
from matplotlib.colors import SymLogNorm
from v3_final_atlas import COLORMAP


def render(prefix,dest,annotated=False):
    with prefix.with_suffix('.csv').open() as f:rows=list(csv.DictReader(f))
    with np.load(prefix.with_suffix('.npz')) as a:l,b,values=a['l'],a['b'],a['values']
    assert len(rows)==len(values) and all('panel' in r and 'linthresh' in r for r in rows)
    r=rows[0];signed=float(r['color_lo'])<0
    norm=SymLogNorm(linthresh=float(r['linthresh']),linscale=.32,vmin=float(r['color_lo']),vmax=float(r['color_hi']))
    ncols=2 if len(values)>1 else 1;nrows=(len(values)+ncols-1)//ncols
    with plt.rc_context(PAPER):
        f,axes=plt.subplots(nrows,ncols,figsize=(7.15,2.35*nrows+.35),subplot_kw={'projection':'aitoff'},layout='constrained',squeeze=False)
        for ax,z,row in zip(axes.flat,values,rows):
            image=ax.pcolormesh(l,b,np.ma.masked_invalid(z),cmap='RdBu_r' if signed else COLORMAP,norm=norm,rasterized=True,shading='flat')
            ax.set_axisbelow(False);ax.grid(color='.49',alpha=.48,ls=':',lw=.5)
            ax.set_xticks(np.deg2rad([-120,-60,0,60,120]),labels=['120°','60°','0°','300°','240°'])
            ax.set_yticks(np.deg2rad([-60,-30,0,30,60]));ax.tick_params(labelsize=7);ax.set_title(row['panel'])
            if annotated:
                import healpy as hp
                from v3_13_clusters import SYSTEMS
                for key,(coord,_) in SYSTEMS.items():
                    ll,bb=hp.Rotator(coord=('C','G'))(*coord,lonlat=True);xx=-((ll+180)%360-180)
                    ax.plot(np.deg2rad(xx),np.deg2rad(bb),'+',ms=3,color='black',lw=.5)
                    ax.annotate(key,(np.deg2rad(xx),np.deg2rad(bb)),xytext=(3,2),textcoords='offset points',fontsize=5)
                from v3_16_gd1 import ROTATION,unit,angles,track,EDGES
                phi1=np.linspace(EDGES[0],EDGES[-1],201)
                ra,dec=angles(ROTATION.T@unit(phi1,track(phi1)))
                ll,bb=hp.Rotator(coord=('C','G'))(ra,dec,lonlat=True);xx=-((ll+180)%360-180)
                ax.plot(np.deg2rad(xx),np.deg2rad(bb),'--',color='.1',lw=.6)
                ax.annotate('GD-1',(np.deg2rad(xx[len(xx)//2]),np.deg2rad(bb[len(bb)//2])),xytext=(3,2),textcoords='offset points',fontsize=5)
        for ax in list(axes.flat)[len(values):]:ax.set_visible(False)
        f.supxlabel('Galactic longitude',fontsize=8);f.supylabel('Galactic latitude',fontsize=8)
        f.colorbar(image,ax=list(axes.flat),location='bottom',fraction=.025,pad=.07,shrink=.8,aspect=48,
                   extend='both' if signed else 'max',label=r['units'])
        dest.mkdir(parents=True,exist_ok=True)
        name=prefix.name+('_known_structures' if annotated else '_redraw')
        figure(dest,name,f,rows)
    return dest/(name+'.png')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('prefix',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--annotated',action='store_true')
    args=p.parse_args();render(args.prefix,args.output,args.annotated)
