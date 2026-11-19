#!/usr/bin/env python3
"""Portable v5-style Galactic RGB and coverage plates from compact arrays."""
import argparse
import healpy as hp
import matplotlib.patheffects as pe
from matplotlib.colors import SymLogNorm,Normalize
from stream_common import track
from v3_final_atlas import COLORMAP
from halo_common import BLUE,ORANGE,GREEN
from sky6_common import *


def overlay(ax):
    x=np.linspace(-np.pi,np.pi,721)
    for lat in (-30,-20,0,20,30):
        ax.plot(x,np.full_like(x,np.deg2rad(lat)),color='.25',lw=.65 if lat==0 else .45,ls='-' if lat==0 else '--' if abs(lat)==20 else ':',alpha=.8,label='Galactic plane (b=0°)' if lat==0 else 'b=±20°' if lat==20 else 'b=±30°' if lat==30 else None)
    for label,name,col in [('Sagittarius','Sagittarius.antoja2020','.12'),('Pal 5 track','Pal5.ibata2024',ORANGE)]:
        t,_=track(name);l,b=hp.Rotator(coord=('C','G'))(t['ra'],t['dec'],lonlat=True);l=-((l+180)%360-180)
        jumps=np.flatnonzero((abs(np.diff(l))>120)|(abs(np.diff(b))>5))+1
        for j,part in enumerate(np.split(np.arange(len(l)),jumps)):
            if len(part)>1:ax.plot(np.deg2rad(l[part]),np.deg2rad(b[part]),color=col,lw=.55,ls='--',label=label if j==0 else None,path_effects=[pe.Stroke(linewidth=1.1,foreground='white',alpha=.6),pe.Normal()])
    from v3_13_clusters import SYSTEMS
    from v3_13_cmd import M5_CENTER
    places={'Pal 5':(229.0205,-.1202),'M13':SYSTEMS['M13'][0],'M5':M5_CENTER,'M30':SYSTEMS['M30'][0]}
    for name,(ra,dec) in places.items():
        l,b=hp.Rotator(coord=('C','G'))(ra,dec,lonlat=True);x=-((l+180)%360-180);ax.plot(np.deg2rad(x),np.deg2rad(b),'+',color='.1',ms=3,lw=.6)
        ax.annotate(name,(np.deg2rad(x),np.deg2rad(b)),xytext=(3,2 if name!='M5' else -7),textcoords='offset points',fontsize=5,path_effects=[pe.withStroke(linewidth=1,foreground='white',alpha=.7)])
    ax.set_axisbelow(False);ax.grid(ls=':',lw=.4,color='.5');ax.set_xticks(np.deg2rad([-120,-60,0,60,120]),labels=['120°','60°','0°','300°','240°']);ax.set_yticks(np.deg2rad([-60,-30,-20,0,20,30,60]),labels=['−60°','−30°','−20°','0°','20°','30°','60°']);ax.tick_params(labelsize=6)


def rgb(a,d,name):
    images=[]
    for row in (0,1):
        if 'rgb' in a:
            images.append(a['rgb'][row].copy());continue
        values=a['density'] if row==0 else np.maximum(a['signed_residual'],0);lo=a['stretch'][row,:,0,None,None];hi=a['stretch'][row,:,1,None,None]
        out=np.arcsinh(10*np.clip((values-lo)/np.maximum(hi-lo,1e-8),0,1))/np.arcsinh(10)
        im=np.moveaxis(out[::-1],0,-1);im[~a['mask']]=.88;images.append(im)
    with plt.rc_context(PAPER):
        f,axes=plt.subplots(2,1,figsize=(7.15,7),subplot_kw={'projection':'aitoff'},layout='constrained')
        for ax,im,label in zip(axes,images,(str(a.get('label','CMD-selected sky')),str(a.get('residual_label','Positive residual')))):
            mesh=ax.pcolormesh(a['l'],a['b'],np.zeros(a['mask'].shape),shading='flat',rasterized=True);mesh.set_array(None);mesh.set_facecolor(np.nan_to_num(im.reshape(-1,3),nan=.88));overlay(ax);ax.set_title(label);ax.set_facecolor('.88')
        axes[0].legend(frameon=False,fontsize=5,loc='lower right')
        f.supxlabel('   |   '.join(f'{c}: {lo:.2f}–{hi:.2f} kpc' for c,(lo,hi) in zip('BGR',a['channel_kpc']))+'\nGalactic longitude',fontsize=8);f.supylabel('Galactic latitude',fontsize=8)
        figure(d,name,f,[dict(scope=str(a['scope']),coordinates='Galactic',latitude_lines='0,+/-20,+/-30',correction='2D catalog full-column')])


def physical(a,d,name='physical_overview'):
    """Retain the v5 scalar companion and its physical colorbar units."""
    with plt.rc_context(PAPER):
        f,axes=plt.subplots(2,3,figsize=(7.15,5.1),subplot_kw={'projection':'aitoff'},layout='constrained')
        for row,key in enumerate(('density','signed_residual')):
            values=a[key];finite=values[np.isfinite(values)];cap=max(float(np.quantile(abs(finite),.995)),1e-5)
            for j,ax in enumerate(axes[row]):
                z=ax.pcolormesh(a['l'],a['b'],np.ma.masked_invalid(values[j]),cmap=COLORMAP if row==0 else 'RdBu_r',norm=SymLogNorm(cap/100,vmin=0 if row==0 else -cap,vmax=cap),rasterized=True)
                overlay(ax);lo,hi=a['channel_kpc'][j];ax.set_title(f'{lo:.2f}–{hi:.2f} kpc')
            decade=10.**np.floor(np.log10(cap));ticks=[0,decade] if row==0 else [-decade,0,decade]
            f.colorbar(z,ax=axes[row].tolist(),location='bottom',fraction=.03,pad=.06,label='CMD-selected source density (deg⁻²)' if row==0 else 'Signed residual (deg⁻²)',extend='max' if row==0 else 'both',ticks=ticks,format=plt.FormatStrFormatter('%g'))
        f.supxlabel('Galactic longitude',fontsize=8);f.supylabel('Galactic latitude',fontsize=8);figure(d,name,f,[dict(scope=str(a['scope']),coordinates='Galactic')])


def coverage(a,d):
    with plt.rc_context(PAPER):
        f,axes=plt.subplots(2,2,figsize=(7.15,5.3),subplot_kw={'projection':'aitoff'},layout='constrained')
        for j,(ax,title,unit) in enumerate(zip(axes.flat,('Native PSF sources','Usable gr sources','Shallower gr 5σ depth','Catalog E(B−V)'),('Sources (deg⁻²)','Sources (deg⁻²)','Observed magnitude','E(B−V) (mag)'))):
            v=a['values'][j];finite=v[np.isfinite(v)];cap=max(float(np.quantile(finite,.995)),1e-5)
            norm=Normalize(20,25) if j==2 else SymLogNorm(cap/100,vmin=0,vmax=cap)
            z=ax.pcolormesh(a['l'],a['b'],np.ma.masked_invalid(v),norm=norm,cmap=COLORMAP,rasterized=True);overlay(ax);ax.set_title(title);ticks=[20,22,24] if j==2 else [0,10**np.floor(np.log10(cap))]
            f.colorbar(z,ax=ax,location='bottom',fraction=.04,pad=.08,label=unit,ticks=ticks,format=plt.FormatStrFormatter('%g'),extend='both' if j==2 else 'max')
        f.supxlabel('Galactic longitude',fontsize=8);f.supylabel('Galactic latitude',fontsize=8);figure(d,'sky_coverage',f,[])


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'manifests/sky_sequence_20261119');p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    for stage in ('rgb','old_domain','no_extinction','prior_display','channel_trials','selected_channels','smoothing_trials'):
        d=args.output/stage;d.mkdir(exist_ok=True)
        for path in sorted((args.root/stage).glob('rgb_*.npz')):
            with np.load(path) as z:a={k:z[k] for k in z.files}
            rgb(a,d,path.stem)
            if path.stem=='rgb_overview':physical(a,d)
            if stage=='selected_channels' and path.stem=='rgb_selected_fixed_gain':physical(a,d,'physical_selected')
    if (args.root/'coverage/sky_coverage.npz').exists():
        with np.load(args.root/'coverage/sky_coverage.npz') as z:a={k:z[k] for k in z.files}
        d=args.output/'coverage';d.mkdir(exist_ok=True);coverage(a,d)
    for stage,filename in [('ridge','ridge_redraw.npz'),('width','width_redraw.npz')]:
        if not (args.root/stage/filename).exists():continue
        with np.load(args.root/stage/filename) as z:a={k:z[k] for k in z.files}
        d=args.output/stage;d.mkdir(exist_ok=True)
        if stage=='ridge':
            from sky6_ridge import render
        else:
            from sky6_width import render
        render(a,d)


if __name__=='__main__':main()
