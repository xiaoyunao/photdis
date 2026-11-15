#!/usr/bin/env python3
"""Plot the complete source-by-class DR11 distance validation atlas."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.metrics import distance_summary
from photdis_figure_style import BLUE, GRAY, GREEN, ORANGE, STYLE
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_distance_evidence import OUT as INPUT
from v3_6_vac_distances import read_columns


OUT = ROOT/'data/v3_6/figures/distance'
SOURCE_LABELS = {'Gaia':'Gaia DR3','SpecDis':'SpecDis v2.1','SPDist':'SPDist v1.0',
                 'DR17_astroNN':'SDSS DR17 astroNN','DR19_Astra':'SDSS DR19 Astra',
                 'DR20_APO':'SDSS DR20 APO','DR20_MINESweeper':'SDSS DR20 LCO'}
CLASSES = ('All classes','Dwarfs','Giants','Other stars','Unknown class')
GATES = (('supported','Before cuts',BLUE),('primary','Primary',ORANGE),('strict','Strict',GREEN))
LOW_N = 100
EDGES = np.linspace(-1,1,41)


def distance_range(q,d):
    values = np.r_[q[q>0],d[d>0]]
    if not len(values):
        return (.2,30.)
    low,high = np.min(values),np.max(values)
    if low==high:
        return (.7*low,1.4*high)
    return (low*.92,high*1.08)


def limits(center,lo,hi,bottom):
    lo = np.where(np.isfinite(lo),np.minimum(center,np.maximum(lo,bottom)),center)
    hi = np.where(np.isfinite(hi),np.maximum(center,hi),center)
    return np.stack([center-lo,hi-center])


def subtitle(summary,count):
    if not count:
        return 'N = 0'
    bias = summary['median_relative']*100
    nmad = summary['nmad_relative']*100
    return f'N = {count:,}   |   Bias {bias:+.1f}%   |   NMAD {nmad:.1f}%'


def plate(a,source,label,class_mask,dest):
    d = a['reference_kpc']
    q = a['qrf_quantiles_kpc']
    finite = np.isfinite(d)&(d>0)&np.isfinite(q).all(axis=1)&(q>0).all(axis=1)
    supported = class_mask&finite
    endpoint = np.c_[a['reference_lo_kpc'][supported],a['reference_hi_kpc'][supported]]
    endpoint = endpoint[np.isfinite(endpoint)&(endpoint>0)]
    domain = distance_range(q[supported],np.r_[d[supported],endpoint])
    bins = np.geomspace(*domain,65)
    residual_all = q[supported,2]/d[supported]-1
    native_counts = np.histogram2d(d[supported],q[supported,2],bins=(bins,bins))[0]
    residual_counts = np.histogram2d(d[supported],residual_all,bins=(bins,EDGES))[0]
    scale = LogNorm(1,max(2,int(max(native_counts.max(),residual_counts.max()))))
    rows = []
    name = source+'_'+label.lower().replace(' ','_')
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(15.8,6.8),layout='constrained')
        grid = fig.add_gridspec(2,6,height_ratios=(3.3,1.1),width_ratios=[5,1.2]*3,hspace=.0,wspace=.0)
        for j,(gate,heading,color) in enumerate(GATES):
            use = supported if gate=='supported' else supported&a[gate]
            top = fig.add_subplot(grid[0,2*j])
            residual = fig.add_subplot(grid[1,2*j],sharex=top)
            hist = fig.add_subplot(grid[1,2*j+1],sharey=residual)
            dist,quant = d[use],q[use]
            frac = quant[:,2]/dist-1
            count = len(dist)
            stat = distance_summary(10+5*np.log10(quant),10+5*np.log10(dist))
            outside = (frac<EDGES[0])|(frac>EDGES[-1])
            rows.append(dict(row_type='summary',selection=gate,n=stat['n'],
                             tail_outside_residual_panel=int(outside.sum()),
                             missing_distance_n=int((class_mask&~finite&(a[gate] if gate!='supported' else True)).sum()),
                             **{k:v for k,v in stat.items() if k!='n'}))
            if 0<count<=LOW_N:
                xerr = limits(dist,a['reference_lo_kpc'][use],a['reference_hi_kpc'][use],domain[0])
                yerr = limits(quant[:,2],quant[:,1],quant[:,3],domain[0])
                top.errorbar(dist,quant[:,2],xerr=xerr,yerr=yerr,fmt='o',ms=4,linewidth=.85,
                             color=color,alpha=.8,capsize=2)
                residual.errorbar(dist,frac,xerr=xerr,yerr=yerr/dist,fmt='o',ms=3,linewidth=.7,
                                  color=color,alpha=.8)
                for index in np.flatnonzero(use):
                    rows.append(dict(row_type='source',selection=gate,legacy_id=a['ls_id'][index],
                        reference_kpc=d[index],reference_lo_kpc=a['reference_lo_kpc'][index],
                        reference_hi_kpc=a['reference_hi_kpc'][index],qrf_median_kpc=q[index,2],
                        qrf_16_kpc=q[index,1],qrf_84_kpc=q[index,3],qrf_training=bool(a['qrf_training'][index])))
            elif count:
                xy,_,_ = np.histogram2d(dist,quant[:,2],bins=(bins,bins))
                xr,_,_ = np.histogram2d(dist,frac,bins=(bins,EDGES))
                for ax,h,x,y in ((top,xy,bins,bins),(residual,xr,bins,EDGES)):
                    ax.pcolormesh(x,y,np.ma.masked_equal(h.T,0),norm=scale,
                                  cmap='viridis',rasterized=True)
                for kind,h,y in (('distance_density',xy,bins),('residual_density',xr,EDGES)):
                    for i,k in zip(*np.nonzero(h)):
                        rows.append(dict(row_type=kind,selection=gate,x_lo=bins[i],x_hi=bins[i+1],
                                         y_lo=y[k],y_hi=y[k+1],n=int(h[i,k])))
            freq,_ = np.histogram(frac,bins=EDGES)
            hist.stairs(freq,EDGES,orientation='horizontal',fill=True,color=color,alpha=.65)
            rows.extend(dict(row_type='central_residual_histogram',selection=gate,y_lo=l,y_hi=h,n=int(v))
                        for l,h,v in zip(EDGES[:-1],EDGES[1:],freq))
            top.plot(domain,domain,color=GRAY,ls='--',lw=1)
            top.set(xscale='log',yscale='log',xlim=domain,ylim=domain,title=heading)
            top.text(.04,.94,subtitle(stat,count),transform=top.transAxes,va='top',fontsize=10)
            top.tick_params(axis='x',labelbottom=False)
            residual.axhline(0,color=GRAY,ls='--',lw=.8)
            residual.set(xlabel='Reference distance [kpc]',ylim=(-1,1))
            residual.set_yticks([-.8,0,.8],['−80%','0','+80%'])
            if outside.any():
                residual.text(.03,.04,f'{outside.sum():,} outside ±100%',fontsize=8,transform=residual.transAxes)
            hist.set(xlabel='Count')
            hist.tick_params(axis='y',length=0,labelleft=False)
            if j==0:
                top.set_ylabel('QRF median distance [kpc]')
                residual.set_ylabel('Relative error')
        if supported.sum()>LOW_N:
            bar = fig.add_subplot(grid[0,-1])
            bar.axis('off')
            ax = bar.inset_axes([.06,.18,.24,.61])
            cb = fig.colorbar(ScalarMappable(norm=scale,cmap='viridis'),cax=ax)
            cb.set_label('Stars per bin',fontsize=11)
        fig.suptitle(f'{SOURCE_LABELS.get(source,source)} · {label} · Distance evidence')
        for ext in ('png','pdf'):
            fig.savefig(dest/f'{name}.{ext}',dpi=300)
        plt.close(fig)
    csv_rows(dest/f'{name}.csv',rows)
    return name


def main():
    require_compute()
    marker = INPUT/'complete.json'
    receipt = json.loads(marker.read_text())
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'complete.json').exists():
        raise FileExistsError(OUT/'complete.json')
    outputs = {}
    keys = ('ls_id','TYPE','qrf_training','primary','strict','consensus_class','reference_kpc',
            'reference_lo_kpc','reference_hi_kpc','qrf_quantiles_kpc')
    for source in receipt['rows']:
        filename = source+'.npz'
        a = read_columns(verified(INPUT/filename,receipt['outputs'][filename]),keys)
        if not (a['TYPE']=='PSF').all():
            raise ValueError('Distance atlas requires verified native PSFs')
        c = a['consensus_class']
        masks = (np.ones(len(c),bool),c==0,c==1,c==2,c<0)
        for label,mask in zip(CLASSES,masks):
            name = plate(a,source,label,mask,OUT)
            outputs.update({f'{name}.{ext}':sha(OUT/f'{name}.{ext}') for ext in ('png','pdf','csv')})
        print(source,len(c),flush=True)
    save_json(OUT/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),dpi=300,
        inputs={str(p.relative_to(ROOT)):sha(p) for p in (marker,Path(__file__))},
        caption='All finite positive reference centers, preserving distances with larger or missing errors. Both cuts use the same matched parent. Low-count panels show every star with its available reference error and QRF 68% interval. Density bins include all other points. Statistics include the tails outside the central residual panel. Source error conventions and training status are retained in the per-source evidence tables.',
        outputs=outputs))


if __name__=='__main__':
    main()
