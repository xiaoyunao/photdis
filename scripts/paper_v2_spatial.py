#!/usr/bin/env python3
"""Render disjoint Aitoff distance layers and archived GD-1 controls."""
import csv
import json
import os

import numpy as np
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import FormatStrFormatter

from photdis.stage1 import require_compute
from v2_1_near import ROOT,csv_rows,save_json,sha
from v3_3_sed import verified
from v3_8_direct_figures import BLUE,ORANGE,plt,save
from v3_10_figures import check_templates
from v3_11_layers import read_receipt
from v3_final_atlas import COLORMAP,MIN_AREA,plotting_grid,read_area
from paper_v1_figures import PAPER
from paper_v2_scan import OUT


def all_sky(dest,inputs):
    p=OUT/'reference/complete.json'
    r=json.loads(p.read_text())
    with np.load(verified(p.with_name('sky.npz'),r['outputs']['sky.npz'])) as q:
        data=q['sky']
    inputs['reference']=sha(p)
    area=read_area(inputs)
    a=np.stack((area['south_high'],area['south_high'],area['south_low'],area['north']))
    use=a[[0,2,3]].sum(axis=0)
    shell=('0–1 kpc','1–10 kpc','≥10 kpc')
    rows=[]
    for i,k in enumerate(('Median distance','Integrated PDF mass')):
        for j,label in enumerate(shell):
            for name,n,d in zip(('south_v35','south_faint','south_low','north'),data[:,i,j],a):
                rows.append(dict(statistic=k,distance=label,region=name,n_or_mass=n.sum(),
                   mapped=n[d>=MIN_AREA].sum(),masked=n[d<MIN_AREA].sum()))
            n=data[:,i,j].sum(axis=0)
            rows.append(dict(statistic=k,distance=label,region='all',n_or_mass=n.sum(),
                          mapped=n[use>=MIN_AREA].sum(),masked=n[use<MIN_AREA].sum()))
    csv_rows(dest/'distance_layers.csv',rows)
    l,b,pix=plotting_grid()
    count=data.sum(axis=0)
    rate=np.divide(count,use[None,None,:],out=np.full_like(count,np.nan),where=use[None,None,:]>=MIN_AREA)
    values=rate[:,0:2][np.isfinite(rate[:,0:2])&(rate[:,0:2]>0)]
    norm=SymLogNorm(linthresh=float(np.quantile(values,.05)),vmin=0,vmax=float(np.quantile(values,.995)),
                    linscale=.32)
    for i,(slug,cblabel) in enumerate((('aitoff_distance','Sources per effective deg²'),
                                      ('aitoff_probability','PDF mass per effective deg²'))):
        f,axs=plt.subplots(1,2,figsize=(7.15,3.2),subplot_kw={'projection':'aitoff'},layout='constrained')
        shown=[]
        for j,ax in enumerate(axs):
            z=ax.pcolormesh(l,b,np.ma.masked_invalid(rate[i,j,pix]),cmap=COLORMAP,norm=norm,
                rasterized=True,shading='flat')
            ax.set_axisbelow(False)
            ax.grid(color='.49',alpha=.48,ls=':',lw=.5)
            ax.set_xticks(np.deg2rad([-120,-60,0,60,120]),labels=['120°','60°','0°','−60°','−120°'])
            ax.set_yticks(np.deg2rad([-60,-30,0,30,60]),labels=['−60°','−30°','0°','30°','60°'])
            ax.annotate('180°',(.055,.43),xycoords='axes fraction',ha='center',fontsize=7)
            ax.annotate('−180°',(.94,.43),xycoords='axes fraction',ha='center',fontsize=7)
            ax.tick_params(labelsize=7)
            ax.set_title(shell[j])
            shown.extend(r for r in rows if r['distance']==shell[j] and r['region']=='all' and
                         r['statistic']==('Median distance','Integrated PDF mass')[i])
        f.supxlabel('Galactic longitude',fontsize=8)
        f.supylabel('Galactic latitude',fontsize=8)
        f.colorbar(z,ax=axs.tolist(),location='bottom',fraction=.032,pad=.10,shrink=.80,
                   aspect=48,extend='max',ticks=[0,1000,2500,5000],
                   format=FormatStrFormatter('%g'),label=cblabel)
        save(f,dest,slug,shown)
    return dict(effective_deg2=float(use.sum()),mapped_deg2=float(use[use>=MIN_AREA].sum()))


def structure(dest,inputs):
    pa=ROOT/'data/v3_17/hess'
    read=read_receipt(pa,'complete.json','hess.npz',('stack_desi','color','absolute'),inputs)
    bm=ROOT/'data/v3_18/members/complete.json'
    j=json.loads(bm.read_text())
    path=verified(bm.with_name('roster.csv'),j['outputs']['roster.csv'])
    inputs['gd1_members']=sha(bm)
    with path.open() as f: rows=[r for r in csv.DictReader(f) if r['science']=='True' and r['TYPE']=='PSF']
    figures,axes=plt.subplots(1,2,figsize=(7.0,3.25),layout='constrained')
    H=read['stack_desi'][0].sum(axis=0)
    x,y=read['color'],read['absolute']
    vmax=max(1.,np.quantile(np.abs(H),.985))
    c=axes[0].pcolormesh(x,y,H.T,cmap='RdBu_r',rasterized=True,
           norm=SymLogNorm(max(.5,vmax*.1),vmin=-vmax,vmax=vmax))
    # The original released Hess bins already fix the reference distance curve.
    axes[0].set(xlim=(-.05,1.15),ylim=(7.8,1.8),xlabel=r'$(g-r)_0$',ylabel=r'$M_r$ (mag)')
    figures.colorbar(c,ax=axes[0],fraction=.045,extend='both',label='Excess PSFs / deg² / bin')
    boxes=[]
    for name,color,label in (('thin',BLUE,'Thin stream'),('cocoon',ORANGE,'Cocoon')):
        jj=[s for s in rows if s['group']==name and np.isfinite(float(s['desi_kpc']))]
        a=np.array([float(s['phi1']) for s in jj])
        b=np.array([5*np.log10(float(s['median_kpc'])/float(s['desi_kpc'])) for s in jj])
        axes[1].scatter(a,b,marker='o' if name=='thin' else 's',s=8,alpha=.56,color=color,
                        edgecolors='none',rasterized=True,label=label)
        boxes.extend(dict(group=name,phi1=l,modulus_residual=m) for l,m in zip(a,b))
    axes[1].axhline(0,color='.6',lw=.6,ls=':')
    axes[1].set(xlabel=r'$\phi_1$ (deg)',ylabel=r'$\Delta\mu$ (mag)',ylim=(-4,4),xlim=(-40,-10))
    axes[1].legend(frameon=False,loc='lower right')
    save(figures,dest,'gd1_hess_member_residuals',boxes,dict(hess=H,color=x,absolute=y))
    csv_rows(dest/'gd1_display_census.csv',[dict(projected_hess_abs_lo=1.8,projected_hess_abs_hi=7.8,
     member_n=len(boxes),members_outside_plot=sum(not (-4<=r['modulus_residual']<=4 and -40<=r['phi1']<=-10)
                                          for r in boxes))])
    path=ROOT/'manifests/v3_15/transfer/comparison.csv'
    with path.open() as f:
        retained=[r for r in csv.DictReader(f) if r['sample']=='strict' and r['template']=='strict']
    csv_rows(dest/'cluster_template_transfer.csv',retained)
    inputs['cluster_transfer']=sha(path)


def main():
    require_compute()
    dest=OUT/'spatial'
    dest.mkdir(exist_ok=True,parents=True)
    inputs={}
    with plt.rc_context(PAPER):
        difference=all_sky(dest,inputs)
        structure(dest,inputs)
    save_json(dest/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        area=difference,style=check_templates(),outputs={p.name:sha(p) for p in dest.iterdir() if p.name!='complete.json'}))


if __name__=='__main__':
    main()
