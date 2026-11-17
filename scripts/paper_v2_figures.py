#!/usr/bin/env python3
"""Render the full-reference figures and reference-distance migration for v2."""
import csv
import json
import os

import numpy as np
from matplotlib.colors import LogNorm, Normalize
from scipy.stats import binned_statistic_2d

from photdis.stage1 import require_compute
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_8_direct_figures import BLUE,GREEN,ORANGE,STYLE,plt,save
from v3_10_figures import check_templates
from v3_6_vac_distances import read_columns
from final_science_catalog import REGIONS
from paper_v1_figures import PAPER
from paper_v2_scan import OUT, CLASSES, REFS, GR, RZ, R, SHELLS, bin_axis
from paper_v2_reference import POP_MIGRATION, metric

LABELS=('Gaia geometry','SpecDis','SPDist')
COLORS=(BLUE,ORANGE,GREEN)
POP_NAMES=('Ordinary dwarfs','Metal-poor dwarfs','Intermediate gravity','Giants')
RAD=(0,1,2,5,10,20,np.inf)
RAD_LIST=('<1','1–2','2–5','5–10','10–20','≥20')
MID=(0,2,5,10,np.inf)
MID_LIST=('<2','2–5','5–10','≥10')


def distance_plot(t,dest,heldout):
    fig,axs=plt.subplots(2,3,figsize=(7.15,4.7),layout='constrained')
    x=np.geomspace(.08,32,90)
    y=np.linspace(-4,4,100)
    table,arrays=[],{}
    maps=[]
    for ref in REFS:
        d=t[ref+'_d']
        m=(t[ref+'_stage']==3)&(t['state']==3)
        if heldout: m &= t['heldout']
        a,b=d[m],t['q'][m,2]
        res=5*np.log10(b/a)
        h=np.histogram2d(a,b,(x,x))[0]
        g=np.histogram2d(a,res,(x,y))[0]
        maps.append((a,b,res,h,g))
    for i,(ref,title,(a,b,res,h,g)) in enumerate(zip(REFS,LABELS,maps)):
        for row,data,e in ((0,h,x),(1,g,y)):
            ax=axs[row,i]
            vmax=max(2,max(w[3+row].max() for w in maps))
            cb=ax.pcolormesh(x,e,np.ma.masked_equal(data.T,0),norm=LogNorm(1,vmax),
                            cmap='viridis',rasterized=True)
            ax.set(xscale='log',xlim=x[[0,-1]],ylim=e[[0,-1]])
            ax.set_xticks([.1,1,10],labels=['0.1','1','10'])
            if row==0:
                ax.plot(x[[0,-1]],x[[0,-1]],ls='--',color=ORANGE,lw=.8)
                ax.set(yscale='log',aspect='equal',title=title)
                ax.set_yticks([.1,1,10],labels=['0.1','1','10'] if i==0 else ['', '', ''])
            else:
                ax.axhline(0,color='0.55',lw=.6,ls=':')
                ax.set_xlabel(r'$d_{\rm ref}$ (kpc)')
                if i>0: ax.set_yticklabels([])
                X,Y,L,H=[],[],[],[]
                for j,(lo,hi) in enumerate(zip(RAD[:-1],RAD[1:])):
                    k=(a>=lo)&(a<hi)
                    if k.sum()<5: continue
                    l,m,n=np.quantile(res[k],[.16,.5,.84])
                    xx=np.median(a[k])
                    table.append(dict(reference=ref,kind='interval',d_lo=lo,d_hi=hi,n=int(k.sum()),
                                       inside_plot=int(((a[k]>=x[0])&(a[k]<=x[-1])&
                                                       (res[k]>=y[0])&(res[k]<=y[-1])).sum()),
                                       median=m,lo68=l,hi68=n))
                    X.append(xx); Y.append(m); L.append(l); H.append(n)
                if X:
                    ax.fill_between(X,L,H,color=ORANGE,alpha=.40)
                    ax.plot(X,Y,color=ORANGE,lw=1)
            table.append(dict(reference=ref,kind='density' if row==0 else 'residual',d_lo=x[0],
                  d_hi=x[-1],n=len(a),inside_plot=int(data.sum()),median=None,lo68=None,hi68=None))
            arrays[f'{ref}_{row}']=data
            if i==2: fig.colorbar(cb,ax=axs[row,:].tolist(),pad=.017,fraction=.028,label='Stars/bin')
    axs[0,0].set_ylabel(r'$d_{50}$ (kpc)')
    axs[1,0].set_ylabel(r'$\Delta\mu$ (mag)')
    name='all_reference_distance' if heldout else 'all_reference_distance_all_matches'
    save(fig,dest,name,table,dict(distance_edges=x,residual_edges=y,**arrays))


def trends(t,dest):
    f,df,g,dg,_,_=t['physical'].T
    re=t['observed_r']
    fig,a=plt.subplots(2,2,figsize=(7.1,4.7),layout='constrained')
    axes=((-2.8,-2.2,-1.8,-1.5,-1.2,-.9,-.6,-.3,0.,.3,.6),
          (0,1.5,2.5,3.2,3.7,4.0,4.2,4.5,4.75,5.1),
          (-4,-2,0,1,2,3,4,5,6,7,8,10),
          (15,16,17,18,19,20,21,22,23))
    labels=('[Fe/H] (dex)',r'$\log g$',r'$M_{r,\rm ref}$ (mag)',r'$r$ (mag)')
    rows=[]
    for k,ref in enumerate(('specdis','spdist')):
        d=t[ref+'_d']
        base=(t[ref+'_stage']==3)&(t['state']==3)&t['heldout']
        mag=np.full(len(d),np.nan)
        mag[base]=t['m'][base,1]-10-5*np.log10(d[base])
        res=np.full(len(d),np.nan)
        res[base]=5*np.log10(t['q'][base,2]/d[base])
        for i,(ax,value,edges,lab) in enumerate(zip(a.ravel(),(f,g,mag,re),axes,labels)):
            for side,color in ((0,BLUE),(1,ORANGE)):
                good=base&((t['region']==3)==bool(side))&np.isfinite(value)
                if i<2:
                    e=(df,dg)[i]
                    good&=(e>0)&(e<=.3)
                X=[]; Y=[]; E=[]
                for lo,hi in zip(edges[:-1],edges[1:]):
                    j=good&(value>=lo)&(value<hi)
                    z=res[j]
                    # Asymptotic fixed-rank median interval. It describes within-source
                    # scatter only. Tile-level sensitivity is tested in the F15 design.
                    if len(z):
                        xx=np.median(value[j])
                        delta=.98/np.sqrt(len(z))
                        y,l,h=np.quantile(z,[.5,max(0,.5-delta),min(1,.5+delta)])
                    else: xx=y=l=h=np.nan
                    rows.append(dict(reference=ref,native=('DECam','BASS/MzLS')[side],variable=i,
                       lo=lo,hi=hi,n=len(z),coordinate=xx,mid=y,lo95=l,hi95=h,
                       outside_plot=int((good&((value<edges[0])|(value>=edges[-1]))).sum())))
                    if len(z)>=20:
                        X.append(xx);Y.append(y);E.append([y-l,h-y])
                if X:
                    ax.errorbar(X,Y,yerr=np.array(E).T,ms=2,marker='o',lw=.75,color=color,
                        ls='-' if k==0 else '--',label=f"{('DECam','BASS/MzLS')[side]} {ref.title()}")
                ax.axhline(0,color='.65',lw=.6,ls=':')
                ax.set(xlabel=lab,xlim=(edges[0],edges[-1]))
    for ax in a[:,0]: ax.set_ylabel(r'Median $\Delta\mu$ (mag)')
    a[0,0].legend(fontsize=6,frameon=False,ncols=2)
    save(fig,dest,'distance_bias_trends',rows)


def optical(t,dest):
    x=np.linspace(-.28,1.5,51); y=np.linspace(-2,11,58); z=np.linspace(-.22,1.08,48)
    g,r,rz=t['m'].T
    c=g-r
    d=t['specdis_d']
    ok=(t['specdis_stage']==3)&(t['state']==3)&t['heldout']
    mu=np.full(len(d),np.nan); mu[ok]=10+5*np.log10(d[ok])
    mm=np.full(len(d),np.nan); mm[ok]=5*np.log10(t['q'][ok,2]/d[ok])
    fig,a=plt.subplots(2,2,figsize=(7.1,4.8),layout='constrained',sharex=True)
    table=[]; data={}
    for i,side in enumerate(('DECam','BASS/MzLS')):
        for j,(v,b,label) in enumerate(((r-mu,y,r'$M_{r,\rm ref}$ (mag)'),(r-rz,z,r'$(r-z)_0$'))):
            k=ok&((t['region']==3)==bool(i))&np.isfinite(v)&np.isfinite(c)
            n=np.histogram2d(c[k],v[k],(x,b))[0]
            m=binned_statistic_2d(c[k],v[k],mm[k],statistic='median',bins=(x,b)).statistic
            plot=a[i,j].pcolormesh(x,b,np.ma.masked_where(n<10,m).T,norm=Normalize(-1,1),
                 cmap='coolwarm',rasterized=True)
            a[i,j].set(ylabel=label,title=side if j==0 else '')
            if j==0: a[i,j].invert_yaxis()
            if i==1: a[i,j].set_xlabel(r'$(g-r)_0$')
            data[f'counts_{i}_{j}']=n;data[f'median_{i}_{j}']=m
            table.append(dict(native=side,projection=j,n=int(k.sum()),inside_plot=int(n.sum()),
                              n_shown=int(n[n>=10].sum())))
    fig.colorbar(plot,ax=a.ravel().tolist(),fraction=.035,extend='both',label=r'Median $\Delta\mu$ (mag)')
    save(fig,dest,'native_cmd_residuals',table,dict(gr=x,mr=y,rz=z,**data))


def migration(t,dest):
    mark=ROOT/'data/final_science/working_ids.json'
    w=json.loads(mark.read_text())
    a=read_columns(verified(mark.with_name('working_ids.npz'),w['outputs']['working_ids.npz']),('north','south'))
    picked=np.isin(t['ls_id'],a['north'])|np.isin(t['ls_id'],a['south'])
    gate_domain=(t['observed_r']<20)&np.isin(t['region'],(2,3))&(t['quality']==7)
    if np.any(picked&~gate_domain):
        raise ValueError('Working sample outside its frozen application domain')
    dd=t['q'][:,2]
    supported=t['state']==3
    reference=[]
    matrices={}
    figure,axes=plt.subplots(2,4,figsize=(7.15,4.45),layout='constrained',sharex=True,sharey=True)
    for i,ref in enumerate(('specdis','spdist')):
        d=t[ref+'_d']
        for j,grp in enumerate(POP_MIGRATION):
            base=(t[ref+'_stage']==3)&(t['pop']==grp)
            for frame,extra in (('all',np.ones(len(dd),bool)),('heldout',t['heldout'])):
                use=base&extra
                h=np.histogram2d(d[use&supported],dd[use&supported],(MID,MID))[0].astype(int)
                sel=np.histogram2d(d[use&supported&picked],dd[use&supported&picked],(MID,MID))[0].astype(int)
                source=np.histogram(d[use],MID)[0]
                exposed=np.histogram(d[use&supported&gate_domain],MID)[0]
                assert np.all(h.sum(axis=1)<=source) and np.all(sel.sum(axis=1)<=exposed)
                assert np.all(sel<=h)
                for k in range(4):
                    missing=int(source[k]-h[k].sum()); removed=int(exposed[k]-sel[k].sum())
                    for l in range(4):
                        reference.append(dict(reference=ref,frame=frame,population=CLASSES[grp],
                            ref_k=k,pred_k=l,parent=int(source[k]),unsupported=missing,
                            supported=int(h[k].sum()),working=int(sel[k].sum()),
                            working_applicable=int(exposed[k]),working_unassessed=int(h[k].sum()-exposed[k]),
                            working_rejected=removed,pred_count=h[k,l],working_pred_count=sel[k,l]))
                matrices[f'{ref}_{frame}_{grp}']=h
                if frame!='heldout': continue
                ax=axes[i,j]
                norm=np.divide(h,source[:,None],out=np.full_like(h,np.nan,dtype=float),where=source[:,None]>0)
                im=ax.imshow(norm,origin='lower',cmap='Blues',norm=Normalize(0,1),aspect='equal')
                for k,n in enumerate(source):
                    if n: ax.text(3.35,k,f'{n:,}',ha='right',va='center',fontsize=6,
                                  color='white' if norm[k,3]>.65 else 'black')
                if i==0: ax.set_title(POP_NAMES[j],fontsize=8)
                if j==0: ax.set_ylabel(f"{LABELS[i+1]}\nReference (kpc)")
                if i==1: ax.set_xlabel('Predicted (kpc)',fontsize=8)
                ax.set_xticks(np.arange(4),labels=MID_LIST,fontsize=7)
                ax.set_yticks(np.arange(4),labels=MID_LIST,fontsize=7)
    figure.colorbar(im,ax=axes.ravel().tolist(),fraction=.025,label='Fraction of reference layer')
    save(figure,dest,'population_distance_migration',reference,matrices)
    return sha(mark)


def matched_branches(t,dest):
    gr=t['m'][:,0]-t['m'][:,1]; rz=t['m'][:,1]-t['m'][:,2]
    x=np.column_stack((t['region'],bin_axis(t['observed_r'],R),bin_axis(gr,GR),bin_axis(rz,RZ),
            bin_axis(t['EBV'],(-np.inf,.01,.02,.03,np.inf)),
            bin_axis(t['err'].max(axis=1),(-np.inf,.045,.09,.16,np.inf)))).astype('i1')
    rows=[]
    for ref in ('specdis','spdist'):
        d=t[ref+'_d']
        for lo,hi in ((2,5),(5,10)):
            allowed=np.flatnonzero((t[ref+'_stage']==3)&(t['state']==3)&t['heldout']&
                 (d>=lo)&(d<hi)&np.isin(t['pop'],(1,2)))
            h,id=np.unique(x[allowed],axis=0,return_inverse=True)
            order=np.argsort(id,kind='stable'); ixs=np.r_[0,np.flatnonzero(np.diff(id[order]))+1,len(order)]
            for i,(a,b) in enumerate(zip(ixs[:-1],ixs[1:])):
                if b==a: continue
                j=allowed[order[a:b]]
                p=[j[t['pop'][j]==v] for v in (1,2)]
                for grp,k in zip((1,2),p):
                    rows.append(dict(reference=ref,d_lo=lo,d_hi=hi,pop=CLASSES[grp],region=int(h[i,0]),
                       r_bin=int(h[i,1]),gr_bin=int(h[i,2]),rz_bin=int(h[i,3]),ebv_bin=int(h[i,4]),
                       error_bin=int(h[i,5]),common=min(map(len,p)),**metric(t['q'][k],d[k])))
    csv_rows(dest/'matched_physical_conditions.csv',rows)


def main():
    require_compute()
    src=OUT/'reference/complete.json'
    r=json.loads(src.read_text())
    p=verified(src.with_name('reference_rows.npz'),r['outputs']['reference_rows.npz'])
    fields=('TYPE','ls_id','region','heldout','state','is_primary','physical','observed_r','m','q','pop','err','quality',
            'EBV',*(k+suffix for k in REFS for suffix in ('_stage','_d')))
    t=read_columns(p,fields)
    assert (t['TYPE']=='PSF').all()
    dest=OUT/'figures'
    dest.mkdir(exist_ok=True,parents=True)
    with plt.rc_context(PAPER):
        distance_plot(t,dest,True)
        distance_plot(t,dest,False)
        trends(t,dest)
        optical(t,dest)
        working=migration(t,dest)
    matched_branches(t,dest)
    save_json(dest/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),reference=sha(src),
         science_working=working,style=check_templates(),outputs={p.name:sha(p) for p in dest.iterdir()
                                                                if p.name!='complete.json'}))


if __name__=='__main__':
    main()
