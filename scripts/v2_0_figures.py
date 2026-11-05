#!/usr/bin/env python3
"""Produce frozen-report figures with one CSV and metadata record per figure."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle
import numpy as np
from scipy.stats import binned_statistic_2d

from v2_0_common import *
from photdis.stage1 import require_compute
from photdis_figure_style import STYLE, BLUE, GREEN, ORANGE, GRAY
from photdis.diagnostics_v1_4 import fixed_examples, cell_area
from photdis.audit import rvs_metallicity
from photdis.baseline_v1 import integrate

REGISTRY={}
ROWS=[]
PANELS=[]
CLASSES=[(-1,'Unknown',GRAY),(0,'Dwarf',BLUE),(1,'Evolved',ORANGE),(2,'Other / transition',GREEN)]
PRIMARY=None


def begin():
    ROWS.clear()
    PANELS.clear()


def save(fig,name,caption,section='main',model='expanded_A',reference='chosen',split='development_test'):
    folder=FIG/section
    folder.mkdir(parents=True,exist_ok=True)
    meta_folder=TABLES/'figures'
    meta_folder.mkdir(exist_ok=True)
    fig.savefig(folder/(name+'.png'),dpi=300,bbox_inches='tight')
    fig.savefig(folder/(name+'.pdf'),dpi=300,bbox_inches='tight')
    plt.close(fig)
    csv_rows(meta_folder/(name+'.csv'),ROWS)
    record=dict(name=name,section=section,model=model,reference=reference,split=split,
                caption=caption,units='distance kpc; distance residual percent; modulus mag',
                binning='edges and counts in CSV',minimum_performance_bin=20,
                sparse='gray crosses indicate 1–19 objects; cream background indicates no evidence',
                panels=list(PANELS),
                png=f'figures/v2_0/{section}/{name}.png',pdf=f'figures/v2_0/{section}/{name}.pdf',
                csv=f'manifests/v2_0/figures/{name}.csv')
    for key in ('png','pdf'):
        record[key+'_sha256']=sha(folder/(name+'.'+key))
    save_json(meta_folder/(name+'.json'),record)
    REGISTRY[name]=record
    print('figure',name,flush=True)


def panel(name,eligible,finite,plotted=None,**kw):
    PANELS.append(dict(panel=name,eligible=int(eligible),finite=int(finite),
                       missing=int(eligible-finite),plotted=int(finite if plotted is None else plotted),
                       outside=int(0 if plotted is None else finite-plotted),**kw))


def limits(v,pad=.03):
    v=np.asarray(v);v=v[np.isfinite(v)]
    if not len(v):
        return (0,1)
    lo,hi=v.min(),v.max()
    margin=max(hi-lo,.1)*pad
    return lo-margin,hi+margin


def density(ax,x,y,xe,ye,label,weight=None):
    ok=np.isfinite(x)&np.isfinite(y)
    h=np.histogram2d(x[ok],y[ok],bins=[xe,ye],weights=None if weight is None else weight[ok])[0]
    count=np.histogram2d(x[ok],y[ok],bins=[xe,ye])[0]
    cm=plt.get_cmap('Blues').copy()
    cm.set_bad('white')
    im=ax.pcolormesh(xe,ye,np.ma.masked_equal(h.T,0),cmap=cm,norm=LogNorm(1,max(2,h.max())))
    panel(label,len(x),ok.sum(),int(count.sum()),x_range=[xe[0],xe[-1]],y_range=[ye[0],ye[-1]])
    for i,j in zip(*np.where(count>0)):
        ROWS.append(dict(panel=label,xlo=xe[i],xhi=xe[i+1],ylo=ye[j],yhi=ye[j+1],n=int(count[i,j]),weight=h[i,j]))
    return im


def map_error(ax,x,y,error,xe,ye,label,absolute=False):
    ok=np.isfinite(x)&np.isfinite(y)&np.isfinite(error)
    v=abs(error[ok]) if absolute else error[ok]
    count=np.histogram2d(x[ok],y[ok],bins=[xe,ye])[0]
    z=binned_statistic_2d(x[ok],y[ok],v,statistic='median',bins=[xe,ye])[0]
    ax.set_facecolor('#f4ebd9')
    cm=plt.get_cmap('Oranges' if absolute else 'coolwarm').copy()
    cm.set_bad((0,0,0,0))
    im=ax.pcolormesh(xe,ye,np.ma.masked_where(count.T<20,z.T),cmap=cm,
                     norm=Normalize(0,100) if absolute else Normalize(-100,100))
    low=(count>0)&(count<20)
    i,j=np.where(low)
    ax.scatter((xe[i]+xe[i+1])/2,(ye[j]+ye[j+1])/2,s=6,c=GRAY,marker='x',linewidths=.4)
    panel(label,len(x),ok.sum(),int(count.sum()),low_occupancy_cells=int(low.sum()),
          display_color_limits=[0,100] if absolute else [-100,100],
          residual_beyond_color=int((abs(v)>100).sum()))
    for i,j in zip(*np.where(count>0)):
        ROWS.append(dict(panel=label,xlo=xe[i],xhi=xe[i+1],ylo=ye[j],yhi=ye[j+1],n=int(count[i,j]),
                         value=z[i,j],low_occupancy=bool(count[i,j]<20)))
    return im


def trend(ax,t,x,edges,key,label,color):
    ref=t['ref_chosen']
    for lo,hi in zip(edges[:-1],edges[1:]):
        use=np.isfinite(x)&(x>=lo)&(x<hi)&np.isfinite(ref)
        stats=percent_metrics(t['mu'][use],ref[use])
        ROWS.append(dict(panel=label,xlo=lo,xhi=hi,**stats))
        if stats['n']<20:
            continue
        center=(lo+hi)/2
        ax.plot(center,stats[key],'o',color=color,ms=5)
    ax.plot([],[],label=label,color=color,marker='o')
    panel(label,len(x),np.isfinite(x).sum(),int(((x>=edges[0])&(x<edges[-1])&np.isfinite(x)).sum()))


def scope():
    begin()
    app=json.loads((TABLES/'snapshot.json').read_text())['application_fields']
    pop=load(DEST/'population.npz')
    fig,axes=plt.subplots(1,2,figsize=(14,5.7),layout='constrained')
    for row in app:
        color=GRAY if row['field']==13 else BLUE
        axes[0].add_patch(Rectangle((row['ra_min'],row['dec_min']),row['ra_max']-row['ra_min'],
                                   row['dec_max']-row['dec_min'],fc=color,alpha=.35,ec=color))
        axes[0].text((row['ra_min']+row['ra_max'])/2,(row['dec_min']+row['dec_max'])/2,str(row['field']),ha='center',fontsize=10)
        ROWS.append(dict(panel='application',**row))
    axes[0].set(xlim=(360,0),ylim=(-60,80),xlabel='RA [deg]',ylabel='Dec [deg]',
                title='14 application fields · blue predicted / gray north')
    for split,label,color in [(0,'Train',BLUE),(1,'Cal',GREEN),(2,'Development test',ORANGE),(3,'Historical/quarantined',GRAY)]:
        use=pop['split']==split
        cells,count=np.unique(pop['cell'][use],return_counts=True)
        for c,n in zip(cells,count):
            ra=(c//18)*10;dec=(c%18)*10-90
            axes[1].add_patch(Rectangle((ra,dec),10,10,fc=color,alpha=.25,ec='none'))
            ROWS.append(dict(panel=label,cell=int(c),n=int(n),selection_density=n/cell_area(c)))
        axes[1].plot([],[],color=color,lw=5,label=label)
    axes[1].set(xlim=(360,0),ylim=(-90,90),xlabel='RA [deg]',ylabel='Dec [deg]',title='Expanded reference selection footprint')
    axes[1].legend(fontsize=10,loc='lower left')
    fig.suptitle('9,390,518 catalog rows · 8,899,334 distances · 3,565,499 reference candidates',fontsize=17)
    save(fig,'01_scope','Application rectangles are configuration boundaries, not masked effective area. Reference cells show selection footprint; counts and stars/deg² are in CSV.',model='baseline_v1 + expanded references',reference='all candidates',split='all')


def coverage(q):
    q=load(DEST/'chosen_expanded_A.npz')
    p=load(DEST/'parent.npz')
    begin()
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    edges=np.array([14,15,16,17,18,19,20,21,22,23,24,25.])
    for data,label,color,weight in [(p,'Weighted parent',GRAY,p['sample_weight']),
                                     (q,'Distance evaluation',BLUE,None)]:
        x=data['r']
        for split,style in ([(None,'-')] if label.startswith('Weighted') else [(1,'--'),(2,'-')]):
            use=np.ones(len(x),bool) if split is None else data['split']==split
            h=np.histogram(x[use],edges,weights=None if weight is None else weight[use])[0]
            axes[0].stairs(h/h.sum(),edges,label=label+('' if split is None else f' split {split}'),color=color,ls=style,lw=2)
            for lo,hi,n in zip(edges[:-1],edges[1:],h):
                ROWS.append(dict(panel=label,split=split,lo=lo,hi=hi,count_or_weight=n))
    import csv
    with open(TABLES/'training_brightness_hist.csv') as stream:
        records=list(csv.DictReader(stream))
    for family,color in [('distance',ORANGE),('class',GREEN)]:
        chosen=[row for row in records if row['family']==family]
        counts=np.array([int(row['n']) for row in chosen])
        e=np.r_[[float(row['lo']) for row in chosen],float(chosen[-1]['hi'])]
        axes[0].stairs(counts/counts.sum(),e,color=color,label=family+' train',lw=2)
        ROWS.extend(dict(panel=family+' train',**row) for row in chosen)
    pop=load(DEST/'population.npz')
    bins=np.array([14,18,20,22,25])
    for key,label,color in [('class_eligible','Class',GREEN),('label_source','Chosen',ORANGE),('parallax_eligible','Gaia parallax',BLUE)]:
        use=(pop['split']==2)&(pop[key]>0)
        counts=np.histogram(pop['r'][use],bins)[0]
        axes[1].plot((bins[:-1]+bins[1:])/2,counts,'o-',color=color,label=label)
        for lo,hi,n in zip(bins[:-1],bins[1:],counts):
            ROWS.append(dict(panel=label,lo=lo,hi=hi,n=int(n)))
    axes[0].set(xlabel='Observed r [mag]',ylabel='Fraction of each sample',title='Parent and spatial evaluation samples')
    axes[1].set(xlabel='Observed r [mag]',ylabel='Test evidence objects',yscale='symlog',title='Three independent evidence qualifications')
    for ax in axes:ax.legend()
    save(fig,'02a_brightness_coverage','Parent retains original probability weights. No common reference mask is used for class, chosen or signed Gaia evidence.',reference='independent eligibility',split='cal/test + parent')
    begin()
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    xe=np.linspace(-1,3,33);ye=np.arange(14.,26)
    for ax,t,label,w in [(axes[0],p,'Weighted parent',p['sample_weight']),(axes[1],subset(q,q['split']==2),'Development test',None)]:
        im=density(ax,t['gr'],t['r'],xe,ye,label,w)
        fig.colorbar(im,ax=ax,label='Parent weight' if w is not None else 'Reference objects')
        ax.set(xlabel='$(g-r)_0$',ylabel='Observed r [mag]',ylim=(25,14),title=label)
    save(fig,'02b_color_magnitude_coverage','Same displayed bins; white cells have no evidence. Counts and out-of-range objects are saved. Parent density represents old DECam scope.')


def distance(q,group='all',section='main'):
    begin()
    use=np.ones(len(q['identity']),bool) if group=='all' else strata(q)[group]
    t=subset(q,use)
    fig,axes=plt.subplots(2,3,figsize=(15.6,9.5),layout='constrained')
    density_artists=[]
    for j,kind in enumerate(('Gaia','specdis','spdist')):
        ref=t['ref_'+kind]
        good=np.isfinite(ref)
        x=10**((ref[good]-10)/5);y=t['distance_kpc'][good,2]
        err=t['epsilon_pct_'+kind][good]
        if not len(x):
            for ax in axes[:,j]:ax.text(.5,.5,'No eligible reference',ha='center',transform=ax.transAxes)
            panel(kind,len(t['identity']),0)
            continue
        lo,hi=limits(np.log10(np.r_[x,y]))
        ed=10**np.linspace(lo,hi,45)
        im=density(axes[0,j],x,y,ed,ed,kind)
        density_artists.append(im)
        fig.colorbar(im,ax=axes[0,j],label='Objects',shrink=.7)
        axes[0,j].plot(ed,ed,'--',c=ORANGE,lw=1.5)
        axes[0,j].set(xscale='log',yscale='log',xlim=(ed[0],ed[-1]),ylim=(ed[0],ed[-1]),
                      xlabel='Reference distance [kpc]',ylabel='Predicted distance [kpc]',title=kind,aspect='equal')
        ax=axes[1,j]
        ax.scatter(x,err,s=2,color=BLUE,alpha=.08,rasterized=True)
        ax.set(xscale='log',yscale='symlog',ylim=limits(np.r_[err,0]),xlabel='Reference distance [kpc]',ylabel='Distance residual [%]')
        for value in (-50,-20,0,20,50):
            ax.axhline(value,color=GRAY,lw=.7,ls='-' if value==0 else ':')
        for a,b in zip(ed[:-1],ed[1:]):
            mask=(x>=a)&(x<b)
            n=int(mask.sum())
            row=dict(panel=kind+'_trend',xlo=a,xhi=b,n=n)
            if n>=20:
                v=np.quantile(err[mask],[.16,.5,.84])
                center=np.sqrt(a*b)
                ax.plot(center,v[1],'o',ms=3,color='black')
                ax.vlines(center,v[0],v[2],color='black',alpha=.4,lw=1)
                row.update(p16=v[0],median=v[1],p84=v[2])
            ROWS.append(row)
        stats=percent_metrics(t['mu'][good],ref[good])
        ax.text(.02,.97,f'N={stats["n"]:,}\nbias={stats["median_pct"]:+.1f}% · NMAD={stats["nmad_pct"]:.1f}%\n1 mag tail={stats["tail1mag"]:.1%}',va='top',transform=ax.transAxes,fontsize=10,bbox=dict(fc='white',alpha=.85,ec='none'))
    if density_artists:
        norm=LogNorm(1,max(2,max(im.get_array().max() for im in density_artists)))
        for im in density_artists:im.set_norm(norm)
    save(fig,'03_distance_'+group,f'expanded_A, {group}, all eligible source-specific objects. Full residuals retained with symlog. Lines at 0, ±20%, ±50%. Black marks show median and 16–84% object spread, not confidence limits.',section=section,reference='Gaia/specdis/spdist')


def cmd(q,group='all',section='main'):
    begin()
    mask=np.isfinite(q['ref_chosen'])
    if group!='all':mask &= strata(q)[group]
    t=subset(q,mask)
    x=t['gr'];yr=t['g0']-t['ref_chosen'];yp=t['g0']-t['mu'][:,2];err=t['epsilon_pct_chosen']
    xe=np.linspace(*limits(x),46);ye=np.linspace(*limits(np.r_[yr,yp]),46)
    fig,axes=plt.subplots(2,2,figsize=(12.8,10.5),layout='constrained')
    density_artists=[]
    for ax,y,label in [(axes[0,0],yr,'Reference CMD'),(axes[0,1],yp,'Predicted CMD')]:
        im=density(ax,x,y,xe,ye,label)
        density_artists.append(im)
        fig.colorbar(im,ax=ax,label='Objects')
        ax.set_title(label)
    shared=LogNorm(1,max(2,max(im.get_array().max() for im in density_artists)))
    for im in density_artists:im.set_norm(shared)
    for ax,absolute,title in [(axes[1,0],False,'Median signed residual [%]'),(axes[1,1],True,'Median absolute residual [%]')]:
        im=map_error(ax,x,yr,err,xe,ye,title,absolute)
        fig.colorbar(im,ax=ax,label=title,extend='max' if absolute else 'both')
        ax.set_title(title)
    for ax in axes.flat:ax.set(xlabel='$(g-r)_0$',ylabel='Legacy $M_g$ [mag]',ylim=(ye[-1],ye[0]))
    save(fig,'04_cmd_'+group,f'Same {len(x):,} objects, expanded_A chosen references. Mg_phot-Mg_ref=-delta_mu. Both CMD coordinates and residuals share distance estimates; associations do not establish physical causes.',section=section)


def color_map(q,kind='grz',group='all',section='main'):
    begin()
    axes_vars={'grz':('gr','rz','$(g-r)_0$','$(r-z)_0$'),
               'grW1':('gr','zW1','$(g-r)_0$','$(z-W1)_0$'),
               'WISE':('zW1','W1W2','$(z-W1)_0$','$(W1-W2)_0$'),
               'griz':('ri','iz','$(r-i)_0$','$(i-z)_0$')}
    xkey,ykey,xlabel,ylabel=axes_vars[kind]
    use=np.isfinite(q['ref_chosen'])
    if group!='all':use &= strata(q)[group]
    t=subset(q,use)
    x,y=t[xkey],t[ykey]
    valid=np.isfinite(x)&np.isfinite(y)
    xe=np.linspace(*limits(x[valid]),38);ye=np.linspace(*limits(y[valid]),38)
    fig,axes=plt.subplots(1,3,figsize=(16,5.2),layout='constrained')
    im=density(axes[0],x,y,xe,ye,'density');fig.colorbar(im,ax=axes[0],label='Objects')
    axes[0].set_title('Reference density')
    for ax,ab,title in [(axes[1],False,'Median residual [%]'),(axes[2],True,'Median absolute residual [%]')]:
        im=map_error(ax,x,y,t['epsilon_pct_chosen'],xe,ye,title,ab)
        fig.colorbar(im,ax=ax,label='Distance residual [%]',extend='max' if ab else 'both')
        ax.set_title(title)
    for ax in axes:ax.set(xlabel=xlabel,ylabel=ylabel)
    save(fig,f'05_colors_{kind}_{group}',f'expanded_A optical residuals in {kind} coordinates, {group}; WISE colors require state4, i colors require usable i. No negative flux-to-magnitude conversion. Full coordinate extents, N>=20 performance cells.',section=section)


def physical(q):
    use=q['rvs_parameter_quality']&np.isfinite(q['ref_chosen'])
    t=subset(q,use)
    begin()
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    xe=np.linspace(2500,12000,48);ye=np.linspace(-1,6,36)
    im=density(axes[0],t['rvs_teff'],t['rvs_logg'],xe,ye,'Kiel density');fig.colorbar(im,ax=axes[0],label='Objects')
    im=map_error(axes[1],t['rvs_teff'],t['rvs_logg'],t['epsilon_pct_chosen'],xe,ye,'Kiel residual')
    fig.colorbar(im,ax=axes[1],label='Median residual [%]',extend='both')
    for ax in axes:ax.set(xlabel='RVS temperature [K]',ylabel='RVS log g',xlim=(12000,2500),ylim=(6,-1))
    save(fig,'06a_kiel','Qualified continuous RVS parameters, expanded_A. Fixed display ranges and excluded counts retained; coarse evolved/other labels do not assign RGB/RC/HB phases.')
    begin()
    fig,axes=plt.subplots(1,3,figsize=(16,5.2),layout='constrained')
    corrected,domain=rvs_metallicity(t['rvs_feh'],t['rvs_teff'],t['rvs_logg'])
    quality=np.isfinite(t['rvs_feh_err'])&(t['rvs_feh_err']>0)
    for ax,value,mask,title in zip(axes,[t['rvs_feh'],t['rvs_feh'],corrected],
                                  [quality,quality&domain,quality&domain],
                                  ['Raw RVS','Raw within correction domain','Corrected within domain']):
        for code,label,color in CLASSES[1:]:
            m=mask&(t['class_label']==code)
            trend(ax,subset(t,m),value[m],np.array([-4,-2,-1.5,-1,-.5,0,.5,1]),'median_pct',label,color)
        ax.axhline(0,c=GRAY,ls='--');ax.set(xlabel='[Fe/H]',ylabel='Median distance residual [%]',title=title)
        ax.legend(fontsize=10)
    save(fig,'06b_metallicity','RVS abundance correction restricted to 4200<=Teff<6600 and 1.8<=logg<5.5. Raw/domain/raw-corrected comparisons use finite FeH and positive formal error. Metal-poor is not halo membership.')
    begin()
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    for ax,key,edges,title in [(axes[0],'rvs_logg',np.linspace(-1,6,15),'RVS log g'),(axes[1],'rvs_teff',np.linspace(2500,12000,20),'RVS temperature [K]')]:
        for code,label,color in CLASSES[1:]:
            m=t['class_label']==code
            trend(ax,subset(t,m),t[key][m],edges,'median_pct',label,color)
        ax.set(xlabel=title,ylabel='Median distance residual [%]');ax.legend()
    save(fig,'06c_spectral_trends','RVS-qualified sources, N>=20 bins. Per-class counts and full percentage metrics in CSV.')


def conditions(q,key='r',section='main'):
    edges={'r':np.array([14,16,18,19,20,21,22,25.]),'optical_error':np.array([0,.01,.03,.1,.3,1.]),
           'EBV':np.linspace(0,.03,8),'snr_w1':np.array([-10,0,2,5,10,20,50,100,1000.]),
           'ref_distance':np.array([0,.25,.5,1,2,5,10,20,50,200,2000.])}[key]
    metrics=[('median_pct','Median bias [%]'),('nmad_pct','NMAD [%]'),('tail1mag','Severe-error fraction')]
    begin()
    fig,axes=plt.subplots(1,3,figsize=(16,5.2),layout='constrained')
    for ax,(metric,label) in zip(axes,metrics):
        for code,name,color in CLASSES:
            t=subset(q,q['class_label']==code)
            trend(ax,t,t[key],edges,metric,name,color)
        t=subset(q,~q['B']);trend(ax,t,t[key],edges,metric,'No W1','#9467bd')
        ax.set(xlabel='Observed r [mag]' if key=='r' else key,ylabel=label)
        if metric=='tail1mag':ax.set_ylim(0,1)
        ax.legend(fontsize=10)
    save(fig,'07_conditions_'+key,'expanded_A by class and no-W1 group. Separate markers show occupied bins; sparse layers not interpolated. Complete bin counts and percentage metrics are in CSV.',section=section)
    if key!='r':return
    begin()
    fig,axes=plt.subplots(1,3,figsize=(16,5.2),layout='constrained')
    for code,label,color in CLASSES:
        t=subset(q,q['class_label']==code)
        trend(axes[0],t,t['r'],edges,'coverage68',label,color)
        trend(axes[1],t,t['r'],edges,'width68_mag',label,color)
        h=np.histogram(t['r'][np.isfinite(t['ref_chosen'])],edges)[0]
        axes[2].stairs(h,edges,label=label,color=color,lw=2)
    axes[0].axhline(.68,c=GRAY,ls=':');axes[0].set(ylim=(0,1),ylabel='68% reference-point coverage')
    axes[1].set_ylabel('Median 68% width [mag]')
    axes[2].set(ylabel='Reference count',yscale='symlog')
    for ax in axes:ax.set_xlabel('Observed r [mag]');ax.legend(fontsize=10)
    save(fig,'07b_intervals_and_counts','Width and coverage are displayed together. Sparse faint layers retain actual N and do not imply broad validation.')


def uncertainty(q):
    begin()
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    for code,label,color in CLASSES:
        t=subset(q,q['class_label']==code)
        s=percent_metrics(t['mu'],t['ref_chosen'])
        if not s['n']:continue
        axes[0].plot(s['width68_mag'],s['coverage68'],'o',c=color,label=f'{label} N={s["n"]:,}')
        axes[1].plot(s['width95_mag'],s['coverage95'],'o',c=color,label=label)
        ROWS.append(dict(panel=label,**s))
    for ax,level in zip(axes,[.68,.95]):
        ax.axhline(level,c=GRAY,ls=':');ax.set(xlabel='Median interval width [mag]',ylabel='Reference-point coverage',ylim=(0,1));ax.legend(fontsize=10)
    save(fig,'08a_coverage_width','Raw expanded_A intervals relative to chosen catalog point estimates. Catalog uncertainties and common upstream information are not removed.')
    begin()
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    for ax,layers in zip(axes,[[('all',np.ones(len(q['identity']),bool),BLUE)],[(name,q['class_label']==code,color) for code,name,color in CLASSES]]):
        for label,m,color in layers:
            m &= q['parallax_eligible']
            h=np.histogram(q['full_pit'][m],np.linspace(0,1,11))[0]
            if h.sum():ax.stairs(h/h.sum(),np.linspace(0,1,11),label=f'{label} N={h.sum():,}',color=color,lw=2)
            for j,n in enumerate(h):ROWS.append(dict(panel=label,lo=j/10,hi=(j+1)/10,n=int(n),negative=int((m&(q['p_corr']<0)).sum())))
        ax.axhline(.1,c=GRAY,ls=':');ax.set(xlabel='Observed-parallax PIT',ylabel='Fraction per bin');ax.legend(fontsize=10)
    save(fig,'08b_gaia_pit','All signed eligible Gaia parallaxes from refined sidecars. Negative observations retained. Class-conditional PIT is diagnostic and does not alone prove unconditional mixture weights wrong.',reference='signed Gaia')


def availability():
    p=load(DEST/'parent.npz')
    begin()
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    edges=np.array([14,16,18,20,22,25.])
    for key,label,color in [('i_usable','i',GREEN),('w1','W1',BLUE),('w2','W2',ORANGE)]:
        high=p[key] if key=='i_usable' else p[key+'_measurement_state']==4
        valid=high if key=='i_usable' else p[key+'_measurement_state']>=3
        for ax,m,title in [(axes[0],high,'High-S/N usable'),(axes[1],valid,'Quality valid, signed weak allowed')]:
            vals=[]
            for lo,hi in zip(edges[:-1],edges[1:]):
                use=(p['r']>=lo)&(p['r']<hi)
                value=np.average(m[use],weights=p['sample_weight'][use]) if use.any() else np.nan
                vals.append(value);ROWS.append(dict(panel=title,band=label,lo=lo,hi=hi,n=int(use.sum()),fraction=value))
            ax.plot((edges[:-1]+edges[1:])/2,vals,'o-',c=color,label=label)
    for ax in axes:ax.set(xlabel='Observed r [mag]',ylabel='Weighted usable fraction',ylim=(0,1));ax.legend()
    save(fig,'09a_band_availability','Original26000 probability sample, original DECam weights. High S/N and signed valid measurements separated. Exact application route counts are a different population statistic.',model='measurement states',reference='none',split='weighted parent')
    begin()
    fig,axes=plt.subplots(1,2,figsize=(14,5.5),layout='constrained')
    names=['Missing','Invalid','Quality fail','Weak','High S/N']
    for ax,band in zip(axes,['w1','w2']):
        state=p[band+'_measurement_state'];v=[]
        for i,name in enumerate(names):
            use=state==i
            fraction=np.average(use,weights=p['sample_weight'])
            v.append(fraction);ROWS.append(dict(panel=band,state=name,n=int(use.sum()),weight=p['sample_weight'][use].sum(),fraction=fraction))
        ax.bar(range(5),v,color=[GRAY,GRAY,ORANGE,GREEN,BLUE]);ax.set(xticks=range(5),xticklabels=names,ylim=(0,1),ylabel='Weighted parent fraction',title=band.upper())
        ax.tick_params(axis='x',rotation=25)
    valid=np.column_stack([p[k+'_measurement_state']>=3 for k in ('w1','w2')])
    for x,y,label in [(1,0,'W1-only'),(0,1,'W2-only'),(1,1,'Both'),(0,0,'Neither')]:
        m=(valid[:,0]==x)&(valid[:,1]==y)
        ROWS.append(dict(panel='joint_valid',state=label,n=int(m.sum()),weight=p['sample_weight'][m].sum(),fraction=np.average(m,weights=p['sample_weight'])))
    save(fig,'09b_wise_states','Five states retain missing, invalid and quality failure apart from signed weak and high-S/N measurements. Joint W1-only/W2-only/both/neither in CSV.',model='measurement states',reference='none',split='weighted parent')


def joint_states():
    p=load(DEST/'parent.npz')
    begin()
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    for ax,threshold,title in [(axes[0],4,'High S/N'),(axes[1],3,'Quality valid incl. weak')]:
        valid=np.column_stack([p[b+'_measurement_state']>=threshold for b in ('w1','w2')])
        fractions=[]
        labels=['W1-only','W2-only','Both','Neither']
        for x,y,label in zip([1,0,1,0],[0,1,1,0],labels):
            m=(valid[:,0]==x)&(valid[:,1]==y)
            f=np.average(m,weights=p['sample_weight']);fractions.append(f)
            ROWS.append(dict(panel=title,state=label,n=int(m.sum()),fraction=f,weight=p['sample_weight'][m].sum()))
        ax.bar(range(4),fractions,color=[BLUE,ORANGE,GREEN,GRAY])
        ax.set(title=title,xticks=range(4),xticklabels=labels,ylabel='Weighted parent fraction',ylim=(0,1))
    save(fig,'09c_joint_states','Independent per-band measurement availability; quality failures remain excluded. Original weighted parent, not exact deployment routing.',model='measurement states',reference='none',split='weighted parent')


def experiments():
    science=json.loads((ROOT/'manifests/baseline_v1_4_science.json').read_text())
    r0=json.loads((ROOT/'manifests/baseline_v1_5_r0_result.json').read_text())['scores']['development_test']
    weak=json.loads((ROOT/'manifests/baseline_v1_6_wise_result.json').read_text())['scores']['development_test']
    begin()
    fig,axes=plt.subplots(1,2,figsize=(15,6.4),layout='constrained')
    keys=['all','dwarf','evolved','other_transition','unknown']
    pairs=[('Expanded A',science['pairs']['expanded_A']['chosen']),
           ('Matched griz',science['pairs']['matched_griz']['chosen']),
           ('Added r0',{k:r0['all' if k=='all' else {'dwarf':'class0','evolved':'class1','other_transition':'class2','unknown':'class-1'}[k]]['chosen']['paired'] for k in keys})]
    for j,(label,source) in enumerate(pairs):
        for i,key in enumerate(keys):
            p=source[key]
            if not p['n']:continue
            x=100*p['tail_difference'];lo,hi=p['tail_ci95'];y=i+(j-1)*.22
            c=[BLUE,GREEN,ORANGE][j]
            if lo is not None:axes[0].hlines(y,100*lo,100*hi,colors=c,lw=1.2)
            axes[0].plot(x,y,'o',c=c,label=label if i==0 else None)
            ROWS.append(dict(panel=label,group=key,**p))
    axes[0].set(yticks=range(5),yticklabels=keys,xlabel='Tail-rate change [pp]',title='Separate fixed matched comparisons')
    axes[0].axvline(0,c=GRAY,ls='--');axes[0].legend(fontsize=10);axes[0].invert_yaxis()
    for j,(group,a,b,title) in enumerate([('r18_20:w1_weak','optical','w1','Weak W1'),
                                         ('r18_20:w2_weak_given_w1','w1','w1w2','Weak W2 with W1')]):
        d=weak[group]['chosen']
        p=d['w1_minus_optical' if a=='optical' else 'w1w2_minus_w1']
        axes[1].barh(j-.16,p['rescued'],height=.3,color=GREEN,label='Rescued' if j==0 else None)
        axes[1].barh(j+.16,p['harmed'],height=.3,color=ORANGE,label='Harmed' if j==0 else None)
        ROWS.append(dict(panel=title,group=group,**p,old_coverage68=d[a]['coverage68'],new_coverage68=d[b]['coverage68'],old_width68=d[a]['width68_mag'] if 'width68_mag' in d[a] else d[a]['width68'],new_width68=d[b]['width68_mag'] if 'width68_mag' in d[b] else d[b]['width68']))
    axes[1].set(yticks=[0,1],yticklabels=['Weak W1','Weak W2 + W1'],xlabel='Same-star transitions',title='18–20 real weak measurements');axes[1].legend()
    save(fig,'10a_experiment_tradeoffs','Each experiment has its own matched identities, training controls and N in CSV. These are not a continuous version ranking. r0 preserves max_features0.75, changing realized split candidates1→2.',model='paired experiments',reference='chosen')

    old=json.loads((ROOT/'manifests/baseline_v1_report.json').read_text())
    begin()
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    for ax,key in zip(axes,['W1','W1W2']):
        source=old['same_star'][key]
        labels=[];values=[]
        for name,models in source.items():
            if 'wide' not in models:continue
            refs=models['wide']
            if 'chosen' not in refs:continue
            d=refs['chosen']['all'];labels.append(name);values.append(d['tail_1'])
            ROWS.append(dict(panel=key,model=name,**d))
        ax.bar(range(len(values)),values,color=[BLUE,GREEN,ORANGE][:len(values)])
        ax.set(xticks=range(len(labels)),xticklabels=labels,ylabel='Severe-error fraction',title='High-S/N matched '+key)
    save(fig,'10b_high_snr_information','Existing v1 A_on_B/B and A_on_C/B_on_C/C matched training and source controls. Original specific model identifiers retained.',model='baseline_v1 matched bands',reference='chosen',split='old development test')


def appendix(q):
    for group in ('dwarf','evolved','other','unknown'):
        distance(q,group,'appendix');cmd(q,group,'appendix')
    for kind in ('grz','grW1','WISE','griz'):
        for group in ('all','dwarf','evolved','other','unknown'):
            if kind=='grz' and group=='all':continue
            color_map(q,kind,group,'appendix')
    for key in ('ref_distance','optical_error','snr_w1','EBV'):conditions(q,key,'appendix')
    begin()
    fig,axes=plt.subplots(1,3,figsize=(16,5.2),layout='constrained')
    for name,mask,color in [('supported',q['supported'],BLUE),('ambiguous',q['ambiguous'],ORANGE),('outside_train_precision',q['outside_train_precision'],GREEN)]:
        stats=[]
        for code,label,_ in CLASSES:
            use=mask&(q['class_label']==code)
            s=percent_metrics(q['mu'][use],q['ref_chosen'][use]);ROWS.append(dict(panel=name,group=label,**s));stats.append(s.get('tail1mag',np.nan))
        ax=axes[['supported','ambiguous','outside_train_precision'].index(name)]
        ax.bar(range(4),stats,color=color);ax.set(xticks=range(4),xticklabels=[r[1] for r in CLASSES],ylim=(0,1),ylabel='Severe-error fraction',title=name);ax.tick_params(axis='x',rotation=25)
    save(fig,'A_flags','Model-state groups do not define a reliable single-distance catalog. Rows may overlap; all counts and risks retained.','appendix')
    begin()
    fig,axes=plt.subplots(1,3,figsize=(16,5.2),layout='constrained')
    groups0=[]
    for cell in np.unique(q['cell']):
        use=q['cell']==cell;s=percent_metrics(q['mu'][use],q['ref_chosen'][use])
        groups0.append((cell,s))
        ROWS.append(dict(panel='sky',cell=int(cell),area_sqdeg=cell_area(cell),**s))
    for ax,key,label in [(axes[0],'n','Reference count'),(axes[1],'median_pct','Median residual [%]'),(axes[2],'tail1mag','Severe-error fraction')]:
        for cell,s in groups0:
            if not s['n']:continue
            ra=cell//18*10;dec=cell%18*10-90
            value=s[key] if key=='n' or s['n']>=20 else np.nan
            if np.isfinite(value):
                norm=LogNorm(1,max(r['n'] for _,r in groups0)) if key=='n' else Normalize(-100,100) if key=='median_pct' else Normalize(0,1)
                cmap=plt.get_cmap('Blues' if key=='n' else 'coolwarm' if key=='median_pct' else 'Oranges')
                ax.add_patch(Rectangle((ra,dec),10,10,fc=cmap(norm(value)),ec=GRAY,lw=.3))
            else:ax.add_patch(Rectangle((ra,dec),10,10,fc='#f4ebd9',hatch='xx',ec=GRAY))
        im=plt.cm.ScalarMappable(norm=norm,cmap=cmap)
        fig.colorbar(im,ax=ax,label=label)
        ax.set(xlim=(360,0),ylim=(-90,90),xlabel='RA [deg]',ylabel='Dec [deg]',title=label)
    save(fig,'A_sky','Chosen test selection only. Sparse cells hatched. Per-cell solid-angle area is retained for density comparisons. No interpolation.','appendix')


def reused_appendices():
    old=json.loads((ROOT/'manifests/baseline_v1_report.json').read_text())
    begin()
    candidates=old['quality_diagnostics']['candidates']
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    for d in candidates:
        f=d['parent_fraction']
        for ax,split,color in [(axes[0],'cal',BLUE),(axes[1],'test',ORANGE)]:
            v=d[split]['all']['raw']
            if v['n']:
                ax.plot(f,v['tail_1'],'o',color=color,alpha=.3,ms=3)
            ROWS.append(dict(panel=split,parent_fraction=f,**d['cuts'],**v))
    for ax,label in zip(axes,['Cal','Old development test']):
        ax.set(xlabel='Exact parent retention fraction',ylabel='Raw severe-error fraction',title=label)
    save(fig,'A_quality_retention','Existing fixed v1 quality grid; no re-optimization. All-class adoption failed. Raw/calibrated intervals remain distinct in the original source.','appendix',model='baseline_v1',split='old cal/test')
    begin()
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    labels=[];v0=[];v1=[];width=[]
    for name,d in old['depth']['groups'].items():
        if name!='all' and not name.startswith('observed_r_'):continue
        labels.append(name);v0.append(d['models']['v0']['metrics']['tail_1']);v1.append(d['models']['v1']['metrics']['tail_1'])
        ROWS.append(dict(panel=name,n_sources=d['n_sources'],n_accepted=d['n_accepted'],v0_tail=v0[-1],v1_tail=v1[-1]))
    for ax,values,title in [(axes[0],v0,'v0'),(axes[1],v1,'v1')]:
        ax.bar(range(len(labels)),values,color=BLUE);ax.set(xticks=range(len(labels)),xticklabels=labels,ylabel='Severe-error fraction',title=title)
    save(fig,'A_flux_depth_stress','Existing fixed real-template artificial flux degradation. Stress-test sources are reused; it does not supply new faint-population truth.','appendix',model='baseline_v0/v1',split='existing depth stress')


def pdf_examples(q):
    import joblib
    from baseline_v1_4_experiments import arrays
    begin()
    ref=q['ref_chosen'];res=q['delta_mu_chosen']
    corrected,domain=rvs_metallicity(q['rvs_feh'],q['rvs_teff'],q['rvs_logg'])
    finite=np.isfinite(ref)
    masks=dict(ordinary_dwarf=finite&(q['class_label']==0)&(abs(res)<.2),
               metal_poor_dwarf=finite&(q['class_label']==0)&domain&(corrected<-1.5),
               evolved_failure=finite&(q['class_label']==1)&(abs(res)>1),
               multimodal_or_wide=finite&(q['ambiguous']|((q['mu'][:,3]-q['mu'][:,1])>3)))
    selected=fixed_examples(q['identity'],masks,per_group=1,seed=20261105)
    model=joblib.load(OUT/'experiments/expanded_A_distance.joblib')
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    payload={}
    for ax,(name,rows) in zip(axes.flat,selected.items()):
        if not rows:
            ax.set_title(name+' · unavailable');continue
        row=rows[0];m,e,ids=arrays(q,np.array([row]))
        prediction=integrate(model,m,e,ids,draws=128)['wide']
        assert np.allclose(prediction['mu'][0],q['mu'][row],rtol=0,atol=1e-6)
        edges=q['g0'][row]-model.edges[::-1];p=prediction['pdf'][0,::-1]/np.diff(edges)
        ax.stairs(p,edges,color=BLUE,lw=1.8)
        ax.axvline(ref[row],c=ORANGE,lw=1.8,label='Reference')
        err=q['sigma_mu'][row]
        ax.axvspan(ref[row]-err,ref[row]+err,color=ORANGE,alpha=.2)
        ax.axvline(q['mu'][row,2],c=GREEN,ls='--',label='Median')
        ax.set(title=f'{name}\nID={q["identity"][row]}',xlabel='Distance modulus [mag]',ylabel='Density [1/mag]')
        ax.legend(fontsize=10)
        for lo,hi,d in zip(edges[:-1],edges[1:],p):ROWS.append(dict(panel=name,identity=int(q['identity'][row]),lo=lo,hi=hi,density=d,reference=ref[row],reference_sigma=err))
        payload[name+'_edges']=edges;payload[name+'_pdf']=prediction['pdf'][0,::-1]
    atomic_npz(DEST/'fixed_pdf_examples.npz',**payload)
    save(fig,'A_fixed_pdf_examples','One object per preregistered group, fixed hash order. Four forward calls to frozen expanded_A, same Sobol128, quantiles reproduced within1e-6. Shading shows catalog reference error; examples do not estimate mechanism frequency.','appendix')


def main():
    require_compute()
    completion=json.loads((DEST/'collection_complete.json').read_text())
    assert sha(TABLES/'snapshot.json')==completion['snapshot_sha256']
    plt.rcParams.update(STYLE)
    q=load(DEST/'chosen_expanded_A.npz')
    q=subset(q,q['split']==2)
    global PRIMARY
    PRIMARY=q
    assert np.isfinite(q['ref_chosen']).sum()==47756
    scope();coverage(q);distance(q);cmd(q);color_map(q);physical(q);conditions(q)
    uncertainty(q);availability();joint_states();experiments();appendix(q);reused_appendices();pdf_examples(q)
    save_json(TABLES/'figure_registry.json',REGISTRY)
    print('COMPLETE figures',len(REGISTRY),flush=True)


if __name__=='__main__':
    main()
