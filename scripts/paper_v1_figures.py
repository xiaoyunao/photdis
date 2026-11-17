#!/usr/bin/env python3
"""Render compact AASTeX figures from the frozen product and new validation tables."""
import csv
import json
import os

import numpy as np
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from scipy.stats import binned_statistic_2d

from photdis.stage1 import require_compute
from v2_1_near import ROOT, save_json, sha
from v3_3_sed import verified
from v3_8_direct_figures import BLUE, GREEN, ORANGE, STYLE, plt, save
from v3_10_figures import check_templates
from v3_final_atlas import COLORMAP, MIN_AREA, plotting_grid
from paper_v1_benchmark import CLASSES, DIMS, EDGES, NMIN, OUT, REFS

PAPER = {**STYLE, 'font.size':8, 'axes.labelsize':9, 'axes.titlesize':9,
         'xtick.labelsize':8, 'ytick.labelsize':8, 'legend.fontsize':8}
GROUPS = ('Ordinary dwarfs','Metal-poor dwarfs','Intermediate gravity')
SYSTEMS = ('DECam','BASS/MzLS')
NAVY, GRAY = '#24427c', '#666666'


def reference(dest,t):
    fig,axs=plt.subplots(2,3,figsize=(7.1,4.95),layout='constrained')
    rows,data=[],{}
    edges=np.geomspace(.055,90,85)
    rex=np.linspace(-6,6,145)
    for j,ref in enumerate(REFS):
        d=t[ref.lower()+'_kpc']
        use=t[ref+'_good']
        records=[]
        for k,(method,name,color) in enumerate((('photdis','Photdis',BLUE),('bj','BJ photogeometric',ORANGE))):
            q=t[method][use,1]
            residual=5*np.log10(q/d[use])
            h=np.histogram2d(d[use],q,(edges,edges))[0]
            data[f'{ref}_{method}_count']=h
            records.append((k,name,color,q,residual,h))
        vmax=max(x[-1].max() for x in records)
        for k,name,color,q,residual,h in records:
            ax=axs[j,k]
            z=ax.pcolormesh(edges,edges,np.ma.masked_equal(h.T,0),norm=LogNorm(1,vmax),
                            cmap='viridis',rasterized=True)
            ax.plot(edges[[0,-1]],edges[[0,-1]],'--',color=ORANGE,lw=.9)
            ax.set(xscale='log',yscale='log',xlim=(1.9,5.15),ylim=edges[[0,-1]],
                   xlabel=f'{ref} (kpc)',title=name if j==0 else '')
            ax.set_xticks([2,3,4,5],labels=['2','3','4','5'])
            if k==0:
                ax.set_ylabel(r'$d_{50}$ (kpc)')
                ax.text(.04,.94,f'{use.sum():,} stars',transform=ax.transAxes,va='top',fontsize=8)
            else:
                ax.set_yticklabels([])
            ax=axs[j,2]
            hh=np.histogram(residual,rex)[0]
            data[f'{ref}_{name.lower().replace(" ","_")}_residual']=hh
            ax.stairs(hh/len(q),rex,color=color,linewidth=1.2,label=name)
            rows.append(dict(reference=ref,method=name,n=len(q),density_displayed=int(h.sum()),
                 residual_displayed=int(hh.sum()),below_density=int((q<edges[0]).sum()),
                 above_density=int((q>=edges[-1]).sum()),residual_left=int((residual<rex[0]).sum()),
                 residual_right=int((residual>=rex[-1]).sum())))
        fig.colorbar(z,ax=axs[j,:2].tolist(),pad=.01,label='Stars/bin',fraction=.045)
        axs[j,2].axvline(0,color='.45',ls=':',lw=.8)
        axs[j,2].set(xlabel=r'$\Delta\mu$ (mag)',ylabel='Fraction/bin',xlim=rex[[0,-1]],
                   yticks=np.arange(0,.26,.05),ylim=(0,.25))
        axs[j,2].set_xscale('symlog',linthresh=.7)
        axs[j,2].set_xticks([-3,0,3],labels=['−3','0','3'])
        if j==0: axs[j,2].legend(frameon=False,fontsize=7,loc='upper left')
    save(fig,dest,'f15_same_star_distances',rows,dict(distance_edges=edges,residual_edges=rex,**data))


def occupied(dest,b):
    m=b/'observables.npz'
    with np.load(m) as z:
        count=z['counts'].reshape((3,*DIMS))
        ref=z['reference_counts'].reshape((2,*DIMS))
    x=np.asarray(EDGES[1][1:-1])
    y=np.asarray(EDGES[0][1:-1])
    sums=lambda a:a.sum(axis=(0,3,4,5,6))[1:-2,1:-2]
    fig,axes=plt.subplots(2,4,figsize=(7.15,4.5),layout='constrained',sharex=True,sharey=True)
    bars={}
    rows=[]
    plates=[]
    for system,region in (('DECam',[0,1,2]),('BASS/MzLS',[3])):
        source=[sums(a[region]) for a in count]
        total=source[1]+source[2]
        covered=sums(((ref>=NMIN).any(axis=0)*(count[1]+count[2]))[region])
        ref_n=sums(ref[0,region])
        plates.append((source[0],total,ref_n,np.divide(covered,total,
                       out=np.full_like(total,np.nan,dtype=float),where=total>0)))
        rows.append(dict(system=system,pdf_n=int(source[0].sum()),core_n=int(total.sum()),
                         common_n=int(covered.sum()),reference_n=int(ref_n.sum())))
    for k,title in enumerate(('Full PDFs','Science core','SpecDis pair','Core overlap')):
        if k<3:
            nz=np.concatenate([p[k][p[k]>0] for p in plates])
            norm=LogNorm(max(1.,float(np.quantile(nz,.02))),float(nz.max()))
            cmap='viridis'
        else:
            norm,cmap=Normalize(0,1),'Blues'
        for j,system in enumerate(('DECam','BASS/MzLS')):
            mat=plates[j][k]
            ax=axes[j,k]
            valid = np.ma.masked_equal(mat,0) if k<3 else np.ma.masked_invalid(mat)
            bars[k]=ax.pcolormesh(x,y,valid,norm=norm,cmap=cmap,rasterized=True)
            if j==0: ax.set_title(title)
            if k==0: ax.set_ylabel(system+'\nObserved r (mag)')
            if j==1: ax.set_xlabel(r'$(g-r)_0$')
            ax.set(xlim=x[[0,-1]],ylim=y[[-1,0]])
            for l in (18,20): ax.axhline(l,color='.6',lw=.55,ls=':')
        fig.colorbar(bars[k],ax=axes[:,k].tolist(),location='bottom',pad=.08,
                     fraction=.09,label='Stars/bin' if k<3 else 'Core fraction')
    save(fig,dest,'observable_support',rows,dict(g_r_edges=x,r_edges=y,catalogue=count,reference=ref))


def pdf_diagnostics(dest,b):
    rs=np.load(b/'spectral_pit.npz')
    signed=np.load(b/'signed_pit.npz')
    x=rs['edges']
    fig,axes=plt.subplots(3,3,figsize=(7.1,5.25),layout='constrained',sharex=True)
    rows=[]
    for j,ref in enumerate(REFS):
        for k,label in enumerate(GROUPS):
            ax=axes[j,k]
            # All r=18–20 references preserve tails outside the 2–5 kpc selection.
            key=f'{ref}_all_{k+1}_18_20_all'
            for tag,color,name in (('center',BLUE,'Reference center'),('forward',ORANGE,'Error-convolved')):
                h=rs[f'{key}_{tag}']
                ax.stairs(h/max(1,h.sum()),x,color=color,label=name,lw=1.15)
                rows.append(dict(reference=ref,population=label,diagnostic=tag,n=int(h.sum())))
            ax.axhline(.1,ls=':',color='.4',linewidth=.7)
            if j==0: ax.set_title(label)
            if k==0: ax.set_ylabel(f'{ref}\nFraction/bin')
    for k,label in enumerate(CLASSES[:3]):
        ax=axes[2,k]
        for tag,c,name in ((4,BLUE,'S/N < 5'),(5,ORANGE,'5 ≤ S/N < 10')):
            h=signed[f'{tag}_all_{label}_qrf_pdf']
            ax.stairs(h/max(1,h.sum()),x,color=c,lw=1.15,label=name)
            rows.append(dict(reference='Gaia signed',population=label,diagnostic=str(tag),n=int(h.sum())))
        ax.axhline(.1,ls=':',color='.4',linewidth=.7)
        ax.set_xlabel('Predictive CDF')
    axes[2,0].set_ylabel('Gaia signed\nFraction/bin')
    axes[0,0].legend(frameon=False,fontsize=7)
    axes[2,0].legend(frameon=False,fontsize=7)
    save(fig,dest,'pdf_calibration',rows)


def physical(dest,t):
    data,rows=[],[]
    x=np.linspace(-.12,1.6,44)
    my=np.linspace(-1.5,10.5,45)
    zy=np.linspace(-.15,1.05,42)
    d=t['specdis_kpc']
    c=t['g0']-t['r0']; rz=t['r0']-t['z0']
    good=np.isfinite(d)
    mr=np.full(len(d),np.nan); mr[good]=t['r0'][good]-10-5*np.log10(d[good])
    dm=np.full(len(d),np.nan); dm[good]=5*np.log10(t['photdis'][good,1]/d[good])
    fig,axs=plt.subplots(2,2,figsize=(7,5.5),layout='constrained')
    for i,(system,region) in enumerate(zip(SYSTEMS,((2,),(3,)))):
        use=good&np.isin(t['region'],region)
        for k,(y,by,title) in enumerate(((mr,my,r'$M_{r,\rm ref}$ (mag)'),(rz,zy,r'$(r-z)_0$'))):
            ax=axs[i,k]
            w=use&np.isfinite(y)&np.isfinite(c)&np.isfinite(dm)
            n=np.histogram2d(c[w],y[w],bins=(x,by))[0]
            val=binned_statistic_2d(c[w],y[w],dm[w],statistic='median',bins=(x,by)).statistic
            g=np.ma.masked_where((n<10)|~np.isfinite(val),val)
            cb=ax.pcolormesh(x,by,g.T,norm=Normalize(-.8,.8),cmap='coolwarm',rasterized=True)
            ax.set(xlabel=r'$(g-r)_0$',ylabel=title,title=system if k==0 else '')
            for v,ls,line in zip(CLASSES[:3],('solid','dotted','dashed'),('#202020',NAVY,'#376b29')):
                group=w&(t['group']==v)
                branch=np.histogram2d(c[group],y[group],bins=(x,by))[0]
                if branch.max()<12:
                    continue
                level=max(4,np.quantile(branch[branch>0],.72))
                ax.contour((x[:-1]+x[1:])*.5,(by[:-1]+by[1:])*.5,branch.T,
                           levels=[level],colors=[line],linestyles=[ls],linewidths=.55)
                data.append((f'branch_{v}_{i}_{k}',branch))
            if k==0: ax.invert_yaxis()
            data += [(f'n_{i}_{k}',n),(f'median_{i}_{k}',val)]
            rows.append(dict(system=system,reference='SpecDis',coordinates='CMD' if k==0 else 'color_color',
                             n=int(w.sum()),inside=int(n.sum()),shown=int(n[n>=10].sum()),n_min=10))
    fig.colorbar(cb,ax=axs.ravel().tolist(),extend='both',pad=.025,shrink=.82,
                 label=r'Median $\Delta\mu$ (mag)')
    save(fig,dest,'native_luminosity_residuals',rows,dict(gr=x,Mr=my,rz=zy,**dict(data)))


def sky(dest,p):
    with np.load(p/'density_cells.npz') as z:
        count=z['catalogue_counts'].sum(axis=0)[:,0]
        area=z['native_area_deg2'].sum(axis=0)
    l,b,pix=plotting_grid()
    a=np.divide(count,area,out=np.full_like(count,np.nan,dtype='f8'),where=area>=MIN_AREA)
    positive=a[np.isfinite(a)&(a>0)]
    norm=SymLogNorm(linthresh=np.quantile(positive,.05),linscale=.32,
                    vmin=0,vmax=np.ceil(np.quantile(positive,.995)))
    fig,axs=plt.subplots(1,3,figsize=(7.1,2.55),subplot_kw={'projection':'aitoff'},layout='constrained')
    rows=[]
    for i,(ax,title) in enumerate(zip(axs,('Owned PSFs','Full PDFs','Science core'))):
        z=ax.pcolormesh(l,b,np.ma.masked_invalid(a[i,pix]),cmap=COLORMAP,norm=norm,rasterized=True)
        ax.grid(True)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(title)
        rows.append(dict(component=title,n=int(count[i].sum()),density_area_deg2=area[area>=MIN_AREA].sum(),
                         display_n=int(count[i,area>=MIN_AREA].sum())))
    fig.colorbar(z,ax=axs.tolist(),shrink=.83,location='bottom',pad=.04,extend='max',
                 label=r'Sources per effective deg$^2$',
                 ticks=(0,200,500,1000,2000,5000,10000),format=plt.FormatStrFormatter('%g'))
    save(fig,dest,'f21_catalogue_density',rows)


def planes(dest,p):
    from final_catalogue_sky import XY
    with np.load(p/'atlas_bins.npz') as z:
        counts=z['xyz']
    pairs=('XY','XZ','YZ')
    fig,axs=plt.subplots(2,3,figsize=(7,4.3),layout='constrained')
    rows=[]
    for i,name in enumerate(('Full PDFs','Science core')):
        norm=LogNorm(1,max(2,counts[i].max()))
        for k,axes in enumerate(pairs):
            a=axs[i,k]
            cb=a.pcolormesh(XY,XY,np.ma.masked_equal(counts[i,k].T,0),norm=norm,
                            cmap='viridis',rasterized=True)
            a.set(xlabel=f'{axes[0]} (kpc)',ylabel=f'{axes[1]} (kpc)',aspect='equal')
            if k==0: a.set_title(name,loc='left')
            rows.append(dict(view=name,projection=axes,n=int(counts[i,k].sum())))
        fig.colorbar(cb,ax=axs[i,:].tolist(),pad=.01,label='Sources/bin')
    save(fig,dest,'f16_heliocentric_projections',rows,dict(edges=XY,counts=counts))


def main():
    require_compute()
    b=OUT/'benchmark'
    dest=OUT/'figures'
    dest.mkdir(exist_ok=True,parents=True)
    marks={'benchmark':b/'complete.json','density':ROOT/'data/v3_final/atlas/complete.json',
           'space':ROOT/'data/final_catalogue/atlas/complete.json'}
    for key,names in (('benchmark',('figure_stars.npz','observables.npz','spectral_pit.npz','signed_pit.npz')),
                      ('density',('density_cells.npz',)),('space',('atlas_bins.npz',))):
        p=marks[key]
        r=json.loads(p.read_text())
        for k in names: verified(p.with_name(k),r['outputs'][k])
    with np.load(b/'figure_stars.npz') as z:
        t={k:z[k] for k in z.files}
    assert (t['TYPE']=='PSF').all()
    with plt.rc_context(PAPER):
        reference(dest,t)
        occupied(dest,b)
        pdf_diagnostics(dest,b)
        physical(dest,t)
        sky(dest,marks['density'].parent)
        planes(dest,marks['space'].parent)
    save_json(dest/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),style=check_templates(),
        inputs={k:sha(v) for k,v in marks.items()},outputs={p.name:sha(p) for p in dest.iterdir() if p.name!='complete.json'}))


if __name__=='__main__':
    main()
