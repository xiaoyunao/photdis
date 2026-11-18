#!/usr/bin/env python3
"""Training luminosity distributions and representative full distance PDFs."""
from matplotlib.colors import LogNorm
from photdis.stage1 import require_compute
from sv18_common import *
from paper_v1_benchmark import read_scan
from final_catalogue_benchmark import load_parent


def main():
    require_compute()
    inputs={};d=out('figures')
    gr_edges=np.linspace(-.1,1.5,65); mr_edges=np.linspace(-3,11,71)
    hist=[];meta=[]
    for side in ('south','north'):
        t=read(DEST/'benchmark',(side+'_training_sample.npz','TYPE','ls_id','m','y'),inputs)
        assert (t['TYPE']=='PSF').all()
        gr=t['m'][:,0]-t['m'][:,1];mr=t['y']-gr
        h=np.histogram2d(gr,mr,(gr_edges,mr_edges))[0];hist.append(h)
        meta.append(dict(native=side,kind='training_CMD',parent=len(gr),inside_plot=int(h.sum()),
            outside_plot=int(len(gr)-h.sum())))
    _,detail,_=read_scan(inputs)
    t,_=load_parent();ix=index_join(t['ls_id'],detail['ls_id']);use=ix>=0
    t={k:v[use] for k,v in t.items()};detail={k:v[ix[use]] for k,v in detail.items()}
    colors=detail['g0']-detail['r0']
    dwarfs=t['class_label']==0
    feh_good=(t['rvs_feh_err']>0)&(t['rvs_feh_err']<=.3)
    groups=(('Ordinary dwarf',dwarfs&feh_good&(t['rvs_feh']>=-1.5),BLUE),
       ('Metal-poor dwarf',dwarfs&feh_good&(t['rvs_feh']<-1.5),ORANGE),
       ('Intermediate gravity',(t['rvs_logg_err']>0)&(t['rvs_logg_err']<=.3)&
          (t['rvs_logg']>=3.2)&(t['rvs_logg']<4.2),GREEN))
    arrays=dict(gr_edges=gr_edges,mr_edges=mr_edges,south_training=hist[0],north_training=hist[1])
    ids=[];curve_rows=[]
    with plt.rc_context(PAPER):
        f,axes=plt.subplots(2,2,figsize=(7.1,5.3),layout='constrained')
        for j,(side,title) in enumerate((('south','DECam'),('north','BASS/MzLS'))):
            h=hist[j]
            im=axes[0,j].pcolormesh(gr_edges,mr_edges,np.ma.masked_equal(h.T,0),
                 cmap='viridis',norm=LogNorm(1,max(x.max() for x in hist)),rasterized=True)
            axes[0,j].set(title=title+' training',xlabel=r'$(g-r)_0$',ylabel=r'$M_r$ (mag)',
                          xlim=gr_edges[[0,-1]],ylim=mr_edges[[-1,0]])
            for name,group,color in groups:
                good=group&((t['region']=='north')==(side=='north'))&(colors>=.25)&(colors<.55)
                good&=(t['r']>=18)&(t['r']<20)
                if not good.any():continue
                indices=np.flatnonzero(good)
                rank=((colors[indices]-.4)/.1)**2+((t['r'][indices]-19)/.5)**2
                index=indices[np.argmin(rank)]
                e=detail['qrf_mu_edges'][index];w=detail['qrf_mu_mass'][index]
                density=w/np.diff(e);mid=(e[1:]+e[:-1])/2
                reference=t['specdis_kpc'][index]
                if not np.isfinite(reference):reference=t['spdist_kpc'][index]
                axes[1,j].stairs(density,e,color=color,lw=1,label=name)
                if np.isfinite(reference):
                    axes[1,j].axvline(10+5*np.log10(reference),color=color,lw=.7,ls=':')
                ids.append(dict(native=side,population=name,TYPE='PSF',ls_id=int(t['ls_id'][index])))
                outside=np.sum(w*np.clip((8-e[:-1])/np.diff(e),0,1))+np.sum(w*(1-np.clip((18-e[:-1])/np.diff(e),0,1)))
                meta.append(dict(native=side,kind=name,parent=1,inside_plot=None,outside_plot_mass=outside))
                curve_rows.extend(dict(native=side,population=name,mu_lo=a,mu_hi=b,pdf_mass=c,
                     reference_mu=10+5*np.log10(reference) if np.isfinite(reference) else None)
                     for a,b,c in zip(e[:-1],e[1:],w))
            axes[1,j].set(xlabel=r'Distance modulus $\mu$ (mag)',ylabel=r'PDF density (mag$^{-1}$)',xlim=(8,18))
        axes[1,0].legend(fontsize=6,frameon=False)
        f.colorbar(im,ax=axes[0,:].tolist(),fraction=.025,label='Training stars / bin')
        figure(d,'training_and_branch_pdfs',f,meta,arrays)
    csv_rows(d/'representative_curves.csv',curve_rows)
    csv_rows(d/'example_sources.csv',ids)
    complete(d,inputs,selection='same blue color and 18–20 mag window, minimum fixed color/magnitude rank per native external class')


if __name__=='__main__':
    main()
