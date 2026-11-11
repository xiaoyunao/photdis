#!/usr/bin/env python3
"""Prospective evidence and independent-scale figures; fixed English style."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from v2_1_near import ROOT,load,sha,save_json,csv_rows
from photdis.stage1 import require_compute
from photdis.v3.controls import optical_magnitudes
from photdis_figure_style import STYLE,BLUE,ORANGE,GREEN,GRAY
from v3_0_development_figures import clip_pair
D=ROOT/'data/v3_0'; OUT=ROOT/'figures/v3_0/validation'; TABLE=ROOT/'manifests/v3_0/figures/validation'
INPUTS={}


def verified(path, marker, key):
    r=json.loads(marker.read_text());assert sha(path)==r['outputs'][key]
    INPUTS[str(path.relative_to(ROOT))]=sha(path)
    return load(path)


def save(fig,name,rows,caption):
    OUT.mkdir(parents=True,exist_ok=True);TABLE.mkdir(parents=True,exist_ok=True)
    assert not any(any('\u4e00'<=c<='\u9fff' for c in x.get_text()) for x in fig.findobj(matplotlib.text.Text))
    fig.savefig(OUT/(name+'.png'),dpi=300);fig.savefig(OUT/(name+'.pdf'));plt.close(fig)
    csv_rows(TABLE/(name+'.csv'),rows)
    save_json(TABLE/(name+'.json'),dict(caption=caption,inputs=INPUTS.copy(),english_text_verified=True,png_dpi=300,
        outputs={str(p.relative_to(ROOT)):sha(p) for p in (OUT/(name+'.png'),OUT/(name+'.pdf'),TABLE/(name+'.csv'))}))
    print('REAL FIGURE',name,flush=True)


def main():
    require_compute();plt.rcParams.update(STYLE)
    examples={}
    for name,title in [('new_sky','Previously unused sky'),('independent_scale','NGC 288 photometric field')]:
        f=D/'evaluation_results'/f'final_{name}_multiband_fit_0'
        t=verified(f/'per_star.npz',f/'complete.json','per_star.npz');examples[name]=t
        phot=load(D/'validation'/name/'photometry.npz');np.testing.assert_array_equal(phot['identity'][t['rows']],t['identity'])
        fig,axes=plt.subplots(1,2,figsize=(11,4.5));rows=[]
        for ax,use,label in [(axes[0],np.isfinite(t['gaia_pit']),'All qualified Gaia'),
                             (axes[1],np.isfinite(t['gaia_pit'])&(t['qnear']>.95),'Photometric P(d < 10 kpc) > 0.95')]:
            weights=phot['design_weight'][t['rows'][use]];edges=np.linspace(0,1,21)
            h,_=np.histogram(t['gaia_pit'][use],edges,weights=weights);h=h/h.sum() if h.sum() else h
            ax.stairs(h*100,edges,color=BLUE,fill=True,alpha=.45);ax.axhline(5,ls='--',color='black',lw=1)
            ax.set(xlabel='Gaia posterior predictive PIT',ylabel='Weighted fraction (%)',title=label)
            ax.text(.03,.96,f'N = {use.sum():,}',transform=ax.transAxes,va='top',fontsize=12)
            for i,v in enumerate(h):rows.append(dict(selection=label,lower=edges[i],upper=edges[i+1],fraction=v,n=int(use.sum())))
        fig.suptitle(title+': signed Gaia observations',fontsize=17);fig.tight_layout(rect=(0,0,1,.94))
        save(fig,name+'_gaia_pit',rows,'Frozen pure-photometry reference prior. Negative parallaxes retained. Weighted counts do not increase independent validation sample size; dashed line is nominal uniform PIT. These cohorts have different selection, not a before/after performance comparison.')
    t=examples['new_sky'];use=t['gaia_precise_valid'];x=10**((t['gaia_precise_mu'][use]-10)/5);y=10**((t['mu'][use,2]-10)/5)
    keep,audit=clip_pair(x,y);x,y=x[keep],y[keep];res=100*(y/x-1)
    fig=plt.figure(figsize=(10,7));gs=fig.add_gridspec(2,3,width_ratios=[4,.16,2],height_ratios=[4,1],wspace=.38,hspace=.06)
    ax=fig.add_subplot(gs[0,0]);ar=fig.add_subplot(gs[1,0],sharex=ax);cb=fig.add_subplot(gs[:,1]);hist=fig.add_subplot(gs[:,2])
    limit=max(x.max(),y.max())*1.03;edges=np.linspace(0,limit,38);redges=np.linspace(-100,100,41)
    h,_,_=np.histogram2d(x,y,(edges,edges));hr,_,_=np.histogram2d(x,res,(edges,redges));norm=LogNorm(1,max(h.max(),hr.max(),2))
    im=ax.pcolormesh(edges,edges,np.ma.masked_equal(h.T,0),norm=norm,cmap='viridis',rasterized=True)
    ar.pcolormesh(edges,redges,np.ma.masked_equal(hr.T,0),norm=norm,cmap='viridis',rasterized=True)
    ax.plot([0,limit],[0,limit],'--',color='black',lw=1);ar.axhline(0,color='black',ls='--',lw=1)
    ax.set(xlim=(0,limit),ylim=(0,limit),ylabel='Photometric distance (kpc)',title='New sky: Gaia S/N > 10')
    ax.text(.04,.94,f'N = {len(x)} / {audit["valid"]}',transform=ax.transAxes,fontsize=12)
    ar.set(xlabel='Gaia reference distance (kpc)',ylabel='Residual (%)',ylim=(-100,100),yticks=[-100,0,100]);plt.setp(ax.get_xticklabels(),visible=False)
    colorbar=fig.colorbar(im,cax=cb,label='Objects per bin');colorbar.ax.yaxis.label.set_size(12)
    hh,_=np.histogram(res,redges);hist.stairs(hh,redges,color=BLUE,fill=True,alpha=.4)
    hist.set(xlabel='Distance residual (%)',ylabel='Objects',xlim=(-100,100));hist.axvline(0,color='black',ls='--',lw=1)
    hist.yaxis.tick_right();hist.yaxis.set_label_position('right');hist.xaxis.label.set_size(13);hist.yaxis.label.set_size(14)
    ar.yaxis.label.set_size(13);ar.xaxis.label.set_size(15);ax.yaxis.label.set_size(15)
    fig.subplots_adjust(left=.09,right=.91,bottom=.11,top=.92)
    rows=[dict(panel='main',x=edges[i],y=edges[j],count=h[i,j]) for i in range(len(edges)-1) for j in range(len(edges)-1)]
    rows += [dict(panel='residual',x=edges[i],y=redges[j],count=hr[i,j]) for i in range(len(edges)-1) for j in range(len(redges)-1)]
    rows += [dict(panel='histogram',x=redges[i],y=0,count=hh[i]) for i in range(len(hh))]
    save(fig,'new_sky_distance',rows,'Display-only independent x/y linear-distance 3sigma clip, shared between main/residual/histogram within this Gaia comparison. Scientific metrics use all247 qualified stars, no clipping. '+json.dumps(audit))
    f=D/'scale_example/reference';r=verified(f/'per_star.npz',f/'complete.json','per_star.npz');meta=json.loads((f/'complete.json').read_text())
    dist=list(csv.DictReader((f/'distance_distribution.csv').open()));INPUTS[str((f/'distance_distribution.csv').relative_to(ROOT))]=sha(f/'distance_distribution.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    labels=[('all','All photometric objects',GRAY),('members','Gaia-selected members',BLUE),('nonmembers','Other photometric objects',ORANGE)]
    for group,label,color in labels:
        rows=[x for x in dist if x['group']==group];mu=np.array([float(x['mu']) for x in rows]);v=np.array([float(x['expected_count']) for x in rows]);n=int(rows[0]['n'])
        axes[0].plot(mu,v/max(n,1)/.05,color=color,label=label,lw=1.5)
    d=meta['distance_kpc'];m=10+5*np.log10(d);ml=10+5*np.log10(d-meta['distance_error_lower']);mh=10+5*np.log10(d+meta['distance_error_upper'])
    axes[0].axvspan(ml,mh,color=GREEN,alpha=.25);axes[0].axvline(m,color=GREEN,ls='--')
    axes[0].set(xlabel='Distance modulus (mag)',ylabel='Mean posterior density',xlim=(7,20),title='Photometric distance distributions');axes[0].legend(fontsize=10,frameon=False)
    member=r['selected_member'];fraction=100*(10**((r['mu'][member,2]-10)/5)/d-1);axes[1].hist(fraction,bins=np.arange(-100,201,5),histtype='stepfilled',color=BLUE,alpha=.4)
    axes[1].axvline(0,color='black',ls='--');axes[1].set(xlabel='Distance residual to dynamical scale (%)',ylabel='Members',title=f'Frozen membership: N = {member.sum():,}',xlim=(-100,200))
    fig.suptitle('NGC 288: marginalization and an independent distance scale',fontsize=17);fig.tight_layout(rect=(0,0,1,.94))
    save(fig,'ngc288_distance_distribution',dist,'Photometric sample frozen before Gaia-conditioned membership. Green band is one shared dynamical-scale uncertainty, not reduced by sqrt(N). Stacked PDFs show expectations, not calibrated intrinsic counts; foreground stars are not assigned cluster truth.')
    phot=load(D/'validation/independent_scale/photometry.npz');mag,_=optical_magnitudes(phot)
    np.testing.assert_array_equal(phot['identity'],r['identity']);color=mag[:,0]-mag[:,1];Mr=mag[:,1]-m;relative=100*(10**((r['mu'][:,2]-10)/5)/d-1)
    fig,axes=plt.subplots(1,2,figsize=(11,5));norm=TwoSlopeNorm(vmin=-100,vcenter=0,vmax=100)
    sc=axes[0].scatter(color[member],Mr[member],c=relative[member],norm=norm,cmap='coolwarm',s=6,rasterized=True)
    axes[0].set(xlabel='Dereddened g - r (mag)',ylabel='Reference M_r (mag)',title='Selected members');axes[0].invert_yaxis()
    axes[1].scatter(r['RA'][~member],r['DEC'][~member],s=1,color=GRAY,alpha=.2,rasterized=True)
    axes[1].scatter(r['RA'][member],r['DEC'][member],c=relative[member],norm=norm,cmap='coolwarm',s=6,rasterized=True)
    axes[1].set(xlabel='Right ascension (deg)',ylabel='Declination (deg)',title='Photometric-first field');axes[1].invert_xaxis()
    fig.subplots_adjust(left=.08,right=.86,bottom=.14,top=.9,wspace=.35);fig.colorbar(sc,cax=fig.add_axes([.9,.16,.02,.7]),label='Relative distance residual (%)')
    points=[dict(gr=float(color[i]),Mr_scale=float(Mr[i]),RA=float(r['RA'][i]),DEC=float(r['DEC'][i]),relative_percent=float(relative[i])) for i in np.flatnonzero(member)]
    save(fig,'ngc288_cmd_sky',points,'Reference CMD uses the independently specified common system distance only for evaluation, never model input. Color range saturates at +/-100%; full residuals remain in CSV. Gaia membership is not an independent class label.')
    save_json(TABLE/'complete.json',dict(figures=5,inputs=INPUTS,outputs={x.name:sha(x) for x in TABLE.glob('*.json') if x.name not in ('complete.json','validation.json')}))


if __name__=='__main__':main()
