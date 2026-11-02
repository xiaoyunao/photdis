#!/usr/bin/env python3
"""Reproduce fixed held-field distances; density is a matched sample diagnostic."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.deps')]
import fitsio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
from photdis.stage1 import require_compute
from photdis.holdouts import load_reference_catalogs
from photdis.route import LEVEL, feature_matrix, route_rows
from photdis.route_policy import choose_policy
from cold_v4 import configured_field, sha
from cold_v4_policy_noise import frozen_models
from cold_v4_native_transfer import EXTRA
from giant_photo_baseline import SOURCE, clean_warm_spdist, estimate, summarize
OUT=Path('data/distance_comparison_v1');FIG=OUT/'figures'


def distance(mu):
    return 10**((np.asarray(mu)-10)/5)


def panel(axes,x,y,label,title):
    x,y=np.asarray(x,float),np.asarray(y,float)
    valid=np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0);n_rejected=int((~valid).sum())
    x,y=x[valid],y[valid]
    if not len(x):
        for ax in axes: ax.text(.5,.5,'No qualified matched distances',ha='center',transform=ax.transAxes)
        return {'n':0,'invalid_pairs':n_rejected}
    residual=y/x-1;med=float(np.median(residual));nmad=float(1.4826*np.median(abs(residual-med)))
    limits=[min(x.min(),y.min())*.85,max(x.max(),y.max())*1.18]
    ax=axes[0]
    if len(x)>=40:
        h=ax.hexbin(x,y,gridsize=45,mincnt=1,bins='log',xscale='log',yscale='log',cmap='viridis',linewidths=0)
        ax.figure.colorbar(h,ax=ax,pad=.02,label='Stars / hexagon (log scale)',shrink=.8)
    else:ax.scatter(x,y,s=14,alpha=.7)
    ax.plot(limits,limits,color='#d95f02',ls='--',lw=1.2,label='1:1')
    ax.set(xscale='log',yscale='log',xlim=limits,ylim=limits,ylabel='Photometric distance [kpc]',title=title)
    ax.legend(loc='lower right',fontsize=8)
    ax.text(.04,.96,f'N={len(x):,}\nmedian = {med:+.1%}\nNMAD = {nmad:.1%}',va='top',transform=ax.transAxes,fontsize=9,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
    ax=axes[1]
    if len(x)>=40:ax.hexbin(x,residual,gridsize=45,mincnt=1,bins='log',xscale='log',cmap='viridis',linewidths=0)
    else:ax.scatter(x,residual,s=14,alpha=.7)
    ax.axhline(0,color='#d95f02',ls='--',lw=1);ax.set(xscale='log',xlabel=label+' [kpc]',ylabel=r'$d_{phot}/d_{ref}-1$')
    for ax in axes:
        ax.grid(alpha=.12)
        ax.xaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,_:f'{value:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0].yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value,_:f'{value:g}'))
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    return {'n':len(x),'invalid_pairs':n_rejected,'relative_median':med,'relative_nmad':nmad,
            'fraction_abs_relative_gt_0p5':float(np.mean(abs(residual)>.5)),
            'reference_range_kpc':[float(x.min()),float(x.max())]}


def cold():
    route,forests=frozen_models();references=load_reference_catalogs("data/reference")
    policy=json.loads(Path('configs/cold_v4_deep_policy.json').read_text())
    # Native-transfer pipeline uses the registered policy member.
    policy=policy['chosen_policy']
    registrations=['configs/cold_v4_native_transfer_fields.json','configs/cold_v4_native_transfer_north_final.json']
    fields=sum([json.loads(Path(p).read_text())['independent_final_fields'] for p in registrations],[])
    fig,axes=plt.subplots(2,3,figsize=(16,9),gridspec_kw={'height_ratios':[3,1.7]},layout='constrained')
    result=[]
    for j,item in enumerate(fields):
        table,strict,_,meta=configured_field(item['field_config'],references)
        with fitsio.FITS(meta['path']) as f:
            rows=np.flatnonzero(f[1].read(columns=['anchor'])['anchor'])
            cols=list(dict.fromkeys([*EXTRA,'gaia_parallax_error']))
            extra=f[1].read(columns=cols,rows=rows)
        assert np.array_equal(extra['source_id'],table['source_id'])
        table.update({n:extra[n] for n in cols if n!='source_id'})
        level=choose_policy(table,route_rows(table,route),policy)
        selected=strict & (level>0) & (table['p_corr']>0) & (table['gaia_parallax_error']>0) & (table['p_corr']/table['gaia_parallax_error']>=20)
        mu=np.full(len(selected),np.nan)
        for tier,num in LEVEL.items():
            use=selected&(level==num)
            if use.any():mu[use]=table['g0'][use]-forests[tier].predict(feature_matrix(table,tier)[use])[:,2]
        gaia=1/table['p_corr'][selected];phot=distance(mu[selected])
        metric=panel(axes[:,j],gaia,phot,'Gaia inverse corrected parallax',item['name'].split('_dec')[0].replace('native_',''))
        metric.update(field=item['name'],strict_anchor_n=int(strict.sum()),selected_n=int(selected.sum()),source_sha256=sha(meta['path']))
        np.savez_compressed(OUT/f'cold_{j}.npz',source_id=table['source_id'][selected],gaia_kpc=gaia,phot_kpc=phot,route=level[selected])
        result.append(metric)
    fig.suptitle('Frozen cold-v4 | strict cold dwarfs, Gaia S/N >= 20, clean + WISE20 fallback\nNative held-out fields; conditional on cold-dwarf identity',fontsize=13)
    fig.savefig(FIG/'01_cold_phot_vs_gaia.png',dpi=170);plt.close(fig)
    return result


def warm():
    rows=fitsio.read(str(SOURCE));mask=clean_warm_spdist(rows);selected=rows[mask]
    role=np.char.strip(selected['role'].astype('U'));train=selected[role=='development'];test=selected[role=='reserved_validation']
    assert not set(train['source_id'])&set(test['source_id'])
    manifest=json.loads(Path('manifests/giant_photo_baseline_v1.json').read_text());best=manifest['chosen']
    values=estimate(test,train,best['bandwidth'],best['magnitude_floor']);metrics=summarize(values)
    for key in ('n','median_mg_residual','nmad_mg_residual'):
        assert np.isclose(metrics[key],manifest['reserved_summary'][key],atol=1e-6),key
    a=np.asarray(values,object);assert np.array_equal(a[:,0].astype(np.int64),test['source_id'])
    phot=distance(test['g0']-a[:,8].astype(float));support=a[:,13].astype(bool)
    fig,axes=plt.subplots(2,2,figsize=(12,9),gridspec_kw={'height_ratios':[3,1.7]},layout='constrained')
    results={}
    for j,(key,label) in enumerate([('spdist','SPDist v1.0'),('specdis','SPECDIS v2.1')]):
        take=test[key+'_valid'] & support
        results[key]=panel(axes[:,j],test[key+'_dist_kpc'][take],phot[take],label,label+' | supported reserved fields')
        results[key]['all_reserved']=int(len(test));results[key]['quality_valid']=int(test[key+'_valid'].sum());results[key]['supported_valid']=int(take.sum())
    fig.suptitle('Warm-giant grz photometric prototype (5000–7000 K)\nSPDist-trained teacher comparison; SPECDIS is a cross-catalog check, neither is geometric truth',fontsize=12)
    fig.savefig(FIG/'02_warm_phot_vs_spectral.png',dpi=170);plt.close(fig)
    # Include all reserved predictions explicitly, to expose support selection.
    fig,axes=plt.subplots(2,2,figsize=(12,9),gridspec_kw={'height_ratios':[3,1.7]},layout='constrained')
    for j,(key,label) in enumerate([('spdist','SPDist v1.0'),('specdis','SPECDIS v2.1')]):
        take=test[key+'_valid']
        results[key+'_all']=panel(axes[:,j],test[key+'_dist_kpc'][take],phot[take],label,label+' | all qualified reserved stars')
    fig.suptitle('Warm-giant prototype | all reserved stars, including unsupported colors\nDiagnostic only: spectral teachers share assumptions; no field-giant distance release',fontsize=12)
    fig.savefig(FIG/'03_warm_all_reserved.png',dpi=170);plt.close(fig)
    np.savez_compressed(OUT/'warm.npz',source_id=test['source_id'],phot_kpc=phot,spdist_kpc=test['spdist_dist_kpc'],specdis_kpc=test['specdis_dist_kpc'],supported=support,specdis_valid=test['specdis_valid'])
    results['source_sha256']=sha(SOURCE);results['frozen_model_manifest_sha256']=sha('manifests/giant_photo_baseline_v1.json');results['reproduced_summary']=metrics
    return results


def main():
    require_compute();FIG.mkdir(parents=True,exist_ok=True)
    if (OUT/'summary.json').exists():raise FileExistsError(OUT/'summary.json')
    result={'cold':cold(),'warm':warm(),'distance_unit':'kpc','uncertainty':'descriptive matched point-distance comparisons; no coverage or truth claim','script_sha256':sha(__file__)}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')

def render_cached():
    require_compute()
    summary=json.loads((OUT/'summary.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(16,9),gridspec_kw={'height_ratios':[3,1.7]},layout='constrained')
    for j,item in enumerate(summary['cold']):
        with np.load(OUT/f'cold_{j}.npz') as f:
            panel(axes[:,j],f['gaia_kpc'],f['phot_kpc'],'Gaia inverse corrected parallax',item['field'].split('_dec')[0].replace('native_',''))
    fig.suptitle('Frozen cold-v4 | strict cold dwarfs, Gaia S/N >= 20, clean + WISE20 fallback\nNative held-out fields; conditional on cold-dwarf identity',fontsize=13)
    fig.savefig(FIG/'01_cold_phot_vs_gaia.png',dpi=170);plt.close(fig)
    with np.load(OUT/'warm.npz') as f:t={k:f[k] for k in f.files}
    for supported,filename in [(True,'02_warm_phot_vs_spectral.png'),(False,'03_warm_all_reserved.png')]:
        fig,axes=plt.subplots(2,2,figsize=(12,9),gridspec_kw={'height_ratios':[3,1.7]},layout='constrained')
        for j,(key,label) in enumerate([('spdist','SPDist v1.0'),('specdis','SPECDIS v2.1')]):
            take=np.ones(len(t['source_id']),bool) if key=='spdist' else t['specdis_valid'].copy()
            if supported:take &= t['supported']
            panel(axes[:,j],t[key+'_kpc'][take],t['phot_kpc'][take],label,label+(' | supported reserved fields' if supported else ' | all qualified reserved stars'))
        fig.suptitle('Warm-giant grz photometric prototype (5000–7000 K)\nSPDist teacher comparison; SPECDIS cross-catalog check; neither is geometric truth',fontsize=12)
        fig.savefig(FIG/filename,dpi=170);plt.close(fig)
    (OUT/'render.json').write_text(json.dumps({'cached_only':True,'script_sha256':sha(__file__),
        'unchanged_scientific_summary_sha256':sha(OUT/'summary.json'),
        'figure_sha256':{str(p):sha(p) for p in sorted(FIG.glob('*.png'))}},indent=2)+'\n')


if __name__=='__main__':
    render_cached() if '--render-only' in sys.argv else main()

