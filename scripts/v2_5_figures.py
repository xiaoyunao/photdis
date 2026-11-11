#!/usr/bin/env python3
"""English-only v2.5 scientific evidence from actual compact tables."""
import sys
import numpy as np
import matplotlib.pyplot as plt
import v2_2_figures as f
from photdis_figure_style import STYLE,BLUE,ORANGE,GREEN,GRAY
f.TABLES=f.ROOT/'manifests/v2_5';f.FIG=f.ROOT/'figures/v2_5'
read,sel,val=f.read,f.sel,f.val


def save(fig,name,sources,caption):
    import csv,json,hashlib,re
    from matplotlib.text import Text
    f.FIG.mkdir(parents=True,exist_ok=True);dest=f.TABLES/'figures';dest.mkdir(exist_ok=True)
    assert all(not re.search(r'[\u3400-\u9fff]',x.get_text()) for x in fig.findobj(match=Text))
    fig.savefig(f.FIG/(name+'.png'),dpi=300,bbox_inches='tight');fig.savefig(f.FIG/(name+'.pdf'),dpi=300,bbox_inches='tight');plt.close(fig)
    rows=[dict(table_source=source,**r) for source in sources for r in read(source)]
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with open(dest/(name+'.csv'),'w') as handle:
        writer=csv.DictWriter(handle,fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    (dest/(name+'.json')).write_text(json.dumps(dict(name=name,caption=caption,
        selection='Native PSF; empirical diagnostic or explicit simulated selection; no independent final-validation claim',
        sources={s:hashlib.sha256((f.TABLES/(s+'.csv')).read_bytes()).hexdigest() for s in sources}),indent=2)+'\n')


def main():
    plt.rcParams.update(STYLE)
    r=read('red_migration');fig,ax=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for a,tail in zip(ax,['low','high']):
        for j,(method,color,label) in enumerate([('historical_raw',GRAY,'Historical: raw PIT'),('historical_conditional',ORANGE,'Historical: conditional PIT'),('application_raw',BLUE,'Application: raw PIT')]):
            q=[sel(r,group=g,method=method,weighting='matched_identity',tail=tail,cluster='reused_control')[0] for g in ['all','red','blue']]
            x=np.arange(3)+(j-1)*.15;y=np.array([val(t,'value') for t in q])*100
            a.plot(x,y,'o-',color=color,label=label);a.vlines(x,[100*val(t,'low') for t in q],[100*val(t,'high') for t in q],color=color)
        a.axhline(5,color='black',ls='--',lw=1);a.set(xticks=range(3),xticklabels=['All matched','Red: r-z >= 0.8','Bluer'],ylabel=tail.title()+' PIT tail (%)')
    ax[0].legend(fontsize=9);fig.suptitle('Outcome-blind controls · different historical selection remains explicit')
    save(fig,'01_red_migration',['red_migration','red_matches'],'Match full711historical objects without PIT. With-replacement controls exclude fitting identities and known nonstellar risks. Bars resample reused control IDs. Raw and historical conditional PIT have different selection meanings; weighted and field intervals remain in CSV.')
    r=read('metal_gaia_pairs');fig,ax=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    for a,metric in zip(ax,['log_score','low_tail','high_tail']):
        for j,(model,color) in enumerate([('control',BLUE),('candidate',ORANGE)]):
            q=[sel(r,model=model,source=s,role='neither_fitted',metric=metric,cluster='control')[0] for s in ['all','specdis','spdist','both_VAC']]
            x=np.arange(4)+(j-.5)*.1;a.plot(x,[val(t,'mean') for t in q],'o-',color=color,label='QRF' if j==0 else 'Corrected XD')
            a.vlines(x,[val(t,'low') for t in q],[val(t,'high') for t in q],color=color)
        a.axhline(0,color=GRAY,ls='--');a.set(xticks=range(4),xticklabels=['All Gaia','SpecDis','SPDist','Both VAC'],ylabel='Poor minus ordinary '+metric.replace('_',' '));a.tick_params(axis='x',labelsize=10)
    ax[0].legend(fontsize=10);fig.suptitle('Original metal pairs · full signed Gaia predictive distributions')
    save(fig,'02_metal_signed_gaia',['metal_gaia_pairs','metal_gaia_availability'],'Both sides eligible for paired differences; no SNR cut or weak-parallax inversion. NativePSF, clean references, neither identity fitted. Original2055denominator and one-sided availability in CSV; bars resample reused ordinary IDs. Source selection and shared teachers limit causality.')
    r=read('band_gain_by_class');fig,ax=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    for a,c,title in zip(ax,range(3),['Dwarf','Evolved reference','Other']):
        for j,(method,color) in enumerate([('fixed_forest',ORANGE),('frozen_partitions_same_labels',BLUE)]):
            q=sel(r,group=c,method=method);x=np.arange(3)+(j-.5)*.12;a.plot(x,[val(t,'mean') for t in q],'o-',color=color,label='Color probe' if j==0 else 'Distance partition');a.vlines(x,[val(t,'low') for t in q],[val(t,'high') for t in q],color=color)
        a.axhline(0,color=GRAY,ls='--');a.set(xticks=range(3),xticklabels=['+ i','+ W1','+ W2'],title=title,ylabel='Paired log-score gain (nats)')
    ax[0].legend(fontsize=10);fig.suptitle('Same matched identities · class-specific sky-cluster intervals')
    save(fig,'03_band_gains_by_class',['band_gain_by_class'],'Group-specific uncertainty, not borrowed overall intervals. Fixed probes and partitions have different fitting support. Matched bands have no faint evolved references; no physical information-bound claim.')
    if '--diagnostics-only' in sys.argv:return
    r=read('recovery_summary');designs=['reference_like','giant_only','poor_only','distance_only','ordinary_dominated','giant_rich','metal_poor_rich','balanced']
    fig,ax=plt.subplots(2,1,figsize=(15,9),constrained_layout=True)
    for a,regime,title in zip(ax,['0','1'],['Fixed photometric entrance','Selection complement: stress only']):
        for j,(method,color) in enumerate([('point',GRAY),('pdf',BLUE),('population',ORANGE)]):
            q=[sel(r,design=d,regime=regime,method=method)[0] for d in designs];x=np.arange(8)+(j-1)*.23
            a.bar(x,[100*val(t,'mean_L1') for t in q],width=.22,color=color,label={'point':'QRF point','pdf':'QRF PDF','population':'Population posterior'}[method]);a.errorbar(x,[100*val(t,'mean_L1') for t in q],yerr=[100*val(t,'sd_L1') for t in q],fmt='none',color='black',lw=.7)
        a.set(xticks=range(8),xticklabels=['Reference','Giant only','Poor only','Distance only','Ordinary','Giant-rich','Poor-rich','Balanced'],ylabel='Distance-fraction L1 error (%)',title=title);a.tick_params(axis='x',labelsize=10)
    ax[0].legend(ncol=3,fontsize=11);fig.suptitle('Independent fresh-noise catalogs · 12 realizations per design')
    save(fig,'04_population_recovery',['recovery_summary','recovery_beyond50'],'Same128objects/catalog for all three methods. L1 includes six distance bins, including QRF mass beyond50kpc; model truth has five fixed nodes. Bars across independent catalogs, errorbars their SD. Empirical templates/teachers and noise law are not real-sky truth.')
    r=read('recovery_distance');fig,ax=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
    for a,design in zip(ax.ravel(),designs[:4]):
        q=sel(r,design=design,regime=0);first=sel(q,catalog=q[0]['catalog']);x=np.arange(5)
        a.plot(x,[val(t,'truth') for t in first],'k--',label='Design selected fraction')
        for method,color in [('point',GRAY),('pdf',BLUE),('population',ORANGE)]:
            a.plot(x,[np.mean([val(t,method) for t in q if t['bin']==str(b)]) for b in range(5)],'o-',color=color,label=method.title())
        a.fill_between(x,[np.mean([val(t,'low') for t in q if t['bin']==str(b)]) for b in range(5)],[np.mean([val(t,'high') for t in q if t['bin']==str(b)]) for b in range(5)],alpha=.18,color=ORANGE,label='Mean parameter interval')
        a.set(xticks=x,xticklabels=['0–2','2–5','5–10','10–20','20–50'],xlabel='Distance (kpc)',ylabel='Selected population fraction',title=design.replace('_',' '))
    ax[0,0].legend(fontsize=9);fig.suptitle('One-factor recovery · independent empirical SED templates')
    save(fig,'05_recovery_distance_shapes',['recovery_distance'],'Means across12independent catalogs; shaded region is mean parameter-interval endpoints, NOT an interval on the average curve. True selected distribution from separate test-library forward Monte Carlo; no response fitting on test identities.')
    r=read('recovery_summary');di=read('identifiability');fig,ax=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
    for regime,color,label in [('0',BLUE,'Photometric entrance'),('1',ORANGE,'Complement stress')]:
        coverage=[];cor=[];grid=[]
        for d in designs:
            q=sel(r,design=d,regime=regime,method='parameter');coverage.append(np.mean([val(t,'coverage') for t in q])*100)
            q=sel(di,design=d,regime=regime);cor.append(np.mean([val(t,'max_correlation') for t in q]));grid.append(max(val(t,'grid_max_delta') for t in q))
        for a,y in zip(ax,[coverage,cor,grid]):a.plot(range(8),y,'o-',color=color,label=label)
    for a in ax:a.set(xticks=range(8),xticklabels=['Ref','Giant','Poor','Dist','Ord','G-rich','P-rich','Bal']);a.tick_params(axis='x',labelsize=9)
    ax[0].axhline(95,color=GRAY,ls='--');ax[0].set(ylabel='Mean parameter coverage (%)',ylim=(-2,102));ax[1].set_ylabel('Maximum |parameter correlation|');ax[2].set_ylabel('Max fraction change: doubled grid');ax[0].legend(fontsize=9)
    fig.suptitle('Coverage and identifiability · numerical limitations are retained')
    save(fig,'06_coverage_identifiability',['recovery_summary','identifiability'],'12independent catalogs per design; Wilson intervals for each distance bin in CSV. Parameter intervals and replicate-count intervals are separate. Boundary mass, quadrature ESS and grid convergence retained; high correlation or narrow wrong intervals prohibit adoption.')
    r=read('uncertainty_sources');fig,ax=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    for a,regime in zip(ax,['0','1']):
        for variant,color in [('teacher_minus',BLUE),('teacher_plus',ORANGE),('labeled_only',GREEN)]:
            q=sel(r,regime=regime,variant=variant);a.plot(range(5),[val(t,'estimate') for t in q],'o-',color=color,label=variant.replace('_',' '))
        q=[x for x in r if x['regime']==regime and x['variant'].startswith('template_bootstrap')]
        boot=np.array([[val(t,'estimate') for t in q if t['bin']==str(b)] for b in range(5)])
        a.fill_between(range(5),np.quantile(boot,.025,axis=1),np.quantile(boot,.975,axis=1),color=GRAY,alpha=.25,label='16 template resamples')
        a.set(xticks=range(5),xticklabels=['0–2','2–5','5–10','10–20','20–50'],xlabel='Distance (kpc)',ylabel='Population posterior mean fraction',title='Photometric entrance' if regime=='0' else 'Complement stress')
    ax[0].legend(fontsize=9);fig.suptitle('Separate template, missing-label and teacher-scale sensitivities')
    save(fig,'07_uncertainty_sources',['uncertainty_sources'],'First reference-like catalog only;16identity-level Bayesian bootstrap responses, coherent±1teacher-sigma calibration-template shifts and labeled-only response. Sensitivity envelopes are not combined calibrated uncertainty.')

    r=read('group_comparison');fig,ax=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    channels=['dwarf','evolved','other','unlabeled_reference']
    for a,design in zip(ax,['reference_like','giant_rich','metal_poor_rich']):
        q=sel(r,design=design,regime=0)
        for j,(method,color) in enumerate([('truth','GRAY'),('QRF_reference','BLUE'),('population','ORANGE')]):
            y=[np.nanmean([val(t,method) for t in q if t['channel']==c]) if c!='unlabeled_reference' or method=='QRF_reference' else np.nan for c in channels]
            a.bar(np.arange(4)+(j-1)*.24,y,width=.23,color={'GRAY':GRAY,'BLUE':BLUE,'ORANGE':ORANGE}[color],label=method.replace('_',' '))
        a.set(xticks=range(4),xticklabels=['Dwarf','Evolved','Other','No label'],ylabel='Selected population fraction',title=design.replace('_',' '));a.tick_params(axis='x',labelsize=10)
    ax[0].legend(fontsize=9);fig.suptitle('Composition recovery · QRF reference contributions are not physical labels')
    save(fig,'08_population_composition',['group_comparison'],'Mean over12catalogs in the fixed photometric entrance. New population model marginalizes missing template labels; QRF retains its unlabeled-reference channel separately. No-label is not a true physical population and has no injected truth bar.')


if __name__=='__main__':main()
