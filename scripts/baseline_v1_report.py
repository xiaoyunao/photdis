#!/usr/bin/env python3
"""One global scorecard; figures expose failures as well as improvements."""
from baseline_v1_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from photdis.stage1 import require_compute

FIG=OUT/'figures'
def save(fig,name):
    fig.tight_layout();fig.savefig(FIG/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)

def quality_histogram(width,weight,disagreement,error,supported):
    """Bins preserve the registered <=, >=, <=, < threshold conventions."""
    valid=supported&np.isfinite(width+weight+disagreement+error)
    bins=[np.searchsorted(np.asarray(c,dtype=x.dtype),x[valid],side=side) for c,x,side in
          [([1,2,4,8],width,'left'),([.5,.7,.9],weight,'right'),
           ([.25,.5,1],disagreement,'left'),([.03,.06,.12,.25],error,'right')]]
    return np.bincount(np.ravel_multi_index(bins,(5,4,4,5)),minlength=400).reshape(5,4,4,5)

def histogram_count(hist,cuts):
    i=[1,2,4,8].index(cuts['width95']);j=[.5,.7,.9].index(cuts['weight'])
    k=[.25,.5,1].index(cuts['disagreement']);l=[.03,.06,.12,.25].index(cuts['error'])
    return int(hist[:i+1,j+1:,:k+1,:l+1].sum())

def quality_diagnostics(t,p,candidates):
    """Describe fixed cuts on cal/test and the parent; never change selection."""
    import fitsio
    from baseline_v1_select import quality_mask
    hist=np.zeros((5,4,4,5),dtype=np.int64);nparent=0
    for field in range(13):
        with fitsio.FITS(str(OUT/f'catalog_{field:02}.fits')) as f:
            n=f[1].get_nrows();nparent+=n
            for start in range(0,n,100000):
                a=f[1].read(rows=np.arange(start,min(start+100000,n)),
                    columns=['mu','branch_weights','disagreement','optical_error','supported'])
                hist+=quality_histogram(a['mu'][:,4]-a['mu'][:,0],np.max(a['branch_weights'][:,:3],axis=1),
                                        a['disagreement'],a['optical_error'],a['supported'])
    result=[];rows=p['rows'];truth=truth_mu(t,'chosen')[rows]
    for candidate in candidates:
        use=quality_mask(t,rows,p,candidate['cuts']);record=dict(candidate)
        record['parent_count']=histogram_count(hist,candidate['cuts'])
        record['parent_fraction']=record['parent_count']/nparent
        for split,name in [(1,'cal'),(2,'test')]:
            mask=use&(t['split'][rows]==split)
            record[name]={label:dict(raw=metrics(p['mu'][m],truth[m]),
                                    calibrated=metrics(p['mu_cal'][m],truth[m]))
                for label,m in [('all',mask)]+[(f'class{k}',mask&(t['class_label'][rows]==k)) for k in range(3)]}
        result.append(record)
    return dict(parent_n=nparent,candidates=result,
        note='Fixed predeclared cuts; no selection or refit after test/parent inspection. Parent counts use stored float32 summaries.')

def main():
    global FIG
    preview='--experiments-only' in sys.argv
    if preview:FIG=OUT/'preview_figures'
    require_compute();FIG.mkdir(exist_ok=True);t=load(OLD/'references.npz')
    numer=json.load(open(OUT/'numerical.json'));selection=json.load(open(OUT/'selection.json'))
    model_scores={n:json.load(open(OUT/(n+'_scores.json'))) for n in MODELS}
    branch_audit=json.load(open(OUT/'v0_branch_audit.json'));depth=json.load(open(OUT/'depth_result.json'))
    result=dict(numerical=numer,selection=selection,model_scores=model_scores,
        v0_branch_audit=branch_audit,depth=depth,depth_input=json.load(open(OUT/'depth_input.json')),
        recommended=json.load(open(OUT/'recommended_scores.json')),
        training={n:json.load(open(OUT/(n+'_train.json'))) for n in MODELS})
    sample=load(OUT/'sample.npz');repeat={}
    for name in 'ABC':
        c=load(OUT/f'convergence_{name}.npz');repeat[name]={}
        delta=np.abs(c['sobol_128_0_mu'][:,[0,2,4]]-c['sobol_128_1_mu'][:,[0,2,4]])
        for layer,mask in [('low',sample['optical_error'][c['rows']]<.03),
                           ('high',sample['optical_error'][c['rows']]>=.03)]:
            repeat[name][layer]=dict(n=int(mask.sum()),p50_abs_mag=np.median(delta[mask],axis=0).tolist(),
                p95_abs_mag=np.quantile(delta[mask],.95,axis=0).tolist(),
                ambiguity_flip=float(np.mean(c['sobol_128_0_ambiguous'][mask]!=c['sobol_128_1_ambiguous'][mask])))
    result['independent_sobol128_repeat']=repeat
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    analytic=numer['analytic'];keys=list(analytic)
    ax[0].bar(np.arange(len(keys)),[analytic[k]['mean_probability95'] for k in keys]);ax[0].axhline(.95,c='r',ls='--')
    ax[0].set(xticks=np.arange(len(keys)),xticklabels=keys,ylim=(.7,1),ylabel='Analytic probability within numerical 95% interval',title='Constant-luminosity counterexample')
    ax[0].tick_params(axis='x',rotation=25)
    for layer in ('low','high'):
        keys=['iid_8_0','iid_32_0','iid_128_0','sobol_32_0','sobol_128_0']
        vals=[max(numer['convergence'][n]['comparisons'][k][layer]['p95_abs_mag'][0::2]) for k in keys for n in ['A']]
        ax[1].plot(np.arange(len(keys)),vals,'o-',label=layer+' optical error')
    ax[1].set(xticks=np.arange(len(keys)),xticklabels=[k[:-2] for k in keys],ylabel='A: 95th percentile endpoint difference [mag]',title='Versus independent 512-point integration');ax[1].tick_params(axis='x',rotation=25);ax[1].legend()
    save(fig,'01_numerical_convergence')

    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,cl in zip(axes,('class_dwarf','class_evolved','class_transition')):
        for variant in ('wide','natural','balanced','oracle'):
            y=[model_scores[n]['variants'][variant]['test']['chosen'][cl].get('tail_1',np.nan) for n in 'ABC']
            ax.plot(range(3),y,'o-',label=variant)
        ax.set(xticks=range(3),xticklabels=list('ABC'),ylim=(0,1),ylabel='Fraction |delta mu| > 1 mag',title=cl+' (available samples)')
    axes[0].legend();save(fig,'02_population_branches')

    same={};fig,axes=plt.subplots(1,2,figsize=(11,4))
    for tag,names in [('W1',['A_on_B','B']),('W1W2',['A_on_C','B_on_C','C'])]:
        same[tag]={}
        for name in names:
            same[tag][name]={v:model_scores[name]['variants'][v]['test'] for v in ('wide','natural','balanced','oracle')}
    for ax,source in zip(axes,('Gaia','specdis')):
        for variant in ('wide','natural','balanced','oracle'):
            vals=[model_scores[n]['variants'][variant]['test'][source]['all']['nmad_mag'] for n in ['A_on_C','B_on_C','C']]
            ax.plot(range(3),vals,'o-',label=variant)
        ax.set(xticks=range(3),xticklabels=['grz','grzW1','grzW1W2'],ylabel='Distance-modulus NMAD [mag]',title=source+': identical training/test identities')
    axes[0].legend();save(fig,'03_same_star_information');result['same_star']=same

    p=load(OUT/'recommended_evaluation.npz')
    if preview:
        candidates=selection['quality_candidates'];parents=(False,)
    else:
        quality=quality_diagnostics(t,p,selection['quality_candidates']);result['quality_diagnostics']=quality
        candidates=quality['candidates'];parents=(False,True)
    ncal=model_scores['A']['variants']['wide']['cal']['chosen']['all']['n']
    fig,axes=plt.subplots(1,len(parents),figsize=(6*len(parents),4),squeeze=False)
    for ax,parent in zip(axes[0],parents):
        for eligible,color in [(False,'gray'),(True,'tab:green')]:
            use=[c for c in candidates if c['eligible']==eligible and min(c['class_counts'])>=30]
            ax.scatter([c['parent_fraction'] if parent else c['n']/ncal for c in use],
                       [max(c['class_tail']) for c in use],s=15,alpha=.6,c=color,label='passes guards' if eligible else 'fails guards')
        ax.axhline(.1,c='r',ls='--');ax.set(xlabel='Fraction retained: '+('complete DECam parent' if parent else 'calibration distance references'),
            ylabel='Worst coarse-class cal catastrophic fraction',title='Same fixed cuts; each cal class N >= 30');ax.legend()
    save(fig,'04_quality_retention')

    fig,axes=plt.subplots(2,3,figsize=(14,8));x=np.arange(4)
    for i,cl in enumerate((0,1,2)):
        keys=[f'class{cl}_observed_r_{lo}_{hi}' for lo,hi in CFG['depth_r_bins']]
        for name in ('v0','v1','v1_cal'):
            for row,metric in enumerate(('tail_1','coverage95')):
                vals=[depth['groups'][k]['models'][name]['metrics'].get(metric,np.nan) for k in keys]
                axes[row,i].plot(x,vals,'o-',label=name)
                cis=[depth['groups'][k]['models'][name]['source_cluster_ci'] for k in keys]
                ci_index=2 if row==0 else 4
                lower=[c['ci95'][ci_index][0] if c else np.nan for c in cis]
                upper=[c['ci95'][ci_index][1] if c else np.nan for c in cis]
                axes[row,i].fill_between(x,lower,upper,alpha=.10)
                axes[row,i].set(xticks=x,xticklabels=[f'{lo}-{hi}' for lo,hi in CFG['depth_r_bins']],ylim=(0,1),xlabel='Final measured r',ylabel=metric,title=['dwarf','evolved','transition'][i])
        axes[0,i].legend()
    save(fig,'05_flux_depth_stress')

    p=load(OUT/'recommended_evaluation.npz');rows=p['rows'];test=t['split'][rows]==2
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for j,source in enumerate(('Gaia','specdis','spdist')):
        truth=truth_mu(t,source)[rows];use=test&np.isfinite(truth);xx=(truth[use]-10)/5;yy=(p['mu'][use,2]-10)/5
        hb=axes[0,j].hexbin(xx,yy,gridsize=65,bins='log',mincnt=1);fig.colorbar(hb,ax=axes[0,j],label='Count per hexagon (log scale)')
        lim=[min(xx.min(),yy.min()),max(xx.max(),yy.max())];axes[0,j].plot(lim,lim,'r--');axes[0,j].set(xlabel=source+' log10(d/kpc)',ylabel='Photometric log10(d/kpc)',title=f'N={use.sum():,}')
        residual=p['mu'][use,2]-truth[use];axes[1,j].hist(residual,bins=np.linspace(-8,8,100),histtype='step');axes[1,j].axvline(0,c='r');axes[1,j].set(xlabel='Delta mu [mag]',ylabel='Number',title=f'Outside plotted range: {np.sum(abs(residual)>8)}')
    save(fig,'06_distance_references')

    truth=truth_mu(t,'chosen')[rows];use=test&np.isfinite(truth);color=t['g0'][rows][use]-t['r0'][rows][use]
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,y,title in zip(axes,[t['g0'][rows][use]-truth[use],t['g0'][rows][use]-p['mu'][use,2]],['Reference CMD','Photometric median CMD']):
        h=ax.hexbin(color,y,gridsize=65,bins='log',mincnt=1);fig.colorbar(h,ax=ax,label='Count per hexagon (log scale)');ax.set(xlabel='(g-r)0',ylabel='Mg',title=title+f'; same N={use.sum():,}');ax.invert_yaxis()
    save(fig,'07_same_source_cmd')

    result['complete_application']=not preview and (OUT/'application.json').exists()
    if result['complete_application']:
        app=json.load(open(OUT/'application.json'));result['application']=app
        fig,axes=plt.subplots(1,2,figsize=(11,4));fields=app['fields'][:13];den=sum(f['n'] for f in fields)
        values=[sum(f['routes'][str(k)] for f in fields)/den for k in range(3)]
        axes[0].bar(list('ABC'),values);axes[0].set(ylabel='Fraction of complete DECam parent',title='Frozen recommended route')
        keys=['supported','ambiguous','research_quality','numerical_converged'];vals=[sum(f[k] for f in fields)/den for k in keys]
        axes[1].bar(range(4),vals);axes[1].set(xticks=range(4),xticklabels=keys,ylabel='Fraction of complete DECam parent',title='Independent flags; not disjoint populations');axes[1].tick_params(axis='x',rotation=25);save(fig,'08_application_states')
    if not preview and (OUT/'validation.json').exists():result['validation']=json.load(open(OUT/'validation.json'))
    result['experiments_only_preview']=preview
    write_json(OUT/('preview_report.json' if preview else 'report.json'),result)
    print('Report complete; application:',result['complete_application'],flush=True)
if __name__=='__main__':main()
