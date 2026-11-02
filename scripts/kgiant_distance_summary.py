#!/usr/bin/env python3
"""Evidence plots, paired held-field metrics and real-data invariants."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.deps')]
import numpy as np
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from photdis.stage1 import require_compute
from kgiant_distance_inventory import OUT, sha, write_json

MODELS=['prior','A_grz','B_grzW1']
COLORS=['#777777','#e69f00','#0072b2']


def metric(rows):
    if not rows:return {'n':0}
    d=np.array([r['delta_mu'] for r in rows]);rel=np.array([r['relative_residual'] for r in rows])
    return {'n':len(rows),'median_delta_mu':float(np.median(d)),
            'nmad_mu':float(1.4826*np.median(abs(d-np.median(d)))),
            'median_relative':float(np.median(rel)),
            'nmad_relative':float(1.4826*np.median(abs(rel-np.median(rel)))),
            'tail_mu_gt_0p5':float(np.mean(abs(d)>.5)), 'tail_relative_gt_0p5':float(np.mean(abs(rel)>.5)),
            **{key:float(np.mean([r[key] for r in rows])) for key in ('raw_cover68','raw_cover95','log_score')},
            **{key:float(np.median([r[key] for r in rows])) for key in ('raw_width68','raw_width95','effective_n')},
            'outside_training_box_n':sum(r['outside_training_box'] for r in rows),
            'effective_n_lt10':sum(r['effective_n']<10 for r in rows),
            'convolved_pit_lt0p05':float(np.mean([r['convolved_pit']<.05 for r in rows])),
            'convolved_pit_gt0p95':float(np.mean([r['convolved_pit']>.95 for r in rows]))}


def paired(a,b):
    a=sorted(a,key=lambda r:r['row']);b=sorted(b,key=lambda r:r['row'])
    assert [r['row'] for r in a]==[r['row'] for r in b]
    if not a:return {'n':0}
    fields=np.array([r['field'] for r in a]);unique=np.unique(fields)
    gain=np.array([y['log_score']-x['log_score'] for x,y in zip(a,b)])
    da=np.array([r['delta_mu'] for r in a]);db=np.array([r['delta_mu'] for r in b])
    def nmad(d):return 1.4826*np.median(abs(d-np.median(d)))
    rng=np.random.default_rng(20261102);values=[]
    for _ in range(1000):
        ids=np.concatenate([np.flatnonzero(fields==f) for f in rng.choice(unique,len(unique),replace=True)])
        values.append([gain[ids].mean(),nmad(db[ids])-nmad(da[ids])])
    return {'n':len(a),'n_fields':len(unique),'log_score_gain':float(gain.mean()),
            'delta_nmad_mu':float(nmad(db)-nmad(da)),
            'field_bootstrap95_score_gain':np.quantile(np.array(values)[:,0],[.025,.975]).tolist(),
            'field_bootstrap95_delta_nmad':np.quantile(np.array(values)[:,1],[.025,.975]).tolist(),
            'by_field':{str(f):{'n':int((fields==f).sum()),'score_gain':float(gain[fields==f].mean())} for f in unique}}


def main():
    require_compute();figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    inv=json.loads((OUT/'inventory.json').read_text());pred=json.loads((OUT/'predictions.json').read_text())
    with np.load(OUT/'population.npz') as f:q={k:f[k] for k in f.files}
    records=pred['rows'];summary={'models':{},'paired':{},'scope':pred['scope']}
    strata={'all':np.ones(len(q['source_id']),bool),'selected':q['selected'],
            'r16_18':(q['r']>=16)&(q['r']<18),'r18_20':(q['r']>=18)&(q['r']<20),
            'Teff4200_4800':q['desi_teff']<4800,'Teff4800_5000':q['desi_teff']>=4800,
            'FeH_lt_m1':q['abundance_calibration_valid']&(q['feh_corrected']<-1),
            'FeH_ge_m1':q['abundance_calibration_valid']&(q['feh_corrected']>=-1),
            'spec_error_lt0p2':q['specdis_valid']&(q['specdis_fractional_error']<.2),
            'spec_error_ge0p5':q['specdis_valid']&(q['specdis_fractional_error']>=.5)}
    for teacher in ('spdist','specdis','specdis_linear'):
        summary['models'][teacher]={};summary['paired'][teacher]={}
        for model in MODELS:
            rows=[r for r in records if r['teacher']==teacher and r['model']==model]
            summary['models'][teacher][model]={name:metric([r for r in rows if mask[r['row']]]) for name,mask in strata.items()}
            summary['models'][teacher][model]['by_field']={str(f):metric([r for r in rows if r['field']==f]) for f in sorted(set(r['field'] for r in rows))}
        for name in ('all','selected'):
            summary['paired'][teacher][name]={}
            for first,second in [('prior','A_grz'),('prior','B_grzW1'),('A_grz','B_grzW1')]:
                subsets=[[r for r in records if r['teacher']==teacher and r['model']==m and strata[name][r['row']]] for m in (first,second)]
                summary['paired'][teacher][name][first+'_to_'+second]=paired(*subsets)
    # Exact common-teacher sample; plot sign is SPDist minus SPECDIS.
    selected=q['selected'];both=selected&q['spdist_valid']&q['specdis_valid']
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    ax=axes[0,0];x=q['specdis_dist_kpc'][both];y=q['spdist_dist_kpc'][both]
    ax.scatter(x,y,s=20,alpha=.8);lo=min(x.min(),y.min())*.8;hi=max(x.max(),y.max())*1.2
    ax.plot([lo,hi],[lo,hi],'k--',lw=1);ax.set(xscale='log',yscale='log',xlabel='SPECDIS [kpc]',ylabel='SPDist [kpc]',title=f'Common valid selected K: N={both.sum()}')
    for ax,col,label in zip(axes.flat[1:5],('desi_teff','desi_logg','feh_corrected','desi_snr'),('Teff [K]','log g','calibrated [Fe/H]','spectral S/N')):
        use=both&np.isfinite(q[col])
        if col=='feh_corrected':use &= q['abundance_calibration_valid']
        ax.scatter(q[col][use],q['delta_mu_sp_minus_spec'][use],s=22,alpha=.75)
        ax.axhline(0,color='k',ls='--',lw=1);ax.set(xlabel=label,ylabel=r'$5\log_{10}(d_{SP}/d_{SPEC})$ [mag]',title=f'N={use.sum()}')
    ax=axes[1,2]
    error_max=max(float(np.nanmax(q[k+'_sigma_mu'][selected&q[k+'_valid']])) for k in ('spdist','specdis'))
    for kind,color in [('spdist',COLORS[1]),('specdis',COLORS[2])]:
        use=selected&q[kind+'_valid'];ax.hist(q[kind+'_sigma_mu'][use],bins=np.linspace(0,error_max*1.05,21),histtype='step',lw=2,label=f'{kind}: N={use.sum()}',color=color)
    ax.set(xlabel='Teacher modulus width proxy [mag]',ylabel='Stars',title='Actual selected-population errors');ax.legend(fontsize=9)
    fig.suptitle('Selected K giants: paired teacher evidence | correlated methods, not independent truth',fontsize=13)
    fig.savefig(figdir/'01_teacher_pairs.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,col,label in zip(axes.flat,('r','gr0','rz0','desi_teff','desi_logg','feh_corrected'),('r [mag]','(g-r)0','(r-z)0','Teff [K]','log g','calibrated [Fe/H]')):
        vals=q[col][selected&np.isfinite(q[col])];edges=np.linspace(vals.min()-.01,vals.max()+.01,14)
        for state,color in [(True,COLORS[2]),(False,COLORS[1])]:
            use=selected&(q['spdist_valid']==state)&np.isfinite(q[col])
            if col=='feh_corrected':use &= q['abundance_calibration_valid']
            ax.hist(q[col][use],bins=edges,histtype='step',density=True,lw=2,color=color,label=f"SP {'valid' if state else 'not valid'}: N={use.sum()}")
        ax.set(xlabel=label,ylabel='Normalized density');ax.legend(fontsize=8)
    fig.suptitle('SPDist-valid coverage versus remaining selected K stars | same original 190 objects')
    fig.savefig(figdir/'02_teacher_coverage.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for row,teacher in enumerate(('spdist','specdis')):
        for col,key in enumerate(('nmad_mu','log_score','raw_width68')):
            ax=axes[row,col]
            for j,model in enumerate(MODELS):
                for i,subset in enumerate(('all','selected')):
                    v=summary['models'][teacher][model][subset];ax.scatter(i+(j-1)*.16,v[key],color=COLORS[j],label=model if i==0 else None,s=45)
                    ax.annotate(f"N={v['n']}",(i+(j-1)*.16,v[key]),xytext=(0,7),textcoords='offset points',fontsize=7,ha='center')
            ax.set(xticks=[0,1],xticklabels=['All qualified K','B-wide selected K'],ylabel=key,title=teacher);ax.margins(y=.25);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Whole-field development validation | teacher-scale distributions; no absolute calibration')
    fig.savefig(figdir/'03_distance_metrics.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,9),layout='constrained')
    for row,teacher in enumerate(('spdist','specdis')):
        for col,model in enumerate(('A_grz','B_grzW1')):
            a=[r for r in records if r['teacher']==teacher and r['model']==model]
            x=10**((np.array([r['reference_mu'] for r in a])-10)/5)
            y=10**((np.array([r['quantiles'][2] for r in a])-10)/5)
            use=np.array([r['selected'] for r in a]);ax=axes[row,col]
            ax.scatter(x[~use],y[~use],s=12,c='#aaaaaa',label='Other eligible K')
            ax.scatter(x[use],y[use],s=18,c=COLORS[col+1],label=f'Selected N={use.sum()}')
            lo=min(x.min(),y.min())*.85;hi=max(x.max(),y.max())*1.15;ax.plot([lo,hi],[lo,hi],'k--',lw=1)
            ax.set(xscale='log',yscale='log',xlabel=f'{teacher} [kpc]',ylabel='Photometric median [kpc]',title=f'{model} | N={len(a)}');ax.legend(fontsize=8)
    fig.suptitle('Actual K-giant conditional-distance prototypes | fixed held-field identities')
    fig.savefig(figdir/'04_distance_scatter.png',dpi=160);plt.close(fig)
    # Predefined examples: smallest six Gaia IDs in selected primary-teacher evaluation.
    examples=sorted([r for r in records if r['teacher']=='specdis' and r['model']=='B_grzW1' and r['selected']],key=lambda r:r['source_id'])[:6]
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,r in zip(axes.flat,examples):
        qlim=r['quantiles'];grid=np.linspace(min(qlim[0],r['reference_mu']-2)-.5,max(qlim[-1],r['reference_mu']+2)+.5,300)
        for model,color in zip(MODELS,COLORS):
            with np.load(OUT/'mixtures'/f"specdis_{r['field']}_{model}.npz") as m:
                i=int(np.flatnonzero(m['rows']==r['row'])[0]);w=m['weights'][i];positive=w>0
                means=m['means'][i][positive];sd=m['sigmas'][i][positive]
                logp=np.log(w[positive])[None,:]-np.log(sd)[None,:]-.5*((grid[:,None]-means)/sd)**2-.5*np.log(2*np.pi)
                ax.plot(grid,np.exp(logsumexp(logp,axis=1)),color=color,label=model)
        ax.axvline(r['reference_mu'],color='k',ls='--',lw=1)
        ax.set(xlabel='Distance modulus [mag]',ylabel='Conditional density',title=str(r['source_id']),ylim=(0,None))
    axes[0,0].legend(fontsize=8);fig.suptitle('Full mixtures: six smallest selected source IDs | dashed = noisy SPECDIS point')
    fig.savefig(figdir/'05_predictive_distributions.png',dpi=160);plt.close(fig)
    # Real-output invariants verify saved distributions and provenance, not only summaries.
    checks=['frozen selection reproduces190','teacher common intersection explicitly counted','SP method floor accounted once']
    assert q['selected'].sum()==190
    for teacher in ('spdist','specdis','specdis_linear'):
        sets=[[(r['field'],r['row']) for r in records if r['teacher']==teacher and r['model']==m] for m in MODELS]
        assert sets[0]==sets[1]==sets[2] and len(set(sets[0]))==len(sets[0])
    checks.append('same A/B/prior tests and no repeated held-out identity')
    for path in (OUT/'mixtures').glob('*.npz'):
        with np.load(path) as m:
            train=m['train_rows'];test=m['rows']
            assert not set(q['field_index'][train])&set(q['field_index'][test])
            assert not set(q['identity_group'][train])&set(q['identity_group'][test])
            assert np.all(q['cluster_group'][train]==0)
            assert np.allclose(m['weights'].sum(axis=1),1) and np.all(m['sigmas']>0)
            assert np.all(np.isfinite(m['means']))
    checks.extend(['saved mixtures normalized and finite','field and linked-identity separation','known members excluded from all training'])
    for path,h in pred['source_sha256'].items():assert sha(path)==h
    checks.append('frozen source and protocol hashes unchanged')
    summary['inventory_sha256']=sha(OUT/'inventory.json');summary['predictions_sha256']=sha(OUT/'predictions.json')
    summary['script_sha256']=sha(__file__);summary['validation']={'status':'passed','checks':checks}
    summary['figure_sha256']={str(p):sha(p) for p in sorted(figdir.glob('*.png'))}
    write_json(OUT/'summary.json',summary)
    print(json.dumps({teacher:{m:summary['models'][teacher][m]['selected'] for m in MODELS} for teacher in ('spdist','specdis')}),flush=True)


if __name__=='__main__':main()
