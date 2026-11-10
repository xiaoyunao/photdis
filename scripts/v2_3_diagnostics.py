#!/usr/bin/env python3
"""Frozen-model comparisons, prospective scope and target-evidence limits."""
import json
import concurrent.futures
import multiprocessing as mp
import os
import numpy as np
from scipy.special import logsumexp, log_ndtr
from scipy.spatial import cKDTree
from v2_3_prepare import ROOT, DEST, TABLES
from v2_1_near import load, subset, csv_rows, save_json, sha, atomic_npz, strata, reference, DISTANCES
from photdis.stage1 import require_compute
from photdis.baseline_v0 import metrics
from photdis.audit import rvs_metallicity
from photdis.probability_v2_1 import interval_score, migration
from photdis.selection_v2_2 import reliability_bins, evidence_cells
from photdis.baseline_v1_2 import full_parallax_log_density


def interval95(mu, truth):
    lo,hi=mu[:,0],mu[:,4]
    return hi-lo+40*np.maximum(lo-truth,0)+40*np.maximum(truth-hi,0)


def masks(t):
    n=len(t['identity']);m={'all':np.ones(n,bool)}
    for c,label in [(-1,'unknown'),(0,'dwarf'),(1,'evolved'),(2,'other')]:m[label]=t['class_label']==c
    m.update(no_W1=~t['B'],W1_only=t['B']&~t['C'],W1W2=t['C'])
    for lo,hi in [(-np.inf,18),(18,20),(20,22),(22,25)]:m[f'r_{lo}_{hi}']=(t['r']>=lo)&(t['r']<hi)
    for lo,hi in [(-np.inf,.3),(.3,.8),(.8,1.3),(1.3,np.inf)]:m[f'gr_{lo}_{hi}']=(t['g0']-t['r0']>=lo)&(t['g0']-t['r0']<hi)
    for lo,hi in [(0,.03),(.03,.1),(.1,np.inf)]:m[f'error_{lo}_{hi}']=(t['optical_error']>=lo)&(t['optical_error']<hi)
    for c in np.unique(t['cell']):m[f'cell_{c}']=t['cell']==c
    if 'rvs_feh' in t:
        feh,domain=rvs_metallicity(t['rvs_feh'],t['rvs_teff'],t['rvs_logg'])
        quality=domain&t['rvs_parameter_quality']&np.isfinite(t['rvs_feh_err'])&(t['rvs_feh_err']>0)
        m['metal_poor_dwarf']=quality&(t['class_label']==0)&(feh< -1.5)
        m['ordinary_dwarf']=quality&(t['class_label']==0)&(feh>= -.5)
    m['low_luminosity']=np.isfinite(t['ref_chosen'])&(t['g0']-t['ref_chosen']>10)
    return m


def references(t,kind):
    ref,sigma=reference(t,kind)
    # Source-valid VAC estimates on known contaminants are audited separately,
    # not silently promoted to qualified stellar distance references.
    clean=~t['nonstellar_risk']&~t.get('identity_conflict',np.zeros(len(ref),bool))
    good=clean&np.isfinite(ref)&np.isfinite(sigma)&(sigma>0)
    ref[~good]=np.nan
    return ref,sigma


def teacher_log(pdf,edges,target,sigma):
    a=(edges[None,:-1]-target[:,None])/sigma[:,None]
    b=(edges[None,1:]-target[:,None])/sigma[:,None]
    hi=np.where(a>0,log_ndtr(-a),log_ndtr(b))
    lo=np.where(a>0,log_ndtr(-b),log_ndtr(a))
    with np.errstate(divide='ignore',invalid='ignore'):
        lm=hi+np.log(-np.expm1(np.minimum(lo-hi,0)))-np.log(np.diff(edges))[None,:]
        lp=np.where(pdf>0,np.log(pdf),-np.inf)
    return logsumexp(lp+lm,axis=1)


_SCORE_T = _SCORE_EDGES = None


def score_chunk(path):
    t=_SCORE_T;edges=_SCORE_EDGES;p=load(path);r=p['rows'];out={}
    assert sha(path)==json.loads(path.with_suffix('.json').read_text())['sha256']
    for kind in ('chosen','Gaia','specdis','spdist'):
        ref,sigma=references(subset(t,r),kind);use=np.isfinite(ref);value=np.full(len(r),np.nan)
        value[use]=teacher_log(p['pdf'][use],edges,t['g0'][r][use]-ref[use],sigma[use]);out[kind]=value
    ga=t['parallax_eligible'][r];check=None
    if ga.any():
        log32=full_parallax_log_density(p['pdf'][ga],edges,t['g0'][r][ga],t['p_corr'][r][ga],t['gaia_parallax_error'][r][ga],order=32)
        check=dict(first_row=int(r[0]),n=int(ga.sum()),max_log16_32=float(np.max(abs(log32-p['gaia_log'][ga]))),mean_log16_32=float(np.mean(log32-p['gaia_log'][ga])))
    return r,out,check


def cached_scores(role,name,t):
    global _SCORE_T,_SCORE_EDGES
    _SCORE_T=t;_SCORE_EDGES=load(DEST/(name+'_grid.npz'))['edges']
    out={k:np.full(len(t['identity']),np.nan) for k in ('chosen','Gaia','specdis','spdist')};checks=[]
    paths=sorted((DEST/'chunks').glob(f'{role}_{name}_*.npz'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=16,mp_context=mp.get_context('fork')) as pool:
        for r,part,check in pool.map(score_chunk,paths):
            for k in out:out[k][r]=part[k]
            if check is not None:checks.append(dict(role=role,model=name,**check))
    atomic_npz(DEST/f'{role}_{name}_teacher.npz',identity=t['identity'],TYPE=t['TYPE'],**out)
    return out,checks


def compare(role,t):
    allrows=[];grows=[];pitrows=[];riskrows=[];relrows=[];quadrature=[]
    groups=masks(t)
    splits=[(1,'cal'),(2,'development_test')] if role=='development' else [(None,'same_sky_new_identity')]
    for name in ('control','candidate'):
        p=load(DEST/f'{role}_{name}.npz');assert np.array_equal(p['identity'],t['identity'])
        scores,check=cached_scores(role,name,t);quadrature+=check
        selections={'all':np.ones(len(p['identity']),bool),'q95_r20':(p['qnear']>.95)&(t['r']<20),'near_scope':p['near_scope']}
        for split,label in splits:
            base=np.ones(len(p['identity']),bool) if split is None else t['split']==split
            for group,mask in groups.items():
                for selection,sel in selections.items():
                    gm=base&mask;use=gm&sel
                    ga=use&t['parallax_eligible'];score=p['gaia_log'][ga];pit=p['gaia_pit'][ga]
                    key=dict(role=label,model=name,group=group,selection=selection)
                    grows.append(dict(**key,n=int(ga.sum()),eligible=int((gm&t['parallax_eligible']).sum()),negative=int((ga&(t['p_corr']<0)).sum()),
                        log_mean=float(score.mean()) if len(score) else None,log_median=float(np.median(score)) if len(score) else None,
                        pit_low=float((pit<.05).mean()) if len(pit) else None,pit_high=float((pit>.95).mean()) if len(pit) else None))
                    hist=np.histogram(pit,bins=np.linspace(0,1,11))[0];assert hist.sum()==ga.sum()
                    pitrows.extend(dict(**key,low=i/10,high=(i+1)/10,n=int(v)) for i,v in enumerate(hist))
                    for kind in ('chosen','Gaia','specdis','spdist'):
                        ref,sigma=references(t,kind);d=10**((ref-10)/5)
                        valid=gm&np.isfinite(ref)
                        riskrows.append(dict(**key,reference=kind,**migration(ref[valid],sel[valid],sigma[valid])))
                        if selection=='all':
                            relrows.extend(dict(**key,reference=kind,**x) for x in reliability_bins(p['qnear'][valid],ref[valid]<15))
                        for lo,hi in [(0,np.inf)]+DISTANCES:
                            eligible=valid&(d>=lo)&(d<hi);take=eligible&sel
                            ls=scores[kind][take]
                            allrows.append(dict(**key,reference=kind,low_kpc=lo,high_kpc=hi,eligible=int(eligible.sum()),
                                **metrics(p['mu'][take],ref[take]),
                                interval68=float(interval_score(p['mu'][take],ref[take]).mean()) if take.any() else None,
                                interval95=float(interval95(p['mu'][take],ref[take]).mean()) if take.any() else None,
                                reference_convolved_log=float(ls.mean()) if len(ls) and np.isfinite(ls).all() else None,
                                nonfinite_log=int((~np.isfinite(ls)).sum())))
    for file,rows in [('distance',allrows),('gaia',grows),('pit',pitrows),('risk',riskrows),('reliability',relrows),('quadrature',quadrature)]:
        csv_rows(TABLES/f'{role}_{file}.csv',rows)
    return allrows,grows,riskrows


def poor_metal(t):
    groups=masks(t);poor=groups['metal_poor_dwarf']&(t['split']==2);ordinary=groups['ordinary_dwarf']&(t['split']==2)
    # Fixed matching scales and calipers, no tuning after inspecting outcomes.
    x=np.column_stack((t['g0']-t['r0'],t['r0']-t['z0'],t['rvs_teff']/1000,t['optical_error'],t['sigma_mu']))
    scales=np.array([.1,.1,.5,.03,.2]);valid=np.isfinite(x).all(axis=1)
    pairs=[]
    for source in (1,2):
        a=np.flatnonzero(poor&valid&(t['label_source']==source));b=np.flatnonzero(ordinary&valid&(t['label_source']==source))
        if not len(a) or not len(b):continue
        dist,index=cKDTree(x[b]/scales).query(x[a]/scales)
        delta=abs(x[a]-x[b[index]])/scales
        keep=np.max(delta,axis=1)<=1
        pairs.extend(zip(a[keep],b[index[keep]],dist[keep]))
    a=np.array([p[0] for p in pairs],int);b=np.array([p[1] for p in pairs],int)
    rows=[]
    for name in ('control','candidate'):
        p=load(DEST/f'development_{name}.npz')
        for group,ids in [('metal_poor',a),('matched_ordinary',b)]:
            ref=t['ref_chosen'][ids];d=10**((ref-10)/5)
            for lo,hi in [(0,2),(2,5),(5,10),(0,np.inf)]:
                for selected in (False,True):
                    use=(d>=lo)&(d<hi)
                    if selected:use&=p['near_scope'][ids]
                    take=ids[use]
                    rows.append(dict(model=name,group=group,selected=selected,low_kpc=lo,high_kpc=hi,
                        total_matched=len(ids),**metrics(p['mu'][take],t['ref_chosen'][take])))
    csv_rows(TABLES/'metal_poor_matching.csv',rows)
    atomic_npz(DEST/'metal_poor_pairs.npz',poor_identity=t['identity'][a],control_identity=t['identity'][b],
        poor_TYPE=t['TYPE'][a],control_TYPE=t['TYPE'][b],match_scaled_distance=np.array([p[2] for p in pairs]))
    save_json(DEST/'metal_poor_matching.json',dict(eligible=int(poor.sum()),matched=len(a),unique_controls=len(np.unique(b)),
        color_scales=[.1,.1],temperature_scale_K=500,optical_error_scale=.03,reference_sigma_scale=.2,
        max_dimension_difference=1,reference_source_exact=True,with_replacement=True,
        claim='Descriptive conditional comparison, not causal attribution; repeated matched controls are not independent',
        median_abs_differences=np.median(abs(x[a]-x[b]),axis=0).tolist() if len(a) else None))


def bootstrap(t):
    a=load(DEST/'development_control.npz');b=load(DEST/'development_candidate.npz');rng=np.random.default_rng(20261110);rows=[]
    for group,mask in masks(t).items():
        if group not in ('all','dwarf','evolved','metal_poor_dwarf','low_luminosity'):continue
        use=mask&(t['split']==2)&np.isfinite(t['ref_chosen']);idx=np.flatnonzero(use)
        if not len(idx):continue
        ref=t['ref_chosen'][idx];values={}
        for metric in ('tail','interval68','relative'):
            def loss(p):
                delta=p['mu'][idx,2]-ref
                if metric=='tail':return (abs(delta)>1).astype(float)
                if metric=='relative':return 10**(delta/5)-1
                return interval_score(p['mu'][idx],ref)
            diff=loss(b)-loss(a);cells=t['cell'][idx];u=np.unique(cells)
            count=np.array([(cells==c).sum() for c in u]);sums=np.array([diff[cells==c].sum() for c in u])
            draw=rng.integers(0,len(u),(2000,len(u)));v=sums[draw].sum(axis=1)/count[draw].sum(axis=1)
            rows.append(dict(group=group,metric=metric,n=len(idx),cells=len(u),candidate_minus_control=float(diff.mean()),
                spatial_low95=float(np.quantile(v,.025)),spatial_high95=float(np.quantile(v,.975))))
    csv_rows(TABLES/'paired_spatial_uncertainty.csv',rows)


def application():
    t=load(ROOT/'data/v2_2/application_inputs.npz');dev=load(ROOT/'data/v2_1/development.npz');reserved=load(DEST/'reserved_inputs.npz')
    c=evidence_cells(t);cd=evidence_cells(dev);cr=evidence_cells(reserved)
    devvalid=np.isfinite(dev['ref_chosen']);rvalid=reserved['parallax_eligible']
    dc=np.bincount(cd[devvalid],minlength=3000);rc=np.bincount(cr[rvalid],minlength=3000)
    records=[];cells=[]
    for name in ('control','candidate'):
        p=load(DEST/f'application_{name}.npz')
        for selection,selected in [('q95_r20',(p['qnear']>.95)&(t['r']<20)),('near_scope',p['near_scope'])]:
            for lo,hi in [(-np.inf,18),(18,20),(20,22),(22,25),(-np.inf,20)]:
                for wise,mask in [('all',np.ones(len(c),bool)),('no_W1',~t['B']),('W1_only',t['B']&~t['C']),('W1W2',t['C'])]:
                    u=selected&(t['r']>=lo)&(t['r']<hi)&mask;w=t['sample_weight'][u];cc=c[u]
                    records.append(dict(model=name,selection=selection,r_low=lo,r_high=hi,WISE=wise,n=int(u.sum()),weight=float(w.sum()),
                        zero_dev_reference_weight=float(w[dc[cc]==0].sum()),sparse_dev_reference_weight=float(w[(dc[cc]>0)&(dc[cc]<20)].sum()),
                        no_new_parallax_weight=float(w[rc[cc]==0].sum()),
                        mean_unlabeled=float(np.average(p['weights'][u,3],weights=w)) if len(w) else None))
            for cell in np.unique(c[selected]):
                u=selected&(c==cell);cells.append(dict(model=name,selection=selection,cell=int(cell),n=int(u.sum()),weight=float(t['sample_weight'][u].sum()),
                    dev_distance_n=int(dc[cell]),reserved_parallax_n=int(rc[cell])))
    csv_rows(TABLES/'application_scope.csv',records);csv_rows(TABLES/'application_coverage.csv',cells)


def admission_effect(t):
    control=load(DEST/'development_control.npz');oldq=load(ROOT/'data/v2_2/probability_canonical.npz')['qnear'];rows=[]
    for split,role in [(1,'cal'),(2,'development_test')]:
        for group,mask in masks(t).items():
            if group not in ('all','dwarf','evolved','unknown','metal_poor_dwarf','low_luminosity'):continue
            for name,mu,q,log in [('historical_expanded_A',t['mu'],oldq,t['full_log_density']),('same_PSF_QRF',control['mu'],control['qnear'],control['gaia_log'])]:
                base=(t['split']==split)&mask;use=base&np.isfinite(t['ref_chosen']);ga=base&t['parallax_eligible']
                rows.append(dict(role=role,group=group,model=name,**metrics(mu[use],t['ref_chosen'][use]),
                    interval68=float(interval_score(mu[use],t['ref_chosen'][use]).mean()) if use.any() else None,
                    full_gaia_log=float(log[ga].mean()) if ga.any() else None,
                    q95_selected=int((use&(q>.95)).sum())))
    csv_rows(TABLES/'admission_effect.csv',rows)


def posterior_examples(t):
    groups=masks(t);d=10**((t['ref_chosen']-10)/5);dev=t['split']==2
    choices={'near_dwarf':groups['dwarf']&(d<2),'poor_metal_dwarf':groups['metal_poor_dwarf']&(d<5),
        'giant_10_20':groups['evolved']&(d>=10)&(d<20),'giant_20_50':groups['evolved']&(d>=20)&(d<50),
        'unlabeled_low_luminosity':groups['unknown']&groups['low_luminosity'],
        'near_boundary':(d>=8)&(d<12)}
    records=[]
    for group,mask in choices.items():
        ids=np.flatnonzero(dev&mask&np.isfinite(t['ref_chosen']))
        if not len(ids):continue
        row=int(ids[np.argmin(t['identity'][ids])])
        for name in ('control','candidate'):
            start=row//256*256;p=load(DEST/'chunks'/f'development_{name}_{start:07}.npz');j=row-start
            assert p['identity'][j]==t['identity'][row] and p['TYPE'][j]=='PSF'
            edges=load(DEST/(name+'_grid.npz'))['edges'];mu=t['g0'][row]-(edges[:-1]+edges[1:])/2
            for x,prob,width in zip(mu,p['pdf'][j],np.diff(edges)):
                records.append(dict(group=group,identity=int(t['identity'][row]),TYPE='PSF',model=name,mu=float(x),density=float(prob/width),bin_width=float(width),
                    ref_mu=float(t['ref_chosen'][row]),ref_sigma=float(t['sigma_mu'][row]),qnear=float(p['qnear'][j]),
                    mu025=float(p['mu'][j,0]),mu975=float(p['mu'][j,4]),selection='Smallest identity in reference-defined development stratum, independent of model outcomes'))
    csv_rows(TABLES/'posterior_examples.csv',records)


def decision(distance,gaia,risk):
    def d(model,group):return next(x for x in distance if x['role']=='cal' and x['model']==model and x['group']==group and x['selection']=='all' and x['reference']=='chosen' and x['low_kpc']==0 and np.isinf(x['high_kpc']))
    def g(model):return next(x for x in gaia if x['role']=='cal' and x['model']==model and x['group']=='all' and x['selection']=='all')
    def r(model):return next(x for x in risk if x['role']=='cal' and x['model']==model and x['group']=='all' and x['selection']=='near_scope' and x['reference']=='chosen')
    a,b=d('control','all'),d('candidate','all');ra,rb=r('control'),r('candidate');ma,mb=d('control','metal_poor_dwarf'),d('candidate','metal_poor_dwarf')
    guards=dict(all_tail=b['tail_1']<=a['tail_1'],evolved_tail=d('candidate','evolved')['tail_1']<=d('control','evolved')['tail_1'],
        dwarf_tail=d('candidate','dwarf')['tail_1']<=d('control','dwarf')['tail_1']+.005,
        full_gaia=g('candidate')['log_mean']>=g('control')['log_mean'],
        near_contamination=rb['far_contamination'] is not None and rb['far_contamination']<=ra['far_contamination']+.005,
        near_completeness=rb['completeness'] is not None and rb['completeness']>=ra['completeness']-.02,
        metal_poor=abs(mb['median_relative'])<abs(ma['median_relative']) or mb['interval68']<ma['interval68'])
    save_json(DEST/'decision.json',dict(guards=guards,cal_pass=all(guards.values()),deployment='unchanged',
        interpretation='No automatic adoption; new identity evaluation cannot reopen tuning.'))


def main():
    require_compute();TABLES.mkdir(exist_ok=True)
    t=load(ROOT/'data/v2_1/development.npz');a,g,r=compare('development',t)
    compare('reserved',load(DEST/'reserved_inputs.npz'));poor_metal(t);bootstrap(t);application();admission_effect(t);posterior_examples(t);decision(a,g,r)
    save_json(DEST/'diagnostics_complete.json',dict(job=os.getenv('SLURM_JOB_ID'),models=json.loads((DEST/'models_frozen.json').read_text()),
        tables={p.name:sha(p) for p in TABLES.glob('*.csv')},
        source_sha256=sha(ROOT/'scripts/v2_3_diagnostics.py'),new_scale_validation=False))
    print('DIAGNOSTICS COMPLETE',flush=True)


if __name__=='__main__':main()
