#!/usr/bin/env python3
"""Allocated per-star scientific diagnostics; primary v1.4 artifacts read-only."""
import json, sys, time, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.deps')]
import numpy as np
import joblib
from scipy.signal import find_peaks
from photdis.stage1 import require_compute, legacy_ids
from photdis.reference_v1_4 import unique_join, optional_photometry
from photdis.diagnostics_v1_4 import CLASS_NAMES,residual_stats,paired_stats,fixed_examples,wise_joint,cell_area
from photdis.baseline_v0 import distance_quantiles
from baseline_v1_4_experiments import arrays,truth,save_json,subset
from baseline_v1_4_extract import sha,atomic_npz
from baseline_v1_4_assemble import load
BASE=ROOT/'data/baseline_v1_4';OUT=BASE/'diagnostics'
NAMES=['frozen_'+k for k in 'ABC']+['expanded_'+k for k in 'ABC']+['matched_grz','matched_griz']
SOURCES=['Gaia','specdis','spdist','chosen']
INPUTS={};CHECKS={}


def check(name,value):
    CHECKS[name]=bool(value)
    if not value:raise AssertionError(name)


def input_file(path,expected=None):
    h=sha(path);INPUTS[str(path.relative_to(ROOT))]=h
    if expected is not None:check(str(path.relative_to(BASE))+'_hash',h==expected)
    return path


def reference_table():
    a=json.loads((BASE/'assembly.json').read_text())
    input_file(BASE/'references.npz',a['reference_sha256'])
    # Retain scientifically useful provenance; source selection bit masks are not
    # interpreted as population labels. All original columns remain in references.
    keys=['identity','ls_id','RA','DEC','cell','split','historical_identity','identity_conflict',
          'photometric_eligible','nonstellar_risk','class_label','class_eligible','class_train_eligible',
          'fit_eligible','B','C','i_usable','g0','r0','z0','i0','w10','w20','r',
          'sigma_g','sigma_r','sigma_i','sigma_z','sigma_w1','sigma_w2','optical_error','EBV',
          'rvs_teff','rvs_logg','rvs_feh','rvs_teff_err','rvs_logg_err','rvs_feh_err',
          'rvs_parameter_quality','rvs_survey','rvs_program','target_mws_target','target_desi_target',
          'ref_distance','label_source','sigma_mu','specdis_distance','spdist_distance',
          'specdis_sigma_mu','spdist_sigma_mu','specdis_valid','spdist_valid','anchor_astrometry',
          'p_corr','gaia_parallax_error','parallax_eligible','zero_id_match','match_epoch_known']
    with np.load(BASE/'references.npz') as z:t={k:z[k] for k in keys}
    check('reference_unique_identity',len(np.unique(t['identity']))==len(t['identity']))
    # Display unknown distinctly even if a raw unqualified label exists.
    t['class_label']=np.where(t['class_eligible'],t['class_label'],-1)
    return t


def population_summary(t):
    clean=t['photometric_eligible']&~t['identity_conflict']&~t['nonstellar_risk']
    keys=['identity','cell','split','historical_identity','g0','r0','z0','r','B','C','i_usable',
          'class_label','fit_eligible','class_train_eligible','rvs_feh','rvs_parameter_quality']
    atomic_npz(OUT/'reference_population.npz',**subset({k:t[k] for k in keys},np.flatnonzero(clean)))
    foot=[]
    for c in np.unique(t['cell']):
        q=t['cell']==c
        foot.append(dict(cell=int(c),n=int(q.sum()),clean=int((q&clean).sum()),
                         historical=int((q&t['historical_identity']).sum()),
                         split_counts={str(k):int((q&(t['split']==k)).sum()) for k in range(4)}))
    return foot


def join_model(name,t):
    ep=BASE/'experiments'/(name+'_distance')
    scores=json.loads((ep/'scores.json').read_text());input_file(ep/'scores.json')
    input_file(ep/'evaluation.npz',scores['evaluation_sha256']);p=load(ep/'evaluation.npz')
    ids=load(input_file(BASE/'experiments'/(name+'_identities.npz')))
    check(name+'_eval_identity',np.array_equal(ids['distance_eval_identity'],p['identity']))
    check(name+'_row_identity',np.array_equal(t['identity'][p['rows']],p['identity']))
    check(name+'_eval_unique',len(np.unique(p['identity']))==len(p['identity']))
    q=subset(t,p['rows']);q.update(mu=p['mu'],supported=p['supported'],ambiguous=p['ambiguous'])
    check(name+'_only_cal_test',np.isin(q['split'],[1,2]).all())
    check(name+'_no_historical',not q['historical_identity'].any())
    check(name+'_no_conflict',not q['identity_conflict'].any())
    cp=BASE/'experiments'/(name+'_class_scores.npz')
    ci=json.loads(cp.with_name(name+'_class_integrity.json').read_text())
    c=load(input_file(cp,ci['sha256']));j=unique_join(q['identity'],c['identity'])
    check(name+'_class_join_complete',np.all(j>=0))
    q['class_scores']=c.get('marginalized',c['nominal'])[j]
    q['class_nominal']=c['nominal'][j]
    gp=BASE/'gaia_refined'/name
    gs=json.loads((gp/'scores.json').read_text());input_file(gp/'scores.json')
    g=load(input_file(gp/'evaluation.npz',gs['evaluation_sha256']))
    j=unique_join(q['identity'],g['identity']);valid=j>=0
    check(name+'_refined_exact_eligibility',np.array_equal(valid,q['parallax_eligible']))
    for key in ['full_pit','full_log_density']:
        q[key]=np.full(len(j),np.nan);q[key][valid]=g[key][j[valid]]
    for kind in SOURCES:
        q['ref_'+kind]=truth(q,kind)
        if kind=='Gaia':
            with np.errstate(divide='ignore',invalid='ignore'):
                q['error_'+kind]=5/np.log(10)*q['gaia_parallax_error']/q['p_corr']
        else:q['error_'+kind]=q['sigma_mu'] if kind=='chosen' else q[kind+'_sigma_mu']
    use=q['split']==2
    report={}
    for kind in SOURCES:
        report[kind]={}
        for label,mask in [('all',np.ones(len(use),bool))]+[(n,q['class_label']==v) for v,n in CLASS_NAMES.items()]:
            s=residual_stats(q['mu'][use&mask],q['ref_'+kind][use&mask]);report[kind][label]=s
            old=scores['groups']['all' if label=='all' else 'class'+str(next(k for k,v in CLASS_NAMES.items() if v==label))][kind]
            check(name+'_'+kind+'_'+label+'_count',s['n']==old['n'])
            for k in ['bias_mag','nmad_mag','tail_1','coverage68','coverage95']:
                if s['n']:check(name+'_'+kind+'_'+label+'_'+k,np.isclose(s[k],old[k],atol=1e-10,rtol=0))
        e=q['error_'+kind][use&np.isfinite(q['ref_'+kind])]
        report[kind]['reference_sigma_mu_quantiles']=np.quantile(e,[.16,.5,.84]).tolist() if len(e) else []
    ga=use&q['parallax_eligible'];old=gs['groups']['all']
    check(name+'_Gaia_refined_count',int(ga.sum())==old['n'])
    check(name+'_Gaia_refined_mean',np.isclose(np.mean(q['full_log_density'][ga]),old['mean_log_density']))
    report['Gaia_refined']=dict(old)
    report['spectral_parameter_rows']=int((use&q['rvs_parameter_quality']).sum())
    # Retain all original sky scores, plus new stratified maps from the joined data.
    report['original_sky']={k:v for k,v in scores['groups'].items() if k.startswith('cell')}
    spatial={}
    strata={'all':np.ones(len(use),bool),'dwarf':q['class_label']==0,'evolved':q['class_label']==1,
            'r_lt18':q['r']<18,'r18_20':(q['r']>=18)&(q['r']<20),'r_ge20':q['r']>=20}
    for kind in SOURCES:
        spatial[kind]={}
        for layer,m in strata.items():
            spatial[kind][layer]=[]
            for cell in np.unique(q['cell'][use]):
                sel=use&m&(q['cell']==cell)
                s=residual_stats(q['mu'][sel],q['ref_'+kind][sel])
                spatial[kind][layer].append(dict(cell=int(cell),area_sqdeg=float(cell_area(cell)),**s))
    report['sky']=spatial
    atomic_npz(OUT/(name+'.npz'),**subset(q,np.flatnonzero(use)))
    print('joined',name,int(use.sum()),'checks',len(CHECKS),flush=True)
    return report


def pairs():
    report={}
    for old,new in [('frozen_'+k,'expanded_'+k) for k in 'ABC']+[('matched_grz','matched_griz')]:
        a=load(OUT/(old+'.npz'));b=load(OUT/(new+'.npz'))
        check(new+'_pair_identity',np.array_equal(a['identity'],b['identity']))
        check(new+'_pair_truth',np.allclose(a['ref_chosen'],b['ref_chosen'],equal_nan=True))
        report[new]={}
        for kind in SOURCES:
            report[new][kind]={}
            for label,m in [('all',np.ones(len(a['identity']),bool))]+[(n,a['class_label']==v) for v,n in CLASS_NAMES.items()]:
                da=a['mu'][:,2]-a['ref_'+kind];db=b['mu'][:,2]-b['ref_'+kind]
                s=paired_stats(da[m],db[m],a['cell'][m]);report[new][kind][label]=s
                if s['n']:
                    check(new+'_'+kind+'_'+label+'_transition_conservation',s['new_bad']-s['old_bad']==s['harmed']-s['rescued'])
    return report


def recover_parent_raw():
    pp=BASE/'parent_coverage.npz';pc=json.loads((BASE/'parent_coverage.json').read_text())
    parent=load(input_file(pp,pc['output_sha256']));n=len(parent['identity'])
    check('parent26000',n==26000)
    check('parent_unique_lsid',len(np.unique(parent['ls_id']))==n)
    fields=[f'{p}_{b}' for b in ['W1','W2'] for p in ['FLUX','FLUX_IVAR','NOBS','MW_TRANSMISSION','WISEMASK','FRACFLUX']]
    raw={k:np.full(n,np.nan) for k in fields};seen=np.zeros(n,bool);repeats=0;files=0
    order=np.argsort(parent['ls_id']);ids=parent['ls_id'][order]
    paths=sorted((BASE/'photometry').glob('*.npz'))
    check('all1600_retained_checkpoints',len(paths)==1600)
    for path in paths:
        with np.load(path) as z:
            if 'RELEASE' not in z.files:continue
            keys={k:z[k] for k in ['RELEASE','BRICKID','OBJID']};lsid=legacy_ids(keys)
            j=np.searchsorted(ids,lsid);good=(j<n)&(ids[np.minimum(j,n-1)]==lsid)
            if not good.any():continue
            marker=json.loads(path.with_suffix('.json').read_text());input_file(path,marker['sha256']);files+=1
            pos=order[j[good]]
            check(path.stem+'_native_unique',len(np.unique(pos))==len(pos))
            for k in fields:
                v=z[k][good]
                check(path.stem+'_'+k+'_duplicates_agree',np.allclose(raw[k][pos[seen[pos]]],v[seen[pos]],equal_nan=True))
                raw[k][pos]=v
            repeats+=int(seen[pos].sum());seen[pos]=True
    check('parent_raw_all_matched',seen.all())
    joint={}
    for band in ['W1','W2']:
        k=band.lower()
        j=wise_joint(*[raw[p+'_'+band] for p in ['FLUX','FLUX_IVAR','NOBS','MW_TRANSMISSION','WISEMASK','FRACFLUX']],parent['sample_weight'])
        joint[band]=j
        snr=raw['FLUX_'+band]*np.sqrt(raw['FLUX_IVAR_'+band])
        mask=(raw['WISEMASK_'+band]==0)&(raw['FRACFLUX_'+band]<1)
        usable=(raw['NOBS_'+band]>0)&np.isfinite(snr)&(raw['FLUX_IVAR_'+band]>0)&(snr>=5)&mask&(raw['MW_TRANSMISSION_'+band]>0)
        check(k+'_original_measurement_state',np.array_equal(usable,parent[k+'_measurement_state']==4))
        check(k+'_joint_total',np.isclose(sum(c['fraction'] for c in j['cells'])+j['unmeasured_fraction'],1))
        parent[k+'_usable']=usable;parent[k+'_snr_raw']=snr
    parent.update(raw);atomic_npz(OUT/'parent.npz',**parent)
    return dict(joint=joint,matched=int(seen.sum()),repeated_rows_agree=repeats,contributing_sweeps=files,
                weighted_i=float(np.average(parent['i_usable'],weights=parent['sample_weight'])))


def pdf_examples():
    t=load(OUT/'matched_griz.npz');ref=t['ref_chosen'];delta=t['mu'][:,2]-ref;finite=np.isfinite(ref)
    masks={'ordinary_success':finite&(t['class_label']==0)&(abs(delta)<.2),
           'evolved_failure':finite&(t['class_label']==1)&(abs(delta)>1),
           'ambiguity_flagged':finite&t['ambiguous'],
           'no_W1':finite&~t['B'],'unknown':finite&(t['class_label']==-1)}
    selected=fixed_examples(t['identity'],masks);rows=np.array([i for v in selected.values() for i in v],int)
    check('pdf_examples_unique',len(np.unique(rows))==len(rows))
    case=dict(rows=rows,identity=t['identity'][rows],category=np.array([k for k,v in selected.items() for _ in v]),
              g0=t['g0'][rows],ref=ref[rows],ref_error=t['error_chosen'][rows],class_label=t['class_label'][rows],
              reference_source=t['label_source'][rows],r=t['r'][rows],B=t['B'][rows])
    report=[]
    for name in ['matched_grz','matched_griz']:
        mp=BASE/'experiments'/(name+'_distance.joblib');meta=json.loads(mp.with_suffix('.json').read_text())
        model=joblib.load(input_file(mp,meta['sha256']));model.forest.n_jobs=1
        m,e,ids=arrays(t,rows,True);p=model.predict(m,e,ids,128)
        mu=distance_quantiles(m[:,0],p['q_Meffective'])[0];original=load(OUT/(name+'.npz'))
        j=unique_join(ids,original['identity']);check(name+'_PDF_quantile_reproduction',np.allclose(mu,original['mu'][j],rtol=0,atol=1e-6))
        check(name+'_PDF_probability',np.allclose(p['pdf'].sum(axis=1),1,atol=1e-8))
        case[name+'_pdf']=p['pdf'][:,::-1];case[name+'_mu_edges']=m[:,0,None]-model.edges[None,::-1]
        case[name+'_quantiles']=mu
        for k,row in enumerate(rows):
            edges=case[name+'_mu_edges'][k];pdf=case[name+'_pdf'][k];center=(edges[:-1]+edges[1:])/2
            peaks,_=find_peaks(pdf,prominence=max(pdf)*.02)
            # Integral in the fixed +-1mag severe-error window, with partial bins.
            overlap=np.maximum(0,np.minimum(edges[1:],ref[row]+1)-np.maximum(edges[:-1],ref[row]-1))/np.diff(edges)
            report.append(dict(model=name,identity=int(ids[k]),category=str(case['category'][k]),
                               mass_within_1mag=float(pdf@overlap),peak_mu=center[peaks].tolist(),
                               peak_mass=pdf[peaks].tolist(),reference_mu=float(ref[row]),
                               median_delta=float(mu[k,2]-ref[row]),width68=float(mu[k,3]-mu[k,1]),
                               supported=bool(p['supported'][k]),ambiguous=bool(p['ambiguous'][k])))
    atomic_npz(OUT/'pdf_examples.npz',**case)
    return report


def main():
    require_compute();OUT.mkdir(exist_ok=True)
    if (OUT/'complete.json').exists():raise FileExistsError('Completed diagnostics exist; do not silently overwrite')
    start=time.time();t=reference_table();foot=population_summary(t)
    report={'models':{},'footprint':foot,'baseline_commit':'46cb68a','seed':20261105,'job_id':os.environ.get('SLURM_JOB_ID')}
    for name in NAMES:report['models'][name]=join_model(name,t)
    del t
    report['pairs']=pairs();print('paired bootstrap complete',flush=True)
    report['parent']=recover_parent_raw();print('parent WISE recovery complete',flush=True)
    report['pdf_examples']=pdf_examples();print('fixed PDF examples complete',flush=True)
    # Verify input immutability at the end, including every checkpoint actually read.
    for path,h in INPUTS.items():check('unchanged:'+path,sha(ROOT/path)==h)
    report['inputs']=INPUTS;report['checks']=CHECKS;report['seconds']=time.time()-start
    report['code_sha256']={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/photdis/diagnostics_v1_4.py',ROOT/'docs/baseline_v1_4_diagnostics_protocol.md']}
    save_json(OUT/'statistics.json',report)
    save_json(OUT/'complete.json',dict(statistics_sha256=sha(OUT/'statistics.json'),checks=len(CHECKS),job_id=os.environ.get('SLURM_JOB_ID')))
    print('COMPLETE',len(CHECKS),'checks',round(time.time()-start),'seconds',flush=True)

if __name__=='__main__':main()
