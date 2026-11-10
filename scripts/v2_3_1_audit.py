#!/usr/bin/env python3
"""Correction deltas, exact inventory replay, and qualified selection evidence."""
import csv
import json
import os
import concurrent.futures
import multiprocessing as mp
import numpy as np
from v2_3_prepare import ROOT,DEST,TABLES
from v2_1_near import load,subset,sha,save_json,csv_rows,atomic_npz
from v2_3_diagnostics import references,masks
from photdis.stage1 import require_compute
from photdis.reference_v1_4 import unique_join
from photdis.probability_v2_1 import psf_admission
from photdis.baseline_parallax import parallax_check
from photdis.baseline_v0 import metrics
from photdis.selection_v2_2 import evidence_cells
from photdis.selection_v2_3_1 import distance_fit_from_snr,selection_upper,draw_observed_parallax
ARCHIVE=DEST/'inference_minus_g_archive'


def inventory():
    keys=['identity','TYPE','photometric_eligible','split','historical_identity','class_eligible','fit_eligible','nonstellar_risk','identity_conflict','optical_error','astro_quality','zpt_valid','specdis_valid','p_corr','gaia_parallax_error','r']
    path=ROOT/'data/v2_0/population.npz'
    with np.load(path) as z:t={k:z[k] for k in keys if k in z.files}
    # The frozen report population is a deliberately projected inventory, not
    # the full scientific source table. Enrich only missing fields by exact ID.
    source_path=ROOT/'data/baseline_v1_4/references.npz'
    with np.load(source_path) as z:
        joined=unique_join(t['identity'],z['identity']);found=joined>=0
        needed=(t['TYPE']=='PSF')&t['photometric_eligible']&(t['split']==0)&~t['historical_identity']
        assert found[needed].all(), 'Unmatched original owner for selection audit'
        for key in keys:
            if key in t:continue
            a=z[key];fill=np.nan if a.dtype.kind=='f' else False
            t[key]=np.full(len(joined),fill,dtype=a.dtype);t[key][found]=a[joined[found]]
        for key in ('photometric_eligible','class_eligible','fit_eligible','split'):
            assert np.array_equal(t[key][needed],z[key][joined[needed]]),key
    old=json.loads((ROOT/'data/v2_1/final_validation_reservation.json').read_text());used=[];sources={str(path.relative_to(ROOT)):sha(path),str(source_path.relative_to(ROOT)):sha(source_path)}
    for path in old['inputs']:
        with np.load(ROOT/path) as z:used.extend(z[k] for k in z.files if 'identity' in k)
        sources[path]=sha(ROOT/path)
    for name in ('information_train','information_eval','optical_probe','nonstellar_predictions','development'):
        path=ROOT/'data/v2_1'/(name+'.npz')
        with np.load(path) as z:
            used.append(z['identity'])
            if 'train_identity' in z:used.append(z['train_identity'])
        sources[str(path.relative_to(ROOT))]=sha(path)
    used=np.unique(np.concatenate(used));occupied=np.isin(t['identity'],used)
    psf,admission=psf_admission(t['TYPE'])
    base=psf&t['photometric_eligible']&(t['split']==0)&~t['historical_identity']
    expected=base&~occupied
    reserve=load(DEST/'reserved_inputs.npz')
    assert np.array_equal(t['identity'][expected],reserve['identity']), 'Inventory replay mismatch'
    clean=t['photometric_eligible']&~t['nonstellar_risk']&~t['identity_conflict']
    snr=t['p_corr']/t['gaia_parallax_error'];astro=t['astro_quality']&t['zpt_valid']
    fit=distance_fit_from_snr(snr,t['optical_error'],clean,astro,t['specdis_valid'])
    assert np.array_equal(fit[base],t['fit_eligible'][base]), 'Historical fit replay mismatch'
    # All clean unknown training-pool identities should be used iff distance-fit.
    # This checks remaining historical views did not introduce another SNR rule.
    resolved=base&clean&~t['class_eligible'];unexpected=resolved&(occupied!=fit)
    reason=np.full(len(base),'outside_base',dtype='U32')
    reason[base&~occupied]='reserved'
    reason[base&occupied]='other_historical_use'
    reason[base&occupied&fit]='distance_fit'
    reason[base&occupied&t['class_eligible']]='class_training'
    records=[]
    for lo,hi in [(-np.inf,18),(18,20),(20,22),(22,25),(25,np.inf),(-np.inf,np.inf)]:
        for why in ['reserved','distance_fit','class_training','other_historical_use']:
            u=base&(t['r']>=lo)&(t['r']<hi)&(reason==why)
            records.append(dict(r_low=lo,r_high=hi,reason=why,n=int(u.sum()),optical_precise=int((u&(t['optical_error']<.03)).sum()),anchor_snr20=int((u&astro&(snr>=20)).sum()),specdis_valid=int((u&t['specdis_valid']).sum()),class_eligible=int((u&t['class_eligible']).sum())))
    csv_rows(TABLES/'selection_history.csv',records)
    good=reserve['parallax_eligible'];c=reserve['photometric_eligible']&~reserve['nonstellar_risk']&~reserve['identity_conflict'];a=reserve['astro_quality']&reserve['zpt_valid']
    upper=selection_upper(reserve['optical_error'],c,a,reserve['specdis_valid'])
    supported=good&~reserve['class_eligible']&np.isfinite(reserve['gaia_parallax_error'])&~np.isnan(upper)
    # If source replay leaves unresolved clean/unknown selection, do not upgrade
    # an anchor-only conditional check to a complete historical-selection claim.
    exact_historical=not unexpected.any()
    assert np.all(reserve['p_corr'][supported]/reserve['gaia_parallax_error'][supported]<upper[supported])
    rows=[]
    for lo,hi in [(-np.inf,18),(18,20),(20,22),(22,25),(-np.inf,np.inf)]:
        u=(reserve['r']>=lo)&(reserve['r']<hi)&good
        rows.append(dict(r_low=lo,r_high=hi,gaia_n=int(u.sum()),analytic_n=int((u&supported).sum()),upper20=int((u&supported&(upper==20)).sum()),untruncated=int((u&supported&np.isposinf(upper)).sum()),unresolved=int((u&~supported).sum())))
    csv_rows(TABLES/'selection_qualification.csv',rows)
    atomic_npz(DEST/'selection_conditions.npz',identity=reserve['identity'],TYPE=reserve['TYPE'],supported=supported,snr_upper=upper)
    save_json(DEST/'selection_replay.json',dict(reserved=len(reserve['identity']),replay_exact=True,fit_rule_exact=True,base=int(base.sum()),clean_unknown=int(resolved.sum()),unexplained_clean_unknown=int(unexpected.sum()),historical_rule_exact_in_clean_unknown=exact_historical,
        inputs=sources,PSF_admission=admission,
        conditioning='Hold non-parallax covariates, source qualification/error/class/spatial flags fixed; dynamic historical distance fit admission replayed. Not a model of parent spectroscopic selection or correlated source evidence.',
        new_blind_test=False))
    print('SELECTION REPLAY',len(reserve['identity']),'unexplained',int(unexpected.sum()),flush=True)


_T=_CONDITIONS=_EDGES=None


def selection_chunk(task):
    name,path=task;t=_T;cond=_CONDITIONS;p=load(path);r=p['rows'];use=cond['supported'][r]
    rr=r[use]
    if not len(rr):return rr,{},[]
    pdf=p['pdf'][use];g=t['g0'][rr];error=t['gaia_parallax_error'][rr];obs=t['p_corr'][rr];upper=cond['snr_upper'][rr]
    check=parallax_check(pdf,_EDGES[name],g,obs,error,np.full(len(rr),-np.inf),upper,order=32)
    # Keep stable pre-existing log-density and subtract selection log mass.
    from photdis.baseline_v1_2 import full_parallax_log_density
    raw=full_parallax_log_density(pdf,_EDGES[name],g,obs,error,order=32)
    probability=check['selection_probability'];assert np.all(probability>0)
    output=dict(raw_log32=raw,raw_pit32=check['pit']*probability,conditional_log=raw-np.log(probability),conditional_pit=check['pit'],selection_probability=probability)
    # Independent direct forward draws; all source/covariate branches of the
    # reconstructed distance-fit rule are reapplied, including SpecDis/error.
    draws=draw_observed_parallax(pdf,_EDGES[name],g,error,np.random.default_rng(20261110+int(r[0])),repeats=2048)
    clean=t['photometric_eligible'][rr]&~t['nonstellar_risk'][rr]&~t['identity_conflict'][rr]
    accepted=~distance_fit_from_snr(draws/error[:,None],t['optical_error'][rr,None],clean[:,None],(t['astro_quality'][rr]&t['zpt_valid'][rr])[:,None],t['specdis_valid'][rr,None])
    empirical=accepted.mean(axis=1)
    output['simulation_probability']=empirical
    output['simulation_observed_pit']=np.divide(((draws<=obs[:,None])&accepted).sum(axis=1),accepted.sum(axis=1),out=np.full(len(rr),np.nan),where=accepted.sum(axis=1)>0)
    # Use independent first/second halves to estimate CDFs and score accepted
    # simulation draws. Uniform conditional PIT is a method check, not data fit.
    records=[]
    for i,row in enumerate(rr):
        reference=np.sort(draws[i,:1024]);selected_reference=np.sort(draws[i,:1024][accepted[i,:1024]])
        simulated=draws[i,1024:][accepted[i,1024:]]
        if len(selected_reference)>=20 and len(simulated):
            rawpit=(np.searchsorted(reference,simulated)+.5)/(len(reference)+1)
            cpit=(np.searchsorted(selected_reference,simulated)+.5)/(len(selected_reference)+1)
            records.append(dict(row=int(row),model=name,n=len(simulated),snr_upper=float(upper[i]),raw_low=int((rawpit<.05).sum()),raw_high=int((rawpit>.95).sum()),conditional_low=int((cpit<.05).sum()),conditional_high=int((cpit>.95).sum())))
    return rr,output,records


def selection_scores():
    global _T,_CONDITIONS,_EDGES
    _T=load(DEST/'reserved_inputs.npz');_CONDITIONS=load(DEST/'selection_conditions.npz')
    _EDGES={n:load(DEST/(n+'_grid.npz'))['edges'] for n in ('control','candidate')}
    records=[];hists=[];simrows=[];simchecks=[]
    for name in ('control','candidate'):
        output={k:np.full(len(_T['identity']),np.nan) for k in ('raw_log32','raw_pit32','conditional_log','conditional_pit','selection_probability','simulation_probability','simulation_observed_pit')}
        simulations=[];paths=sorted((DEST/'chunks').glob('reserved_'+name+'_*.npz'))
        with concurrent.futures.ProcessPoolExecutor(max_workers=16,mp_context=mp.get_context('fork')) as pool:
            for r,values,sim in pool.map(selection_chunk,[(name,p) for p in paths]):
                for k,v in values.items():output[k][r]=v
                simulations+=sim
        p=load(DEST/f'reserved_{name}.npz');supported=_CONDITIONS['supported'];upper=_CONDITIONS['snr_upper']
        atomic_npz(DEST/f'reserved_{name}_selection.npz',identity=_T['identity'],TYPE=_T['TYPE'],**output)
        groups={'all':np.ones(len(supported),bool),'r_lt20':_T['r']<20,'no_W1':~_T['B'],'upper20':upper==20,'untruncated':np.isposinf(upper)}
        for group,base in groups.items():
            for selection,sel in [('all',np.ones(len(supported),bool)),('near_scope',p['near_scope'])]:
                u=base&sel&supported;key=dict(model=name,group=group,selection=selection,n=int(u.sum()))
                for kind,log,pit in [('raw',output['raw_log32'],output['raw_pit32']),('conditional',output['conditional_log'],output['conditional_pit'])]:
                    x=pit[u];records.append(dict(**key,score=kind,log_mean=float(log[u].mean()) if u.any() else None,pit_low=float((x<.05).mean()) if len(x) else None,pit_high=float((x>.95).mean()) if len(x) else None,negative=int((u&(_T['p_corr']<0)).sum()),mean_selection_probability=float(output['selection_probability'][u].mean()) if u.any() else None))
                    hist=np.histogram(x,np.linspace(0,1,11))[0];assert hist.sum()==u.sum()
                    hists.extend(dict(**key,score=kind,low=i/10,high=(i+1)/10,count=int(v)) for i,v in enumerate(hist))
                ss=[r for r in simulations if u[r['row']]];total=sum(r['n'] for r in ss)
                simrows.append(dict(**key,draws=total,**{k:sum(r[k] for r in ss)/total if total else None for k in ('raw_low','raw_high','conditional_low','conditional_high')}))
        diff=output['simulation_probability'][supported]-output['selection_probability'][supported]
        pitdiff=output['simulation_observed_pit'][supported]-output['conditional_pit'][supported]
        simchecks.append(dict(model=name,n=int(supported.sum()),mean_probability_difference=float(np.mean(diff)),rms_probability_difference=float(np.sqrt(np.mean(diff**2))),median_abs_observed_pit_difference=float(np.nanmedian(abs(pitdiff))),draws_per_object=2048,max_original_order8_vs32_PIT_difference=float(np.max(abs(output['raw_pit32'][supported]-p['gaia_pit'][supported])))))
    csv_rows(TABLES/'reserved_selection_scores.csv',records);csv_rows(TABLES/'reserved_selection_pit.csv',hists);csv_rows(TABLES/'selection_forward_simulation.csv',simrows);csv_rows(TABLES/'selection_simulation_checks.csv',simchecks)
    print('SELECTION SCORES COMPLETE',flush=True)


def correction_changes():
    rows=[]
    for role,source in [('development',ROOT/'data/v2_1/development.npz'),('reserved',DEST/'reserved_inputs.npz'),('application',ROOT/'data/v2_2/application_inputs.npz')]:
        t=load(source);old=load(ARCHIVE/f'{role}_candidate.npz');new=load(DEST/f'{role}_candidate.npz')
        assert np.array_equal(old['identity'],new['identity'])
        for split,label in ([(1,'cal'),(2,'development_test')] if role=='development' else [(None,role)]):
            base=np.ones(len(t['identity']),bool) if split is None else t['split']==split
            for lo,hi in [(0,.01),(.01,.03),(.03,.1),(.1,np.inf),(0,np.inf)]:
                u=base&(t['optical_error']>=lo)&(t['optical_error']<hi)
                ref,sigma=references(t,'chosen') if 'label_source' in t else (np.full(len(u),np.nan),None)
                valid=u&np.isfinite(ref);ga=u&t.get('parallax_eligible',np.zeros(len(u),bool));n=int(u.sum())
                for version,p in [('minus_g_old',old),('plus_g_corrected',new)]:
                    rows.append(dict(role=label,error_low=lo,error_high=hi,version=version,n=n,selected=int((u&p['near_scope']).sum()),
                        median_q=float(np.median(p['qnear'][u])) if n else None,gaia_n=int(ga.sum()),gaia_log=float(p['gaia_log'][ga].mean()) if ga.any() else None,
                        changed_scope=int((u&(old['near_scope']!=new['near_scope'])).sum()),median_abs_mu_change=float(np.median(abs(new['mu'][u,2]-old['mu'][u,2]))) if n else None,
                        mean_abs_q_change=float(np.mean(abs(new['qnear'][u]-old['qnear'][u]))) if n else None,
                        **{'distance_'+k:v for k,v in metrics(p['mu'][valid],ref[valid]).items()}))
    csv_rows(TABLES/'inference_correction_by_error.csv',rows)


def far_mass():
    t=load(ROOT/'data/v2_1/development.npz');results=[]
    for name,folder in [('control',DEST),('candidate_old',ARCHIVE),('candidate_corrected',DEST)]:
        model='control' if name=='control' else 'candidate';grid=load(DEST/(model+'_grid.npz'))['edges']
        values={k:np.full(len(t['identity']),np.nan) for k in ('specdis','spdist')}
        for path in sorted((folder/'chunks').glob(f'development_{model}_*.npz')):
            p=load(path);r=p['rows'];u=(t['split'][r]==2)&(t['class_label'][r]==1)
            if not u.any():continue
            rr=r[u]
            for kind in values:
                ref,_=references(subset(t,rr),kind);target=t['g0'][rr]-ref
                fraction=np.maximum(0,np.minimum(grid[None,1:],target[:,None]+1)-np.maximum(grid[None,:-1],target[:,None]-1))/np.diff(grid)
                values[kind][rr]=np.sum(p['pdf'][u]*fraction,axis=1)
        score=load(folder/f'development_{model}_teacher.npz')
        for kind,mass in values.items():
            ref,_=references(t,kind);d=10**((ref-10)/5)
            for lo,hi in [(10,20),(20,50)]:
                u=(t['split']==2)&(t['class_label']==1)&(d>=lo)&(d<hi)&np.isfinite(mass)
                results.append(dict(model=name,reference=kind,low_kpc=lo,high_kpc=hi,n=int(u.sum()),mean_mass_within1=float(mass[u].mean()),median_mass_within1=float(np.median(mass[u])),fraction_mass_below01=float((mass[u]<.1).mean()),reference_convolved_log=float(score[kind][u].mean())))
    csv_rows(TABLES/'far_full_pdf_mass.csv',results)


def paired_metal():
    t=load(ROOT/'data/v2_1/development.npz');pairs=load(ARCHIVE/'metal_poor_pairs.npz');a=unique_join(pairs['poor_identity'],t['identity']);b=unique_join(pairs['control_identity'],t['identity']);assert (a>=0).all() and (b>=0).all()
    rows=[];audit=[]
    for name in ('control','candidate'):
        p=load(DEST/f'development_{name}.npz');da=10**((t['ref_chosen'][a]-10)/5);db=10**((t['ref_chosen'][b]-10)/5)
        for lo,hi in [(0,2),(2,5),(5,10),(0,np.inf)]:
            for selected in (False,True):
                ua=(da>=lo)&(da<hi);ub=(db>=lo)&(db<hi)
                if selected:ua&=p['near_scope'][a];ub&=p['near_scope'][b]
                both=ua&ub
                audit.append(dict(model=name,low_kpc=lo,high_kpc=hi,selected=selected,pairs=len(a),both=int(both.sum()),poor_only=int((ua&~ub).sum()),ordinary_only=int((ub&~ua).sum()),neither=int((~ua&~ub).sum()),unique_controls_both=len(np.unique(b[both]))))
                for rule,mask in [('both_in_stratum',both),('poor_side_defines_stratum',ua)]:
                    for group,idx in [('metal_poor',a),('matched_ordinary',b)]:
                        take=idx[mask];rows.append(dict(model=name,rule=rule,group=group,low_kpc=lo,high_kpc=hi,selected=selected,**metrics(p['mu'][take],t['ref_chosen'][take])))
    csv_rows(TABLES/'metal_poor_pair_retention.csv',audit);csv_rows(TABLES/'metal_poor_pair_conserving.csv',rows)


def coverage():
    dev=load(ROOT/'data/v2_1/development.npz');t=load(ROOT/'data/v2_2/application_inputs.npz');cd=evidence_cells(dev);c=evidence_cells(t);rows=[]
    for evidence,mask in [('development_only',dev['split']==2),('cal_plus_development',np.isin(dev['split'],[1,2]))]:
        good=mask&np.isfinite(dev['ref_chosen']);count=np.bincount(cd[good],minlength=3000)
        for name in ('control','candidate'):
            p=load(DEST/f'application_{name}.npz')
            for selection,u in [('all',np.ones(len(c),bool)),('q95_r20',(p['qnear']>.95)&(t['r']<20)),('near_scope',p['near_scope'])]:
                w=t['sample_weight'][u];n=count[c[u]];rows.append(dict(model=name,selection=selection,evidence=evidence,n=int(u.sum()),weight=float(w.sum()),zero_weight=float(w[n==0].sum()),sparse1_19_weight=float(w[(n>0)&(n<20)].sum()),reference_n=int(good.sum())))
    csv_rows(TABLES/'application_coverage_definitions.csv',rows)


def validate():
    m=json.loads((DEST/'inference_plus_g_migration.json').read_text());checks={}
    def check(k,v):checks[k]=bool(v);assert v,k
    for n,h in m['models'].items():check('unchanged_model:'+n,sha(DEST/(n+'.joblib'))==h)
    for path,h in m['controls'].items():check('unchanged_control:'+path,sha(DEST/path)==h)
    for item in m['archived']:check('old_preserved:'+item['path'],sha(ARCHIVE/item['path'])==item['sha256'])
    from v2_3_run import candidate_inference_stamp
    for path in (DEST/'chunks').glob('*_candidate_*.json'):check('inference_version:'+path.stem,json.loads(path.read_text())['inference']==candidate_inference_stamp())
    check('production_validation',json.loads((DEST/'validation.json').read_text())['all_passed'])
    for name in ('selection_replay','inference_plus_g_migration'):check(name,(DEST/(name+'.json')).exists())
    with open(TABLES/'metal_poor_pair_retention.csv') as f:
        for i,row in enumerate(csv.DictReader(f)):
            check('pair_conservation:'+str(i),sum(int(row[k]) for k in ('both','poor_only','ordinary_only','neither'))==int(row['pairs']))
    with open(TABLES/'selection_simulation_checks.csv') as f:
        for row in csv.DictReader(f):
            check('forward_probability:'+row['model'],abs(float(row['mean_probability_difference']))<.01)
            check('forward_observed_CDF:'+row['model'],float(row['median_abs_observed_pit_difference'])<.03)
    conditions=load(DEST/'selection_conditions.npz');u=conditions['supported']
    for name in ('control','candidate'):
        p=load(DEST/f'reserved_{name}_selection.npz')
        check('selected_CDF_valid:'+name,np.all(np.isfinite(p['conditional_pit'][u])) and np.all((p['conditional_pit'][u]>=0)&(p['conditional_pit'][u]<=1)))
        check('selected_log_valid:'+name,np.isfinite(p['conditional_log'][u]).all())
    replay=json.loads((DEST/'selection_replay.json').read_text())
    check('replay_exact',replay['replay_exact'] and replay['fit_rule_exact'])
    save_json(DEST/'correction_validation.json',dict(actual_checks=len(checks),all_passed=all(checks.values()),checks=checks,job=os.getenv('SLURM_JOB_ID')))
    print('CORRECTION VALIDATION',len(checks),flush=True)


def main():
    require_compute();inventory();selection_scores();correction_changes();far_mass();paired_metal();coverage();validate()
    save_json(DEST/'correction_audit_complete.json',dict(source=sha(ROOT/'scripts/v2_3_1_audit.py'),tables={p.name:sha(p) for p in TABLES.glob('*.csv')},already_inspected_reserve=True))
    print('CORRECTION AUDIT COMPLETE',flush=True)


if __name__=='__main__':main()
