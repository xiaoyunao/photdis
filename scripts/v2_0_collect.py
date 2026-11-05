#!/usr/bin/env python3
"""Collect frozen tables, native morphology, ranges and percentage performance."""
import json
import os
from pathlib import Path
import numpy as np
import fitsio
import joblib

from v2_0_common import *
from photdis.stage1 import require_compute, legacy_ids
from photdis.audit import rvs_metallicity
from baseline_v1_6_wise_raw import states
from baseline_v1_5_r0 import SPECTRAL

RAW_FIELDS=['TYPE','EBV']
for band in ('G','R','I','Z','W1','W2'):
    RAW_FIELDS += [f'{key}_{band}' for key in ('FLUX','FLUX_IVAR','MW_TRANSMISSION','NOBS')]
RAW_FIELDS += ['WISEMASK_W1','WISEMASK_W2','FRACFLUX_W1','FRACFLUX_W2']


def recover_raw(t,parent):
    path=DEST/'native_report.npz'
    completion=path.with_suffix('.json')
    ids=np.unique(np.r_[t['ls_id'],parent['ls_id']])
    if completion.exists():
        old=json.loads(completion.read_text())
        audited(path,old['sha256'])
        raw=load(path)
        assert np.array_equal(raw['ls_id'],ids)
        return raw
    raw={k:np.full(len(ids),np.nan) for k in RAW_FIELDS if k!='TYPE'}
    raw['TYPE']=np.full(len(ids),'','U4')
    seen=np.zeros(len(ids),bool)
    hashes=json.loads((OUT/'native_photometry.json').read_text())['sweep_hashes']
    for i,(name,digest) in enumerate(sorted(hashes.items()),1):
        p=OUT/'photometry'/f'{name}.npz'
        with np.load(p) as z:
            if 'RELEASE' not in z.files:
                continue
            keys=legacy_ids({k:z[k] for k in ('RELEASE','BRICKID','OBJID')})
            pos=np.searchsorted(ids,keys)
            take=(pos<len(ids))&(ids[np.minimum(pos,len(ids)-1)]==keys)
            if take.any():
                audited(p,digest)
                q=pos[take]
                assert len(np.unique(q))==len(q) and not seen[q].any()
                for key in RAW_FIELDS:
                    raw[key][q]=np.char.strip(z[key][take].astype('U')) if key=='TYPE' else z[key][take]
                seen[q]=True
        if i%200==0:
            print('native morphology',i,int(seen.sum()),len(ids),flush=True)
    zero=OUT/'zero_candidates_raw.npz'
    audited(zero,json.loads(zero.with_suffix('.json').read_text())['sha256'])
    with np.load(zero) as z:
        q=np.flatnonzero(~seen)
        j=unique_join(ids[q],z['ls_id'])
        assert (j>=0).all()
        for key in RAW_FIELDS:
            raw[key][q]=np.char.strip(z[key][j].astype('U')) if key=='TYPE' else z[key][j]
    assert (raw['TYPE']!='').all()
    raw['ls_id']=ids
    atomic_npz(path,**raw)
    save_json(completion,dict(sha256=sha(path),rows=len(ids),zero_fallback=len(q)))
    audited(path)
    return raw


def enrich_raw(t,raw):
    j=unique_join(t['ls_id'],raw['ls_id'])
    assert (j>=0).all()
    t=dict(t)
    for key in RAW_FIELDS:
        t[key]=raw[key][j]
    t['wise_state']=states(t)
    for b,name in enumerate(('w1','w2')):
        if name+'_measurement_state' in t:
            assert np.array_equal(t['wise_state'][:,b],t[name+'_measurement_state'])
        t[name+'_measurement_state']=t['wise_state'][:,b]
    for band in ('G','R','I','Z','W1','W2'):
        f,v,tr=(t[key+'_'+band] for key in ('FLUX','FLUX_IVAR','MW_TRANSMISSION'))
        good=np.isfinite(f)&(f>0)&np.isfinite(tr)&(tr>0)
        mag=np.full(len(f),np.nan)
        mag[good]=22.5-2.5*np.log10(f[good])
        key=band.lower()
        t[key]=mag
        t[key+'0']=mag+2.5*np.log10(np.where(tr>0,tr,np.nan))
        t['snr_'+key]=np.where(np.isfinite(v)&(v>0),f*np.sqrt(np.maximum(v,0)),np.nan)
    ra=t.get('RA',t.get('ra'))
    dec=t.get('DEC',t.get('dec'))
    if ra is not None:
        t['galactic_b']=galactic(ra,dec)[1]
    t['gr']=t['g0']-t['r0'];t['rz']=t['r0']-t['z0']
    t['zW1']=np.where(t['wise_state'][:,0]==4,t['z0']-t['w10'],np.nan)
    t['W1W2']=np.where((t['wise_state']==4).all(axis=1),t['w10']-t['w20'],np.nan)
    t['ri']=np.where(t['i_usable'],t['r0']-t['i0'],np.nan)
    t['iz']=np.where(t['i_usable'],t['i0']-t['z0'],np.nan)
    return t


def model_table(name,t):
    ep=OUT/'experiments'/f'{name}_distance'
    score=json.loads((ep/'scores.json').read_text())
    p=load(audited(ep/'evaluation.npz',score['evaluation_sha256']))
    ids=load(audited(OUT/'experiments'/f'{name}_identities.npz'))
    assert np.array_equal(p['identity'],t['identity'][p['rows']])
    assert np.array_equal(p['identity'],ids['distance_eval_identity'])
    train=ids['distance_train_identity']
    legacy=OUT/'experiments'/f'{name}_legacy_training.npz'
    if legacy.exists():
        train=np.r_[train,load(audited(legacy))['distance_train_identity']]
    assert len(np.unique(train))==len(train)
    assert not np.intersect1d(train,p['identity']).size
    q=subset(t,p['rows'])
    q.update(mu=p['mu'],supported=p['supported'],ambiguous=p['ambiguous'])
    q['in_model_training']=np.isin(q['identity'],train)
    q['outside_train_precision']=q['optical_error']>=.03
    q['mu_raw']=q['mu']
    q['mu_calibrated_available']=np.zeros(len(q['identity']),bool)
    q['distance_kpc']=10**((q['mu']-10)/5)
    q['pdf_source_row']=p['rows']
    q['cached_full_pdf']=np.zeros(len(q['identity']),bool)
    gp=OUT/'gaia_refined'/name
    gs=json.loads((gp/'scores.json').read_text())
    g=load(audited(gp/'evaluation.npz',gs['evaluation_sha256']))
    j=unique_join(q['identity'],g['identity'])
    use=j>=0
    assert np.array_equal(use,q['parallax_eligible'])
    for key in ('full_pit','full_log_density'):
        q[key]=np.full(len(j),np.nan);q[key][use]=g[key][j[use]]
    for kind in ('chosen','Gaia','specdis','spdist'):
        ref=truth(q,kind)
        q['ref_'+kind]=ref
        error=q['mu'][:,2]-ref
        q['delta_mu_'+kind]=error
        q['epsilon_pct_'+kind]=100*np.expm1(np.log(10)*error/5)
    modelpath=OUT/'experiments'/f'{name}_distance.joblib'
    audited(modelpath,json.loads(modelpath.with_suffix('.json').read_text())['sha256'])
    atomic_npz(DEST/f'chosen_{name}.npz',**q)
    source_rows=[]
    source_code=[]
    for code,kind in enumerate(('Gaia','specdis','spdist')):
        valid=np.isfinite(q['ref_'+kind])
        source_rows.extend(np.flatnonzero(valid))
        source_code.extend([code]*int(valid.sum()))
    source_rows=np.asarray(source_rows)
    source_code=np.asarray(source_code)
    atomic_npz(DEST/f'sources_{name}.npz',chosen_view_row=source_rows,source=source_code,
               identity=q['identity'][source_rows])
    rows=[]
    gaia=[]
    masks=strata(q)
    for split,label in [(1,'cal'),(2,'development_test')]:
        for layer,mask in masks.items():
            use=mask&(q['split']==split)
            for kind in ('chosen','Gaia','specdis','spdist'):
                rows.append(dict(model=name,split=label,group=layer,reference=kind,
                                 **percent_metrics(q['mu'][use],q['ref_'+kind][use])))
            use &= q['parallax_eligible']
            pit=q['full_pit'][use]
            gaia.append(dict(model=name,split=label,group=layer,n=int(use.sum()),
                             negative=int((q['p_corr'][use]<0).sum()),
                             mean_log_density=np.mean(q['full_log_density'][use]) if len(pit) else None,
                             pit_extreme=np.mean((pit<.05)|(pit>.95)) if len(pit) else None,
                             **{f'pit_bin{i}':int(v) for i,v in enumerate(np.histogram(pit,np.linspace(0,1,11))[0])}))
    csv_rows(TABLES/f'performance_{name}.csv',rows)
    csv_rows(TABLES/f'gaia_{name}.csv',gaia)
    print('unified',name,len(q['identity']),flush=True)
    return dict(model=name,train_n=len(train),chosen_view=f'data/v2_0/chosen_{name}.npz',
                source_view=f'data/v2_0/sources_{name}.npz',model_path=str(modelpath.relative_to(ROOT)),
                model_sha256=sha(modelpath),evaluation_sha256=score['evaluation_sha256'],
                identities=str((OUT/'experiments'/f'{name}_identities.npz').relative_to(ROOT)),
                pdf='Quantiles cached; full PDF requires frozen model with stored source rows/Sobol128 unless explicit example cache.',
                n=len(q['identity']))


def range_tables(t,parent):
    with np.load(ROOT/'data/baseline_v1_3/references.npz') as z:
        old={k:z[k] for k in z.files}
    old['EBV']=old['ebv'];old['RA']=old['ra'];old['DEC']=old['dec']
    old['galactic_b']=galactic(old['ra'],old['dec'])[1]
    old.update({new:old[prior] for new,prior in [('rvs_teff','desi_teff'),('rvs_logg','desi_logg'),('rvs_feh','desi_feh')]})
    for b in ('g','r','z','w1','w2'):
        old['snr_'+b]=2.5/np.log(10)/old['sigma_'+b]
        if b not in old:
            old[b]=np.full(len(old['identity']),np.nan)
    old['gr']=old['g0']-old['r0'];old['rz']=old['r0']-old['z0']
    old['zW1']=np.where(old['B'],old['z0']-old['w10'],np.nan)
    old['W1W2']=np.where(old['C'],old['w10']-old['w20'],np.nan)
    ids=load(OUT/'experiments/expanded_A_identities.npz')
    legacy=load(OUT/'experiments/expanded_A_legacy_training.npz')
    variables=[*list('griz'),'w1','w2','g0','r0','i0','z0','w10','w20','gr','rz','ri','iz','zW1','W1W2',
               *['snr_'+b for b in ('g','r','i','z','w1','w2')],
               'optical_error','EBV','galactic_b','ref_distance','sigma_mu','rvs_teff','rvs_logg','rvs_feh']
    selections=[('parent_weighted',parent,None,parent['sample_weight'])]
    for family,key in [('class_train','class_train_identity'),('distance_train','distance_train_identity')]:
        j=unique_join(ids[key],t['identity']);k=unique_join(legacy[key],old['identity'])
        assert (j>=0).all() and (k>=0).all()
        combined={}
        for var in variables:
            combined[var]=np.r_[old[var][k] if var in old else np.full(len(k),np.nan),
                                t[var][j] if var in t else np.full(len(j),np.nan)]
        combined['identity']=np.r_[old['identity'][k],t['identity'][j]]
        selections.append((family,combined,None,None))
    clean=t['photometric_eligible']&~t['identity_conflict']&~t['nonstellar_risk']
    for split,label in [(1,'cal'),(2,'test')]:
        for name,mask in [('class',t['class_eligible']),('chosen',t['label_source']>0),
                          ('Gaia',t['parallax_eligible']),('specdis',t['specdis_valid']),('spdist',t['spdist_valid'])]:
            selections.append((label+'_'+name,t,clean&(t['split']==split)&mask,None))
    rows=[]
    for label,data,mask,weight in selections:
        n=len(data['identity'])
        if mask is None:
            mask=np.ones(n,bool)
        for var in variables:
            a=data.get(var,np.full(n,np.nan))[mask]
            record=ranges(a,None if weight is None else weight[mask])
            rows.append(dict(sample=label,variable=var,weighted=weight is not None,
                             range_definition='finite measured values; unavailable bands retained as missing',
                             selection_threshold='distance train max(grz error)<0.03' if label=='distance_train' else 'see methods selection table',
                             plot_range='per-figure metadata',**record))
    csv_rows(TABLES/'B_sample_ranges.csv',rows)


def application():
    prep=json.loads((ROOT/'manifests/baseline_v0_prepared.json').read_text())
    app=json.loads((ROOT/'manifests/baseline_v1_application.json').read_text())
    rows=[]
    for record,field in zip(app['fields'],prep['fields']):
        config=json.loads((ROOT/field['config']).read_text())
        path=ROOT/'data/baseline_v1'/f'catalog_{record["field"]:02}.fits'
        audited(path,record['sha256'])
        counts={}
        with fitsio.FITS(str(path)) as h:
            assert h[1].get_nrows()==record['n']
            for start in range(0,record['n'],100000):
                flags=h[1].read(rows=np.arange(start,min(start+100000,record['n'])),
                               columns=['route','supported','ambiguous','outside_train_precision','research_quality','numerical_converged'])
                for key in flags.dtype.names:
                    if key=='route':
                        for code in (-1,0,1,2):
                            counts[f'route_{code}']=counts.get(f'route_{code}',0)+int((flags[key]==code).sum())
                    else:
                        counts[key]=counts.get(key,0)+int(flags[key].sum())
        assert counts['supported']==record['supported'] and sum(counts[f'route_{i}'] for i in (-1,0,1,2))==record['n']
        bounds=config['bounds_deg']
        rows.append(dict(field=record['field'],name=field['name'],config=field['config'],
                         ra_min=bounds[0],dec_min=bounds[1],ra_max=bounds[2],dec_max=bounds[3],
                         input_rows=field['parent_records'],unique_rows=record['n'],
                         predicted=record['n']-counts['route_-1'],unpredicted_reason='north_no_native_model' if record['field']==13 else '',
                         effective_masked_area_sqdeg=None,**counts))
    assert sum(r['unique_rows'] for r in rows)==9390518
    assert sum(r['predicted'] for r in rows)==8899334
    csv_rows(TABLES/'C_application_fields.csv',rows)
    return rows


def flows(t):
    rows=[]
    masks=dict(photometric=t['photometric_eligible'],no_identity_conflict=~t['identity_conflict'],
               no_nonstellar_risk=~t['nonstellar_risk'],new_geometry=t['split']<3)
    keep=np.ones(len(t['identity']),bool)
    for name,mask in masks.items():
        before=int(keep.sum());keep &= mask
        rows.append(dict(stage=name,input=before,pass_n=int(keep.sum()),sequential_loss=before-int(keep.sum()),
                         independent_failure=int((~mask).sum()),definition='independent failures overlap'))
    for name in ('class_eligible','fit_eligible','parallax_eligible'):
        rows.append(dict(stage=name,input=len(keep),pass_n=int((keep&t[name]).sum()),
                         sequential_loss=None,independent_failure=int((~t[name]).sum()),definition='parallel independent eligibility'))
    csv_rows(TABLES/'A_selection_flow.csv',rows)
    csv_rows(TABLES/'A_existing_source_flow.csv',[
        dict(source=fn,**{k:v for k,v in json.loads((ROOT/'manifests'/fn).read_text()).items() if isinstance(v,(int,float,str))})
        for fn in ('baseline_v1_4_source_evidence.json','baseline_v1_4_assembly.json','baseline_v1_4_extraction_complete.json')])


def morphology(t):
    rows=[]
    clean=t['photometric_eligible']&~t['identity_conflict']&~t['nonstellar_risk']
    selections={'assembled':np.ones(len(clean),bool),'photometric_eligible':t['photometric_eligible']}
    for split,label in [(0,'train'),(1,'cal'),(2,'test'),(3,'quarantined')]:
        selections[label]=t['split']==split
        for key in ('class_train_eligible','fit_eligible','class_eligible','parallax_eligible'):
            selections[label+':'+key]=(t['split']==split)&t[key]&clean
        selections[label+':chosen']=(t['split']==split)&(t['label_source']>0)&clean
    for key,mask in selections.items():
        for kind in np.unique(t['TYPE']):
            rows.append(dict(sample=key,TYPE=kind,n=int((mask&(t['TYPE']==kind)).sum())))
    csv_rows(TABLES/'morphology.csv',rows)


def main():
    require_compute()
    TABLES.mkdir(parents=True,exist_ok=True)
    refs=OUT/'references.npz'
    audited(refs,json.loads((OUT/'assembly.json').read_text())['reference_sha256'])
    keys=['identity','ls_id','source_id','targetid','RA','DEC','EBV','cell','split','class_label','class_eligible',
          'class_train_eligible','fit_eligible','photometric_eligible','identity_conflict','nonstellar_risk',
          'historical_identity','zero_id_match','match_epoch_known','match_separation_arcsec',
          'label_source','ref_distance','sigma_mu','specdis_distance','spdist_distance','specdis_sigma_mu',
          'spdist_sigma_mu','specdis_valid','spdist_valid','anchor_astrometry','p_corr','gaia_parallax_error',
          'parallax_eligible','optical_error','B','C','i_usable','sigma_g','sigma_r','sigma_z','sigma_i','sigma_w1','sigma_w2',
          'w1_measurement_state','w2_measurement_state','rvs_parameter_quality',*SPECTRAL,
          'rvs_logg_err','rvs_teff_err','rvs_vsini','rvs_sn_r']
    keys=list(dict.fromkeys(keys))
    with np.load(refs) as z:
        t={k:z[k] for k in keys}
    t['class_label']=np.where(t['class_eligible'],t['class_label'],-1)
    parent=load(audited(OUT/'diagnostics/parent.npz'))
    # Recover parent sky and dust from its frozen owner row, without a new sky scan.
    for field in np.unique(parent['field']):
        m=parent['field']==field
        with np.load(ROOT/'data/baseline_v0'/f'parent_{field:02}.npz') as z:
            for key,new in [('ra','RA'),('dec','DEC'),('ebv','EBV')]:
                if new not in parent:
                    parent[new]=np.full(len(parent['identity']),np.nan)
                parent[new][m]=z[key][parent['parent_row'][m]]
    raw=recover_raw(t,parent)
    t=enrich_raw(t,raw);parent=enrich_raw(parent,raw)
    morphology(t);flows(t)
    range_tables(t,parent)
    app=application()
    models=[]
    for name in CFG['models']:
        models.append(model_table(name,t))
    q=load(DEST/'chosen_expanded_A.npz')
    test=q['split']==2
    assert int((test&np.isfinite(q['ref_chosen'])).sum())==47756
    assert int((test&q['parallax_eligible']).sum())==49327
    assert int((test&q['parallax_eligible']&(q['p_corr']<0)).sum())==4139
    atomic_npz(DEST/'parent.npz',**parent)
    atomic_npz(DEST/'population.npz',**{k:t[k] for k in ['identity','RA','DEC','cell','split','TYPE','historical_identity','photometric_eligible','class_eligible','fit_eligible','label_source','parallax_eligible','r','B','C','gr','r0']})
    save_json(TABLES/'snapshot.json',dict(config=CFG,models=models,application_fields=app,inputs=INPUTS,
              unified_schema='chosen_model is one row per identity; sources_model is normalized many-source view keyed by chosen_view_row.',
              job_id=os.getenv('SLURM_JOB_ID'),counts=dict(assembled=len(t['identity']),test_chosen=47756,test_gaia=49327)))
    save_json(DEST/'collection_complete.json',dict(snapshot_sha256=sha(TABLES/'snapshot.json'),job_id=os.getenv('SLURM_JOB_ID')))
    print('COMPLETE v2 collection',flush=True)


if __name__=='__main__':
    main()
