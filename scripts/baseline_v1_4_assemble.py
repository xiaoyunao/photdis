#!/usr/bin/env python3
"""Assemble independent evidence and freeze identities before any model scores."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.deps')]
import fitsio
import numpy as np
from photdis.assembly_v1_4 import representative_rows,duplicate_mask
from photdis.reference_v1_4 import unique_join,pair_keys,spatial_split
from photdis.source_evidence_v1_4 import vac_evidence
from photdis.gaia import join_gaia,derive_gaia_labels
from photdis.stage1 import require_compute,write_json,galactic
from baseline_v1_4_extract import sha,atomic_npz
from baseline_v1_4_source_audit import VAC_COLUMNS
OUT=ROOT/'data/baseline_v1_4'


def load(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def audit_cells(t,mask):
    result={}
    for name,lo,hi in [('bright',-np.inf,18),('18_20',18,20),('20_22',20,22),('22_25',22,25),('25_plus',25,np.inf)]:
        for wise,m in [('all',np.ones(len(mask),bool)),('no_W1',~t['B'])]:
            take=mask&m&(t['r']>=lo)&(t['r']<hi)
            result[name+':'+wise]=dict(n=int(take.sum()),classes={str(k):int((take&t['class_eligible']&(t['class_label']==k)).sum()) for k in [0,1,2]},
                distance=int((take&(t['label_source']>0)).sum()),specdis=int((take&t['specdis_valid']).sum()),spdist=int((take&t['spdist_valid']).sum()),
                gaia=int((take&t['parallax_eligible']).sum()),negative=int((take&t['parallax_eligible']&(t['p_corr']<0)).sum()),i=int((take&t['i_usable']).sum()))
    return result


def parent_coverage(native):
    parent=load(ROOT/'data/baseline_v1_1/parent_sample.npz');n=len(parent['identity']);lsid=np.zeros(n,'i8')
    for field in np.unique(parent['field']):
        use=parent['field']==field
        raw=fitsio.read(str(ROOT/f'data/baseline_v0/catalog_{field:02}.fits'),columns=['identity','ls_id'],rows=parent['parent_row'][use])
        assert np.array_equal(raw['identity'],parent['identity'][use]);lsid[use]=raw['ls_id']
    ix=unique_join(lsid,native['ls_id']);matched=ix>=0;safe=np.maximum(ix,0)
    parent['ls_id']=lsid;parent['native_match']=matched
    for k in ['i_usable','i0','sigma_i','w1_measurement_state','w2_measurement_state']:
        a=native[k][safe].copy()
        a[~matched]=False if a.dtype.kind=='b' else (np.nan if a.dtype.kind=='f' else -1)
        parent[k]=a
    atomic_npz(OUT/'parent_coverage.npz',**parent)
    groups={'all':np.ones(n,bool),'no_W1':~parent['B']}
    for key,edges in [('r',[0,18,20,22,25,np.inf]),('optical_error',[0,.03,.1,np.inf]),('gr',[-np.inf,.5,1,np.inf])]:
        value=parent['g0']-parent['r0'] if key=='gr' else parent[key]
        for lo,hi in zip(edges[:-1],edges[1:]):
            m=(value>=lo)&(value<hi);groups[f'{key}:{lo}:{hi}']=m;groups[f'{key}:{lo}:{hi}:no_W1']=m&~parent['B']
    report={}
    for name,take in groups.items():
        w=parent['sample_weight'][take];total=w.sum();i=parent['i_usable'][take]
        report[name]=dict(n=int(take.sum()),weight_sum=float(total),matched=int(matched[take].sum()),i_count=int(i.sum()),
            weighted_i_fraction=float(w@i/total) if total else None,
            wise={band:{str(state):dict(n=int((parent[band+'_measurement_state'][take]==state).sum()),
                weighted_fraction=float(w@(parent[band+'_measurement_state'][take]==state)/total) if total else None) for state in [-1,0,1,2,3,4]} for band in ['w1','w2']})
    write_json(OUT/'parent_coverage.json',dict(groups=report,source_sha256=sha(ROOT/'data/baseline_v1_1/parent_sample.npz'),output_sha256=sha(OUT/'parent_coverage.npz')))
    return dict(n=n,matched=int(matched.sum()),original_weights_unchanged=True)


def main():
    require_compute()
    if (OUT/'assembly.json').exists():raise FileExistsError('Frozen assembly exists; do not overwrite')
    for p,h in json.loads((OUT/'freeze.json').read_text())['hashes'].items():assert sha(ROOT/p)==h
    for name,key in [('sources','source_sha256'),('specdis_raw','sha256'),('spdist_raw','sha256')]:
        assert sha(OUT/(name+'.npz'))==json.loads((OUT/(name+'.json')).read_text())[key]
    assert sha(OUT/'source_evidence.npz')==json.loads((OUT/'source_evidence.json').read_text())['output_sha256']
    for name in ['native_photometry','zero_match']:
        assert sha(OUT/(name+'.npz'))==json.loads((OUT/(name+'.json')).read_text())['sha256']
    assert sha(OUT/'zero_photometry.npz')==json.loads((OUT/'zero_match.json').read_text())['photometry_sha256']
    native=load(OUT/'native_photometry.npz');extra=load(OUT/'zero_photometry.npz');zm=load(OUT/'zero_match.npz')
    new=~np.isin(extra['ls_id'],native['ls_id']);native={k:np.concatenate([v,extra[k][new]]) for k,v in native.items()}
    parent=parent_coverage(native)
    print('parent matched',parent,flush=True)
    source=load(OUT/'sources.npz');evidence=load(OUT/'source_evidence.npz')
    assert np.array_equal(source['original_row'],evidence['original_row'])
    rep=representative_rows(source['identity'],source['rvs_primary'],source['rvs_sn_r'])
    sid=source['source_id'][rep]
    # Ambiguous multiple native objects with a single Gaia ID are quarantined.
    positive_native=np.flatnonzero(native['source_id']>0)
    native_multi_source_ids=np.unique(native['source_id'][positive_native][duplicate_mask(native['source_id'][positive_native])])
    match=unique_join(sid,native['source_id'][positive_native]);ok=match>=0
    match[ok]=positive_native[match[ok]];match[sid<=0]=-1
    zjoin=unique_join(source['identity'][rep],zm['identity']);zok=(zjoin>=0)&(sid<=0)
    q=np.flatnonzero(zok);valid=zm['candidate_row'][zjoin[q]]>=0;q=q[valid]
    extraids=extra['ls_id'][zm['candidate_row'][zjoin[q]]]
    match[q]=unique_join(extraids,native['ls_id'])
    known_epoch=np.ones(len(rep),bool);separation=np.full(len(rep),np.nan)
    known_epoch[q]=~zm['unknown_epoch'][zjoin[q]];separation[q]=zm['separation_arcsec'][zjoin[q]]
    print('source representatives',len(rep),'matched',int((match>=0).sum()),flush=True)
    # Positive-ID VAC-only sources get native evidence too. Exact target/source
    # uniqueness remains mandatory for each distance label.
    vacs={};vacstats={};extras=[];extra_targets=[]
    source_pairs=np.unique(pair_keys(source['rvs_targetid'],source['source_id']))
    for kind,cols in VAC_COLUMNS.items():
        with np.load(OUT/(kind+'_raw.npz')) as z:v={k:z[k] for k in cols}
        vacs[kind]=v
        keys=pair_keys(v['targetid'],v['source_id']);dup=duplicate_mask(keys)
        missing_pair=unique_join(keys,source_pairs)<0
        missing_sid=~np.isin(v['source_id'],source['source_id'])
        usable_sid=(v['source_id']>0)&(v['source_id']!=999999)
        # Pair-level absent rows remain separate audit statistics; an existing MWS
        # identity can receive a unique-source VAC below, with provenance recorded.
        extras.append(v['source_id'][missing_sid&usable_sid]);extra_targets.append(v['targetid'][missing_sid&usable_sid])
        vacstats[kind]=dict(rows=len(keys),duplicate_pair_rows=int(dup.sum()),zero_or_invalid_id_rows=int((~usable_sid).sum()),
            vac_only_pair_rows=int(missing_pair.sum()),vac_only_source_rows=int((missing_sid&usable_sid).sum()))
    xs=np.concatenate(extras);xt=np.concatenate(extra_targets)
    xs,first=np.unique(xs,return_index=True);xt=xt[first]
    xm=unique_join(xs,native['source_id'][positive_native]);keep=xm>=0;xs,xt,xm=xs[keep],xt[keep],positive_native[xm[keep]]
    take=match>=0;sr=rep[take];ni=np.r_[match[take],xm];n=len(ni);nsource=len(sr)
    t={k:v[ni] for k,v in native.items()};del native,extra
    t['source_row']=np.r_[sr,np.full(len(xs),-1,'i8')]
    t['source_identity']=np.r_[source['identity'][sr],xs]
    t['identity']=np.where(t['source_id']>0,t['source_id'],-t['ls_id'])
    t['mws_source_id']=np.r_[source['source_id'][sr],np.zeros(len(xs),'i8')]
    t['targetid']=np.r_[source['rvs_targetid'][sr],xt]
    t['zero_id_match']=np.r_[source['source_id'][sr]<=0,np.zeros(len(xs),bool)]
    t['match_epoch_known']=np.r_[known_epoch[take],np.ones(len(xs),bool)]
    t['match_separation_arcsec']=np.r_[separation[take],np.full(len(xs),np.nan)]
    # Preserve all MWS selection/provenance columns, linked to immutable source rows.
    for k,v in source.items():
        if k.startswith(('target_','rvs_')):
            fill='' if v.dtype.kind in 'SU' else (np.nan if v.dtype.kind=='f' else 0)
            t[k]=np.concatenate([v[sr],np.full(len(xs),fill,dtype=v.dtype)])
    for k,v in evidence.items():
        if k in ['identity','source_id','original_row','cell','split'] or k.startswith(('specdis_','spdist_')):continue
        fill=np.nan if v.dtype.kind=='f' else (-1 if k=='class_label' else 0)
        t[k]=np.concatenate([v[sr],np.full(len(xs),fill,dtype=v.dtype)])
    # Matched zero-ID sources that acquire Gaia identities and VAC-only identities
    # need fresh exact-ID astrometry; all other rows reuse the audited MWS payload.
    ga_rows=np.flatnonzero((np.arange(n)>=nsource)|t['zero_id_match'])
    if len(ga_rows):
        ga=join_gaia(t['source_id'][ga_rows],'/global/cfs/cdirs/cosmo/data/gaia/dr3/healpix')
        for k,v in derive_gaia_labels(ga).items():t[k][ga_rows]=v
        t['gaia_parallax_error'][ga_rows]=ga['parallax_error']
    for kind,v in vacs.items():
        # First exact original pair (including valid zero SOURCE_ID pairs), then
        # recover only a unique positive SOURCE_ID if its TARGETID differs.
        j=unique_join(pair_keys(t['targetid'],t['mws_source_id']),pair_keys(v['targetid'],v['source_id']))
        fallback=(j<0)&(t['source_id']>0)
        bysource=unique_join(t['source_id'],v['source_id']);j[fallback]=bysource[fallback]
        t[kind+'_source_only_join']=fallback&(j>=0)
        # Existing quality recipe requires positive pair IDs; assess each unique
        # raw row using a synthetic unique positive key, with true provenance kept.
        q=np.arange(len(v['targetid']),dtype='i8')+1
        synthetic=dict(v,targetid=q,source_id=q)
        target=np.where(j>=0,j+1,0)
        values=vac_evidence(target,target,synthetic,kind)
        values[kind+'_row']=j
        t.update(values)
        vacstats[kind]['joined_reference_rows']=int((j>=0).sum())
        vacstats[kind]['source_only_join_rows']=int(t[kind+'_source_only_join'].sum())
        vacstats[kind]['quality_passing_rows']=int(t[kind+'_valid'].sum())
    del vacs,evidence
    bounds=json.loads((OUT/'freeze.json').read_text())['legacy_bounds']
    cell,split=spatial_split(t['RA'],t['DEC'],bounds)
    source_split=source['split'][sr];source_cell=source['cell'][sr]
    bad_geometry=(source_split!=split[:nsource])|(source_cell!=cell[:nsource])
    split[:nsource][bad_geometry]=3
    native_collision=duplicate_mask(t['identity'])|duplicate_mask(t['ls_id'])
    t['identity_conflict']=t['class_identity_conflict']|t['target_identity_conflict']|native_collision
    t['nonstellar_risk']=t['redrock_nonstellar_evidence']|t['contaminant_candidate']
    old=load(ROOT/'data/baseline_v0/references.npz')
    historical=np.isin(t['identity'],old['identity'])|np.isin(t['ls_id'],old['ls_id'])|np.isin(t['source_identity'],old['identity'])
    oldtarget=np.zeros(len(old['identity']),'i8');m=old['desi_match']&~old['desi_match_ambiguous']
    oldtarget[m]=np.char.strip(old['desi_object_id'][m].astype('U')).astype('i8')
    historical|=(t['targetid']>0)&np.isin(t['targetid'],oldtarget[oldtarget>0])
    for path in sorted((ROOT/'data/baseline_v0').glob('parent_*.npz')):
        with np.load(path) as z:historical|=np.isin(t['identity'],z['identity'])|np.isin(t['ls_id'],z['ls_id'])
    split[historical|t['identity_conflict']]=3
    t['historical_identity']=historical;t['cell']=cell;t['split']=split;t['field']=cell
    b=galactic(t['RA'],t['DEC'])[1]
    t['photometric_eligible']=t['grz_base']&t['grz_clean']&(abs(b)>30)&(t['EBV']>=0)&(t['EBV']<.03)&(t['RELEASE']==11010)
    t['B']=t['w1_clean']&(t['snr_w1']>=5);t['C']=t['B']&t['w2_clean']&(t['snr_w2']>=5)
    t['optical_error']=np.maximum.reduce([t[k] for k in ['sigma_g','sigma_r','sigma_z']])
    clean=t['photometric_eligible']&~t['identity_conflict']&~t['nonstellar_risk']
    t['class_eligible']=clean&(t['class_label']>=0)
    # Class qualification is independent of distance, FeH, and precise errors.
    t['class_train_eligible']=t['class_eligible'].copy()
    ga=clean&t['anchor_astrometry']&(t['p_corr']>0)
    sp=clean&t['specdis_valid']
    t['label_source']=np.where(ga,1,np.where(sp,2,0)).astype('i1')
    t['ref_distance']=np.where(ga,1/np.maximum(t['p_corr'],1e-10),np.where(sp,t['specdis_distance'],np.nan))
    t['sigma_mu']=np.where(ga,5/np.log(10)*t['gaia_parallax_error']/np.maximum(t['p_corr'],1e-10),t['specdis_sigma_mu'])
    t['Mg']=t['g0']-10-5*np.log10(t['ref_distance'])
    t['fit_eligible']=(t['label_source']>0)&(t['optical_error']<.03)&np.isfinite(t['Mg'])
    t['parallax_eligible']=clean&t['astro_quality']&t['zpt_valid']&np.isfinite(t['p_corr'])&(t['gaia_parallax_error']>0)
    t['weak_parallax_quality']=t['parallax_eligible']&(t['p_corr']/t['gaia_parallax_error']<5)
    # Preserve collision records in audit counts; frozen table has one row per
    # canonical identity, with all colliding identities excluded, never relabeled.
    _,first=np.unique(t['identity'],return_index=True)
    t={k:v[first] for k,v in t.items()}
    checks=dict(unique_identity=len(np.unique(t['identity']))==len(t['identity']),
        unique_eligible_native=len(np.unique(t['ls_id'][t['split']<3]))==int((t['split']<3).sum()),
        historical_excluded=not bool(np.any(t['historical_identity']&(t['split']<3))),
        conflicting_excluded=not bool(np.any(t['identity_conflict']&(t['split']<3))),
        class_independent_distance=np.array_equal(t['class_eligible'],t['class_train_eligible']),
        parent_all26000=parent['n']==26000 and parent['matched']==26000,
        has_new_cal=bool(np.any(t['class_eligible']&(t['split']==1))),has_new_test=bool(np.any(t['class_eligible']&(t['split']==2))))
    for kind in ['specdis','spdist']:
        good=t[kind+'_valid']
        checks[kind+'_finite_positive_distance_error']=bool(np.all(np.isfinite(t[kind+'_distance'][good])&(t[kind+'_distance'][good]>0)&np.isfinite(t[kind+'_sigma_mu'][good])&(t[kind+'_sigma_mu'][good]>0)))
    checks['signed_gaia_finite_error']=bool(np.all(np.isfinite(t['p_corr'][t['parallax_eligible']])&(t['gaia_parallax_error'][t['parallax_eligible']]>0)))
    report=dict(candidate_rows=n,unique_reference_rows=len(first),matched_mws=nsource,unmatched_mws_identities=int((match<0).sum()),ambiguous_native_gaia_ids=len(native_multi_source_ids),vac_only_native=len(xs),native_collision_rows=int(native_collision.sum()),
        geometry_disagreement_rows=int(bad_geometry.sum()),historical_rows=int(historical.sum()),vac_audit=vacstats,parent=parent,
        new_split_counts={str(k):int((t['split']==k).sum()) for k in range(4)},
        coverage={str(k):audit_cells(t,t['split']==k) for k in range(4)},checks=checks,
        matching=json.loads((OUT/'zero_match.json').read_text()),
        limitation='Reference-selected sky, not representative population fractions. Unknown epoch matches flagged. No model scores yet;22–25 validation limited to actual evidence.',
        class_training='Independent of distance, FeH and precise optical errors; photometric quality still required',
        historical_reference_sha256=sha(ROOT/'data/baseline_v0/references.npz'))
    atomic_npz(OUT/'references.npz',**t)
    report['reference_sha256']=sha(OUT/'references.npz')
    report['code_sha256']={p:sha(ROOT/p) for p in ['scripts/baseline_v1_4_assemble.py','src/photdis/assembly_v1_4.py']}
    write_json(OUT/'assembly.json',report)
    print('ASSEMBLY',json.dumps({k:v for k,v in report.items() if k not in ['coverage']}),flush=True)
    assert all(checks.values()),checks

if __name__=='__main__':main()
