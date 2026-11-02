#!/usr/bin/env python3
"""Exact-identity K-population evidence; no distance model scores inspected here."""
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'.deps')]
import fitsio
import numpy as np
from photdis.giant_candidate import observable_masks
from photdis.holdouts import load_reference_catalogs, annotate_holdouts
from photdis.stage1 import require_compute

OUT = Path('data/kgiant_distance_v1')
VAC = [Path('data/giant_vac_v3/same_star.fits'), Path('data/giant_ranked_vac_v1/same_star.fits')]
SP = Path('/global/cfs/cdirs/desi/public/dr1/vac/dr1/mws-spdist/v1.0/mws-iron-SPDist-240613.fits')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def text(x):
    return (x.decode() if isinstance(x, bytes) else str(x)).strip()


def distribution(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if not len(v): return {'n': 0}
    return {'n': len(v), 'q0_16_50_84_100': np.quantile(v, [0,.16,.5,.84,1]).tolist(),
            'nmad': float(1.4826*np.median(abs(v-np.median(v))))}


def write_json(path, value):
    def convert(x):
        if isinstance(x, dict): return {str(k): convert(v) for k,v in x.items()}
        if isinstance(x, (list, tuple)): return [convert(v) for v in x]
        if isinstance(x, np.ndarray): return convert(x.tolist())
        if isinstance(x, np.generic): return convert(x.item())
        if isinstance(x, float) and not np.isfinite(x): return None
        return x
    Path(path).write_text(json.dumps(convert(value), indent=2, allow_nan=False)+'\n')


def main():
    require_compute(); OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'inventory.json').exists(): raise FileExistsError(OUT/'inventory.json')
    labelpath = Path('data/giant_information_v1/labels.npz')
    with np.load(labelpath) as f: t = {k:f[k] for k in f.files}
    cfg = json.loads(Path('configs/giant_information_v1.json').read_text())
    masks = observable_masks(t,cfg)
    k = masks['optical'] & (t['label']==1) & (t['desi_teff']>=4200) & (t['desi_teff']<5000)
    indices = np.flatnonzero(k); q = {name:val[indices] for name,val in t.items()}
    q['original_row'] = indices; q['w1_qualified'] = masks['w1'][indices]
    selected = np.zeros(len(t['source_id']),bool)
    folds = json.loads(Path('data/giant_wide_v1/results.json').read_text())['folds']
    for fold in folds:
        with np.load(f"data/giant_wide_v1/{fold['test_field']}_scores.npz") as f:
            threshold = fold['models']['B-wide']['domains']['common_K']['thresholds']['0.01']['threshold']
            selected[f['rows']] = (f['score_B-wide']>=threshold) & ~f['outside_B-wide']
    q['selected'] = selected[indices]
    assert int(q['selected'].sum())==190 and int(q['w1_qualified'].sum())==289
    assert len(np.unique(q['identity_group'])) == len(indices)
    lookup = {}; ambiguous = set()
    for path in VAC:
        for row in fitsio.read(str(path)):
            pair = (int(row['source_id']), text(row['desi_object_id']))
            if pair in lookup: ambiguous.add(pair)
            else: lookup[pair] = row
    pairs = [(int(s),text(d)) for s,d in zip(q['source_id'],q['desi_object_id'])]
    matched = [lookup.get(p) if p not in ambiguous else None for p in pairs]
    q['vac_matched'] = np.array([v is not None for v in matched])
    q['vac_ambiguous'] = np.array([p in ambiguous for p in pairs])
    columns = ['specdis_match','specdis_valid','specdis_dist_kpc','specdis_disterr_kpc',
               'specdis_binary_flag','specdis_sigma_mg','spdist_match','spdist_valid',
               'spdist_dist_kpc','spdist_sigma_mg','spdist_flag_good','spdist_qflag_input',
               'p_corr','gaia_parallax_error','astro_quality','zpt_valid','cmd_spectral_conflict']
    flags = [c for c in columns if c.endswith(('_match','_valid')) or c in
             ('spdist_flag_good','spdist_qflag_input','astro_quality','zpt_valid','cmd_spectral_conflict')]
    for c in columns:
        q[c] = np.array([v[c] if v is not None else (False if c in flags else np.nan) for v in matched])
    # Read only matching raw SPDist rows, preserving target/source pairing and duplicate flags.
    with fitsio.FITS(str(SP)) as f:
        rawids = f['SPDIST'].read(columns=['SOURCE_ID','TARGETID'])
        where = np.flatnonzero(np.isin(rawids['SOURCE_ID'],q['source_id']))
        raw = f['SPDIST'].read(columns=['SOURCE_ID','TARGETID','MG_16','MG_50','MG_84',
                                       'E_GMAG','QFLAG_COMMENTS'],rows=where)
    rawmap = {}; repeated = set()
    for row in raw:
        pair = (int(row['SOURCE_ID']),str(int(row['TARGETID'])))
        if pair in rawmap: repeated.add(pair)
        else: rawmap[pair] = row
    for dest,c in [('sp_mg16','MG_16'),('sp_mg50','MG_50'),('sp_mg84','MG_84'),('sp_gaia_g_error','E_GMAG')]:
        q[dest] = np.array([rawmap[p][c] if p in rawmap and p not in repeated else np.nan for p in pairs])
    q['sp_comment'] = np.array([text(rawmap[p]['QFLAG_COMMENTS']) if p in rawmap else 'no_exact_pair' for p in pairs])
    q['sp_duplicate'] = np.array([p in repeated for p in pairs])
    for kind in ('spdist','specdis'):
        valid = q[kind+'_valid'].astype(bool)
        q[kind+'_mg'] = np.full(len(indices),np.nan)
        q[kind+'_mg'][valid] = q['g0'][valid]-10-5*np.log10(q[kind+'_dist_kpc'][valid])
    q['spdist_sigma_mu'] = np.sqrt(((q['sp_mg84']-q['sp_mg16'])/2)**2+q['sp_gaia_g_error']**2+.167**2)
    sp = q['spdist_valid']; assert np.allclose(np.hypot(q['spdist_sigma_mu'][sp],q['sigma_g'][sp]),q['spdist_sigma_mg'][sp],rtol=1e-5)
    frac = q['specdis_disterr_kpc']/q['specdis_dist_kpc']
    q['specdis_fractional_error'] = frac
    q['specdis_sigma_mu'] = 5/np.log(10)*np.sqrt(np.log1p(frac**2))
    q['specdis_sigma_mu_linear'] = 5/np.log(10)*frac
    q['delta_mu_sp_minus_spec'] = np.full(len(indices),np.nan)
    both = q['spdist_valid'] & q['specdis_valid']
    q['delta_mu_sp_minus_spec'][both] = 5*np.log10(q['spdist_dist_kpc'][both]/q['specdis_dist_kpc'][both])
    refs = load_reference_catalogs('data/reference')
    hold = annotate_holdouts(refs,q['source_id'],q['ra'],q['dec'])
    q['cluster_group'] = hold['cluster_member_group']
    names = json.loads(Path('data/reference/holdout_group_names.json').read_text())
    result = {'n_optical_K':len(indices), 'n_w1_K':int(q['w1_qualified'].sum()),
              'n_selected_K':int(q['selected'].sum()),'vac_ambiguous':int(q['vac_ambiguous'].sum()),
              'populations':{},'teacher_covariance':'Unknown shared Gaia/spectral systematics; not independent measurements.',
              'source_sha256':{str(p):sha(p) for p in [labelpath,*VAC,SP,Path('data/giant_wide_v1/results.json')]}}
    parameters = ['g0','r','desi_teff','desi_logg','desi_snr','feh_corrected']
    q['gr0'] = q['g0']-q['r0']; q['rz0'] = q['r0']-q['z0'];q['zw10'] = q['z0']-q['w10']
    parameters += ['gr0','rz0','zw10']
    for pop,mask in [('optical_all',np.ones(len(indices),bool)),('w1_all',q['w1_qualified']),('selected',q['selected'])]:
        r = {'n':int(mask.sum()),'matched':int((mask&q['vac_matched']).sum()),
             'both_valid':int((mask&both).sum()),'paired_delta_mu':distribution(q['delta_mu_sp_minus_spec'][mask&both])}
        for kind in ('spdist','specdis'):
            valid=mask&q[kind+'_valid'];r[kind]={'n_valid':int(valid.sum()),'sigma_mu':distribution(q[kind+'_sigma_mu'][valid]),
                    'distance_kpc':distribution(q[kind+'_dist_kpc'][valid]),'Mg':distribution(q[kind+'_mg'][valid])}
        r['specdis']['fractional_error']=distribution(frac[mask&q['specdis_valid']])
        r['specdis']['error_gt_20pct']=int((mask&q['specdis_valid']&(frac>.2)).sum())
        r['specdis']['error_gt_50pct']=int((mask&q['specdis_valid']&(frac>.5)).sum())
        r['sp_valid_vs_rest']={}
        for group,within in [('valid',sp),('rest',~sp)]:
            r['sp_valid_vs_rest'][group]={c:distribution(q[c][mask&within&(q['abundance_calibration_valid'] if c=='feh_corrected' else True)]) for c in parameters}
        r['quality_failures_overlapping']={
            'no_unique_vac_pair':int((mask&~q['vac_matched']).sum()),
            'sp_no_unique_match':int((mask&~q['spdist_match']).sum()),
            'sp_duplicate_raw_pair':int((mask&q['sp_duplicate']).sum()),
            'sp_bad_parameter_flag':int((mask&q['spdist_match']&~q['spdist_flag_good']).sum()),
            'sp_outside_input_domain':int((mask&q['spdist_match']&~q['spdist_qflag_input']).sum()),
            'sp_other_invalid':int((mask&q['spdist_match']&q['spdist_flag_good']&q['spdist_qflag_input']&~sp).sum()),
            'spec_no_match':int((mask&~q['specdis_match']).sum()),
            'spec_binary_flag_not_single':int((mask&q['specdis_match']&(q['specdis_binary_flag']!=1)).sum()),
            'spec_nonpositive_or_large_error':int((mask&q['specdis_match']&~(np.isfinite(frac)&(frac>0)&(frac<1))).sum())}
        comments=q['sp_comment'][mask&~sp]; r['sp_invalid_comments']={c:int((comments==c).sum()) for c in np.unique(comments)}
        r['paired_strata']={}
        for c,edges in [('desi_teff',[4200,4600,4800,5000]),('desi_logg',[0,1.8,2.2,2.6,3.2]),('feh_corrected',[-5,-1.5,-1,-.5,1]),('desi_snr',[0,20,60,100,1e9])]:
            r['paired_strata'][c]=[{'bounds':[lo,hi],**distribution(q['delta_mu_sp_minus_spec'][mask&both&(q[c]>=lo)&(q[c]<hi)&(q['abundance_calibration_valid'] if c=='feh_corrected' else True)])} for lo,hi in zip(edges[:-1],edges[1:])]
        members=mask&(q['cluster_group']>0)
        r['independent_members']={str(g):{'n':int((mask&(q['cluster_group']==g)).sum()),'metadata':names.get(str(g))} for g in np.unique(q['cluster_group'][members])}
        geom=mask&q['astro_quality']&q['zpt_valid']&(q['p_corr']>0)&(q['gaia_parallax_error']>0)&(q['p_corr']/q['gaia_parallax_error']>=20)
        r['high_snr_gaia_n']=int(geom.sum());r['high_snr_gaia_source_ids']=q['source_id'][geom].astype(int).tolist()
        result['populations'][pop]=r
    np.savez_compressed(OUT/'population.npz',**q)
    columns=['source_id','desi_object_id','field_index','g0','r','gr0','rz0','zw10','desi_teff','desi_logg','desi_snr','feh_corrected','abundance_calibration_valid','spdist_match','spdist_valid','spdist_flag_good','spdist_qflag_input','sp_duplicate','sp_comment','spdist_dist_kpc','spdist_sigma_mu','sp_mg16','sp_mg50','sp_mg84','sp_gaia_g_error','specdis_match','specdis_valid','specdis_binary_flag','specdis_dist_kpc','specdis_disterr_kpc','specdis_fractional_error','specdis_sigma_mu','delta_mu_sp_minus_spec','cluster_group']
    with (OUT/'selected190_evidence.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(columns)
        for i in np.flatnonzero(q['selected']): writer.writerow([text(q[c][i]) for c in columns])
    result['population_sha256']=sha(OUT/'population.npz');result['script_sha256']=sha(__file__)
    write_json(OUT/'inventory.json',result)
    print(json.dumps({k:{c:v[c] for c in ('n','both_valid','paired_delta_mu','high_snr_gaia_n','independent_members')} for k,v in result['populations'].items()}),flush=True)


if __name__=='__main__':main()
