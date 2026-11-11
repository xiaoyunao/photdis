#!/usr/bin/env python3
"""Final real validation: method freeze, photometry, predictions, then evidence.

The reservation is conditional on thirteen already-seen footprints. New-sky and
independent-system layers retain all photometric admissions, even without Gaia.
This driver cannot open outcomes before a versioned method freeze exists.
"""
import argparse
import json
import os
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, atomic_npz
from photdis.stage1 import require_compute
from photdis.reference_v1_4 import unique_join
from photdis.v3.data import primary_channel, specdis_qualification
from photdis.gaia import join_gaia, derive_gaia_labels
from photdis.assembly_v1_4 import representative_rows
from photdis.source_evidence_v1_4 import vac_evidence
from v3_0_prepare import projected, gather
from v3_0_cohorts import INPUT_KEYS, EVIDENCE_KEYS
D = ROOT/'data/v3_0'; OLD = ROOT/'data/baseline_v1_4'


def check_freeze():
    path = D/'method_freeze.json'
    if not path.exists(): raise RuntimeError('Final method/priors/metrics/numerics freeze required before E')
    record = json.loads(path.read_text())
    for name, digest in record['hashes'].items(): assert sha(ROOT/name) == digest, name
    return sha(path)


def prepare(name, method):
    out = D/'validation'/name; out.mkdir(parents=True, exist_ok=True)
    if (out/'photometry.json').exists():
        old = json.loads((out/'photometry.json').read_text()); assert old['method_freeze'] == method
        assert sha(out/'photometry.npz') == old['sha256']; return
    paths = []; pieces = []
    if name == 'reservation':
        path = ROOT/'data/v2_4/prospective_identity_reservation.npz'; reserved = load(path); paths.append(path)
        for field in range(13):
            use = reserved['field'] == field
            source_path = D/'application_inputs'/f'field_{field:02}.npz'; t = load(source_path); paths.append(source_path)
            index = unique_join(reserved['ls_id'][use], t['ls_id']); assert (index >= 0).all()
            np.testing.assert_array_equal(t['identity'][index], reserved['identity'][use])
            selected = {k: t[k][index] for k in INPUT_KEYS+['source_id']}
            selected.update(field=np.full(len(index), field, 'i2'), design_weight=np.ones(len(index)))
            pieces.append(selected)
        assert sum(len(p['identity']) for p in pieces) == 3250
    else:
        sources = [D/'independent_scale/photometry.npz'] if name == 'independent_scale' else [D/'new_sky_photometry'/f'field_{i:02}.npz' for i in range(2)]
        for path in sources:
            t = load(path); paths.append(path); selected = {k: t[k] for k in INPUT_KEYS+['source_id', 'field']}
            selected['design_weight'] = t['design_weight'] if 'design_weight' in t else t['sample_weight']
            pieces.append(selected)
    result = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    assert (result['TYPE'] == 'PSF').all() and len(np.unique(result['identity'])) == len(result['identity'])
    ref = projected(D/'references.npz', ['identity', 'ls_id', 'role'])
    overlaps = {}
    for role in (1, 2, 3, 4):
        selected = ref['role'] == role
        overlaps[str(role)] = int((np.isin(result['identity'], ref['identity'][selected]) | np.isin(result['ls_id'], ref['ls_id'][selected])).sum())
    if any(overlaps.values()): raise RuntimeError('Final real evidence role isolation failed '+str(overlaps))
    atomic_npz(out/'photometry.npz', **result)
    save_json(out/'photometry.json', dict(method_freeze=method, sha256=sha(out/'photometry.npz'), n=len(result['identity']),
        source_hashes={str(p.relative_to(ROOT)): sha(p) for p in paths}, role_overlaps=overlaps,
        outcomes_opened=False, missing_TYPE=0, nonPSF=0, job=os.getenv('SLURM_JOB_ID'),
        caveat='Original3250is selected within13previouslyusedfields, not newsky; other layers are distinct and not pooled into an absolute truth claim.'))
    print('FINAL PHOTOMETRY', name, len(result['identity']), flush=True)


def associate(name, method):
    out = D/'validation'/name; phot_path = out/'photometry.npz'
    meta = json.loads((out/'photometry.json').read_text()); assert meta['method_freeze'] == method and sha(phot_path) == meta['sha256']
    prediction = out/'predictions/complete.json'
    if not prediction.exists(): raise RuntimeError('Complete pure-photometry prediction must precede evidence association')
    t = load(phot_path); n = len(t['identity']); marker = out/'evidence.json'
    if marker.exists():
        old = json.loads(marker.read_text()); assert old['method_freeze'] == method and sha(out/'evidence.npz') == old['sha256']; return
    # Fixed schema defaults copied without materializing the large reference table.
    evidence = {}
    with np.load(D/'references.npz') as z:
        for key in EVIDENCE_KEYS:
            a = z[key]; fill = False if a.dtype.kind == 'b' else np.nan if a.dtype.kind == 'f' else '' if a.dtype.kind in 'SU' else 0 if a.dtype.kind == 'u' else -1
            evidence[key] = np.full((n,)+a.shape[1:], fill, dtype=a.dtype)
    evidence.update(identity=t['identity'], TYPE=t['TYPE'], role=np.full(n, 5, 'i1'), historical_fit=np.zeros(n, bool))
    # Raw full MWS inventory, independent of whether v1.4 found native photometry.
    skeys = ['identity', 'source_id', 'rvs_primary', 'rvs_sn_r', 'rvs_targetid', 'rvs_feh', 'rvs_feh_err', 'rvs_teff', 'rvs_logg']
    source = projected(OLD/'sources.npz', skeys)
    representative = representative_rows(source['identity'], source['rvs_primary'], source['rvs_sn_r'])
    index = unique_join(t['source_id'], source['source_id'][representative]); source_rows = np.full(n, -1, 'i8')
    matched = (index >= 0) & (t['source_id'] > 0); source_rows[matched] = representative[index[matched]]
    raw_evidence = gather(OLD/'source_evidence.npz', source_rows,
        ['class_label', 'rvs_quality_reason', 'rvs_parameter_quality', 'redrock_nonstellar_evidence', 'class_identity_conflict', 'target_identity_conflict'])
    conflict = raw_evidence['class_identity_conflict'] | raw_evidence['target_identity_conflict']
    for key in ('class_label', 'rvs_parameter_quality', 'redrock_nonstellar_evidence'): evidence[key] = raw_evidence[key]
    evidence['class_label'][conflict] = -1; evidence['rvs_parameter_quality'][conflict] = False
    for key in ('rvs_feh', 'rvs_feh_err', 'rvs_teff', 'rvs_logg'):
        evidence[key][matched] = source[key][source_rows[matched]]
    target = np.zeros(n, 'i8'); target[matched] = source['rvs_targetid'][source_rows[matched]]
    source_audit = dict(raw_MWS_match=int(matched.sum()), raw_identity_conflicts=int(conflict.sum()))
    # Unique positive source joins also audit VAC-only records. Duplicated raw
    # source keys fail association; they are not arbitrarily picked by distance.
    for kind, columns in [('specdis', ['targetid', 'source_id', 'dist', 'disterr', 'binary_flag']),
                          ('spdist', ['targetid', 'source_id', 'g0', 'mg_50', 'mg_16', 'mg_84', 'e_gmag', 'flag_good', 'qflag_input'])]:
        raw = projected(OLD/(kind+'_raw.npz'), columns)
        from photdis.reference_v1_4 import pair_keys
        j = unique_join(pair_keys(target, t['source_id']), pair_keys(raw['targetid'], raw['source_id']))
        fallback = (j < 0) & (t['source_id'] > 0); j[fallback] = unique_join(t['source_id'][fallback], raw['source_id'])
        j[t['source_id'] <= 0] = -1
        present = j >= 0; q = np.flatnonzero(present)
        source_audit[kind+'_raw_matches'] = int(present.sum())
        source_audit[kind+'_source_only_matches'] = int((present & fallback).sum())
        if kind == 'specdis':
            spectral = matched & ~conflict & ((raw_evidence['rvs_quality_reason'].astype('i8') & ~(1 << 4)) == 0)
            distance = np.full(n, np.nan); error = distance.copy()
            distance[q] = raw['dist'][j[q]]; error[q] = raw['disterr'][j[q]]
            valid, sigma = specdis_qualification(distance, error, spectral)
            evidence['specdis_valid'] = valid; evidence['specdis_distance'] = distance; evidence['specdis_sigma_mu'] = sigma
            evidence['specdis_raw_binary_flag'][q] = raw['binary_flag'][j[q]]
        else:
            local = {k: v[j[q]] for k, v in raw.items()}; artificial = np.arange(len(q), dtype='i8')+1
            local['source_id'] = artificial; local['targetid'] = artificial
            values = vac_evidence(artificial, artificial, local, 'spdist')
            for key in ('spdist_valid', 'spdist_distance', 'spdist_sigma_mu'): evidence[key][q] = values[key]
            evidence['spdist_valid'][conflict] = False
    # Fresh signed Gaia observations. No parallax sign/SNR cut for predictive tests.
    gaia = join_gaia(t['source_id'], '/global/cfs/cdirs/cosmo/data/gaia/dr3/healpix'); labels = derive_gaia_labels(gaia)
    evidence['p_corr'] = labels['p_corr']; evidence['gaia_parallax_error'] = gaia['parallax_error']
    evidence['parallax_eligible'] = labels['astro_quality'] & labels['zpt_valid'] & labels['finite_parallax']
    evidence['nonstellar_risk'] = labels['contaminant_candidate'] | evidence['redrock_nonstellar_evidence']
    evidence['reference_stellar_eligible'] = ~evidence['redrock_nonstellar_evidence']
    evidence['primary_channel'] = primary_channel(evidence['specdis_valid'], evidence['spdist_valid'], evidence['parallax_eligible'])
    reason = np.full(n, 'qualified_reference', 'U32')
    no = evidence['primary_channel'] == 0
    reason[no] = 'reference_quality_failed'; reason[no & ~gaia['gaia_match']] = 'no_exact_Gaia_match'
    reason[no & gaia['gaia_file_missing']] = 'Gaia_tile_unavailable'; reason[no & (t['source_id'] <= 0)] = 'no_native_Gaia_ID'
    evidence.update(no_reference_reason=reason, raw_source_row=source_rows, source_identity_conflict=conflict)
    atomic_npz(out/'evidence.npz', **evidence)
    atomic_npz(out/'gaia_payload.npz', identity=t['identity'], TYPE=t['TYPE'], **gaia, **{k: v for k, v in labels.items() if k not in gaia})
    save_json(marker, dict(method_freeze=method, sha256=sha(out/'evidence.npz'), gaia_sha256=sha(out/'gaia_payload.npz'),
        photometry_sha256=sha(phot_path), prediction_sha256=sha(prediction), n=n, retained_without_reference=int(no.sum()),
        reference_counts={str(k): int((evidence['primary_channel'] == k).sum()) for k in range(4)},
        signed_Gaia=int(evidence['parallax_eligible'].sum()), negative_Gaia=int((evidence['parallax_eligible'] & (evidence['p_corr'] < 0)).sum()),
        source_audit=source_audit, zero_id_scope='No Gaia source ID retained as unvalidated; prior audited source-zero coordinate associations cannot create new independent evidence.',
        raw_source_hashes={str(p.relative_to(ROOT)): sha(p) for p in [OLD/'sources.npz', OLD/'source_evidence.npz', OLD/'specdis_raw.npz', OLD/'spdist_raw.npz']},
        once_only=True, job=os.getenv('SLURM_JOB_ID')))
    print('FINAL EVIDENCE', name, n, 'without reference', int(no.sum()), flush=True)


def main():
    require_compute(); p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'associate'])
    p.add_argument('cohort', choices=['reservation', 'new_sky', 'independent_scale']); a = p.parse_args()
    method = check_freeze(); (prepare if a.stage == 'prepare' else associate)(a.cohort, method)


if __name__ == '__main__': main()
