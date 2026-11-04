#!/usr/bin/env python3
"""Allocated independent evidence audit of frozen raw sources; no model scores.

May run while sweep extraction continues: reads immutable raw checkpoints and
writes its own source_evidence files. No native photometric admission yet.
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'.deps')]
import numpy as np
from photdis.source_evidence_v1_4 import class_evidence, gaia_input, vac_evidence
from photdis.gaia import derive_gaia_labels
from photdis.reference_v1_4 import pair_keys
from photdis.stage1 import require_compute, write_json
from baseline_v1_4_extract import sha, atomic_npz

OUT = ROOT/'data/baseline_v1_4'
VAC_COLUMNS = dict(specdis=['targetid','source_id','dist','disterr','binary_flag'],
                   spdist=['targetid','source_id','flag_good','qflag_input','g0','e_gmag','mg_16','mg_50','mg_84'])


def main():
    require_compute()
    output, report_path = OUT/'source_evidence.npz', OUT/'source_evidence.json'
    if output.exists() or report_path.exists():
        raise FileExistsError('Source evidence audit already exists; inspect it, do not overwrite')
    freeze = json.loads((OUT/'freeze.json').read_text())
    for path, checksum in freeze['hashes'].items():
        assert sha(ROOT/path) == checksum, path
    provenance = {}
    for kind, filename, key in [('sources','sources.npz','source_sha256'),
                               ('specdis_raw','specdis_raw.npz','sha256'),
                               ('spdist_raw','spdist_raw.npz','sha256')]:
        expected = json.loads((OUT/(kind+'.json')).read_text())[key]
        assert sha(OUT/filename) == expected, filename
        provenance[filename] = expected
    base_keys = ['identity','source_id','original_row','cell','split']
    with np.load(OUT/'sources.npz') as z:
        source = {k:z[k] for k in z.files if k in base_keys or k.startswith(('rvs_', 'gaia_'))}
    ids = source['identity']; n = len(ids)
    _, first, inverse = np.unique(ids, return_index=True, return_inverse=True)
    evidence = {k:source[k] for k in base_keys}
    evidence.update(class_evidence(source))
    gaia = gaia_input(source)
    evidence.update(derive_gaia_labels(gaia))
    evidence['gaia_parallax_error'] = gaia['parallax_error']
    label = evidence['class_label']
    bits = np.zeros(len(first), 'u1')
    row_bits = np.where(label >= 0, 1 << np.maximum(label, 0), 0).astype('u1')
    np.bitwise_or.at(bits, inverse, row_bits)
    conflict = (bits & (bits - np.uint8(1))) != 0
    evidence['class_identity_conflict'] = conflict[inverse]
    target, sid = source['rvs_targetid'], source['source_id']
    positive = (target > 0) & (sid > 0)
    pairs = np.unique(pair_keys(target[positive], sid[positive]))
    targets, count = np.unique(pairs['target'], return_counts=True)
    evidence['target_identity_conflict'] = np.isin(target, targets[count > 1])
    with np.load(ROOT/'data/baseline_v0/references.npz') as old:
        evidence['historical_reference_identity'] = np.isin(ids, old['identity'])
    del pairs
    print('Raw class/Gaia and identity evidence prepared:', n, flush=True)
    for kind, columns in VAC_COLUMNS.items():
        with np.load(OUT/(kind+'_raw.npz')) as z:
            catalog = {k:z[k] for k in columns}
        evidence.update(vac_evidence(target, sid, catalog, kind))
        print(kind, 'unique pair matches', int(evidence[kind+'_match'].sum()),
              'quality-passing rows', int(evidence[kind+'_valid'].sum()), flush=True)
        del catalog
    weak = (evidence['astro_quality'] & evidence['zpt_valid']
            & np.isfinite(evidence['p_corr']) & (evidence['snr_corr'] < 5))
    evidence['weak_parallax_quality'] = weak
    metrics = dict(class_rows={str(k):int((label==k).sum()) for k in [-1,0,1,2]},
        qualified_class_no_gaia_rows=int(((label>=0)&(sid<=0)).sum()),
        class_conflict_identities=int(conflict.sum()),
        target_multi_gaia_identities=int((count>1).sum()),
        gaia_anchor_rows=int(evidence['anchor_astrometry'].sum()),
        weak_parallax_quality_rows=int(weak.sum()),
        negative_parallax_quality_rows=int((weak & (evidence['p_corr']<0)).sum()))
    split_counts = {}
    for split in range(4):
        take = source['split'] == split
        split_counts[str(split)] = dict(rows=int(take.sum()),
            identities=int((source['split'][first]==split).sum()),
            class_rows={str(k):int((take&(label==k)).sum()) for k in [-1,0,1,2]},
            gaia_anchor=int((take&evidence['anchor_astrometry']).sum()),
            weak_parallax=int((take&weak).sum()),
            specdis_valid=int((take&evidence['specdis_valid']).sum()),
            spdist_valid=int((take&evidence['spdist_valid']).sum()))
    checks = dict(row_alignment=np.array_equal(source['original_row'],np.arange(n)),
        identity_split_consistent=np.array_equal(source['split'],source['split'][first][inverse]),
        new_cal_test_excludes_historical_references=not bool(np.any(
            np.isin(source['split'],[1,2]) & evidence['historical_reference_identity'])),
        unknown_class_not_qualified=bool(np.array_equal(label>=0,evidence['rvs_parameter_quality'])))
    for kind in VAC_COLUMNS:
        valid = evidence[kind+'_valid']
        checks[kind+'_valid_has_unique_match'] = bool(np.all(evidence[kind+'_match'][valid]))
        checks[kind+'_valid_has_finite_positive_distance_and_error'] = bool(np.all(
            np.isfinite(evidence[kind+'_distance'][valid]) & (evidence[kind+'_distance'][valid]>0)
            & np.isfinite(evidence[kind+'_sigma_mu'][valid]) & (evidence[kind+'_sigma_mu'][valid]>0)))
    atomic_npz(output, **evidence)
    report = dict(rows=n, identities=len(first), metrics=metrics, split_counts=split_counts,
        checks=checks, input_sha256=provenance, output_sha256=sha(output),
        code_sha256={p:sha(ROOT/p) for p in ['scripts/baseline_v1_4_source_audit.py','src/photdis/source_evidence_v1_4.py']},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        job_id=os.environ['SLURM_JOB_ID'], hostname=socket.gethostname(),
        limitation='Raw source-row evidence, not deduplicated DR11-qualified train/cal/test references. Photometry, zero-ID/VAC-only completeness, conflicts and final eligibility still require assembly. No model scores.')
    write_json(report_path, report)
    print(json.dumps(report), flush=True)
    if not all(checks.values()):
        raise AssertionError('Source audit checks fail; inspect report before downstream assembly')


if __name__ == '__main__':
    main()
