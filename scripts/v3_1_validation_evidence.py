#!/usr/bin/env python3
"""Open only frozen same-identity external evidence after full prediction freeze."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, atomic_npz
from photdis.stage1 import require_compute
from photdis.reference_v1_4 import unique_join
from photdis.gaia import join_gaia, derive_gaia_labels, SOURCE_ID_HEALPIX_FACTOR
from photdis.v3_1.freeze import verify_freeze
from v3_1_validation_channels import associate_mws


def main():
    require_compute()
    p = argparse.ArgumentParser(); p.add_argument('--cohort',choices=['field','common'],required=True)
    a = p.parse_args(); root = ROOT/'data/v3_1'; final = root/'final_validation'
    freeze = final/'freeze.json'; digest = verify_freeze(ROOT,freeze)
    out = final/a.cohort; out.mkdir(exist_ok=True)
    photo = root/f'validation_roles/{a.cohort}_photometry.npz'
    pointers = json.loads((root/'validation_roles/references.json').read_text())
    paths = [photo,root/'validation_roles/references.json',ROOT/'scripts/v3_1_validation_evidence.py',
             ROOT/'src/photdis/gaia.py',ROOT/'src/photdis/reference_v1_4.py']
    if a.cohort == 'field':
        paths.append(ROOT/pointers['field']['path'])
    else:
        paths += [ROOT/pointers['common']['path'],ROOT/pointers['common']['membership']]
    mws_base=ROOT/'data/baseline_v1_4'
    mws_paths=[mws_base/name for name in ['sources.npz','source_evidence.npz','specdis_raw.npz','spdist_raw.npz']]
    inputs = {str(path.relative_to(ROOT)):sha(path) for path in paths}; inputs['freeze_sha256'] = digest
    inputs.update({str(path.relative_to(ROOT)):sha(path) for path in mws_paths})
    marker = out/'complete.json'
    if marker.exists():
        record = json.loads(marker.read_text()); assert record['inputs'] == inputs
        for name, checksum in record['outputs'].items():
            assert sha(out/name) == checksum
        print('VERIFIED EXISTING EVIDENCE',a.cohort,record['n'],flush=True)
        return
    started = out/'started.json'
    if started.exists():
        assert json.loads(started.read_text())['inputs'] == inputs
    else:
        save_json(started,dict(inputs=inputs,utc=datetime.now(timezone.utc).isoformat(),
            job=os.getenv('SLURM_JOB_ID'),scope='Start once-only evidence opening against immutable prediction freeze; resumable, no refit.'))
    t = load(photo); n = len(t['identity'])
    assert (t['TYPE'] == 'PSF').all() and (t['RELEASE'] == 11010).all()
    evidence = {k:t[k] for k in ['identity','source_id','ls_id','TYPE','role','cell']}
    evidence.update(class_label=np.full(n,-1,'i1'),feh=np.full(n,np.nan),feh_error=np.full(n,np.nan),
        feh_valid=np.zeros(n,bool),distance_kpc=np.full(n,np.nan),distance_error_kpc=np.full(n,np.nan),
        distance_available=np.zeros(n,bool),distance_precise=np.zeros(n,bool),
        scale_system=np.full(n,-1,'i2'),membership_probability=np.full(n,np.nan))
    if a.cohort == 'field':
        raw = load(paths[-1]); j = unique_join(t['identity'],raw['identity'])
        assert (j >= 0).all()
        reference = {k:v[j] for k,v in raw.items()}
        np.testing.assert_array_equal(reference['ls_id'],t['ls_id'])
        assert (reference['TYPE'] == 'PSF').all()
        atomic_npz(out/'astronn_apogee.npz',**reference)
        for name in ['distance_kpc','distance_error_kpc','distance_available','distance_precise']:
            evidence[name] = reference[name]
        # Uncertain/ambiguous/nonstellar labels are unknown, never class2.
        quality = reference['quality'] & reference['is_stellar'] & ~reference['known_contaminant']
        dwarf = quality & reference['label_dwarf']; giant = quality & reference['label_giant']
        assert not (dwarf & giant).any()
        evidence['class_label'][dwarf] = 0; evidence['class_label'][giant] = 1
        evidence['feh'] = reference['feh']; evidence['feh_error'] = reference['feh_err']
        evidence['feh_valid'] = quality & np.isfinite(reference['feh']) & np.isfinite(reference['feh_err']) & (reference['feh_err'] >= 0)
    else:
        raw = load(paths[-1]); j = unique_join(t['identity'],raw['identity']); assert (j >= 0).all()
        reference = {k:v[j] for k,v in raw.items()}
        assert (reference['TYPE'] == 'PSF').all() and (reference['probability'] > .9).all()
        np.testing.assert_array_equal(reference['scale_system'],t['scale_system'])
        atomic_npz(out/'membership.npz',**reference)
        evidence['scale_system'] = reference['scale_system']; evidence['membership_probability'] = reference['probability']
        save_json(out/'scales.json',json.loads(paths[-2].read_text()))
    gaia_root = Path('/global/cfs/cdirs/cosmo/data/gaia/dr3/healpix')
    # Exact tile identities/statistics allow an interrupted join to detect changed sources.
    def inventory():
        rows = []
        for pixel in np.unique(t['source_id'][t['source_id'] > 0] // SOURCE_ID_HEALPIX_FACTOR):
            path = gaia_root/f'healpix-{int(pixel):05d}.fits'
            stat = path.stat() if path.exists() else None
            rows.append(dict(path=str(path),exists=stat is not None,
                size=None if stat is None else stat.st_size,mtime_ns=None if stat is None else stat.st_mtime_ns))
        return rows
    tiles = inventory(); tile_record = out/'gaia_sources.json'
    if tile_record.exists():
        assert json.loads(tile_record.read_text()) == tiles
    else:
        save_json(tile_record,tiles)
    gaia = join_gaia(t['source_id'],gaia_root); labels = derive_gaia_labels(gaia)
    assert inventory() == tiles
    atomic_npz(out/'gaia_payload.npz',identity=t['identity'],TYPE=t['TYPE'],**gaia,
        **{k:v for k,v in labels.items() if k not in gaia})
    evidence.update(p_corr=labels['p_corr'],gaia_parallax_error=gaia['parallax_error'],
        parallax_eligible=labels['astro_quality'] & labels['zpt_valid'] & labels['finite_parallax'],
        gaia_match=gaia['gaia_match'],gaia_file_missing=gaia['gaia_file_missing'],
        nonstellar_risk=labels['contaminant_candidate'])
    mws,mws_audit=associate_mws(t,mws_base)
    atomic_npz(out/'mws_vac_payload.npz',identity=t['identity'],TYPE=t['TYPE'],**mws)
    evidence.update(mws)
    atomic_npz(out/'evidence.npz',**evidence)
    outputs = ['evidence.npz','gaia_payload.npz','gaia_sources.json','mws_vac_payload.npz']
    outputs += ['astronn_apogee.npz'] if a.cohort == 'field' else ['membership.npz','scales.json']
    save_json(marker,dict(inputs=inputs,outputs={name:sha(out/name) for name in outputs},cohort=a.cohort,n=n,
        all_PSF=True,non_PSF_excluded=0,missing_TYPE_excluded=0,admission_exclusions_retained_upstream=True,
        signed_Gaia=int(evidence['parallax_eligible'].sum()),negative_Gaia=int((evidence['parallax_eligible'] & (evidence['p_corr'] < 0)).sum()),
        missing_Gaia=int((~gaia['gaia_match']).sum()),distance_available=int(evidence['distance_available'].sum()),
        distance_precise=int(evidence['distance_precise'].sum()),class_labels={str(c):int((evidence['class_label']==c).sum()) for c in [-1,0,1]},
        final_outcomes_opened=True,prediction_residuals_computed=False,job=os.getenv('SLURM_JOB_ID'),
        MWS_VAC_channel_audit=mws_audit,
        scope='Same frozen PSF identities; source-qualified channels separate. AstroNN is Gaia-trained; common scales are shared; no calibrated fullapplication contamination claim.'))
    print('FINAL EVIDENCE OPENED',a.cohort,n,flush=True)


if __name__ == '__main__':
    main()
