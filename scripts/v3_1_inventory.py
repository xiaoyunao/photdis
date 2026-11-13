#!/usr/bin/env python3
"""Allocated reference availability and historical-role audit, without predictions.

Raw source rows without native TYPE are an acquisition inventory, not an admitted
scientific sample. This does not declare any historical role independently blind.
"""
import json
import os
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, csv_rows
from photdis.stage1 import require_compute
from photdis.audit import rvs_metallicity
from photdis.assembly_v1_4 import representative_rows
from photdis.v3.controls import optical_magnitudes
from photdis.baseline_v1 import support
import joblib


def projected(path, columns):
    with np.load(path) as z:
        return {k: z[k] for k in columns}


def main():
    require_compute(); d = ROOT/'data/v3_0'; old = ROOT/'data/baseline_v1_4'
    out = ROOT/'data/v3_1/inventory'; out.mkdir(parents=True, exist_ok=True)
    paths = [d/'references.npz', old/'sources.npz', old/'source_evidence.npz',
        d/'cohorts/application.npz', d/'controls/qrf_grz.joblib',
        ROOT/'scripts/v3_1_inventory.py', ROOT/'docs/v3_1_protocol.md']
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    marker = out/'complete.json'
    if marker.exists():
        old_record = json.loads(marker.read_text()); assert old_record['inputs'] == inputs
        for name, digest in old_record['outputs'].items(): assert sha(out/name) == digest
        return
    t = projected(paths[0], ['identity','TYPE','role','cell','class_label','historical_fit',
        'specdis_valid','specdis_distance','spdist_valid','spdist_distance','parallax_eligible',
        'p_corr','gaia_parallax_error','rvs_feh','rvs_teff','rvs_logg','rvs_parameter_quality'])
    types = np.char.strip(t['TYPE'].astype('U')); psf = types == 'PSF'
    morphology = dict(parent_rows=len(types), nonPSF=int(((types != '') & ~psf).sum()),
        missing_TYPE=int((types == '').sum()), admitted=int(psf.sum()))
    feh, valid = rvs_metallicity(t['rvs_feh'],t['rvs_teff'],t['rvs_logg'])
    poor = valid & t['rvs_parameter_quality'] & (feh < -1.5)
    rows = []; bounds = [0,2,5,10,20,50,np.inf]
    for role in np.unique(t['role']):
        for label in [None, -1,0,1,2]:
            for metal in ['all','poor']:
                mask = psf & (t['role'] == role)
                if label is not None: mask &= t['class_label'] == label
                if metal == 'poor': mask &= poor
                for source in ['specdis','spdist','gaia_precise']:
                    if source == 'gaia_precise':
                        available = t['parallax_eligible'] & (t['p_corr'] > 10*t['gaia_parallax_error'])
                        distance = np.full(len(mask),np.nan)
                        distance[available] = 1/t['p_corr'][available]
                    else: available = t[source+'_valid']; distance = t[source+'_distance']
                    for low,high in zip(bounds[:-1],bounds[1:]):
                        selected = mask & available & (distance >= low) & (distance < high)
                        rows.append(dict(role=int(role), reference_class='all' if label is None else label,
                            metal=metal, reference=source, low_kpc=low, high_kpc=None if not np.isfinite(high) else high,
                            n=int(selected.sum()), sky_cells=len(np.unique(t['cell'][selected])),
                            historical_fit=int(t['historical_fit'][selected].sum()),
                            validation_role='historical; not a new blind test'))
    csv_rows(out/'historical_coverage.csv',rows)
    raw = projected(paths[1], ['identity','cell','split','rvs_primary','rvs_sn_r'])
    evidence = projected(paths[2], ['identity','class_label','rvs_parameter_quality','historical_reference_identity',
        'class_identity_conflict','target_identity_conflict','specdis_valid','spdist_valid',
        'specdis_distance','spdist_distance','astro_quality','zpt_valid','finite_parallax','p_corr'])
    np.testing.assert_array_equal(raw['identity'], evidence['identity'])
    representative = representative_rows(raw['identity'],raw['rvs_primary'],raw['rvs_sn_r'])
    raw = {k:v[representative] for k,v in raw.items()}; evidence = {k:v[representative] for k,v in evidence.items()}
    used_id = np.isin(raw['identity'],t['identity'])
    used_sky = np.isin(raw['cell'],t['cell'])
    source_rows = []; raw_masks = dict(all=np.ones(len(used_id),bool), outside_v3_id=~used_id,
        outside_v3_sky=~used_sky, outside_v3_sky_and_old_reference=~used_sky & ~evidence['historical_reference_identity'])
    for name, selected in raw_masks.items():
        conflict = evidence['class_identity_conflict'] | evidence['target_identity_conflict']
        signed = evidence['astro_quality'] & evidence['zpt_valid'] & evidence['finite_parallax']
        source_rows.append(dict(group=name, identities=int(selected.sum()), cells=len(np.unique(raw['cell'][selected])),
            qualified_class_without_conflict=int((selected & ~conflict & (evidence['class_label'] >= 0)).sum()),
            evolved_without_conflict=int((selected & ~conflict & (evidence['class_label'] == 1)).sum()),
            signed_gaia=int((selected & signed).sum()), negative_gaia=int((selected & signed & (evidence['p_corr'] < 0)).sum()),
            admitted_scientific_sample=0, native_TYPE_unverified=int(selected.sum()),
            status='acquisition_inventory_only; native PSF/photometry/historical-use audit still required'))
    csv_rows(out/'raw_source_availability.csv',source_rows)
    del raw,evidence
    app = load(paths[3]); types = np.char.strip(app['TYPE'].astype('U')); psf = types == 'PSF'
    mag,error = optical_magnitudes(app); r = mag[:,1]-2.5*np.log10(app['transmission'][:,1])
    supported = support(joblib.load(paths[4]),mag)
    bright = psf & (r < 20) & (error.max(axis=1) < .1) & supported & (app['photometric_system'] == 'DECam_DR11')
    app_rows = []
    for field in np.unique(app['field']):
        for name, mask in [('all26k',psf),('candidate_envelope_subset',bright)]:
            selected = mask & (app['field'] == field); w = app['design_weight'][selected]
            app_rows.append(dict(sample=name, field=int(field), n=int(selected.sum()), weight_sum=float(w.sum()),
                effective_n=float(w.sum()**2/(w@w)) if len(w) else 0,
                missing_TYPE=int(((types == '') & (app['field'] == field)).sum()),
                nonPSF=int(((types != 'PSF') & (types != '') & (app['field'] == field)).sum())))
    csv_rows(out/'target_domain_inventory.csv',app_rows)
    save_json(marker,dict(inputs=inputs, outputs={p.name:sha(p) for p in out.glob('*.csv')},
        reference_morphology=morphology,job=os.getenv('SLURM_JOB_ID'),
        prior_validation='v3 new_sky/NGC288/reservation already opened; seen regression only',
        raw_source_limitation='Outside v3 ID/sky is not all-history independence. No raw acquisition row admitted before native TYPE verification. Source raw tables were inventoried historically; no prediction residuals read.',
        next='Audit all-history source/sky/system use, verify native PSF and reference availability, then freeze new final roles before any correction fit.',
        scientific_acceptance=False))
    print('V3.1 AVAILABILITY INVENTORY COMPLETE',flush=True)


if __name__ == '__main__':main()
