#!/usr/bin/env python3
"""Allocated, resumable same-object v3.1 mechanism pilot and diagnostic cohort."""
import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import time
import joblib
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, atomic_npz, csv_rows
from photdis.stage1 import require_compute
from photdis.audit import rvs_metallicity
from photdis.v3.data import stable_rank
from photdis.v3.model import inference_model
from photdis.v3.systematics import calibration_vectors
from photdis.v3.metrics import observed_parallax, normal_reference_score, distance_summary
from photdis.v3_1.diagnostics import infer_control
from v3_0_response import observation
import v3_0_evaluate as evaluator

D = ROOT/'data/v3_0'
_TABLE = _EVIDENCE = _MODELS = _CALIBRATION = _VIEWS = _OUT = _INPUTS = _FEH = _FEH_VALID = _CONFIG = None


def one_object(task):
    row, stratum, inclusion = task
    identity = int(_TABLE['identity'][row]); marker = _OUT/f'object_{identity}.json'
    if marker.exists():
        record = json.loads(marker.read_text())
        assert record['inputs'] == _INPUTS
        assert sha(marker.with_suffix('.npz')) == record['pdf_sha256']
        return record['results']
    o = observation(_TABLE, row); start = time.monotonic()
    label = int(_EVIDENCE['class_label'][row])
    options = dict(nominal={}, response12={}, **{'class': {'reference_class': label}},
        feh={'reference_feh': float(_FEH[row]), 'reference_feh_error': float(_EVIDENCE['rvs_feh_err'][row])},
        class_feh={'reference_class': label, 'reference_feh': float(_FEH[row]),
                   'reference_feh_error': float(_EVIDENCE['rvs_feh_err'][row])},
        grz={'bands': np.array([1,1,0,1,0,0], bool)}, griz={'bands': np.array([1,1,1,1,0,0], bool)},
        single={'binary_fraction': 0.})
    for key, bands in [('error_i2', [2]), ('error_w12', [4]), ('error_w22', [5]), ('error_optical2', [0,1,3])]:
        scale = np.ones(6); scale[bands] = 2; options[key] = dict(error_scale=scale)
    records = []; pdfs = {}
    for name in _CONFIG['controls']:
        if name in ('class', 'class_feh') and label < 0: continue
        if name in ('feh', 'class_feh') and not _FEH_VALID[row]: continue
        nominal = name == 'nominal'; model = _MODELS[int(not nominal)]
        control = dict(options[name])
        if not nominal: control['calibration_mag'] = _CALIBRATION[row]
        result = infer_control(model, o, initial_step=_CONFIG['numerical']['initial_step'],
            tolerance=_CONFIG['numerical']['probability_tolerance'], **control)
        p = result['posterior']; logpdf = result['full_logpdf']
        record = dict(identity=identity, TYPE='PSF', row=int(row), stratum=stratum,
            inclusion_probability=inclusion, design_weight=1/inclusion,
            cell=int(_TABLE['cell'][row]), class_label=label, reference_feh=float(_FEH[row]) if _FEH_VALID[row] else None,
            control=name, model='nominal' if nominal else 'response12', prior='reference_N15_sigma5',
            diagnostic_only=result['diagnostic_only'], mu=result['mu'].tolist(), qnear=result['qnear'],
            class_distance=result['class_distance'].tolist(), binary_probability=result['binary_probability'],
            kernel_negative_support_proxy_by_band=result['kernel_negative_support_proxy_by_band'].tolist(),
            modes=result['modes'], convergence=p.convergence,
            component_effective_count=float(1/(result['component_probability']@result['component_probability'])))
        if _EVIDENCE['parallax_eligible'][row]:
            record['gaia'] = observed_parallax(p.mu, logpdf, _EVIDENCE['p_corr'][row], _EVIDENCE['gaia_parallax_error'][row])
        record['references'] = {}
        for source, (mu, error, valid, fraction) in _VIEWS.items():
            if not valid[row]: continue
            broad_source = {1:'specdis', 2:'spdist'}.get(int(_EVIDENCE['primary_channel'][row])) if source == 'primary' else source
            broad_error = np.hypot(_EVIDENCE[broad_source+'_sigma_mu'][row], 2.) if broad_source in ('specdis','spdist') else 2.
            record['references'][source] = dict(mu=float(mu[row]), error=float(error[row]),
                logscore=normal_reference_score(p.mu, logpdf, mu[row], error[row], broad_fraction=fraction[row], broad_error=broad_error))
        records.append(record); pdfs[name+'_mu'] = p.mu; pdfs[name+'_logpdf'] = logpdf
    atomic_npz(marker.with_suffix('.npz'), **pdfs)
    save_json(marker, dict(inputs=_INPUTS, results=records, pdf_sha256=sha(marker.with_suffix('.npz')),
        elapsed_seconds=time.monotonic()-start, job=os.getenv('SLURM_JOB_ID')))
    return records


def main():
    global _TABLE, _EVIDENCE, _MODELS, _CALIBRATION, _VIEWS, _OUT, _INPUTS, _FEH, _FEH_VALID, _CONFIG
    require_compute(); parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['pilot', 'diagnostic'], default='pilot')
    parser.add_argument('--workers', type=int, default=16); args = parser.parse_args()
    config_path = ROOT/'configs/v3_1_diagnostics.json'; _CONFIG = json.loads(config_path.read_text())
    paths = [config_path, D/'cohorts/development.npz', D/'evaluation/development.npz', D/'calibration.npz',
        D/'sed_fit.joblib', D/'uncertainty_refit/draw_12/model.joblib', D/'uncertainty/shared_draws.npz',
        ROOT/'docs/v3_1_protocol.md', ROOT/'scripts/v3_1_diagnostics.py', ROOT/'scripts/v3_0_evaluate.py',
        ROOT/'scripts/v3_0_response.py', ROOT/'src/photdis/audit.py', ROOT/'src/photdis/v3_1/diagnostics.py']
    paths += [ROOT/'src/photdis/v3'/name for name in ('inference.py','likelihood.py','fast.py','model.py','systematics.py','metrics.py','data.py')]
    _INPUTS = {str(p.relative_to(ROOT)): sha(p) for p in paths}; _INPUTS['stage'] = args.stage
    _TABLE, _EVIDENCE = load(paths[1]), load(paths[2])
    np.testing.assert_array_equal(_TABLE['identity'], _EVIDENCE['identity'])
    assert len(np.unique(_TABLE['identity'])) == len(_TABLE['identity'])
    types = np.char.strip(_TABLE['TYPE'].astype('U')); etypes = np.char.strip(_EVIDENCE['TYPE'].astype('U'))
    admission = (types == 'PSF') & (etypes == 'PSF')
    psf_audit = dict(input_rows=len(types), admitted=int(admission.sum()),
        missing_TYPE=int(((types == '') | (etypes == '')).sum()),
        nonPSF=int(((types != 'PSF') & (types != '') | (etypes != 'PSF') & (etypes != '')).sum()))
    evaluator._CAL = load(paths[3]); _VIEWS = evaluator.evidence_views(_EVIDENCE)
    _MODELS = [inference_model(joblib.load(x)) for x in paths[4:6]]
    for model in _MODELS:
        assert model.converged
        assert not np.intersect1d(_TABLE['identity'], model.training_identity).size
    _CALIBRATION, _ = calibration_vectors(_TABLE, load(paths[6]), _CONFIG['response_draw'])
    _FEH, valid = rvs_metallicity(_EVIDENCE['rvs_feh'], _EVIDENCE['rvs_teff'], _EVIDENCE['rvs_logg'])
    _FEH_VALID = valid & _EVIDENCE['rvs_parameter_quality'] & np.isfinite(_EVIDENCE['rvs_feh_err']) & (_EVIDENCE['rvs_feh_err'] > 0)
    metal = np.where(_FEH_VALID, np.digitize(_FEH, _CONFIG['feh_edges']), 3)
    _OUT = ROOT/'data/v3_1/diagnostics'/args.stage; _OUT.mkdir(parents=True, exist_ok=True)
    tasks = []; strata = []; cap = _CONFIG[args.stage+'_per_stratum']
    for label in (-1, 0, 1, 2):
        for met in range(4):
            rows = np.flatnonzero(admission & (_EVIDENCE['class_label'] == label) & (metal == met))
            chosen = rows[np.argsort(stable_rank(_TABLE['identity'][rows], _CONFIG['seed']))[:cap]]
            name = f'class{label}_feh{met}'; inclusion = len(chosen)/len(rows) if len(rows) else 0.
            tasks.extend((int(row), name, inclusion) for row in chosen)
            strata.append(dict(stratum=name, parent=len(rows), selected=len(chosen), inclusion_probability=inclusion))
    freeze = dict(inputs=_INPUTS, psf=psf_audit, strata=strata,
        identities=[int(_TABLE['identity'][row]) for row, _, _ in tasks], role='seen_development_diagnostic',
        selection='Disjoint class/FeH strata, uniform stable rank; no residual/outcome success selection')
    freeze_path = _OUT/'cohort.json'
    if freeze_path.exists(): assert json.loads(freeze_path.read_text()) == freeze
    else: save_json(freeze_path, freeze)
    records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('fork')) as pool:
        for result in pool.map(one_object, tasks):
            records.extend(result)
            print('V3.1 DIAGNOSTIC', args.stage, len(records), 'views', flush=True)
    summaries = []
    for name in _CONFIG['controls']:
        for label in [None, -1, 0, 1, 2]:
            for source in ('primary','specdis','spdist','gaia_precise'):
                rr = [r for r in records if r['control'] == name and (label is None or r['class_label'] == label) and source in r['references']]
                if not rr: continue
                summaries.append(dict(control=name, reference_class='all' if label is None else label,
                    reference=source, scope='diagnostic-stratified-unweighted',
                    **distance_summary(np.array([r['mu'] for r in rr]), np.array([r['references'][source]['mu'] for r in rr])),
                    reference_logscore_mean=float(np.mean([r['references'][source]['logscore'] for r in rr]))))
    csv_rows(_OUT/'distance.csv', summaries)
    save_json(_OUT/'complete.json', dict(inputs=_INPUTS, cohort_sha256=sha(freeze_path), stars=len(tasks), views=len(records),
        nonconverged=sum(not r['convergence']['converged'] for r in records),
        objects={str(p.name):sha(p) for p in sorted(_OUT.glob('object_*.json'))},
        summary_sha256=sha(_OUT/'distance.csv'), job=os.getenv('SLURM_JOB_ID'),
        scientific_acceptance=False, next='Paired mechanism analysis with sampling/sky uncertainty; target-prior and negative-latent-posterior audits still pending'))
    print('V3.1 DIAGNOSTICS COMPLETE',len(tasks),len(records),flush=True)


if __name__ == '__main__': main()
