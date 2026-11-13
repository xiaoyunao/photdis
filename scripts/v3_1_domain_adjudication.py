#!/usr/bin/env python3
"""Join the completed v3.1 evidence without promoting failed numerical products."""
import csv
import json
import os
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, csv_rows
from photdis.stage1 import require_compute


def main():
    require_compute()
    root = ROOT / 'data/v3_1'
    out = root / 'domain_adjudication'
    inputs, markers = {}, {}
    stages = ['domain_evidence', 'domain_assessment', 'candidate_modes',
              'target_ensemble/candidate', 'highq_counts', 'response_probes/report',
              'radial_report', 'radial_summary', 'final_validation/report',
              'heldout_transfer/report', 'structure/posterior', 'structure/report']
    for stage in stages:
        path = root / stage / 'complete.json'
        inputs[str(path.relative_to(ROOT))] = sha(path)
        markers[stage] = json.loads(path.read_text())

    def source(stage, name):
        path = root / stage / name
        assert sha(path) == markers[stage]['outputs'][name]
        inputs[str(path.relative_to(ROOT))] = sha(path)
        return path

    def rows(stage, name):
        with source(stage, name).open() as stream:
            return list(csv.DictReader(stream))

    evidence = markers['domain_evidence']
    ensemble = markers['target_ensemble/candidate']
    main = next(v for v in ensemble['views'] if v['view'] == 'bootstrap_evidence')
    assert ensemble['states'] == 34 and ensemble['all_PSF'] and main['inputs_passed']
    # This is a synthesis of the already evaluated, immutable failed candidate.
    # A different accepted product would require its own scientific adjudication.
    assert not main['accepted'], 'Do not automatically certify a different candidate'
    assert evidence['candidate_n'] == 3540 and evidence['cells'] == 75 and evidence['all_PSF']
    assert markers['candidate_modes']['n'] == 3540
    assert not markers['candidate_modes']['inherited_ensemble_accepted']
    assert markers['response_probes/report']['states'] == 64
    assert markers['radial_report']['catalogs'] == 176 and markers['radial_report']['arms'] == 352
    assert markers['radial_summary']['all_outcomes_retained']
    assert markers['final_validation/report']['all_frozen_identities_retained']
    assert not markers['final_validation/report']['scientific_acceptance']
    assert markers['heldout_transfer/report']['states'] == 102
    for stage in ['structure/posterior', 'structure/report']:
        assert markers[stage]['states'] == 32 and markers[stage]['n'] == 512 and markers[stage]['all_PSF']
    for stage, dependency in [('domain_evidence', 'target_ensemble/candidate'),
                              ('domain_evidence', 'final_validation/report'),
                              ('domain_evidence', 'response_probes/report'),
                              ('highq_counts', 'target_ensemble/candidate')]:
        name = f'data/v3_1/{dependency}/complete.json'
        assert markers[stage]['inputs'][name] == inputs[name]

    objects = load(source('domain_evidence', 'candidate_domains.npz'))
    assert len(objects['identity']) == len(set(objects['identity'])) == 3540
    assert (objects['TYPE'] == 'PSF').all() and int(objects['highq'].sum()) == 3003
    assert np.array_equal(objects['highq'], objects['qnear'] > .95)
    assert np.isin(objects['cell'], np.arange(75)).all()
    mode_rows = rows('candidate_modes', 'full_summary.csv')
    modes = {int(r['identity']): r for r in mode_rows}
    assert len(mode_rows) == len(modes) == 3540 and set(modes) == set(objects['identity'])
    assert all(r['TYPE'] == 'PSF' for r in mode_rows)
    cells = rows('domain_evidence', 'cells.csv')
    assert len(cells) == 75 and {int(r['cell']) for r in cells} == set(range(75))
    roles = ['new_sky_field', 'new_source_seen_sky_field']
    screens = {}
    for role in roles:
        values = rows('domain_assessment', role + '_cells.csv')
        selected = [r for r in values if r['group'] == 'all']
        assert len(selected) == 75 and {int(r['cell']) for r in selected} == set(range(75))
        assert all(r['absolute_calibration_established'] == 'False' for r in values)
        screens[role] = {int(r['cell']): r for r in selected}
    references = rows('domain_evidence', 'reference_distance_cells.csv')
    precise = {(channel, cell): 0 for channel in ['astronn', 'specdis', 'spdist'] for cell in range(75)}
    for r in references:
        if r['role'] == 'new_sky_field' and r['quality'] == 'precise':
            precise[r['source'], int(r['cell'])] += int(r['n'])
    transfer_inventory = rows('heldout_transfer/report', 'inventory.csv')
    assert len(transfer_inventory) == 102 and len({(r['case'], r['correction']) for r in transfer_inventory}) == 102
    transfer_support = rows('heldout_transfer/report', 'predictive_summary.csv')
    # Authenticate the actual outcomes used by the written integrated review.
    for stage, names in [
        ('radial_report', ['status.csv', 'structure.csv', 'paired.csv']),
        ('radial_summary', ['source_inventory.csv', 'coverage95_defined_structure.csv']),
        ('final_validation/report', ['new_sky_field_availability.csv', 'new_sky_field_paired_gaia_all_objects.csv', 'common_scale_summary.csv']),
        ('response_probes/report', ['sets.json', 'weight_numerics.csv']),
        ('structure/posterior', ['summary.csv', 'numerical_checks.csv']),
        ('structure/report', ['counts.csv', 'membership_comparison.csv']),
        ('highq_counts', ['counts.csv'])]:
        for name in names:
            source(stage, name)

    object_rows = []
    for i, identity in enumerate(objects['identity']):
        mode = modes[int(identity)]
        assert (mode['fixed_main_highq'] == 'True') == bool(objects['highq'][i])
        object_rows.append(dict(identity=int(identity), TYPE='PSF', cell=int(objects['cell'][i]),
            fixed_main_highq=bool(objects['highq'][i]), scientific_disposition='failed_main_numerical_gate',
            absolute_calibration_established=False, independent_contamination_measured=False,
            allowed_use='conditional_model_diagnostics_only',
            significant_modes=int(mode['significant_modes']), multimodal=int(mode['significant_modes']) >= 2,
            mode_numerical_passed=mode['numerical_passed'] == 'True',
            native_fine_audited=bool(objects['native_fine_audited'][i]),
            distance_mass_without_precise_newsky_reference=float(objects['posterior_distance_mass_without_precise_newsky_reference'][i])))
    result = []
    for original in cells:
        cell = int(original['cell'])
        selected = [r for r in object_rows if r['cell'] == cell]
        assert len(selected) == int(original['candidate_n'])
        assert sum(r['fixed_main_highq'] for r in selected) == int(original['highq_n'])
        r = dict(original)
        r.pop('scientific_adjudication')
        r.update(scientific_disposition='failed_main_numerical_gate' if selected else 'outside_candidate',
            absolute_calibration_established=False, calibrated_domain=False,
            scientifically_accepted_ambiguous_domain=False,
            allowed_use='conditional_model_diagnostics_only' if selected else 'no_candidate_claim',
            empirical_absolute_distance_status='unvalidated',
            multimodal_n=sum(row['multimodal'] for row in selected),
            highq_multimodal_n=sum(row['multimodal'] and row['fixed_main_highq'] for row in selected),
            mode_numerical_failed_n=sum(not row['mode_numerical_passed'] for row in selected))
        for channel in ['astronn', 'specdis', 'spdist']:
            n = precise[channel, cell]
            r['precise_newsky_' + channel] = n
            r['precise_newsky_' + channel + '_below20'] = n < 20
        for role in roles:
            for key in ['n', 'original_sky_cells', 'main_numerical_failed', 'relative_assessment',
                        'sufficient_mean_failures', 'sufficient_mean_passes', 'numerical_metric_failures',
                        'sparse_or_missing_metrics', 'sky_intervals_within_margin']:
                r[role + '_' + key] = screens[role][cell][key]
        result.append(r)
    assert sum(int(r['candidate_n']) for r in result) == 3540
    assert sum(r['highq_multimodal_n'] for r in result) == sum(r['multimodal'] and r['fixed_main_highq'] for r in object_rows)
    for path in [ROOT / 'scripts/v3_1_domain_adjudication.py', ROOT / 'docs/v3_1_domain_adjudication_notes.md',
                 ROOT / 'docs/v3_1_use_domain_protocol.md', ROOT / 'docs/v3_1_domain_assessment_protocol.md']:
        inputs[str(path.relative_to(ROOT))] = sha(path)
    out.mkdir(exist_ok=True)
    marker = out / 'complete.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        assert saved['inputs'] == inputs
        for name, digest in saved['outputs'].items():
            assert sha(out / name) == digest
        return
    csv_rows(out / 'cells.csv', result)
    csv_rows(out / 'objects.csv', object_rows)
    csv_rows(out / 'transfer_support.csv', transfer_support)
    save_json(out / 'global_limits.json', dict(
        candidate_main=main,
        highq_views=markers['highq_counts']['views'],
        independent_SED_probe_sets=markers['response_probes/report']['sets'],
        radial=dict(catalogs=176, arms=352, numerically_passed=markers['radial_report']['numerically_passed'],
            selection_approximation=markers['radial_report']['selection_approximation'],
            shared_library_uncertainty_measured=markers['radial_summary']['shared_library_uncertainty_measured']),
        final_external_scientific_acceptance=markers['final_validation/report']['scientific_acceptance'],
        independent_fullcandidate_contamination_measured=markers['final_validation/report']['independent_fullcandidate_contamination_measured'],
        heldout_transfer_views=markers['heldout_transfer/report']['views'],
        structure_posterior_summary=markers['structure/posterior']['summary'],
        structure_count_report_passed=markers['structure/report']['passed'],
        scope='Shared globals are not local cell evidence. Seen NGC288 supplied-scale application is not independent final-candidate calibration. Signed Gaia is never inverted. Teacher channels and the one common scale are not independent replicas.'))
    save_json(marker, dict(inputs=inputs, outputs={name: sha(out / name) for name in
        ['cells.csv', 'objects.csv', 'transfer_support.csv', 'global_limits.json']},
        cells=75, candidate_n=3540, highq_n=3003, all_PSF=True,
        non_PSF_excluded=0, missing_TYPE_excluded=0, new_sample_constructed=False,
        occupied_cells=sum(int(r['candidate_n']) > 0 for r in result),
        calibrated_cells=0, scientifically_accepted_ambiguous_cells=0,
        failed_candidate_objects=3540, integrated_written_review_pending=True,
        scientific_domain_adjudication_complete=False, job=os.getenv('SLURM_JOB_ID'),
        scope='Complete actual evidence join for the immutable failed v3.1 main candidate. Numerical failure does not erase model ambiguity or missing empirical evidence. Final written scientific interpretation and figure review remain required.'))


if __name__ == '__main__':
    main()
