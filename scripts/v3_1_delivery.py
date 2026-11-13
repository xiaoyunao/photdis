#!/usr/bin/env python3
"""Close the complete research delivery without granting scientific acceptance."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'manifests/v3_1'
REPORTS = {
    'mechanism': ('mechanism_result_sources.json', 'report_sha256'),
    'radial': ('radial_report/result_sources.json', 'document_sha256'),
    'response': ('response_summary/result_sources.json', 'document_sha256'),
    'final_validation': ('final_validation/result_sources.json', 'output_sha256'),
    'candidate_modes': ('candidate_modes/result_sources.json', 'document_sha256'),
    'domain_evidence': ('domain_evidence/result_sources.json', 'document_sha256'),
    'transfer': ('transfer_result_sources.json', 'report_sha256'),
    'structure': ('structure_result_sources.json', 'report_sha256'),
    'domain': ('domain_result_sources.json', 'report_sha256'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    required = [BASE / name for name, _ in REPORTS.values()]
    required += [BASE / name for name in [
        'heldout_transfer/compact_inventory_102_states.json',
        'structure/refined_entry_32_states.json',
        'structure/matrices/complete.json', 'structure/posterior/complete.json',
        'structure/report/complete.json', 'domain_adjudication/complete.json',
        'figure_delivery.json', 'final_unit_validation.json',
        'final_narrative_review.json',
    ]]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError('Complete delivery required; no output written. Missing: ' + ', '.join(missing))
    inputs, checks = {}, []

    def check(name, value):
        if not value:
            raise AssertionError(name)
        checks.append(dict(check=name, passed=True))

    def record(path):
        inputs[str(path.relative_to(ROOT))] = sha(path)

    def read(name):
        path = BASE / name
        record(path)
        return json.loads(path.read_text())

    def authenticate(mapping, label):
        for name, digest in mapping.items():
            path = ROOT / name
            check(label + ':' + name, path.is_file() and sha(path) == digest)
            record(path)

    for stem, (name, key) in REPORTS.items():
        proof = read(name)
        document = ROOT / f'docs/v3_1_{stem}_result.md'
        check('report:' + stem, sha(document) == proof[key])
        record(document)
        authenticate(proof['inputs'], 'report_source:' + stem)
        if 'checks' in proof:
            check('report_checks:' + stem, proof['actual_checks'] == len(proof['checks']) and
                  all(c['passed'] for c in proof['checks']))

    transfer = read('heldout_transfer/compact_inventory_102_states.json')
    check('all102_transfer_states', transfer['all102_states_present'] and
          transfer['completed_states'] == 102 and not transfer['remaining_states'] and
          transfer['compact_consistency_passed'])
    authenticate(transfer['inputs'], 'transfer_inventory')
    refined = read('structure/refined_entry_32_states.json')
    check('all32_refined_states', refined['all32_present'] and refined['states'] == 32 and
          refined['checked_branch_rows'] == 32768 and refined['compact_consistency_passed'])
    authenticate(refined['inputs'], 'refined_inventory')
    for name, digest in refined['outputs'].items():
        authenticate({'manifests/v3_1/structure/' + name: digest}, 'refined_summary')
    assembly = read('structure/matrices/complete.json')
    posterior = read('structure/posterior/complete.json')
    counts = read('structure/report/complete.json')
    for name, value in [('assembly', assembly), ('posterior', posterior), ('counts', counts)]:
        check('complete_structure:' + name, value['states'] == 32 and value['n'] == 512 and value['all_PSF'])
    for kind in ['native', 'native_refined', 'entry', 'entry_refined']:
        for index in range(32):
            name = f'structure/{kind}/draw_{index:02}/complete.json'
            path = BASE / name
            check('assembly_state:' + name, sha(path) == assembly['inputs']['data/v3_1/' + name])
            record(path)
    # Production assembly authenticated the large arrays on an allocated node.
    # This local closure authenticates its marker and downstream linkage only.
    check('posterior_assembly_identity', posterior['inputs']['data/v3_1/structure/matrices/complete.json'] ==
          sha(BASE / 'structure/matrices/complete.json'))
    check('counts_posterior_identity', counts['inputs']['data/v3_1/structure/posterior/complete.json'] ==
          sha(BASE / 'structure/posterior/complete.json'))
    domain = read('domain_adjudication/complete.json')
    check('same_fixed_candidate', domain['candidate_n'] == 3540 and domain['highq_n'] == 3003 and
          domain['cells'] == 75 and domain['all_PSF'] and not domain['new_sample_constructed'])
    review = read('domain_result_sources.json')
    check('written_domain_review', review['passed'] and review['written_review_complete'])
    check('no_scientific_acceptance', not review['scientific_acceptance'] and domain['calibrated_cells'] == 0)

    figures = read('figure_delivery.json')
    check('all22_visually_reviewed_figures', figures['passed'] and figures['complete_figure_delivery'] and
          figures['figures'] == 22 and len(figures['inventory']) == 13 and not figures['missing_groups'])
    check('all_figure_checks', figures['actual_checks'] == len(figures['checks']) and
          all(c['passed'] for c in figures['checks']))
    authenticate(figures['inputs'], 'figure_provenance')
    unit = read('final_unit_validation.json')
    check('actual_tests_passed', unit['exit_code'] == 0 and unit['failed'] == 0 and unit['passed'] >= 179)
    authenticate(unit['inputs'], 'unit_source')
    check('test_transcript', sha(BASE / 'final_unit_validation.txt') == unit['stdout_sha256'])
    record(BASE / 'final_unit_validation.txt')
    narrative = read('final_narrative_review.json')
    check('complete_narrative_review', narrative['delivery_ready'] and narrative['requirements_reviewed'] and
          narrative['scientific_acceptance'] is False)
    required_docs = {'docs/v3_1_result.md', 'docs/v3_1_requirements_result.md',
                     'docs/v3_1_methods.md', 'docs/v3_1_atlas.md', 'docs/v3_1_status.md',
                     'README.md', 'AGENTS.md', 'WORKLOG.md', 'PLAN.md'}
    check('all_delivery_documents', required_docs <= set(narrative['inputs']))
    authenticate(narrative['inputs'], 'narrative_source')
    for item in narrative['links']:
        check('document_link:' + item['document'] + ':' + item['target'],
              (ROOT / item['document']).parent.joinpath(item['target'].split('#')[0]).exists())
    frozen = read('final_validation/freeze_receipt.json')
    score = read('final_validation/score_audit/complete.json')
    check('freeze_before_reference_opening', frozen['predictions_frozen'] and
          not frozen['outcomes_opened_at_freeze'] and frozen['hashed_artifacts'] == 53725)
    check('score_uses_frozen_predictions', score['inputs']['data/v3_1/final_validation/freeze.json'] == frozen['sha256'])
    check('actual_external_score_audit', score['artifact_audit_passed'] and score['actual_checks'] == 432132 and
          not score['scientific_acceptance'])
    record(Path(__file__))
    result = dict(inputs=inputs, checks=checks, actual_checks=len(checks), passed=True,
                  research_delivery_complete=True, scientific_acceptance=False,
                  scientific_domain_adjudication_complete=True, model_adopted=False,
                  independent_contamination_measured=False, deployed_v1_replaced=False,
                  figures=22, transfer_states=102, structure_states=32,
                  original_structure_failures=refined['original_failed_contexts'],
                  unresolved_structure_contexts=refined['unresolved_contexts'],
                  scope='Local compact provenance and complete research delivery closure. Existing allocated production audits cover large payloads; no fresh full-payload scan or independent scientific calibration is claimed. Final git synchronization is recorded separately.')
    (BASE / 'delivery.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['actual_checks', 'research_delivery_complete', 'scientific_acceptance']}))


if __name__ == '__main__':
    main()
