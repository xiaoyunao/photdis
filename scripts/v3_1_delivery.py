#!/usr/bin/env python3
"""Close the complete research delivery without granting scientific acceptance."""
import csv
import hashlib
import json
from pathlib import Path
import re

from v3_1_preservation_check import examine as examine_preservation


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
        'structure/matrices/complete.json', 'structure/matrices/status.csv',
        'structure/posterior/complete.json',
        'structure/report/complete.json', 'domain_adjudication/complete.json',
        'figure_delivery.json', 'final_unit_validation.json',
        'final_narrative_review.json', 'final_requirements_review.json',
        'baseline_preservation_current.json',
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

    preservation = read('baseline_preservation_current.json')
    authenticate(preservation['inputs'], 'preservation_source')
    current_preservation = examine_preservation()
    check('baseline_preserved_now', preservation['passed'] and current_preservation['passed'] and
          preservation['baseline_commit'] == current_preservation['baseline_commit'] and
          preservation['protected_baseline_files'] == current_preservation['protected_baseline_files'])

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
    state_records = {}
    for kind in ['native', 'native_refined', 'entry', 'entry_refined']:
        for index in range(32):
            name = f'structure/{kind}/draw_{index:02}/complete.json'
            path = BASE / name
            check('assembly_state:' + name, sha(path) == assembly['inputs']['data/v3_1/' + name])
            record(path)
            state_records[kind, index] = json.loads(path.read_text())
    status_path = BASE / 'structure/matrices/status.csv'
    check('assembled_status_hash', sha(status_path) == assembly['outputs']['status.csv'])
    record(status_path)
    with status_path.open() as stream:
        status_rows = list(csv.DictReader(stream))
    expected = {(draw, row) for draw in range(32) for row in range(512)}
    check('all16384_state_objects', len(status_rows) == 16384 and
          {(int(r['draw']), int(r['row'])) for r in status_rows} == expected and
          all(r['TYPE'] == 'PSF' for r in status_rows))
    flag_columns = ['original_native_passed', 'original_entry_passed',
                    'native_passed', 'entry_passed', 'passed']
    check('explicit_boolean_flags', all(r[k] in ['True', 'False']
          for r in status_rows for k in flag_columns))
    identity_by_row = {int(r['row']): r['identity'] for r in status_rows if int(r['draw']) == 0}
    check('identical_row_order_across_states', all(r['identity'] == identity_by_row[int(r['row'])]
          for r in status_rows))
    membership_path = BASE / 'structure/report/membership.csv'
    check('reported_membership_hash', sha(membership_path) == counts['outputs']['membership.csv'])
    record(membership_path)
    with membership_path.open() as stream:
        member_ids = {r['identity'] for r in csv.DictReader(stream)}
    check('same512_structure_objects', len(member_ids) == 512 and
          {r['identity'] for r in status_rows} == member_ids)
    for draw in range(32):
        rows = [r for r in status_rows if int(r['draw']) == draw]
        check('same_identities:' + str(draw), len({r['identity'] for r in rows}) == 512 and
              {r['identity'] for r in rows} == member_ids)
        for kind in ['native', 'entry']:
            old, new = state_records[kind, draw], state_records[kind + '_refined', draw]
            check('original_flags:' + kind + ':' + str(draw),
                  sum(r['original_' + kind + '_passed'] == 'False' for r in rows) ==
                  old['failures'] == new['original_failures'])
            check('refined_flags:' + kind + ':' + str(draw),
                  sum(r[kind + '_passed'] == 'False' for r in rows) == new['unresolved'])
        check('combined_flags:' + str(draw), all((r['passed'] == 'True') ==
              (r['native_passed'] == r['entry_passed'] == 'True') for r in rows))
    check('assembled_failure_total', sum(r['passed'] == 'False' for r in status_rows) ==
          assembly['failed_state_objects'])
    # Production assembly authenticated the large arrays on an allocated node.
    # This local closure checks compact flags, markers and downstream linkage.
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
    check('actual_tests_passed', unit['exit_code'] == 0 and unit['failed'] == 0 and unit['passed'] >= 183)
    check('preservation_regression_tests', {'scripts/v3_1_preservation_check.py',
          'tests/test_v3_1_preservation.py'} <= set(unit['inputs']) and
          'tests/test_v3_1_preservation.py' in unit['command'])
    authenticate(unit['inputs'], 'unit_source')
    check('test_transcript', sha(BASE / 'final_unit_validation.txt') == unit['stdout_sha256'])
    record(BASE / 'final_unit_validation.txt')
    requirements = read('final_requirements_review.json')
    authority = {'docs/pro_review_31f1277.md', 'docs/v3_1_protocol.md',
                 'docs/v3_1_requirements_result.md'}
    check('requirements_authority', set(requirements['authority']) == authority)
    authenticate(requirements['authority'], 'requirements_authority')
    ledger = ROOT / 'docs/v3_1_requirements_result.md'
    expected = {line.split('|')[1].strip(): line.split('|')[2].strip()
                for line in ledger.read_text().splitlines() if re.match(r'^\| [A-F]\d', line)}
    items = requirements['items']
    check('every_explicit_requirement_reviewed', expected and
          len(items) == requirements['requirements'] == len(expected) and
          {item['id']: item['requirement'] for item in items} == expected)
    check('no_pending_requirement', requirements['delivery_ready'] and
          not requirements['pending'] and requirements['scientific_acceptance'] is False)
    for item in items:
        check('requirement_disposition:' + item['id'], item['status'] in
              {'implemented_with_limitations', 'evidence_gap_with_scope_limit'} and
              bool(item['finding'].strip()) and bool(item['limitations'].strip()) and
              bool(item['evidence']))
        authenticate(item['evidence'], 'requirement_evidence:' + item['id'])
    # Source binding/completeness complements the written scientific review;
    # it does not automatically judge the truth of the reviewed claims.
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
                  current_tracked_preservation=current_preservation,
                  scope='Local compact provenance and complete research delivery closure. Existing allocated production audits cover large payloads; no fresh full-payload scan or independent scientific calibration is claimed. Final git synchronization is recorded separately.')
    (BASE / 'delivery.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['actual_checks', 'research_delivery_complete', 'scientific_acceptance']}))


if __name__ == '__main__':
    main()
