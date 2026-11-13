#!/usr/bin/env python3
"""Authenticate the complete 22-plate delivery and persisted visual reviews."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    'paired_figures': ['target_prior_transfer', 'class_metallicity_contrasts'],
    'calibration_figure': ['luminosity_correction'],
    'latent_figure': ['latent_support_failure'],
    'target_candidate_figures': ['conditional_counts', 'weights_numerics'],
    'domain_photometry_figures': ['photometric_occupancy'],
    'final_validation_figures': ['field_score_gains', 'common_scale_agreement'],
    'radial_figures': ['paired_radial_error', 'radial_nonzero_coverage', 'radial_structure_parameters'],
    'response_figures': ['response_evidence', 'response_source_transfer'],
    'candidate_mode_figures': ['mode_responsibilities'],
    'domain_evidence_figures': ['precise_newsky_references', 'missing_reference_mass'],
    'heldout_transfer_figures': ['heldout_joint_counts', 'heldout_predictive_support'],
    'structure_figures': ['structure_posterior', 'structure_membership'],
    'domain_adjudication_figures': ['scientific_use_and_ambiguity'],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--completed-only', action='store_true',
                        help='Explicit partial audit; never writes final delivery acceptance')
    partial = parser.parse_args().completed_only
    base = ROOT / 'manifests/v3_1'
    missing = [g for g in GROUPS if not (base / g / 'complete.json').exists()]
    if missing and not partial:
        raise FileNotFoundError('All 22 plates required; missing groups: ' + ', '.join(missing))
    inputs, checks, inventory = {}, [], []

    def check(name, value):
        if not value:
            raise AssertionError(name)
        checks.append(dict(check=name, passed=True))

    def record(path):
        inputs[str(path.relative_to(ROOT))] = sha(path)

    for group, stems in GROUPS.items():
        if group in missing:
            continue
        folder = base / group
        marker = folder / 'complete.json'
        done = json.loads(marker.read_text())
        review = folder / ('delivery_validation.json' if group == 'paired_figures' else 'acceptance.json')
        accepted = json.loads(review.read_text())
        record(marker)
        record(review)
        check(group + ':recorded_artifact_checks',
              accepted.get('passed', accepted.get('visual_review_passed', False)) is True)
        visual = accepted.get('visual_review', accepted.get('visual_review_passed',
                              accepted.get('visually_reviewed')))
        if isinstance(visual, dict) and 'completed' in visual:
            visual = visual['completed']
        check(group + ':persisted_visual_review', bool(visual))
        check(group + ':English_renderer_record', done['english_text_verified'] is True)
        # Some early visual records predate explicit manifest hashes. Bind all
        # reviews to their actual committed image manifest, not an inferred view.
        review_name = str(review.relative_to(ROOT))
        commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', review_name],
                                         cwd=ROOT, text=True).strip()
        check(group + ':committed_review', len(commit) == 40)
        for path in [review, marker]:
            original = subprocess.check_output(['git', 'show', f'{commit}:{path.relative_to(ROOT)}'], cwd=ROOT)
            check(group + ':visual_commit_binding:' + path.name,
                  hashlib.sha256(original).hexdigest() == sha(path))
        for name, digest in done['inputs'].items():
            path = ROOT / name.replace('data/v3_1/', 'manifests/v3_1/', 1)
            check(group + ':input:' + name, sha(path) == digest)
            record(path)
        outputs = {}
        for name, digest in done['outputs'].items():
            path = ROOT / name
            if '/' not in name:
                path = (ROOT / 'figures/v3_1' / group if Path(name).suffix in ['.png', '.pdf'] else folder) / name
            check(group + ':output:' + name, sha(path) == digest)
            record(path)
            outputs[path] = digest
        for suffix in ['.png', '.pdf']:
            paths = [p for p in outputs if p.suffix == suffix]
            check(group + ':exact_inventory:' + suffix,
                  len(paths) == len(stems) and {p.stem for p in paths} == set(stems))
        for path in outputs:
            if path.suffix == '.png':
                with Image.open(path) as im:
                    im.load()
                    check(group + ':' + path.stem + ':PNG300dpi',
                          min(im.size) >= 1000 and len(im.info.get('dpi', [])) == 2 and
                          all(abs(v - 300) < 1 for v in im.info['dpi']))
            elif path.suffix == '.pdf':
                reader = PdfReader(path)
                check(group + ':' + path.stem + ':PDFonepage', len(reader.pages) == 1)
                text = reader.pages[0].extract_text()
                check(group + ':' + path.stem + ':PDFEnglish', len(text) > 100 and
                      not any('\u4e00' <= c <= '\u9fff' for c in text))
        inventory.append(dict(group=group, figures=len(stems), stems=stems,
                              visual_record_commit=commit, visual_record_sha256=sha(review),
                              manifest_sha256=sha(marker)))
    n = sum(r['figures'] for r in inventory)
    check('total_inventory', n + sum(len(GROUPS[g]) for g in missing) == 22)
    record(Path(__file__))
    result = dict(inputs=inputs, checks=checks, actual_checks=len(checks), inventory=inventory,
                  figures=n, missing_groups=missing, partial_requested=partial,
                  complete_figure_delivery=not partial and not missing and n == 22,
                  passed=True, new_visual_inspection_performed=False, scientific_acceptance=False,
                  scope='Current bytes, source tables, exact plate inventory, formats and committed visual-review provenance. Existing visual inspections authenticated, not repeated or inferred. Scientific and full-version acceptance remain separate.')
    name = f'figure_delivery_partial_{n}.json' if partial else 'figure_delivery.json'
    (base / name).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['figures', 'actual_checks', 'missing_groups', 'complete_figure_delivery']}))


if __name__ == '__main__':
    main()
