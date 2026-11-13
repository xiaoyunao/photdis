#!/usr/bin/env python3
"""Check the actual final five plates against their complete source tables."""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from PIL import Image
from pypdf import PdfReader
from v2_1_near import ROOT, sha, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', required=True, choices=['transfer', 'structure', 'domain'])
    group = parser.parse_args().group
    config = {
        'transfer': ('heldout_transfer_figures', 'heldout_transfer/report', 2, 12),
        'structure': ('structure_figures', 'structure', 2, 8),
        'domain': ('domain_adjudication_figures', 'domain_adjudication', 1, 6),
    }
    meta_name, source_name, figures, panels = config[group]
    meta = ROOT / 'manifests/v3_1' / meta_name
    source = ROOT / 'manifests/v3_1' / source_name
    marker = meta / 'complete.json'
    done = json.loads(marker.read_text())
    inputs = {str(marker.relative_to(ROOT)): sha(marker)}
    checks = []

    def check(name, condition):
        checks.append(dict(check=name, passed=bool(condition)))
        if not condition:
            raise AssertionError(name)

    def read(path):
        with path.open() as stream:
            return list(csv.DictReader(stream))

    for section in ['inputs', 'outputs']:
        for name, digest in done[section].items():
            check(section + ':' + name, sha(ROOT / name) == digest)
    check('plate_inventory', done['figures'] == figures and done['panels'] == panels)
    check('renderer_English_check', done['english_text_verified'])
    plotted = read(meta / 'plotted.csv')
    check('plotted_count', len(plotted) == done['plotted_rows'])
    required_sources = {
        'transfer': ['counts.csv', 'predictive_summary.csv'],
        'structure': ['report/counts.csv', 'posterior/summary.csv',
                      'posterior/numerical_checks.csv', 'report/membership.csv'],
        'domain': ['cells.csv'],
    }[group]
    check('only_declared_sources', {r['source'] for r in plotted} <= set(required_sources))
    sources = {}
    for name in required_sources:
        path = source / name
        parent_marker = path.parent / 'complete.json'
        record = json.loads(parent_marker.read_text())
        check('source_output:' + name, sha(path) == record['outputs'][path.name])
        check('source_in_renderer:' + name,
              done['inputs'][str(path.relative_to(ROOT))] == sha(path))
        inputs[str(parent_marker.relative_to(ROOT))] = sha(parent_marker)
        inputs[str(path.relative_to(ROOT))] = sha(path)
        sources[name] = read(path)

    # Count every intended source row per panel, including repeated source rows
    # used for distinct metrics. This catches missing/duplicate plotted points.
    expected = Counter()
    actual = Counter()

    def key(figure, panel, name, row, metric=''):
        return (figure, panel, name, metric, tuple(sorted(row.items())))

    def expect(figure, panel, name, row, metric=''):
        expected[key(figure, panel, name, row, metric)] += 1

    if group == 'transfer':
        for r in sources.get('counts.csv', []):
            if r['view'] == 'bootstrap_evidence' and r['variant'] == 'base':
                panel = str(['broad', 'candidate'].index(r['envelope'])) + ':' + r['class_index']
                expect('heldout_joint_counts', panel, 'counts.csv', r)
        for r in sources.get('predictive_summary.csv', []):
            if r['view'] == 'bootstrap_evidence' and r['variant'] == 'base':
                for column in range(3):
                    panel = str(['broad', 'candidate'].index(r['envelope'])) + ':' + str(column)
                    expect('heldout_predictive_support', panel, 'predictive_summary.csv', r)
    elif group == 'structure':
        for r in sources['report/counts.csv']:
            expect('structure_posterior', 'A', 'report/counts.csv', r)
            if r['component'] == 'member':
                expect('structure_posterior', 'F', 'report/counts.csv', r)
        for r in sources['posterior/summary.csv']:
            for panel in ['B', 'C', 'D']:
                expect('structure_posterior', panel, 'posterior/summary.csv', r)
        for r in sources['posterior/numerical_checks.csv']:
            if r['comparison'] != 'quadrature_first':
                for metric in ['maximum_member_probability_delta', 'maximum_mean_count_delta',
                               'maximum_common_mu_quantile_delta', 'log_evidence_delta']:
                    expect('structure_posterior', 'E', 'posterior/numerical_checks.csv', r, metric)
        for r in sources['report/membership.csv']:
            expect('structure_membership', 'A', 'report/membership.csv', r)
            if r['membership_available'] == 'True':
                expect('structure_membership', 'B', 'report/membership.csv', r)
    else:
        check('75_domain_cells', len(sources['cells.csv']) == 75 and
              {int(r['cell']) for r in sources['cells.csv']} == set(range(75)))
        for r in sources['cells.csv']:
            for panel in ['disposition', 'conditional_multimodality']:
                expect('', panel, 'cells.csv', r)

    for i, row in enumerate(plotted):
        name = row['source']
        columns = set(sources[name][0])
        original = {k: row[k] for k in columns}
        actual[key(row.get('figure', ''), row['panel'], name, original,
                   row.get('display_metric', ''))] += 1
        check('source_row:' + str(i), original in sources[name])
        if group == 'domain':
            n, multi = int(row['candidate_n']), int(row['multimodal_n'])
            check('domain_counts:' + str(i), 0 <= multi <= n)
            if row['panel'] == 'disposition':
                check('disposition_value:' + str(i), float(row['plotted_value']) == int(n > 0) and
                      row['display'] == (str(n) if n else '—'))
            elif n:
                check('multimodality_value:' + str(i),
                      math.isclose(float(row['plotted_value']), multi / n, rel_tol=1e-12) and
                      row['display'] == f'{100 * multi / n:.0f}%')
            else:
                check('empty_not_zero:' + str(i), row['plotted_value'] == '' and row['display'] == '—')
    check('exact_source_panel_multiplicity', actual == expected)
    counts = Counter(Path(name).suffix for name in done['outputs'])
    check('paired_PNG_PDF', counts['.png'] == counts['.pdf'] == figures)
    for name in done['outputs']:
        path = ROOT / name
        if path.suffix == '.png':
            with Image.open(path) as im:
                im.load()
                check(name + ':300dpi', min(im.size) >= 1000 and
                      all(abs(v - 300) < 1 for v in im.info.get('dpi', (0, 0))))
        elif path.suffix == '.pdf':
            pdf = PdfReader(path)
            check(name + ':one_page', len(pdf.pages) == 1)
            text = pdf.pages[0].extract_text()
            check(name + ':English', len(text) > 100 and
                  not any('\u4e00' <= c <= '\u9fff' for c in text))
    inputs[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    save_json(meta / 'integrity.json', dict(inputs=inputs, checks=checks,
        actual_checks=len(checks), passed=True, figures=figures, panels=panels,
        visual_review_completed=False, scientific_acceptance=False,
        scope='Actual source rows/panel multiplicities/hash chains and PNG/PDF formats. Visual review is separate and cannot be inferred from these checks.'))
    print(json.dumps(dict(group=group, actual_checks=len(checks), passed=True)))


if __name__ == '__main__':
    main()
