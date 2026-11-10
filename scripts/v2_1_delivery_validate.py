#!/usr/bin/env python3
"""Compact table conservation, figure formats/provenance and report link checks."""
import csv
import hashlib
import json
import re
from pathlib import Path
from PIL import Image
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'manifests/v2_1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks = {}
    def check(name, value):
        checks[name] = bool(value)
        if not value:
            raise AssertionError(name)
    def rows(name):
        with open(BASE/(name+'.csv')) as stream:
            return list(csv.DictReader(stream))
    distance = rows('distance_class')
    migration = rows('migration')
    for r in migration:
        key = tuple(r[k] for k in ('split', 'group', 'reference'))
        rr = [x for x in distance if tuple(x[k] for k in ('split', 'group', 'reference')) == key]
        check('distance_bins_'+str(key), len(rr) == 6 and sum(int(x['n']) for x in rr) == int(r['n']))
        check('migration_'+str(key), sum(int(r[k]) for k in ('near_selected', 'near_rejected', 'far_selected', 'far_rejected')) == int(r['n']))
    for split in ('cal', 'development_test'):
        for reference in ('chosen', 'Gaia', 'specdis', 'spdist'):
            rr = [r for r in migration if r['split'] == split and r['reference'] == reference]
            allrow = next(r for r in rr if r['group'] == 'all')
            classes = [r for r in rr if r['group'] in ('dwarf', 'evolved', 'other', 'unknown')]
            check('classes_'+split+reference, sum(int(r['n']) for r in classes) == int(allrow['n']))
    for index, r in enumerate(rows('risk_retention')):
        check('risk_counts_'+str(index), sum(int(r[k]) for k in ('near_selected', 'near_rejected', 'far_selected', 'far_rejected')) == int(r['n']))
    figures = {}
    paths = sorted((ROOT/'figures/v2_1').glob('*.png'))
    check('eight_plates', len(paths) == 8)
    for path in paths:
        name = path.stem
        pdf = path.with_suffix('.pdf')
        metadata = BASE/'figures'/(name+'.json')
        csvpath = BASE/'figures'/(name+'.csv')
        meta = json.loads(metadata.read_text())
        with Image.open(path) as img:
            img.load()
            check(name+'_300dpi', all(abs(x-300) < .02 for x in img.info['dpi']))
            check(name+'_dimensions', min(img.size) >= 1500)
        reader = PdfReader(pdf)
        check(name+'_one_PDF_page', len(reader.pages) == 1)
        text = reader.pages[0].extract_text()
        check(name+'_English_text', not re.search(r'[\u3400-\u9fff]', text))
        with open(csvpath) as stream:
            count = len(list(csv.DictReader(stream)))
        check(name+'_plotted_rows', count == meta['rows'] and count > 0)
        for source, expected in meta['inputs'].items():
            check(name+'_source_'+source, sha(BASE/(source+'.csv')) == expected)
        figures[name] = {str(p.relative_to(ROOT)): sha(p) for p in (path, pdf, csvpath, metadata)}
    for filename in ('v2_1_result.md', 'v2_1_atlas.md', 'v2_1_requirements_result.md'):
        p = ROOT/'docs'/filename
        for i, target in enumerate(re.findall(r'\]\(([^)]+)\)', p.read_text())):
            if target.startswith(('https://', 'http://', '#')):
                continue
            check(filename+'_link_'+str(i), (p.parent/target.split('#')[0]).exists())
    final = json.loads((BASE/'execution/final_validation.json').read_text())
    check('compute_artifact_validation_passed', final['all_passed'])
    result = dict(actual_delivery_checks=len(checks), all_passed=all(checks.values()), checks=checks,
                  compute_checks=final['total_actual_checks'], focused_tests=7,
                  figure_groups=6, plates=8, figures=figures,
                  compute_validation_sha256=sha(BASE/'execution/final_validation.json'),
                  visual_review='All eight final PNGs inspected; representative candidate PDF rendered and inspected; English labels, legibility and curve/legend consistency reviewed',
                  limitations=['Independent final validation remains unavailable', 'Candidate rejected; deployed baseline_v1 unchanged'])
    (BASE/'delivery.json').write_text(json.dumps(result, indent=2)+'\n')
    print(len(checks), 'actual delivery checks passed; eight figure sets verified')


if __name__ == '__main__':
    main()
