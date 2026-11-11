#!/usr/bin/env python3
"""Validate small figure artifacts against provenance and recorded visual review."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(group, reviewed):
    images = sorted((ROOT/'figures/v3_0'/group).glob('*.png'))
    assert images and set(reviewed) == {p.stem for p in images}
    checks, files = [], {}
    for png in images:
        pdf = png.with_suffix('.pdf')
        meta = ROOT/'manifests/v3_0/figures'/group/(png.stem+'.json')
        table = meta.with_suffix('.csv')
        record = json.loads(meta.read_text())
        for kind, path in [('png', png), ('pdf', pdf), ('csv', table)]:
            actual = digest(path)
            expected = record.get(kind+'_sha256')
            if expected is None:
                expected = record['outputs'][str(path.relative_to(ROOT))]
            assert actual == expected, path
            checks.append(str(path.relative_to(ROOT))+':hash')
            files[str(path.relative_to(ROOT))] = actual
        files[str(meta.relative_to(ROOT))] = digest(meta)
        with Image.open(png) as image:
            assert image.format == 'PNG' and min(image.size) >= 800
            assert all(abs(d-300) < .1 for d in image.info['dpi'])
        checks += [png.stem+':PNG', png.stem+':300dpi']
        reader = PdfReader(pdf)
        assert len(reader.pages) == 1
        text = reader.pages[0].extract_text()
        assert text.strip() and not any('\u4e00' <= c <= '\u9fff' for c in text)
        assert table.stat().st_size > 20
        checks += [png.stem+':PDFpage', png.stem+':English', png.stem+':CSV']
    return dict(passed=True, figures=len(images), checks=checks, files=files,
                visually_reviewed=list(reviewed),
                review_scope='Actual human-visible plate inspection recorded at invocation; format checks do not substitute for visual review.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group')
    parser.add_argument('--reviewed', default='')
    args = parser.parse_args()
    if args.group:
        record = audit(args.group, args.reviewed.split(','))
        output = ROOT/'manifests/v3_0/figures'/args.group/'delivery_validation.json'
    else:
        files, groups = {}, {}
        for group in ['development', 'diagnostics', 'recovery', 'validation', 'final']:
            path = ROOT/'manifests/v3_0/figures'/group/'delivery_validation.json'
            previous = json.loads(path.read_text())
            fresh = audit(group, previous['visually_reviewed'])
            assert fresh['files'] == previous['files'], 'Artifacts changed after visual review'
            groups[group] = dict(figures=fresh['figures'], checks=len(fresh['checks']), review_sha256=digest(path))
            files.update(fresh['files'])
        n = sum(g['figures'] for g in groups.values())
        assert n == 42
        record = dict(passed=True, figures=n, visually_reviewed=n, groups=groups, files=files)
        output = ROOT/'manifests/v3_0/figure_delivery_validation.json'
    output.write_text(json.dumps(record, indent=2)+'\n')
    print(output.relative_to(ROOT), record['figures'], 'figure sets passed')


if __name__ == '__main__':
    main()
