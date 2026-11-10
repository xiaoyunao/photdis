#!/usr/bin/env python3
"""Inspect compact CSV/JSON, PDF/PNG formats, provenance and delivery links."""
import csv
import json
import re
import hashlib
import subprocess
from pathlib import Path
from PIL import Image
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT/'manifests/v2_2'
STEMS = ['01_reliability', '02_selected_distance', '03_application', '03b_pdf_structure', '04_noise_transfer', '05_far_shape', '06_evidence_limits']


def main():
    checks = {}
    def check(name, value):
        checks[name] = bool(value)
        assert value, name
    compute = json.loads((TABLES/'execution/validation.json').read_text())
    check('actual_compute_checks', compute['all_passed'] and compute['actual_checks'] > 2000)
    for name in STEMS:
        png = ROOT/'figures/v2_2'/(name+'.png'); pdf = png.with_suffix('.pdf')
        with Image.open(png) as im:
            check(name+':PNG300dpi', im.format == 'PNG' and abs(im.info['dpi'][0]-300) < .1 and min(im.size) > 1000)
        info = subprocess.check_output(['pdfinfo', str(pdf)], text=True)
        check(name+':PDFpage', re.search(r'Pages:\s+1\b', info) is not None)
        text = '\n'.join(page.extract_text() or '' for page in PdfReader(pdf).pages)
        check(name+':English_text', bool(text.strip()) and not re.search(r'[\u3400-\u9fff]', text))
        meta = json.loads((TABLES/'figures'/(name+'.json')).read_text())
        expected = 0
        for source, digest in meta['sources'].items():
            path = TABLES/(source+'.csv')
            check(name+':source:'+source, hashlib.sha256(path.read_bytes()).hexdigest() == digest)
            with open(path) as f:
                expected += len(list(csv.DictReader(f)))
        with open(TABLES/'figures'/(name+'.csv')) as f:
            check(name+':CSV_rows', len(list(csv.DictReader(f))) == expected)
    for document in ['v2_2_result.md', 'v2_2_atlas.md', 'v2_2_requirements_result.md']:
        path = ROOT/'docs'/document
        for i, link in enumerate(re.findall(r'\]\(([^)]+)\)', path.read_text())):
            if not link.startswith(('http:', 'https:', '#')):
                check(document+':link'+str(i), (path.parent/link.split('#')[0]).exists())
    unchanged = subprocess.check_output(['git', 'diff', 'fbb8ea0', '--', 'src/photdis/probability_v2_1.py', 'scripts/v2_1_joint.py', 'docs/v2_1_protocol.md'], cwd=ROOT, text=True)
    check('frozen_v2_1_code_protocol', not unchanged)
    (TABLES/'delivery_validation.json').write_text(json.dumps(dict(actual_checks=len(checks), all_passed=all(checks.values()), checks=checks, compute_checks=compute['actual_checks'], visual_review='Recorded separately after viewing all seven plates and representative PDF renders'), indent=2)+'\n')
    print('DELIVERY', len(checks), 'checks passed')


if __name__ == '__main__':
    main()
