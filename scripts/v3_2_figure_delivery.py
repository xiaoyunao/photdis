#!/usr/bin/env python3
"""Authenticate Photdis v3.2 plates and independently recorded visual checks."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'manifests/v3_2'
FIGURES = ROOT / 'figures/v3_2'
GROUPS = {
    'external_validation': ('final_validation_dr20/figure.json', ('dr20_reliability', 'gaia_and_hst')),
    'opened_audits': ('opened_audits_figure/complete.json', ('probe_and_count_audit',)),
    'opened_diagnostics': ('opened_diagnostics/figure.json', ('class_reliability',)),
    'radial_design': ('radial_design/figure.json', ('selected_membership',)),
    'structure_odds_audit': ('structure_odds_audit/figure.json', ('fixed_distance_odds',)),
    'tilt_factorial': ('tilt_report/initial/figure.json', ('opened_factorial',)),
    'tilt_size_control': ('tilt_size_control/comparison/figure.json', ('size_control',)),
    'radial_recovery': ('radial_report/figure.json', ('fixed_coverage', 'prior_coverage')),
    'structure_joint': ('structure_joint/figure.json', ('likelihood_contrast',)),
}
FINAL_GROUPS = {'radial_recovery', 'structure_joint'}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def check_formats(png, pdf):
    with Image.open(png) as image:
        image.load()
        width, height = image.size
        if image.format != 'PNG' or min(width, height) < 1000 or not all(
            abs(v-300) < 1 for v in image.info.get('dpi', (0, 0))
        ):
            raise ValueError(f'Invalid PNG size or resolution: {png}')
    reader = PdfReader(pdf)
    if len(reader.pages) != 1:
        raise ValueError(f'Expected a one-page PDF: {pdf}')
    page = reader.pages[0]
    if (abs(float(page.mediabox.width)*300/72-width) > 2
            or abs(float(page.mediabox.height)*300/72-height) > 2):
        raise ValueError(f'PDF and PNG sizes differ: {pdf}')
    content = page.extract_text()
    if not re.search(r'[A-Za-z]{3,}', content) or re.search(r'[\u2e80-\u9fff\uf900-\ufaff]', content):
        raise ValueError(f'Expected extractable English figure labels: {pdf}')
    return [width, height]


def audit(*, completed_only=False):
    review_path = BASE/'figure_visual_review.json'
    review = json.loads(review_path.read_text())
    atlas_path = ROOT/'docs/v3_2_atlas.md'
    atlas = atlas_path.read_text()
    inputs = {str(p.relative_to(ROOT)): digest(p) for p in (review_path, atlas_path)}
    inventory = {}
    groups = [group for group, (receipt_name, _) in GROUPS.items()
              if not completed_only or group not in FINAL_GROUPS or (BASE/receipt_name).is_file()]
    for group in groups:
        receipt_name, stems = GROUPS[group]
        receipt = BASE/receipt_name
        record = json.loads(receipt.read_text())
        inputs[str(receipt.relative_to(ROOT))] = digest(receipt)
        sources = record.get('sources', record)
        paths = {FIGURES/group/f'{stem}.{ext}' for stem in stems for ext in ('pdf', 'png')}
        recorded = {ROOT/name for name in sources if name.startswith('figures/v3_2/')}
        if recorded != paths:
            raise ValueError(f'Wrong figure inventory: {receipt}')
        if {p for p in (FIGURES/group).iterdir() if p.suffix in {'.png', '.pdf'}} != paths:
            raise ValueError(f'Unregistered files in {group}')
        for name, checksum in sources.items():
            path = ROOT / name
            if digest(path) != checksum:
                raise ValueError(f'Figure receipt source changed: {name}')
            inputs[name] = checksum
        for stem in stems:
            png, pdf = (FIGURES/group/f'{stem}.{ext}' for ext in ('png', 'pdf'))
            key = str(png.relative_to(ROOT))
            item = review[key]
            if item['sha256'] != sources[key] or item['english_labels'] is not True or item['visually_reviewed'] is not True:
                raise ValueError(f'Visual review not bound to this version: {key}')
            if any(f'../{p.relative_to(ROOT)}' not in atlas for p in (png, pdf)):
                raise ValueError(f'Figure is missing from the atlas: {group}/{stem}')
            inventory[f'{group}/{stem}'] = dict(png=key, pdf=str(pdf.relative_to(ROOT)), pixels=check_formats(png, pdf))
    if not completed_only and set(review) != {r['png'] for r in inventory.values()}:
        raise ValueError('Visual review inventory differs from the final plates')
    return dict(passed=True, complete=not completed_only, plates=len(inventory), inventory=inventory, inputs=inputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completed-only', action='store_true', help='Check plates with available receipts')
    print(json.dumps(audit(completed_only=parser.parse_args().completed_only), indent=2))
