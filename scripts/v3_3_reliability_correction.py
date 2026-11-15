#!/usr/bin/env python3
"""Repair roundoff at probability endpoints in opened development reporting."""
import json
import os
from pathlib import Path

import numpy as np

from photdis.stage1 import require_compute
from v2_1_near import ROOT, csv_rows, load, save_json, sha
from v3_3_development import OUT as DEVELOPMENT
from v3_3_development_report import COLORS, LABELS, VIEWS, plt, reliability
from v3_3_sed import verified

OUT = DEVELOPMENT.parent/'reliability_corrected'


def main():
    require_compute()
    marker = DEVELOPMENT/'complete.json'
    receipt = json.loads(marker.read_text())
    views = {name: load(verified(DEVELOPMENT/f'{name}.npz', receipt['outputs'][f'{name}.npz']))
             for name in VIEWS}
    qrf = views['qrf']
    reference = qrf['reference_mu']
    matched = np.isfinite(reference) & qrf['supported']
    label = qrf['reference_class']
    rows, endpoints = [], {}
    for name, t in views.items():
        np.testing.assert_array_equal(t['identity'], qrf['identity'])
        np.testing.assert_array_equal(t['reference_mu'], reference)
        np.testing.assert_array_equal(t['reference_class'], label)
        if not (t['TYPE'] == 'PSF').all():
            raise ValueError('Expected the original development PSFs')
        groups = [('Nearer than 10 kpc', t['qnear'], reference < 15., matched)]
        if name != 'qrf':
            groups.append(('Giant class', t['class_probability'][:, 1], label == 1, label >= 0))
        for target, p, truth, use in groups:
            v = p[use]
            if not np.isfinite(v).all() or (v < -1e-12).any() or (v > 1+1e-12).any():
                raise ValueError(f'Non-roundoff probability violation for {name}/{target}')
            endpoints[f'{name}/{target}'] = int(((v < 0) | (v > 1)).sum())
            points = list(reliability(np.clip(p, 0, 1), truth, use))
            if sum(r['n'] for r in points) != int(use.sum()):
                raise ValueError(f'Missing reliability rows for {name}/{target}')
            rows.extend(dict(view=name, target=target, **point) for point in points)
    OUT.mkdir(parents=True, exist_ok=False)
    csv_rows(OUT/'reliability.csv', rows)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
    for ax, target in zip(axes, ('Giant class', 'Nearer than 10 kpc')):
        for name, title, color in zip(VIEWS, LABELS, COLORS):
            points = [p for p in rows if p['view'] == name and p['target'] == target]
            if points:
                ax.plot([r['predicted'] for r in points], [r['observed'] for r in points], 'o-', ms=4, lw=1,
                        color=color, label=title)
        ax.plot([0, 1], [0, 1], ':', color='0.4')
        ax.set(xlim=(0, 1), ylim=(0, 1), title=target, xlabel='Mean predicted probability',
               ylabel='Fraction in development references')
        ax.legend(frameon=False, fontsize=8)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT/f'reference_reliability.{ext}', dpi=220)
    plt.close(fig)
    save_json(OUT/'complete.json', dict(job=os.getenv('SLURM_JOB_ID'), matched_near=int(matched.sum()),
        roundoff_endpoints=endpoints, development_only=True,
        inputs={str(p.relative_to(ROOT)): sha(p) for p in (marker, Path(__file__))},
        outputs={p.name: sha(p) for p in sorted(OUT.iterdir())}))


if __name__ == '__main__':
    main()
