#!/usr/bin/env python3
"""Isolate odds-axis quadrature error on the frozen NGC288 distance grid."""
import json
import os

import numpy as np
from scipy.special import logsumexp

from photdis.stage1 import require_compute
from photdis.v3_1.structure import structure_quadrature
from photdis.v3_1.structure_posterior import common_structure_posterior
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_1_structure_infer import compare
from v3_2_tilt_inputs import verified


def main():
    require_compute()
    source = ROOT / 'data/v3_1/structure/matrices'
    marker = source / 'complete.json'
    registered = ROOT / 'manifests/v3_1/structure/matrices/complete.json'
    record = json.loads(verified(marker, sha(registered)).read_text())
    if not record['passed']:
        raise ValueError('Structure entry matrices did not pass')
    data = load(verified(source / 'matrices.npz', record['outputs']['matrices.npz']))
    if not (data['TYPE'] == 'PSF').all() or len(data['identity']) != 512:
        raise ValueError('Frozen PSF structure cohort required')
    np.testing.assert_array_equal(data['response_ids'], np.arange(32))
    output = ROOT / 'data/v3_2/structure_odds_audit'
    output.mkdir(parents=True, exist_ok=False)
    checks = []
    for index, branch in enumerate(('full', 'gaussian')):
        response = data['logresponse'][1, index]
        entry = data['entry'][0]
        background = logsumexp(response[:, :, :5], axis=-1) - np.log(5)
        background_entry = entry[:, :, :5].mean(axis=-1)
        results = {}
        for order in (256, 512, 1024):
            mu, lmu, z, lz = structure_quadrature(64, order)
            indices = np.searchsorted(data['common_mu'], mu)
            np.testing.assert_array_equal(data['common_mu'][indices], mu)
            results[order] = common_structure_posterior(
                np.take(response, 5 + indices, axis=-1), background,
                np.take(entry, 5 + indices, axis=-1), background_entry,
                data['radius_deg'], mu, lmu, z, lz)
            atomic_npz(output / f'{branch}_{order}.npz', identity=data['identity'],
                       TYPE=data['TYPE'], response_ids=data['response_ids'], **results[order])
            print(branch, order, results[order]['mean'], flush=True)
        for order in (512, 1024):
            checks.append(dict(branch=branch, distance_order=64, odds_order=order,
                               previous_odds_order=order // 2,
                               **compare(results[order], results[order // 2])))
    csv_rows(output / 'checks.csv', checks)
    sources = [marker, ROOT / 'scripts/v3_2_structure_odds_audit.py',
               ROOT / 'src/photdis/v3_1/structure.py',
               ROOT / 'src/photdis/v3_1/structure_posterior.py',
               ROOT / 'scripts/v3_1_structure_infer.py']
    save_json(output / 'complete.json', dict(
        inputs={str(p.relative_to(ROOT)): sha(p) for p in sources},
        outputs={p.name: sha(p) for p in sorted(output.iterdir())}, all_PSF=True,
        job=os.getenv('SLURM_JOB_ID'), n=512, responses=32,
        scope='Odds axis only on frozen 64-point distance grid. Joint quadrature not certified.'))


if __name__ == '__main__':
    main()
