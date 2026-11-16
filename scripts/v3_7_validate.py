#!/usr/bin/env python3
"""Check recorded products and paper denominators from the compact delivery."""
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from v3_7_collect import CSV, ROOT

MANIFEST = ROOT/'manifests/v3_7'
FIGURES = ROOT/'figures/v3_7'
DISTANCE = ('0–2','2–5','5–10','10–20','20–50','>50')


def table(stage,name):
    with (MANIFEST/stage/name).open() as f:
        return list(csv.DictReader(f))


def outputs():
    checks = 0
    for stage,files in CSV.items():
        root = MANIFEST/stage
        expected = json.loads((root/'complete.json').read_text())['outputs']
        small = json.loads((root/'compact.json').read_text()) if (root/'compact.json').exists() else {}
        for name in files:
            if name in small:
                assert expected[name]==small[name]['complete_sha256'],(stage,name)
            assert sha256((root/name).read_bytes()).hexdigest()==(
                small[name]['compact_sha256'] if name in small else expected[name]),(stage,name)
            checks += 1
    selection = MANIFEST/'selection'
    derived = json.loads((selection/'derived.json').read_text())
    for name,digest in (derived['inputs']|derived['outputs']).items():
        assert sha256((selection/name).read_bytes()).hexdigest()==digest,name
        checks += 1

    for figure in sorted((MANIFEST/'figures').glob('F*.json')):
        info = json.loads(figure.read_text())
        assert info['dpi']==300
        for name,expected in info['outputs'].items():
            p = (FIGURES if name.endswith(('.pdf','.png')) else figure.parent)/name
            assert sha256(p.read_bytes()).hexdigest()==expected,name
            checks += 1
    assert len(list((MANIFEST/'figures').glob('F*.json')))==6
    return checks


def migration():
    dense = defaultdict(dict)
    for row in table('reference_comparison','qualified_migration.csv'):
        cell = tuple(row[k] for k in ('source','selection','history','reference_class','reference_shell'))
        dense[cell][row['predicted_shell']] = row
    for k,cell in dense.items():
        assert set(cell)==set(DISTANCE),k
        n = int(cell['0–2']['selected'])
        assert sum(int(r['hard_count']) for r in cell.values())==n,k
        np.testing.assert_allclose(sum(float(r['pdf_mass']) for r in cell.values()),n,rtol=1e-12)
        assert n<=int(cell['0–2']['reference_denominator']),k
    checks = len(dense)
    for r in table('selection','reference.csv'):
        k = tuple(r[f] for f in ('source','selection','history','reference_class','reference_shell'))
        n = int(r['n'])
        assert sum(int(r[f'hard_{i}']) for i in range(len(DISTANCE)))==n,k
        np.testing.assert_allclose(sum(float(r[f'pdf_{i}']) for i in range(len(DISTANCE))),n,rtol=1e-12)
        assert n<=int(r['reference_denominator']),k
        if r['domain']=='all' and r['selection'] in ('primary','strict'):
            for i,d in enumerate(DISTANCE):
                old = dense[k][d]
                assert int(old['hard_count'])==int(r[f'hard_{i}']),k
                np.testing.assert_allclose(float(old['pdf_mass']),float(r[f'pdf_{i}']),rtol=1e-12)
        checks += 1
    return checks


def calibration():
    weak = table('weak_gaia','scores.csv')
    all_rows = [r for r in weak if r['selection']=='supported' and r['photometry']=='all' and r['history']=='all'][0]
    receipt = json.loads((MANIFEST/'weak_gaia/complete.json').read_text())
    assert int(all_rows['n'])==receipt['rows']
    assert 0<int(all_rows['negative'])<receipt['rows']
    for r in weak:
        assert 0<int(r['blocks'])<=int(r['n'])
        assert np.isfinite(float(r['qrf_minus_broad']))
    odds = table('lf_odds','odds_leakage.csv')
    for r in odds:
        den = float(r['denominator'])
        if den:
            v = float(r['fraction'])
            assert -1e-12<=v<=1.+1e-12
            np.testing.assert_allclose(v,float(r['expected'])/den)
    cases = table('lf_known_branches','case_probabilities.csv')
    for r in cases:
        assert r['TYPE']=='PSF'
        assert 0.<=float(r['correct_reference_shell_probability'])<=1.+1e-12
        if r['correct_joint_branch_probability']:
            assert 0.<=float(r['correct_joint_branch_probability'])<=1.+1e-12
    classifier = table('classifier_sky','retention.csv')
    for r in classifier:
        assert r['stored']==r['published_replayed'],r
        assert int(r['archived_score_agreement'])<=int(r['n']),r
    neighborhoods = table('blue_neighborhood','neighborhoods.csv')
    counts = Counter((r['cohort'],r['ls_id']) for r in neighborhoods if r['TYPE']=='PSF')
    assert len(counts)==305 and set(counts.values())=={7} and sum(counts.values())==len(neighborhoods)
    return len(weak)+len(odds)+len(cases)+len(classifier)+len(neighborhoods)


if __name__=='__main__':
    print(f'Validated {outputs()+migration()+calibration():,} output and science-table checks')
