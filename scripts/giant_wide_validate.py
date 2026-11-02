#!/usr/bin/env python3
"""Verify real row identities, paired controls, threshold budgets and aggregates."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.deps')]
import joblib
import numpy as np
from photdis.stage1 import require_compute
from cold_v4 import sha
from distance_comparison_plot import distance
OUT=Path('data/giant_wide_v1')


def main():
    require_compute();raw=json.loads((OUT/'results.json').read_text());summary=json.loads((OUT/'summary.json').read_text())
    with np.load('data/giant_information_v1/labels.npz') as f:t={k:f[k] for k in f.files}
    seen=[];checks=[]
    for rec in raw['folds']:
        assert not set(rec['training_fields']) & set(rec['calibration_fields'])
        assert rec['test_field'] not in rec['training_fields']+rec['calibration_fields']
        field=rec['test_field'];models={name:joblib.load(OUT/f'{field}_{name}.joblib') for name in rec['models']}
        np.testing.assert_array_equal(models['B-K']['scale'].mean_,models['B-wide']['scale'].mean_)
        np.testing.assert_array_equal(models['B-K']['scale'].scale_,models['B-wide']['scale'].scale_)
        with np.load(OUT/f'{field}_scores.npz') as f:
            rows=f['rows'];cal=f['calibration_rows'];seen.extend(t['identity_group'][rows].tolist())
            assert not set(t['identity_group'][rows])&set(t['identity_group'][cal])
            assert set(t['field_index'][rows])=={field}
            for name,m in rec['models'].items():
                assert m['model_sha256_before_calibration']==m['model_sha256_after_test']==sha(OUT/f'{field}_{name}.joblib')
                assert np.isclose(m['total_weight'],rec['n_reference'])
                np.testing.assert_allclose(m['class_weight_totals'],rec['n_reference']/2)
                assert np.isfinite(f['score_'+name]).all()
                for domain,d in m['domains'].items():
                    for key,rule in d['thresholds'].items():
                        if key!='0.5':
                            assert rule['calibration_false_positives']<=rule['allowed_false_positives']
                            assert rule['calibration_fpr']['rate']<=float(key)+1e-12
                        chosen=(f['score_'+name]>=rule['threshold'])&~f['outside_'+domain+'_'+name]
                        stats=d['test'][key]['strata']['all']
                        assert chosen.sum()==stats['reference_total']['success']
                        k=(t['label'][rows]==1)&(t['desi_teff'][rows]>=4200)&(t['desi_teff'][rows]<5000)
                        assert (chosen&k).sum()==stats['K_giant_total']['success']
    assert len(seen)==len(set(seen))
    checks+=['unique held-out source groups across folds','train/cal/test role isolation','matched B scaling','constant total/class weights','serialized model unchanged across calibration/test','threshold calibration budgets','real-score selection counts']
    for domain in summary['aggregate']:
        for key in summary['aggregate'][domain]:
            for name,a in summary['aggregate'][domain][key].items():
                expected=sum(r['models'][name]['domains'][domain]['test'][key]['strata']['all']['K_giant_total']['success'] for r in raw['folds'])
                assert expected==a['strata']['all']['K_giant_total']['success']
    checks.append('aggregate counts equal field sums')
    np.testing.assert_allclose(distance([5.,10.,15.]),[.1,1.,10.])
    dist=json.loads(Path('data/distance_comparison_v1/summary.json').read_text())
    assert dist['warm']['reproduced_summary']['n']==588
    for j,item in enumerate(dist['cold']):
        with np.load(f'data/distance_comparison_v1/cold_{j}.npz') as f:
            assert len(f['source_id'])==item['n'] and len(np.unique(f['source_id']))==item['n']
            relative=f['phot_kpc']/f['gaia_kpc']-1
            assert np.isclose(np.median(relative),item['relative_median'])
    checks+=['distance modulus to kpc normalization','frozen warm reserved summary reproduced','cold figure row counts and residuals']
    report={'status':'passed','checks':checks,'n_checks':len(checks),'n_test_identities':len(seen),
            'input_sha256':{str(p):sha(p) for p in [OUT/'results.json',OUT/'summary.json',Path('data/distance_comparison_v1/summary.json')]}}
    (OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
