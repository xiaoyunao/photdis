#!/usr/bin/env python3
"""Audit scoped execution and summarize fixed independent-scale comparisons."""
import csv
import json

import numpy as np

from faint_signed_20261120 import C, CONFIG, OUT, inputs, projections
from photdis.stage1 import require_compute
from v2_1_near import csv_rows, load, save_json, sha


def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def cluster_summary(folder):
    roster = json.loads((folder/'cluster_freeze.json').read_text())
    assert roster['config']==sha(CONFIG) and roster['inventory']==sha(folder/'cluster_native_inventory.csv')
    rows = table(folder/'cluster_fixed_scores.csv')
    output = []
    for name,ids in roster['admitted'].items():
        region = [r for r in rows if r['system']==name]
        np.testing.assert_array_equal([int(r['ls_id']) for r in region],ids)
        assert len(set(ids))==len(ids) and all(r['TYPE']=='PSF' for r in region)
        q = load(OUT/(name+'_fixed_pdfs.npz'))
        assert (q['TYPE']=='PSF').all()
        np.testing.assert_array_equal(q['ls_id'],ids)
        np.testing.assert_allclose(q['mass'].sum(1),1,atol=1e-10)
        value = np.array([float(r['delta_mu']) for r in region])
        crps = np.array([float(r['crps']) for r in region])
        rmag = np.array([float(r['r']) for r in region])
        lower = np.array([r['lower_sequence_proxy']=='True' for r in region])
        bricks = (q['ls_id']>>20) & ((1<<20)-1)
        for group,mask in [('all',np.ones(len(value),bool)),('bright',rmag<20),
                           ('faint_lower_sequence',(rmag>=20)&(rmag<22)&lower)]:
            a=value[mask]
            if not len(a):
                continue
            relative = bricks[mask]
            bins = [np.flatnonzero(relative==k) for k in np.unique(relative)]
            rng=np.random.default_rng(C['seed'])
            samples = np.array([np.median(a[np.concatenate([bins[k] for k in draw])])
                                for draw in rng.integers(0,len(bins),size=(600,len(bins)))])
            center = np.median(a)
            original = C['cluster_candidates'][name]
            ci = np.quantile(samples,[.025,.975])
            sensitivity = np.quantile(samples+rng.normal(0,np.hypot(original['rrl_sigma_mag'],.1),len(samples)),[.025,.975])
            output.append(dict(system=name,subset=group,n=len(a),bricks=len(bins),
                reference_mu=original['rrl_mu'],bias=center,nmad=1.4826*np.median(abs(a-center)),
                severe=np.mean(abs(a)>1),crps=crps[mask].mean(),brick_lo=ci[0],brick_hi=ci[1],
                shared_reference_lo=sensitivity[0],shared_reference_hi=sensitivity[1],
                bias_plus_032_anchor=center-.32,bias_minus_032_anchor=center+.32))
    csv_rows(folder/'cluster_scale_summary.csv',output)


def audit():
    require_compute()
    folder = OUT/'results'
    gate = json.loads((OUT/'power/gate.json').read_text())
    assert gate['config']==sha(CONFIG) and gate['power']==sha(OUT/'power/power.csv')
    seen,checks = [],[]
    for cell,h in gate['pilot'].items():
        path=OUT/'pilot'/cell
        done=path/'complete.json'
        assert sha(done)==h
        receipt=json.loads(done.read_text())
        assert all(sha(path/k)==v for k,v in receipt['outputs'].items())
        t,e,w=inputs(path)
        y=load(path/'signed_observation.npz')
        np.testing.assert_array_equal(t['ls_id'],y['ls_id'])
        seen.extend(t['source_id'])
        np.testing.assert_allclose(w.sum(axis=1),1,atol=1e-10)
        assert np.all(y['parallax']/t['gaia_parallax_error']<5)
        p8=projections(e[:12],w[:12],t['gaia_parallax_error'][:12],8)
        p32=projections(e[:12],w[:12],t['gaia_parallax_error'][:12],32)
        error=np.max(abs(p8-p32))
        assert error<1e-5
        checks.append(dict(cell=cell,n=receipt['n'],negative=int((y['parallax']<0).sum()),
                           projection_error=float(error)))
    assert len(seen)==len(set(seen))
    summary = table(folder/'signed_summary.csv')
    pit = table(folder/'pit.csv')
    snr = table(folder/'signed_snr.csv')
    for r in summary:
        def same(q):
            return all(q[k]==r[k] for k in ('mode','field','r_lo'))
        n=int(r['n'])
        assert sum(int(q['count']) for q in pit if same(q))==n
        assert sum(int(q['observed']) for q in snr if same(q))==n
        assert abs(sum(float(q['predicted']) for q in snr if same(q))-n)<1e-8
        if r['mode']=='full':
            assert int(r['r_lo']) in gate['approved_r_lo']
    cluster_summary(folder)
    archive = json.loads((folder/'population_receipt.json').read_text())
    assert all(sha(folder/k)==v for k,v in archive['outputs'].items())
    stars = table(folder/'population_stars.csv')
    assert len(stars)==len({r['ls_id'] for r in stars}) and all(r['TYPE']=='PSF' for r in stars)
    picture = ('TYPE','field','group','match_weight','gaia_bp_rp','gaia_MG','gr','rz')
    csv_rows(folder/'population_figure_input.csv',
             ({k:r[k] for k in picture} for r in stars if float(r['match_weight'])>0))
    tables={p.name:sha(p) for p in sorted(folder.glob('*.csv'))}
    save_json(folder/'validation.json',dict(config=sha(CONFIG),power=gate['power'],
        pilot_checks=checks,unique_pilot_stars=len(seen),authorized_nine_field_bins=gate['approved_r_lo'],
        bright_archive_stars=len(stars),outputs=tables))
    print('Verified signed conservation, actual-source integrals, archived populations and candidates.',flush=True)


if __name__=='__main__':
    audit()
