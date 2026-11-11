#!/usr/bin/env python3
"""Frozen NGC288 scale check and photometric marginalization demonstration.

Membership is Gaia-conditioned; its dynamical distance is a shared system-scale
constraint, not 1442 independent distance measurements. No model is fitted here.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, atomic_npz, csv_rows
from photdis.stage1 import require_compute
from photdis.v3.metrics import distance_summary, normal_reference_score
from v3_0_validation_inputs import check_freeze
D = ROOT/'data/v3_0'


def main():
    require_compute(); p = argparse.ArgumentParser(); p.add_argument('--target', default=''); a = p.parse_args()
    method = check_freeze()
    source = load(D/'validation/independent_scale/photometry.npz')
    membership = load(D/'independent_scale/membership.npz')
    original = load(D/'independent_scale/photometry.npz')
    np.testing.assert_array_equal(source['identity'], membership['identity'])
    np.testing.assert_array_equal(source['identity'], original['identity'])
    cfg = json.loads((ROOT/'configs/v3_0_independent_scale.json').read_text())
    association = json.loads((D/'independent_scale/association.json').read_text())
    assert association['method_freeze_sha256'] == method
    base = D/'validation/independent_scale'
    folder = Path(a.target) if a.target else base/'predictions'
    marker = folder/'complete.json'; record = json.loads(marker.read_text())
    if a.target: assert record['integration_passed']
    name = folder.name if a.target else 'reference'
    out = D/'scale_example'/name; out.mkdir(parents=True, exist_ok=True)
    inputs = {str(x.relative_to(ROOT)):sha(x) for x in [marker, D/'independent_scale/membership.npz',
        D/'independent_scale/association.json', ROOT/'configs/v3_0_independent_scale.json', Path(__file__)]}
    mu0 = 10+5*np.log10(cfg['distance_kpc'])
    mu_lo = 10+5*np.log10(cfg['distance_kpc']-cfg['distance_lower_error_kpc'])
    mu_hi = 10+5*np.log10(cfg['distance_kpc']+cfg['distance_upper_error_kpc'])
    n = len(source['identity']); quantiles = np.full((n,5),np.nan); probability = np.zeros((n,3,6))
    qnear = np.full(n,np.nan); logscore = qnear.copy(); seen = np.zeros(n,bool)
    bins = np.arange(-5,30.0001,.05); centers = (bins[1:]+bins[:-1])/2
    expected = {k:np.zeros(len(centers)) for k in ('all','members','nonmembers')}
    count = {k:0 for k in expected}
    paths = (folder/'chunks' if a.target else folder).glob('chunk_*.npz')
    for path in sorted(paths):
        if path.stem.endswith(('_joint','_counts')): continue
        s = load(path); meta = json.loads(path.with_suffix('.json').read_text())
        assert sha(path) == (meta['outputs'][path.name] if a.target else meta['summary_sha256'])
        packed = s if a.target else load(path.with_name(path.stem+'_joint.npz'))
        if not a.target: assert sha(path.with_name(path.stem+'_joint.npz')) == meta['joint_sha256']
        order = np.argsort(source['identity']); rows = order[np.searchsorted(source['identity'][order],s['identity'])]
        np.testing.assert_array_equal(source['identity'][rows],s['identity'])
        assert not seen[rows].any() and (s['TYPE']=='PSF').all(); seen[rows] = True
        quantiles[rows] = s['mu']; probability[rows] = s['class_distance']; qnear[rows] = s['qnear']
        for j,row in enumerate(rows):
            lo,hi = packed['full_grid_offset'][j:j+2]
            grid,lp = packed['full_mu'][lo:hi],packed['full_logpdf'][lo:hi]
            density = np.exp(lp-np.max(lp)); area = np.diff(grid)*(density[1:]+density[:-1])/2
            cdf = np.r_[0,np.cumsum(area)]; cdf /= cdf[-1]
            mass = np.diff(np.interp(bins,grid,cdf,left=0,right=1))
            for group in ('all','members' if membership['selected_member'][row] else 'nonmembers'):
                expected[group] += mass; count[group] += 1
            # Symmetrized reported scale uncertainty is only a per-star score.
            # The system-scale error must never be divided by sqrt(member count).
            logscore[row] = normal_reference_score(grid,lp,mu0,(mu_hi-mu_lo)/2)
    assert seen.all()
    use = membership['selected_member']
    groups = {'all_photometric':np.ones(n,bool), 'members':use, 'nonmembers':~use,
              'members_near95':use&(qnear>.95)}
    stats = []
    for group,mask in groups.items():
        for label,value in [('central',mu0),('scale_minus_1sigma',mu_lo),('scale_plus_1sigma',mu_hi)]:
            stats.append(dict(group=group, scale=label, **distance_summary(quantiles[mask],np.full(mask.sum(),value)),
                mean_scale_convolved_logscore=float(logscore[mask].mean()) if mask.any() else None))
    atomic_npz(out/'per_star.npz', identity=source['identity'], TYPE=source['TYPE'], RA=source['RA'],DEC=source['DEC'],
        radius_deg=original['radius_deg'], selected_member=use,membership_probability=membership['membership_probability'],
        mu=quantiles, probability=probability,qnear=qnear,scale_logscore=logscore)
    csv_rows(out/'metrics.csv',stats)
    csv_rows(out/'distance_distribution.csv',[dict(group=k, mu=centers[i],lower_mu=bins[i],upper_mu=bins[i+1],
        expected_count=float(v),n=count[k]) for k in expected for i,v in enumerate(expected[k])])
    csv_rows(out/'joint_counts.csv',[dict(group=g,class_index=c,distance_bin=b,n=int(mask.sum()),
        expected_count=float(probability[mask,c,b].sum())) for g,mask in groups.items() for c in range(3) for b in range(6)])
    save_json(out/'complete.json',dict(inputs=inputs,method_freeze=method,n=n,members=int(use.sum()),
        scope='Photometric-first 1deg field. Membership selected after prediction at frozen p>=0.9; Gaia-conditioned membership, independently anchored dynamical scale. All-photometric/nonmember scale residuals are foreground diagnostics, not accuracy measures.',
        uncertainty='Shared dynamical distance uncertainty shown by common +/-1sigma shifts. Mean per-star convolved score is descriptive, not product likelihood for the common cluster distance. Stacked probabilities are expectations; shared target count covariance is in the target product, not independent-star error bars.',
        distance_kpc=cfg['distance_kpc'],distance_error_lower=cfg['distance_lower_error_kpc'],distance_error_upper=cfg['distance_upper_error_kpc'],
        outputs={x.name:sha(x) for x in out.iterdir() if x.suffix in ('.csv','.npz')}, job=os.getenv('SLURM_JOB_ID')))
    print('SCALE EXAMPLE',name,n,int(use.sum()),[r for r in stats if r['group']=='members' and r['scale']=='central'],flush=True)


if __name__=='__main__': main()
