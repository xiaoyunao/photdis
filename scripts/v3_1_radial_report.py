#!/usr/bin/env python3
"""Summarize all registered radial controls without hiding numerical failures."""
import csv
import json
import os
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, csv_rows
from photdis.stage1 import require_compute
from photdis.v3.compression import DISTANCE_MU_BOUNDARIES
from photdis.v3_1.radial_products import generator_joint_parameters


def quantile(values,weights,levels=(.025,.5,.975)):
    order=np.argsort(values); w=np.asarray(weights,float)[order]
    assert np.isfinite(w).all() and (w>=0).all() and w.sum()>0
    cumulative=np.cumsum(w/w.sum()); cumulative[-1]=1.
    return np.asarray(values)[order[np.searchsorted(cumulative,levels,side='left')]]


def main():
    require_compute(); root=ROOT/'data/v3_1'; out=root/'radial_report'; out.mkdir(exist_ok=True)
    paths=[root/'radial_catalogs/catalogs.json',root/'radial_catalogs/truth.npz',
        ROOT/'scripts/v3_1_radial_report.py',ROOT/'src/photdis/v3_1/radial_products.py']
    inputs={str(p.relative_to(ROOT)):sha(p) for p in paths}
    catalogs=json.loads(paths[0].read_text()); truth=load(paths[1]); assert len(catalogs)==176
    status=[]; recovery=[]; totals=[]; structures=[]; distance=[]; covariances=[]; missing=[]
    for catalog in catalogs:
        c=catalog['catalog']; use=truth['catalog']==c; t={k:v[use] for k,v in truth.items()}
        assert len(t['identity'])==128
        bins=np.searchsorted(DISTANCE_MU_BOUNDARIES,t['mu'],side='right')
        realized=np.bincount(t['class_label']*6+bins,minlength=18)
        for arm in ('broad','local'):
            marker=root/'radial_refinement'/f'catalog_{c:03}_{arm}/complete.json'
            base=dict(catalog=c,scenario=catalog['scenario'],repeat=catalog['repeat'],field=catalog['field'],arm=arm)
            if not marker.exists(): missing.append(base); continue
            refinement=json.loads(marker.read_text()); inputs[str(marker.relative_to(ROOT))]=sha(marker)
            attempt=refinement['attempts'][-1]; folder=ROOT/attempt['recovery']; record=json.loads(folder.read_text())
            assert sha(folder)==attempt['sha256']; folder=folder.parent
            for name in ('counts.npz','population_parameters.npz','per_star.npz','distance.csv'):
                assert sha(folder/name)==record['outputs'][name]
            fine_path=attempt.get('fine'); fine=None
            if fine_path:
                fine_marker=ROOT/fine_path; assert sha(fine_marker)==attempt['fine_sha256']; fine=json.loads(fine_marker.read_text())
                for name,digest in fine['outputs'].items(): assert sha(fine_marker.parent/name)==digest
            accepted=bool(refinement['passed']); usefine=bool(fine and fine['passed'])
            count=load(fine_marker.parent/'counts.npz' if usefine else folder/'counts.npz')
            parameters=load(folder/'population_parameters.npz'); stars=load(folder/'per_star.npz')
            np.testing.assert_array_equal(stars['identity'],t['identity']); assert (stars['TYPE']=='PSF').all()
            weights=count['draw_weights']; fractions=parameters['joint_fractions']; draws=count['count_draws']
            assert len(weights)==len(fractions)==len(draws)
            assert draws.dtype==np.int64 and (draws.sum(axis=1)==128).all()
            if usefine:
                correction=load(fine_marker.parent/'correction.npz')
                np.testing.assert_array_equal(correction['identity'],t['identity'])
                np.testing.assert_array_equal(correction['weights'],weights)
                np.testing.assert_array_equal(correction['draw_ids'],count['draw_ids'])
            covariance=count['covariance']; eigen=np.linalg.eigvalsh(covariance)
            assert abs(covariance.sum(axis=1)).max()<1e-6 and eigen.min()>-1e-6
            assert abs(count['mean'].sum()-128)<1e-8
            state=dict(**base,numerically_passed=accepted,selected_draws=refinement['selected_draws'],
                fine_weighted=usefine,max_rhat=record['max_rhat'],minimum_ess=record['minimum_ess'],
                divergent=record['divergent'],marginal_half_delta=count['marginal_half_delta'],
                covariance_min_eigen=float(eigen.min()),covariance_sum_max=float(abs(covariance.sum(axis=1)).max()))
            status.append(state)
            covariances.append(dict(**base,numerically_passed=accepted,covariance=covariance.tolist(),
                shared_covariance=count['shared_covariance'].tolist(),conditional_covariance=count['conditional_covariance'].tolist()))
            for name,values,true,mean in [('realized_selected_count',draws,realized,count['mean']),
                ('pre_entry_population_fraction',fractions,generator_joint_parameters(catalog).ravel(),weights@fractions)]:
                lohi=np.array([quantile(values[:,i],weights) for i in range(18)])
                covered=(lohi[:,0]<=true)&(true<=lohi[:,2])
                for i in range(18):
                    recovery.append(dict(**base,numerically_passed=accepted,fine_weighted=usefine,estimand=name,
                        class_index=i//6,distance_bin=i%6,truth=float(true[i]),mean=float(mean[i]),
                        lower95=float(lohi[i,0]),upper95=float(lohi[i,2]),covered95=bool(covered[i])))
                totals.append(dict(**base,numerically_passed=accepted,fine_weighted=usefine,estimand=name,
                    absolute_L1=float(abs(mean-true).sum()),fraction_L1=float(abs(mean-true).sum()/(128 if name=='realized_selected_count' else 1)),
                    bins_covered95=int(covered.sum()),bins=18,nonzero_truth_bins=int((true>0).sum()),
                    nonzero_truth_covered95=int(covered[true>0].sum())))
            if arm=='local':
                for key,col in [('location_mu',3),('width_mu',4)]:
                    ref=record['structure'][key]['truth']; q=quantile(parameters['structure'][:,col],weights)
                    structures.append(dict(**base,numerically_passed=accepted,fine_weighted=usefine,parameter=key,
                        truth=ref,lower95=float(q[0]),median=float(q[1]),upper95=float(q[2]),
                        covered95=None if ref is None else bool(q[0]<=ref<=q[2])))
            with (folder/'distance.csv').open() as stream:
                for row in csv.DictReader(stream): distance.append(dict(**base,numerically_passed=accepted,grid='coarse0.02',**row))
            distance.append(dict(**base,numerically_passed=accepted,grid='coarse0.02',group='proper_scores',
                n=128,mean_truth_mu_logscore=float(stars['logscore'].mean()),
                mean_signed_gaia_logscore=float(stars['gaia_logscore'].mean()),
                signed_gaia_tail_fraction=float(((stars['gaia_pit']<.025)|(stars['gaia_pit']>.975)).mean())))
    save_json(out/'progress.json',dict(expected=352,available=len(status),missing=missing))
    if missing: raise RuntimeError(f'{len(missing)} registered arm outcomes not complete; no final report')
    for name,rows in [('status.csv',status),('recovery.csv',recovery),('totals.csv',totals),('structure.csv',structures),('distance.csv',distance)]:csv_rows(out/name,rows)
    save_json(out/'covariances.json',covariances)
    paired=[]
    for c in range(176):
        for estimand in ('realized_selected_count','pre_entry_population_fraction'):
            a=next(r for r in totals if r['catalog']==c and r['estimand']==estimand and r['arm']=='broad')
            b=next(r for r in totals if r['catalog']==c and r['estimand']==estimand and r['arm']=='local')
            paired.append(dict(catalog=c,scenario=a['scenario'],repeat=a['repeat'],field=a['field'],estimand=estimand,
                both_numerically_passed=a['numerically_passed'] and b['numerically_passed'],
                broad_fraction_L1=a['fraction_L1'],local_fraction_L1=b['fraction_L1'],
                local_minus_broad_fraction_L1=b['fraction_L1']-a['fraction_L1']))
    csv_rows(out/'paired.csv',paired)
    save_json(out/'complete.json',dict(inputs=inputs,outputs={p.name:sha(p) for p in out.iterdir() if p.suffix in ('.csv','.json') and p.name!='complete.json'},
        job=os.getenv('SLURM_JOB_ID'),catalogs=176,arms=352,numerically_passed=sum(r['numerically_passed'] for r in status),
        all_outcomes_retained=True,selection_approximation='Original entry tables omit local dust covariance; separately audited, not exact matched selection.',
        scope='Conditional empirical-source simulation recovery. Parameters and selected realized counts distinct; fine weights applied where accepted. Failed arms retained. Coverage includes structural-zero bins, with separate nonzero denominators. Proper scores retain coarse grid label; no independent real calibration claim.'))


if __name__=='__main__':main()
