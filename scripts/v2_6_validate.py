#!/usr/bin/env python3
"""Actual scientific artifact validation and aligned estimand summaries."""
import csv
import json
import numpy as np
from scipy.special import logsumexp
from v2_1_near import ROOT,load,sha,save_json,csv_rows
from photdis.stage1 import require_compute
from photdis.population_v2_6 import responsibilities
from photdis.population_v2_5 import component_weights,observation_cell
D=ROOT/'data/v2_6';T=D/'tables'


def read(path):
    with open(path) as f:return list(csv.DictReader(f))


def main():
    require_compute();checks={}
    def check(k,v):
        checks[k]=bool(v)
        if not v:raise AssertionError(k)
    cat=load(ROOT/'data/v2_5/catalogs.npz');base=np.array(json.loads((ROOT/'data/v2_5/templates.json').read_text())['base'])
    for marker in ('population_freeze','luminosity_freeze'):
        for name,h in json.loads((D/(marker+'.json')).read_text()).items():check(marker+':'+name,sha(ROOT/name)==h)
    for directory,expected in [('observed',192),('uncertain_observed',36)]:
        paths=sorted((D/directory).glob('*.npz'));check(directory+':count',len(paths)==expected)
        for path in paths:
            p=load(path);i=int(path.stem);u=cat['catalog']==i;pre=directory+':'+str(i)
            check(pre+':identity',np.array_equal(p['identity'],cat['identity'][u]));check(pre+':PSF',np.all(p['TYPE']=='PSF'))
            check(pre+':finite',np.isfinite(p['joint']).all() and np.all(p['joint']>=0));check(pre+':normalization',np.allclose(p['joint'].sum(axis=1),1,atol=1e-12))
            check(pre+':joint_count_size',np.all(p['count_draws'].sum(axis=1)==128));check(pre+':covariance_PSD',np.linalg.eigvalsh(p['count_covariance']).min()>-1e-8)
            check(pre+':count_mean_MC',abs(p['count_draws'].mean(axis=0)-p['joint'].sum(axis=0)).max()<3)
    for regime in (0,1):
        old=load(ROOT/f'data/v2_5/posterior_regime{regime}.npz');i=int(old['catalog'][0]);p=load(D/'observed'/f'{i:03}.npz');u=cat['catalog']==i
        eta=component_weights(old['theta'][p['theta_index']],base);total=np.zeros_like(p['joint'])
        for j in range(0,len(eta),256):total+=responsibilities(eta[j:j+256],old['response'],observation_cell(cat['mags'][u])).sum(axis=0)
        check('independent_responsibility_sum:'+str(regime),np.allclose(total/len(eta),p['joint'],atol=1e-14))
    rp=load(D/'response_prior.npz');grid=load(D/'uncertain_grid.npz')
    check('response_evidence_weights',np.allclose(grid['response_weight'],np.exp(grid['log_evidence']-logsumexp(grid['log_evidence'],axis=1)[:,None])))
    check('response_theta_weights',np.allclose(grid['theta_weight'].sum(axis=2),1))
    prior=load(ROOT/'data/v2_5/cal_templates.npz');last=prior['identity']
    test=load(ROOT/'data/v2_5/test_templates.npz')['identity'];fit=load(ROOT/'data/v2_3/training.npz')['identity']
    for scale in range(3):
        c=load(D/f'templates_{scale}.npz');check(f'scale{scale}:nested',np.isin(last,c['identity']).all());last=c['identity']
        check(f'scale{scale}:PSF',np.all(c['TYPE']=='PSF'));check(f'scale{scale}:no_fit_test',not np.intersect1d(last,np.r_[test,fit]).size)
        check(f'scale{scale}:membership',np.allclose(c['member'].sum(axis=1),1) and np.all(c['member']>=0))
    for source in ('specdis','spdist'):
        p=load(D/f'photometric_luminosity_{source}.npz');c=load(D/f'luminosity_posterior_{source}.npz')
        check(source+':identity_isolation',not np.intersect1d(p['identity'],c['identity']).size);check(source+':PSF',np.all(p['TYPE']=='PSF') and np.all(c['TYPE']=='PSF'))
        for side in ('before','after'):check(source+':latent_'+side,np.allclose(p['joint_'+side].sum(axis=(1,2)),1))
        check(source+':calibration_weight',np.isclose(c['weight'].sum(),1))
    # Same realized-catalog estimand for all methods, including QRF >50kpc mass.
    old=read(ROOT/'data/v2_5/tables/recovery_distance.csv');tails={int(r['catalog']):r for r in read(ROOT/'data/v2_5/tables/recovery_beyond50.csv')}
    new=read(T/'observed_counts.csv')+read(T/'uncertain_counts.csv');summary=[]
    for design in np.unique(cat['design']):
        for regime in (0,1):
            ids=[i for i in np.unique(cat['catalog']) if cat['design'][cat['catalog']==i][0]==design and cat['regime'][cat['catalog']==i][0]==regime]
            for method in ('point','pdf','population','fixed_response','response_marginalized'):
                values=[];cov=[];width=[]
                for i in ids:
                    if method in ('point','pdf','population'):
                        rows=[r for r in old if int(r['catalog'])==i];value=sum(abs(float(r[method])-float(r['realized'])) for r in rows)
                        if method in ('point','pdf'):value+=float(tails[i][method])
                    else:
                        rows=[r for r in new if int(r['catalog'])==i and r['target']=='distance' and r['method']==method]
                        if not rows:continue
                        value=sum(abs(float(r['mean_count'])-float(r['realized_count']))/128 for r in rows)
                        q=next(r for r in rows if r['bin']=='1');cov.append(float(q['low'])<=float(q['realized_count'])<=float(q['high']));width.append((float(q['high'])-float(q['low']))/128)
                    values.append(value)
                if values:summary.append(dict(design=design,regime=regime,method=method,catalogs=len(values),realized_L1=np.mean(values),sd_L1=np.std(values),coverage_2_5=np.mean(cov) if cov else None,width_2_5=np.mean(width) if width else None))
    csv_rows(T/'observed_summary.csv',summary)
    bounds=read(T/'reachable_range.csv');reach=[]
    for design in np.unique(cat['design']):
        for regime in (0,1):
            for b in range(5):
                row=next(r for r in old if r['design']==design and int(r['regime'])==regime and int(r['bin'])==b);bound=next(r for r in bounds if int(r['regime'])==regime and int(r['bin'])==b)
                truth=float(row['truth']);reach.append(dict(design=design,regime=regime,bin=b,truth=truth,minimum=float(bound['minimum']),maximum=float(bound['maximum']),inside=bool(float(bound['minimum'])<=truth<=float(bound['maximum']))))
    csv_rows(T/'reachable_truth.csv',reach)
    check('controls108',len(load(D/'control_counts.npz')['counts'])==108)
    check('count_summaries_all192',len({r['catalog'] for r in new if r['method']=='fixed_response'})==192)
    save_json(D/'validation.json',dict(actual_checks=len(checks),all_passed=all(checks.values()),checks=checks,scientific_acceptance=False,reason='Passing implementation checks does not establish response calibration or real-sky validity'))
    print('VALIDATED',len(checks),flush=True)


if __name__=='__main__':main()
