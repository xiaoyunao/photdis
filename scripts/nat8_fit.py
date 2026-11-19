#!/usr/bin/env python3
"""Two folds then full M5, every bootstrap refit shares the system reference."""
from nat8_common import *
from nat8_math import fit,objective
from concurrent.futures import ProcessPoolExecutor
from photdis.stage1 import require_compute
from functools import lru_cache


@lru_cache(maxsize=1)
def data():return load(OUT/'prepare/roster.npz'),load(OUT/'prepare/M5_E.npz')


def one(task):
    label,draw,point=task;t,z=data();ii=z['ii'];take=t['calibration_eligible'][ii]&((t['fold'][ii]!=int(label)) if label!='full' else True);rows=ii[take];counts=np.ones(len(rows)) if draw<0 else t['mult'][draw,rows].astype(float);mu=float(10+5*np.log10(t['ref'][rows[0]]));starts=C['fit']['starts'] if draw<0 else [point,[0,0]]
    result=fit(z['means'][take],z['logw'][take],z['sigma'][take],z['x'],mu,counts,starts,maxiter=C['fit']['maxiter']);result.update(label=label,draw=draw,N=len(rows),effective_N=float(counts.sum()),mu_ref=mu,reference_shift=0. if draw<0 else float(t['shift_M5'][draw]));theta=result['a'].copy();result['theta']=theta;result['a']=theta-np.array([result['reference_shift'],0])
    return result


def main():
    require_compute();d=folder('fit');inputs={'prepare':sha(OUT/'prepare/complete.json')};t,z=data();results=[];points={}
    # These are the only candidate fits. Numerical starts optimize one objective.
    for label in ('0','1','full'):
        r=one((label,-1,None));results.append(r);points[label]=r;print('Point',label,r['a'],r['objective'],r['gradient'],flush=True)
    tasks=[(label,k,points[label]['theta']) for label in points for k in range(C['bootstrap'])]
    with ProcessPoolExecutor(max_workers=C['workers']) as pool:
        for at,r in enumerate(pool.map(one,tasks,chunksize=1)):
            results.append(r)
            if at%64==0:print('Bootstrap refit',at,len(tasks),flush=True)
    params={};rows=[]
    for label in points:
        rr=[r for r in results if r['label']==label and r['draw']>=0];assert [r['draw'] for r in rr]==list(range(256));a=np.array([r['a'] for r in rr]);theta=np.array([r['theta'] for r in rr]);point=points[label];params[label]=dict(point=point,parameter_covariance=np.cov(a.T),conditional_theta_covariance=np.cov(theta.T),a0_CI=np.quantile(a[:,0],[.025,.975]),a1_CI=np.quantile(a[:,1],[.025,.975]),bootstrap_a=a,bootstrap_theta=theta)
        for r in [point]+rr:rows.append(dict(fit=label,replicate=r['draw'],N=r['N'],resampled_N=r['effective_N'],a0=float(r['a'][0]),a1=float(r['a'][1]),theta0=float(r['theta'][0]),objective=r['objective'],success=r['success'],gradient=r['gradient'],iterations=r['iterations'],reference_shift=r['reference_shift'],local_starts=len(r['solutions']),message=r['message']))
    save_json(d/'parameters.json',params);csv_rows(d/'fit_draws.csv',rows);save_json(d/'all_solutions.json',results)
    npz(d/'parameter_draws.npz',**{label+'_a':params[label]['bootstrap_a'] for label in params},**{label+'_theta':params[label]['bootstrap_theta'] for label in params},M5_reference_shift=t['shift_M5'],M13_reference_shift=t['shift_M13'],M30_reference_shift=t['shift_M30'])
    # Real-data proof of the shared-distance translation and one reference term.
    use=t['calibration_eligible'][z['ii']];mu=float(10+5*np.log10(t['ref'][z['ii'][use][0]]));a=points['full']['a'];counts=np.ones(sum(use));base,_=objective(a,z['means'][use],z['logw'][use],z['sigma'][use],z['x'],mu,counts);proof=[]
    for delta in (-.05,0,.05):
        shifted,_=objective(a-np.array([delta,0]),z['means'][use],z['logw'][use],z['sigma'][use],z['x'],mu+delta,counts);assert abs(shifted-base)<1e-8;proof.append(dict(shared_reference_shift=delta,intercept_shift=-delta,likelihood_difference=shifted-base,reference_factor_count=1))
    csv_rows(d/'shared_reference_identity.csv',proof);complete(d,inputs,only_two_parameters=True,common_reference_factor_count=1,fit_order=['fold0','fold1','fullM5','bootstrap'],factorization='theta0=a0+mu-mu_ref; joint likelihood=L(theta0,a1)*R(mu), unit Jacobian; point mu_MAP=mu_ref. Bootstrap intercept theta0_boot minus one shared reference shift.',predictions_not_opened=True);print('All fixed fits frozen',flush=True)


if __name__=='__main__':main()
