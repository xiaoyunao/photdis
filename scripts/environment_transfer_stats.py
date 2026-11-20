#!/usr/bin/env python3
"""Identical-star PDF scoring and paired sky-block confidence intervals."""
import csv
import json

import numpy as np

import environment_dust as dust
import environment_pilot as pilot
import environment_results as baseline
from environment_transfer import C, CONFIG, FROZEN, OUT, REGISTRY, TABLES, configure
from photdis.stage1 import require_compute
from v2_1_near import csv_rows, load, save_json, sha

METRICS = ('bias','nmad','severe','crps','log_score','coverage68','coverage95','width68','width95')
ARMS = ('A_full','B','C')
COVERAGE = ((5,.68),(6,.95))


def rows_for_pdf(edges, weight, target, sigma):
    np.testing.assert_allclose(weight.sum(1),1,rtol=0,atol=1e-9)
    q = dust.modulus_quantiles(weight,edges)
    log, crps = baseline.histogram_scores(edges,weight,target,sigma)
    delta = q[:,2]-target
    return np.column_stack((delta,abs(delta)>C['severe_error_mu_mag'],crps,log,
                            (q[:,1]<=target)&(target<=q[:,3]),
                            (q[:,0]<=target)&(target<=q[:,4]),q[:,3]-q[:,1],q[:,4]-q[:,0]))


def summarize(rows):
    center = np.median(rows[:,0])
    mad = 1.4826*np.median(abs(rows[:,0]-center))
    return np.r_[center,mad,rows[:,1:].mean(axis=0)]


def compare(a, sky, rng):
    values = np.stack([summarize(rows) for rows in a])
    cells = [np.flatnonzero(sky==x) for x in np.unique(sky)]
    if len(cells)<2:
        pair = values[2]-values[1]
        missing = np.full((2,len(METRICS)),np.nan)
        return values, np.broadcast_to(missing[:,None,:],(2,len(a),len(METRICS))), pair, missing, np.full((2,2),np.nan)
    reps = []
    for drawn in rng.integers(0,len(cells),size=(C['bootstrap_draws'],len(cells))):
        idx = np.concatenate([cells[x] for x in drawn])
        reps.append([summarize(rows[idx]) for rows in a])
    r = np.asarray(reps)
    pair = r[:,2]-r[:,1]
    coverage_loss = np.column_stack([abs(r[:,2,k]-nominal)-abs(r[:,1,k]-nominal)
                                     for k,nominal in COVERAGE])
    return values,np.quantile(r,[.025,.975],axis=0),values[2]-values[1],np.quantile(pair,[.025,.975],axis=0),np.quantile(coverage_loss,[.025,.975],axis=0)


def reference_bootstrap(a, sky, parallax, sigma, rng):
    """Draw one Gaia error per identity and one common offset per field."""
    blocks = [np.flatnonzero(sky==x) for x in np.unique(sky)]
    if len(blocks)<2:
        return np.full((2,4),np.nan)
    scale = REGISTRY['common_parallax_sensitivity_mas']
    draw = []
    for _ in range(C['bootstrap_draws']):
        i = np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))])
        perturb = rng.normal(0,sigma) + rng.normal(0,scale)
        delta_mu = -5*np.log10((parallax+perturb)/parallax)
        centers = np.median((a[:,:,0]-delta_mu)[:,i],axis=1)
        draw.append(np.r_[centers,centers[2]-centers[1]])
    return np.quantile(draw,[.025,.975],axis=0)


def groups(t, rows):
    yield 'all','all',np.ones(len(rows),bool)
    values = {
        'observed_r':(t['r'][rows],[14,17,19,20,21,22,23]),
        'sfd_gr':(np.diff(t['mag_grz'][rows,:2],axis=1)[:,0]*-1,[-10,.45,.75,1.05,1.40,10]),
        'reference_kpc':(t['gaia_distance_kpc'][rows],[0,.5,1,2,5,10,100]),
        'sfd_ebv':(t['EBV'][rows],[0,.08,.2,.5,10]),
    }
    for feature,(x,edges) in values.items():
        for a,b in zip(edges[:-1],edges[1:]):
            yield feature,f'{a:g}-{b:g}',(x>=a)&(x<b)
    tag = baseline.tag(t)[rows]
    for label in sorted(set(tag)|{'dwarf_proxy','giant_proxy','uncertain'}):
        yield 'class',label,tag==label


def decision(values, pair_ci, cov_ci, n, blocks):
    median = abs(values[2,0])
    limits = REGISTRY
    return dict(enough_reference=n>=limits['minimum_gaia'] and blocks>=limits['minimum_blocks'],
                scale_improves=median<abs(values[1,0]), scale_target=median<=limits['absolute_bias_target_mag'],
                crps_improves=pair_ci[1,3]<0,
                log_guard=not bool(pair_ci[1,4] < -limits['log_score_loss_nat']),
                severe_guard=not bool(pair_ci[0,2] > limits['severe_rate_increase']),
                coverage_guard=not bool(np.any(cov_ci[0] > limits['coverage_error_increase'])))


def main():
    require_compute()
    configure()
    TABLES.mkdir(parents=True,exist_ok=True)
    frozen = json.loads(FROZEN.read_text())
    t = load(OUT/'forecast/roster.npz')
    a = load(OUT/'apogee/references.npz')
    np.testing.assert_array_equal(t['ls_id'],a['ls_id'])
    t.update(a)
    p, d = map(load,(OUT/'forecast/pdf.npz',OUT/'dust/forecasts.npz'))
    rows = d['rows']
    at = np.searchsorted(p['rows'],rows)
    np.testing.assert_array_equal(p['ls_id'][at],d['ls_id'])
    assert (d['TYPE']=='PSF').all()
    block = baseline.covariance_blocks(t)[rows]
    mu, sig = d['reference_mu'],t['gaia_sigma_mu'][rows]
    mapped = d['mu_max']>d['mu_min']
    inside = mapped & (mu>=d['mu_min']) & (mu<=d['mu_max'])
    retained = mapped & (d['base_probability']>=C['dust']['minimum_histogram_mass'])
    converged = d['iteration_state']==5
    common = inside & retained & converged
    original_n = np.count_nonzero(t['gaia_evaluation'])
    assert original_n >= len(rows) and np.isfinite(sig).all()
    all_masks = (np.ones(len(rows),bool),mapped,inside,retained,converged,common)
    all_labels = ('registered','map_valid','reference_inside','sfd_half_mass','converged','common')
    methods = ((p['mu_edges'][at],p['mass'][at]),(d['base_edges'],d['base_mass']),
               (d['iteration_edges'],d['iteration_mass']))
    rng = np.random.default_rng(C['sample_seed'])
    ref_rng = np.random.default_rng(C['sample_seed']+1)
    flow, scores, contrasts, pairs, judgments, shift, oracle, ref_ci = [],[],[],[],[],[],[],[]
    for field in frozen['selected']:
        f = t['field'][rows]==field
        source = t['field']==field
        pred = t['field'][p['rows']]==field
        on_application = pred & t['application_sample'][p['rows']]
        working = on_application & (p['dwarf_score']>=.80) & (p['mass_2_10']>=.50)
        flow.append(dict(field=field,unique_precise=int(source.sum()), nominal_support=int((source&t['support']).sum()),
                         gaia_before_cap=int((source&t['gaia_evaluation']).sum()),
                         application_draw=int((source&t['application_sample']).sum()),
                         application_predicted=int(on_application.sum()), working_sample=int(working.sum()),
                         **{key:int((f&mask).sum()) for key,mask in zip(all_labels,all_masks)},
                         **{f'state{k}':int((f&(d['iteration_state']==k)).sum()) for k in range(6)}))
        ax, = np.where(f&common)
        common_rows = rows[ax]
        arr = np.stack([rows_for_pdf(e[ax],w[ax],mu[ax],sig[ax]) for e,w in methods]) if len(ax) else None
        for i,k in enumerate(ax):
            row = rows[k]
            pairs.append(dict(field=field,TYPE=t['TYPE'][row],ls_id=int(t['ls_id'][row]),
                              reference_kpc=t['gaia_distance_kpc'][row], reference_sigma_mu=sig[k],
                              obs_r=t['r'][row], sfd=t['EBV'][row], bayestar=d['iteration_ebv'][k],
                              A_full_delta_mu=arr[0,i,0], B_delta_mu=arr[1,i,0], C_delta_mu=arr[2,i,0]))
        dx, = np.where(f & retained & (d['oracle_state']==3))
        if len(dx):
            base = dust.modulus_quantiles(d['base_mass'][dx],d['base_edges'][dx])[:,2]
            oracle.append(dict(field=field,n=len(dx),B_bias=float(np.median(base-mu[dx])),
                               D_bias=float(np.median(d['oracle_center'][dx]-mu[dx]))))
        for name,label,mask in groups(t,rows):
            idx, = np.where(f&common&mask)
            n, nblock = len(idx),len(np.unique(block[idx]))
            key = dict(field=field, stratum=name, bin=label, n=n, blocks=nblock,
                       **{s:int((f&mask&m).sum()) for s,m in zip(all_labels,all_masks) if s!='common'})
            if not n:
                contrasts.append(key)
                if name=='all':
                    judgments.append(dict(field=field,n=0,passed=False))
                continue
            a = arr[:,np.searchsorted(ax,idx)]
            value, ci, pair, pair_ci, cov_ci = compare(a,block[idx],rng)
            for j,arm in enumerate(ARMS):
                scores.extend(dict(**key,arm=arm,metric=m,value=value[j,k],lo=ci[0,j,k],hi=ci[1,j,k])
                              for k,m in enumerate(METRICS))
            contrasts.extend(dict(**key,metric=m,delta=pair[k],lo=pair_ci[0,k],hi=pair_ci[1,k])
                             for k,m in enumerate(METRICS))
            if name=='all':
                gate = decision(value,pair_ci,cov_ci,n,nblock)
                judgments.append(dict(field=field,n=n,blocks=nblock,
                    converged_fraction=float((f&converged).sum()/f.sum()),
                    common_fraction=float(n/f.sum()), **gate, passed=bool(all(gate.values()))))
                interval = reference_bootstrap(a,block[idx],t['gaia_parallax'][rows[idx]],
                                               t['gaia_parallax_error'][rows[idx]],ref_rng)
                ref_ci.extend(dict(field=field,arm=arm,bias_lo=lo,bias_hi=hi)
                              for arm,lo,hi in zip((*ARMS,'C_minus_B'),*interval))
                # Perturb the same reference parallax for both arms; this is a scale sensitivity,
                # with Gaussian measurement uncertainty already integrated once in the log score.
                for offset in (-REGISTRY['common_parallax_sensitivity_mas'],0,
                               REGISTRY['common_parallax_sensitivity_mas']):
                    parallax = t['gaia_parallax'][rows[idx]]
                    target = mu[idx] - 5*np.log10((parallax+offset)/parallax)
                    shifted = [summarize(rows_for_pdf(e[idx],w[idx],target,sig[idx])) for e,w in methods]
                    shift.extend(dict(field=field,offset_mas=offset,arm=arm,bias=v[0],crps=v[3],log_score=v[4])
                                 for arm,v in zip(ARMS,shifted))
    files = dict(flow=flow,scores=scores,contrasts=contrasts,pairs=pairs,judgments=judgments,
                 oracle=oracle,common_parallax_sensitivity=shift,reference_bootstrap=ref_ci)
    for name, records in files.items():
        csv_rows(TABLES/(name+'.csv'),records)
    if judgments:
        for row in judgments:
            print(row,flush=True)
    save_json(TABLES/'transfer_receipt.json',dict(registry=sha(CONFIG),freeze=sha(FROZEN),
        inputs={s:sha(OUT/s/'complete.json') for s in ('census','forecast','dust','apogee')},
        outputs={name+'.csv':sha(TABLES/(name+'.csv')) for name in files}))


if __name__=='__main__':
    main()
