#!/usr/bin/env python3
"""Compare dust responses in the existing bright-star common support."""
import json
import numpy as np

import environment_dust as dust
import environment_pilot as pilot
import environment_results as baseline
from environment_transfer import OLD, OUT as TRANSFER
from faint_signed_20261120 import C, OUT
from photdis.stage1 import require_compute
from v2_1_near import csv_rows, load, save_json, sha


def wrong_branch(edges, mass, ref):
    width = np.diff(edges,axis=1)
    rel = edges[:,:-1]-ref[:,None]
    below = np.divide(-1-rel,width,out=np.zeros_like(width),where=width>0)
    above = np.divide(rel+width-1,width,out=np.zeros_like(width),where=width>0)
    return np.column_stack([(mass*np.clip(q,0,1)).sum(axis=1) for q in (below,above)])


def read_archive(root):
    d,t,p,a = [load(root/f) for f in ('dust/forecasts.npz','forecast/roster.npz',
                                     'forecast/pdf.npz','apogee/references.npz')]
    np.testing.assert_array_equal(t['ls_id'],a['ls_id'])
    i = d['rows']
    np.testing.assert_array_equal(t['ls_id'][i],d['ls_id'])
    fit = t['historical_overlap'][i] | t['qrf_fit_overlap'][i]
    common = ((d['iteration_state']==5)&(d['reference_mu']>=d['mu_min'])&
              (d['reference_mu']<=d['mu_max'])&~fit)
    j = np.flatnonzero(common)
    i = i[j]
    obs = pilot.join_gaia(t['source_id'][i],pilot.GAIA)
    mu,sigma = d['reference_mu'][j],t['gaia_sigma_mu'][i]
    pdf_rows = np.searchsorted(p['rows'],i)
    np.testing.assert_array_equal(t['ls_id'][i],p['ls_id'][pdf_rows])
    assert (p['TYPE'][pdf_rows]=='PSF').all() and (t['TYPE'][i]=='PSF').all()
    grz = t['mag_grz'][i]
    q = dict(ls_id=t['ls_id'][i], TYPE=t['TYPE'][i], field=t['field'][i],
             group=t['gaia_proxy'][i], gr=grz[:,0]-grz[:,1], rz=grz[:,1]-grz[:,2],
             r=t['r'][i], mu_ref=mu, ebv=t['EBV'][i], block=baseline.covariance_blocks(t)[i],
             gaia_bp_rp=obs['bp_rp'], gaia_MG=obs['phot_g_mean_mag']-mu,
             spectra_class=t['classification'][i], astro_class=a['astro_classification'][i],
             spectra_split=t['spectra_split'][i])
    for arm,e,w in [
        ('A',p['mu_edges'][pdf_rows],p['mass'][pdf_rows]),
        ('B',d['base_edges'][j],d['base_mass'][j]),
        ('C',d['iteration_edges'][j],d['iteration_mass'][j])]:
        q[arm+'_crps'] = baseline.histogram_scores(e,w,mu,sigma)[1]
        q[arm+'_bias'] = dust.modulus_quantiles(w,e)[:,2]-mu
        q[arm+'_near'],q[arm+'_far'] = wrong_branch(e,w,mu).T
    q['delta_gr'] = (q['ebv']-d['iteration_ebv'][j])*(dust.DECam_PER_BAYESTAR[0]-dust.DECam_PER_BAYESTAR[1])
    q['delta_rz'] = (q['ebv']-d['iteration_ebv'][j])*(dust.DECam_PER_BAYESTAR[1]-dust.DECam_PER_BAYESTAR[2])
    return q


def matching(x, keep):
    features = tuple(C['covariate_match'])
    z = np.column_stack([np.floor(x[k]/C['covariate_match'][k]) for k in features])
    cell = np.unique(z,axis=0,return_inverse=True)[1]
    dwarf = x['group']=='dwarf_proxy'
    uncertain = x['group']=='uncertain'
    counts = np.stack([np.bincount(cell[keep&g],minlength=z.shape[0]) for g in (dwarf,uncertain)])
    overlap = counts.min(axis=0)[cell]
    a = np.divide(overlap,counts[0,cell],out=np.zeros(len(cell)),where=counts[0,cell]>0)*dwarf*keep
    b = np.divide(overlap,counts[1,cell],out=np.zeros(len(cell)),where=counts[1,cell]>0)*uncertain*keep
    return a,b


def field_rows(x, field, a, b):
    labels = tuple(f'C_{m}' for m in ('crps','bias','near','far'))
    diffs = np.column_stack([x[k]-x['B'+k[1:]] for k in labels])
    if min(a.sum(),b.sum())==0:
        return [dict(field=field,n_dwarf=0,n_uncertain=0,status='no joint support')]
    cells = np.unique(x['block'][(a+b)>0])
    wb = np.array([[(a*(x['block']==k)).sum(),(b*(x['block']==k)).sum()] for k in cells])
    wx = np.array([[(diffs*(a*(x['block']==k))[:,None]).sum(0),
                    (diffs*(b*(x['block']==k))[:,None]).sum(0)] for k in cells])
    rng = np.random.default_rng(C['seed'])
    resample = rng.integers(0,len(cells),(600,len(cells)))
    den = wb[resample].sum(1)
    keep = (den>0).all(1)
    boot = wx[resample][keep].sum(1)/den[keep,:,None]
    ci = np.quantile(boot[:,1]-boot[:,0],[.025,.975],axis=0) if len(boot)>2 else np.full((2,len(labels)),np.nan)
    raw = wx.sum(0)/wb.sum(0)[:,None]
    count = [int((v>0).sum()) for v in (a,b)]
    eff = [v.sum()**2/np.dot(v,v) for v in (a,b)]
    return [dict(field=field,metric=key,blocks=len(cells),matched_equivalent=a.sum(),
                 n_dwarf=count[0],n_uncertain=count[1],ess_dwarf=eff[0],ess_uncertain=eff[1],
                 dwarf_change=raw[0,k],uncertain_change=raw[1,k],
                 change_difference=raw[1,k]-raw[0,k],lo=ci[0,k],hi=ci[1,k])
            for k,key in enumerate(labels)]


def main():
    require_compute()
    results = OUT/'results'
    results.mkdir(parents=True,exist_ok=True)
    old,new = [read_archive(p) for p in (OLD,TRANSFER)]
    x = {key:np.r_[old[key],new[key]] for key in old}
    assert len(np.unique(x['ls_id']))==len(x['ls_id'])
    available, matches, balance, data = [],[],[],[]
    for field in np.unique(x['field']):
        f = x['field']==field
        a,b = matching(x,f)
        x['match_weight'] = a+b
        matches.extend(field_rows(x,field,a,b))
        for name in C['covariate_match']:
            for label,w in [('dwarf_proxy',a),('uncertain',b)]:
                k = f & (x['group']==label)
                if k.any():
                    balance.append(dict(field=field,variable=name,group=label,n=int(k.sum()),
                         raw=np.mean(x[name][k]),post=np.average(x[name],weights=w) if w.sum()>0 else ''))
        for label in np.unique(x['group'][f]):
            j=f&(x['group']==label)
            available.append(dict(field=field,group=label,common=int(j.sum()),
                spectroscopic_class=int(np.count_nonzero(j&(x['spectra_class']>=0))),
                astro_class=int(np.count_nonzero(j&(x['astro_class']>=0))),
                source_dwarf=int(np.count_nonzero(j&((x['spectra_class']==0)|(x['astro_class']==0)))),
                source_evolved=int(np.count_nonzero(j&((x['spectra_class']==1)|(x['astro_class']==1)))),
                mean_crps_change=np.mean(x['C_crps'][j]-x['B_crps'][j])))
        data.extend(dict(zip(x,record)) for record in zip(*(k[f] for k in x.values())))
    tables={'population_availability':available,'population_matches':matches,
            'population_balance':balance,'population_stars':data}
    for name,rows in tables.items():
        csv_rows(results/(name+'.csv'),rows)
    save_json(results/'population_receipt.json',dict(
        inputs={str(d):sha(d/'dust/complete.json') for d in (OLD,TRANSFER)},
        outputs={name+'.csv':sha(results/(name+'.csv')) for name in tables}))
    print('Existing common PDF sources',len(x['ls_id']),flush=True)


if __name__=='__main__':
    main()
