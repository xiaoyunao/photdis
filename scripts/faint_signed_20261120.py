#!/usr/bin/env python3
"""Frozen QRF signed-Gaia experiment on the registered nine apertures."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from multiprocessing import get_context

import numpy as np

import environment_pilot as pilot
import environment_results as spatial
from environment_transfer import OLD, OUT as TRANSFER
from environment_transfer_support import faint_field
from photdis.faint_signed import components, scale_weights, selected_moments, selected_score, simulation
from photdis.stage1 import require_compute
from photdis.v3.data import stable_rank
from photdis.v3_3 import frozen_qrf
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_4_near_forecast import prepare

CONFIG = ROOT/'configs/faint_signed_20261120.json'
C = json.loads(CONFIG.read_text())
OUT = ROOT/'data/faint_signed_20261120'
PREVIOUS = ROOT/'manifests/environment_transfer_20261120/results/faint_signed_parallax.csv'
MODEL = INPUT = None


def registered_fields():
    f = json.loads((ROOT/'configs/environment_20261120.json').read_text())['fields']
    new = json.loads((ROOT/'configs/environment_transfer_20261120.json').read_text())['fields']
    selected = json.loads((TRANSFER/'preflight/freeze.json').read_text())['selected']
    return {name:(root,[name,l,b]) for root,defs in ((OLD,f),(TRANSFER,[q for q in new if q[0] in selected]))
            for name,l,b in defs}


def replay_qrf(rows):
    m = INPUT
    p = frozen_qrf(MODEL, m['mag_grz'][rows], m['error_grz'][rows], m['identity'][rows])
    assert np.all(p['supported'])
    return p['mu_edges'], p['mass']


def make_cohort(field, root, coordinates, magnitude, pilot_only):
    previous = list(csv.DictReader(PREVIOUS.open()))
    expected = next(r for r in previous if r['field']==field and int(r['r_lo'])==magnitude)
    t = load(root/'forecast/roster.npz')
    p = load(root/'census/qualified.npz')
    np.testing.assert_array_equal(t['ls_id'],p['ls_id'])
    if root == TRANSFER:
        masks = load(TRANSFER/'preflight/identity_masks.npz')
        np.testing.assert_array_equal(t['ls_id'],masks['ls_id'])
        independent = masks['clean']
    else:
        independent = ~(t['qrf_fit_overlap'] | t['historical_overlap'])
    with (root/'census/raw_admission.csv').open() as f:
        files = [q['sweep'] for q in csv.DictReader(f) if q['field']==field]
    _, parent = faint_field((coordinates,files))
    ids = parent['source_id']
    unique = (ids>0) & (pilot.unique_join(ids,ids)>=0)
    take = ((t['field']==field) & independent & t['support'] & (t['gaia_state']==4) &
            (t['r']>=magnitude) & (t['r']<magnitude+1) & np.isin(t['ls_id'],parent['ls_id'][unique]))
    j = np.flatnonzero(take)
    assert len(j)==int(expected['qrf_history_clean_weak']), (field,magnitude,len(j),expected)
    assert np.count_nonzero(t['gaia_parallax'][j]<0)==int(expected['qrf_history_clean_negative'])
    if pilot_only:
        rank = stable_rank(t['ls_id'][j],C['seed'])
        j = j[np.argsort(rank,kind='stable')[:C['pilot_cap']]]
    keep = ('TYPE','ls_id','source_id','r','longitude','latitude','gaia_parallax_error')
    m = {k:t[k][j] for k in keep}
    m.update({k:p[k][j] for k in ('mag_grz','error_grz','identity')})
    assert len(np.unique(m['source_id']))==len(j) and (m['TYPE']=='PSF').all()
    obs = t['gaia_parallax'][j]
    assert np.all(np.isfinite(obs)) and np.all(obs<m['gaia_parallax_error']*C['weak_snr_upper'])
    return m, obs


def prepare_run(mode):
    global MODEL, INPUT
    is_pilot = mode=='pilot'
    groups = list(C['bins'][:-1])
    if not is_pilot:
        gate = json.loads((OUT/'power/gate.json').read_text())
        assert gate['config']==sha(CONFIG) and gate['power']==sha(OUT/'power/power.csv')
        groups = gate['approved_r_lo']
    if not groups:
        return
    models, _, sources = prepare()
    MODEL = models['qrf']
    for field,(root,position) in registered_fields().items():
        if is_pilot and field not in C['pilot_fields']:
            continue
        for mag in groups:
            folder = OUT/mode/f'{field}_{mag}'
            folder.mkdir(parents=True,exist_ok=True)
            done = folder/'complete.json'
            if done.exists():
                receipt = json.loads(done.read_text())
                assert receipt['config']==sha(CONFIG)
                assert all(sha(folder/f)==h for f,h in receipt['outputs'].items())
                continue
            m, obs = make_cohort(field,root,position,mag,is_pilot)
            INPUT = m
            m['block'] = spatial.covariance_blocks(m)
            atomic_npz(folder/'inputs.npz', **{k:m[k] for k in ('TYPE','ls_id','source_id','r','block','gaia_parallax_error')})
            atomic_npz(folder/'signed_observation.npz',ls_id=m['ls_id'],TYPE=m['TYPE'],parallax=obs)
            if len(obs):
                chunks = np.array_split(np.arange(len(obs)),max(1,int(np.ceil(len(obs)/256))))
                with ProcessPoolExecutor(max_workers=20,mp_context=get_context('fork')) as pool:
                    result = list(pool.map(replay_qrf,chunks))
                edges,mass = [np.concatenate(x) for x in zip(*result)]
                atomic_npz(folder/'pdf.npz',ls_id=m['ls_id'],TYPE=m['TYPE'],edges=edges,mass=mass)
            output = {f.name:sha(f) for f in folder.glob('*.npz')}
            save_json(done,dict(config=sha(CONFIG),roster=sha(root/'forecast/complete.json'),
                                sources={str(p):sha(p) for p in sources},outputs=output,n=len(obs)))
            print(mode,field,mag,len(obs),flush=True)


def inputs(folder):
    t,p = [load(folder/f) for f in ('inputs.npz','pdf.npz')]
    np.testing.assert_array_equal(t['ls_id'],p['ls_id'])
    assert (p['TYPE']=='PSF').all()
    return t,p['edges'],p['mass']


def projections(edges,mass,sigma,order=8):
    shifts = C['shifts_mag']
    lim = C['weak_snr_upper']*sigma
    table = np.empty((len(sigma),len(shifts)+3))
    for start in range(0,len(sigma),512):
        j = slice(start,start+512)
        p,w = components(edges[j],mass[j],order)
        for i,shift in enumerate(shifts):
            m,v,a = selected_moments(p,w,sigma[j],lim[j],shift=shift)
            table[j,i] = m
            if shift==0:
                table[j,-3:-1] = np.column_stack((v,a))
        upper = selected_moments(p,w,sigma[j],lim[j],offset=.005)[0]
        lower = selected_moments(p,w,sigma[j],lim[j],offset=-.005)[0]
        table[j,-1] = (upper-lower)/.010
    assert np.isfinite(table).all() and np.all(table[:,-3:-1]>0)
    return table


def traits(t,p,stress=1.):
    blocks = np.unique(t['block'],return_inverse=True)[1]
    rms_field,rms_block = [C[k]*stress for k in ('field_offset_rms_mas','block_offset_rms_mas')]
    shifts = C['shifts_mag']
    slope = (p[:,shifts.index(.1)]-p[:,shifts.index(-.1)])/.2
    weights = scale_weights(slope,p[:,-1],p[:,-3],blocks,rms_field,rms_block)
    sd = np.sqrt(np.dot(weights**2,p[:,-3]) + (rms_field*np.dot(weights,p[:,-1]))**2 +
                 rms_block**2*np.square(np.bincount(blocks,weights=weights*p[:,-1])).sum())
    return blocks,weights,sd


def power_cell(item):
    field,mag = item
    t,e,w = inputs(OUT/'pilot'/f'{field}_{mag}')
    sig = t['gaia_parallax_error']
    lim = C['weak_snr_upper']*sig
    table = projections(e,w,sig)
    rng = np.random.default_rng(C['seed']+mag+sum(map(ord,field))*31)
    rows = []
    for stress in (1,C['systematic_stress_factor']):
        block,coef,sd = traits(t,table,stress)
        base = (e,w,sig,lim,block,coef,table[:,2])
        params = (C['field_offset_rms_mas']*stress,C['block_offset_rms_mas']*stress,rng)
        null = simulation(*base,0.,C['zero_calibration'],*params)
        ci = np.quantile(null,[C['null_rejection']/2,1-C['null_rejection']/2])
        for shift in C['shifts_mag']:
            n = C['zero_evaluation'] if shift==0 else C['injection_repetitions']
            draws = simulation(*base,shift,n,*params)
            rate = np.mean((draws<ci[0])|(draws>ci[1]))
            directional = np.mean(draws<ci[0]) if shift<0 else np.mean(draws>ci[1])
            rows.append(dict(field=field,r_lo=mag,systematics_multiplier=stress,n=len(sig),
                     blocks=len(np.unique(block)),shift_mag=shift,replicates=n,rejection=rate,
                     directional_power=directional if shift!=0 else '',null_lo=ci[0],null_hi=ci[1],
                     median_recovered=np.median(draws),working_sd_mag=sd))
    return rows


def power():
    dest = OUT/'power'
    dest.mkdir(exist_ok=True)
    jobs = [(f,m) for f in C['pilot_fields'] for m in C['bins'][:-1]]
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        rows = [x for sub in pool.map(power_cell,jobs) for x in sub]
    csv_rows(dest/'power.csv',rows)
    lo,hi = C['allowed_realized_null_rejection']
    passed = []
    for mag in C['bins'][:-1]:
        q = [x for x in rows if x['r_lo']==mag and x['systematics_multiplier']==1]
        if len(q)==len(C['pilot_fields'])*len(C['shifts_mag']) and all(
            x['blocks']>=C['minimum_blocks'] and
            (lo<=x['rejection']<=hi if x['shift_mag']==0 else x['directional_power']>=C['minimum_power'])
            for x in q):
            passed.append(mag)
    save_json(dest/'gate.json',dict(config=sha(CONFIG),power=sha(dest/'power.csv'),
                                   approved_r_lo=passed,
                                   pilot={f'{f}_{m}':sha(OUT/'pilot'/f'{f}_{m}'/'complete.json') for f,m in jobs}))
    print('Nine-field magnitude bins authorized by power:',passed,flush=True)


def summarize(folder, mode, field, mag):
    t,e,w = inputs(folder)
    obs = load(folder/'signed_observation.npz')
    np.testing.assert_array_equal(obs['ls_id'],t['ls_id'])
    y,sig = obs['parallax'],t['gaia_parallax_error']
    limit = sig*C['weak_snr_upper']
    table = projections(e,w,sig)
    block,coef,sd = traits(t,table)
    _,stress_c,stress_sd = traits(t,table,C['systematic_stress_factor'])
    metric = np.empty((len(y),len(C['shifts_mag']),3))
    verify = 0.
    for start in range(0,len(y),512):
        j = slice(start,start+512)
        p,weight = components(e[j],w[j],8)
        for k,shift in enumerate(C['shifts_mag']):
            metric[j,k] = np.column_stack(selected_score(p,weight,y[j],sig[j],limit[j],shift))
        if start==0:
            p16,w16 = components(e[j],w[j],16)
            hi = np.column_stack(selected_score(p16,w16,y[j],sig[j],limit[j]))
            verify = np.max(abs(hi-metric[j,2]))
    assert verify<1e-4 and np.all((metric[:,:,1]>=0)&(metric[:,:,1]<=1+1e-10))
    residual = y-table[:,2]
    record = dict(mode=mode,field=field,r_lo=mag,n=len(y),blocks=int(block.max()+1),
        observed_negative=float(np.mean(y<0)),predicted_negative=float(metric[:,2,2].mean()),
        scale_shift_linear_mag=float(coef@residual),systematic_stress_scale_mag=float(stress_c@residual),
        model_sd_mag=sd,systematic_stress_sd_mag=stress_sd,p08_p16_error=verify)
    contrasts = []
    rng = np.random.default_rng(C['seed']+mag)
    cells = [np.flatnonzero(block==k) for k in np.unique(block)]
    groups = [np.concatenate([cells[k] for k in z]) for z in rng.integers(0,len(cells),size=(500,len(cells)))]
    for i,shift in enumerate(C['shifts_mag']):
        diff = metric[:,i,0]-metric[:,2,0]
        lo,hi=np.quantile([diff[j].mean() for j in groups],[.025,.975])
        contrasts.append(dict(mode=mode,field=field,r_lo=mag,shift_mag=shift,
            log_score=metric[:,i,0].mean(),paired_gain=diff.mean(),gain_lo=lo,gain_hi=hi,
            expected_negative=metric[:,i,2].mean()))
    hist,edge = np.histogram(metric[:,2,1],bins=np.linspace(0,1,11))
    histrows = [dict(mode=mode,field=field,r_lo=mag,pit_lo=a,pit_hi=b,count=int(n),uniform=len(y)/10)
                for a,b,n in zip(edge[:-1],edge[1:],hist)]
    slo = np.r_[-np.inf,np.arange(-4,5),5.]; moments = []
    for a,b in zip(slo[:-1],slo[1:]):
        count = np.count_nonzero((y/sig>=a)&(y/sig<b))
        cdfs=[]
        for z in (a,b):
            if not np.isfinite(z):
                cdfs.append(0.)
            elif z==5.:
                cdfs.append(float(len(y)))
            else:
                total=0.
                for start in range(0,len(y),512):
                    j=slice(start,start+512)
                    pp,ww=components(e[j],w[j],8)
                    total += selected_score(pp,ww,z*sig[j],sig[j],limit[j])[1].sum()
                cdfs.append(float(total))
        moments.append(dict(mode=mode,field=field,r_lo=mag,snr_lo=a,snr_hi=b,observed=count,
                            predicted=cdfs[1]-cdfs[0]))
    return record,contrasts,histrows,moments


def score():
    approved = json.loads((OUT/'power/gate.json').read_text())['approved_r_lo']
    rows,comparisons,pit,count = [],[],[],[]
    for mode in ('pilot','full'):
        for path in sorted((OUT/mode).glob('*_*')):
            if not (path/'pdf.npz').exists():
                continue
            field,r = path.name.split('_')
            r=int(r)
            if mode=='full':
                assert r in approved
            a,b,c,d = summarize(path,mode,field,r)
            rows.append(a); comparisons.extend(b); pit.extend(c); count.extend(d)
            print('Scored',mode,field,r,a['n'],flush=True)
    target=OUT/'results'
    target.mkdir(exist_ok=True)
    for name,records in [('signed_summary',rows),('signed_scores',comparisons),('pit',pit),('signed_snr',count)]:
        csv_rows(target/(name+'.csv'),records)


if __name__=='__main__':
    require_compute()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['pilot','power','full','score'])
    phase=parser.parse_args().phase
    if phase in ('pilot','full'):
        prepare_run(phase)
    elif phase=='power':
        power()
    else:
        score()
