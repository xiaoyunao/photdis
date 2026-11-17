#!/usr/bin/env python3
"""Check frozen QRF distance calibration at securely identified GD-1 members."""
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from astropy.io import fits

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute, tile_bounds
from photdis.v3.metrics import distance_summary
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_8_direct_figures import BLUE, GRAY, ORANGE, STYLE, plt, save
from v3_10_evidence import rvs_evidence
from v3_10_figures import check_templates
from v3_10_working_sample import shell_mass
from v3_11_layers import read_receipt
from v3_13_clusters import model_and_history, phase
from v3_13_delivery import bind, figure_check
from v3_16_sample import D_EDGES
from v3_17_hess import MODULUS_CURVE
from v3_17_members import MEMBERS, table
from v3_17_sample import DEST as V317, INPUT, NORTH

DEST = ROOT/'data/v3_18'
CONFIG = ROOT/'configs/v3_18_gd1.json'
C = json.loads(CONFIG.read_text())
CUTS = np.r_[-np.inf,10+5*np.log10(D_EDGES),np.inf]
SOURCE = V317/'sample'/'complete.json'


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def read_pdfs(t,inputs):
    source = json.loads(SOURCE.read_text())['inputs']
    inputs[str(SOURCE.relative_to(ROOT))] = sha(SOURCE)
    paths = []
    for p,s in source.items():
        if '/predict/' not in p or not p.endswith('complete.json'):
            continue
        path = ROOT/p
        a,b,c,d = tile_bounds(path.parent.name+'.fits')
        near = (t['RA']>=a-.002)&(t['RA']<=c+.002)&(t['DEC']>=b-.002)&(t['DEC']<=d+.002)
        if near.any():
            paths.append((path,s))
    ids = t['ls_id']
    def scan(args):
        path,digest = args
        info = json.loads(verified(path,digest).read_text())
        answer = []
        for name,h in info['outputs'].items():
            key = 'state_'+name
            if name.startswith('state_') or key not in info['outputs']:
                continue
            r = read_columns(verified(path.parent/key,info['outputs'][key]),('TYPE','ls_id','state'))
            j = unique_join(r['ls_id'],ids)
            use = (j>=0)&(r['state']==3)
            if not use.any():
                continue
            p = read_columns(verified(path.parent/name,h),('TYPE','ls_id','qrf_mu_edges','qrf_mu_mass','qrf_quantiles_kpc'))
            k = unique_join(r['ls_id'][use],p['ls_id'])
            assert (k>=0).all() and np.all(p['TYPE'][k]=='PSF') and np.all(r['TYPE'][use]=='PSF')
            answer.append((j[use],p['qrf_mu_edges'][k],p['qrf_mu_mass'][k],p['qrf_quantiles_kpc'][k]))
        return answer
    seen = np.zeros(len(ids),bool)
    raw = {}
    with ThreadPoolExecutor(max_workers=min(6,int(os.getenv('SLURM_CPUS_PER_TASK',1)))) as pool:
        for group in pool.map(scan,paths):
            for j,ed,m,q in group:
                assert not seen[j].any() and len(np.unique(j))==len(j)
                seen[j] = True
                for i,e,w,x in zip(j,ed,m,q):
                    raw[i] = (e,w,x)
    for path,digest in paths:
        inputs[str(path.relative_to(ROOT))] = digest
    assert np.all(seen[t['qrf_archived']])
    keep = np.flatnonzero(seen)
    return keep,[np.stack([raw[j][i] for j in keep]) for i in range(3)]


def refs(phi1,inputs):
    p = INPUT/'v4_Figure2_track_jarvis.fits'
    t = np.sort(fits.getdata(verified(p,MODULUS_CURVE),1),order='phi1')
    result = {'DESI':10**((np.interp(phi1,t['phi1'],t['r-Mr'])-10)/5)}
    inputs['desi_reference']=MODULUS_CURVE
    for name in ('cfht','sdss'):
        p = ROOT/f'manifests/v3_16/{name}_reference.csv'
        x = read(p)
        a,b = ([float(r[k]) for r in x] for k in ('phi1','distance_kpc'))
        result[name.upper()] = np.interp(phi1,a,b,left=np.nan,right=np.nan)
        inputs[name+'_reference']=sha(p)
    return result


def summary(t,reference,q,use):
    k = use&np.isfinite(reference)&np.isfinite(q).all(axis=1)
    stats = distance_summary(10+5*np.log10(q[k]),10+5*np.log10(reference[k]))
    if k.sum()<2:
        return stats
    diff = q[k,2]/reference[k]-1
    blocks = np.floor(t['phi1'][k])
    chunks = [diff[blocks==b] for b in np.unique(blocks)]
    rng = np.random.default_rng(C['bootstrap_seed'])
    if len(chunks)>1:
        draws = [np.median(np.concatenate([chunks[j] for j in ix])) for ix in
                 rng.integers(len(chunks),size=(C['bootstrap_draws'],len(chunks)))]
        lo,hi=np.quantile(draws,[.025,.975])
        stats.update(median_bias_p025=lo,median_bias_p975=hi)
    return dict(**stats,angular_blocks=len(chunks))


def rosters(t):
    yield 'all','all',np.ones(len(t['ls_id']),bool)
    yield 'purity','p>0.9 and velocity',t['high_purity']
    yield 'location','target ribbon',t['desi_ribbon']==0
    for name,key in [('phase','phase'),('segment','segment'),('gravity','gravity'),('metallicity','metallicity')]:
        for j in np.unique(t[key]):
            yield name,str(j),t[key]==j
    for name,quant,edges in [('color',t['color'],C['color_edges']),
                             ('r',t['r'],C['depth_edges'])]:
        for a,b in zip(edges,edges[1:]):
            yield name,f'{a:g} to {b:g}',(quant>=a)&(quant<b)


def run():
    inputs = {'config':sha(CONFIG),'style':check_templates()}
    bind(MEMBERS/'complete.json',inputs)
    raw = [r for r in read(MEMBERS/'members.csv') if r['catalog']=='DESI' and r['group'] in ('thin','cocoon')]
    assert len(raw)==len({r['source_id'] for r in raw})
    assert all(r['native_type']=='PSF' for r in raw)
    names = ('ls_id','source_id','phi1','segment','desi_ribbon','color','r','r0','p_thin','p_cocoon','ra','dec')
    integer = {'ls_id','source_id','segment','desi_ribbon'}
    t = {k:np.array([(int(r[k]) if k in integer else float(r[k]) if r[k] else np.nan) for r in raw]) for k in names}
    t['RA'],t['DEC'] = t.pop('ra'),t.pop('dec')
    t['TYPE'] = np.array([a['native_type'] for a in raw])
    t['group'] = np.array([a['group'] for a in raw])
    t['clean'],t['qrf_archived'] = (np.array([x[k]=='True' for x in raw]) for k in ('clean','qrf_archived'))
    _,_,training=model_and_history('north',inputs)
    ledger = read_receipt(NORTH/'native_model','complete.json','roles.npz',('ls_id','role'),inputs)
    cut = np.isin(ledger['role'],('gaia_fit','specdis_fit','calibration'))
    t['train_cal']=np.isin(t['source_id'],training)|np.isin(t['ls_id'],ledger['ls_id'][cut])
    z = table('v4_Table7_all.fits')
    k=unique_join(t['source_id'],z['SOURCE_ID'])
    assert (k>=0).all()
    t.update(feh=z['FEH'][k],feh_err=z['FEH_ERR'][k],v_error=z['V_ERR'][k],delta_v=z['DELTA_VGSR'][k])
    valid = (t['v_error']>0)&(t['v_error']<=C['high_purity_velocity_error_km_s'])
    t['high_purity'] = valid&(abs(t['delta_v'])<=C['high_purity_dvg_km_s'])&(t['p_thin']>=C['high_purity_pthin'])
    rvs = rvs_evidence(t,inputs)
    t.update(rvs)
    good = (t['rvs_logg_err']>0)&(t['rvs_logg_err']<=C['maximum_logg_error_dex'])
    t['gravity'] = np.select([good&(t['rvs_logg']>=4),good&(t['rvs_logg']>=3.5),good],
                             ['high','intermediate','low'],default='unavailable')
    good = (t['feh_err']>0)&(t['feh_err']<=C['maximum_feh_error_dex'])&np.isfinite(t['feh'])
    t['metallicity'] = np.select([good&(t['feh']<-2.3),good],['below -2.3','at least -2.3'],default='unavailable')
    ref=refs(t['phi1'],inputs)
    t['phase']=phase(t['color'],t['r0']-10-5*np.log10(ref['DESI']))
    indices,(edges,mass,q0) = read_pdfs(t,inputs)
    used=t['clean']&~t['train_cal']&t['qrf_archived']
    pred = np.full((len(used),5),np.nan)
    pred[indices]=q0
    pred[~used]=np.nan
    assert np.array_equal(np.isfinite(pred).all(axis=1),used)
    rows=[]
    for group in ('thin','cocoon'):
        g=t['group']==group
        for name,key,use in rosters(t):
            for source,d in ref.items():
                u=g&use
                if not u.any():
                    continue
                rows.append(dict(group=group,stratum=name,value=key,reference=source,
                    inventory=int(u.sum()),clean=int((u&t['clean']).sum()),archived=int((u&t['qrf_archived']).sum()),
                    training_cal=int((u&t['train_cal']).sum()),**summary(t,d,pred,u)))
    out=DEST/'members'
    out.mkdir(parents=True,exist_ok=True)
    csv_rows(out/'residuals.csv',rows)
    csv_rows(out/'roster.csv',[{k:v[j] for k,v in dict(**t,desi_kpc=ref['DESI'],
             cfht_kpc=ref['CFHT'],sdss_kpc=ref['SDSS'],median_kpc=pred[:,2],science=used).items()} for j in range(len(used))])
    b=shell_mass(edges,mass,CUTS)
    np.testing.assert_allclose(b.sum(axis=1),1.,rtol=0,atol=2e-7)
    marks = (('Validation segment',1,10.125),('Exploration segment',2,10.375))
    peak_rows=[]
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
        for ax,(name,seg,field) in zip(axes,marks):
            for g,color in [('thin',BLUE),('cocoon',ORANGE)]:
                take=used&(t['group']==g)&(t['segment']==seg)&(t['desi_ribbon']==0)
                mask=take[indices]
                values=b[mask,1:-1].sum(axis=0)
                if mask.any():
                    ax.stairs(values/mask.sum(),D_EDGES,label=g.capitalize(),color=color)
                    p=float((D_EDGES[:-1]+.125)[values.argmax()])
                    peak_rows.append(dict(segment=seg,group=g,n=int(mask.sum()),member_peak=p,
                        signed_field_peak=field,member_reference_median=float(np.median(ref['DESI'][take]))))
            ax.axvline(field,color=GRAY,ls='--',label='Signed field peak')
            ax.set(title=name,xlabel='Heliocentric distance (kpc)',ylabel='Mean PDF mass per bin',xlim=(5,14))
            ax.legend()
        save(fig,out,'members_vs_field',peak_rows)
    figure_check(out/'members_vs_field.png')
    atomic_npz(out/'posteriors.npz',TYPE=t['TYPE'][indices],ls_id=t['ls_id'][indices],
        source_id=t['source_id'][indices],science=used[indices],mu_edges=edges,mass=mass,quantiles_kpc=q0)
    save_json(out/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        outputs={p.name:sha(p) for p in out.iterdir() if p.name!='complete.json'}))


if __name__=='__main__':
    require_compute()
    run()
