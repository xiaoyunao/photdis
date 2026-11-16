#!/usr/bin/env python3
"""Train on native DR11 north photometry and evaluate shared Gaia identities."""
import json
from pathlib import Path
from types import SimpleNamespace

import fitsio
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier

from photdis.baseline_v0 import HistogramQRF
from photdis.photometry import CLEAN_REJECT_MASKBITS
from photdis.stage1 import galactic, legacy_ids, require_compute, tile_bounds
from photdis.v3.data import stable_rank
from photdis.v3.metrics import distance_summary
from photdis.v3_3 import class_log_likelihood, frozen_qrf
from v3_3_catalog import joint_summary
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_6_distance_evidence import OUT as EVIDENCE
from v3_6_vac_distances import read_columns
from v3_8_candidate import read_cache

OUT = ROOT/'data/v3_8/north'
SWEEPS = Path('/global/cfs/cdirs/cosmo/data/legacysurvey/dr11/north/sweep/11.0')
SYSTEM = 'BASS_MzLS_DR11'
BANDS = np.array([0,1,3])
META = ('RA','DEC','TYPE','RELEASE','BRICKID','OBJID','REF_ID','REF_CAT','EBV','MASKBITS')
PHOTOMETRY = ('FLUX','FLUX_IVAR','MW_TRANSMISSION','NOBS','FRACFLUX')
FIELDS = (*META,*(f'{q}_{b}' for q in PHOTOMETRY for b in 'GRZ'))


def north_matches(south):
    rows = []
    log = []
    sort = np.argsort(south['source_id'])
    ids = south['source_id'][sort]
    for path in sorted(SWEEPS.glob('sweep-*.fits')):
        r0,b0,_,b1 = tile_bounds(path)
        if r0 not in (130,140,150,160,170,180,190,200,210,220,230,240) or b0 not in (30,35,40):
            continue
        with fitsio.FITS(str(path)) as f:
            raw = f[1].read(columns=list(FIELDS))
        morphology = np.char.strip(raw['TYPE'].astype('U'))
        positions = np.searchsorted(ids,raw['REF_ID'])
        idx = np.flatnonzero(positions<len(ids))
        idx = idx[ids[positions[idx]]==raw['REF_ID'][idx]]
        idx = idx[np.char.startswith(np.char.strip(raw['REF_CAT'][idx].astype('U')),'G')]
        nref = len(idx)
        kind = morphology[idx]
        logrow = dict(sweep=path.name,raw_rows=len(raw),matched_ref_key=nref,
                      non_psf=int(np.sum((kind!='')&(kind!='PSF'))),missing_type=int(np.sum(kind=='')))
        idx = idx[(kind=='PSF')&(raw['RELEASE'][idx]==11011)]
        a = raw[idx]
        j = sort[positions[idx]]
        separation = np.hypot(((a['RA']-south['RA'][j]+180)%360-180)*np.cos(np.deg2rad(a['DEC'])),
                              a['DEC']-south['DEC'][j])*3600
        _,b = galactic(a['RA'],a['DEC'])
        bits = sum(1<<i for i in CLEAN_REJECT_MASKBITS)
        clean = ((abs(b)>30)&(a['EBV']>=0)&(a['EBV']<.03)&(separation<1.)&
                 ((a['MASKBITS']&bits)==0))
        f,v,t = [np.column_stack([a[f'{field}_{b}'] for b in 'GRZ']).astype('f8')
                 for field in ('FLUX','FLUX_IVAR','MW_TRANSMISSION')]
        for band in 'GRZ':
            clean &= (a[f'NOBS_{band}']>0)&(a[f'FRACFLUX_{band}']<1)
        clean &= ((f>0)&(v>0)&(t>0)&(t<=1)&(f*np.sqrt(np.maximum(v,0))>=5)).all(axis=1)
        with np.errstate(divide='ignore',invalid='ignore'):
            clean &= 22.5-2.5*np.log10(f[:,1])<20
        a,j,f,v,t = (x[clean] for x in (a,j,f,v,t))
        log.append(dict(**logrow,verified_pairs=len(j),size_bytes=path.stat().st_size,
                        mtime_ns=path.stat().st_mtime_ns))
        if len(j):
            flux = np.zeros((len(j),6))
            error,trans = np.ones_like(flux),np.ones_like(flux)
            mask = np.zeros_like(flux,dtype=bool)
            flux[:,BANDS],error[:,BANDS],trans[:,BANDS],mask[:,BANDS] = f,1/np.sqrt(v),t,True
            rows.append(dict(ls_id=legacy_ids(a),TYPE=np.full(len(j),'PSF'),source_id=a['REF_ID'],
                north_ra=a['RA'],north_dec=a['DEC'],ebv=a['EBV'],flux=flux,error=error,trans=trans,
                mask=mask,mag=22.5-2.5*np.log10(f/t),magerr=(2.5/np.log(10))/(f*np.sqrt(v)),south=j))
    if not rows:
        raise ValueError('No verified north–south Gaia pairs in the registered sweeps')
    data = {k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
    x,inv,counts = np.unique(data['source_id'],return_inverse=True,return_counts=True)
    take = counts[inv]==1
    return {k:v[take] for k,v in data.items()},log,int((~take).sum())


def native_sed(p,d,cls,train):
    xyz = p['flux'][train]/p['trans'][train]*d[train,None]**2
    binary = .25
    mag = p['mag'][train]
    f = np.column_stack((mag[:,0]-mag[:,1],mag[:,1]-mag[:,2],mag[:,0]-10-5*np.log10(d[train])))
    lab = KMeans(n_clusters=12,n_init=3,random_state=3803).fit_predict(f/np.maximum(f.std(axis=0),.05))
    c = np.bincount(lab,minlength=12)
    model = SimpleNamespace(weights=c/c.sum(),system=SYSTEM,binary_reference_fraction=binary,
        binary_brightness_factor=2.,means=np.zeros((12,6)),covariances=np.zeros((12,6,6)),
        classes=np.empty((12,3)))
    for i in range(12):
        g = lab==i
        u = xyz[g]
        m = u.mean(axis=0)
        cov = np.cov(u,rowvar=False) if len(u)>1 else np.zeros((6,6))
        model.means[i] = m
        model.covariances[i] = cov + np.diag(np.maximum(.06*abs(m),1e-4)**2)
        state = cls[train][g]
        evidence = np.bincount(state[state>=0],minlength=3)+.5
        model.classes[i] = evidence/evidence.sum()
    return model


def main():
    require_compute()
    OUT.mkdir(parents=True,exist_ok=True)
    marker = OUT/'complete.json'
    if marker.exists():
        raise FileExistsError(marker)
    a = read_columns(read_cache(EVIDENCE,'Gaia.npz'),('ls_id','source_id','RA','DEC','TYPE','reference_kpc',
        'primary','high_quality','qrf_training','consensus_class','qrf_quantiles_kpc','dereddened_g_r'))
    selected = (a['TYPE']=='PSF')&a['high_quality']&~a['qrf_training']&(a['source_id']>0)
    a = {k:v[selected] for k,v in a.items()}
    p,logs,duplicates = north_matches(a)
    csv_rows(OUT/'sweeps.csv',logs)
    s = p['south']
    d,cls = a['reference_kpc'][s],a['consensus_class'][s]
    field = np.floor(p['north_ra']/10).astype(int)
    part = field%4
    fit,cal,test = part<2,part==2,part==3
    # Limit pilot fitting and validation after assigning disjoint sky blocks.
    def subset(mask,limit):
        j = np.flatnonzero(mask)
        return j[np.argsort(stable_rank(p['source_id'][j],3804))[:limit]]
    train,hold,t = subset(fit,16000),subset(cal,4000),subset(test,6000)
    if min(len(train),len(hold),len(t))<128:
        raise ValueError(f'Pilot needs all spatial roles: {len(train)}, {len(hold)}, {len(t)}')
    mu = 10+5*np.log10(d)
    qrf = HistogramQRF(2)
    qrf.forest.n_jobs = 8
    qrf.fit(p['mag'][train],p['mag'][train,0]-mu[train],cls[train])
    qrf.forest.n_jobs = 1
    x = np.column_stack((p['mag'][:,0]-p['mag'][:,1],p['mag'][:,1]-p['mag'][:,2],p['mag'][:,1]))
    q = frozen_qrf(qrf,p['mag'][np.r_[hold,t]],p['magerr'][np.r_[hold,t]],p['source_id'][np.r_[hold,t]])
    cal_q,qt = np.split(q['distance'],[len(hold)])
    calibrated = np.median(mu[hold]-10-5*np.log10(cal_q[:,2]))
    qt = qt*10**(calibrated/5)
    keep = q['supported'][len(hold):]
    model = native_sed(p,d,cls,train)
    spec_train = train[cls[train]>=0]
    learner = None
    if len(np.unique(cls[spec_train]))>=2:
        learner = HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=12,random_state=3804).fit(x[spec_train],cls[spec_train])
    dwarf = np.full(len(t),np.nan)
    if learner is not None and 0 in learner.classes_:
        dwarf = learner.predict_proba(x[t])[:,np.flatnonzero(learner.classes_==0)[0]]
    grid = np.linspace(-2,25,271)
    prior = model.weights@model.classes
    subset_sed = np.flatnonzero(keep)[:512]
    qnear = np.full(len(t),np.nan)
    for index in subset_sed:
        j = t[index]
        obs = dict(flux=p['flux'][j],flux_error=p['error'][j],transmission=p['trans'][j],
                   band_mask=p['mask'][j],photometric_system=SYSTEM,EBV=float(p['ebv'][j]))
        curve = class_log_likelihood(model,obs,grid)
        qnear[index] = joint_summary(grid,curve,prior)['qnear']
    np.testing.assert_equal(len(np.unique(p['source_id'][np.r_[train,hold,t]])),len(train)+len(hold)+len(t))
    south = a['qrf_quantiles_kpc'][s[t]]
    use = np.isfinite(qt).all(axis=1)&keep
    summaries = [dict(model=name,**distance_summary(10+5*np.log10(qv[use]),mu[t][use]))
                 for name,qv in [('South published',south),('North native',qt)]]
    summaries.append(dict(model='North versus south',**distance_summary(10+5*np.log10(qt[use]),
                      10+5*np.log10(south[use,2]))))
    csv_rows(OUT/'distance.csv',summaries)
    csv_rows(OUT/'selection.csv',[dict(sed_evaluated=len(subset_sed),north_nominal_selected=int(np.sum((dwarf>=.80)&(qnear>=.90))),
         south_primary_same_rows=int(a['primary'][s[t[subset_sed]]].sum()),
         known_test_labels=int(np.sum(cls[t]>=0)),available_north_dwarf_scores=int(np.isfinite(dwarf).sum()))])
    atomic_npz(OUT/'pairs.npz',ls_id=p['ls_id'][t],TYPE=p['TYPE'][t],south_ls_id=a['ls_id'][s[t]],
        gaia_id=p['source_id'][t],north_q=qt,south_q=south,reference_kpc=d[t],north_mag=p['mag'][t],
        south_gr=a['dereddened_g_r'][s[t]],support=keep,dwarf_score=dwarf,sed_near=qnear,north_ra=p['north_ra'][t],
        north_dec=p['north_dec'][t])
    atomic_npz(OUT/'roles.npz',ls_id=p['ls_id'][np.r_[train,hold,t]],gaia_id=p['source_id'][np.r_[train,hold,t]],
        TYPE=np.full(len(train)+len(hold)+len(t),'PSF'),role=np.repeat(['fit','calibration','test'],[len(train),len(hold),len(t)]))
    joblib.dump(dict(qrf=qrf,mu_offset=calibrated,classifier=learner,sed=model),OUT/'prototype.joblib')
    paths = (EVIDENCE/'complete.json',Path(__file__).resolve())
    save_json(marker,dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in paths},native_system=SYSTEM,
        scope='Gaia identity and angular matched south–north overlap; Gaia reference shapes are shared',
        train=len(train),calibration=len(hold),test=len(t),duplicate_north_gaia_exclusions=duplicates,
        calibration_offset_mag=float(calibrated),outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p!=marker}))


if __name__=='__main__':
    main()
