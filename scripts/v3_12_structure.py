#!/usr/bin/env python3
"""Compare structural information across frozen and probability-weighted forecasts."""

import csv
import json
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.special import ndtr

from photdis.baseline_v1 import support
from photdis.photometry import CLEAN_REJECT_MASKBITS
from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute, tile_bounds
from photdis.v3.controls import apply_temperature, probe_features
from photdis.v3_3 import frozen_qrf
from v2_1_near import ROOT, atomic_npz, save_json, sha
from v3_3_sed import verified
from v3_4_near_forecast import prepare
from v3_5_dr11_area import BANDS, random_memmap
from v3_6_vac_distances import read_columns
from v3_8_direct_figures import BLUE, GRAY, ORANGE, STYLE, plt, save
from v3_9_catalog import read_native, scan_chunk, settings
from v3_9_delivery import DEST as RELEASE
from v3_9_references import join, subset
from v3_10_coverage import ANNULI
from v3_10_figures import check_templates
from v3_10_science import DISTANCE
from v3_10_selection import DEST as CLASS_OLD
from v3_10_working_sample import RADIAL_MASS, SCORE, shell_mass
from v3_11_selection import DEST as CLASS_NEW
from v3_11_structure import CLUSTERS, DEST as ARCHIVE, overlaps
from v3_12_reference import DEST as REFERENCE, M30, predict, separation

DEST = ROOT/'data/v3_12/structure'
INPUT = ROOT/'data/v3_12/input/members.csv'
MEMBERS_SHA = 'e868d6e012a440e61984af1cbd16492662ef91ce2e99dca48d56dd734c2edde0'
EXTINCTION_LIMIT = .08
D30 = 8.49


def zones(ra,dec):
    d = separation(ra,dec,M30)
    return tuple((d>=lo)&(d<hi) for lo,hi in ANNULI)


def south_area(inputs):
    cfg = settings()['south']
    path = Path(cfg['sweep_root']).parent.parent/'randoms/randoms-south-1-0.fits'
    raw,density,names = random_memmap(path)
    masks = sum(1<<k for k in CLEAN_REJECT_MASKBITS)
    limit = (2.5/np.log(10)/.1/(10**((22.5-20)/2.5)))**2
    count = np.zeros(2)
    for i in range(0,len(raw),300000):
        a = raw[i:i+300000]
        ra,dec = a[names['RA']],a[names['DEC']]
        k = (abs(ra-M30[0])<1.5)&(abs(dec-M30[1])<1.25)
        if not k.any():
            continue
        b = a[k]
        good = (b[names['RELEASE']]==cfg['release'])&(np.char.strip(b[names['PHOTSYS']])==b'S')
        good &= (b[names['EBV']]>=0)&(b[names['EBV']]<EXTINCTION_LIMIT)&((b[names['MASKBITS']].astype('i8')&masks)==0)
        for band in BANDS:
            good &= (b[names['NOBS_'+band]]>0)&(b[names['PSFDEPTH_'+band]]>=limit)
        count += [(good&j).sum() for j in zones(ra[k],dec[k])]
    inputs['m30_random_catalog'] = dict(file=str(path),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
    return count/density


def load_model(root, file, inputs):
    marker = root/'complete.json'
    d = json.loads(marker.read_text())
    inputs[str(marker.relative_to(ROOT))] = sha(marker)
    return joblib.load(verified(root/file,d['outputs'][file]))


def class_probability(fit, v):
    return apply_temperature(fit['model'].predict_proba(v),fit['temperature'])[:,0]


def m30(version, inputs):
    cfg = settings()['south']
    cfg['maximum_grz_error_mag'] = settings()['maximum_grz_error_mag']
    cfg['domains'] = {'south_test':{'r_max':20.,'abs_b_min':30.,'ebv_max':EXTINCTION_LIMIT}}
    root = Path(cfg['sweep_root'])
    tiles = []
    for source in sorted(root.glob('sweep-*.fits')):
        r0,b0,r1,b1 = tile_bounds(source)
        if not (r0<M30[0]+1.5 and r1>M30[0]-1.5 and b0<M30[1]+1.25 and b1>M30[1]-1.25):
            continue
        data = read_native(source)
        on,off = zones(data['RA'],data['DEC'])
        in_field = on|off
        if not in_field.any():
            continue
        stat = source.stat()
        inputs[str(source)] = dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
        raw = data[in_field]
        p = join([v for v,_ in scan_chunk(raw,(r0,b0,r1,b1),cfg,0) if v])
        if p:
            p = subset(p,p['row_state']==7)
            tiles.append(p)
    p = join(tiles)
    if not p or not len(p['ls_id']):
        raise ValueError('No qualified native PSF in the predeclared M30 footprint')
    if not np.all(p['TYPE']=='PSF') or len(np.unique(p['ls_id']))!=len(p['ls_id']):
        raise ValueError('M30 native source identity conflict')
    models,_,bound = prepare()
    inputs.update({str(path.relative_to(ROOT)):sha(path) for path in bound})
    qrf = models['qrf']
    train_match = np.isin(p['identity'],qrf.training_identity)
    n_train = int(train_match.sum())
    p = subset(p,~train_match)
    old = load_model(CLASS_OLD/'south','model.joblib',inputs)
    repaired = load_model(CLASS_NEW/'south','model.joblib',inputs)
    train_hard = []
    for root,c in ((CLASS_OLD,'training'),(CLASS_NEW,'new_train')):
        marker = root/'south/complete.json'
        inputs[str(marker.relative_to(ROOT))] = sha(marker)
        keys = ('ls_id',c,'calibration' if c=='training' else 'new_cal')
        a = read_columns(verified(marker.parent/'scores.npz',json.loads(marker.read_text())['outputs']['scores.npz']),keys)
        train_hard.extend(a['ls_id'][a[c]|a[keys[-1]]])
    touched = np.isin(p['ls_id'],train_hard)
    direct_class = int(touched.sum())
    p = subset(p,~touched)
    q = []
    valid = np.flatnonzero(support(qrf,np.pad(p['mag_grz'],((0,0),(0,2)))))
    for i in range(0,len(valid),1024):
        j = valid[i:i+1024]
        a = frozen_qrf(qrf,p['mag_grz'][j],p['error_grz'][j],p['identity'][j])
        k = a['supported']
        q.append(dict(i=j[k],edges=a['mu_edges'][k],mass=a['mass'][k]))
    a = join(q)
    if not a:
        raise ValueError('No in-support M30 frozen QRF forecasts')
    p = subset(p,a['i'])
    base = probe_features(p,'grz')
    cuts = 10+5*np.log10(DISTANCE)
    x = version['ablation']['full_feature_control']
    score = class_probability(x,base)
    rec = dict(TYPE=p['TYPE'],ls_id=p['ls_id'],source_id=p['source_id'],RA=p['RA'],DEC=p['DEC'],
               target=zones(p['RA'],p['DEC'])[0],g0=p['g0'],r0=p['r0'],EBV=p['EBV'],
               score=class_probability(old,base),candidate=class_probability(repaired,base),
               candidate_threshold=np.full(len(base),repaired['threshold']),clean_score=score,
               pdf_mass=shell_mass(a['edges'],a['mass'],cuts),
               radial=shell_mass(a['edges'],a['mass'],10+5*np.log10([2,10]))[:,0]>=RADIAL_MASS)
    return rec,south_area(inputs),n_train,direct_class


def north(name,inputs):
    marker = ARCHIVE/'complete.json'
    receipt = json.loads(marker.read_text())
    inputs[str(marker.relative_to(ROOT))] = sha(marker)
    file = f'{name.lower()}_sources.npz'
    t = read_columns(verified(ARCHIVE/file,receipt['outputs'][file]),
                     ('TYPE','ls_id','g0','r0','RA','DEC','state','new_score','score','radial_mass','target','pdf_mass'))
    t['candidate'] = t['new_score']
    c = CLASS_NEW/'north/complete.json'
    meta = json.loads(c.read_text())
    inputs[str(c.relative_to(ROOT))] = sha(c)
    t['candidate_threshold'] = np.full(len(t['ls_id']),meta['threshold'])
    t['radial'] = t['radial_mass'].sum(axis=1)>=RADIAL_MASS
    r = RELEASE/'complete.json'
    inputs[str(r.relative_to(ROOT))] = sha(r)
    with verified(RELEASE/'unified_parts.csv',json.loads(r.read_text())['outputs']['unified_parts.csv']).open() as f:
        parts = [part for part in csv.DictReader(f) if part['side']=='north' and overlaps(name,part['sweep'])]
    matched = np.zeros(len(t['ls_id']),bool)
    source = np.zeros(len(matched),dtype='i8')
    for part in parts:
        p = read_columns(verified(ROOT/part['photo_file'],part['photo_sha256']),('TYPE','ls_id','source_id'))
        i = unique_join(t['ls_id'],p['ls_id'])
        ok = i>=0
        if np.any(matched&ok) or not np.all(p['TYPE'][i[ok]]=='PSF'):
            raise ValueError('Ambiguous native ID in cluster photo match')
        source[ok] = p['source_id'][i[ok]]
        matched |= ok
    if not matched.all():
        raise ValueError('Missing DR11 photo rows for northern cluster')
    t['source_id'] = source
    m = receipt['clusters'][name]
    return t,np.array([m['target_area_deg2'],m['control_area_deg2']])


def stats(pdf,weight,zone,area,d,member,mask):
    target,control = mask&zone,mask&~zone
    use = (target,control)
    profiles = np.array([(pdf[k]*weight[k,None]).sum(axis=0)/den for k,den in zip(use,area)])
    net = profiles[0]-profiles[1]
    peak = int(np.argmax(net))
    mid = (DISTANCE[:-1]+DISTANCE[1:])/2
    w = np.clip((np.minimum(DISTANCE[1:],d+1)-np.maximum(DISTANCE[:-1],d-1))/np.diff(DISTANCE),0,1)
    window = (pdf@w)*weight
    v = sum(np.square(window[k]/den).sum() for k,den in zip(use,area))
    excess = float(net@w)
    n = int((member&target).sum())
    n_weight = float(weight[member&target].sum())
    return profiles,net,dict(peak_kpc=float(mid[peak]) if net[peak]>0 else None,
        peak_bias=float(mid[peak]/d-1) if net[peak]>0 else None,reference_window_excess=excess,
        reference_window_poisson_error=float(np.sqrt(v)),
        reference_window_poisson_z=float(excess/np.sqrt(v)) if v>0 else None,
        target_rows=int(target.sum()),background_rows=int(control.sum()),
        target_weight=float(weight[target].sum()),background_weight=float(weight[control].sum()),
        member_matches=n,member_weight=n_weight,member_retention=n_weight/n if n else None,
        background_window=float(profiles[1]@w),
        background_weight_without_confirmed_member=float(weight[control&~member].sum()))


def render(name,t,area,d,rel,member,scale):
    color = t['g0']-t['r0']
    median,width = predict(rel,color,t['r0'])
    support_simple = np.isfinite(median)
    cdf = ndtr((10+5*np.log10(DISTANCE)[None,:]-median[:,None])/width[:,None])
    simple = np.where(support_simple[:,None],np.diff(cdf,axis=1),0.)
    prob = t['clean_score'] if name=='M30' else t['candidate']
    one = np.ones(len(t['ls_id']))
    q = t['pdf_mass']
    versions = [('QRF supported',q,one),('Original hard',q,(t['score']>=SCORE)&t['radial']),
                ('Revised hard',q,(t['candidate']>=t['candidate_threshold'])&t['radial']),
                ('Soft dwarf',q,prob),('Soft matched budget',q,prob*scale),
                ('QRF common color support',q,support_simple),('Color relation',simple,support_simple)]
    rows, curves = [],{}
    for row,(label,b) in enumerate((('All colors',np.isfinite(color)),('Blue stars',color<.5))):
        with plt.rc_context(STYLE):
            f,axs = plt.subplots(2,3,figsize=(18,9),layout='constrained',sharex=True)
            panels = (0,1,2,3,5,6)
            for i,(view,pdf,weight) in enumerate(versions):
                p,net,out = stats(pdf,weight.astype(float),t['target'],area,d,member,b)
                curves[f'{row}_{i}'] = np.vstack((p,net))
                rows.append(dict(cluster=name,color=label,method=view,reference_kpc=d,
                                 target_deg2=area[0],background_deg2=area[1],**out))
                if i not in panels:
                    continue
                ax = axs.flat[panels.index(i)]
                ax.stairs(p[0],DISTANCE,color=BLUE,label='Cluster annulus')
                ax.stairs(p[1],DISTANCE,color=ORANGE,label='Control annulus')
                ax.stairs(net,DISTANCE,color=GRAY,label='Excess')
                ax.axvline(d,color=GRAY,ls='--',lw=1.25)
                ax.set(xlabel='Distance (kpc)',ylabel='Stars per cell per sq. deg.',title=view,xlim=(1,20))
            axs.flat[0].legend()
            f.suptitle(f'{name} · {label} · full distance distributions')
            save(f,DEST,f'{name.lower()}_{row}',[r for r in rows if r['color']==label],
                 {'distance_bins':DISTANCE,**{k:v for k,v in curves.items() if k.startswith(f'{row}_')}})
    return rows


def run(group):
    inputs = {'style':check_templates(),'membership':MEMBERS_SHA,
              'membership_url':'https://zenodo.org/records/4891252',
              'cluster_distance_source':'https://academic.oup.com/view-large/353698133',
              'm30_ebv_max':EXTINCTION_LIMIT}
    DEST.mkdir(parents=True,exist_ok=True)
    reg = 'south' if group=='m30' else 'north'
    model = load_model(REFERENCE/reg,'models.joblib',inputs)
    catalog = {}
    with verified(INPUT,MEMBERS_SHA).open() as f:
        for row in csv.DictReader(f):
            if float(row['memberprob'])>.9:
                catalog.setdefault(row['cluster'],[]).append(int(row['source_id']))
    results = {}
    for name in (('M30',) if group=='m30' else tuple(CLUSTERS)):
        if name=='M30':
            t,area,qrf_overlap,class_overlap = m30(model,inputs)
            d = D30
        else:
            t,area = north(name,inputs)
            d = CLUSTERS[name][1]
            qrf_overlap,class_overlap = None,None
        matches = np.isin(t['source_id'],catalog[name])
        matches &= unique_join(t['source_id'],t['source_id'])>=0
        if not np.all(t['TYPE']=='PSF') or (area<=0).any():
            raise ValueError(f'Ineligible PSF or sky area in {name}')
        render(name,t,area,d,model['relation'],matches,model['scale'])
        atomic_npz(DEST/f'{name.lower()}_members.npz',
                    **{k:t[k] for k in ('TYPE','ls_id','source_id','RA','DEC','g0','r0','target')},member=matches)
        results[name] = dict(rows=len(matches),member_matches=int(matches.sum()),
            member_target=int((matches&t['target']).sum()),published_members=len(catalog[name]),
            excluded_frozen_qrf_fit=qrf_overlap,excluded_archived_class_fit_or_cal=class_overlap,
            area=area,comparison_fits_exclude_m30_two_degrees=True)
    names = {name.lower() for name in results}
    save_json(DEST/f'{group}.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,results=results,
        outputs={f.name:sha(f) for f in DEST.iterdir() if f.stem.split('_')[0] in names}))


if __name__=='__main__':
    require_compute()
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group',choices=('north','m30'))
    run(parser.parse_args().group)
