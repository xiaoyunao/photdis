#!/usr/bin/env python3
"""Joint application validation and targeted spectral evidence for full DECam."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import chain
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from photdis.gaia import derive_gaia_labels, join_gaia
from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from photdis.v3.data import stable_rank
from photdis.v3.metrics import distance_summary, histogram_parallax
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_3_sed import verified
from v3_4_near_recover_labels import archived_labels
from v3_4_near_reference_support import past_evidence, separation, spectroscopic_inventory
from v3_4_near_validate import GAIA, OUT as V34_EVIDENCE, geometry, read_targeted, tile_inventory
from v3_5_dr11_pilot_evidence import PRODUCT as PILOT_EVIDENCE
from v3_5_dr11_production import CUTS, FREEZE, PRODUCTION
from v3_5_dr11_spatial import DELIVERY, read_tile

OUT = PRODUCTION.parent/'validation'
KEYS = ('ls_id','identity','source_id','TYPE','RA','DEC','observed_r','dereddened_g_r','stratum',
        'primary','strict','qrf_bins','qrf_quantiles_kpc','reused_v34')
PDF_KEYS = ('qrf_mu_edges','qrf_mu_mass')
SEED = 350419
MIN_WEAK_SNR = 5.
REFERENCE_PRIOR = np.array([5.,10+5*np.log10(20.)])
_STATE = None


def join_parts(parts):
    return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else {}


def take_rows(fields, index):
    return {k:v[index] for k,v in fields.items()}


def draw_tile(path, sorted_targets):
    marker = json.loads((path/'complete.json').read_text())
    files = sorted(f for f in marker['outputs'] if f.startswith('part_'))
    inputs, spectra = [], []
    for j,f in enumerate(files):
        with np.load(path/f) as z:
            selection = z['primary']
            if selection.any():
                native = {k:z[k][selection] for k in KEYS}
                native['file'] = np.full(int(selection.sum()),j)
                native['row'] = np.flatnonzero(selection)
                inputs.append(native)
                source_id = native['source_id']
                position = np.searchsorted(sorted_targets,source_id)
                matched = position<len(sorted_targets)
                matched[matched] = sorted_targets[position[matched]]==source_id[matched]
                if matched.any():
                    spectra.append(take_rows(native,matched))
    if not inputs:
        return {},[],{}
    t = join_parts(inputs)
    main = t['primary']
    if not (t['TYPE']=='PSF').all() or not main.all() or len(np.unique(t['ls_id']))!=len(main):
        raise ValueError(f'Invalid validation frame: {path.name}')
    groups = 6*t['stratum'] + np.argmax(t['qrf_bins'],axis=1)
    salt = stable_rank(t['ls_id'],SEED)
    take, weights = [], []
    for key in np.unique(groups):
        i = np.flatnonzero(groups==key)
        # Oversample the joint faint/blue range and nominal far shells. The
        # cap is constant within each defined design stratum.
        j = i[0]
        focus = t['observed_r'][j]>=18 and .5<=t['dereddened_g_r'][j]<.8
        cap = 18 if focus or key%6 in (1,2) else 6
        picked = i[np.argsort(salt[i],kind='stable')[:cap]]
        take.extend(picked.tolist())
        weights.extend([len(i)/len(picked)]*len(picked))
    sample = take_rows(t,np.asarray(take,dtype=int))
    sample['design_weight'] = np.asarray(weights)
    np.testing.assert_allclose(sample['design_weight'].sum(),len(main),atol=1e-8,rtol=0)
    for name in PDF_KEYS:
        holders = {}
        for j in np.unique(sample['file']):
            m = sample['file']==j
            with np.load(path/files[j]) as z:
                holders[j] = z[name][sample['row'][m]]
        first = next(iter(holders.values()))
        values = np.empty((len(take),first.shape[1]),dtype=first.dtype)
        for j,m in holders.items():
            values[sample['file']==j] = m
        sample[name] = values
    sample['sweep'] = np.full(len(take),path.name)
    return sample,spectra,dict(rows=len(main),sampled=len(take),strict=int(t['strict'].sum()))


def acquire_spectral_metadata(hashes):
    # Retain native matched reference identities and row pointers before
    # opening stellar class or distance in the outcome step.
    import fitsio
    from v3_2_minesweeper_inventory import COLUMNS,DIGEST,FILENAME
    from v3_2_reference_inventory import text

    spec = spectroscopic_inventory(hashes)
    path = ROOT/'data/v3_2/reference_sources/dr20_minesweeper'/FILENAME
    verified(path,DIGEST)
    with fitsio.FITS(str(path)) as f:
        names = {n.lower():n for n in f[1].get_colnames()}
        z = f[1].read(columns=[names[k] for k in COLUMNS])
    sid,ra,dec,obs,mjd,visits = [z[names[k]] for k in COLUMNS]
    use = (sid>0)&(text(obs)=='APO')&(visits>0)&(mjd>0)&(mjd<=60708)&np.isfinite(ra)&np.isfinite(dec)&(abs(dec)<=90)
    rows = np.flatnonzero(use)
    rows = rows[unique_join(sid[rows],sid[rows])>=0]
    spec['DR20_APO'] = dict(source_id=sid[rows],RA=ra[rows],DEC=dec[rows],reference_row=rows)
    return spec



def validation_history(hashes, group):
    """Include the recent evidence cohorts alongside the legacy exposure list."""
    prior,legacy = past_evidence(hashes)
    by_id,by_legacy = [prior],[legacy]

    def retain(path,digest):
        hashes[str(path.relative_to(ROOT))] = digest
        with np.load(verified(path,digest),allow_pickle=False) as rows:
            by_legacy.append(rows['ls_id'])
            for key in ('identity','source_id','native_source_id'):
                if key in rows:
                    by_id.append(rows[key])

    v34 = V34_EVIDENCE/'complete.json'
    pilot = PILOT_EVIDENCE/'complete.json'
    for marker in (v34,pilot):
        hashes[str(marker.relative_to(ROOT))] = sha(marker)
    for name,digest in json.loads(v34.read_text())['outputs'].items():
        if name.endswith('.npz'):
            retain(v34.parent/name,digest)
    for key,digest in json.loads(pilot.read_text())['inputs'].items():
        path = Path(key)
        if path.parts[:3] == ('data','v3_5','pilot') and path.name == 'complete.json':
            marker = verified(ROOT/path,digest)
            item = json.loads(marker.read_text())['outputs'].get('prediction.npz')
            if item is not None:
                retain(marker.parent/'prediction.npz',item)
    for result in sorted((OUT.parent/'evidence_replay').glob('*.json')):
        if not result.stem.isdecimal():
            continue
        marker = result.with_name(result.stem+'_roster.json')
        receipt = json.loads(marker.read_text())
        verified(result,receipt['replay_sha256'])
        hashes[str(marker.relative_to(ROOT))] = sha(marker)
        hashes[str(result.relative_to(ROOT))] = receipt['replay_sha256']
        retain(marker.with_suffix('.npz'),receipt['roster_sha256'])
    if group is None:
        for marker in sorted(OUT.parent.glob('validation_group_*/complete.json')):
            hashes[str(marker.relative_to(ROOT))] = sha(marker)
            outputs = json.loads(marker.read_text())['outputs']
            for name in ('gaia_qualification_roster.npz','spectra_reference_roster.npz'):
                if name in outputs:
                    retain(marker.parent/name,outputs[name])
    return np.unique(np.concatenate(by_id)),np.unique(np.concatenate(by_legacy))

def qualification(photo,valid_id):
    gaia = join_gaia(valid_id,GAIA)
    labels = derive_gaia_labels(gaia)
    close = separation(photo['RA'],photo['DEC'],gaia['ra'],gaia['dec'])
    result = labels['astro_quality']&labels['zpt_valid']&labels['finite_parallax']&close
    return gaia,labels,result


def one_score(i):
    t,p,error = _STATE
    row = histogram_parallax(t['qrf_mu_edges'][i],t['qrf_mu_mass'][i],p[i],error[i],order=32)
    prior = histogram_parallax(REFERENCE_PRIOR,np.array([1.]),p[i],error[i],order=32)
    return row['log_score'],row['pit'],prior['log_score']


def group_masks(t):
    r,c = t['observed_r'],t['dereddened_g_r']
    m = {'all':np.ones(len(r),bool),'r18_20':(r>=18)&(r<20),'color05_08':(c>=.5)&(c<.8)}
    m['r18_20_color05_08'] = m['r18_20']&m['color05_08']
    for a,b in ((14,16),(16,18),(18,19),(19,20)):
        for x,y in ((.2,.5),(.5,.8),(.8,1.1),(1.1,1.4)):
            m[f'r{a}_{b}_color{x:g}_{y:g}'] = (r>=a)&(r<b)&(c>=x)&(c<y)
    return m


def stellar_statistics(t,rosters,valid,archived_history,usable_gaia,weak_gaia):
    output = []
    for survey,z in rosters.items():
        j = unique_join(valid,z['source_id'])
        assigned = (valid>0)&(j>=0)
        assigned &= separation(t['RA'],t['DEC'],z['RA'][j],z['DEC'][j])
        rows = np.flatnonzero(assigned)
        if not len(rows):
            continue
        ref = read_targeted(survey,{k:v[j[rows]] for k,v in z.items()})
        d,u,cls = [ref[k] for k in ('distance_kpc','distance_error_kpc','class_label')]
        near = ref['distance_precise']&(d+1.96*u<10.)
        far = ref['distance_precise']&(d-1.96*u>10.)
        for cut in ('primary','strict'):
            for name,g in group_masks(t).items():
                a = t[cut][rows]&g[rows]
                if not a.any():
                    continue
                for low,high in ((0.,2.),(2.,5.),(5.,10.)):
                    dwarfs = a&(cls==0)&ref['distance_precise']&(d>=low)&(d<high)
                    distances = {}
                    for prefix,subset in (('dwarf',dwarfs),('dwarf_weak_gaia',dwarfs&weak_gaia[rows]),
                                          ('dwarf_no_qualified_gaia',dwarfs&~usable_gaia[rows])):
                        scores = distance_summary(10+5*np.log10(t['qrf_quantiles_kpc'][rows[subset]]),
                                                  10+5*np.log10(d[subset]))
                        distances.update({f'{prefix}_{k}':v for k,v in scores.items()})
                    output.append(dict(survey=survey,selection=cut,group=name,shell_lo=low,shell_hi=high,
                        matched=int(a.sum()),outside_recorded_history=int((a&~archived_history[rows]).sum()),
                        dwarfs=int((a&(cls==0)).sum()),giants=int((a&(cls==1)).sum()),
                        near_giant=int((a&(cls==1)&near).sum()),far_giant=int((a&(cls==1)&far).sum()),
                        unresolved_giant=int((a&(cls==1)&~(near|far)).sum()),no_label=int((a&(cls<0)).sum()),
                        **distances))
    return output


def parallax_interval(p,error,usable,sigma,lower_kpc,upper_kpc):
    """Classify interval overlap with a distance shell in parallax space."""
    lower,upper = p-sigma*error,p+sigma*error
    compatible = usable&(upper>0)
    p_min = 1/upper_kpc if np.isfinite(upper_kpc) else 0.
    p_max = 1/lower_kpc if lower_kpc else np.inf
    contained = compatible&(lower>p_min)&(upper<p_max)
    excluded = compatible&((upper<p_min)|(lower>p_max))
    return contained,excluded


def gaia_statistics(t,gaia,labels,usable,old,weights):
    cfg = json.loads(CUTS.read_text())
    p,error = labels['p_corr'],gaia['parallax_error']
    center,_,_ = geometry(p,error,usable,cfg)
    score,pit,prior = np.full((3,len(p)),np.nan)
    eligible_rows = np.flatnonzero(usable)
    global _STATE
    _STATE = t,p,error
    with ProcessPoolExecutor(max_workers=int(os.getenv('SLURM_CPUS_PER_TASK','16')),
                             mp_context=mp.get_context('fork')) as pool:
        if len(eligible_rows):
            score[eligible_rows],pit[eligible_rows],prior[eligible_rows] = np.asarray(
                list(pool.map(one_score,eligible_rows,chunksize=64))).T
    _STATE = None
    archived,_,_,held = archived_labels(t,{})
    sigma = cfg['geometry_sigma_multiplier']
    near,far = parallax_interval(p,error,usable,sigma,0.,10.)
    inconsistent = usable&(p+sigma*error<=0)
    weighted = lambda v,m: float(weights[m]@v[m]/weights[m].sum()) if m.any() else np.nan
    report = []
    for cut in ('primary','strict'):
        for name,group in group_masks(t).items():
            a = t[cut]&group
            if not a.any():
                continue
            use = a&usable
            weak = use&(p<MIN_WEAK_SNR*error)
            weak_new = weak&~old
            totals = dict(selection=cut,group=name,draw=n_count(a),projected_n=float(weights[a].sum()),
                gaia_link=int((a&t['gaia_unique']).sum()),gaia_reference=n_count(use),
                gaia_astrometry_quality=n_count(a&labels['astro_quality']),
                gaia_zero_point_available=n_count(a&labels['zpt_valid']),
                gaia_positive_parallax_inconsistent=n_count(a&inconsistent),
                gaia_outside_recorded_history=n_count(use&~old),weak_gaia=n_count(weak),
                weak_gaia_outside_history=n_count(weak_new),gaia_far_interval=n_count(a&far),gaia_near_interval=n_count(a&near),
                far_fraction_lower=float(weights[a&far].sum()/weights[a].sum()),
                far_fraction_upper=float(1-weights[a&near].sum()/weights[a].sum()),
                weak_signed_score_qrf=weighted(score,weak_new),weak_signed_score_prior=weighted(prior,weak_new),
                gaia_pit_outside_90=weighted((pit<.05)|(pit>.95),use),archived_heldout=n_count(a&held),
                archived_dwarf=n_count(a&(archived==0)),archived_giant=n_count(a&(archived==1)),no_archived_class=n_count(a&(archived<0)))
            for j,(low,high) in enumerate(((0.,2.),(2.,5.),(5.,10.))):
                ref = a&(center>=low)&(center<high)
                b = distance_summary(10+5*np.log10(t['qrf_quantiles_kpc'][ref]),10+5*np.log10(center[ref]))
                inside,outside = parallax_interval(p,error,usable,sigma,low,high)
                report.append(dict(**totals,lower_kpc=low,upper_kpc=high,
                    qrf_mass=float(weights[a]@t['qrf_bins'][a,j]),geom_outside_history=n_count(ref&~old),
                    shell_interval_inside=n_count(a&inside),shell_interval_excluded=n_count(a&outside),
                    shell_fraction_lower=float(weights[a&inside].sum()/weights[a].sum()),
                    shell_fraction_upper=float(1-weights[a&outside].sum()/weights[a].sum()),
                    **{f'geometry_{k}':v for k,v in b.items()}))
    return report


def n_count(mask):
    return int(mask.sum())


def reference_frame(group,plan,hashes):
    if group is None:
        catalog = DELIVERY/'complete.json'
        rec = json.loads(catalog.read_text())
        if sha(FREEZE)!=rec['production_freeze_sha256']:
            raise ValueError('Validation follows the completed production manifest')
        hashes[str(catalog.relative_to(ROOT))] = sha(catalog)
        spatial = load(verified(DELIVERY/'spatial.npz',rec['outputs']['spatial.npz']))
        return OUT,list(chain.from_iterable(plan['groups'])),spatial['ambiguous_gaia_aliases'],rec['totals']
    names = plan['groups'][group]
    aliases,totals = [],Counter()
    digest = sha(FREEZE)
    for name in names:
        rec,ids,_,receipt,_ = read_tile(name,digest)
        aliases.append(ids['source_id'][ids['source_id']>0])
        marker = PRODUCTION/Path(name).stem/'complete.json'
        hashes[str(marker.relative_to(ROOT))] = receipt
        totals.update(rec['counts'])
    sid,n = np.unique(np.concatenate(aliases),return_counts=True)
    return OUT.parent/f'validation_group_{group:02d}',names,sid[n>1],totals


def main():
    require_compute()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group',type=int,help='Evaluate a completed production group separately')
    args = parser.parse_args()
    plan = json.loads(FREEZE.read_text())
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in (FREEZE,Path(__file__),CUTS)}
    destination,names,ambiguous,totals = reference_frame(args.group,plan,hashes)
    specs = acquire_spectral_metadata(hashes)
    targets = np.unique(np.concatenate([x['source_id'] for x in specs.values()]))
    previous, lsids = validation_history(hashes,args.group)
    samples,spectra,counts = [],[],Counter()
    paths = (PRODUCTION/Path(name).stem for name in names)
    workers = min(8, int(os.getenv('SLURM_CPUS_PER_TASK','1')))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Collect in the frozen sweep order to retain roster and score ordering.
        for sample, matched, stats in pool.map(lambda path:draw_tile(path,targets),paths):
            if sample:
                samples.append(sample)
                spectra.extend(matched)
                counts.update(stats)
    if counts['rows']!=totals['primary'] or counts['strict']!=totals['strict']:
        raise ValueError('Validation denominator does not match the production selections')
    photo,spectral = join_parts(samples),join_parts(spectra)
    valid = lambda t: np.where(np.isin(t['source_id'],ambiguous),0,t['source_id'])
    historical = lambda t: np.isin(t['identity'],previous)|np.isin(t['source_id'],previous[previous>0])|np.isin(t['ls_id'],lsids)
    ids = valid(photo)
    prior_match = historical(photo)
    photo['gaia_unique'] = ids>0
    spectral_ids = valid(spectral) if spectral else np.empty(0,'i8')
    joint_ids = np.r_[ids,spectral_ids]
    positions = {k:np.r_[photo[k],spectral.get(k,np.empty(0))] for k in ('RA','DEC')}
    tiles = tile_inventory(joint_ids[joint_ids>0])
    destination.mkdir(parents=True,exist_ok=True)
    freeze = destination/'freeze.json'
    binding = dict(hashes=hashes,gaia_tiles=tiles,rows=counts,spectral_n=len(spectral_ids),
                   reference_prior_mu=REFERENCE_PRIOR.tolist())
    if freeze.exists():
        if json.loads(freeze.read_text())!=binding:
            raise ValueError('Cannot reopen validation with a changed roster')
    else:
        save_json(freeze,binding)
    joined,zpt,qualified = qualification(positions,joint_ids)
    if tiles!=tile_inventory(joint_ids[joint_ids>0]):
        raise ValueError('Gaia tiles changed while accessing reference data')
    usable = qualified[:len(ids)]
    csv_rows(destination/'gaia_joint.csv',gaia_statistics(photo,take_rows(joined,slice(len(ids))),
             take_rows(zpt,slice(len(ids))),usable,prior_match,photo['design_weight']))
    if spectral:
        spec_good = qualified[len(ids):]
        spec_weak = spec_good & (zpt['p_corr'][len(ids):]<MIN_WEAK_SNR*joined['parallax_error'][len(ids):])
        stats = stellar_statistics(spectral,specs,spectral_ids,historical(spectral),spec_good,spec_weak)
        csv_rows(destination/'targeted_spectra_joint.csv',stats)
        atomic_npz(destination/'spectra_reference_roster.npz',
                   **{k:spectral[k] for k in ('ls_id','identity','TYPE')},
                   native_source_id=spectral_ids,qualified_gaia=spec_good,weak_gaia=spec_weak)
    counts.update(gaia_usable=n_count(usable),gaia_outside_recorded_history=n_count(usable&~prior_match),
                  targeted_candidates=len(spectral.get('ls_id',[])))
    atomic_npz(destination/'gaia_qualification_roster.npz',ls_id=photo['ls_id'],TYPE=photo['TYPE'],
         native_source_id=ids,qualified=usable,history=prior_match,design_weight=photo['design_weight'])
    save_json(destination/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),freeze_sha256=sha(freeze),
        all_PSF=True,production_group=args.group,counters=counts,outputs={p.name:sha(p) for p in sorted(destination.iterdir()) if p.name!='complete.json'}))


if __name__=='__main__':
    main()
