#!/usr/bin/env python3
"""Bind source-specific distance evidence to common DR11 prediction and class fields."""
import json
import os
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from photdis.v3.metrics import distance_summary
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_3_sed import BASE, MANIFESTS, verified
from v3_5_dr11_validate import OUT as VALIDATION, join_parts, validation_history
from v3_6_blue_replay import OUT as BLUE_REPLAY
from v3_6_external_reference import OUT as EXTERNAL
from v3_6_gaia_census import OUT as GAIA, STATES
from v3_6_reference_index import OUT as INDEX
from v3_6_vac_distances import ARCHIVE, DISTANCE_EDGES, read_columns
from v3_6_vac_flow import OUT as VAC


OUT = ROOT/'data/v3_6/distance_evidence'
COMMON = ('ls_id','identity','source_id','catalog_row','TYPE','RA','DEC','observed_r','dereddened_g_r',
          'band_state','nominal_qnear','qrf_qnear','dwarf_selection_score','qrf_quantiles_kpc',
          'qrf_bins','primary','strict')
CLASSES = ('dwarf','giant','other')
SHELLS = ('0–2','2–5','5–10','10–20','20–50','>50')
COVERAGE_GATES = ('supported','primary','strict')
COVERAGE_PHOTOMETRY = ('all','faint_blue','bright_red')
COVERAGE_SCOPES = ('all_native_psf','at_least_one_qualified_spectroscopic_distance')
RED_DOMAIN = ROOT/'configs/v3_4_near_release_domain.json'


def class_index(paths,size,inputs):
    marker = INDEX/'complete.json'
    ref = json.loads(marker.read_text())
    inputs[str(marker.relative_to(ROOT))] = sha(marker)
    first = np.full(size,-1,'i1')
    conflict = np.zeros(size,bool)
    archive_roles = {}
    for name,path in paths.items():
        x = load(verified(INDEX/f'{name}.npz',ref['outputs'][f'{name}.npz']))
        cols = ('class_label','class_eligible','split') if name=='baseline_v1_4' else ('class_label','class_eligible','role')
        t = read_columns(verified(path,ref['inputs'][str(path.relative_to(ROOT))]),cols)
        ok = (x['catalog_row']>=0)&t['class_eligible']&np.isin(t['class_label'],np.arange(3))
        row = x['catalog_row'][ok]
        label = t['class_label'][ok]
        # Compare all repeated classifications before constructing consensus.
        for k in range(3):
            j = np.unique(row[label==k])
            changed = (first[j]>=0)&(first[j]!=k)
            conflict[j[changed]] = True
            first[j[first[j]<0]] = k
        mapped = x['catalog_row']>=0
        label_key = 'split' if name=='baseline_v1_4' else 'role'
        archive_roles[name] = (x['catalog_row'][mapped],t[label_key][mapped])
    first[conflict] = -1
    return first,conflict,archive_roles


def desi_rvs(inputs):
    # The class inventory needs its own roster because a usable DESI spectrum
    # need not have a usable SpecDis or SPDist distance.
    marker = INDEX/'complete.json'
    index = json.loads(marker.read_text())
    filename = 'baseline_v1_4.npz'
    attached = load(verified(INDEX/filename,index['outputs'][filename]))
    archive_path = ARCHIVE/'references.npz'
    archive_key = str(archive_path.relative_to(ROOT))
    archive = read_columns(verified(archive_path,index['inputs'][archive_key]),
        ('ls_id','source_id','identity','source_row','targetid','class_label','class_eligible',
         'split','historical_identity','rvs_teff','rvs_teff_err','rvs_logg','rvs_logg_err',
         'rvs_feh','rvs_feh_err','rvs_parameter_quality','specdis_row','spdist_row'))
    keep = (attached['catalog_row']>=0)&(archive['source_row']>=0)&(archive['targetid']>0)
    supported = attached['catalog_row'][keep]
    catalog = read_columns(verified(INDEX/'catalog_rows.npz',index['outputs']['catalog_rows.npz']),
                           ('ls_id','source_id'))
    archive['archive_ls_id'] = archive.pop('ls_id')
    archive['archive_source_id'] = archive.pop('source_id')
    archive['archive_identity'] = archive.pop('identity')
    inputs[archive_key] = index['inputs'][archive_key]
    native_id = catalog['ls_id'][supported]
    _,index_back,multiplicity = np.unique(native_id,return_inverse=True,return_counts=True)
    return dict(**{key:value[keep] for key,value in archive.items()},catalog_row=supported,
                archive_row=np.flatnonzero(keep),match_status=attached['match_status'][keep],
                ls_id=native_id,source_id=catalog['source_id'][supported],
                native_identity_repeated=multiplicity[index_back]>1,TYPE=np.full(len(supported),'PSF'))



def source(path,inputs):
    receipt = json.loads(path.read_text())
    inputs[str(path.relative_to(ROOT))] = sha(path)
    return receipt


def coverage_counts(z,has_spec,domain):
    n = len(z['ls_id'])
    r,c = z['observed_r'],z['dereddened_g_r']
    lo,hi = domain['observed_r']
    a,b = domain['dereddened_g_r']
    bright_red = ((r>=lo)&(r<hi)&(c>=a)&(c<b)&np.isin(z['band_state'][:,4],domain['w1_states']))
    faint_blue = (r>=18)&(r<20)&(c>=.5)&(c<.8)
    all_rows = np.ones(n,bool)
    state = z['gaia_state']
    hard = np.searchsorted(DISTANCE_EDGES,z['qrf_quantiles_kpc'][:,2],side='right')-1
    shape = (len(STATES),len(SHELLS))
    cell = np.ravel_multi_index((state,hard),shape)
    qrf = z['qrf_bins']
    result = np.empty((len(COVERAGE_SCOPES),len(COVERAGE_GATES),len(COVERAGE_PHOTOMETRY),*shape,2))
    for h,scope in enumerate((all_rows,has_spec)):
        for i,gate in enumerate((all_rows,z['primary'],z['strict'])):
            for j,photo in enumerate((all_rows,faint_blue,bright_red)):
                use = scope&gate&photo
                binary = np.bincount(cell[use],minlength=np.prod(shape)).reshape(shape)
                weighted = np.stack([np.bincount(state[use],weights=qrf[use,k],minlength=len(STATES))
                                     for k in range(len(SHELLS))],axis=1)
                np.testing.assert_allclose(weighted.sum(axis=1),binary.sum(axis=1),rtol=0,atol=1e-7)
                result[h,i,j] = np.stack((binary,weighted),axis=-1)
    return result


def geometry_rows(inputs,has_spec):
    r = source(GAIA/'complete.json',inputs)
    domain, = json.loads(RED_DOMAIN.read_text())['domains']
    inputs[str(RED_DOMAIN.relative_to(ROOT))] = sha(RED_DOMAIN)
    chunks = []
    state = np.full(len(has_spec),-1,'i1')
    coverage = np.zeros((len(COVERAGE_SCOPES),len(COVERAGE_GATES),len(COVERAGE_PHOTOMETRY),
                         len(STATES),len(SHELLS),2))
    for name,digest in r['child_receipts'].items():
        marker = verified(GAIA/name,digest)
        child = json.loads(marker.read_text())
        path = marker.with_suffix('.npz')
        with np.load(verified(path,child['outputs'][path.name]),allow_pickle=False) as z:
            keys = {'TYPE','gaia_state','ls_id','catalog_row','observed_r','dereddened_g_r',
                    'band_state','qrf_quantiles_kpc','qrf_bins','primary','strict'}
            p = {k:z[k] for k in keys}
            ix = p['catalog_row']
            if np.any((ix<0)|(ix>=len(state))) or np.any(state[ix]>=0) or not (p['TYPE']=='PSF').all():
                raise ValueError('Gaia coverage requires unique native catalog rows and verified PSF')
            state[ix] = p['gaia_state']
            coverage += coverage_counts(p,has_spec[ix],domain)
            mask = p['gaia_state']==STATES.index('geometry_reference')
            if mask.any():
                for key in (*COMMON,'reference_kpc','reference_lo_kpc','reference_hi_kpc','p_corr','gaia_parallax_error'):
                    if key not in p:
                        p[key] = z[key]
                chunks.append({k:v[mask] for k,v in p.items()})
    a = join_parts(chunks)
    a['high_quality'] = np.ones(len(a['ls_id']),bool)
    if np.any(state<0) or coverage[0,0,0,:,:,0].sum()!=len(has_spec) or coverage[1,0,0,:,:,0].sum()!=has_spec.sum():
        raise ValueError('Gaia and spectroscopic coverage do not reconcile with the parent frame')
    return a,state,coverage


def vac_rows(kind,inputs):
    p = VAC/kind
    r = source(p/'complete.json',inputs)
    keys = ('class_label','distance_kpc','sigma_mu','historical_quality','usable_uncertainty','vac_row',
            'rvs_logg','rvs_teff','rvs_feh','rvs_logg_err','rvs_teff_err','rvs_feh_err',
            'rvs_parameter_quality','historical_identity','split')
    a = read_columns(verified(p/'matched.npz',r['outputs']['matched.npz']),(*COMMON,*keys))
    a['reference_kpc'] = a['distance_kpc']
    width = np.where(a['usable_uncertainty'],a['sigma_mu'],np.nan)
    a['reference_lo_kpc'] = a['reference_kpc']*10**(-width/5)
    a['reference_hi_kpc'] = a['reference_kpc']*10**(width/5)
    a['high_quality'] = a['historical_quality']
    return a


def external_parameters(name, a):
    # The archived channels only needed a hard gravity label and metallicity.
    # Reuse their stellar quality cuts without tying temperature or gravity
    # availability to the iron flag or the distance qualification.
    if name=='DR17_astroNN':
        teff,logg,err,teff_err,valid = (a[f'raw_apogee_{k}'] for k in
                                      ('teff','logg','logg_err','teff_err','quality'))
        lo = hi = err
    elif name=='DR19_Astra':
        teff,logg,err,snr,flag = (a[f'raw_stellar_{k}'] for k in ('teff','logg','e_logg','snr','flag_bad'))
        lo = hi = err
        teff_err = np.full(len(teff),np.nan)
        valid = ~flag.astype(bool)&np.isfinite(snr)&(snr>=50)
    elif name in ('DR20_APO','DR20_MINESweeper'):
        teff,logg,lo,hi,flag = (a[f'raw_{k}'] for k in ('teff','logg','logg_lerr','logg_uerr','flag'))
        teff_err = np.full(len(teff),np.nan)
        valid = flag==0
    else:
        raise ValueError(f'Unrecognized spectroscopic source: {name}')
    valid = (valid & np.isfinite(teff) & (teff>0) & np.isfinite(logg) & (logg>-9000)
             & np.isfinite(lo) & np.isfinite(hi) & (lo>0) & (hi>0))
    if name!='DR17_astroNN':
        valid &= teff<7000
    return dict(teff=teff,teff_error=teff_err,teff_valid=valid,
                logg=logg,logg_error_lo=lo,logg_error_hi=hi,logg_valid=valid)


def external_rows(inputs):
    r = source(EXTERNAL/'complete.json',inputs)
    for f,h in r['outputs'].items():
        if not f.endswith('_matched.npz'):
            continue
        name = f.removesuffix('_matched.npz')
        keys = ('class_label','distance_kpc','distance_error_kpc','distance_available','distance_precise',
                'historical_evidence','inventory_row','feh','feh_error','feh_valid')
        with np.load(verified(EXTERNAL/f,h),allow_pickle=False) as payload:
            a = {k:payload[k] for k in (*COMMON,*keys)}
            a.update(external_parameters(name,payload))
            if name.startswith('DR20_'):
                a['distance_error_lo_kpc'] = payload['raw_dist_lerr']
                a['distance_error_hi_kpc'] = payload['raw_dist_uerr']
        d,s = a['distance_kpc'],a['distance_error_kpc']
        a['reference_kpc'] = d
        a['reference_lo_kpc'] = np.maximum(d-a.get('distance_error_lo_kpc',s),0.)
        a['reference_hi_kpc'] = d+a.get('distance_error_hi_kpc',s)
        a['high_quality'] = a['distance_precise']
        middle = a['logg_valid'] & (a['logg']>=3.2) & (a['logg']<=4.2)
        a['class_label_historical'] = a['class_label'].copy()
        a['class_label'][middle & (a['class_label']<0)] = 2
        yield name,a


def attach(a,first,conflict,roles,training,history):
    row = a['catalog_row']
    if not (a['TYPE']=='PSF').all() or (row<0).any():
        raise ValueError('Distance reconciliation requires native PSFs and checked ownership')
    own = a.get('class_label',np.full(len(row),-1,'i1'))
    label = first[row]
    valid = np.isin(own,np.arange(3))
    disagree = conflict[row]|(valid&(label>=0)&(own!=label))
    a['source_class'] = own
    a['consensus_class'] = np.where(disagree,-1,np.where(valid,own,label)).astype('i1')
    a['class_disagreement'] = disagree
    for k,identities in training.items():
        a[k] = np.isin(a['identity'],identities)|np.isin(a['source_id'],identities)
    identities,legacy = history
    a['recorded_history'] = (np.isin(a['identity'],identities)|np.isin(a['source_id'],identities)
                             |np.isin(a['ls_id'],legacy)|a.get('historical_evidence',False)
                             |a.get('historical_identity',False))
    for name,(pointer,key) in roles.items():
        ix = unique_join(row,pointer)
        have = ix>=0
        role = np.full(len(row),-1,dtype='i4') if key.dtype.kind in 'iu' else np.full(len(row),'',dtype=key.dtype)
        role[have] = key[ix[have]]
        a[f'{name}_role'] = role


def reference_agreement(a):
    """Reference-error probability inside the stored interval, with each source's error law."""
    q = a['qrf_quantiles_kpc']
    d = a['reference_kpc']
    result = []
    for lo,hi in ((1,3),(0,4)):
        if 'p_corr' in a:
            p,err = a['p_corr'],a['gaia_parallax_error']
            of_reference = (ndtr((1/q[:,lo]-p)/err)-ndtr((1/q[:,hi]-p)/err))/ndtr(p/err)
        elif 'sigma_mu' in a:
            e = a['sigma_mu']
            with np.errstate(divide='ignore',invalid='ignore'):
                of_reference = ndtr(5*np.log10(q[:,hi]/d)/e)-ndtr(5*np.log10(q[:,lo]/d)/e)
            of_reference[e<=0] = np.nan
        else:
            e = a['distance_error_kpc']
            with np.errstate(divide='ignore',invalid='ignore'):
                of_reference = (ndtr((q[:,hi]-d)/e)-ndtr((q[:,lo]-d)/e))/ndtr(d/e)
            of_reference[e<=0] = np.nan
        result.append(of_reference)
    return np.stack(result,axis=1)


def summaries(a,name):
    distance = a['reference_kpc']
    pred = a['qrf_quantiles_kpc']
    with np.errstate(divide='ignore',invalid='ignore'):
        ref = 10+5*np.log10(distance)
        mu = 10+5*np.log10(pred)
    valid = np.isfinite(ref)
    uncertainty = a['reference_error_agreement']
    r,c = a['observed_r'],a['dereddened_g_r']
    all_rows = np.ones(len(distance),bool)
    labels = a['consensus_class']
    conditions = [('all',all_rows),('blue',(c>=.5)&(c<.8)),('faint_blue',(r>=18)&(c>=.5)&(c<.8))]
    summary,migration = [],[]
    for cut,gate in (('supported',all_rows),('primary',a['primary']),('strict',a['strict'])):
        for group,cl in [('all',all_rows),*[(key,labels==i) for i,key in enumerate(CLASSES)],('unknown',labels<0)]:
            selected = gate&cl
            for case,qual in (('available',all_rows),('high_quality',a['high_quality'])):
                for label,history in (('all',all_rows),('outside_qrf_train',~a['qrf_training']),
                                      ('outside_recorded_history',~a['recorded_history']&~a['qrf_training'])):
                    scope = selected&qual&history
                    for observed,valid_obs in conditions:
                        for j,(lo,hi) in enumerate(zip(DISTANCE_EDGES[:-1],DISTANCE_EDGES[1:])):
                            keep = scope&valid_obs&valid&(distance>=lo)&(distance<hi)
                            agreement = uncertainty[keep]
                            means = [float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else None
                                     for x in agreement.T]
                            summary.append(dict(source=name,selection=cut,reference_class=group,quality=case,
                                history=label,observed=observed,reference_lo_kpc=lo,reference_hi_kpc=hi,
                                row_denominator=int((scope&valid_obs).sum()),
                                qrf_training_n=int((keep&a['qrf_training']).sum()),
                                qrf_d50_gt10_n=int((keep&(pred[:,2]>10.)).sum()),
                                qrf_qnear_lt_half_n=int((keep&(a['qrf_qnear']<.5)).sum()),
                                ref_error_agreement68=means[0],ref_error_agreement95=means[1],
                                error_defined_n=int(np.isfinite(agreement).all(axis=1).sum()),
                                **distance_summary(mu[keep],ref[keep])))
                            if case=='high_quality' and label=='all' and observed=='all':
                                for k,(plo,phi) in enumerate(zip(DISTANCE_EDGES[:-1],DISTANCE_EDGES[1:])):
                                    migration.append(dict(source=name,selection=cut,reference_class=group,
                                        reference_shell=SHELLS[j],predicted_shell=SHELLS[k],
                                        hard_count=int((keep&(pred[:,2]>=plo)&(pred[:,2]<phi)).sum()),
                                        qrf_weighted_count=float(a['qrf_bins'][keep,k].sum())))
    return summary,migration


def blue_spectrum_rows(blue,a,source):
    if not (blue['TYPE']=='PSF').all() or not (a['TYPE']=='PSF').all():
        raise ValueError('Blue spectroscopic comparisons require native PSF')
    j = unique_join(blue['ls_id'],a['ls_id'])
    elsewhere = np.isin(blue['source_id'],a['source_id'])
    occupied = np.isin(blue['ls_id'],a['ls_id'])
    n = len(a['ls_id'])
    empty = np.full(n,np.nan)
    labels = a.get('source_class',a['class_label'])
    valid_class = a.get('class_eligible',labels>=0)
    for i,k in enumerate(j):
        r = dict(source=source,cohort=blue['sample'][i],ls_id=blue['ls_id'][i],
                 gaia_source_id=blue['source_id'][i],strict=bool(blue['strict'][i]),
                 gaia_distance_kpc=blue['reference_kpc'][i])
        if k<0:
            status = ('repeated_legacy_candidate' if occupied[i] else
                      'gaia_key_only' if elsewhere[i] else 'unmatched')
            yield dict(**r,match_status=status)
            continue
        if a['source_id'][k]!=blue['source_id'][i] or blue['source_id'][i]<=0:
            yield dict(**r,match_status='conflicting_gaia_key')
            continue
        r.update(match_status='both_keys',reference_class=int(labels[k]) if valid_class[k] else -1,
                 spectroscopic_distance_kpc=a.get('reference_kpc',empty)[k],
                 spectroscopic_distance_low_kpc=a.get('reference_lo_kpc',empty)[k],
                 spectroscopic_distance_high_kpc=a.get('reference_hi_kpc',empty)[k])
        for key in ('logg','teff','feh'):
            if key in a:
                v,good = a[key][k],a[f'{key}_valid'][k]
                symmetric = a.get(f'{key}_error',empty)
                lo,hi = (a.get(f'{key}_error_{side}',symmetric)[k] for side in ('lo','hi'))
            else:
                v = a.get(f'rvs_{key}',empty)[k]
                quality = a['rvs_parameter_quality'][k] if 'rvs_parameter_quality' in a else False
                good = quality and np.isfinite(v)
                lo = hi = a.get(f'rvs_{key}_err',empty)[k]
            r.update({key:float(v) if good else None,f'{key}_qualified':bool(good),
                      f'{key}_error_lo':float(lo) if good and np.isfinite(lo) else None,
                      f'{key}_error_hi':float(hi) if good and np.isfinite(hi) else None})
        yield r



def main():
    require_compute()
    inputs = {str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    size = source(INDEX/'complete.json',inputs)['supported_rows']
    first,conflict,roles = class_index({'baseline_v1_4':ARCHIVE/'references.npz','v3_0':BASE/'references.npz'},size,inputs)
    controls = source(MANIFESTS/'controls/complete.json',inputs)
    q = read_columns(verified(BASE/'controls/training.npz',controls['outputs']['training.npz']),('identity','class_label'))
    cloud = source(MANIFESTS/'sed_cloud.json',inputs)
    sed = read_columns(verified(BASE/'sed_cloud.npz',cloud['sha256']),('identity',))['identity']
    np.testing.assert_array_equal(sed,q['identity'])
    training = dict(qrf_training=q['identity'],nominal_sed_training=sed,
                    selector_training=q['identity'][q['class_label']>=0])
    previous,legacy = validation_history(inputs,None)
    original = source(VALIDATION/'complete.json',inputs)
    for kind in ('gaia_qualification_roster.npz','spectra_reference_roster.npz'):
        path = verified(VALIDATION/kind,original['outputs'][kind])
        with np.load(path,allow_pickle=False) as z:
            legacy = np.union1d(legacy,z['ls_id'])
            if 'native_source_id' in z:
                previous = np.union1d(previous,z['native_source_id'])
    history = previous,legacy
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'complete.json').exists():
        raise FileExistsError(OUT/'complete.json')
    cases = [('SpecDis',vac_rows('specdis',inputs)),('SPDist',vac_rows('spdist',inputs)),*external_rows(inputs)]
    has_spec = np.zeros(size,bool)
    for _,a in cases:
        owner = a['catalog_row']
        if np.any((owner<0)|(owner>=size)) or not (a['TYPE']=='PSF').all():
            raise ValueError('Spectroscopic coverage requires verified native catalog ownership')
        qualified = a['high_quality']&np.isfinite(a['reference_kpc'])&(a['reference_kpc']>0)
        has_spec[owner[qualified]] = True
    geometry,gaia_state,coverage = geometry_rows(inputs,has_spec)
    cases.insert(0,('Gaia',geometry))
    coverage_rows = []
    for h,i,j,k,l in np.ndindex(coverage.shape[:-1]):
        count,mass = coverage[h,i,j,k,l]
        coverage_rows.append(dict(scope=COVERAGE_SCOPES[h],selection=COVERAGE_GATES[i],
            photometry=COVERAGE_PHOTOMETRY[j],gaia_state=STATES[k],qrf_shell=SHELLS[l],
            median_count=int(count),posterior_count=mass))
    csv_rows(OUT/'photometric_reference_coverage.csv',coverage_rows)
    all_rows,all_migration = [],[]
    for name,a in cases:
        attach(a,first,conflict,roles,training,history)
        a['gaia_state'] = gaia_state[a['catalog_row']]
        a['reference_error_agreement'] = reference_agreement(a)
        atomic_npz(OUT/f'{name}.npz',**a)
        summary,migration = summaries(a,name)
        all_rows.extend(summary)
        all_migration.extend(migration)
        print(name,len(a['ls_id']),int(a['qrf_training'].sum()),int(a['class_disagreement'].sum()),flush=True)
    csv_rows(OUT/'distance_cells.csv',all_rows)
    csv_rows(OUT/'migration.csv',all_migration)
    rvs = desi_rvs(inputs)
    atomic_npz(OUT/'desi_mws_rvs.npz',**rvs)
    rvs_counts = dict(rows=len(rvs['ls_id']),identities=len(np.unique(rvs['ls_id'])),
                      class_eligible=int(rvs['class_eligible'].sum()),
                      parameter_quality=int(rvs['rvs_parameter_quality'].sum()))
    print('DESI MWS/RVS',rvs_counts,flush=True)
    blue_marker = BLUE_REPLAY/'complete.json'
    blue_record = source(blue_marker,inputs)
    blue = read_columns(verified(BLUE_REPLAY/'blue_objects.npz',blue_record['outputs']['blue_objects.npz']),
                        ('TYPE','sample','ls_id','source_id','reference_kpc','strict'))
    spectra = [('DESI_MWS_RVS',rvs),*[(name,a) for name,a in cases if name!='Gaia']]
    csv_rows(OUT/'blue_spectroscopy.csv',(r for name,a in spectra for r in blue_spectrum_rows(blue,a,name)))
    save_json(OUT/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        rows={name:len(a['ls_id']) for name,a in cases},desi_mws_rvs=rvs_counts,
        interpretation=('Per-source native PSF projections. Reference sets overlap. QRF training '
                        'matches are fit consistency. Error-integrated agreement uses Gaussian '
                        'parallax, modulus, or distance reference errors as available. DR20 intervals '
                        'retain their published asymmetric errors, with the larger error used for '
                        'the Gaussian agreement diagnostic. Qualified intermediate gravities are '
                        'retained as other stars, with historical labels preserved. The DESI class '
                        'roster does not require a VAC distance. Photometric reference coverage '
                        'counts each native object once across the union of source-qualified '
                        'spectroscopic distances. It includes historical objects and preserves '
                        'each Gaia evidence state.'),
        outputs={p.name:sha(p) for p in OUT.iterdir() if p.name!='complete.json'}))


if __name__=='__main__':
    main()
