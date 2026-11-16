#!/usr/bin/env python3
"""Reference-error-aware distance-interval calibration and candidate stellar census."""
import json
from pathlib import Path

import healpy as hp
import numpy as np
from scipy.optimize import brentq

from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_6_distance_evidence import OUT as EVIDENCE, reference_agreement
from v3_6_vac_distances import read_columns
from v3_8_candidate import decisions, gates, read_cache
from v3_8_direct_figures import catalog, matched_roster

OUT = ROOT/'data/v3_8/reliability'
RULES = ROOT/'configs/v3_7_use_domains.json'
COLS = ('ls_id','catalog_row','TYPE','RA','DEC','qrf_quantiles_kpc','reference_kpc','p_corr',
        'gaia_parallax_error','high_quality','primary','qrf_training','band_state','observed_r',
        'dereddened_g_r','identity','source_id')


def transform(q,scale):
    """Positive, monotone transform of the distance-modulus axis about its median."""
    return q[:,2,None]*(q/q[:,2,None])**scale


def calibration(a):
    r,c,d = (a[k] for k in ('observed_r','dereddened_g_r','qrf_quantiles_kpc'))
    block = hp.ang2pix(4,a['RA'],a['DEC'],nest=True,lonlat=True)
    fit = (block*73+19)%7<4
    # All uses of a Gaia identity must fall in the same geographic split.
    ids,inv = np.unique(a['source_id'],return_inverse=True)
    if (ids<=0).any() or np.any(np.bincount(inv,weights=fit)*np.bincount(inv,weights=~fit)>0):
        raise ValueError('Calibration and test identities overlap')
    a['sky_block'] = block
    a['fit'] = fit
    rows,scaled = [],d.copy()
    applied = np.zeros(len(r),bool)
    for c0,c1 in ((.8,1.1),(1.1,1.4)):
        for w in (2,4):
            m = (r>=16)&(r<18)&(c>=c0)&(c<c1)&(d[:,2]>0)&(d[:,2]<2)&(a['band_state'][:,4]==w)
            train,test = (m&fit,m&~fit)
            row = dict(color_lo=c0,color_hi=c1,w1_state=w,n_train=int(train.sum()),n_test=int(test.sum()),
                       train_blocks=len(np.unique(block[train])),test_blocks=len(np.unique(block[test])))
            if min(row['n_train'],row['n_test'])<500 or min(row['train_blocks'],row['test_blocks'])<5:
                rows.append(dict(**row,status='insufficient_fit_or_test'))
                continue
            def agreement(mask,factor):
                take = {k:v[mask] for k,v in a.items()}
                take['qrf_quantiles_kpc'] = transform(take['qrf_quantiles_kpc'],factor)
                return reference_agreement(take)
            def target(factor):
                return agreement(train,factor)[:,0].mean()-.68
            low,high = .5,3.
            if target(low)*target(high)>0:
                rows.append(dict(**row,status='target_outside_registered_scale'))
                continue
            scale = brentq(target,low,high,xtol=1e-5)
            before,after = agreement(test,1.),agreement(test,scale)
            scoped = transform(d[m],scale)
            if not np.isfinite(scoped).all() or np.any(np.diff(scoped,axis=1)<0):
                raise ValueError('Nonfinite or unordered calibrated interval')
            scaled[m] = scoped
            applied[m] = True
            rows.append(dict(**row,status='evaluated',scale=scale,before_68=before[:,0].mean(),
                after_68=after[:,0].mean(),before_95=before[:,1].mean(),after_95=after[:,1].mean(),
                median_width_before=float(np.median(5*np.log10(d[test,3]/d[test,1]))),
                median_width_after=float(np.median(5*np.log10(scaled[test,3]/scaled[test,1])))))
    return rows,scaled,applied


def newly_recovered(replay):
    fields = ('ls_id','catalog_row','TYPE','rvs_teff','rvs_teff_err','rvs_logg','rvs_logg_err','rvs_feh',
        'rvs_parameter_quality','reference_kpc','high_quality','qrf_training','consensus_class',
        'recorded_history','primary','dwarf_selection_score','nominal_qnear','dereddened_g_r',
        'observed_r','qrf_quantiles_kpc')
    diagnostics,rows = [],[]
    for source in ('SpecDis','SPDist'):
        a = read_columns(read_cache(EVIDENCE,source+'.npz'),fields)
        use = (a['TYPE']=='PSF')&a['high_quality']&~a['qrf_training']
        a = {k:v[use] for k,v in a.items()}
        cand = decisions(a,replay)
        d = a['reference_kpc']
        good = a['rvs_parameter_quality']&np.isfinite(a['rvs_teff'])&np.isfinite(a['rvs_logg'])
        robust = good&(a['rvs_teff']<7000)&(a['rvs_logg']-a['rvs_logg_err']>4.2)
        transition = good&(a['rvs_logg']>=3.5)&(a['rvs_logg']<=4.4)
        mag = read_columns(read_cache(ROOT/'data/v3_6/reference_photometry','photometry.npz'),('catalog_row','ls_id','magnitude'))
        ref_far = (d>=5)&(d<10)
        take = np.flatnonzero(ref_far&(a['primary']|cand))
        idx = np.searchsorted(mag['catalog_row'],a['catalog_row'][take])
        np.testing.assert_array_equal(mag['ls_id'][idx],a['ls_id'][take])
        absr = mag['magnitude'][idx,1]-10-5*np.log10(d[take])
        diagnostics.append(dict(source=np.repeat(source,len(take)),ls_id=a['ls_id'][take],TYPE=a['TYPE'][take],
            primary=a['primary'][take],new=(cand&~a['primary'])[take],reference_Mr=absr,
            gr=a['dereddened_g_r'][take],**{k:a[k][take] for k in ('rvs_teff','rvs_logg','rvs_feh',
                'rvs_logg_err','consensus_class','recorded_history')},robust_dwarf=robust[take],
                transition_region=transition[take]))
        for name,m in (('primary',a['primary']),('newly_recovered',cand&~a['primary']),('rejected_both',~cand)):
            g = m&ref_far
            rows.append(dict(source=source,group=name,n=int(g.sum()),reliable_parameters=int((g&good).sum()),
                robust_dwarf=int((g&robust).sum()),transition_region=int((g&transition).sum()),
                class_dwarf=int((g&(a['consensus_class']==0)).sum()),
                class_giant=int((g&(a['consensus_class']==1)).sum()),
                history=int((g&a['recorded_history']).sum())))
    return rows,diagnostics


def counts():
    cat = catalog()
    ref,_,known,_ = matched_roster(cat)
    r,gr,d = cat['observed_r'],cat['dereddened_g_r'],cat['qrf_quantiles_kpc'][:,2]
    has_wise = np.isin(cat['band_state'][:,4],[2,4])
    rows = []
    for name,g in (
        ('bright_red_near',cat['primary']&(r>=16)&(r<18)&(gr>=.8)&(gr<1.4)&has_wise&(d<2)),
        ('bright_spectroscopic_dwarf_analog',cat['primary']&(r>=16)&(r<18)&(gr<.5)&has_wise&(d>=2)&(d<5))):
        rows.append(dict(domain=name,rows=int(g.sum()),qualified_geometry_or_spec=int((g&known).sum()),
                         without_qualified_match=int((g&~known).sum())))
    return rows


def main():
    require_compute()
    OUT.mkdir(parents=True,exist_ok=True)
    marker = OUT/'complete.json'
    if marker.exists():
        raise FileExistsError(marker)
    a = read_columns(read_cache(EVIDENCE,'Gaia.npz'),COLS)
    use = (a['TYPE']=='PSF')&a['high_quality']&a['primary']&~a['qrf_training']
    fields = {k:v[use] for k,v in a.items()}
    report,q,applied = calibration(fields)
    csv_rows(OUT/'interval_calibration.csv',report)
    atomic_npz(OUT/'calibrated_intervals.npz',ls_id=fields['ls_id'][applied],TYPE=fields['TYPE'][applied],
               sky_block=fields['sky_block'][applied],fit=fields['fit'][applied],distance_quantiles=q[applied])
    cohort,diags = newly_recovered(gates())
    csv_rows(OUT/'recovered_types.csv',cohort)
    atomic_npz(OUT/'recovered_types.npz',**{k:np.concatenate([r[k] for r in diags]) for k in diags[0]})
    csv_rows(OUT/'reference_gap.csv',counts())
    paths = (EVIDENCE/'complete.json',RULES,Path(__file__).resolve())
    save_json(marker,dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in paths},
        excluded_morphology=dict(non_psf=int(np.sum((a['TYPE']!='')&(a['TYPE']!='PSF'))),
                                unverified_type=int(np.sum(a['TYPE']==''))),
        outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p!=marker},
        reference_gap='Qualified-or-available is limited to the searched frozen reference inventory and includes nominal numerical flags.'))


if __name__=='__main__':
    main()
