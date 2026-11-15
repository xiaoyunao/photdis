#!/usr/bin/env python3
"""Check two distance VACs on the same stars and resolve the far-giant counts."""
import json
import os
from pathlib import Path

import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.metrics import distance_summary
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import DISTANCE_EDGES, read_columns
from v3_6_vac_flow import OUT as FLOW


OUT = ROOT/'data/v3_6/vac_paired'
FIELDS = ('ls_id','source_id','catalog_row','vac_row','TYPE','primary','strict','qrf_training',
          'qrf_quantiles_kpc','qrf_qnear','dwarf_selection_score','nominal_qnear',
          'observed_r','dereddened_g_r','class_label',
          'distance_kpc','sigma_mu','usable_uncertainty','historical_quality')
CUTS = ('supported','primary','strict')


def owners(ls_id):
    ids,i,counts = np.unique(ls_id,return_index=True,return_counts=True)
    return ids[counts==1],i[counts==1]


def pair(spec,sp):
    left,li = owners(spec['catalog_row'])
    right,ri = owners(sp['catalog_row'])
    _,i,j = np.intersect1d(left,right,assume_unique=True,return_indices=True)
    i,j = li[i],ri[j]
    same = ('ls_id','source_id','TYPE','primary','strict','qrf_training','qrf_quantiles_kpc',
            'qrf_qnear','dwarf_selection_score','nominal_qnear','observed_r','dereddened_g_r')
    for field in same:
        if spec[field].dtype.kind=='f':
            np.testing.assert_allclose(spec[field][i],sp[field][j],rtol=0,atol=0,equal_nan=True)
        else:
            np.testing.assert_array_equal(spec[field][i],sp[field][j])
    base = dict(**{k:spec[k][i] for k in (*same,'catalog_row')})
    for name,src,ix in (('specdis',spec,i),('spdist',sp,j)):
        for key in ('vac_row','class_label','distance_kpc','sigma_mu','usable_uncertainty','historical_quality'):
            base[f'{name}_{key}'] = src[key][ix]
    x,y = (base[f'{name}_class_label'] for name in ('specdis','spdist'))
    conflict = (x>=0)&(y>=0)&(x!=y)
    base['class_disputed'] = conflict
    base['paired_class'] = np.where(conflict,-1,np.maximum(x,y))
    return base


def cuts(a):
    return (np.ones(len(a['ls_id']),bool),a['primary'],a['strict'])


def photometric_domains(a):
    color = a['dereddened_g_r']
    blue = (color>=.5)&(color<.8)
    return (('all',np.ones(len(blue),bool)),('blue',blue),('faint_blue',blue&(a['observed_r']>=18)))


def paired_metrics(a):
    d,s = (a[f'{name}_distance_kpc'] for name in ('specdis','spdist'))
    with np.errstate(divide='ignore',invalid='ignore'):
        ref = [10+5*np.log10(x) for x in (d,s)]
        mu = 10+5*np.log10(a['qrf_quantiles_kpc'])
        log_ratio = ref[1]-ref[0]
        ratio = s/d
    finite = np.isfinite(log_ratio)
    qualified = a['specdis_historical_quality']&a['spdist_historical_quality']
    cls = a['paired_class']
    for name,selected in zip(CUTS,cuts(a)):
        for group,domain in photometric_domains(a):
            for label in ('all',-1,0,1,2):
                use = (selected&domain & finite & qualified & ~a['qrf_training']
                       & (True if label=='all' else cls==label))
                for lo,hi in zip(DISTANCE_EDGES[:-1],DISTANCE_EDGES[1:]):
                    for anchor,distance in (('specdis',d),('spdist',s)):
                        keep = use&(distance>=lo)&(distance<hi)
                        # Holding the roster fixed, compare the QRF to each
                        # source separately without treating their errors as independent.
                        out = dict(selection=name,observed=group,class_code=label,bin_source=anchor,
                                   lower_kpc=lo,upper_kpc=hi,n=int(keep.sum()),
                                   both_in_shell=int((keep&(d>=lo)&(d<hi)&(s>=lo)&(s<hi)).sum()))
                        if keep.any():
                            x = log_ratio[keep]
                            out.update(spdist_to_specdis_median_ratio=float(np.median(ratio[keep])),
                                       inter_reference_bias_mu=float(np.median(x)),
                                       inter_reference_nmad_mu=float(1.4826*np.median(abs(x-np.median(x)))))
                            for source,center in zip(('specdis','spdist'),ref):
                                out.update({f'{source}_{key}':v for key,v in
                                            distance_summary(mu[keep],center[keep]).items()})
                        yield out


def far_giant_counts(a,source):
    d = a['distance_kpc']
    lower = d*10**(-np.where(a['usable_uncertainty'],a['sigma_mu'],np.nan)/5)
    giant = a['class_label']==1
    before_train_exclusion = np.ones(len(d),bool)
    for cut,selected in zip(CUTS,cuts(a)):
        for quality,qual in (('legacy_quality',a['historical_quality']),
                             ('finite_error',a['usable_uncertainty'])):
            for history,history_mask in (('all',before_train_exclusion),('outside_qrf_train',~a['qrf_training'])):
                use = selected & qual & history_mask
                for lo,hi in zip(DISTANCE_EDGES[3:-1],DISTANCE_EDGES[4:]):
                    center = use&(d>=lo)&(d<hi)
                    far = center & giant & (d>10)
                    interval = far & (lower>10)
                    for definition,group in (('center_beyond_10',far),('lower_68_bound_beyond_10',interval)):
                        yield dict(source=source,selection=cut,reference_quality=quality,
                            history=history,definition=definition,lower_center_kpc=lo,upper_center_kpc=hi,
                            reference_matches_in_shell=int(center.sum()),giant_matches=int(group.sum()),
                            giant_qrf_median_below_10=int((group&(a['qrf_quantiles_kpc'][:,2]<10)).sum()),
                            giant_qrf_qnear_at_least_half=int((group&(a['qrf_qnear']>=.5)).sum()),
                            qrf_expected_near_giant_count=float(a['qrf_qnear'][group].sum()))


def main():
    require_compute()
    if (OUT/'complete.json').exists():
        raise FileExistsError(OUT/'complete.json')
    inputs = {str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    sources = {}
    for name in ('specdis','spdist'):
        path = FLOW/name/'complete.json'
        record = json.loads(path.read_text())
        inputs[str(path.relative_to(ROOT))] = sha(path)
        src = read_columns(verified(path.parent/'matched.npz',record['outputs']['matched.npz']),FIELDS)
        if not (src['TYPE']=='PSF').all():
            raise ValueError('Pairing requires both source rosters to be native PSF')
        sources[name] = src
    same = pair(sources['specdis'],sources['spdist'])
    OUT.mkdir(parents=True,exist_ok=True)
    atomic_npz(OUT/'pairs.npz',**same)
    csv_rows(OUT/'paired_metrics.csv',paired_metrics(same))
    csv_rows(OUT/'far_giants.csv',[r for name,a in sources.items() for r in far_giant_counts(a,name)])
    save_json(OUT/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        pairs=len(same['ls_id']),primary=int(same['primary'].sum()),strict=int(same['strict'].sum()),
        interpretation=('Paired metrics hold the stars fixed and use both original distance-quality '
                        'rules outside current QRF training. The two spectroscopic distances can '
                        'share measurements. The giant lower endpoint is the available Gaussian '
                        'modulus-error approximation. Source rows and catalog contamination '
                        'percentages require separate interpretation.'),
        outputs={p.name:sha(p) for p in OUT.iterdir() if p.name!='complete.json'}))
    print(len(same['ls_id']),int(same['primary'].sum()),int(same['strict'].sum()),flush=True)


if __name__=='__main__':
    main()
