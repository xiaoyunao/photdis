"""Observable candidate application and finite low-leakage diagnostics."""
from __future__ import annotations
import numpy as np
from .giant_information import BANDS, ERRORS, wilson, class_metrics, colors_and_covariance


def observable_masks(t, cfg):
    finite = np.isfinite(np.column_stack([t[k] for k in BANDS+ERRORS]))
    base = (t['representative'] & ~t['identity_conflict'] & (t['release']==11010)
            & finite[:, [0, 1, 2, 5, 6, 7]].all(axis=1))
    w1 = base & finite[:, [3, 8]].all(axis=1) & t['w1_clean'] & (t['snr_w1']>=cfg['wise_snr_min'])
    w2 = w1 & finite[:, [4, 9]].all(axis=1) & t['w2_clean'] & (t['snr_w2']>=cfg['wise_snr_min'])
    return {'optical': base, 'w1': w1, 'common5': w2,
            'common20': w2 & (t['snr_w1']>=20) & (t['snr_w2']>=20)}


def population_categories(t):
    temperature = t['desi_teff']
    out = np.full(len(temperature), 'unknown', dtype='U32')
    text = t['class_evidence'].astype('U')
    out[text=='other_star_or_transition'] = 'intermediate_or_transition'
    out[(text=='other_star_or_transition') & (temperature>=7000)] = 'hot_qualified_unclassified'
    out[t['redrock_nonstellar_evidence']] = 'raw_nonstellar_unassessed'
    for value, name in ((0, 'dwarf'), (1, 'giant')):
        for lo, hi, tag in ((0, 4200, 'cool'), (4200, 5000, 'K'), (5000, 7000, 'warm')):
            out[(t['label']==value) & (temperature>=lo) & (temperature<hi)] = tag+'_'+name
    return out


def strata(t, indices, w2, outside):
    n = len(indices)
    r, temp = t['r'][indices], t['desi_teff'][indices]
    result = {'all': np.ones(n, bool), 'inside': ~outside, 'outside': outside,
              'w2_pass': w2[indices], 'w2_fail': ~w2[indices]}
    for lo, hi in ((12,16),(16,18),(18,20),(20,22),(22,25)):
        result[f'r{lo}_{hi}'] = (r>=lo) & (r<hi)
        result[f'inside_r{lo}_{hi}'] = result[f'r{lo}_{hi}'] & ~outside
    for lo, hi in ((0,4200),(4200,4500),(4500,5000),(5000,7000),(7000,100000)):
        result[f'teff{lo}_{hi}'] = (temp>=lo) & (temp<hi)
    return result


def selection_summary(t, rows, score, outside, w2, threshold, reject_outside=False):
    chosen = (score>=threshold) & (~outside if reject_outside else True)
    categories = population_categories(t)[rows]
    result = {}
    for name, mask in strata(t, rows, w2, outside).items():
        counts = {c: wilson(int((chosen & mask & (categories==c)).sum()), int((mask & (categories==c)).sum()))
                  for c in np.unique(categories)}
        kgiant = categories=='K_giant'; dwarf = t['label'][rows]==0
        counts['K_giant_total'] = wilson(int((mask & chosen & kgiant).sum()), int((mask & kgiant).sum()))
        counts['all_dwarf_total'] = wilson(int((mask & chosen & dwarf).sum()), int((mask & dwarf).sum()))
        counts['reference_total'] = wilson(int((mask & chosen).sum()), int(mask.sum()))
        result[name] = counts
    selected_giants = rows[chosen & (t['label'][rows]==1)]
    pop = {}
    for col in ('desi_teff', 'desi_logg', 'feh_corrected', 'r'):
        valid = np.isfinite(t[col][selected_giants])
        if col=='feh_corrected':
            valid &= t['abundance_calibration_valid'][selected_giants]
        values = t[col][selected_giants][valid]
        pop[col] = {'n': len(values), 'q16_50_84': np.quantile(values,[.16,.5,.84]).tolist() if len(values) else None}
    return {'strata': result, 'selected_reliable_giant_parameters': pop}


def threshold_at_fpr(dwarf_scores, target):
    """Lowest threshold excluding ties that admits at most floor(alpha*N)."""
    values = np.sort(np.asarray(dwarf_scores, float))[::-1]
    if not len(values):
        return {'threshold': None, 'n_calibration_dwarfs': 0, 'resolution_ok': False}
    k = int(np.floor(target*len(values)))
    threshold = float(max(0., np.nextafter(values[k], np.inf)))
    count = int((values>=threshold).sum())
    return {'threshold': threshold, 'n_calibration_dwarfs': len(values),
            'allowed_false_positives': k, 'calibration_false_positives': count,
            'calibration_fpr': wilson(count, len(values)),
            'resolution_ok': bool(len(values)*target>=1),
            'target_fpr': target}


def feasibility(t, rows, score, outside, targets):
    effective = np.where(outside, -1., score)
    label = t['label'][rows]
    k = (t['desi_teff'][rows]>=4200) & (t['desi_teff'][rows]<5000)
    answer = {}
    for alpha in targets:
        rule = threshold_at_fpr(effective[label==0], alpha)
        if rule['threshold'] is not None:
            selected = effective>=rule['threshold']
            rule['K_giant_recall'] = wilson(int((selected & (label==1) & k).sum()), int(((label==1)&k).sum()))
        answer[str(alpha)] = rule
    return answer


def mechanism_features(t, bands, errors, dimension):
    x = np.log10(t['desi_teff']/5000)/.1
    spec = np.column_stack([x, x*x, t['desi_logg']])
    color, covariance = colors_and_covariance(bands, errors, dimension) if dimension else (np.empty((len(x),0)), np.empty((len(x),0,0)))
    features = np.column_stack([spec, color])
    cov = np.zeros((len(x), 3+dimension, 3+dimension))
    cov[:, 3:, 3:] = covariance
    return features, cov
