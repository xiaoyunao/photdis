"""Independent raw-source evidence before native photometry admission."""
import numpy as np
from .class_evidence import desi_class_evidence_v3
from .spectroscopy import desi_quality_reasons
from .gaia import _decode_boolean
from .reference_v1_4 import pair_keys, unique_join


def class_evidence(source):
    rv = {k[4:].upper(): v for k, v in source.items() if k.startswith('rvs_')}
    reason = desi_quality_reasons(rv)
    n = len(reason)
    table = dict(desi_match=np.ones(n, bool), desi_match_ambiguous=np.zeros(n, bool),
                 desi_quality_reason=reason, desi_spectype=rv['RR_SPECTYPE'])
    for name in ['TEFF', 'TEFF_ERR', 'LOGG', 'LOGG_ERR', 'FEH', 'FEH_ERR']:
        table['desi_' + name.lower()] = rv[name]
    evidence = desi_class_evidence_v3(table)
    text = evidence['class_evidence'].astype('U')
    label = np.full(n, -1, 'i1')
    for value, name in enumerate(['conservative_dwarf', 'conservative_giant', 'other_star_or_transition']):
        label[text == name] = value
    return dict(class_label=label, rvs_quality_reason=reason,
                rvs_parameter_quality=evidence['rvs_parameter_quality'],
                redrock_nonstellar_evidence=evidence['redrock_nonstellar_evidence'])


def gaia_input(source):
    ga = {k[5:]: v for k, v in source.items() if k.startswith('gaia_')}
    match = np.asarray(source['source_id']) > 0
    ga['gaia_match'] = match
    for field in ['duplicated_source', 'in_qso_candidates', 'in_galaxy_candidates']:
        decoded = np.zeros(len(match), bool)
        decoded[match] = _decode_boolean(ga[field][match], field)
        ga[field] = decoded
    return ga


def vac_evidence(target, source_id, catalog, kind):
    """Unique exact positive (TARGETID,SOURCE_ID) pairs and frozen v0 quality cuts."""
    index = unique_join(pair_keys(target, source_id), pair_keys(catalog['targetid'], catalog['source_id']))
    index[(np.asarray(source_id) <= 0) | (np.asarray(target) <= 0)] = -1
    matched = index >= 0
    distance = np.full(len(index), np.nan)
    sigma = np.full(len(index), np.nan)
    valid = np.zeros(len(index), bool)
    q = np.flatnonzero(matched)
    v = {k: x[index[q]] for k, x in catalog.items()}
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        if kind == 'specdis':
            d, e = v['dist'].astype(float), v['disterr'].astype(float)
            ok = np.isfinite(d) & np.isfinite(e) & (d > 0) & (e > 0) & (e < d) & (v['binary_flag'] == 1)
            s = 5 / np.log(10) * np.sqrt(np.log1p((e / d)**2))
        elif kind == 'spdist':
            g, m, lo, hi, e = [v[k].astype(float) for k in ['g0', 'mg_50', 'mg_16', 'mg_84', 'e_gmag']]
            ok = (_decode_boolean(v['flag_good'], 'flag_good') & _decode_boolean(v['qflag_input'], 'qflag_input')
                  & np.isfinite(g) & np.isfinite(m) & np.isfinite(lo) & np.isfinite(hi)
                  & np.isfinite(e) & (e >= 0) & (g > 10) & (g < 30)
                  & (m > -8) & (m < 15) & (hi >= lo) & (hi - lo < 4))
            d = 10**((g - m - 10) / 5)
            s = np.sqrt(e**2 + ((hi - lo) / 2)**2 + .167**2)
        else:
            raise ValueError(kind)
    valid[q] = ok
    distance[q[ok]], sigma[q[ok]] = d[ok], s[ok]
    return {kind + '_match': matched, kind + '_valid': valid,
            kind + '_distance': distance, kind + '_sigma_mu': sigma,
            kind + '_row': index}
