"""Check v3.9 native ownership and full histogram semantics independently of inference."""
import json

import numpy as np

from v2_1_near import ROOT, load
from v3_8_candidate import read_cache

Q = np.array([.025,.16,.5,.84,.975])
SHELLS = np.array([0.,2.,5.,10.,20.,50.,np.inf])
INDEX = ROOT/'data/v3_6/reference_index'


def verify_pdf(edges, mass, quantiles, bins):
    if (edges.shape!=(len(mass),mass.shape[1]+1) or mass.shape[1]==0
            or quantiles.shape!=(len(mass),len(Q)) or bins.shape!=(len(mass),len(SHELLS)-1)):
        raise ValueError('Inconsistent histogram, shell or quantile shape')
    max_quantile_error = 0.
    max_shell_error = 0.
    for start in range(0,len(mass),1024):
        j = slice(start,start+1024)
        e,w,d,s = edges[j],mass[j],quantiles[j],bins[j]
        width = np.diff(e,axis=1)
        if not np.isfinite(e).all() or not (width>0).all():
            raise ValueError('Nonfinite or unordered distance modulus edges')
        if not np.isfinite(w).all() or (w<0).any() or not np.allclose(w.sum(axis=1),1.,atol=1e-11,rtol=0):
            raise ValueError('Distance histogram mass is negative or unnormalized')
        if not np.isfinite(d).all() or (d<=0).any() or (np.diff(d,axis=1)<0).any():
            raise ValueError('Distance quantiles are nonpositive or unordered')
        # Independently recover the reported quantile and shell CDF values
        # from uniform density within each distance-modulus interval.
        a = 10+5*np.log10(d)
        q_cdf = np.column_stack([np.sum(w*np.clip((u[:,None]-e[:,:-1])/width,0,1),axis=1) for u in a.T])
        cdf = np.column_stack([np.sum(w*np.clip((u-e[:,:-1])/width,0,1),axis=1)
                               for u in 10+5*np.log10(SHELLS[1:])])
        shells = np.diff(cdf,axis=1,prepend=0.)
        if not np.isfinite(s).all() or (s < -1e-13).any():
            raise ValueError('Nonfinite or negative distance-shell mass')
        max_quantile_error = max(max_quantile_error,float(np.max(abs(q_cdf-Q))))
        max_shell_error = max(max_shell_error,float(np.max(abs(shells-s))))
    if max_quantile_error>2e-6 or max_shell_error>1e-11:
        raise ValueError(f'Distance PDF differs from saved summaries: {max_quantile_error}, {max_shell_error}')
    return max_quantile_error,max_shell_error


def parent_part(p, state, raw_state, release):
    rows,lsid,level = p['parent_row'],p['ls_id'],p['row_state']
    if ((rows<0)|(rows>=len(raw_state))).any() or not np.array_equal(level,raw_state[rows]):
        raise ValueError('Export row does not belong to its native sweep census')
    if not np.all((lsid>>40)==release) or not np.all(p['TYPE']=='PSF'):
        raise ValueError('Incorrect packed release or unverified morphology')
    if not np.array_equal(state['ls_id'],lsid) or not np.array_equal(state['TYPE'],p['TYPE']):
        raise ValueError('Prediction state has changed native identity or morphology')
    code = state['state']
    low = state['noise_control_photo_row']
    weights = state['noise_control_weight']
    if np.any(~np.isin(level,[5,6,7])) or np.any(~np.isin(code,[0,1,2,3])):
        raise ValueError('Unknown parent or prediction state')
    if np.any((low<0)|(low>=len(code))) or len(low)!=len(weights) or not np.all(np.isfinite(weights)&(weights>=1)):
        raise ValueError('Invalid noise-control design rows or weights')
    if not np.array_equal(np.sort(low),np.flatnonzero((level==6)&(code>=2))):
        raise ValueError('Noise-control weights do not map to evaluated parent rows')
    if not (np.array_equal(code==0,level==5) and np.all(code[level==7]>=2) and np.all(code[low]>=2)):
        raise ValueError('Prediction state contradicts the optical-quality census')


def unique_catalog(ids, side, *, historical_ids=None):
    if not len(ids) or len(np.unique(ids))!=len(ids):
        raise ValueError(f'Empty or repeated packed Legacy identity in the {side} catalog')
    if historical_ids is not None and np.intersect1d(ids,historical_ids).size:
        raise ValueError('v3.9 southern domains duplicate the frozen southern parent')


def all_catalog_ids(root,side):
    count = len(json.loads((root/side/'inventory.json').read_text())['groups'])
    base = root/'validation'/side
    items = []
    certified = checked = 0
    worst_q = worst_shell = 0.
    for i in range(count):
        a = json.loads((base/f'{i:02d}.json').read_text())
        certified += a['audit']['pdfs']
        checked += a['audit']['rows']
        worst_q = max(worst_q,a['audit']['maximum_quantile_cdf_error'])
        worst_shell = max(worst_shell,a['audit']['maximum_shell_probability_error'])
        name = f'{i:02d}_identities.npz'
        items.append(load(read_cache(base,name,f'{i:02d}.json'))['ls_id'])
    ids = np.concatenate(items)
    historical = None
    if side=='south':
        with np.load(read_cache(INDEX,'catalog_rows.npz'),allow_pickle=False) as z:
            historical = z['ls_id']
    unique_catalog(ids,side,historical_ids=historical)
    if len(ids)!=checked:
        raise ValueError('Global native identity count differs from row audits')
    return dict(rows=checked,pdfs=certified,groups=count,maximum_quantile_cdf_error=worst_q,
                maximum_shell_probability_error=worst_shell)
