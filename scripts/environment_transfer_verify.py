#!/usr/bin/env python3
"""Verify source joins, counts, receipts and independent PDF quadrature."""
import csv
import json

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

from environment_transfer import FROZEN, OLD, OUT, TABLES
from environment_transfer_stats import rows_for_pdf
from photdis.stage1 import require_compute
from v2_1_near import load, save_json, sha


def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def cdf_crps(edges,mass,target):
    """Integrate (CDF - observed step)^2 using the original bins."""
    cdf = 0.
    total = max(0.,edges[0]-target,target-edges[-1])
    for lo,hi,m in zip(edges[:-1],edges[1:],mass):
        if hi<=lo:
            assert m==0.
            continue
        density = m/(hi-lo)
        mid = np.clip(target,lo,hi)
        for a,b,step in ((lo,mid,0),(mid,hi,1)):
            if b>a:
                f = lambda x: (cdf+density*(x-lo)-step)**2
                total += quad(f,a,b,epsabs=1e-13)[0]
        cdf += m
    assert abs(cdf-1)<1e-9
    return total


def audit_pdf(pdf, d, sigma, indices):
    max_crps, max_log = 0.,0.
    for edges,mass in ((pdf['mu_edges'],pdf['mass']),(d['base_edges'],d['base_mass']),
                       (d['iteration_edges'],d['iteration_mass'])):
        mu = d['reference_mu'][indices]
        sig = sigma[indices]
        e,w = edges[indices],mass[indices]
        compact = rows_for_pdf(e,w,mu,sig)
        for i,(bins,weight,target,err) in enumerate(zip(e,w,mu,sig)):
            crps = cdf_crps(bins,weight,target)
            max_crps = max(max_crps,abs(crps-compact[i,2]))
            z = (bins-target)/err
            a,b = z[:-1],z[1:]
            flip = a+b>0
            gaussian = np.where(flip,ndtr(-a)-ndtr(-b),ndtr(b)-ndtr(a))
            width = np.diff(bins)
            value = np.divide(gaussian*weight,width,out=np.zeros_like(weight),where=width>0).sum()
            assert value>0
            max_log=max(max_log,abs(np.log(value)-compact[i,3]))
    assert max_crps<1e-9 and max_log<1e-9,(max_crps,max_log)
    return dict(n=len(indices)*3,crps_abs_error=max_crps,log_abs_error=max_log)


def main():
    require_compute()
    t,p,d = map(load,(OUT/'forecast/roster.npz',OUT/'forecast/pdf.npz',OUT/'dust/forecasts.npz'))
    old = load(OLD/'forecast/roster.npz')
    refs = t['source_id'][d['rows']]
    assert not np.intersect1d(refs,old['source_id']).size
    assert len(np.unique(refs))==len(refs) and len(refs)==1500
    assert (d['TYPE']=='PSF').all() and (t['TYPE'][d['rows']]=='PSF').all()
    order = np.searchsorted(p['rows'],d['rows'])
    np.testing.assert_array_equal(p['ls_id'][order],d['ls_id'])
    p = {key:value[order] for key,value in p.items()}
    flows = table(TABLES/'flow.csv')
    pairs = table(TABLES/'pairs.csv')
    assert all(r['TYPE']=='PSF' for r in pairs)
    inside = (d['reference_mu']>=d['mu_min'])&(d['reference_mu']<=d['mu_max'])
    common = inside&(d['iteration_state']==5)
    confirmed = []
    for row in flows:
        f = t['field'][d['rows']]==row['field']
        assert sum(int(row[f'state{k}']) for k in range(6))==int(row['registered'])
        assert int(row['common'])==(f&common).sum()
        ids = [int(r['ls_id']) for r in pairs if r['field']==row['field']]
        np.testing.assert_array_equal(ids,d['ls_id'][f&common])
        confirmed.extend(np.flatnonzero(f&common)[:12])
    audit = audit_pdf(p,d,t['gaia_sigma_mu'][d['rows']],np.array(confirmed))
    faint = table(TABLES/'faint_signed_parallax.csv')
    assert len(faint)==27
    for r in faint:
        for stage in ('all','grz','precise','qrf_support','qrf_history_clean'):
            assert 0<=int(r[stage+'_negative'])<=int(r[stage+'_weak'])<=int(r[stage+'_signed_usable'])<=int(r[stage+'_n'])
    l219 = table(TABLES/'l219_stars.csv')
    assert len(l219)==500 and all(r['TYPE']=='PSF' for r in l219)
    assert sum(int(r['state'])==5 for r in l219)==198
    stage_folders = (OUT/'census',OUT/'forecast',OUT/'apogee',OUT/'dust',OUT/'figures')
    matches = 0
    for path in [*(d/'complete.json' for d in stage_folders),
                 TABLES/'transfer_receipt.json',TABLES/'inventory_receipt.json']:
        record = json.loads(path.read_text())
        for name,digest in record['outputs'].items():
            assert sha(path.parent/name)==digest,name
            matches += 1
    style = json.loads((OUT/'figures/style_regression.json').read_text())
    save_json(TABLES/'validation.json',dict(fields=len(flows),registered=len(refs),scored=len(pairs),
        l219_archived=500,gaia_faint_cells=len(faint),receipt_outputs=matches,style_regression=style,
        independent_quadrature=audit,preoutcome_freeze=sha(FROZEN)))
    print('Verified',audit,'and',matches,'receipt files.',flush=True)


if __name__=='__main__':
    main()
