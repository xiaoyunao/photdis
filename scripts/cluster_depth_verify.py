#!/usr/bin/env python3
"""Verify original membership and independent scores on actual model PDFs."""
import csv
import json

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

from cluster_depth_inputs import C, CONFIG, OLD, OUT
from cluster_depth_physics import MODELS
from cluster_depth_results import PRODUCT
from photdis.stage1 import require_compute
from v2_1_near import load, save_json, sha


def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def numeric_crps(means,weights,sigma,ref):
    tail=12*sigma
    def cdf(x):
        return float(np.dot(weights,ndtr((x-means)/sigma)))
    low=min(means.min()-tail,ref)
    high=max(means.max()+tail,ref)
    return quad(lambda x:cdf(x)**2,low,ref,epsabs=1e-10)[0]+quad(
        lambda x:(1-cdf(x))**2,ref,high,epsabs=1e-10)[0]


def main():
    require_compute()
    result=json.loads((PRODUCT/'decision.json').read_text())
    assert result['config']==sha(CONFIG)
    assert all(sha(PRODUCT/k)==v for k,v in result['outputs'].items())
    seal=json.loads((OLD/'results/cluster_freeze.json').read_text())
    cmd=table(PRODUCT/'cmd.csv')
    rows=[]
    for key in C['systems']:
        a=load(OUT/(key+'_diagnostics.npz'))
        n=len(a['ls_id'])
        np.testing.assert_array_equal(seal['admitted'][key],a['ls_id'])
        assert (a['TYPE']=='PSF').all()
        for color in ('gr','rz'):
            assert sum(int(r['N']) for r in cmd if r['system']==key and r['color']==color)==n
        for model in MODELS:
            p=load(OUT/(key+'_'+model+'_posterior.npz'))
            np.testing.assert_array_equal(p['ls_id'],a['ls_id'])
            assert (p['TYPE']=='PSF').all()
            np.testing.assert_allclose(np.exp(p['logw']).sum(1),1,atol=1e-10)
            supported=np.flatnonzero(p['supported'])
            assert len(supported)>0
            target=a['qrf_median']-a['qrf_bias']
            probe=supported[np.linspace(0,len(supported)-1,4,dtype=int)]
            errors=[]
            for j in probe:
                ncrps=numeric_crps(p['means'][j],np.exp(p['logw'][j]),p['sigma'][j],target[j])
                errors.append(abs(ncrps-a[model+'_crps'][j]))
            assert max(errors)<2e-8, (key,model,errors)
            rows.append(dict(system=key,model=model,roster=n,replayed_crps=len(probe),
                             max_error=max(errors),pdf_archive_sha256=sha(OUT/(key+'_'+model+'_posterior.npz'))))
    save_json(PRODUCT/'validation.json',dict(config=sha(CONFIG),total_members=sum(map(len,seal['admitted'].values())),
                  checked_output_tables=len(result['outputs']),all_cmd_members=True,checks=rows))
    print('Verified all member rows and 36 independent mixture integrations',flush=True)


if __name__=='__main__':
    main()
