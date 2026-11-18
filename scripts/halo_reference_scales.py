#!/usr/bin/env python3
"""Separate member scatter from a shared published cluster distance-scale error."""
from photdis.stage1 import require_compute
from halo_common import *
from sv18_common import interval


def main():
    require_compute();d=folder('reference_scales');inputs={}
    p=ROOT/'configs/halo_system_references_20261118.json';ref=json.loads(p.read_text());inputs['reference_config']=sha(p)
    t=read(OUT/'sequence',('same_source_predictions.npz','TYPE','ls_id','system','r','q','ref','mask','common','simple','sequence'),inputs)
    rows=[]
    for system in ('M5','M13','M30','M92'):
        rr=ref[system];D=rr['distance_kpc'];sigma=rr['sigma_kpc']
        common_D=RNG.normal(D,sigma,size=256);assert (common_D>0).all()
        for low,high in ((14,18),(18,20),(20,21),(21,22),(22,23),(14,23)):
            use=(t['system']==system)&(t['r']>=low)&(t['r']<high)&t['common']&~t['mask']
            ii=np.flatnonzero(use)
            if not len(ii):continue
            np.testing.assert_allclose(t['ref'][ii],D,atol=1e-12)
            bootstrap=RNG.integers(len(ii),size=(256,len(ii)))
            for key,method in (('q','QRF'),('simple','Label grid'),('sequence','Old sequence')):
                med=np.median(t[key][ii,2][bootstrap],axis=1)
                conditional=interval(med/D-1);combined=interval(med/common_D-1)
                rows.append(dict(system=system,r_lo=low,r_hi=high,method=method,n=len(ii),
                    reference_kpc=D,reference_common_sigma_kpc=sigma,reference_common_sigma_mu=5/np.log(10)*sigma/D,
                    median_bias=np.median(t[key][ii,2]/D-1),
                    conditional_ci_lo=conditional['ci_lo'],conditional_ci_hi=conditional['ci_hi'],
                    common_scale_ci_lo=combined['ci_lo'],common_scale_ci_hi=combined['ci_hi'],
                    scope='shared-reference Gaussian sensitivity on fixed center-defined sample; member bootstrap, one common nuisance per cluster/draw'))
    csv_rows(d/'cluster_common_scale_sensitivity.csv',rows)
    complete(d,inputs,published_reference_source=ref['source_url'],
        common_error='one common Gaussian distance draw per cluster reused across methods and magnitude groups; never divided by sqrt(N)',
        GD1='cached distance-curve centers do not contain a joint scale covariance; no invented independent per-member errors',
        scope='supplemental uncertainty sensitivity, no new member selection or zero-point fitting')


if __name__=='__main__':main()
