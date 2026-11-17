#!/usr/bin/env python3
"""Validate the paper science products and replay the portable access example."""
import csv
import json
import os

from astropy.io import fits
import numpy as np
from PIL import Image

from photdis.stage1 import require_compute
from v2_1_near import ROOT, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from paper_v1_read import CORE, SUPPORT, read
from paper_v1_scan import OUT
from v3_final_scan import OUT as BASELINE


def release(inputs):
    package=BASELINE/'release'
    bench=OUT/'benchmark'
    p=package/'complete.json'
    info=json.loads(p.read_text())
    for k,h in info['outputs'].items():
        verified(p.with_name(k),h)
    inputs['frozen_staging']=sha(p)
    assert info['n_pdfs']==43347612 and info['core_rows']==2281023
    with fits.open(package/CORE,memmap=True) as f, fits.open(bench/SUPPORT,memmap=True) as s:
        c=f['PHOTDIS_CORE'].data
        a=s['OBS_SUPPORT'].data
        assert len(a)==len(c)==2281023 and np.all(c['TYPE']=='PSF')
        for name in ('LS_ID','SCIENCE_TIER'):
            np.testing.assert_array_equal(a[name],c[name])
        for n in ('SPECDIS','SPDIST'):
            np.testing.assert_array_equal(a[n+'_SUPPORT'],a['N_'+n]>=20)
    with (package/'pdf_access_examples.csv').open() as f:
        examples=list(csv.DictReader(f))
    results=[]
    for case in examples:
        if case['region']=='south_faint':
            continue
        row=read(package,bench,int(case['ls_id']),ROOT)
        assert abs(row['distance_kpc'][2]-float(case['d50_kpc']))<1e-5
        np.testing.assert_allclose(sum(row.pop('mu_mass')),1,rtol=0,atol=1e-10)
        row.pop('mu_edges')
        results.append(dict(region=case['region'],**row))
    cal=ROOT/'manifests/v3_9/near_calibration/summary.json'
    meta=json.loads(cal.read_text())
    assert meta['rows']==1142280
    shard=meta['shards'][0]
    marker=verified(ROOT/shard['receipt'],shard['receipt_sha256'])
    values=json.loads(marker.read_text())['outputs']
    p=next(k for k in values if k.endswith('.npz'))
    calibrated=read_columns(verified(marker.with_name(p),values[p]),
               ('ls_id','TYPE','qrf_quantiles_kpc','qrf_mu_mass','qrf_mu_edges'))
    with (package/'pdf_partitions.csv').open() as f:
        part=next(r for r in csv.DictReader(f) if r['region']=='south_v35' and
                   r['sweep']+'_'+r['photo_file'].split('/')[-1]==p)
    folder=verified(ROOT/part['pdf_file'],part['pdf_sha256'])
    raw=read_columns(folder,tuple(calibrated))
    _,i,j=np.intersect1d(raw['ls_id'],calibrated['ls_id'],return_indices=True)
    assert len(i)>0 and np.all(raw['TYPE'][i]=='PSF') and np.all(calibrated['TYPE'][j]=='PSF')
    np.testing.assert_array_equal(raw['qrf_mu_mass'][i],calibrated['qrf_mu_mass'][j])
    np.testing.assert_allclose(raw['qrf_quantiles_kpc'][i,2],calibrated['qrf_quantiles_kpc'][j,2],rtol=0,atol=1e-10)
    assert np.any(raw['qrf_mu_edges'][i]!=calibrated['qrf_mu_edges'][j])
    inputs['calibrated_near']=sha(cal)
    return results,len(i)


def main():
    require_compute()
    inputs={}
    for stage in ('benchmark','provenance','figures'):
        mark=OUT/stage/'complete.json'
        r=json.loads(mark.read_text())
        for k,digest in r['outputs'].items():
            verified(mark.with_name(k),digest)
        inputs[stage]=sha(mark)
    p=OUT/'figures'
    plates=[]
    for fig in sorted(p.glob('*.png')):
        with Image.open(fig) as im:
            im.verify()
        with Image.open(fig) as im:
            assert im.width>=2000 and all(abs(x-300)<.02 for x in im.info['dpi'])
        assert fig.with_suffix('.pdf').read_bytes().startswith(b'%PDF')
        plates.append(fig.stem)
    assert len(plates)==6
    replays,cal=release(inputs)
    save_json(OUT/'delivery.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,plates=plates,
                      core_pdf_examples=replays,calibration_example_rows=cal))


if __name__=='__main__':
    main()
