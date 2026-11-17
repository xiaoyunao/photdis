#!/usr/bin/env python3
"""Check final receipts and replay representative external-user PDF lookups."""
import csv
import json
import os

from astropy.io import fits
import numpy as np
from PIL import Image

from photdis.stage1 import require_compute
from v2_1_near import ROOT, save_json, sha
from v3_10_figures import check_templates
from v3_final_release import FITS_NAME, lookup
from v3_final_scan import OUT


def read_receipt(stage):
    root = OUT/stage
    r = json.loads((root/'complete.json').read_text())
    for name, digest in r['outputs'].items():
        if sha(root/name)!=digest:
            raise ValueError(f'Product checksum failed: {root/name}')
    return r


def main():
    require_compute()
    stages = ('science','atlas','release')
    receipts = {s:read_receipt(s) for s in stages}
    images = []
    for stage in ('science','atlas'):
        for name in receipts[stage]['outputs']:
            if not name.endswith('.png'):
                continue
            p = OUT/stage/name
            assert p.with_suffix('.pdf').read_bytes().startswith(b'%PDF-')
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                assert im.width>=1800 and im.height>=1200
                assert all(abs(v-300)<.02 for v in im.info['dpi'])
            images.append(name)
    assert len(images)==6
    release = OUT/'release'
    r = receipts['release']
    assert r['core_rows']==2281023 and r['tiers']==[0,1115077,1165946]
    assert r['n_pdfs']==43347612 and r['partitions']==28917
    with fits.open(release/FITS_NAME,memmap=True) as f:
        t = f['PHOTDIS_CORE'].data
        assert len(t)==r['core_rows'] and np.all(np.diff(t['LS_ID'])>0)
        assert np.all(t['TYPE']=='PSF')
    with (release/'pdf_access_examples.csv').open() as f:
        cases = list(csv.DictReader(f))
    replays = []
    for row in cases:
        if row['region']=='south_faint':
            continue
        q = lookup(release,ROOT,int(row['ls_id']))
        assert int(q['ls_id'])==int(row['ls_id'])
        np.testing.assert_allclose(q['qrf_quantiles_kpc'][2],float(row['d50_kpc']),rtol=0,atol=1e-10)
        np.testing.assert_allclose(sum(q['qrf_mu_mass']),1,rtol=0,atol=1e-10)
        replays.append(row['region'])
    assert len(replays)==3 and len(cases)==4
    save_json(OUT/'delivery.json',dict(job=os.getenv('SLURM_JOB_ID'),style=check_templates(),
        inputs={s:sha(OUT/s/'complete.json') for s in stages},
        valid_plates=sorted(images), pdf_core_replays=replays, pdf_raw_replays=len(cases),
        full_pdf_rows=r['n_pdfs'],core_rows=r['core_rows']))


if __name__=='__main__':
    main()
