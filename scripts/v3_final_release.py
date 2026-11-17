#!/usr/bin/env python3
"""Stage the versioned core FITS and a verified download index for the PDF archive."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import tempfile

from astropy.io import fits
from astropy.table import Table
import numpy as np

from photdis.stage1 import require_compute
from v2_1_near import ROOT, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_10_working_sample import DEST as WORK
from final_science_catalog import indices, REGIONS
from release_readiness_access import get_partition
from v3_final_scan import CORE_FIELDS, OUT, SHARDS

VERSION = '3-final.1'
FITS_NAME = f'photdis-dr11-core-v{VERSION}.fits'
SCHEMA = ROOT/'configs/photdis_dr11_release.json'
FIELDS = ('ls_id','TYPE','RA','DEC','source_id','EBV','r','color','q','bins','science','region','gaia_state')


def read_core(inputs):
    marker = ROOT/'data/final_science/evidence/complete.json'
    r = json.loads(marker.read_text())
    inputs['selected'] = sha(marker)
    t = read_columns(verified(marker.with_name('selected.npz'),r['outputs']['selected.npz']),FIELDS)
    k = np.flatnonzero(np.isin(t['science'],(1,2)))
    k = k[np.argsort(t['ls_id'][k])]
    t = {key:v[k] for key,v in t.items()}
    assert len(t['ls_id'])==2281023 and np.all(np.diff(t['ls_id'])>0) and np.all(t['TYPE']=='PSF')
    return t


def join_links(t, inputs):
    z = []
    for i in range(SHARDS):
        mark = OUT/'scan'/f'{i:02}.json'
        r = json.loads(mark.read_text())
        inputs[str(mark.relative_to(ROOT))] = sha(mark)
        f = mark.with_suffix('.npz')
        z.append(read_columns(verified(f,r['outputs'][f.name]),CORE_FIELDS))
    links = {k:np.concatenate([s[k] for s in z]) for k in CORE_FIELDS}
    order = np.argsort(links['ls_id'])
    np.testing.assert_array_equal(links['ls_id'][order], t['ls_id'])
    t.update({k:v[order] for k,v in links.items() if k!='ls_id'})
    assert np.all(t['row_state']==7) and not t['near_numerical_flag'].any()
    assert np.isin(t['w1_state'][t['science']==1], (2,4)).all()


def working_score(t, parts, inputs):
    found = np.zeros(len(t['ls_id']), bool)
    names = np.array([p['pdf_file'] for p in parts])
    for side in ('north','south'):
        for i in range(48):
            mark = WORK/side/f'{i:02}.json'
            r = json.loads(mark.read_text())
            inputs[str(mark.relative_to(ROOT))] = sha(mark)
            if not r['rows']:
                continue
            files = r['outputs']
            p = mark.with_suffix('.npz')
            a = read_columns(verified(p,files[p.name]),
                             ('ls_id','TYPE','state','score','part_index','pdf_rows'))
            sub = np.flatnonzero(a['state']==4)
            where = np.searchsorted(t['ls_id'],a['ls_id'][sub])
            valid = where < len(found)
            valid[valid] = t['ls_id'][where[valid]]==a['ls_id'][sub[valid]]
            j, k = where[valid], sub[valid]
            assert not found[j].any() and np.all(t['science'][j]==2) and np.all(a['TYPE'][k]=='PSF')
            np.testing.assert_array_equal(t['pdf_row'][j],a['pdf_rows'][k])
            f = mark.with_suffix('.csv')
            with verified(f,files[f.name]).open() as stream:
                locators = np.array([row['file'] for row in csv.DictReader(stream)])
            np.testing.assert_array_equal(names[t['partition_id'][j]],locators[a['part_index'][k]])
            t['selection_score'][j] = a['score'][k]
            found[j] = True
    np.testing.assert_array_equal(found,t['science']==2)
    assert np.all(t['selection_score'][found]>=.80)


def write_fits(dest, t):
    cols = dict(LS_ID=t['ls_id'], TYPE=t['TYPE'].astype('S3'), RA=t['RA'], DEC=t['DEC'],
                GAIA_SOURCE_ID=t['source_id'], EBV=t['EBV'].astype('f4'),
                OBSERVED_R=t['r'].astype('f4'), DERED_G_R=t['color'].astype('f4'),
                D_KPC=t['q'].astype('f4'), DIST_BIN_MASS=t['bins'].astype('f4'),
                SCIENCE_TIER=t['science'], REGION=t['region'], GAIA_STATE=t['gaia_state'],
                PARTITION_ID=t['partition_id'].astype('i4'), PDF_ROW=t['pdf_row'].astype('i4'),
                PHOTO_ROW=t['photo_row'].astype('i4'), GRZ_ROW_STATE=t['row_state'],
                NOMINAL_NUMERICAL_FLAG=t['near_numerical_flag'], W1_STATE=t['w1_state'].astype('i2'),
                SELECTION_SCORE=t['selection_score'])
    table = Table(cols)
    for name,unit in (('RA','deg'),('DEC','deg'),('EBV','mag'),('OBSERVED_R','mag'),
                      ('DERED_G_R','mag'),('D_KPC','kpc')):
        table[name].unit = unit
    primary = fits.PrimaryHDU()
    primary.header['VERSION'] = VERSION
    primary.header['RELEASE'] = 'DR11'
    primary.header['SRCV3'] = 'c530007c'
    data = fits.BinTableHDU(table, name='PHOTDIS_CORE')
    data.header['QPROBS'] = '.025,.16,.5,.84,.975'
    data.header['DBINS'] = '0,2,5,10,20,50,inf'
    data.header['ROWBASE'] = (0,'Original zero-based rows, before ownership masks')
    with tempfile.TemporaryDirectory(dir=dest) as scratch:
        path = Path(scratch)/FITS_NAME
        fits.HDUList([primary,data]).writeto(path,checksum=True)
        with fits.open(path) as hdus:
            for h in hdus:
                assert h.verify_checksum()==h.verify_datasum()==1
            q = hdus[1].data
            for k in cols:
                np.testing.assert_array_equal(q[k],cols[k].astype('U3') if k=='TYPE' else cols[k])
            assert np.all(np.diff(q['D_KPC'],axis=1)>=0) and np.all(q['D_KPC']>0)
            np.testing.assert_allclose(q['DIST_BIN_MASS'].sum(axis=1),1,atol=2e-7,rtol=0)
        path.replace(dest/FITS_NAME)


def full_pdf_index(dest, parts, overrides, t):
    csv_rows(dest/'pdf_partitions.csv',[dict(partition_id=i,**p) for i,p in enumerate(parts)])
    counts = dict(partitions=len(parts), downloadable_pdf_files=sum(bool(p['pdf_file']) for p in parts),
                  n_pdfs=sum(int(p['owned_pdf_rows']) for p in parts))
    assert counts['partitions']==28917 and counts['n_pdfs']==43347612
    assert len(set(p['pdf_file'] for p in parts if p['pdf_file']))==counts['downloadable_pdf_files']
    atomic = ROOT/'data/v3_9/delivery/seam/ownership_overrides.npz'
    shutil.copyfile(atomic,dest/'ownership_overrides.npz')
    assert sha(atomic)==sha(dest/'ownership_overrides.npz')
    rows = []
    for i,region in enumerate(REGIONS):
        match = np.flatnonzero(t['region']==i)
        j = int(match[len(match)//2]) if len(match) else None
        pidx = int(t['partition_id'][j]) if j is not None else max(
               (k for k,p in enumerate(parts) if p['region']==region),key=lambda k:int(parts[k]['owned_pdf_rows']))
        part = parts[pidx]
        raw = get_partition(part,overrides)
        example_id = int(t['ls_id'][j]) if j is not None else int(raw['ls_id'][0])
        loc, = np.flatnonzero(raw['ls_id']==example_id)
        d = raw['qrf_quantiles_kpc'][loc]
        if j is not None:
            np.testing.assert_array_equal(d,t['q'][j])
            np.testing.assert_array_equal(raw['qrf_bins'][loc],t['bins'][j])
            with np.load(ROOT/part['pdf_file'],allow_pickle=False) as p:
                assert p['ls_id'][t['pdf_row'][j]]==example_id
        rows.append(dict(region=region,partition_id=pidx,pdf_file=part['pdf_file'],ls_id=example_id,
                         d50_kpc=d[2],n_owned=len(raw['ls_id'])))
    csv_rows(dest/'pdf_access_examples.csv',rows)
    return counts


def build():
    dest = OUT/'release'
    dest.mkdir(parents=True,exist_ok=True)
    parts, overrides, inputs = indices()
    t = read_core(inputs)
    join_links(t, inputs)
    for region in np.unique(t['region']):
        bins = np.unique(t['partition_id'][t['region']==region])
        assert all(parts[i]['region']==REGIONS[region] for i in bins)
    working_score(t, parts, inputs)
    pdf = full_pdf_index(dest, parts, overrides, t)
    write_fits(dest, t)
    source = json.loads(SCHEMA.read_text())
    source.pop('data_root')
    source['source_index'] = 'pdf_partitions.csv'
    source['sweep_ownership']['override_file'] = 'ownership_overrides.npz'
    source['core_fits'] = dict(path=FITS_NAME, hdu='PHOTDIS_CORE', key='LS_ID', rows=len(t['ls_id']),
        selection_score={'tier_1':'v3.5 dwarf_selection_score',
                         'tier_2':'v3.10 native calibrated classifier score'},
        raw_pdf_locator=dict(partition='PARTITION_ID',row='PDF_ROW',base=0),
        quality_flags=('SCIENCE_TIER','GAIA_STATE','GRZ_ROW_STATE','NOMINAL_NUMERICAL_FLAG','W1_STATE'))
    save_json(dest/'schema.json',source)
    files = [FITS_NAME,'schema.json','pdf_partitions.csv','pdf_access_examples.csv','ownership_overrides.npz']
    (dest/'SHA256SUMS').write_text(''.join(f'{sha(dest/k)}  {k}\n' for k in files))
    save_json(dest/'complete.json',dict(version=VERSION,job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        core_rows=len(t['ls_id']),tiers=np.bincount(t['science']).tolist(),**pdf,
        status='staged_on_nersc',outputs={k:sha(dest/k) for k in (*files,'SHA256SUMS')}))


def lookup(package, archive_root, ls_id):
    # Archive root is the downloaded directory tree, using the relative paths in the manifest.
    with fits.open(package/FITS_NAME,memmap=True) as f:
        table = f['PHOTDIS_CORE'].data
        j = np.searchsorted(table['LS_ID'],ls_id)
        if j==len(table) or table['LS_ID'][j]!=ls_id:
            raise KeyError(ls_id)
        part, row = (int(table[k][j]) for k in ('PARTITION_ID','PDF_ROW'))
    with (package/'pdf_partitions.csv').open() as f:
        p = next(r for r in csv.DictReader(f) if int(r['partition_id'])==part)
    file = verified(archive_root/p['pdf_file'],p['pdf_sha256'])
    with np.load(file,allow_pickle=False) as data:
        if data['ls_id'][row]!=ls_id or data['TYPE'][row]!='PSF':
            raise ValueError('Core PDF lookup did not resolve to the verified source')
        return {k:data[k][row].tolist() for k in ('ls_id','qrf_quantiles_kpc','qrf_mu_edges',
                                                 'qrf_mu_mass','qrf_bins')}


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ls-id',type=int)
    parser.add_argument('--package',type=Path,default=OUT/'release')
    parser.add_argument('--archive-root',type=Path,default=ROOT)
    a = parser.parse_args()
    if a.ls_id is None:
        require_compute()
        build()
    else:
        print(json.dumps(lookup(a.package,a.archive_root,a.ls_id)))
