#!/usr/bin/env python3
"""Read a staged core source, its coverage sidecar, and optionally its frozen PDF."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from astropy.io import fits
import numpy as np

CORE = 'photdis-dr11-core-v3-final.1.fits'
SUPPORT = 'core_observable_support.fits'


def _entry(table,ls_id):
    j=np.searchsorted(table['LS_ID'],ls_id)
    if j==len(table) or table['LS_ID'][j]!=ls_id:
        raise KeyError(ls_id)
    return table[j]


def read(package,support,ls_id,archive=None):
    with fits.open(package/CORE,memmap=True) as f, fits.open(support/SUPPORT,memmap=True) as s:
        c=_entry(f['PHOTDIS_CORE'].data,ls_id)
        flags=_entry(s['OBS_SUPPORT'].data,ls_id)
        if c['TYPE']!='PSF' or c['SCIENCE_TIER']!=flags['SCIENCE_TIER']:
            raise ValueError('Invalid join between the frozen catalogue and coverage flags')
        result=dict(ls_id=ls_id,tier=int(c['SCIENCE_TIER']),distance_kpc=c['D_KPC'].tolist(),
             obs_cell=int(flags['OBS_CELL']),f15_specdis_n=int(flags['N_SPECDIS']),
             f15_spdist_n=int(flags['N_SPDIST']),f15_specdis_supported=bool(flags['SPECDIS_SUPPORT']),
             f15_spdist_supported=bool(flags['SPDIST_SUPPORT']))
        part,pdf=int(c['PARTITION_ID']),int(c['PDF_ROW'])
    if archive is not None:
        with (package/'pdf_partitions.csv').open() as f:
            row=next(r for r in csv.DictReader(f) if int(r['partition_id'])==part)
        path=archive/row['pdf_file']
        with path.open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=row['pdf_sha256']:
                raise ValueError('Archived PDF bytes do not match the partition index')
        with np.load(path,allow_pickle=False) as q:
            if q['ls_id'][pdf]!=ls_id or q['TYPE'][pdf]!='PSF':
                raise ValueError('The archived PDF offset did not resolve to the FITS source')
            result.update(mu_edges=q['qrf_mu_edges'][pdf].tolist(),mu_mass=q['qrf_mu_mass'][pdf].tolist())
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--support',type=Path,required=True)
    p.add_argument('--archive',type=Path)
    p.add_argument('--ls-id',type=int,required=True)
    a=p.parse_args()
    print(json.dumps(read(a.package,a.support,a.ls_id,a.archive)))
