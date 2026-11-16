#!/usr/bin/env python3
"""Cross-check the fixed Gaia faint cohort against published WD candidate tables."""
import argparse
from pathlib import Path

import fitsio
import numpy as np

from photdis.reference_v1_4 import unique_join
from v2_1_near import csv_rows, save_json, sha
from v3_6_vac_distances import read_columns

SOURCE = 'https://warwick.ac.uk/fac/sci/physics/research/astro/research/catalogues/'
WD_CLASSES = ('DA','DB','DBA','DAB','DO','DAO','DC','DAZ','DZA','DZB','DBAZ','DABZ','DZBA',
              'DZAB','DZ','DQ','hotDQ','pecDQ','DAH','DBH','DZH','MWD','PG1159','WD')


def match(path,keys,ids):
    data = fitsio.read(path,ext=1,columns=['source_id',*keys])
    j = unique_join(ids,data['source_id'])
    use = j>=0
    return data,j,use


def run(cohort,spectra,main,output):
    keys = ('ls_id','source_id','TYPE','clean_low_luminosity','reference_kpc','qrf_quantiles_kpc')
    t = read_columns(cohort,keys)
    ids = t['source_id']
    if len(np.unique(ids))!=2003 or not np.all(t['TYPE']=='PSF'):
        raise ValueError('Expected the frozen 2,003 unique native PSF and Gaia identities')
    table = fitsio.read(spectra,ext=1,columns=['source_id','Pwd','spec_class'])
    table = table[np.isin(table['source_id'],ids)]
    label = np.full(len(ids),'',dtype='U8')
    found = np.isin(ids,table['source_id'])
    wd = np.zeros(len(ids),bool)
    pwd = np.full(len(ids),np.nan)
    for i in np.flatnonzero(found):
        p = table[table['source_id']==ids[i]]
        values = np.unique(np.char.strip(p['spec_class'].astype('U')))
        good = values[~np.isin(values,('','Unreli','UNKN'))]
        label[i] = good[0] if len(good)==1 else ('multiple' if len(good)>1 else 'unreli')
        wd[i] = len(good)>0 and np.isin(good,WD_CLASSES).all()
        available = p['Pwd'][np.isfinite(p['Pwd'])]
        if len(available):
            pwd[i] = np.median(available)
    meta = dict(cohort_sha256=sha(cohort),spectra_sha256=sha(spectra),
                spectra_url=SOURCE+'gaiaedr3_wd_sdssspec.fits.gz')
    if main is not None:
        source,i,has = match(main,['Pwd'],ids)
        pwd[has] = source['Pwd'][i[has]]
        meta.update(main_sha256=sha(main),main_url=SOURCE+'gaiaedr3_wd_main.fits.gz')
    ratio = t['qrf_quantiles_kpc'][:,2]/t['reference_kpc']
    severe = abs(5*np.log10(ratio))>1
    output.mkdir(parents=True,exist_ok=True)
    table = output/'wd_evidence.csv'
    csv_rows(table,(dict(ls_id=int(ls_id),source_id=int(sid),TYPE='PSF',spec_class=label[i],
        published_pwd=float(pwd[i]),spectroscopic_wd=bool(wd[i]),severe=bool(severe[i]),
        cmd_candidate=bool(t['clean_low_luminosity'][i]),distance_ratio=float(ratio[i]))
        for i,(ls_id,sid) in enumerate(zip(t['ls_id'],ids))))
    save_json(output/'complete.json',dict(**meta,rows=len(ids),pwd_rows=int(np.isfinite(pwd).sum()),
        pwd_above_point75=int((pwd>.75).sum()),spectroscopic_matches=int(found.sum()),
        spectroscopic_wd=int(wd.sum()),spectroscopic_wd_severe=int((wd&severe).sum()),
        spectroscopic_wd_cmd=int((wd&t['clean_low_luminosity']).sum()),table_sha256=sha(table)))


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cohort',type=Path)
    p.add_argument('spectra',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--main',type=Path)
    a = p.parse_args()
    run(a.cohort,a.spectra,a.main,a.output)
