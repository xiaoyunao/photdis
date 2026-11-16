#!/usr/bin/env python3
"""Qualify the already indexed expansion nonstellar controls in DESI DR1."""
import argparse
import json
import os

import fitsio
import numpy as np

from photdis.reference_v1_4 import unique_join
from photdis.stage1 import require_compute
from v2_1_controls import COLS, ZCAT
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_6_vac_distances import read_columns
from v3_9_references import OLDER, join, matched, sorted_key
from v3_10_evidence import DEST as EVIDENCE, REGIONS
from v3_10_selection import DEST as SELECTION, THRESHOLDS

DEST = ROOT/'data/v3_10/redrock'


def source_roster():
    keys = ('source_id','ls_id','targetid','RA','DEC')
    s = read_columns(OLDER,keys)
    for name,index in (('source_id','_gaia'),('ls_id','_legacy')):
        s[index] = sorted_key(s[name],np.flatnonzero(s[name]>0))
    parts,inputs = [],{str(OLDER.relative_to(ROOT)):sha(OLDER)}
    for name in REGIONS:
        root = EVIDENCE/name
        receipt = root/'complete.json'
        meta = json.loads(receipt.read_text())
        t = read_columns(verified(root/'references.npz',meta['outputs']['references.npz']),
            ('source_id','ls_id','TYPE','RA','DEC','spectype','row_state','r','redrock_quality_assessed'))
        wanted = np.isin(t['spectype'],('QSO','GALAXY'))&(t['row_state']==7)
        t = {key:value[wanted] for key,value in t.items()}
        if not np.all(t['TYPE']=='PSF'):
            raise ValueError('Redrock controls require independently verified native PSF rows')
        row = matched(t,s,legacy=True)
        j = np.flatnonzero(row>=0)
        target = np.full(len(row),-1,'i8')
        target[j] = s['targetid'][row[j]]
        parts.append(dict(**t,targetid=target,region=np.full(len(row),name,dtype='U11')))
        inputs[str(receipt.relative_to(ROOT))] = sha(receipt)
    return join(parts),inputs


def scan(targets):
    rows = []
    with fitsio.FITS(ZCAT) as fits:
        table = fits[1]
        n = table.get_nrows()
        for start in range(0,n,500_000):
            block = table.read(rows=np.arange(start,min(start+500_000,n)),columns=['TARGETID','ZCAT_PRIMARY'])
            use = np.flatnonzero(np.isin(block['TARGETID'],targets)&block['ZCAT_PRIMARY'])
            if len(use):
                part = table.read(rows=start+use,columns=COLS)
                rows.append({k:part[k] for k in COLS})
        if not rows:
            empty = table.read(rows=np.array([0]),columns=COLS)[:0]
            return {key:empty[key] for key in COLS}
    return join(rows)


def qualify(t,r):
    rr = unique_join(t['targetid'],r['TARGETID'])
    j = np.flatnonzero(rr>=0)
    q = rr[j]
    dra = ((r['TARGET_RA'][q]-t['RA'][j]+180)%360-180)*np.cos(np.deg2rad(t['DEC'][j]))
    offsets = 3600*np.hypot(dra,r['TARGET_DEC'][q]-t['DEC'][j])
    kind = np.char.strip(r['SPECTYPE'][q].astype('U'))
    target = np.char.strip(r['OBJTYPE'][q].astype('U'))
    good = ((offsets<1)&np.isin(kind,('QSO','GALAXY'))&(target=='TGT')&(r['ZWARN'][q]==0)&
            (r['COADD_FIBERSTATUS'][q]==0)&np.isfinite(r['DELTACHI2'][q])&(r['DELTACHI2'][q]>100)&
            np.isfinite(r['ZERR'][q])&(r['ZERR'][q]>0)&(r['Z'][q]>np.maximum(.02,5*r['ZERR'][q])))
    label = np.full(len(rr),'',dtype='U8')
    label[j[good]] = kind[good]
    t.update(redrock_row=rr,qualified=label,position_offset_arcsec=np.full(len(rr),np.nan))
    t['position_offset_arcsec'][j] = offsets
    for field in ('ZWARN','DELTACHI2','Z','ZERR'):
        value = np.full(len(rr),np.nan)
        value[j] = r[field][q]
        t[field] = value
    return t


def main():
    require_compute()
    t,inputs = source_roster()
    targets = np.unique(t['targetid'][t['targetid']>0])
    before = os.stat(ZCAT)
    t = qualify(t,scan(targets))
    after = os.stat(ZCAT)
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('Redrock input changed during scan')
    DEST.mkdir(parents=True,exist_ok=True)
    summaries = []
    for region in REGIONS:
        for a,b in ((-np.inf,20),(20,21),(21,22),(22,23)):
            k = (t['region']==region)&(t['r']>=a)&(t['r']<b)
            summaries.append(dict(region=region,r_min=a,r_max=b,raw=int(k.sum()),
                targetid=int((k&(t['targetid']>0)).sum()),primary=int((k&(t['redrock_row']>=0)).sum()),
                qualified_qso=int((k&(t['qualified']=='QSO')).sum()),
                qualified_galaxy=int((k&(t['qualified']=='GALAXY')).sum())))
    atomic_npz(DEST/'qualified.npz',**t)
    csv_rows(DEST/'outcomes.csv',summaries)
    save_json(DEST/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,targets=len(targets),
        zcatalog=ZCAT,size=before.st_size,mtime_ns=before.st_mtime_ns,
        outputs={p.name:sha(p) for p in DEST.iterdir() if p.name!='complete.json'}))


def evaluate():
    require_compute()
    marker = DEST/'complete.json'
    r = json.loads(marker.read_text())
    controls = read_columns(verified(DEST/'qualified.npz',r['outputs']['qualified.npz']),
                            ('TYPE','ls_id','region','qualified','r'))
    inputs = {str(marker.relative_to(ROOT)):sha(marker)}
    rows = []
    for region in REGIONS:
        side = 'north' if region=='north' else 'south'
        folder = SELECTION/side
        score = folder/'complete.json'
        s = json.loads(score.read_text())
        t = read_columns(verified(folder/'scores.npz',s['outputs']['scores.npz']),
                         ('TYPE','ls_id','score','score_original','held_out'))
        folders = EVIDENCE/region
        marker = folders/'complete.json'
        roster = json.loads(marker.read_text())
        p = read_columns(verified(folders/'references.npz',roster['outputs']['references.npz']),
                         ('ls_id','TYPE','nominal_qnear','nominal_numerical_flag'))
        j = unique_join(controls['ls_id'],t['ls_id'])
        k = unique_join(controls['ls_id'],p['ls_id'])
        use = np.flatnonzero((controls['region']==region)&(controls['qualified']!='')&(j>=0)&(k>=0))
        np.testing.assert_array_equal(controls['TYPE'][use],t['TYPE'][j[use]])
        np.testing.assert_array_equal(controls['TYPE'][use],p['TYPE'][k[use]])
        support = t['held_out'][j[use]]&np.isfinite(t['score_original'][j[use]])&np.isfinite(t['score'][j[use]])
        use = use[support]
        inputs[str(score.relative_to(ROOT))] = sha(score)
        inputs[str(marker.relative_to(ROOT))] = sha(marker)
        for a,b in ((-np.inf,20),(20,21),(21,22),(22,23)):
            rr = controls['r'][use]
            for kind in ('QSO','GALAXY'):
                ix = use[(rr>=a)&(rr<b)&(controls['qualified'][use]==kind)]
                gate = p['nominal_qnear'][k[ix]]
                common = np.isfinite(gate)&~p['nominal_numerical_flag'][k[ix]]
                for label in ('score_original','score'):
                    val = t[label][j[ix]]
                    for cut in THRESHOLDS:
                        rows.append(dict(region=region,r_min=a,r_max=b,class_label=kind,method=label,
                            threshold=cut,total=len(ix),passed=int((val>=cut).sum()),
                            with_sed=int(common.sum()),passed_both=int(((val>=cut)&common&(gate>=.90)).sum())))
    out = DEST/'comparison'
    out.mkdir(exist_ok=True)
    csv_rows(out/'held_out.csv',rows)
    save_json(out/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
                                     outputs={'held_out.csv':sha(out/'held_out.csv')}))


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluate',action='store_true')
    (evaluate if p.parse_args().evaluate else main)()
