#!/usr/bin/env python3
"""Compare original and fixed candidate gates on the preserved Redrock controls."""
import json
from pathlib import Path

import joblib
import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from v2_1_near import DEST, ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_catalog import FREEZE, read_frozen
from v3_3_sed import verified
import v3_4_near_forecast as forecast
from v3_5_dr11_inventory import CONFIG
from v3_5_dr11_validate import join_parts
from v3_6_catalog_projection import OUT as PROJECTION
from v3_6_lf_sensitivity import raw_photo
from v3_6_reference_index import OUT as INDEX
from v3_6_vac_distances import ARCHIVE, read_columns
from v3_7_selection_replay import POINTER, probabilities
from v3_8_candidate import gates, read_cache

OUT = ROOT/'data/v3_8/redrock'
ASSESSED = DEST/'nonstellar_assessed_psf.npz'
CLASSES = ('GALAXY','QSO')


def labels():
    a = read_columns(ARCHIVE/'references.npz',('identity','TYPE','rvs_rr_spectype'))
    owners = read_columns(read_cache(INDEX,'baseline_v1_4.npz'),('catalog_row',))['catalog_row']
    b = read_columns(ASSESSED,('identity','TYPE','redrock_SPECTYPE'))
    ids,index,counts = np.unique(a['identity'],return_index=True,return_counts=True)
    loc = np.searchsorted(ids,b['identity'])
    use = np.flatnonzero((loc<len(ids))&(b['TYPE']=='PSF'))
    use = use[(ids[loc[use]]==b['identity'][use])&(counts[loc[use]]==1)]
    assessed = np.full(len(owners),'',dtype='U8')
    assessed[index[loc[use]]] = np.char.strip(b['redrock_SPECTYPE'][use].astype('U'))
    raw = np.char.strip(a['rvs_rr_spectype'].astype('U'))
    label = np.isin(raw,CLASSES)|np.isin(assessed,CLASSES)
    sel = np.flatnonzero(label&(a['TYPE']=='PSF')&(owners>=0))
    _,rev,duplicates = np.unique(owners[sel],return_inverse=True,return_counts=True)
    sel = sel[duplicates[rev]==1]
    key = np.argsort(owners[sel])
    sel = sel[key]
    return owners[sel],raw[sel],assessed[sel],dict(missing_type=int((label&(a['TYPE']=='')).sum()),
        non_psf=int((label&(a['TYPE']!='')&(a['TYPE']!='PSF')).sum()),
        ambiguous_owners=int(duplicates[duplicates>1].sum()))


def main():
    require_compute()
    marker = OUT/'complete.json'
    if marker.exists():
        raise FileExistsError(marker)
    OUT.mkdir(parents=True,exist_ok=True)
    ids,raw,assessed,excluded = labels()
    meta = json.loads((PROJECTION/'complete.json').read_text())
    chunks = []
    for key,h in meta['child_receipts'].items():
        p = read_columns(read_cache(verified(ROOT/key,h).parent,'rows.npz'),
            (*POINTER,'dwarf_selection_score','nominal_qnear','primary','qrf_quantiles_kpc'))
        use = np.isin(p['catalog_row'],ids)
        if use.any():
            chunks.append({k:v[use] for k,v in p.items()})
    p = join_parts(chunks)
    idx = np.searchsorted(ids,p['catalog_row'])
    np.testing.assert_array_equal(ids[idx],p['catalog_row'])
    if len(idx)!=len(ids) or not (p['TYPE']=='PSF').all():
        raise ValueError('Redrock table requires verified and unique native PSFs')
    q = probabilities(p,gates())
    eligible = p['dwarf_selection_score']>=.65
    missing = eligible&~np.isfinite(q)
    if missing.any():
        names = read_columns(read_cache(INDEX,'catalog_rows.npz'),('sweep',))['sweep']
        data = {k:v[missing] for k,v in p.items()}
        photo = raw_photo(data,names,Path(json.loads(CONFIG.read_text())['sweep_root']))
        entry = read_frozen()['models']['nominal']
        m = joblib.load(verified(ROOT/entry['path'],entry['sha256']))
        forecast._STATE = photo,inference_model(m),m.weights@m.classes,False
        for j,i in enumerate(np.flatnonzero(missing)):
            q[i],_,_ = forecast.nominal_row(j)
        forecast._STATE = None
    cand = eligible&(q>=.90)
    np.testing.assert_array_equal(p['primary'],(p['dwarf_selection_score']>=.80)&(q>=.90))
    d = p['qrf_quantiles_kpc'][:,2]
    result = []
    for label,arr in [('Archived MWS labels',raw[idx]),('Qualified DR1 Redrock subset',assessed[idx])]:
        for kind in CLASSES:
            members = arr==kind
            for label1,g in [('primary',p['primary']),('candidate',cand)]:
                use = g&members
                result.append(dict(scope=label,spectype=kind,selection=label1,controls=int(members.sum()),
                    passes=int(use.sum()),median_5_10=int((use&(d>=5)&(d<10)).sum())))
    csv_rows(OUT/'redrock.csv',result)
    atomic_npz(OUT/'controls.npz',ls_id=p['ls_id'],TYPE=p['TYPE'],raw=raw[idx],qualified=assessed[idx],
               primary=p['primary'],candidate=cand,nominal=q)
    files = (INDEX/'complete.json',PROJECTION/'complete.json',ASSESSED,FREEZE,Path(__file__).resolve())
    save_json(marker,dict(inputs={str(x.relative_to(ROOT)):sha(x) for x in files},
        excluded_morphology=excluded,missing_replayed=int(missing.sum()),
        outputs={x.name:sha(x) for x in OUT.iterdir() if x.is_file() and x!=marker}))


if __name__=='__main__':
    main()
