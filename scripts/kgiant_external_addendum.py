#!/usr/bin/env python3
"""Reproducible scope/reference addendum; does not alter frozen predictions."""
import json,hashlib
from pathlib import Path
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

ROOT=Path(__file__).resolve().parents[1]

def main():
    inv=json.loads((ROOT/'manifests/kgiant_external_inventory_v1.json').read_text())
    pred=json.loads((ROOT/'manifests/kgiant_external_predictions_v1.json').read_text())
    refs=json.loads((ROOT/'manifests/kgiant_bv21_reference_v1.json').read_text())
    ref=next(x for x in refs['rows'] if x['name']=='Ter 7');shift=17.05-ref['mu']
    targets=[]
    for r in inv['rows']:
        b=float(SkyCoord(r['ra']*u.deg,r['dec']*u.deg).galactic.b.deg)
        targets.append(dict(name=r['name'],cluster=r['cluster'],galactic_b=b,old_latitude_scope=abs(b)>30,
            old_dust_scope=r.get('old_dust_scope'),quality_raw=r.get('quality_raw'),photometry=r.get('photometry')))
    support={x['name']:x['support'] for x in pred['audit']};rows=[]
    for r in pred['rows']:
        row={k:r[k] for k in ('name','teacher','model','effective_n','delta_mu_median','spdist_temperature_extrapolation')}
        row.update(delta_mu_BV21=r['delta_mu_median']+shift,W68=r['quantiles_mu'][3]-r['quantiles_mu'][1],box_excursions=[])
        if r['model']!='prior':
            a=support[r['name']][r['teacher']+'_'+r['model']]
            for c,v,lo,hi in zip(('gr','rz','zw1'),a['query_colors'],a['train_min'],a['train_max']):
                if v<lo or v>hi:row['box_excursions'].append(dict(color=c,direction='below' if v<lo else 'above',magnitude=float(lo-v if v<lo else v-hi)))
        rows.append(row)
    out=dict(reference=ref,registered_mu=17.05,residual_shift=shift,targets=targets,rows=rows,
         interpretation='3 stars, 1 external system, 18 configurations; no independent-error combination of literature scales',
         source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ('manifests/kgiant_external_inventory_v1.json','manifests/kgiant_external_predictions_v1.json','manifests/kgiant_bv21_reference_v1.json')})
    (ROOT/'manifests/kgiant_external_addendum_v1.json').write_text(json.dumps(out,indent=2)+'\n')
    print('BV21 shift',shift,'b',[(x['name'],x['galactic_b']) for x in targets])
if __name__=='__main__':main()
