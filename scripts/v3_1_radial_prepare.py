#!/usr/bin/env python3
"""Preserve all112 old catalogs and construct the64 registered narrow controls."""
import json
import os
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,atomic_npz
from photdis.stage1 import require_compute
from photdis.v3.simulation import empirical_library
from photdis.v3_1.simulation import draw_narrow


def main():
    require_compute(); d=ROOT/'data/v3_0';out=ROOT/'data/v3_1/radial_catalogs';out.mkdir(parents=True,exist_ok=True)
    paths=[d/'cohorts/simulations.npz',d/'simulations/truth.npz',d/'simulations/catalogs.json',d/'simulations/prepared.json',
        d/'empirical_transfer_bank.npz',d/'cohorts/application.npz',ROOT/'docs/v3_1_radial_protocol.md',
        ROOT/'scripts/v3_1_radial_prepare.py',ROOT/'src/photdis/v3_1/simulation.py',ROOT/'src/photdis/v3/simulation.py']
    inputs={str(p.relative_to(ROOT)):sha(p) for p in paths};marker=out/'complete.json'
    if marker.exists():
        record=json.loads(marker.read_text());assert record['inputs']==inputs
        for name,digest in record['outputs'].items():assert sha(out/name)==digest
        return
    frozen=json.loads(paths[3].read_text())
    for p in paths[:3]:assert sha(p)==frozen['outputs'][str(p.relative_to(d))]
    oldphot,oldtruth=load(paths[0]),load(paths[1]);catalogs=json.loads(paths[2].read_text())
    assert len(catalogs)==112 and len(np.unique(oldtruth['scenario']))==14
    bank,context=load(paths[4]),load(paths[5]);assert (bank['TYPE']=='PSF').all() and (context['TYPE']=='PSF').all() and (oldphot['TYPE']=='PSF').all()
    library=empirical_library(bank,seed=311501);library['TYPE']=bank['TYPE'].copy()
    atomic_npz(out/'independent_empirical_library.npz',**library)
    phot=[oldphot];truth=[oldtruth];fields=np.unique(context['field']);index=112
    for distance in [2.,4.,8.,16.]:
        for width in [.08,.25]:
            scenario=f'narrow_{distance:g}kpc_w{width:g}'
            for repeat in range(8):
                field=fields[repeat%len(fields)];local={k:v[context['field']==field] for k,v in context.items()}
                p,t,info=draw_narrow(library,local,distance,width,n=128,seed=311600+index,first_identity=128*index)
                t.update(identity=p['identity'],catalog=np.full(128,index,'i4'),scenario=np.full(128,scenario),repeat=np.full(128,repeat,'i2'))
                rng=np.random.default_rng(312600+index);t['parallax_error']=np.exp(rng.uniform(np.log(.02),np.log(.3),128))
                t['parallax']=10**((10-t['mu'])/5)+rng.normal(size=128)*t['parallax_error']
                assert set(p)==set(oldphot) and set(t)==set(oldtruth)
                phot.append(p);truth.append(t);catalogs.append(dict(catalog=index,scenario=scenario,repeat=repeat,field=int(field),
                    unique_template_ids=len(np.unique(t['template_identity'])),common_library_seed=311501,**info));index+=1
    p={k:np.concatenate([x[k] for x in phot]) for k in oldphot};t={k:np.concatenate([x[k] for x in truth]) for k in oldtruth}
    np.testing.assert_array_equal(p['identity'],t['identity']);assert len(np.unique(p['identity']))==len(p['identity'])
    assert (p['TYPE']=='PSF').all() and index==176
    atomic_npz(out/'photometry.npz',**p);atomic_npz(out/'truth.npz',**t);save_json(out/'catalogs.json',catalogs)
    save_json(marker,dict(inputs=inputs,outputs={p.name:sha(p) for p in out.iterdir() if p.name!='complete.json'},
        original_catalogs=112,new_catalogs=64,total_catalogs=176,rows=len(p['identity']),job=os.getenv('SLURM_JOB_ID'),
        source_PSF_exclusions=0,context_PSF_exclusions=0,missing_TYPE=0,
        role='Conditional transfer recovery; teacher-scale empirical source uncertainty is not new absolute truth. All old catalog identities and arrays preserved as prefix.'))
    print('V3.1 RADIAL CATALOGS COMPLETE',len(p['identity']),flush=True)


if __name__=='__main__':main()
