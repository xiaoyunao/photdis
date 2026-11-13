#!/usr/bin/env python3
"""Authenticate all structure states without discarding numerical failures."""
import json
import os
import numpy as np
from scipy.special import logsumexp
from v2_1_near import ROOT,load,sha,save_json,atomic_npz,csv_rows
from photdis.stage1 import require_compute


def main():
    require_compute();root=ROOT/'data/v3_1/structure';out=root/'matrices';out.mkdir(exist_ok=True)
    paths=[root/'complete.json',root/'photometry.npz',ROOT/'scripts/v3_1_structure_assemble.py',
        ROOT/'docs/v3_1_structure_protocol.md',ROOT/'docs/v3_1_structure_numerical_protocol.md']
    paths += [root/f'{kind}/draw_{draw:02}/complete.json' for kind in ('native','native_refined','entry','entry_refined') for draw in range(32)]
    missing=[str(p.relative_to(ROOT)) for p in paths if not p.exists()]
    if missing:
        save_json(out/'progress.json',dict(missing=missing,complete=False,job=os.getenv('SLURM_JOB_ID')))
        raise RuntimeError('All32response states and all512objects required')
    inputs={str(p.relative_to(ROOT)):sha(p) for p in paths};prep=json.loads(paths[0].read_text())
    assert prep['all_PSF'] and sha(paths[1])==prep['outputs']['photometry.npz'];table=load(paths[1]);assert len(table['identity'])==512 and (table['TYPE']=='PSF').all()
    freeze=out/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==inputs
    else:save_json(freeze,inputs)
    logresponse=entry=None;common=means=sigmas=None;flags=[];sources={}
    for draw in range(32):
        native=root/f'native/draw_{draw:02}';selection=root/f'entry/draw_{draw:02}';refined=root/f'native_refined/draw_{draw:02}'
        nd=json.loads((native/'complete.json').read_text());ed=json.loads((selection/'complete.json').read_text())
        rd=json.loads((refined/'complete.json').read_text());assert rd['n']==512 and rd['all_PSF'] and rd['response_draw']==draw
        assert rd['inputs'][str((native/'complete.json').relative_to(ROOT))]==sha(native/'complete.json')
        entry_refined=root/f'entry_refined/draw_{draw:02}';erd=json.loads((entry_refined/'complete.json').read_text())
        assert erd['n']==512 and erd['all_PSF'] and erd['response_draw']==draw
        assert erd['inputs'][str((selection/'complete.json').relative_to(ROOT))]==sha(selection/'complete.json')
        assert nd['n']==ed['n']==512 and nd['all_PSF'] and ed['all_PSF'] and nd['response_draw']==ed['response_draw']==draw
        for row,identity in enumerate(table['identity']):
            name=f'object_{int(identity)}.json';p=native/name;assert sha(p)==nd['objects'][name]
            n=json.loads(p.read_text());assert n['inputs']==nd['inputs'] and n['identity']==identity and n['row']==row and n['TYPE']=='PSF'
            payload=p.with_suffix('.npz');assert sha(payload)==n['sha256']
            refmarker=refined/f'object_{int(identity)}/complete.json';assert sha(refmarker)==rd['objects'][str(refmarker.relative_to(refined))]
            refined_record=json.loads(refmarker.read_text());assert refined_record['inputs']==rd['inputs'] and refined_record['identity']==identity and refined_record['TYPE']=='PSF'
            assert refined_record['source_marker_sha256']==sha(p) and refined_record['source_payload_sha256']==sha(payload)
            selected=ROOT/refined_record['selected_payload'];assert sha(selected)==refined_record['selected_sha256'];a=load(selected)
            sources[str(refmarker.relative_to(ROOT))]=sha(refmarker)
            sources[str(p.relative_to(ROOT))]=sha(p)
            if logresponse is None:
                common=a['common_mu'];means=a['radial_means'];sigmas=a['radial_sigmas']
                logresponse=np.empty((2,2,32,512,len(means)));entry=np.empty((3,32,512,len(means)))
            np.testing.assert_array_equal(a['common_mu'],common);np.testing.assert_array_equal(a['radial_means'],means);np.testing.assert_array_equal(a['radial_sigmas'],sigmas)
            assert a['identity'][0]==identity and a['TYPE'][0]=='PSF';frequency=a['class_frequency'];assert np.isclose(frequency.sum(),1)
            for gi,grid in enumerate(('coarse','fine')):
                for bi,branch in enumerate(('full','gaussian')):
                    logresponse[gi,bi,draw,row]=logsumexp(a[f'{grid}_{branch}_logresponse']+np.log(frequency)[:,None],axis=0)
            q=selection/f'context_{row}/complete.json';assert sha(q)==ed['contexts'][str(q.relative_to(selection))]
            e=json.loads(q.read_text());assert e['inputs']==ed['inputs'] and e['identity']==identity and e['row']==row and e['TYPE']=='PSF'
            sources[str(q.relative_to(ROOT))]=sha(q)
            eq=entry_refined/f'context_{row}/complete.json';assert sha(eq)==erd['contexts'][str(eq.relative_to(entry_refined))]
            er=json.loads(eq.read_text());assert er['inputs']==erd['inputs'] and er['source_marker_sha256']==sha(q) and er['identity']==identity and er['row']==row and er['TYPE']=='PSF'
            sources[str(eq.relative_to(ROOT))]=sha(eq)
            for branch in er['results']:
                label=branch['branch'];cols=slice(0,5) if label=='background' else slice(5,None)
                for variant,name in enumerate(('base','second_seed','previous_level')):
                    selected=branch['selected'][name];path=ROOT/selected['path'];assert sha(path)==selected['sha256'];b=load(path)
                    np.testing.assert_array_equal(b['means'],means[cols]);np.testing.assert_array_equal(b['sigmas'],sigmas[cols])
                    entry[variant,draw,row,cols]=frequency@b['alpha']
            flags.append(dict(draw=draw,row=row,identity=int(identity),TYPE='PSF',original_native_passed=n['passed'],
                native_passed=refined_record['passed'],original_entry_passed=e['passed'],entry_passed=er['passed'],passed=refined_record['passed'] and er['passed']))
        print('STRUCTURE ASSEMBLY',draw,flush=True)
    assert np.isfinite(logresponse).all() and np.isfinite(entry).all() and (entry>=0).all()
    atomic_npz(out/'matrices.npz',identity=table['identity'],TYPE=table['TYPE'],radius_deg=table['radius_deg'],response_ids=np.arange(32),
        common_mu=common,radial_means=means,radial_sigmas=sigmas,grid_names=np.array(['coarse','fine']),branch_names=np.array(['full','gaussian']),
        entry_variants=np.array(['base','second_seed','previous_level']),logresponse=logresponse,entry=entry)
    csv_rows(out/'status.csv',flags);save_json(out/'source_markers.json',sources)
    save_json(out/'complete.json',dict(inputs=inputs,outputs={n:sha(out/n) for n in ['matrices.npz','status.csv','source_markers.json']},
        states=32,n=512,all_PSF=True,passed=all(r['passed'] for r in flags),failed_state_objects=sum(not r['passed'] for r in flags),
        job=os.getenv('SLURM_JOB_ID'),scope='All32response/512object structure matrices, no source/response deletion. Exact and fair Gaussian likelihoods share generative entry and all global priors. Posterior/quadrature/count checks still required.'))


if __name__=='__main__':main()
