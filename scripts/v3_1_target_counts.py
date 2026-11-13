#!/usr/bin/env python3
"""Conditional shared joint counts and actual numerical-matrix transfers."""
import argparse
import json
import os
import numpy as np
from scipy.special import logsumexp,softmax
from v2_1_near import ROOT,load,sha,save_json,atomic_npz,csv_rows
from photdis.stage1 import require_compute
from photdis.v3.population import SharedCounts


def weight_audit(logratio):
    weight=softmax(logratio);ess=float(1/(weight@weight))
    halves=[float(logsumexp(logratio[j::2])-np.log(len(logratio[j::2]))) for j in (0,1)]
    delta=abs(halves[0]-halves[1])
    return weight,dict(importance_ess=ess,draws=len(weight),half_log_normalizer_delta=delta,
        passed=bool(ess>.5*len(weight) and delta<.05))


def main():
    require_compute();p=argparse.ArgumentParser();p.add_argument('--case',choices=['broad','candidate','holdout'],required=True)
    p.add_argument('--correction',type=int,required=True);p.add_argument('--draws',type=int,choices=[512,2048,8192],default=512);a=p.parse_args();assert 0<=a.correction<34
    root=ROOT/'data/v3_1';fit=root/'target_fits'/a.case/f'correction_{a.correction:02}_s{a.draws}';matrix=root/'target_matrices';jointroot=root/'target_joint'
    paths=[fit/'complete.json',fit/'population.npz',matrix/'complete.json',matrix/'matrices.npz',
        ROOT/'scripts/v3_1_target_counts.py',ROOT/'src/photdis/v3/population.py',ROOT/'docs/v3_1_target_counts_protocol.md',ROOT/'docs/v3_1_target_numerical_transfer.md']
    paths += [jointroot/f'shard_{i}/complete.json' for i in range(4)]
    inputs={str(p.relative_to(ROOT)):sha(p) for p in paths};fd=json.loads(paths[0].read_text());md=json.loads(paths[2].read_text())
    assert fd['sampler_passed'] and md['passed'] and fd['inputs'][str(paths[3].relative_to(ROOT))]==sha(paths[3])
    assert sha(paths[1])==fd['outputs']['population.npz'] and sha(paths[3])==md['outputs']['matrices.npz']
    data=load(paths[3]);population=load(paths[1]);rows=population['rows'];ids=population['identity'];theta=population['theta'];draw_ids=population['draw_ids'];n=len(rows);s=len(theta)
    np.testing.assert_array_equal(ids,data['identity'][rows]);assert (population['TYPE']=='PSF').all() and (data['TYPE'][rows]=='PSF').all()
    joint_done=[json.loads(p.read_text()) for p in paths[-4:]];assert all(d['passed'] and d['all_PSF'] for d in joint_done)
    out=fit/'joint_counts';out.mkdir(exist_ok=True);freeze=out/'inputs.json'
    if freeze.exists():assert json.loads(freeze.read_text())==inputs
    else:save_json(freeze,inputs)
    basejoint=np.empty((n,3,5,6));finejoint=np.full_like(basejoint,np.nan);source_hashes={}
    for j,row in enumerate(rows):
        name=f'object_{int(ids[j])}.json';path=jointroot/name;assert sha(path)==joint_done[row%4]['objects'][name]
        record=json.loads(path.read_text());assert sha(path.with_suffix('.npz'))==record['sha256'];value=load(path.with_suffix('.npz'))
        assert value['identity'][0]==ids[j] and value['TYPE'][0]=='PSF' and value['row'][0]==row
        assert float(value['logscale'])==data['logscale'][row]
        basejoint[j]=value['coarse_joint_response'][a.correction]
        if record['fine_audited']:finejoint[j]=value['fine_joint_response'][a.correction]
        np.testing.assert_allclose(basejoint[j].sum(axis=-1),data['response'][row,a.correction],atol=1e-12,rtol=1e-8)
        source_hashes[str(path.relative_to(ROOT))]=sha(path)
    fine_mask=data['fine_audited'][rows];assert np.isfinite(finejoint[fine_mask]).all()
    response=data['response'][rows,a.correction];fine_response=data['fine_response'][rows,a.correction]
    label='candidate' if a.case=='candidate' else 'broad'
    entry=data['selection_'+label][rows,a.correction]
    alternatives={'entry_seed':data['selection_second_'+label][rows,a.correction],
        'entry_level':data['selection_previous_'+label][rows,a.correction]}
    fields=data['field'][rows];case_weight=population['normalized_case_weights'];logratio={k:np.zeros(s) for k in [*alternatives,'native_fine']}
    physical={k:bool(np.isfinite(v).all() and (v>=0).all() and (v<=1).all()) for k,v in alternatives.items()}
    for fi,field in enumerate(population['fields']):
        all_use=np.flatnonzero(fields==field);flat=theta[:,fi].reshape(s,-1)
        for start in range(0,len(all_use),256):
            use=all_use[start:start+256];base=flat@entry[use].reshape(len(use),-1).T;assert (base>0).all()
            for name,alt in alternatives.items():
                other=flat@alt[use].reshape(len(use),-1).T;assert (other>0).all()
                logratio[name]+=(np.log(base)-np.log(other))@case_weight[use]
            audit=use[fine_mask[use]]
            if len(audit):
                coarse=flat@response[audit].reshape(len(audit),-1).T;fine=flat@fine_response[audit].reshape(len(audit),-1).T
                assert (coarse>0).all() and (fine>0).all();logratio['native_fine']+=(np.log(fine)-np.log(coarse))@case_weight[audit]
    weights={'base':np.full(s,1/s)};audits={}
    for name,ratio in logratio.items():
        weights[name],audits[name]=weight_audit(ratio)
        audits[name]['physical_entries']=physical.get(name,True);audits[name]['passed'] &= physical.get(name,True)
    counts={name:SharedCounts(draw_ids,18,w,seed=319931) for name,w in weights.items()}
    means={name:np.empty((n,3,6)) for name in weights}
    for fi,field in enumerate(population['fields']):
        all_use=np.flatnonzero(fields==field)
        for start in range(0,len(all_use),32):
            use=all_use[start:start+32]
            value=np.einsum('scd,ncdb->sncb',theta[:,fi],basejoint[use],optimize=True)
            value/=value.sum(axis=(2,3),keepdims=True)
            refined=value.copy();audit=np.flatnonzero(fine_mask[use])
            if len(audit):
                fine=np.einsum('scd,ncdb->sncb',theta[:,fi],finejoint[use[audit]],optimize=True)
                fine/=fine.sum(axis=(2,3),keepdims=True);refined[:,audit]=fine
            for name,accumulator in counts.items():
                probability=refined if name=='native_fine' else value
                means[name][use]=accumulator.add(probability.reshape(s,len(use),18),draw_ids).reshape(len(use),3,6)
    results={name:acc.result() for name,acc in counts.items()};checks=[]
    for name,result in results.items():
        covariance=result['covariance'];conserve=bool(abs(result['mean'].sum()-n)<1e-7 and abs(covariance.sum(axis=1)).max()<1e-6 and np.linalg.eigvalsh(covariance).min()>-1e-6)
        assert conserve
        atomic_npz(out/f'{name}_counts.npz',**result)
        prob_delta=float(abs(means[name]-means['base']).max());count_delta=float(abs(result['mean']-results['base']['mean']).max())
        passed=result['marginal_half_delta']<.005 and prob_delta<.003 and count_delta<.5 and audits.get(name,{'passed':True})['passed']
        checks.append(dict(variant=name,maximum_probability_delta=prob_delta,maximum_count_mean_delta=count_delta,
            marginal_half_delta=result['marginal_half_delta'],count_covariance_conserved=conserve,**audits.get(name,{})))
        # A transfer may have adequate ESS but still fail the product gates.
        checks[-1]['passed']=bool(passed)
    atomic_npz(out/'per_star.npz',identity=ids,TYPE=population['TYPE'],rows=rows,field=fields,fine_audited=fine_mask,
        **{name+'_probability':v for name,v in means.items()})
    atomic_npz(out/'numerical_weights.npz',draw_ids=draw_ids,**{name+'_weight':w for name,w in weights.items()},**{name+'_logratio':v for name,v in logratio.items()})
    csv_rows(out/'checks.csv',checks);save_json(out/'source_markers.json',source_hashes)
    save_json(out/'complete.json',dict(inputs=inputs,outputs={p.name:sha(p) for p in out.iterdir() if p.suffix in ('.npz','.csv') or p.name=='source_markers.json'},
        case=a.case,correction=a.correction,draws=a.draws,objects=n,fine_audited=int(fine_mask.sum()),all_PSF=True,
        sampler_passed=True,bridge_passed=fd['bridge_passed'],passed=all(r['passed'] for r in checks),checks=checks,job=os.getenv('SLURM_JOB_ID'),
        scope='Conditional observed-sample shared class x true-distance counts. Native fine transfer audits only the registered subset; seed/level entry transfers use all fitted objects. Conditional categorical/shared covariance separate. No parent expansion, final response ensemble, high-q selection, or new-field validation claimed.'))


if __name__=='__main__':main()
