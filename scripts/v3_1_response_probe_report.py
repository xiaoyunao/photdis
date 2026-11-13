#!/usr/bin/env python3
"""All-state matched-response evidence, numerical transfers and seen-data diagnostics."""
import json
import os
import joblib
import numpy as np
from scipy.special import logsumexp,softmax
from v2_1_near import ROOT,load,sha,save_json,csv_rows,atomic_npz
from photdis.stage1 import require_compute
from photdis.v3.model import inference_model
from photdis.v3.controls import optical_magnitudes
from photdis.v3_1.comparison import paired_mean
from photdis.audit import rvs_metallicity


def axes(table,evidence=None):
    mag,_=optical_magnitudes(table);r=mag[:,1]-2.5*np.log10(table['transmission'][:,1]);color=mag[:,0]-mag[:,1]
    result=dict(all=np.full(len(r),'all'),apparent_r=np.array(['lt18','18_20','20_22','ge22'])[np.digitize(r,[18,20,22])],
        intrinsic_gr=np.array(['lt0p3','0p3_0p8','0p8_1p3','ge1p3'])[np.digitize(color,[.3,.8,1.3])],
        optional_bands=np.array([''.join(str(int(x)) for x in b) for b in table['band_mask'][:,[2,4,5]]]),sky=table['cell'].astype(str))
    if evidence is not None:
        feh,valid=rvs_metallicity(evidence['rvs_feh'],evidence['rvs_teff'],evidence['rvs_logg'])
        valid&=evidence['rvs_parameter_quality']&np.isfinite(evidence['rvs_feh_err'])&(evidence['rvs_feh_err']>0)
        result.update(reference_class=evidence['class_label'].astype(str),reference_feh=np.where(valid,np.array(['poor','middle','rich'])[np.digitize(feh,[-1.5,-.5])],'missing'))
    return result


def main():
    require_compute();base=ROOT/'data/v3_1/response_probes';out=base/'report';out.mkdir(exist_ok=True)
    draws=list(range(32))+list(range(1000,1032));missing=[];completion=[];inventory=[]
    paths=[base/'complete.json',base/'target_photometry.npz',base/'development_photometry.npz',base/'development_evidence.npz',
        base/'development_design.json',ROOT/'data/v3_0/sed_fit.joblib',ROOT/'scripts/v3_1_response_probe_report.py',
        ROOT/'docs/v3_1_response_probe_report_protocol.md',ROOT/'src/photdis/v3_1/comparison.py']
    inputs={str(p.relative_to(ROOT)):sha(p) for p in paths}
    for draw in draws:
        p=base/f'draw_{draw:04}/complete.json'
        if not p.exists():missing.append(draw);continue
        d=json.loads(p.read_text());assert d['all_PSF'] and d['global_draw_id']==draw
        inputs[str(p.relative_to(ROOT))]=sha(p);completion.append(d)
        inventory.append(dict(draw=draw,library=d['library'],target_passed=d['target_passed'],development_passed=d['development_passed'],
            target_rows=d['target_rows'],development_rows=d['development_rows'],target_training_overlap=d['target_training_overlap'],
            target_conditional_log_evidence=d['target_conditional_log_evidence']))
    csv_rows(out/'inventory.csv',inventory)
    if missing:
        save_json(out/'progress.json',dict(inputs=inputs,missing=missing,complete=False,job=os.getenv('SLURM_JOB_ID')))
        raise RuntimeError('Incomplete registered response set; inventory retained, no posterior normalization')
    prep=json.loads(paths[0].read_text())
    for p in paths[1:5]:assert sha(p)==prep['outputs'][p.name]
    target,development,evidence=map(load,paths[1:4]);design=json.loads(paths[4].read_text())
    nominal=inference_model(joblib.load(paths[5]));frequency=nominal.weights@nominal.classes
    np.testing.assert_array_equal(development['identity'],evidence['identity'])
    np.testing.assert_array_equal(development['identity'],[r['identity'] for r in design])
    assert (target['TYPE']=='PSF').all() and (development['TYPE']=='PSF').all()
    design_weight=1/np.array([r['inclusion_probability'] for r in design]);cells=development['cell']
    targets={};devs={};conditional=np.zeros((64,4,128));flags=[]
    for index,(draw,d) in enumerate(zip(draws,completion)):
        folder=base/f'draw_{draw:04}';targets[draw]=[];devs[draw]=[]
        for cohort,table,records in [('target',target,targets[draw]),('development',development,devs[draw])]:
            assert len(table['identity'])==d[cohort+'_rows']
            for row,identity in enumerate(table['identity']):
                name=f'{cohort}/object_{int(identity)}.json';p=folder/name;assert sha(p)==d['objects'][name]
                r=json.loads(p.read_text());assert r['inputs']==d['inputs'] and r['TYPE']=='PSF' and r['identity']==identity and r['row']==row
                payload=p.with_suffix('.npz');assert sha(payload)==r['sha256'];records.append(r)
                if not r['passed']:flags.append(dict(draw=draw,cohort=cohort,identity=int(identity),native_passed=r['native_passed'],score_passed=r['score_passed'],entry_passed=r['entry_passed']))
                if cohort=='target':
                    data=load(payload);last=r['selection']['attempts'][-1]['draws'];previous=r['selection']['attempts'][-2]['draws']
                    alpha=[float(frequency@data[f'entry_{n}_{s}'][0,:,0]) for n,s in [(last,320011),(last,320012),(previous,320011)]]
                    assert min(alpha)>0
                    conditional[index,:,row]=[r['fine']['log_evidence']-np.log(alpha[0]),r['coarse']['log_evidence']-np.log(alpha[0]),
                        r['fine']['log_evidence']-np.log(alpha[1]),r['fine']['log_evidence']-np.log(alpha[2])]
        np.testing.assert_allclose(conditional[index,0].sum(),d['target_conditional_log_evidence'],atol=1e-8,rtol=0)
        print('RESPONSE REPORT',draw,flush=True)
    weights=[];sets=[];numerics=[];mixtures={};log_evidence=conditional.sum(axis=-1)
    for label,indices in [('original',np.arange(32)),('new',np.arange(32,64)),('pooled',np.arange(64))]:
        allpass=all(completion[i]['target_passed'] for i in indices)
        item=dict(set=label,states=len(indices),target_inputs_passed=allpass,accepted=False)
        if allpass:
            value=log_evidence[indices];w=softmax(value[:,0]);logz=float(logsumexp(value[:,0])-np.log(len(indices)));good=True
            for j,name in enumerate(['native_coarse','entry_second_seed','entry_previous_level'],1):
                alt=softmax(value[:,j]);delta=float(abs(w-alt).max());zd=float(logsumexp(value[:,j])-np.log(len(indices))-logz)
                passed=delta<.01 and abs(zd)<.05;good&=passed
                numerics.append(dict(set=label,alternative=name,maximum_weight_delta=delta,L1_weight_delta=float(abs(w-alt).sum()),
                    log_evidence_delta=zd,alternative_ESS=float(1/(alt@alt)),passed=bool(passed)))
            item.update(accepted=bool(good),log_evidence=logz,ESS=float(1/(w@w)),maximum_weight=float(w.max()),
                top_draw=draws[indices[w.argmax()]],original_library_mass=float(w[indices<32].sum()),new_library_mass=float(w[indices>=32].sum()))
            lw=value[:,0]-logsumexp(value[:,0]);mixtures[label]=(indices,lw,bool(good))
            weights.extend(dict(set=label,draw=draws[i],posterior_weight=float(v),log_posterior_weight=float(l),numerically_accepted=bool(good)) for i,v,l in zip(indices,w,lw))
        sets.append(item)
    csv_rows(out/'response_weights.csv',weights);csv_rows(out/'weight_numerics.csv',numerics);save_json(out/'sets.json',sets);save_json(out/'numerical_flags.json',flags)
    atomic_npz(out/'conditional_evidence.npz',draw_ids=np.array(draws),identity=target['identity'],TYPE=target['TYPE'],
        variants=np.array(['fine','native_coarse','entry_second_seed','entry_previous_level']),per_object_log_evidence=conditional)
    decomposition=[]
    for axis,groups in axes(target).items():
        for group in sorted(set(groups)):
            mask=groups==group
            for j,draw in enumerate(draws):
                decomposition.append(dict(draw=draw,axis=axis,group=group,n=int(mask.sum()),conditional_log_evidence=float(conditional[j,0,mask].sum()),
                    gain_over_response12=float((conditional[j,0,mask]-conditional[12,0,mask]).sum())))
        for j in range(64):np.testing.assert_allclose(sum(r['conditional_log_evidence'] for r in decomposition if r['axis']==axis and r['draw']==draws[j]),log_evidence[j,0],atol=1e-8,rtol=0)
    csv_rows(out/'target_decomposition.csv',decomposition)
    paired=[];baseline=devs[12]
    for draw in draws:
        rr=devs[draw]
        for axis,groups in axes(development,evidence).items():
            for group in sorted(set(groups)):
                rows=np.flatnonzero(groups==group)
                def append(metric,values,selected,source='none'):
                    paired.append(dict(draw=draw,baseline=12,axis=axis,group=group,source=source,metric=metric,
                        all_numerically_passed=all(rr[i]['passed'] and baseline[i]['passed'] for i in selected),
                        **paired_mean(values,design_weight[selected],cells[selected])))
                append('mu50_shift',[rr[i]['fine']['mu'][2]-baseline[i]['fine']['mu'][2] for i in rows],rows)
                append('qnear_shift',[rr[i]['fine']['qnear']-baseline[i]['fine']['qnear'] for i in rows],rows)
                for source in ['primary','specdis','spdist','gaia_precise']:
                    selected=np.array([i for i in rows if source in rr[i]['fine']['references']],int)
                    if not len(selected):continue
                    values=[(rr[i]['fine'],baseline[i]['fine']) for i in selected]
                    for v,b in values:assert v['references'][source]['mu']==b['references'][source]['mu']
                    append('reference_logscore_gain',[v['references'][source]['logscore']-b['references'][source]['logscore'] for v,b in values],selected,source)
                    append('absolute_mu_error_change',[abs(v['mu'][2]-v['references'][source]['mu'])-abs(b['mu'][2]-b['references'][source]['mu']) for v,b in values],selected,source)
                    append('severe_mu_error_change',[float(abs(v['mu'][2]-v['references'][source]['mu'])>1)-float(abs(b['mu'][2]-b['references'][source]['mu'])>1) for v,b in values],selected,source)
                selected=np.array([i for i in rows if 'gaia' in rr[i]['fine']],int)
                if len(selected):
                    append('signed_gaia_logscore_gain',[rr[i]['fine']['gaia']['log_score']-baseline[i]['fine']['gaia']['log_score'] for i in selected],selected,'signed_gaia')
                    append('signed_gaia_PIT_tail_change',[float(rr[i]['fine']['gaia']['pit']<.025 or rr[i]['fine']['gaia']['pit']>.975)-float(baseline[i]['fine']['gaia']['pit']<.025 or baseline[i]['fine']['gaia']['pit']>.975) for i in selected],selected,'signed_gaia')
    # Split by axis to keep durable compact tables below source-control limits.
    for axis in axes(development,evidence):csv_rows(out/f'paired_{axis}.csv',[r for r in paired if r['axis']==axis])
    mixture_objects=[];mixture_scores=[]
    for label,(indices,lw,accepted) in mixtures.items():
        for row,identity in enumerate(development['identity']):
            records=[devs[draws[i]][row] for i in indices];fine=[r['fine'] for r in records]
            numerical=accepted and all(r['native_passed'] and r['score_passed'] for r in records)
            joint=np.einsum('s,scd->cd',np.exp(lw),np.array([v['class_distance'] for v in fine]))
            np.testing.assert_allclose(joint.sum(),1,atol=1e-10,rtol=0)
            common=dict(set=label,identity=int(identity),TYPE='PSF',reference_class=int(evidence['class_label'][row]),
                cell=int(cells[row]),design_weight=float(design_weight[row]),numerically_accepted=bool(numerical))
            mixture_objects.append(dict(**common,qnear=float(joint[:,:3].sum()),
                **{f'class{c}_distance{d}':float(joint[c,d]) for c in range(3) for d in range(6)}))
            for source in baseline[row]['fine']['references']:
                refs=[v['references'][source] for v in fine];b=baseline[row]['fine']['references'][source]
                assert all(r['mu']==b['mu'] and r['error']==b['error'] for r in refs)
                score=float(logsumexp(lw+np.array([r['logscore'] for r in refs])))
                mixture_scores.append(dict(**common,source=source,logscore=score,gain_over_response12=score-b['logscore']))
            if 'gaia' in baseline[row]['fine']:
                score=float(logsumexp(lw+np.array([v['gaia']['log_score'] for v in fine])))
                pit=float(np.exp(lw)@np.array([v['gaia']['pit'] for v in fine]))
                mixture_scores.append(dict(**common,source='signed_gaia',logscore=score,gain_over_response12=score-baseline[row]['fine']['gaia']['log_score'],pit=pit))
    csv_rows(out/'mixture_objects.csv',mixture_objects);csv_rows(out/'mixture_scores.csv',mixture_scores)
    save_json(out/'complete.json',dict(inputs=inputs,outputs={p.name:sha(p) for p in out.iterdir() if p.suffix in ('.csv','.npz') or p.name in ('sets.json','numerical_flags.json')},
        states=64,target_rows=128,development_rows=256,numerical_flags=len(flags),sets=sets,all_PSF=True,
        job=os.getenv('SLURM_JOB_ID'),scope='Matched fixed-prior conditional probes and seen-development pairs, not26kpopulation response evidence or independent calibration. All64states retained; failed sets have no accepted weights.'))


if __name__=='__main__':main()
