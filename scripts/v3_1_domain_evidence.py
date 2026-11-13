#!/usr/bin/env python3
"""Evidence availability and numerical support, not automatic domain certification."""
import json
import os
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,atomic_npz,csv_rows
from photdis.stage1 import require_compute
from photdis.v3_1.freeze import verify_freeze
from photdis.v3_1.use_domains import (N_CELLS,cell_descriptors,distance_availability,
    missing_reference_mass,computational_status)


def main():
    require_compute(); root=ROOT/'data/v3_1'; out=root/'domain_evidence'; out.mkdir(exist_ok=True)
    freeze=root/'final_validation/freeze.json'; verify_freeze(ROOT,freeze)
    paths=[freeze,root/'domain_grid/complete.json',root/'final_validation/report/complete.json',
        root/'target_ensemble/candidate/complete.json',root/'target_ensemble/candidate/states.json',
        root/'response_probes/report/complete.json',ROOT/'scripts/v3_1_domain_evidence.py',
        ROOT/'src/photdis/v3_1/use_domains.py',ROOT/'docs/v3_1_use_domain_protocol.md']
    inputs={str(p.relative_to(ROOT)):sha(p) for p in paths}
    grid_done=json.loads(paths[1].read_text()); grid={}
    assert grid_done['all_PSF'] and not grid_done['references_read']
    for cohort in ['application','field','common']:
        p=root/f'domain_grid/{cohort}.npz'; assert sha(p)==grid_done['outputs'][p.name]
        grid[cohort]=load(p); assert (grid[cohort]['TYPE']=='PSF').all()
    cells=cell_descriptors(); evidence={}; score_pass={}; reference_rows=[]; available={}
    for cohort in ['field','common']:
        source=root/f'final_validation/{cohort}'; done=json.loads((source/'complete.json').read_text())
        inputs[str((source/'complete.json').relative_to(ROOT))]=sha(source/'complete.json')
        assert sha(source/'evidence.npz')==done['outputs']['evidence.npz']; evidence[cohort]=load(source/'evidence.npz')
        np.testing.assert_array_equal(evidence[cohort]['identity'],grid[cohort]['identity'])
        assert (evidence[cohort]['TYPE']=='PSF').all()
        folder=root/f'final_validation/scores/{cohort}'; sd=json.loads((folder/'complete.json').read_text())
        inputs[str((folder/'complete.json').relative_to(ROOT))]=sha(folder/'complete.json'); passed=[]
        for identity in grid[cohort]['identity']:
            name=f'object_{int(identity)}.json'; p=folder/name; assert sha(p)==sd['objects'][name]
            record=json.loads(p.read_text()); assert record['identity']==identity and record['TYPE']=='PSF'
            passed.append(next(c['accepted'] for c in record['checks'] if c['view']=='candidate/bootstrap_evidence'))
        score_pass[cohort]=np.array(passed,bool)
    for role in ['new_sky_field','new_source_seen_sky_field']:
        available[role],rows=distance_availability(grid['field']['cell'],evidence['field'],role,admitted=grid['field']['candidate'])
        reference_rows+=rows
    app=grid['application']; ensemble=json.loads(paths[3].read_text()); states=json.loads(paths[4].read_text())['states']
    assert sha(paths[4])==ensemble['outputs']['states.json'] and len(states)==34
    main=next(v for v in ensemble['views'] if v['view']=='bootstrap_evidence')
    assert main['inputs_passed'],'A missing/failed main state is not an available calibrated product'
    viewdata={}; model_rows=[]; support=[]
    for view in ensemble['views']:
        if not view['inputs_passed']:continue
        name=view['view']; folder=root/f'target_ensemble/candidate/{name}'; marker=folder/'complete.json'
        assert sha(marker)==ensemble['view_markers'][name+'/complete.json']; done=json.loads(marker.read_text())
        p=folder/'base_probabilities.npz'; assert sha(p)==done['outputs'][p.name]; value=load(p)
        np.testing.assert_array_equal(value['rows'],np.flatnonzero(app['candidate']))
        np.testing.assert_array_equal(value['identity'],app['identity'][value['rows']]); assert (value['TYPE']=='PSF').all()
        viewdata[name]=value
    primary=viewdata['bootstrap_evidence']; rows=primary['rows']; cell=app['cell'][rows]; selected=primary['qnear']>.95
    gap=missing_reference_mass(primary['class_distance'],cell,available['new_sky_field'])
    status=computational_status(np.ones(len(rows),bool),np.ones(len(rows),bool),main['accepted'])
    atomic_npz(out/'candidate_domains.npz',identity=primary['identity'],TYPE=primary['TYPE'],rows=rows,cell=cell,
        field=primary['field'],qnear=primary['qnear'],highq=selected,native_fine_audited=primary['fine_audited'],
        conditional_status=status,posterior_distance_mass_without_precise_newsky_reference=gap,
        newsky_precise_reference_available=available['new_sky_field'][cell])
    for view in ensemble['views']:
        name=view['view']
        if name not in viewdata:continue
        value=viewdata[name]
        for subset,mask in [('all_candidate',np.ones(len(rows),bool)),('fixed_main_highq',selected)]:
            for c in range(3):
                for b in range(6):
                    counts=np.bincount(cell[mask],weights=value['class_distance'][mask,c,b],minlength=N_CELLS)
                    for k in range(N_CELLS):model_rows.append(dict(view=name,subset=subset,cell=k,class_index=c,distance_bin=b,
                        conditional_expected_number=float(counts[k]),cohort_numerical_passed=view['accepted'],cell_count_intervals_computed=False))
    for k,row in enumerate(cells):
        m=cell==k; field=grid['field']; common=grid['common']
        row.update(application_n=int((app['cell']==k).sum()),candidate_n=int(m.sum()),highq_n=int((m & selected).sum()),
            native_fine_audited_n=int((m & primary['fine_audited']).sum()),
            model_expected_near=float(primary['qnear'][m].sum()),
            mean_distance_mass_without_precise_newsky_reference=float(gap[m].mean()) if m.any() else None,
            precise_newsky_distance_bins=int(available['new_sky_field'][k].sum()),
            cohort_numerical_passed=main['accepted'],scientific_adjudication='pending_evidence_review')
        for role in ['new_sky_field','new_source_seen_sky_field']:
            fm=(field['cell']==k) & field['candidate'] & (evidence['field']['role']==role)
            row.update({role+'_n':int(fm.sum()),role+'_Gaia':int((fm & evidence['field']['parallax_eligible']).sum()),
                role+'_numerical_accepted':int((fm & score_pass['field']).sum())})
        cm=(common['cell']==k) & common['candidate']; systems=evidence['common']['scale_system'][cm]
        row.update(common_members=int(cm.sum()),common_independent_scales=len(np.unique(systems[systems>=0])),
            common_numerical_accepted=int((cm & score_pass['common']).sum()))
    for state in states:
        marker=ROOT/state['fit']; assert sha(marker)==state['fit_sha256']; fit=json.loads(marker.read_text())
        counts=ROOT/state['counts']; assert sha(counts)==state['counts_sha256']; cd=json.loads(counts.read_text())
        bridge=fit['bridge'] or {}
        for check in cd['checks']:
            support.append(dict(correction=state['correction'],draws_per_chain=fit['draws'],state_passed=state['passed'],
                max_rhat=fit['max_rhat'],minimum_sampler_ess=fit['minimum_ess'],divergent=fit['divergent'],
                bridge_passed=fit['bridge_passed'],bridge_proposal_ess=bridge.get('proposal_bridge_ess'),
                bridge_half_delta=bridge.get('half_log_integral_difference'),**check))
    probes=json.loads(paths[5].read_text()); assert probes['states']==64 and probes['target_rows']==128
    save_json(out/'support_scopes.json',dict(candidate_luminosity_bootstrap_views=ensemble['views'],
        independent_SED_probe_sets=probes['sets'],SED_probe_target_rows=128,
        candidate_response=12,candidate_calibration=12,full_target_SED_response_marginalization=False,
        source_qualified_final_report=str(paths[2].relative_to(ROOT)),
        highq_shared_count_product='data/v3_1/highq_counts/complete.json',
        warning='Finite luminosity-bootstrap weights and fixed-prior128object64SED probes are different conditional experiments.'))
    csv_rows(out/'cells.csv',cells); csv_rows(out/'reference_distance_cells.csv',reference_rows)
    csv_rows(out/'conditional_class_distance_cells.csv',model_rows); csv_rows(out/'numerical_support.csv',support)
    names=['candidate_domains.npz','cells.csv','reference_distance_cells.csv','conditional_class_distance_cells.csv','numerical_support.csv','support_scopes.json']
    save_json(out/'complete.json',dict(inputs=inputs,outputs={name:sha(out/name) for name in names},all_PSF=True,
        candidate_n=len(rows),cells=N_CELLS,job=os.getenv('SLURM_JOB_ID'),scientific_domain_adjudication_complete=False,
        scope='Exactcandidate conditional means, source/role availability and actual numerical support. Not a calibration certificate; final scientific domain decisions and English map figures still required.'))


if __name__=='__main__':main()
