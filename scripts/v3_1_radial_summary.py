#!/usr/bin/env python3
"""All-scenario count/parameter summaries with actual source-reuse blocks."""
import csv
import json
import os
import numpy as np
from v2_1_near import ROOT,load,sha,save_json,csv_rows
from photdis.stage1 import require_compute
from photdis.v3_1.catalog_blocks import overlap_blocks,block_mean


def main():
    require_compute();base=ROOT/'data/v3_1';source=base/'radial_report';out=base/'radial_summary';out.mkdir(exist_ok=True)
    paths=[source/'complete.json',base/'radial_catalogs/complete.json',
        ROOT/'scripts/v3_1_radial_summary.py',ROOT/'src/photdis/v3_1/catalog_blocks.py',
        ROOT/'src/photdis/v3_1/comparison.py',ROOT/'docs/v3_1_radial_summary_protocol.md']
    inputs={str(p.relative_to(ROOT)):sha(p) for p in paths}
    done=json.loads(paths[0].read_text());prep=json.loads(paths[1].read_text())
    assert done['catalogs']==176 and done['arms']==352 and done['all_outcomes_retained']
    def read(name):
        path=source/name;assert sha(path)==done['outputs'][name];inputs[str(path.relative_to(ROOT))]=sha(path)
        with path.open() as stream:return list(csv.DictReader(stream))
    tables={name:read(name+'.csv') for name in ['status','recovery','totals','paired','structure']}
    data=[]
    for name in ['catalogs.json','truth.npz','photometry.npz']:
        path=base/'radial_catalogs'/name;assert sha(path)==prep['outputs'][name]
        inputs[str(path.relative_to(ROOT))]=sha(path);data.append(json.loads(path.read_text()) if name.endswith('.json') else load(path))
    catalogs,truth,photo=data
    np.testing.assert_array_equal(truth['identity'],photo['identity']);assert len(photo['identity'])==176*128 and (photo['TYPE']=='PSF').all()
    assert {c['catalog'] for c in catalogs}==set(range(176))
    status={(int(r['catalog']),r['arm']):r for r in tables['status']}
    assert len(status)==len(tables['status'])==352
    assert set(status)=={(c,arm) for c in range(176) for arm in ['broad','local']}
    metadata={c['catalog']:c for c in catalogs}
    for table in tables.values():
        for row in table:
            catalog=metadata[int(row['catalog'])]
            assert row['scenario']==catalog['scenario'] and int(row['field'])==catalog['field']
            assert int(row['repeat'])==catalog['repeat']
    paired={(int(r['catalog']),r['estimand']):r for r in tables['paired']}
    assert len(paired)==len(tables['paired'])==352
    recovery={(int(r['catalog']),r['arm'],r['estimand'],int(r['class_index']),int(r['distance_bin'])):r for r in tables['recovery']}
    assert len(recovery)==len(tables['recovery'])==352*2*18
    def passed(c,arm):return status[c,arm]['numerically_passed']=='True'
    for key,r in recovery.items():
        assert (r['numerically_passed']=='True')==passed(key[0],key[1])
    structures={(int(r['catalog']),r['parameter']):r for r in tables['structure']}
    assert len(structures)==len(tables['structure'])==352
    results=[];inventory=[]
    scenarios=sorted({c['scenario'] for c in catalogs});assert len(scenarios)==22
    for scenario in scenarios:
        records=sorted([c for c in catalogs if c['scenario']==scenario],key=lambda c:c['catalog'])
        assert len(records)==8 and {c['repeat'] for c in records}==set(range(8))
        ids=np.array([c['catalog'] for c in records]);fields=np.array([c['field'] for c in records])
        sources=[]
        for c in ids:
            use=truth['catalog']==c;assert use.sum()==128
            assert (truth['scenario'][use]==scenario).all()
            assert (photo['field'][use]==metadata[c]['field']).all()
            sources.append(set(truth['template_identity'][use].tolist()))
        blocks=dict(field=fields,source_overlap=overlap_blocks(sources),
            source_or_field=overlap_blocks(sources,fields))
        for i,c in enumerate(ids):
            inventory.append(dict(scenario=scenario,catalog=int(c),field=int(fields[i]),
                distinct_used_templates=len(sources[i]),source_component=int(blocks['source_overlap'][i]),
                joint_component=int(blocks['source_or_field'][i]),broad_passed=passed(c,'broad'),local_passed=passed(c,'local')))
        def summarize(values,eligible,*,metric,estimand,arm,admission,class_index=-1,distance_bin=-1):
            values=np.asarray(values,float);eligible=np.asarray(eligible,bool)
            assert values.shape==eligible.shape==ids.shape
            for label,group in blocks.items():
                results.append(dict(scenario=scenario,metric=metric,estimand=estimand,arm=arm,
                    admission=admission,class_index=class_index,distance_bin=distance_bin,
                    all_catalogs=8,eligible_catalogs=int(eligible.sum()),
                    broad_failed=sum(not passed(c,'broad') for c in ids),local_failed=sum(not passed(c,'local') for c in ids),
                    blocking=label,**block_mean(values[eligible],group[eligible])))
        for estimand in ['realized_selected_count','pre_entry_population_fraction']:
            delta=[]
            for c in ids:
                r=paired[c,estimand];assert r['scenario']==scenario
                assert (r['both_numerically_passed']=='True')==(passed(c,'broad') and passed(c,'local'))
                value=float(r['local_fraction_L1'])-float(r['broad_fraction_L1'])
                assert abs(value-float(r['local_minus_broad_fraction_L1']))<1e-12;delta.append(value)
            for admission,eligible in [('all_outcomes',np.ones(8,bool)),('numerically_accepted',np.array([passed(c,'broad') and passed(c,'local') for c in ids]))]:
                summarize(delta,eligible,metric='local_minus_broad_fraction_L1',estimand=estimand,arm='paired',admission=admission)
            for arm in ['broad','local']:
                for cls in range(3):
                    for dbin in range(6):
                        rows=[recovery[c,arm,estimand,cls,dbin] for c in ids]
                        true=np.array([float(r['truth']) for r in rows]);mean=np.array([float(r['mean']) for r in rows])
                        cover=np.array([r['covered95']=='True' for r in rows],float)
                        expected=np.array([float(r['lower95'])<=t<=float(r['upper95']) for r,t in zip(rows,true)])
                        np.testing.assert_array_equal(cover,expected)
                        for admission,eligible in [('all_outcomes',np.ones(8,bool)),('numerically_accepted',np.array([passed(c,arm) for c in ids]))]:
                            kw=dict(estimand=estimand,arm=arm,admission=admission,class_index=cls,distance_bin=dbin)
                            summarize(mean-true,eligible,metric='mean_minus_truth',**kw)
                            summarize(cover,eligible,metric='coverage95_all_bins',**kw)
                            summarize(cover,eligible & (true>0),metric='coverage95_nonzero_truth',**kw)
        for parameter in ['location_mu','width_mu']:
            rows=[structures[c,parameter] for c in ids]
            defined=np.array([r['truth'] not in ('',None) for r in rows]);true=np.array([float(r['truth']) if r['truth'] not in ('',None) else np.nan for r in rows])
            median=np.array([float(r['median']) for r in rows]);cover=np.array([r['covered95']=='True' for r in rows],float)
            for admission,eligible in [('all_outcomes',defined),('numerically_accepted',defined & np.array([passed(c,'local') for c in ids]))]:
                kw=dict(estimand=parameter,arm='local',admission=admission)
                summarize(median-true,eligible,metric='median_minus_truth',**kw)
                summarize(cover,eligible,metric='coverage95_defined_structure',**kw)
    csv_rows(out/'source_inventory.csv',inventory)
    outputs=['source_inventory.csv']
    for metric in sorted({r['metric'] for r in results}):
        name=metric+'.csv';csv_rows(out/name,[r for r in results if r['metric']==metric]);outputs.append(name)
    save_json(out/'complete.json',dict(inputs=inputs,outputs={name:sha(out/name) for name in outputs},
        catalogs=176,arms=352,scenarios=22,rows=len(results),all_PSF=True,all_outcomes_retained=True,
        shared_library_uncertainty_measured=False,scientific_acceptance=False,job=os.getenv('SLURM_JOB_ID'),
        scope='Equal-catalog conditional summaries; field, transitive used-source and joint blocks separately. No interval for one block. All/accepted and zero/nonzero-truth denominators separate. No independent library, source-teacher scale or real population calibration claim. Original selection approximation and numerical failures remain.'))


if __name__=='__main__':main()
