#!/usr/bin/env python3
"""Validate compact response tables and summarize saved mixture scores by sky blocks."""
import csv
import json
import numpy as np
from v2_1_near import ROOT,sha,save_json,csv_rows
from photdis.v3_1.comparison import paired_mean


def main():
    source=ROOT/'manifests/v3_1/response_probes/report';marker=source/'complete.json'
    done=json.loads(marker.read_text());inputs={str(marker.relative_to(ROOT)):sha(marker)};checks=[];tables={}
    def check(name,condition):
        checks.append(dict(check=name,passed=bool(condition)))
        if not condition:raise AssertionError(name)
    for name,digest in done['outputs'].items():
        if name.endswith('.npz'):continue
        path=source/name;check('hash:'+name,sha(path)==digest);inputs[str(path.relative_to(ROOT))]=sha(path)
        if name.endswith('.csv'):
            with path.open() as stream:tables[name]=list(csv.DictReader(stream))
    inventory=tables['inventory.csv'];check('all64states',len(inventory)==64 and {int(r['draw']) for r in inventory}==set(range(32))|set(range(1000,1032)))
    check('samePSFdesign',done['all_PSF'] and done['target_rows']==128 and done['development_rows']==256)
    check('matched_rows_no_training_overlap',all(int(r['target_rows'])==128 and int(r['development_rows'])==256 and int(r['target_training_overlap'])==0 for r in inventory))
    flags=json.loads((source/'numerical_flags.json').read_text());old=json.loads((source/'original_numerical_flags.json').read_text())
    check('flag_counts',len(flags)==done['numerical_flags'] and len(old)==done['original_numerical_flags'])
    for item in done['sets']:
        rows=[r for r in tables['response_weights.csv'] if r['set']==item['set']]
        if not item['target_inputs_passed']:
            check('no_survivor_normalization:'+item['set'],not rows and not item['accepted']);continue
        check('complete_weights:'+item['set'],len(rows)==item['states'])
        w=np.array([float(r['posterior_weight']) for r in rows]);lw=np.array([float(r['log_posterior_weight']) for r in rows])
        check('normalized_weights:'+item['set'],np.isfinite(lw).all() and abs(w.sum()-1)<1e-12 and np.max(abs(w-np.exp(lw)))<1e-12)
        check('actual_ESS:'+item['set'],abs(1/(w@w)-item['ESS'])<1e-10)
    for row in inventory:
        for axis in sorted({r['axis'] for r in tables['target_decomposition.csv']}):
            group=[r for r in tables['target_decomposition.csv'] if r['draw']==row['draw'] and r['axis']==axis]
            check('decomposition:'+row['draw']+':'+axis,sum(int(r['n']) for r in group)==128 and abs(sum(float(r['conditional_log_evidence']) for r in group)-float(row['target_conditional_log_evidence']))<1e-8)
    for label in {r['set'] for r in tables['target_counts.csv']}:
        rows=[r for r in tables['target_counts.csv'] if r['set']==label]
        check('joint_count_endpoints:'+label,len(rows)==18 and abs(sum(float(r['mean']) for r in rows)-128)<1e-8 and all(0<=int(r['lower95'])<=int(r['median'])<=int(r['upper95'])<=128 for r in rows))
    scores=tables['mixture_scores.csv'];check('mixture_PSF_identity',all(r['TYPE']=='PSF' for r in scores) and len({(r['set'],r['identity'],r['source']) for r in scores})==len(scores))
    summary=[]
    for label in sorted({r['set'] for r in scores}):
        for group in ['all']+sorted({r['reference_class'] for r in scores}):
            for channel in sorted({r['source'] for r in scores}):
                rows=[r for r in scores if r['set']==label and r['source']==channel and (group=='all' or r['reference_class']==group)]
                if not rows:continue
                summary.append(dict(set=label,reference_class=group,source=channel,all_numerically_accepted=all(r['numerically_accepted']=='True' for r in rows),
                    **paired_mean([float(r['gain_over_response12']) for r in rows],[float(r['design_weight']) for r in rows],[int(r['cell']) for r in rows])))
    out=ROOT/'manifests/v3_1/response_summary';out.mkdir(exist_ok=True);csv_rows(out/'mixture_source_summary.csv',summary)
    for path in [ROOT/'scripts/v3_1_response_summary.py',ROOT/'src/photdis/v3_1/comparison.py']:inputs[str(path.relative_to(ROOT))]=sha(path)
    save_json(out/'complete.json',dict(inputs=inputs,outputs={'mixture_source_summary.csv':sha(out/'mixture_source_summary.csv')},checks=checks,actual_checks=len(checks),passed=True,
        mixture_summary_rows=len(summary),sky_bootstrap_repeats=1024,sky_bootstrap_seed=311302,
        scope='Compact source integrity and source-qualified saved-mixture log-score gains on seen development data. Whole-sky bootstrap, original inclusion weights; no new fitting or independent validation.'))
    print(json.dumps(dict(actual_checks=len(checks),passed=True,mixture_summary_rows=len(summary))))


if __name__=='__main__':main()
