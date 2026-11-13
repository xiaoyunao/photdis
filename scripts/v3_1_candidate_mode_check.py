#!/usr/bin/env python3
"""Reconcile actual full-candidate tables, grouped counts and rendered source rows."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
from pypdf import PdfReader
from v2_1_near import ROOT,sha,save_json


def main():
    source=ROOT/'manifests/v3_1/candidate_modes';meta=ROOT/'manifests/v3_1/candidate_mode_figures'
    done=json.loads((source/'complete.json').read_text());figure=json.loads((meta/'complete.json').read_text());checks=[]
    def check(name,value):
        checks.append(dict(check=name,passed=bool(value)))
        if not value:raise AssertionError(name)
    tables={}
    for name in ['full_summary.csv','full_overview.csv','full_histograms.csv']:
        path=source/name;check('source_hash:'+name,sha(path)==done['outputs'][name])
        with path.open() as stream:tables[name]=list(csv.DictReader(stream))
    for section in ['inputs','outputs']:
        for name,digest in figure[section].items():check('figure_'+section+':'+name,sha(ROOT/name)==digest)
    rows=tables['full_summary.csv'];byid={r['identity']:r for r in rows}
    check('full3540nativePSF',len(rows)==len(byid)==3540 and all(r['TYPE']=='PSF' for r in rows) and done['all_PSF'])
    check('fixed3003highq',sum(r['fixed_main_highq']=='True' for r in rows)==3003==done['fixed_main_highq'])
    check('failure_inventory',sum(r['numerical_passed']!='True' for r in rows)==done['failed'])
    overview={r['group']:r for r in tables['full_overview.csv']};hist=tables['full_histograms.csv']
    for group,r in overview.items():
        selected=rows if group=='all' else [row for row in rows if row['fixed_main_highq']=='True']
        check('group_n:'+group,len(selected)==int(r['n']))
        check('multimodal:'+group,sum(int(row['significant_modes'])>=2 for row in selected)==int(r['multimodal']))
        for key in ['different_class','same_class_different_binary','same_class_different_parent']:
            check('contrast:'+group+':'+key,sum(row[key]=='True' for row in selected)==int(r[key]))
        check('binary_mean:'+group,abs(np.mean([float(row['binary_probability']) for row in selected])-float(r['mean_binary_probability']))<1e-12)
        for metric in ['significant_modes','binary_probability']:
            check('histogram_conservation:'+group+':'+metric,sum(int(row['n']) for row in hist if row['group']==group and row['metric']==metric)==len(selected))
    with (meta/'plotted.csv').open() as stream:plotted=list(csv.DictReader(stream))
    check('plotted_n',len(plotted)==figure['plotted_rows'])
    for i,r in enumerate(plotted):
        if r['panel'] in ['A','C']:
            check(f'source_row:{i}',any(all(r[k]==row[k] for k in row) for row in hist))
        elif r['panel']=='B':
            value=overview[r['group']];check(f'source_aggregate:{i}',r['n']==value[r['metric']] and r['denominator']==value['n'])
        else:
            value=byid[r['identity']];check(f'source_scatter:{i}',r['panel']=='D' and all(r[k]==value[k] for k in ['original_probability_delta','same_basin_joint_mass_delta','numerical_passed']))
    for name in figure['outputs']:
        path=ROOT/name
        if path.suffix=='.png':
            with Image.open(path) as im:im.load();check(name+':PNG300dpi',min(im.size)>=1000 and all(abs(x-300)<1 for x in im.info.get('dpi',(0,0))))
        elif path.suffix=='.pdf':
            doc=PdfReader(path);check(name+':one_page',len(doc.pages)==1);text=doc.pages[0].extract_text()
            check(name+':English_text',len(text)>100 and not any('\u4e00'<=c<='\u9fff' for c in text))
    maxima={k:max(float(r[k]) for r in rows) for k in ['original_probability_delta','probability_delta','quantile_delta','same_basin_joint_mass_delta','marginal_half_delta','component_conservation_delta']}
    save_json(meta/'integrity.json',dict(inputs={str((source/'complete.json').relative_to(ROOT)):sha(source/'complete.json'),str((meta/'complete.json').relative_to(ROOT)):sha(meta/'complete.json'),str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},
        checks=checks,actual_checks=len(checks),passed=True,actual_maxima=maxima,visual_review_completed=False,
        scope='Exact actual source and plot integrity only. Model responsibilities do not identify physical binaries or erase inherited ensemble failures.'))
    print(json.dumps(dict(actual_checks=len(checks),passed=True,n=len(rows),plotted_rows=len(plotted))))


if __name__=='__main__':main()
