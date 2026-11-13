#!/usr/bin/env python3
"""Check actual domain evidence and aggregate map cells without certifying calibration."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from PIL import Image
from pypdf import PdfReader
from v2_1_near import ROOT,sha,save_json


def main():
    source=ROOT/'manifests/v3_1/domain_evidence';meta=ROOT/'manifests/v3_1/domain_evidence_figures'
    done=json.loads((source/'complete.json').read_text());figure=json.loads((meta/'complete.json').read_text());checks=[]
    def check(name,value):
        checks.append(dict(check=name,passed=bool(value)))
        if not value:raise AssertionError(name)
    for name,digest in done['outputs'].items():
        if not name.endswith('.npz'):check('source_hash:'+name,sha(source/name)==digest)
    for section in ['inputs','outputs']:
        for name,digest in figure[section].items():check('figure_'+section+':'+name,sha(ROOT/name)==digest)
    read=lambda path:list(csv.DictReader(path.open()))
    cells=read(source/'cells.csv');refs=read(source/'reference_distance_cells.csv');counts=read(source/'conditional_class_distance_cells.csv')
    check('75unique_cells',len(cells)==75 and {int(r['cell']) for r in cells}==set(range(75)))
    bycell={int(r['cell']):r for r in cells}
    for key,total in [('application_n',26000),('candidate_n',3540),('highq_n',3003),('native_fine_audited_n',36),
                      ('new_sky_field_n',167),('new_source_seen_sky_field_n',352),('common_members',1732)]:
        check('cohort_total:'+key,sum(int(r[key]) for r in cells)==total)
    check('reference_inventory',len(refs)==5400 and len({tuple(r[k] for k in ['role','source','quality','cell','distance_bin']) for r in refs})==5400)
    check('nonnegative_reference_counts',all(int(r['n'])>=0 for r in refs))
    check('numerical_support_inventory',len(read(source/'numerical_support.csv'))==136)
    groups=defaultdict(list)
    for r in counts:groups[r['view'],r['subset'],int(r['cell'])].append(r)
    check('joint_count_groups',len(groups)==600)
    for (view,subset,cell),rows in groups.items():
        n=int(bycell[cell]['candidate_n' if subset=='all_candidate' else 'highq_n'])
        check(f'joint_count_conservation:{view}:{subset}:{cell}',len(rows)==18 and len({(r['class_index'],r['distance_bin']) for r in rows})==18 and abs(sum(float(r['conditional_expected_number']) for r in rows)-n)<1e-8)
    plotted=read(meta/'plotted.csv');check('plot_inventory',len(plotted)==300 and figure['panels']==12 and figure['plotted_rows']==300)
    reference_values=defaultdict(int)
    for r in refs:
        if r['role']=='new_sky_field' and r['quality']=='precise':reference_values[r['source'],int(r['cell'])]+=int(r['n'])
    for i,r in enumerate(plotted):
        cell=int(r['cell']);source_row=bycell[cell]
        check(f'cell_geometry:{i}',all(r[k]==source_row[k] for k in ['r_bin','color_bin','wise_bin']))
        if r['figure']=='precise_newsky_references':
            channel={'AstroNN (Gaia-trained teacher)':'astronn','SpecDis source-qualified':'specdis','SPDist source-qualified':'spdist'}[r['panel']]
            expected=reference_values[channel,cell];check(f'plotted_source_value:{i}',float(r['value'])==expected and r['display']==str(expected))
        else:
            check(f'gap_figure:{i}',r['figure']=='missing_reference_mass')
            if int(source_row['candidate_n']):
                expected=float(source_row['mean_distance_mass_without_precise_newsky_reference'])
                check(f'plotted_gap:{i}',0<=expected<=1+1e-8 and float(r['value'])==expected and r['display']==f'{100*expected:.0f}%')
            else:check(f'empty_not_zero:{i}',r['value']=='' and r['display']=='—')
    for name in figure['outputs']:
        path=ROOT/name
        if path.suffix=='.png':
            with Image.open(path) as im:im.load();check(name+':PNG300dpi',min(im.size)>=1000 and all(abs(x-300)<1 for x in im.info.get('dpi',(0,0))))
        elif path.suffix=='.pdf':
            doc=PdfReader(path);check(name+':one_page',len(doc.pages)==1);text=doc.pages[0].extract_text()
            check(name+':English_text',len(text)>100 and not any('\u4e00'<=c<='\u9fff' for c in text))
    gap=sum(int(r['candidate_n'])*float(r['mean_distance_mass_without_precise_newsky_reference']) for r in cells if int(r['candidate_n']))/3540
    facts=dict(candidate_n=3540,occupied_cells=sum(int(r['candidate_n'])>0 for r in cells),
        candidates_in_cells_without_precise_newsky_reference=sum(int(r['candidate_n']) for r in cells if int(r['precise_newsky_distance_bins'])==0),
        candidate_mean_distance_mass_without_precise_newsky_reference=gap,
        precise_newsky_references_by_source={channel:sum(v for (s,c),v in reference_values.items() if s==channel) for channel in ['astronn','specdis','spdist']})
    save_json(meta/'integrity.json',dict(inputs={str((source/'complete.json').relative_to(ROOT)):sha(source/'complete.json'),str((meta/'complete.json').relative_to(ROOT)):sha(meta/'complete.json'),str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},
        checks=checks,actual_checks=len(checks),passed=True,facts=facts,visual_review_completed=False,scientific_domain_adjudication_complete=False,
        scope='Source/geometry/count/format checks and reference-gap summary. No calibrated-domain or validated probability claim.'))
    print(json.dumps(dict(actual_checks=len(checks),passed=True,facts=facts)))


if __name__=='__main__':main()
