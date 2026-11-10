#!/usr/bin/env python3
"""Local compact-artifact and English-figure validation after real computation."""
import csv
import hashlib
import json
import re
from pathlib import Path
from PIL import Image
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1];M=ROOT/'manifests/v2_3_1';F=ROOT/'figures/v2_3_1'


def main():
    checks={}
    def check(k,v): checks[k]=bool(v);assert v,k
    for meta in sorted((M/'figures').glob('*.json')):
        q=json.loads(meta.read_text());name=q['name']
        with Image.open(F/(name+'.png')) as im:check(name+':PNG300dpi',im.format=='PNG' and abs(im.info['dpi'][0]-300)<.1)
        pdf=PdfReader(F/(name+'.pdf'));text='\n'.join(p.extract_text() or '' for p in pdf.pages)
        check(name+':PDFEnglish',len(pdf.pages)==1 and bool(text.strip()) and not re.search(r'[\u3400-\u9fff]',text))
        expected=0
        for source,h in q['sources'].items():
            path=M/(source+'.csv');check(name+':source:'+source,hashlib.sha256(path.read_bytes()).hexdigest()==h)
            with open(path) as f:expected+=len(list(csv.DictReader(f)))
        with open(meta.with_suffix('.csv')) as f:check(name+':source_rows',len(list(csv.DictReader(f)))==expected)
    check('sixteen_figure_sets',len(list((M/'figures').glob('*.json')))==16)
    for name in ('v2_3_1_result','v2_3_1_atlas','v2_3_1_requirements_result'):
        doc=ROOT/'docs'/(name+'.md')
        for i,target in enumerate(re.findall(r'\]\(([^)]+)\)',doc.read_text())):
            if '://' not in target:
                check(name+':link:'+str(i),(doc.parent/target.split('#')[0]).exists())
    visual=json.loads((M/'visual_review.json').read_text())
    for name,h in visual['png_sha256'].items():
        check(name+':visual_version',hashlib.sha256((F/name).read_bytes()).hexdigest()==h)
    check('representative_PDF_reviewed',visual['representative_pdf_rendered_and_reviewed'])
    actual=M/'execution/validation.json'
    compute=json.loads(actual.read_text()) if actual.exists() else None
    if compute is not None:check('real_compute',compute['all_passed'])
    result=dict(actual_checks=len(checks),all_passed=all(checks.values()),checks=checks,compute_validation=compute is not None,
        note='Format/provenance checks; visual review recorded separately. No whole-stage completion claim when compute_validation=false.')
    (M/'delivery_validation.json').write_text(json.dumps(result,indent=2)+'\n');print('DELIVERY',len(checks),'compute',compute is not None)


if __name__=='__main__':main()
