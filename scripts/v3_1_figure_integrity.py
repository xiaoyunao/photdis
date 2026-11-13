#!/usr/bin/env python3
"""Authenticate figure formats and exact source rows; visual review is separate."""
import argparse
import csv
import json
from pathlib import Path
from pypdf import PdfReader
from PIL import Image
from v2_1_near import ROOT,sha,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--source',required=True)
    a=p.parse_args();meta=ROOT/a.manifest;source=ROOT/a.source
    marker=meta/'complete.json';done=json.loads(marker.read_text());checks=[]
    def check(name,condition):
        checks.append(dict(check=name,passed=bool(condition)))
        if not condition:raise AssertionError(name)
    for section in ['inputs','outputs']:
        for name,digest in done[section].items():check(section+':'+name,sha(ROOT/name)==digest)
    check('renderer_English_text',done['english_text_verified'])
    images=pdfs=0
    for name in done['outputs']:
        path=ROOT/name
        if path.suffix=='.png':
            images+=1
            with Image.open(path) as im:
                im.load();check(name+':dimensions',min(im.size)>=1000)
                check(name+':300dpi',all(abs(x-300)<1 for x in im.info.get('dpi',(0,0))))
        elif path.suffix=='.pdf':
            pdfs+=1
            document=PdfReader(path)
            check(name+':one_page',len(document.pages)==1)
            page=document.pages[0];text=page.extract_text()
            check(name+':extractable_text',len(text)>100)
            check(name+':English_text',not any('\u4e00'<=c<='\u9fff' for c in text))
            check(name+':positive_page_size',float(page.mediabox.width)>0 and float(page.mediabox.height)>0)
    check('PNG_count',images==done['figures']);check('PDF_count',pdfs==done['figures'])
    with (meta/'plotted.csv').open() as stream:plotted=list(csv.DictReader(stream))
    check('plotted_row_count',len(plotted)==done['plotted_rows'])
    tables={}
    for i,row in enumerate(plotted):
        name=row.get('source_table') or row['source'];relative=Path(name)
        check(f'safe_source_path_{i}',not relative.is_absolute() and '..' not in relative.parts)
        if name not in tables:
            path=source/name;key=str(path.relative_to(ROOT))
            check('authenticated_source:'+name,key in done['inputs'] and sha(path)==done['inputs'][key])
            with path.open() as stream:
                reader=csv.DictReader(stream);headers=reader.fieldnames;rows=list(reader)
            tables[name]=(headers,{tuple(r[k] for k in headers) for r in rows})
        headers,rows=tables[name]
        check(f'exact_source_row_{i}',tuple(row[k] for k in headers) in rows)
    save_json(meta/'integrity.json',dict(figure_manifest_sha256=sha(marker),validator_sha256=sha(Path(__file__)),
        source=str(source.relative_to(ROOT)),checks=checks,actual_checks=len(checks),passed=True,
        visual_review_completed=False,scope='Actual artifact and exact plotted source-row checks only; visual review and scientific acceptance remain separate.'))
    print(json.dumps(dict(actual_checks=len(checks),passed=True,figures=images,plotted_rows=len(plotted))))


if __name__=='__main__':main()
