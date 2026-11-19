#!/usr/bin/env python3
"""Check the compact scientific package without access to NERSC raw catalogs."""
import argparse,hashlib,json,re,subprocess,sys
from pathlib import Path
from PIL import Image
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'manifests/stream_response_20261119'
FIG=ROOT/'figures/stream_response_20261119'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--redraw',type=Path);a=p.parse_args()
    receipt=json.loads((PUBLIC/'delivery.json').read_text());checks=[]
    for relative,h in receipt['outputs'].items():
        target=ROOT/relative;assert target.is_file() and sha(target)==h,relative
    for stage,h in receipt['inputs'].items():assert sha(PUBLIC/stage/'complete.json')==h,stage
    checks.append(dict(check='all published source/output receipt hashes',n=len(receipt['outputs']),passed=True))
    pdfs=sorted(FIG.rglob('*.pdf'));assert len(pdfs)==36
    for pdf in pdfs:
        reader=PdfReader(pdf);assert len(reader.pages)==1
        text=reader.pages[0].extract_text();assert len(text.strip())>20
        assert not re.search('[\u3400-\u9fff]',text),pdf
        with Image.open(pdf.with_suffix('.png')) as im:assert min(im.size)>=900
    checks.append(dict(check='36 one-page PDFs and high-resolution PNGs, English labels',passed=True))
    for md in PUBLIC.glob('*.md'):
        for target in re.findall(r'\]\(([^)#]+)(?:#[^)]*)?\)',md.read_text()):
            if '://' not in target:assert (md.parent/target).exists(),(md,target)
    checks.append(dict(check='scientific documentation figure/file links',passed=True))
    if a.redraw:
        for stage,extra in (('rgb',[]),('rgb',['--rgb-stage','full_rgb']),('local',[]),('phase',[]),('sequence',[]),('conditioning',[])):
            name='full_rgb' if extra else stage;dest=a.redraw/name
            subprocess.run([sys.executable,str(ROOT/'scripts/stream_redraw.py'),stage,'--output',str(dest),*extra],check=True)
        assert len(list(a.redraw.rglob('*.pdf')))==36
        checks.append(dict(check='all 36 plates independently redrawn using compact public evidence only',passed=True))
    dest=PUBLIC/'delivery';dest.mkdir(exist_ok=True)
    (dest/'public_checks.json').write_text(json.dumps(dict(checks=checks,production=receipt['production_unchanged'],
        files={str(p.relative_to(ROOT)):sha(p) for directory in (PUBLIC,FIG) for p in sorted(directory.rglob('*')) if p.is_file() and p!=dest/'public_checks.json'}),indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
