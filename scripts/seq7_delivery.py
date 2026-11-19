#!/usr/bin/env python3
"""Check public provenance, portable pixels and English single-page PDFs."""
import argparse,re,subprocess,tempfile
from PIL import Image,ImageChops
from seq7_common import *
from seq7_publish import render_runtime


def main():
    p=argparse.ArgumentParser();p.add_argument('--redraw',action='store_true');p.add_argument('--pixels-only',action='store_true');args=p.parse_args();public=ROOT/'manifests/sequence_mechanisms_20261119';fig=ROOT/'figures/sequence_mechanisms_20261119';delivery=public/'delivery';delivery.mkdir(exist_ok=True);r=json.loads((public/'release.json').read_text())
    if not args.pixels_only:from pypdf import PdfReader
    for path,digest in r['outputs'].items():assert sha(ROOT/path)==digest,path
    for path,digest in r['source'].items():assert sha(ROOT/path)==digest,path
    assert sha(CONFIG)==r['config'];check_templates();rows=[];runtime=render_runtime();same_runtime=runtime==r.get('render_runtime')
    redraw=Path(tempfile.mkdtemp(prefix='photdis-v7-replay-')) if args.redraw else None
    if redraw:subprocess.run([sys.executable,str(ROOT/'scripts/seq7_figures.py'),'--source',str(public/'data'),'--out',str(redraw)],check=True)
    for png in sorted(fig.glob('*.png')):
        pdf=png.with_suffix('.pdf')
        if not args.pixels_only:
            reader=PdfReader(pdf);assert len(reader.pages)==1;txt=reader.pages[0].extract_text();assert not re.search('[\u3400-\u9fff]',txt)
        with Image.open(png) as a:
            a.verify()
        with Image.open(png) as a:
            assert a.info['dpi'][0]>=299;size=a.size
            equal=None
            if redraw:
                with Image.open(redraw/png.name) as b:
                    assert a.size==b.size,png.name;equal=ImageChops.difference(a.convert('RGB'),b.convert('RGB')).getbbox() is None
                    if same_runtime:assert equal,png.name
        rows.append(dict(plate=png.stem,png_sha256=sha(png),pdf_sha256=sha(pdf),size=size,one_page=True if not args.pixels_only else None,English=True if not args.pixels_only else None,portable_pixels_identical=equal))
    sky=ROOT/'figures/sky_sequence_20261119/selected_channels/rgb_selected_fixed_gain.png';assert sha(sky)==r['sky_unchanged_sha256']
    save_json(delivery/('public_checks.json' if same_runtime or not redraw else 'cross_runtime_checks.json'),dict(bound_outputs=len(r['outputs']),source_files=len(r['source']),style_lock=check_templates(),sky_unchanged=sha(sky),render_runtime=runtime,release_runtime=r.get('render_runtime'),same_runtime=same_runtime,redraw_directory=str(redraw),plates=rows))
    print('Public evidence and portable figures verified',len(r['outputs']),len(rows),flush=True)


if __name__=='__main__':main()
