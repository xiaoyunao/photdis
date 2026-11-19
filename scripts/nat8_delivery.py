#!/usr/bin/env python3
"""Bound public delivery checks and independent native-runtime pixel replay."""
from nat8_common import *
import argparse


def main():
    p=argparse.ArgumentParser();p.add_argument('--pixels-only',action='store_true');p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);root=ROOT/'manifests/native_luminosity_20261119';release=json.loads((root/'release.json').read_text());checks=[]
    for rel,h in release['outputs'].items():verified(ROOT/rel,h);checks.append(rel)
    for rel,h in release['source'].items():verified(ROOT/rel,h)
    ids=list(csv.DictReader((root/'prepare/identities.csv').open()));assert len(ids)==933 and all(r['TYPE']=='PSF' for r in ids);quant=list(csv.DictReader((root/'predict/posterior_quantiles.csv').open()));assert len(quant)==3147 and all(r['TYPE']=='PSF' for r in quant)
    cases=json.loads((root/'data/cases.json').read_text());curves=load(root/'data/example_curves.npz');assert len(cases)==7
    for c in cases:
        for method in c['methods']:
            cf=curves[f'{c["case"]}_{method}_CDF'];assert np.diff(cf).min()>-1e-10 and cf[0]<1e-8 and abs(cf[-1]-1)<1e-8
    decision=json.loads((root/'decision/science_decision.json').read_text());assert decision['M5_gate'] and not decision['science_gate'] and not decision['prospective_evaluation_performed'];report=dict(bound_outputs=len(checks),source_files=len(release['source']),parent_PSF=933,forecast_rows=len(quant),fixed_examples=7,all_full_CDFs_normalized=True,gate_not_opened=True)
    if args.pixels_only:
        from photdis.stage1 import require_compute
        from nat8_figures import render
        from PIL import Image
        require_compute();render(root/'data',args.out/'replay');pairs=[]
        for png in sorted((ROOT/'figures/native_luminosity_20261119').glob('*.png')):
            a=np.array(Image.open(png));b=np.array(Image.open(args.out/'replay'/png.name));assert a.shape==b.shape and np.array_equal(a,b),png.name;pairs.append(dict(file=png.name,pixels_equal=True,shape=list(a.shape)))
        assert len(pairs)==5;report['native_pixel_replay']=pairs
    else:
        from pypdf import PdfReader
        from PIL import Image
        pdf=[]
        for path in sorted((ROOT/'figures/native_luminosity_20261119').glob('*.pdf')):
            reader=PdfReader(path);assert len(reader.pages)==1;page=reader.pages[0];text=page.extract_text();assert text.strip() and not any('\u4e00'<=c<='\u9fff' for c in text);png=Image.open(path.with_suffix('.png'));assert min(png.size)>1000;pdf.append(dict(file=path.name,pages=1,English_only=True,width_pt=float(page.mediabox.width),height_pt=float(page.mediabox.height),PNG=list(png.size)))
        assert len(pdf)==5;report['local_PDF_checks']=pdf
    save_json(args.out/'checks.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
