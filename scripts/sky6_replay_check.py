#!/usr/bin/env python3
"""Check all portable science plates and the new preview-only main protection."""
from PIL import Image,ImageChops
from photdis.stage1 import require_compute
from sky6_common import *
from sky6_figures import rgb


def main():
    require_compute();replay=OUT/'independent_redraw';dest=replay/'smoothing_trials';dest.mkdir(exist_ok=True)
    for path in sorted((ROOT/'manifests/sky_sequence_20261119/smoothing_trials').glob('rgb_*.npz')):
        with np.load(path) as z:a={k:z[k] for k in z.files}
        rgb(a,dest,path.stem)
    checks=[]
    for source in sorted((ROOT/'figures/sky_sequence_20261119').rglob('*.png')):
        rel=source.relative_to(ROOT/'figures/sky_sequence_20261119')
        if rel.parts[0]=='comparisons':continue
        target=replay/rel;assert target.exists(),rel
        with Image.open(source) as a,Image.open(target) as b:assert a.size==b.size and ImageChops.difference(a.convert('RGB'),b.convert('RGB')).getbbox() is None,rel
        checks.append(dict(plate=str(rel),original_sha256=sha(source),replay_sha256=sha(target),pixels_identical=True))
    for stage in ('selected_channels','smoothing_trials'):
        d=OUT/stage;r=json.loads((d/'complete.json').read_text())
        for name,h in r['outputs'].items():verified(d/name,h)
    kept=json.loads((OUT/'smoothing_trials/main_preserved.json').read_text());assert sha(ROOT/kept['path'])==kept['sha256_before']==kept['sha256_after']
    d=OUT/'validation';save_json(d/'independent_replay.json',dict(scientific_plates=len(checks),checks=checks,main_unchanged=kept,scope='All scientific PNGs independently redrawn from public compact evidence, no raw source catalogs.2/10degree backgrounds remain separate previews.'))
    r=json.loads((d/'complete.json').read_text());meta={k:v for k,v in r.items() if k not in ('config','job','inputs','code','style','outputs')};complete(d,r['inputs'],**meta,portable_redraw_plates=len(checks),main_protection_passed=True)


if __name__=='__main__':main()
