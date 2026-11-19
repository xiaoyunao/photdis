#!/usr/bin/env python3
"""Publish compact result evidence; raw imaging and the full cube remain private."""
import shutil
from stream_common import *


def main():
    stages=('rgb','local_source','local_area','local','core','sequence','conditioning')
    public=ROOT/'manifests/stream_response_20261119';fig=ROOT/'figures/stream_response_20261119'
    receipts={}
    for stage in stages:
        source=OUT/stage;record=json.loads((source/'complete.json').read_text());dest=public/stage;dest.mkdir(parents=True,exist_ok=True)
        for name,h in record['outputs'].items():
            p=verified(source/name,h)
            if p.suffix in ('.png','.pdf'):
                target=fig/stage;target.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target/p.name)
            elif p.suffix in ('.csv','.json') or (p.suffix=='.npz' and ((stage=='rgb' and p.name.startswith('rgb_')) or (stage=='local' and p.name.endswith('_redraw.npz')) or (stage=='sequence' and p.name in ('predictions.npz','template_redraw.npz','ordinary_predictions.npz')))):
                shutil.copy2(p,dest/p.name)
        shutil.copy2(source/'complete.json',dest/'complete.json');receipts[stage]=sha(source/'complete.json')
    save_json(public/'delivery.json',dict(inputs=receipts,outputs={str(p.relative_to(ROOT)):sha(p) for root in (public,fig) for p in root.rglob('*') if p.is_file() and p.name!='delivery.json'},
        private='raw 61-node cube, physical full-sky channels, full source rows, full random rows and weighting identities remain on NERSC with hashes in per-stage receipts',
        production_unchanged=C['production'],style_lock=check_templates()))


if __name__=='__main__':main()
