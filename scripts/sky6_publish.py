#!/usr/bin/env python3
"""Publish only compact arrays/tables/figures; leave raw sources and cube on NERSC."""
import shutil
from sky6_common import *


def main():
    public=ROOT/'manifests/sky_sequence_20261119';fig=ROOT/'figures/sky_sequence_20261119';records={}
    for stage in (*stages(),'no_extinction','display','band','prior_display','ridge','width'):
        source=OUT/stage
        if not (source/'complete.json').exists():continue
        r=json.loads((source/'complete.json').read_text());dest=public/stage;dest.mkdir(parents=True,exist_ok=True)
        for name,h in r['outputs'].items():
            p=source/name
            if p.suffix in ('.png','.pdf'):
                target=fig/stage;target.mkdir(parents=True,exist_ok=True);shutil.copy2(verified(p,h),target/name)
            elif p.suffix in ('.csv','.json') or (p.suffix=='.npz' and (p.name.startswith('rgb_') or p.name in ('sky_coverage.npz','ridge_redraw.npz','width_redraw.npz'))):shutil.copy2(verified(p,h),dest/name)
        shutil.copy2(source/'complete.json',dest/'complete.json');records[stage]=sha(source/'complete.json')
    save_json(public/'delivery.json',dict(stages=records,production=C['production'],outputs={str(p.relative_to(ROOT)):sha(p) for root in (fig,public) for p in root.rglob('*') if p.is_file() and p.name!='delivery.json'},scope='display A delivered independently of sequence-science B; full raw arrays/individual rows and cube private on NERSC'))


if __name__=='__main__':main()
