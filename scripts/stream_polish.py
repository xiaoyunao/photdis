#!/usr/bin/env python3
"""Plot-only replay of frozen grids; retain numerical files and original receipts."""
import shutil
import subprocess
from photdis.stage1 import require_compute
from stream_common import *


def main():
    require_compute();execution=folder('execution')
    for stage in ('rgb','local','sequence','conditioning'):
        source=OUT/stage;marker=source/'complete.json';r=json.loads(marker.read_text());before={name:sha(source/name) for name in r['outputs'] if not name.endswith(('.png','.pdf'))}
        archive=execution/(stage+'_before_plot_polish.json')
        if not archive.exists():archive.write_text(marker.read_text())
        dest=folder('polished')/stage
        subprocess.run([sys.executable,str(ROOT/'scripts/stream_redraw.py'),stage,'--root',str(OUT),'--output',str(dest)],check=True)
        if stage=='local':subprocess.run([sys.executable,str(ROOT/'scripts/stream_redraw.py'),'phase','--root',str(OUT),'--output',str(dest)],check=True)
        for p in dest.iterdir():
            if p.suffix in ('.png','.pdf'):shutil.copy2(p,source/p.name)
        for name,h in before.items():assert sha(source/name)==h,(stage,name)
        r['figure_revision']=dict(renderer=sha(ROOT/'scripts/stream_redraw.py'),revision_receipt=sha(archive),numerical_arrays_and_tables_unchanged=True)
        r['outputs']={p.name:sha(p) for p in source.iterdir() if p.is_file() and p.name!='complete.json'}
        save_json(marker,r)


if __name__=='__main__':main()
