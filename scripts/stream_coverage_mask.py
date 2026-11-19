#!/usr/bin/env python3
"""An undefined zero-area rate must be masked, not treated as a zero rate."""
import shutil
from photdis.stage1 import require_compute
from stream_common import *


def main():
    require_compute();d=folder('local');execution=folder('execution');marker=d/'complete.json';rec=json.loads(marker.read_text());receipt=execution/'local_before_undefined_area_mask.json'
    if not receipt.exists():receipt.write_text(marker.read_text())
    corrected=[]
    for p in d.glob('*_redraw.npz'):
        if '_phase_' in p.name:continue
        roi=p.stem.replace('_redraw','')
        with np.load(p) as a:z={k:a[k] for k in a.files}
        modified=False
        for model in ('FOS','Pal5_control') if roi.startswith('Pal') else ('FOS',):
            table=list(csv.DictReader((d/(roi+'_'+model+'_recovery.csv')).open()))
            for row in table:
                key=model+'_hess_'+row['variant']
                for index,col in enumerate(('area_on','area_off')):
                    if float(row[col])<=0 and np.isfinite(z[key][index]).any():
                        archive=execution/(p.stem+'_before_undefined_area_mask.npz')
                        if not archive.exists():shutil.copy2(p,archive)
                        z[key][index]=np.nan;modified=True;corrected.append(dict(ROI=roi,model=model,variant=row['variant'],zone=col))
        if modified:npz(p,**z)
    rec['undefined_area_correction']=dict(code=sha(Path(__file__)),parent_receipt=sha(receipt),cells=corrected,scope='only normalized Hess rates with zero independent random area masked; original raw source/count arrays, parameters and recovery decisions unchanged')
    rec['outputs']={p.name:sha(p) for p in d.iterdir() if p.is_file() and p.name!='complete.json'};save_json(marker,rec)


if __name__=='__main__':main()
