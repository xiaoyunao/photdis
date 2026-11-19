#!/usr/bin/env python3
"""Separate z-only pixel vetoes from a truly gr-only imaging entrance."""
from photdis.stage1 import require_compute
from stream_common import *
from stream_extract import ROIS,VARS,footprints


def main():
    require_compute();d=folder('mask_revision');changes=[]
    for stage,israndom in (('local_source',False),('local_area',True)):
        source=OUT/stage;receipt=json.loads((source/'complete.json').read_text());archive=folder('execution')/'before_pure_gr_mask'/stage;archive.mkdir(parents=True,exist_ok=True)
        (archive/'complete.json').write_text((source/'complete.json').read_text());variants=[]
        for roi in ROIS:
            p=source/(roi+'.npz')
            with np.load(p) as a:t={k:a[k] for k in a.files}
            old=t['selected'];new=footprints(t,random=israndom);np.testing.assert_array_equal(old[[0,1,3,5]],new[[0,1,3,5]])
            assert np.all(old<=new);npz(archive/(roi+'_old_selected.npz'),selected=old)
            changes.extend(dict(stage=stage,ROI=roi,variant=v,before=int(a.sum()),after=int(b.sum()),added=int((b&~a).sum())) for v,a,b in zip(VARS,old,new))
            t['selected']=new;npz(p,**t)
            for v,keep in zip(VARS,new):
                row=dict(roi=roi,variant=v,n_random=int(keep.sum()),area_deg2=keep.sum()/receipt['density']) if israndom else dict(roi=roi,variant=v,n=int(keep.sum()))
                variants.append(row)
        if israndom:csv_rows(source/'areas.csv',variants)
        else:
            original=list(csv.DictReader((source/'source_cutflow.csv').open()))
            for row in original:
                if row['file']=='unique_owned' and row['stage'] in VARS:row['n']=next(x['n'] for x in variants if x['roi']==row['roi'] and x['variant']==row['stage'])
            csv_rows(source/'source_cutflow.csv',original)
        receipt['pure_gr_mask_revision']=dict(code=sha(Path(__file__)),parent=sha(archive/'complete.json'),config=CONFIG_SHA,gr_bits=[2,3,5,6,10],strict_z_addback_bits=[2,3,4,5,6,7,10],raw_measurements_and_IDs_unchanged=True)
        receipt['outputs']={p.name:sha(p) for p in source.iterdir() if p.is_file() and p.name!='complete.json'};save_json(source/'complete.json',receipt)
    csv_rows(d/'gr_pixel_mask_effect.csv',changes);complete(d,{},interpretation='all flux/error/ownership/TYPE conditions fixed; this isolates z-only pixel-mask vetoes from gr imaging. Original masks retained in execution history.')


if __name__=='__main__':main()
