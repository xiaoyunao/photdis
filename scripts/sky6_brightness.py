#!/usr/bin/env python3
"""Restore the user's previous v5 stretch without changing any physical map."""
import hashlib,shutil
from photdis.stage1 import require_compute
from sky6_common import *
from sky6_figures import rgb


def main():
    require_compute();config=ROOT/'configs/sky_sequence_display_20261119.json';settings=json.loads(config.read_text());source=ROOT/settings['stretch_source']
    prior=json.loads((source.parent/'complete.json').read_text());verified(source,prior['outputs'][source.name]);table=list(csv.DictReader(source.open()))
    inputs=dict(previous_v5_stretch=sha(source),display_settings=sha(config));stats=[];checks=[]
    archive=folder('execution/display_before_v5_brightness');groups=['overview','bin_6.3_10','bin_10.0_15','bin_15.0_26','bin_26.0_50','bin_50.0_100']
    for stage in ('rgb','old_domain','no_extinction'):
        d=OUT/stage;record=json.loads((d/'complete.json').read_text());saved=archive/stage;saved.mkdir(exist_ok=True)
        for path in d.iterdir():
            if path.is_file() and path.name!='physical_channels.npz' and not (saved/path.name).exists():shutil.copy2(path,saved/path.name)
        with np.load(verified(d/'physical_channels.npz',record['outputs']['physical_channels.npz'])) as z:density=z['density'];signed=z['signed_residual'];mask=z['mask']
        for path in sorted(d.glob('rgb_*.npz')):
            with np.load(verified(path,record['outputs'][path.name])) as z:a={k:z[k] for k in z.files}
            group=path.stem.removeprefix('rgb_');start=0 if stage=='no_extinction' else groups.index(group)*3
            old_hash={k:hashlib.sha256(np.ascontiguousarray(a[k]).tobytes()).hexdigest() for k in ('density','signed_residual','mask','l','b','channel_kpc')}
            stretch=[]
            for row,view in enumerate(('CMD density','Positive residual')):
                limits=[]
                for j,ch in enumerate('BGR'):
                    rows=[r for r in table if r['group']==group and r['view']==view and r['channel']==ch];assert len(rows)==1
                    lo,hi=[float(rows[0][k]) for k in ('lo','hi')];limits.append([lo,hi]);v=(density if row==0 else signed)[start+j]
                    stats.append(dict(stage=stage,group=group,kind=('density','positive_residual')[row],channel=ch,lo=lo,hi=hi,saturated_fraction=float((v[mask]>hi).mean()),stretch_source='unchanged previous v5 full_RGB'))
                stretch.append(limits)
            a['stretch']=np.array(stretch);npz(path,**a);rgb(a,d,path.stem)
            with np.load(path) as z:
                assert all(hashlib.sha256(np.ascontiguousarray(z[k]).tobytes()).hexdigest()==h for k,h in old_hash.items())
            checks.append(dict(stage=stage,group=group,unchanged_physical_arrays=old_hash,passed=True))
        metadata={k:v for k,v in record.items() if k not in ('config','job','inputs','code','style','outputs')};metadata['display_stretch']='exact previous v5 per-group/channel limits, requested by user'
        complete(d,dict(**record['inputs'],**inputs),**metadata)
    d=OUT/'display';csv_rows(d/'stretch.csv',stats);complete(d,inputs,render_style='previous v5 exact brightness, existing Pal5/v3.8 base, Galactic references with dashed Sagittarius and Pal5 tracks')
    d=folder('brightness');save_json(d/'unchanged_maps.json',dict(checks=checks,rule=settings,physical_data_unchanged=True));complete(d,inputs,restored_previous_v5_stretch=True)


if __name__=='__main__':main()
