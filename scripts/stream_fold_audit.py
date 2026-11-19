#!/usr/bin/env python3
"""Audit turnoff-fold noise frequencies in each actual native color system."""
import shutil
from photdis.baseline_v1 import qmc_noise
from photdis.stage1 import require_compute
from stream_common import *


def native_bounds(template,side):
    m=template[side];valid=(template['phase']==0)&(m[:,1]>=3)&(m[:,1]<=4.5)
    if side=='north':valid&=template['north_support']
    gr=m[:,0]-m[:,1];left=np.flatnonzero(valid[:-1]&valid[1:]&(np.diff(gr)>1e-7))
    if not len(left):return np.array([np.nan,np.nan])
    return np.array([np.min(gr[left]),np.max(gr[left+1])])


def frequency(m,e,ids,bounds):
    e=np.maximum(e,.001);result=np.zeros(len(ids))
    for draw in range(128):
        z=qmc_noise(ids,draw);gr=m[:,0]-m[:,1]+z[:,0]*e[:,0]-z[:,1]*e[:,1]
        result+=(gr>=bounds[0])&(gr<=bounds[1])
    return result/128


def main():
    require_compute();d=folder('sequence');archive=folder('execution');p=d/'predictions.npz'
    with np.load(p) as a:t={k:a[k] for k in a.files}
    with np.load(d/'template_redraw.npz') as a:template={k:a[k] for k in a.files}
    ids=np.where(t['source_id']>0,t['source_id'],-t['ls_id']);region={};old=t['fold_noise_fraction'].copy()
    for code,side in enumerate(('south','north')):
        bounds=native_bounds(template,side);ii=np.flatnonzero(t['native']==code);t['fold_noise_fraction'][ii]=frequency(t['m'][ii],t['err'][ii],ids[ii],bounds)
        region[side]=dict(gr_lo=float(bounds[0]),gr_hi=float(bounds[1]))
    before=archive/'sequence_before_native_fold.npz'
    if not before.exists():shutil.copy2(p,before)
    npz(p,**t)
    with np.load(before) as a,np.load(p) as b:
        for key in a.files:
            if key!='fold_noise_fraction':np.testing.assert_array_equal(a[key],b[key])
    table=list(csv.DictReader((d/'per_star_branches.csv').open()));byid={int(k):v for k,v in zip(t['ls_id'],t['fold_noise_fraction'])}
    for row in table:row['fold_noise_fraction']=byid[int(row['ls_id'])]
    csv_rows(d/'per_star_branches.csv',table);save_json(d/'native_fold_regions.json',dict(regions=region,changed_rows=int((old!=t['fold_noise_fraction']).sum()),definition='native phase0 redward turnoff-hook color range in original physical order, inside fixed reference-Mr3..4.5 turnoff/support interval; 128 identity-matched g/r Sobol perturbations with the same0.001mag numerical floor as the likelihood',distance_predictions_unchanged=True))
    r=json.loads((d/'complete.json').read_text());r['native_fold_diagnostic']=dict(code=sha(Path(__file__)),parent_prediction=sha(before),scope='fold frequency corrected for actual north native color transform; all distance/posterior/zero-point/group-statistic values unchanged');r['outputs']={p.name:sha(p) for p in d.iterdir() if p.is_file() and p.name!='complete.json'};save_json(d/'complete.json',r)


if __name__=='__main__':main()
