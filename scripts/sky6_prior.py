#!/usr/bin/env python3
"""Overlay actual Galactic references on the unchanged previous v5 RGB."""
from photdis.stage1 import require_compute
from stream_common import track
from sky6_common import *
from sky6_figures import rgb,physical


def main():
    require_compute();d=folder('prior_display');inputs={}
    a=read(STREAM/'full_rgb',('rgb_overview.npz','l','b','rgb','density','signed_residual','mask','channel_kpc'),inputs)
    for name in C['annotations']['tracks']:
        _,path=track(name);inputs[str(path.relative_to(ROOT))]=sha(path)
    a.update(scope=np.array('Exact previous v5 full RGB pixels and physical arrays; only Galactic references added'),label=np.array('Previous high-latitude CMD sky'))
    npz(d/'rgb_overview.npz',**a);rgb(a,d,'rgb_overview');physical(a,d)
    latitude=np.rad2deg((a['b'][:-1]+a['b'][1:])/2)
    save_json(d/'latitude_support.json',dict(covered_display_cells=int(a['mask'].sum()),covered_abs_b_below29=int(a['mask'][abs(latitude)<29].sum()),covered_abs_b_below20=int(a['mask'][abs(latitude)<20].sum()),scope='Exact previous v5 mask; a displayed bright band wholly outside these empty low-latitude cells cannot be the Galactic plane. CMD compatibility alone does not identify stream membership.'))
    complete(d,inputs,scope='unchanged v5 display data/RGB/stretch, add requested Galactic latitude and actual galstreams overlays')


if __name__=='__main__':main()
