#!/usr/bin/env python3
"""Compress exact plotted-grid arrays and publish a private raw-cube index."""
from photdis.stage1 import require_compute
from halo_common import *


def main():
    require_compute();d=folder('redraw');inputs={}
    marker=OUT/'atlas/complete.json';r=json.loads(marker.read_text());inputs['atlas']=sha(marker)
    rows=[]
    for name,h in r['outputs'].items():
        if not name.endswith('.npz') or name=='smoothed_maps.npz':continue
        p=verified(marker.parent/name,h)
        with np.load(p) as a:
            np.savez_compressed(d/name,**{k:a[k] for k in a.files})
        with np.load(p) as original,np.load(d/name) as compressed:
            assert original.files==compressed.files
            for key in original.files:np.testing.assert_array_equal(original[key],compressed[key])
        rows.append(dict(file=name,original_bytes=p.stat().st_size,compressed_bytes=(d/name).stat().st_size,
            source_sha256=h,compressed_sha256=sha(d/name),lossless=True))
    csv_rows(d/'compression.csv',rows)
    cube=OUT/'atlas/cmd_filter_cube.fits'
    save_json(d/'private_cube.json',dict(path=str(cube),bytes=cube.stat().st_size,sha256=r['outputs'][cube.name],
        nside=512,ordering='NESTED',coordsys='C',axes='COUNTS: bright/deep,all/score>=.8/PDF-common,mu,pixel',
        units=dict(COUNTS='PSFs',NODE_SUMW2='binary weight squared equals node counts',BROAD_SUMW2='sum of squared per-source overlapping-channel weights, main gate only',
          BROAD_WEIGHTS='unit-normalized overlap of each broad modulus interval with fixed mu cells',QRF_MASS='seven exclusive PDF integrals including below/above displayed range',
          NATIVE_AREAS='deg2: south/north,geometric/unmasked/depth20,pixel',DEPTH='mean 0.1mag g/r/z sensitivity',EBV='mean random E(B-V)',
          PHASE_VISIBLE='bit0 MS,bit1 SGB-RGB; version,native,mu,pixel',DISPLAY_MASK='raw coverage>=.1,smoothed coverage>=.7'),
        sample='strict native imaging PSFs, unique ownership; both versions use shared area counted once per southern bright/faint union'))
    complete(d,inputs,scope='lossless zipped actual physical display-grid arrays; use halo_redraw.py with matching atlas CSV scales',array_count=len(rows))


if __name__=='__main__':main()
