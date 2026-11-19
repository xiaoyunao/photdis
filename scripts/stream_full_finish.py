#!/usr/bin/env python3
"""Conserve the conditional full-footprint source and node accounting."""
from photdis.stage1 import require_compute
from stream_common import *
from stream_full import FH,gate


def main():
    require_compute();gate();d=folder('full_validation');inputs={}
    with np.load(OUT/'full_replay/fixed_error_curve.npz') as a,np.load(OUT/'local/Pal5_tail_redraw.npz') as b:
        for key in ('g_centers','sigma_g'):np.testing.assert_array_equal(a[key],b[key])
    roster=list(csv.DictReader((OUT/'full_inventory/raw_to_gr_complete.csv').open()));counts={}
    fields=('raw','missing_TYPE','non_PSF','PSF','owned','latitude_E010','clean_mask','gr_NOBS','gr_error','gr_masked','gr_fracin','gr_fracflux','gr_anymask','gr0_lt23','gr_complete_depth')
    for side in ('south','north'):
        rows=[r for r in roster if r['side']==side]
        counts[side]={key:sum(int(r[key]) for r in rows) for key in fields}
        assert all(counts[side][a]>=counts[side][b] for a,b in zip(fields[3:],fields[4:]))
        assert counts[side]['raw']==counts[side]['missing_TYPE']+counts[side]['non_PSF']+counts[side]['PSF']
    identity=json.loads((OUT/'full_identities/identity_counts.json').read_text());part=list(csv.DictReader((OUT/'full_count/source_node_counts.csv').open()))
    assert sum(counts[s]['gr_complete_depth'] for s in counts)==identity['native_unique']
    assert sum(int(r['sources']) for r in part)==identity['admitted_after_aliases']
    nodes=read(OUT/'full_count',('native_nodes.npz','mu','totals'),inputs)
    assert sum(int(r['acceptances']) for r in part)==int(nodes['totals'].sum())
    cube=json.loads((OUT/'full_atlas/complete.json').read_text());inputs['full_cube_receipt']=sha(OUT/'full_atlas/complete.json')
    from astropy.io import fits
    with fits.open(verified(OUT/'full_atlas/cmd_filter_cube.fits',cube['outputs']['cmd_filter_cube.fits']),memmap=True) as h:
        actual=h['NATIVE_COUNTS'].data.sum(axis=2);np.testing.assert_array_equal(actual,nodes['totals'])
        assert (h['COUNTS'].data[0,0]==0).all();np.testing.assert_array_equal(h['COUNTS'].data[1,0].sum(axis=1),actual.sum(axis=0))
    tables=[dict(side=s,stage=k,retained=v) for s,c in counts.items() for k,v in c.items()];csv_rows(d/'native_cutflow_totals.csv',tables)
    csv_rows(d/'node_totals.csv',[dict(side=('south','north')[s],mu=float(mu),accepted=int(nodes['totals'][s,i])) for s in (0,1) for i,mu in enumerate(nodes['mu'])])
    provenance=[]
    for p in sorted((OUT/'full_cache').glob('*/complete.json')):
        r=json.loads(p.read_text());provenance.append(dict(part=p.parent.name,receipt=sha(p),**r));assert r['full_config']==FH
    save_json(d/'source_provenance.json',provenance)
    seam=ROOT/'data/v3_9/delivery/seam';r=json.loads((seam/'complete.json').read_text());verified(seam/'ownership_overrides.npz',r['outputs']['ownership_overrides.npz']);inputs['ownership_overrides']=sha(seam/'ownership_overrides.npz')
    save_json(d/'full_summary.json',dict(sweeps=len(roster),native_counts=counts,identities=identity,
        CMD_union=sum(int(r['union']) for r in part),node_acceptances=int(nodes['totals'].sum()),mean_node_acceptances_per_union=int(nodes['totals'].sum())/sum(int(r['union']) for r in part),
        area_deg2=cube['eligible_area_deg2'],display_area_deg2=cube['display_area_deg2'],cube_sha256=cube['outputs']['cmd_filter_cube.fits'],cube_bytes=(OUT/'full_atlas/cmd_filter_cube.fits').stat().st_size,
        cube_axes='NATIVE_COUNTS[native,mu,pixel], COUNTS[1,0,mu,pixel]=combinedMS_TO; COUNTS[0,0] unused zero compatibility slot, not a bright sample; NATIVE_AREAS three identical eligibility slots, gr_complete area in index2',
        role='conditional full-footprint CMD-response imaging; these are not newly measured distances or spectral phase labels',north_scope='native template support and seam ownership inherited; independent local stream recovery in DECam only'))
    for stage in ('full_replay','full_inventory','full_identities','full_count','full_area/south','full_area/north','full_atlas','full_rgb'):
        r=json.loads((OUT/stage/'complete.json').read_text());inputs[stage]=sha(OUT/stage/'complete.json')
        for name,h in r['outputs'].items():verified(OUT/stage/name,h)
    complete(d,inputs,scientific_count_conservation=True,duplicate_admission_removed=True,production_model_unchanged=C['production'])


if __name__=='__main__':main()
