#!/usr/bin/env python3
"""Verify scientific conservation, actual delivery files and immutable plate style."""
from PIL import Image
from astropy.io import fits
from photdis.stage1 import require_compute
from halo_common import *
from v3_10_figures import regression,STYLE

STAGES=('native','parent','power','noise_fixed','noise_imaging','roles','balance','balance_comparison',
        'sequence','sequence_views','hess','hess_views','known','seam','selection','atlas','redraw','reference_scales')


def table(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    require_compute();d=folder('delivery');inputs={};checks=[];plates=[]
    for stage in STAGES:
        p=OUT/stage;r=json.loads((p/'complete.json').read_text())
        assert r['style']==check_templates()
        assert r['config']==CONFIG_HASH
        for name,h in r['outputs'].items():
            verified(p/name,h)
        inputs[stage]=sha(p/'complete.json')
        for png in sorted(p.glob('*.png')):
            with Image.open(png) as image:
                image.verify()
            with Image.open(png) as image:
                assert image.width>=2000 and abs(image.info['dpi'][0]-300)<.03
            assert png.with_suffix('.pdf').read_bytes().startswith(b'%PDF')
            assert png.with_suffix('.csv').exists()
            plates.append(str(png.relative_to(OUT)))
        checks.append(dict(stage=stage,outputs=len(r['outputs'])))
    assert all(int(r['unlinked'])==0 for r in table(OUT/'roles/role_links.csv'))
    rr=table(OUT/'power/primary_replay.csv')
    gaia=next(r for r in rr if r['reference']=='gaia')
    np.testing.assert_allclose([float(gaia[k]) for k in ('mean_log_score','coverage_68','coverage_95')],
                              [.3400042823542448,.6455078125,.94091796875],atol=1e-12)
    branch=table(OUT/'power/branch_conditioning.csv')[0]
    np.testing.assert_allclose(float(branch['analytic_selected_fraction']),float(branch['joint_component_selected_fraction']),atol=1e-14)
    with fits.open(OUT/'atlas/cmd_filter_cube.fits',memmap=True,checksum=True) as h:
        assert h[0].header['NSIDE']==512 and h[0].header['ORDERING']=='NESTED' and h[0].header['COORDSYS']=='C'
        np.testing.assert_allclose(h['BROAD_WEIGHTS'].data.sum(axis=1),1.)
        np.testing.assert_allclose(h['MU'].data,MU)
        for v in (0,1):
            np.testing.assert_array_equal(h['NODE_SUMW2'].data[v],h['COUNTS'].data[v,0])
            assert (h['COUNTS'].data[v,1:]<=h['COUNTS'].data[v,:1]).all()
    with plt.rc_context(STYLE):
        baseline=regression(d)
    csv_rows(d/'validation.csv',checks)
    complete(d,inputs,plates=plates,plate_count=len(plates),regression=baseline,
             pixel_comparison='both original v3.8 PNGs match exactly',
             production='8706940843975e9f9af657ea298aa5b7c94f64bc unchanged',
             scientific_scope='numerical validity independent of whether scale/recovery/balance criteria pass')


if __name__=='__main__':main()
