#!/usr/bin/env python3
"""Source-count, area, fixed-population and figure-data conservation checks."""
from photdis.stage1 import require_compute
from stream_common import *
from v3_10_figures import regression
from photdis_figure_style import STYLE


def rows(path):return list(csv.DictReader(path.open()))


def main():
    require_compute();d=folder('validation');checks=[]
    for stage in ('rgb','local_source','local_area','local','core','sequence','conditioning'):
        p=OUT/stage;r=json.loads((p/'complete.json').read_text())
        for name,h in r['outputs'].items():verified(p/name,h)
        checks.append(dict(check='all output hashes',stage=stage,n=len(r['outputs']),passed=True))
    # Every grouped source is admitted PSF and unique; coordinate/native
    # ownership was established at extraction, before any CMD condition.
    aliases=[]
    for p in (OUT/'local_source').glob('Pal5*.npz'):
        with np.load(p) as a:
            assert (a['TYPE']=='PSF').all();assert len(np.unique(a['ls_id']))==len(a['ls_id'])
            gaia=a['source_id'];gaia=gaia[gaia>0];aliases.append(dict(ROI=p.stem,PSF=len(a['ls_id']),positive_gaia=len(gaia),duplicate_positive_Gaia=len(gaia)-len(np.unique(gaia))))
    for p in (OUT/'local_source').glob('Sgr*.npz'):
        with np.load(p) as a:
            assert (a['TYPE']=='PSF').all();assert len(np.unique(a['ls_id']))==len(a['ls_id'])
            gaia=a['source_id'];gaia=gaia[gaia>0];aliases.append(dict(ROI=p.stem,PSF=len(a['ls_id']),positive_gaia=len(gaia),duplicate_positive_Gaia=len(gaia)-len(np.unique(gaia))))
    csv_rows(d/'identity_audit.csv',aliases)
    for p in (OUT/'rgb').glob('rgb_*.npz'):
        with np.load(p) as a:
            mask=a['mask'];assert a['rgb'].shape==(2,*mask.shape,3)
            assert np.isfinite(a['rgb']).all();assert np.allclose(a['rgb'][:,~mask],.88)
            assert np.isnan(a['density'][:,~mask]).all();assert np.isnan(a['signed_residual'][:,~mask]).all()
        checks.append(dict(check='RGB display mask and physical units retained',stage=p.stem,passed=True))
    segments=rows(OUT/'local/segment_excess.csv');response=rows(OUT/'local/modulus_response.csv')
    for key in {(r['roi'],r['model'],r['phase'],r['variant']) for r in segments}:
        ss=[r for r in segments if tuple(r[k] for k in ('roi','model','phase','variant'))==key]
        rr=[r for r in response if tuple(r[k] for k in ('roi','model','phase','variant'))==key and abs(float(r['mu'])-float(ss[0]['mu']))<1e-9];assert len(rr)==1
        assert sum(int(r['n_on']) for r in ss)==int(rr[0]['on']);assert sum(int(r['n_off']) for r in ss)==int(rr[0]['off'])
        np.testing.assert_allclose(sum(float(r['area_on']) for r in ss),float(rr[0]['on_area']),atol=1e-10)
    checks.append(dict(check='all raw segment counts and random areas conserve at exact reference mu',n=len(segments),passed=True))
    # Both weighting designs use precisely the same registered source IDs and
    # support, even when one propensity refit changes weight or positivity.
    with np.load(OUT/'conditioning/weighted_ids.npz') as a:
        for ref in np.unique(a['reference']):
            for lo in (2,5):
                for domain in np.unique(a['domain']):
                    g=(a['reference']==ref)&(a['lo']==lo)&(a['domain']==domain)
                    ii=[np.flatnonzero(g&(a['weight_design']==k)) for k in ('without_distance','with_distance')]
                    if not len(ii[0]):continue
                    np.testing.assert_array_equal(a['ls_id'][ii[0]],a['ls_id'][ii[1]]);np.testing.assert_array_equal(a['common'][ii[0]],a['common'][ii[1]])
                    assert (a['weight'][ii[0]][~a['common'][ii[0]]]==0).all()
    checks.append(dict(check='same IDs and same support in both conditioning designs',passed=True))
    with np.load(OUT/'sequence/predictions.npz') as a,np.load(OUT/'sequence/shared_bootstrap.npz') as b:
        assert (a['TYPE']=='PSF').all() and len(np.unique(a['ls_id']))==len(a['ls_id'])
        np.testing.assert_array_equal(a['ls_id'],b['ls_id']);assert b['multiplicity'].shape==(256,len(a['ls_id']))
        for name in np.unique(a['system']):
            assert b['scale_'+name].shape==(256,) and (b['scale_'+name]>0).all()
        cal=(a['system']=='M13')&(a['reference_phase']=='lower_MS')&~a['mask']&a['new_common'];assert cal.sum()==json.loads((OUT/'sequence/calibration.json').read_text())['n']
    checks.append(dict(check='verified PSF members, one common system-scale vector and one member multiplicity matrix for every group',passed=True))
    (d/'v38_pixels').mkdir(parents=True,exist_ok=True)
    with plt.rc_context(STYLE):r=regression(d/'v38_pixels')
    save_json(d/'validation.json',dict(checks=checks,style_pixel_regression=r,scope='scientific conservation plus original v3.8 pixel baseline; figure visual/PDF audit separately recorded'))
    complete(d,{},checks=len(checks),science_passed=True)


if __name__=='__main__':main()
