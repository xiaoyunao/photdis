import numpy as np
import pytest

from photdis.cluster_rgb_external import (
    observe, matched_errors, cluster_bands, identity_hash, abundance_conditions,
    prior_only, verify_hashes)
from photdis.cluster_rgb_conditional import RG, RR, RZ


def test_shared_photometry_and_reference_shift_keep_sign_and_covariance():
    c = np.zeros((2, 2)); m = np.array([1., 2.])
    noise = np.array([[.03, -.02, .01], [-.01, .02, -.04]])
    zero = np.array([.01, .02, -.01])
    colors, ref = observe(c, m, noise, zero=zero, dust=.01, dmu=.1)
    shift = noise + zero - .01*np.array([RG, RR, RZ])
    np.testing.assert_allclose(colors[:, 0], shift[:, 0]-shift[:, 1])
    np.testing.assert_allclose(colors[:, 1], shift[:, 1]-shift[:, 2])
    np.testing.assert_allclose(ref-m, shift[:, 0]-.1)


def test_actual_error_matching_preserves_band_vectors_and_nearest_empty_bin():
    errors = np.array([[1,2,3],[4,5,6]])*.001
    got = matched_errors([.7, 1.2, 2.7], [.7, 2.7], errors, np.random.default_rng(1))
    np.testing.assert_array_equal(got, errors[[0,0,1]])


def test_cluster_bands_do_not_shrink_shared_shift_with_star_count():
    values=[]
    for n, shift in [(10,-.1),(1000,.1)]:
        row=dict(n=n,median=shift,nmad=.02,cover68=.7,cover95=.95,width68=.4,width95=.9)
        values.append(dict(overall=row,bins={'a':row}))
    result=cluster_bands(values)
    np.testing.assert_allclose(result['overall']['median'],[-.095,0,.095])


def test_identity_rejects_duplicates_and_ignores_order():
    assert identity_hash([3,1,2])==identity_hash([1,2,3])
    with pytest.raises(ValueError): identity_hash([1,1])


def test_frozen_file_change_is_rejected(tmp_path):
    p=tmp_path/'input';p.write_text('changed')
    with pytest.raises(ValueError):verify_hashes({str(p):'wrong'})


def test_abundance_quadrature_has_registered_mean_and_variance():
    cfg=dict(feh=-1.7,feh_sigma=.2,feh_alternatives={},unknown_weights={})
    z,w=abundance_conditions(cfg)['reference'];z=np.array(z)
    np.testing.assert_allclose(np.dot(z,w),-1.7)
    np.testing.assert_allclose(np.dot((z+1.7)**2,w),.2**2)


def test_prior_only_analytic_bounds_and_outside_score():
    stats,q=prior_only([1,2.5])
    np.testing.assert_allclose(q[0],[.5625,.9,1.75,2.6,2.9375])
    np.testing.assert_allclose(stats['overall']['log_density'],-np.log(2.5))
    assert prior_only([.49])[0]['overall']['log_density'] is None
