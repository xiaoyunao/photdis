import numpy as np
from photdis.selection_v2_3_1 import distance_fit_from_snr,selection_upper,draw_observed_parallax
from photdis.baseline_parallax import parallax_check


def test_selection_rule_not_global_snr_cut():
    error=np.array([.02,.04,.02,.02]);clean=np.ones(4,bool);astro=np.array([1,1,0,1],bool);spec=np.array([0,0,0,1],bool)
    hi=selection_upper(error,clean,astro,spec)
    assert hi[0]==20 and np.isposinf(hi[1:3]).all() and np.isnan(hi[3])
    snr=np.array([[-5,19.9,20,25]]*4)
    fit=distance_fit_from_snr(snr,error[:,None],clean[:,None],astro[:,None],spec[:,None])
    np.testing.assert_array_equal(fit,[[0,0,1,1],[0,0,0,0],[0,0,0,0],[1,1,1,1]])


def test_forward_selected_gaussian_and_conditional_cdf():
    from scipy.special import ndtr
    # Near-delta distance: parallax 1mas; select observed/error<2, i.e. <=1mas.
    pdf=np.ones((1,1));edges=np.array([4.999999,5.000001]);g=np.array([15.]);sig=np.array([.5]);obs=np.array([.5])
    raw=parallax_check(pdf,edges,g,obs,sig,np.array([-np.inf]),np.array([np.inf]))
    cond=parallax_check(pdf,edges,g,obs,sig,np.array([-np.inf]),np.array([2.]))
    np.testing.assert_allclose(raw['pit'],ndtr(-1),atol=1e-10)
    np.testing.assert_allclose(cond['pit'],2*ndtr(-1),atol=1e-10)
    np.testing.assert_allclose(cond['selection_probability'],.5,atol=1e-10)
    draws=draw_observed_parallax(pdf,edges,g,sig,np.random.default_rng(4),repeats=100000)[0]
    accepted=draws<1
    assert abs(accepted.mean()-.5)<.006
    assert abs((draws[accepted]<.5).mean()-cond['pit'][0])<.008
