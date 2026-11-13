import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import norm
from photdis.v3.inference import DISTANCE_BASIS_MEAN,DISTANCE_BASIS_SIGMA
from photdis.v3.likelihood import trapezoid_weights
from photdis.v3_1.candidate_modes import population_pdf,basin_partition,basin_joint_mass,describe_basins
from photdis.v3_1.gaussian_joint import interval_linear_weights
from photdis.v3_1.luminosity_log import log_luminosity_transform


@pytest.mark.parametrize('offset',[0.,-20000.])
def test_exact_population_mode_adjoint_matches_explicit_forward_components(offset):
    grid=np.linspace(4,24,501)
    native=np.array([[norm.logpdf(grid,m,s)+offset for m,s in [(10,.4),(16,.6)]] for _ in range(2)])
    native[1]=np.array([norm.logpdf(grid,m,s)+offset for m,s in [(10.75,.4),(16.75,.6)]])
    conditional=np.array([[.8,.1,.4],[.2,.9,.6]]);binary=np.array([.9,.1])
    class_ll=np.array([logsumexp(native+np.log(binary)[:,None,None]+np.log(conditional[:,c])[None,:,None],axis=(0,1)) for c in range(3)])
    parameters=np.array([[[.2,.15],[-.3,.2],[0.,0.]],[[.1,.3],[.4,.1],[-.2,.2]]])
    rng=np.random.default_rng(81);theta=rng.dirichlet(np.ones(15),size=(2,4)).reshape(2,4,3,5);lw=np.log([.3,.7])
    pdf=population_pdf(class_ll,grid,parameters,theta,lw);boundaries=np.array([11.17,14.13,17.46])
    actual=basin_joint_mass(native,grid,conditional,binary,parameters,lw,pdf,boundaries)
    weights=interval_linear_weights(grid,boundaries);radial=norm.logpdf(grid[None],DISTANCE_BASIS_MEAN[:,None],DISTANCE_BASIS_SIGMA[:,None])
    explicit=np.zeros_like(actual)
    for h in range(2):
        for c in range(3):
            transformed=log_luminosity_transform(native,grid,*parameters[h,c])
            density=transformed+logsumexp(pdf['log_coefficient'][h,c,:,None]+radial,axis=0)[None,None]
            density+=lw[h]+np.log(binary)[:,None,None]+np.log(conditional[:,c])[None,:,None]
            density-=pdf['state_log_normalization'][h]+pdf['log_normalization']
            explicit[:,:,:,c]+=np.einsum('bkg,mg->mbk',np.exp(density),weights)
    np.testing.assert_allclose(actual,explicit,atol=2e-11,rtol=2e-10)
    assert abs(actual.sum()-1)<2e-10
    np.testing.assert_allclose(actual.sum(axis=(1,2)),(np.exp(pdf['class_logpdf'])@weights.T).T,atol=2e-10)
    assert pdf['joint_delta']<1e-10
    peak,split=basin_partition(pdf['logpdf'],grid)
    modes=basin_joint_mass(native,grid,conditional,binary,parameters,lw,pdf,split)
    records,pairs=describe_basins(peak,split,grid,modes,np.array([0,1]))
    assert len(records)>=2 and len(pairs)>=1
    assert abs(sum(r['mass'] for r in records)-1)<2e-10
    for r in records:assert np.isclose(sum(r['parent_probability']),1) and np.isclose(sum(r['class_probability']),1)


def test_parent_aggregation_and_weak_modes_are_retained():
    masses=np.zeros((3,2,3,3));masses[0,0,0,0]=.48;masses[1,1,1,0]=.49;masses[2,0,2,1]=.03
    records,pairs=describe_basins(np.array([1.,3.,5.]),np.array([2.,4.]),np.arange(7.),masses,np.array([0,0,1]))
    assert len(records)==3 and records[-1]['weak_basin']
    assert pairs==[dict(first=0,second=1,different_class=False,same_class_different_binary=True,same_class_different_parent=False)]
    with pytest.raises(ValueError):describe_basins(np.arange(3.),np.array([2.,4.]),np.arange(7.),masses,np.array([-1,0,1]))


def test_saved_evidence_subtraction_roundoff_is_retained_not_reweighted():
    grid=np.linspace(-20,50,701);native=np.array([norm.logpdf(grid,m,1.) for m in [10,14,17]])
    theta=np.full((2,2,3,5),1/15);parameters=np.zeros((2,3,2));lw=np.log([.4,.6])-1.6e-12;original=lw.copy()
    result=population_pdf(native,grid,parameters,theta,lw)
    reference=population_pdf(native,grid,parameters,theta,np.log([.4,.6]))
    np.testing.assert_array_equal(lw,original)
    assert abs(float(result['state_weight_log_normalization'])+1.6e-12)<1e-15
    np.testing.assert_allclose(result['class_logpdf'],reference['class_logpdf'],atol=1e-12,rtol=0)
    with pytest.raises(ValueError):population_pdf(native,grid,parameters,theta,lw-1e-6)
