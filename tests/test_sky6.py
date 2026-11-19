"""Direct scientific invariants for the explicitly expanded display entrance."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from sky6_scan import entry,domain


def test_display_admits_faint_noise_and_z_only_flags():
    f=np.full((4,2),10.);iv=np.full((4,2),.25);tr=np.ones((4,2));obs=np.ones((4,2));masked=np.zeros((4,2));fracin=np.ones((4,2))
    use,geom,mag,err,_=entry(f,iv,tr,obs,np.array([0,1<<4,1<<7,1<<3]),masked,fracin)
    np.testing.assert_array_equal(use,[True,True,True,False]);assert (err>.1).all()
    np.testing.assert_array_equal(domain(np.array([0.,15.,60.,60.]),np.array([1.,.2,.01,.01]),np.array([[24,24],[24,24],[21,24],[24,24]])),[False,False,False,True])


def test_display_rejects_invalid_gr_without_sky_cut():
    f=np.array([[10.,10.],[10.,-1],[10.,10.],[10.,10.]]);iv=np.full((4,2),.25);tr=np.ones((4,2));tr[2,0]=0;obs=np.ones((4,2));obs[3,1]=0
    use,_,_,_,_=entry(f,iv,tr,obs,np.zeros(4,'i8'),np.zeros((4,2)),np.ones((4,2)))
    np.testing.assert_array_equal(use,[True,False,False,False])


def test_original_segments_preserve_multiple_intersections():
    from sky6_ridge import intersections
    a=np.array([[3.,.2,.1],[4.,.3,.2],[5.,.2,.1]])
    found=intersections(a,np.array([0,1]),1,.25)
    np.testing.assert_allclose(found[:,0],[3.5,4.5])


def test_full_pdf_reference_convolution_retains_tiny_tail_scores():
    from sky6_width import uniform_convolution,mixture_probability
    p,score=uniform_convolution(np.array([[-1.,1.]]),np.array([[1.]]),np.array([0.]),.1)
    np.testing.assert_allclose(p,.5,atol=1e-12);np.testing.assert_allclose(score,-np.log(2),atol=1e-10)
    _,tail=uniform_convolution(np.array([[-1.,1.]]),np.array([[1.]]),np.array([10.]),.1)
    assert np.isfinite(tail).all() and tail[0]<-3000
    p,_=mixture_probability(np.array([0.]),np.array([[0.]]),np.array([[0.]]),np.array([.1]));np.testing.assert_allclose(p,.5)


def test_parallel_identity_proof_keeps_exact_alias_rule():
    import pytest
    from sky6_count import assert_unique_partitioned,gaia_aliases
    ids=np.array([(11010<<40)|(k<<20)|k for k in range(100)],dtype='i8')
    assert_unique_partitioned(ids,'test')
    with pytest.raises(AssertionError):assert_unique_partitioned(np.r_[ids,ids[35]],'duplicate')
    gaia=np.arange(100,dtype='i8')+1;gaia[40]=gaia[20];gaia[60]=gaia[20];gaia[80]=0
    np.testing.assert_array_equal(gaia_aliases(ids,gaia),ids[[40,60]])
