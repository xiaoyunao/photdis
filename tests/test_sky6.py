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
    assert not domain(np.array([0.,15.,60.,60.]),np.array([1.,.2,.01,.01]),np.array([[24,24],[24,24],[21,24],[24,24]])).tolist()==[False,False,False,False]
    np.testing.assert_array_equal(domain(np.array([0.,15.,60.,60.]),np.array([1.,.2,.01,.01]),np.array([[24,24],[24,24],[21,24],[24,24]])),[False,False,False,True])


def test_display_rejects_invalid_gr_without_sky_cut():
    f=np.array([[10.,10.],[10.,-1],[10.,10.],[10.,10.]]);iv=np.full((4,2),.25);tr=np.ones((4,2));tr[2,0]=0;obs=np.ones((4,2));obs[3,1]=0
    use,_,_,_,_=entry(f,iv,tr,obs,np.zeros(4,'i8'),np.zeros((4,2)),np.ones((4,2)))
    np.testing.assert_array_equal(use,[True,False,False,False])
