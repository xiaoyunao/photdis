import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from photdis.audit import checked_gaia_color, rvs_metallicity, ps1_decam, failure_groups
from photdis.photometry import derive_photometry
from test_photometry import catalog

class AuditTests(unittest.TestCase):
    def test_missing_photometry_and_real_zero_color(self):
        d={"gaia_match":np.ones(4,bool),"bp_rp":np.array([0.,0.,1.,0.])}
        for b in ("bp","rp"):
            d[f"phot_{b}_mean_flux"]=np.array([0.,100.,100.,100.])
            d[f"phot_{b}_mean_flux_error"]=np.ones(4)
            d[f"phot_{b}_n_obs"]=np.array([0,10,10,10])
            d[f"phot_{b}_mean_mag"]=np.array([0.,15.,15.,15.])
        d["phot_rp_mean_mag"][2]=14.
        d["phot_bp_mean_mag"][3]=16.
        result=checked_gaia_color(d)
        self.assertTrue(np.isnan(result["bp_rp_checked"][0]))
        self.assertEqual(result["bp_rp_checked"][1],0.)
        np.testing.assert_array_equal(result["bp_rp_catalog_disagrees"],[False,False,False,True])
    def test_author_metallicity_formula_and_bounds(self):
        raw=np.full(6,-1.)
        cal,good=rvs_metallicity(raw,np.array([5000,5000,4199,6600,4200,5000]),np.array([3.,4.5,4.5,4.5,1.8,5.5]))
        np.testing.assert_array_equal(good,[True,True,False,False,True,False])
        np.testing.assert_allclose(cal[:2],[-1.197,-1.039])
        np.testing.assert_array_equal(raw,np.full(6,-1.))
    def test_joint_failures_partition_overlapping_flags(self):
        raw=catalog(7)
        raw["MASKBITS"][[1,3,4,5,6]]=1<<4
        raw["FRACMASKED_Z"][[2,3,4]]=.5
        raw["MASKBITS"][4] |= 1<<7
        raw["FLUX_IVAR_R"][5]=0
        raw["FRACIN_G"][6]=.1
        phot=derive_photometry(raw)
        out=failure_groups(raw,phot)
        np.testing.assert_array_equal(out["group"],[0,1,2,3,4,5,6])
        self.assertTrue(out["satur"][[1,3,4,5,6]].all())
    def test_decam_color_term_sign(self):
        mags=np.array([[16.,15.5,15.,14.8,14.7]])
        self.assertAlmostEqual(ps1_decam(mags,"g")[0],16+.00062+.03604+.01028-.00613)
        self.assertAlmostEqual(ps1_decam(mags,"z")[0],14.8+.02583-.07690+.02824-.00898)
if __name__=="__main__": unittest.main()
