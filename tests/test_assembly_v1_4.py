import unittest
import numpy as np
from photdis.assembly_v1_4 import epoch_positions,positional_join,representative_rows,uppercase_fields


def coords(ra,dec=None,epoch=2016,pmra=None):
    ra=np.asarray(ra,float);n=len(ra)
    return dict(ra=ra,dec=np.zeros(n) if dec is None else np.asarray(dec),epoch=np.full(n,epoch),pmra=np.zeros(n) if pmra is None else np.asarray(pmra),pmdec=np.zeros(n))


class AssemblyTests(unittest.TestCase):
    def test_lowercase_fits_schema_preserves_data(self):
        t=np.zeros(2,dtype=[("ra","f8"),("dec","f8"),("flux","f4",(3,))])
        t["ra"]=[1,2];t["flux"]=3
        u=uppercase_fields(t)
        np.testing.assert_array_equal(u["RA"],[1,2])
        self.assertEqual(u["FLUX"].shape,(2,3));self.assertTrue(np.shares_memory(t,u))

    def test_wrap_and_motion(self):
        s=coords([359.999],epoch=2000,pmra=[450.])
        c=coords([.001])
        ix,sep,unknown,amb=positional_join(s,c)
        self.assertEqual(ix[0],0);self.assertLess(sep[0],1e-5);self.assertFalse(unknown[0])
    def test_ambiguity_and_collisions(self):
        ix,*_=positional_join(coords([10]),coords([10,10+.8/3600]))
        self.assertEqual(ix[0],-1)
        ix,*_=positional_join(coords([10,10+.1/3600]),coords([10]))
        self.assertTrue((ix==-1).all())
    def test_unknown_epoch_is_conservative(self):
        ix,_,unknown,_=positional_join(coords([10,20],epoch=0),coords([10+.3/3600,20+.7/3600]))
        np.testing.assert_array_equal(ix,[0,-1]);self.assertTrue(unknown[0])
    def test_representative_is_label_independent(self):
        ix=representative_rows(np.array([3,2,3,2]),np.array([0,1,1,1]),np.array([100,5,1,10]))
        np.testing.assert_array_equal(ix,[3,2])

if __name__=='__main__':unittest.main()
