from types import SimpleNamespace
import numpy as np
import pytest
from photdis.baseline_v1 import support
from photdis.v3.controls import optical_magnitudes
from photdis.v3_1.admission import candidate_admission


class Neighbors:
    def kneighbors(self,x,n_neighbors):
        assert n_neighbors==10
        return np.full((len(x),10),.5),np.zeros((len(x),10),int)


def test_candidate_gate_matches_frozen_expression_and_uses_observed_r():
    r=np.array([19.5,20.01,19.5,19.5,19.5])
    flux=np.repeat((10**((22.5-r)/2.5))[:,None],6,axis=1)
    errors=flux*.03*np.log(10)/2.5; errors[2]*=4
    table=dict(identity=np.arange(5,dtype='i8'),TYPE=np.array(['PSF','PSF','PSF','DEV','PSF']),
        photometric_system=np.array(['DECam_DR11']*4+['BASS_MzLS_DR9']),flux=flux,flux_error=errors,
        transmission=np.full((5,6),.9),band_mask=np.ones((5,6),bool))
    qrf=SimpleNamespace(feature_matrix=lambda x:x,nn=Neighbors(),mean=np.zeros(3),scale=np.ones(3),
        radius=1.,lo=np.zeros(3),hi=np.full(3,30.))
    result=candidate_admission(table,qrf)
    mag,error=optical_magnitudes(table)
    original=(table['TYPE']=='PSF') & (table['photometric_system']=='DECam_DR11') & \
        (mag[:,1]-2.5*np.log10(table['transmission'][:,1])<20) & (error.max(axis=1)<.1) & support(qrf,mag)
    np.testing.assert_array_equal(result['candidate'],original)
    np.testing.assert_array_equal(result['candidate'],[True,False,False,False,False])
    assert result['r0'][1]<20<result['observed_r'][1]
    np.testing.assert_allclose(result['observed_r'],r,atol=1e-14)
    del table['TYPE']
    with pytest.raises(ValueError,match='TYPE'):candidate_admission(table,qrf)
