import numpy as np
import pytest
from photdis.v3.data import WEAK,SIGNIFICANT,MISSING
from photdis.v3_1.use_domains import (observable_cells,cell_descriptors,distance_availability,
    missing_reference_mass,computational_status,source_distances,N_CELLS)


def test_observable_grid_retains_signed_W1_and_separates_admission():
    mags=np.array([[18.4,18.1,18.0,17.8,18.,18.],[20.9,20.1,20.,19.8,20.,20.],[16.2,16.1,16.,15.8,16.,16.]])
    flux=10**((22.5-mags)/2.5); error=flux*.02
    states=np.full((3,6),SIGNIFICANT); states[:,4]=[WEAK,SIGNIFICANT,MISSING]
    flux[:,4]=[-2,10,np.nan]; error[:,4]=[1,1,np.nan]
    table=dict(identity=np.arange(3,dtype='i8'),TYPE=np.array(['PSF']*3),flux=flux,flux_error=error,
        transmission=np.ones((3,6)),band_mask=np.ones((3,6),bool),band_state=states)
    admission=dict(identity=table['identity'],TYPE=table['TYPE'],observed_r=mags[:,1],candidate=np.array([True,False,True]))
    result=observable_cells(table,admission)
    np.testing.assert_array_equal(result['wise_bin'],[1,2,0])
    np.testing.assert_array_equal(result['candidate'],[True,False,True])
    assert len(cell_descriptors())==N_CELLS and set(r['cell'] for r in cell_descriptors())==set(range(75))
    table['TYPE'][0]='DEV'
    with pytest.raises(ValueError,match='PSF'):observable_cells(table,admission)


def test_missing_reference_mass_is_availability_not_calibration_or_Gaia_inversion():
    e=dict(role=np.array(['new_sky_field','new_sky_field','new_source_seen_sky_field']),
        distance_kpc=np.array([3.,14.,14.]),distance_available=np.ones(3,bool),distance_precise=np.array([True,False,True]),
        specdis_distance=np.array([np.nan,14.,np.nan]),specdis_valid=np.array([False,True,False]),specdis_sigma_mu=np.array([np.nan,.1,np.nan]),
        spdist_distance=np.full(3,np.nan),spdist_valid=np.zeros(3,bool),spdist_sigma_mu=np.full(3,np.nan),p_corr=np.array([-1.,.01,.2]))
    cells=np.array([0,1,1]); available,rows=distance_availability(cells,e,'new_sky_field',admitted=np.array([True,False,True]))
    assert available.sum()==1 and available[0,1]
    assert all(r['source']!='gaia' for r in rows) and set(source_distances(e))=={'astronn','specdis','spdist'}
    probability=np.full((2,3,6),1/18)
    mass=missing_reference_mass(probability,np.array([0,1]),available)
    np.testing.assert_allclose(mass,[5/6,1.])
    broader,_=distance_availability(cells,e,'new_sky_field')
    assert broader[1,3] and broader.sum()==2


def test_computational_availability_never_certifies_calibration():
    result=computational_status([False,True,True],[False,False,True],True)
    assert list(result)==['outside_candidate_domain','candidate_prediction_missing','conditional_model_available']
    assert computational_status([True],[True],False)[0]=='numerical_flags_retained'
