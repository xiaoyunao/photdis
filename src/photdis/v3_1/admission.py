"""Exact observed-photometry candidate gate and separately retained reasons."""
import numpy as np
from photdis.v3.controls import optical_magnitudes
from photdis.baseline_v1 import support


def candidate_admission(table,qrf):
    n=len(table['identity'])
    if 'TYPE' not in table:raise ValueError('Missing TYPE fails scientific admission')
    types=np.char.strip(np.asarray(table['TYPE']).astype('U'))
    native_psf=(types=='PSF') & (table['photometric_system']=='DECam_DR11')
    mags,errors=optical_magnitudes(table)
    # Match the frozen pilot expression, including observed r rather than r0.
    observed_r=mags[:,1]-2.5*np.log10(table['transmission'][:,1])
    features=qrf.feature_matrix(mags)
    distance=qrf.nn.kneighbors((features-qrf.mean)/qrf.scale,n_neighbors=10)[0][:,-1]
    radius=distance<=qrf.radius
    box=np.all((features>=qrf.lo)&(features<=qrf.hi),axis=1)
    supported=radius & box
    np.testing.assert_array_equal(supported,support(qrf,mags))
    bright=observed_r<20; precise=errors.max(axis=1)<.1
    result=dict(identity=np.asarray(table['identity']),TYPE=types,
        candidate=native_psf & bright & precise & supported,
        native_psf=native_psf,observed_r=observed_r,r0=mags[:,1],
        maximum_grz_error=errors.max(axis=1),bright=bright,precise=precise,
        qrf_support=supported,qrf_radius_pass=radius,qrf_feature_box_pass=box,
        qrf_tenth_neighbor_distance=distance,qrf_radius=np.full(n,qrf.radius))
    return result
