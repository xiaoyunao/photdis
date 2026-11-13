import numpy as np
import pytest
from photdis.v3_1.simulation import draw_narrow


def test_narrow_generator_preserves_sources_psf_and_observed_selection():
    library=dict(TYPE=np.full(6,'PSF'),class_label=np.tile([0,1,2],2),mask=np.ones((6,6),bool),
        flux=np.full((6,6),100.),identity=np.arange(10,16),feh=np.full(6,-1.),feh_valid=np.ones(6,bool))
    context=dict(TYPE=np.full(2,'PSF'),identity=np.arange(20,22),flux_error=np.full((2,6),.2),
        transmission=np.full((2,6),.9),band_mask=np.ones((2,6),bool),EBV=np.full(2,.01),
        photometric_system=np.full(2,'DECam_DR11'),RA=np.ones(2),DEC=np.ones(2),cell=np.ones(2,int),field=np.ones(2,int))
    p,t,info=draw_narrow(library,context,4.,.08,n=1000,seed=10)
    assert np.all(p['flux'][:,[0,1,3]]>=5*p['flux_error'][:,[0,1,3]])
    assert set(t['template_identity'])<=set(library['identity'])
    assert set(t['context_identity'])<=set(context['identity'])
    assert abs(t['mu'].mean()-(10+5*np.log10(4)))<.02
    assert abs(t['mu'].std()-.08)<.01
    assert (p['TYPE']=='PSF').all() and len(np.unique(p['identity']))==1000
    bad=dict(library);bad.pop('TYPE')
    with pytest.raises(ValueError,match='native PSF'):draw_narrow(bad,context,4.,.08)
