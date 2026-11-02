import numpy as np
from photdis.kgiant_distance import COLOR, predict_mixture, evaluate_mixture


def test_joint_g_conditioning_matches_full_gaussian():
    bands=np.array([[17.,16.,15.,14.]])
    e=np.array([[.12,.08,.06,.04]]); qe=np.array([.15,.09,.07,.05])
    query=bands[0]+np.array([.2,-.1,.05,0])
    result=predict_mixture(bands,e,[16.],[.25],query,qe)
    # Independent construction in the original four-band coordinates.
    transform=np.vstack([COLOR,[1,0,0,0]])
    joint=transform @ np.diag(e[0]**2+qe**2) @ transform.T
    joint[:3,:3]+=.12**2*np.eye(3);joint[3,3]+=.25**2+.15**2
    c=joint[3,:3] @ np.linalg.inv(joint[:3,:3])
    expected_z=1+c @ (COLOR @ (query-bands[0]))
    assert np.allclose(result['means'],query[0]-expected_z)
    assert np.allclose(result['sigmas']**2,joint[3,3]-c @ joint[:3,3])
    # The shared-g correction is large enough to detect a missing/wrong-sign term.
    assert abs(expected_z-1)>.01


def test_no_color_preserves_training_mixture_and_quantiles():
    bands=np.array([[17.,16.,15.,14.],[19.,18.,17.,16.]])
    out=predict_mixture(bands,np.ones((2,4))*.03,[16.,16.],[.2,.2],
                        [18,17,16,15],[.04]*4,dimension=0)
    assert np.allclose(out['weights'],[.5,.5])
    assert np.allclose(out['means'],[17,15])
    assert np.allclose(out['sigmas']**2,.03**2+.04**2+.2**2+.15**2)
    result=evaluate_mixture(out,16,.1)
    assert abs(result['quantiles'][2]-16)<1e-5
    assert np.isclose(result['convolved_pit'],.5)
    assert result['raw_cover95']


def test_unused_w1_does_not_poison_optical_model():
    b=np.array([[17.,16.,15.,np.nan],[18.,17.2,16.5,np.nan]])
    e=np.array([[.02,.03,.04,np.nan],[.03,.04,.05,np.nan]])
    out=predict_mixture(b,e,[16,16],[.2,.3],[17.5,16.6,15.8,np.nan],
                        [.02,.03,.04,np.nan],dimension=2)
    assert np.all(out['sigmas']>0) and np.isclose(out['weights'].sum(),1)
