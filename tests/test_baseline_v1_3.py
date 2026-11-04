import numpy as np
from photdis.baseline_v1_3 import class_eligible,evolved_candidate


def test_class_evidence_does_not_require_distance():
    t=dict(class_label=np.array([1,1,-1,1,1]),identity_conflict=np.array([0,1,0,0,0],bool),
           nonstellar_risk=np.array([0,0,0,1,0],bool),g0=np.array([20,20,20,20,np.nan]),
           r0=np.ones(5)*19,z0=np.ones(5)*18,Mg=np.full(5,np.nan),fit_eligible=np.zeros(5,bool))
    assert class_eligible(t).tolist()==[True,False,False,False,False]


def test_candidate_uses_prediction_and_support_only():
    p=dict(weights=np.array([[.02,.95,.02,.01],[.95,.02,.02,.01],[.1,.7,.1,.1],[.02,.95,.02,.01]]),
           main_index=np.array([1,0,1,1]),branch_support=np.ones((4,3),bool),supported=np.array([1,1,1,0],bool))
    assert evolved_candidate(p).tolist()==[True,False,False,False]
