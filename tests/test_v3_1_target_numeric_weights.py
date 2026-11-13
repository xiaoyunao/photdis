import numpy as np
from v3_1_target_counts import weight_audit


def test_numerical_weights_keep_constants_and_detect_collapse():
    values=np.repeat(np.linspace(-.2,.2,128),2)
    weights,audit=weight_audit(values)
    shifted,other=weight_audit(values-20000)
    np.testing.assert_allclose(weights,shifted,atol=1e-13,rtol=0)
    assert audit['passed'] and other['passed'] and audit['half_log_normalizer_delta']==0
    collapsed=np.zeros(256);collapsed[0]=100
    weight,failure=weight_audit(collapsed)
    assert not failure['passed'] and failure['importance_ess']<2
    assert weight[0]>.99999
