import numpy as np
from photdis.report_v2 import percent_metrics, ranges


def test_percentage_metrics_are_computed_per_star_and_keep_tails():
    ref=np.array([10.,10.,10.,np.nan])
    delta=5*np.log10(np.array([.5,1,2,1]))
    mu=np.repeat((ref+delta)[:,None],5,axis=1)
    mu[:,0]-=2;mu[:,1]-=1;mu[:,3]+=1;mu[:,4]+=2
    result=percent_metrics(mu,ref)
    assert result['n']==3
    np.testing.assert_allclose(result['median_pct'],0)
    np.testing.assert_allclose(result['nmad_pct'],1.4826*50)
    np.testing.assert_allclose(result['tail1mag'],2/3)
    assert ranges([1,np.nan,3])['missing']==1
    assert ranges([1,10],[99,1])['p50']<2
