import numpy as np
from photdis.giant_wide import training_weights, field_roles, preprocessing, fit_controlled
from photdis.giant_information import band_realizations


def test_weights_preserve_regularization_and_balance():
    for labels in ([0,0,1], [0]*100+[1]*3):
        w=training_weights(labels,37,16)
        assert np.isclose(w.sum()*16,37)
        for k in (0,1): assert np.isclose(w[np.array(labels)==k].sum()*16,18.5)


def test_roles_partition_and_test_once():
    fields=list(range(13)); seen=[]
    for j in fields:
        train,cal,test=field_roles(fields,j)
        assert not set(train)&set(cal) and test not in train+cal
        assert sorted(train+cal+[test])==fields
        seen.append(test)
    assert seen==fields


def test_paired_fit_shared_scale():
    rng=np.random.default_rng(3); mag=rng.normal(size=(40,5)); err=np.full_like(mag,.02)
    labels=np.arange(40)%2; prep=preprocessing(mag[:20],err[:20],3)
    realized=band_realizations(mag,err,16,3)
    small=fit_controlled(prep,realized[:20],labels[:20],20)
    wide=fit_controlled(prep,realized,labels,20)
    np.testing.assert_array_equal(small['scale'].mean_,wide['scale'].mean_)
    assert small['total_weight']==wide['total_weight']==20
