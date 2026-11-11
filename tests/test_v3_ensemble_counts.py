import numpy as np
from photdis.v3.population import SharedCounts
from photdis.v3.ensemble_counts import combine_counts, combine_star_means


def test_response_marginal_covariance_matches_one_global_joint_draw_catalog():
    rng = np.random.default_rng(78)
    p = rng.dirichlet([1, 2, 3], size=(2, 16, 30))
    records = []
    for j in range(2):
        c = SharedCounts(np.arange(16), 3, seed=j); c.add(p[j], np.arange(16)); records.append(c.result())
    logz = np.log([.2, .8]); got = combine_counts(records, logz)
    allcounts = SharedCounts(np.arange(32), 3, draw_weights=np.repeat([.2/16, .8/16], 16), seed=9)
    allcounts.add(p.reshape(32, 30, 3), np.arange(32)); expected = allcounts.result()
    np.testing.assert_allclose(got['mean'], expected['mean'], atol=1e-14)
    np.testing.assert_allclose(got['covariance'], expected['covariance'], atol=2e-13)
    assert got['count_draws'].dtype == np.int64 and (got['count_draws'].sum(axis=1) == 30).all()
    # Tiny-weight conditional errors cannot masquerade as final marginal errors.
    means=p.mean(axis=1); a=means.copy(); b=means.copy(); a[0, 0]=[1, 0, 0];b[0, 0]=[0, 0, 1]
    _, delta=combine_star_means(means,a,b,np.log([1e-8,1-1e-8]));assert delta < 2e-8


def test_chunks_merge_shared_means_before_covariance():
    from photdis.v3.ensemble_counts import merge_chunk_counts
    rng=np.random.default_rng(3); p=rng.dirichlet([1,2,3],size=(20,40))
    records=[]
    for rows in (slice(0,13),slice(13,40)):
        c=SharedCounts(np.arange(20),3,seed=2); c.add(p[:,rows],np.arange(20)); records.append(c.result())
    merged=merge_chunk_counts(records)
    c=SharedCounts(np.arange(20),3,seed=4);c.add(p,np.arange(20));expected=c.result()
    np.testing.assert_allclose(merged['mean'],expected['mean'],atol=1e-14)
    np.testing.assert_allclose(merged['covariance'],expected['covariance'],atol=1e-13)
    assert not np.allclose(merged['shared_covariance'],sum(r['shared_covariance'] for r in records))
