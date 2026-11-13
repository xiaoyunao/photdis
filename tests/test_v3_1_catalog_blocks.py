import numpy as np
import pytest
from photdis.v3_1.catalog_blocks import overlap_blocks,block_mean


def test_transitive_source_reuse_and_duplicates():
    np.testing.assert_array_equal(overlap_blocks([[9,9],[9,10],[10],[22]]),[0,0,0,3])


def test_fields_bridge_disjoint_sources():
    sources=[[1],[2],[2,3],[4]]
    np.testing.assert_array_equal(overlap_blocks(sources,[7,7,8,9]),[0,0,0,3])
    np.testing.assert_array_equal(overlap_blocks(sources),[0,1,1,3])


def test_single_source_block_cannot_create_replication_interval():
    r=block_mean([0.,1.,2.],[0,0,0])
    assert r['mean']==1. and r['blocks']==1 and r['n']==3
    assert r['lower95'] is None and r['upper95'] is None


def test_cluster_bootstrap_preserves_whole_block_and_catalog_weights():
    r=block_mean([0.,2.,10.],[0,0,1])
    assert r['mean']==4. and r['lower95']==1. and r['upper95']==10.
    assert r['blocks']==2 and r['n']==3


def test_empty_and_missing_source_rejected():
    assert block_mean([],[])['mean'] is None
    np.testing.assert_array_equal(overlap_blocks([]),[])
    with pytest.raises(ValueError):overlap_blocks([[1],[]])
    with pytest.raises(ValueError):overlap_blocks([[1]],[1,2])
    with pytest.raises(ValueError):block_mean([np.nan],[0])
