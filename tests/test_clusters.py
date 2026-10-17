"""Tests for function-cluster mining (`miner.find_clusters`)."""

from pathlib import Path

import pytest

from kernel_port import java_ast as J
from kernel_port import miner

CLUSTER_REPO = Path(__file__).parent / "fixtures" / "cluster_kernels"


@pytest.fixture()
def cluster_repo():
    return J.scan_repo(CLUSTER_REPO)


def by_entry(clusters):
    return {c.entry: c for c in clusters}


def test_pure_helper_chain_forms_one_cluster(cluster_repo):
    clusters = miner.find_clusters(cluster_repo)
    norm = [c for c in clusters if c.entry == "com.example.Cluster.normalize"]
    assert len(norm) == 1, "expected exactly one cluster rooted at normalize"
    c = norm[0]
    assert c.members == [
        "com.example.Cluster.normalize",
        "com.example.Cluster.dot",
        "com.example.Cluster.scale",
    ]
    assert c.pure
    assert not c.reasons


def test_cluster_calls_count_external_only(cluster_repo):
    # normalize is called 2x by ClusterApp.main, which is outside its cluster
    c = by_entry(miner.find_clusters(cluster_repo))["com.example.Cluster.normalize"]
    assert c.calls == 2


def test_impure_helper_cluster_still_emitted(cluster_repo):
    c = by_entry(miner.find_clusters(cluster_repo))["com.example.Cluster.loudSum"]
    assert c.members == [
        "com.example.Cluster.loudSum",
        "com.example.Cluster.trace",
    ]
    assert not c.pure
    assert any("com.example.Cluster.trace" in r for r in c.reasons), c.reasons


def test_lone_pure_methods_produce_no_cluster(cluster_repo):
    clusters = miner.find_clusters(cluster_repo)
    entries = {c.entry for c in clusters}
    assert "com.example.Cluster.loneFib" not in entries
    for c in clusters:
        assert "com.example.Cluster.loneFib" not in c.members
    # helpers that are never entries of their own cluster either
    assert "com.example.Cluster.dot" not in entries
    assert "com.example.Cluster.scale" not in entries
    assert "com.example.Cluster.ident" not in entries


def test_ambiguous_overload_noted_not_guessed(cluster_repo):
    c = by_entry(miner.find_clusters(cluster_repo))["com.example.Cluster.blend"]
    # only the uniquely resolvable helper is a member; mix(...) is excluded
    assert c.members == [
        "com.example.Cluster.blend",
        "com.example.Cluster.ident",
    ]
    assert any(
        "mix" in r and "overload" in r for r in c.reasons
    ), c.reasons


def test_clusters_sort_pure_first(cluster_repo):
    clusters = miner.find_clusters(cluster_repo)
    assert clusters, "expected clusters"
    assert clusters[0].entry == "com.example.Cluster.normalize"
    pure_flags = [c.pure for c in clusters]
    assert pure_flags == sorted(pure_flags, reverse=True)


def test_cluster_score_uses_member_intensities(cluster_repo):
    clusters = by_entry(miner.find_clusters(cluster_repo))
    norm = clusters["com.example.Cluster.normalize"]
    assert norm.intensity > 0
    assert norm.score == round((1 + norm.calls) * norm.intensity, 1)
