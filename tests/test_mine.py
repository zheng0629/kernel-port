"""Tests for `mine`: ranking, call counts, impurity reporting."""

from kernel_port import miner


def test_mine_ranks_pure_first(repo):
    cands = miner.mine(repo)
    pure = [c for c in cands if c.pure]
    impure = [c for c in cands if not c.pure]
    assert pure, "expected pure candidates"
    assert impure, "expected impure candidates"
    # all pure candidates sort before any impure one
    first_impure = min(cands.index(c) for c in impure)
    assert all(cands.index(c) < first_impure for c in pure)


def test_mine_call_counts(repo):
    by_name = {c.qualname: c for c in miner.mine(repo)}
    assert by_name["com.example.Kernels.dot"].calls == 3
    assert by_name["com.example.Kernels.fib"].calls == 2
    assert by_name["com.example.Kernels.square"].calls == 1  # called by norm()
    assert by_name["com.example.Kernels.sieve"].calls == 0


def test_mine_scores_positive_for_pure(repo):
    by_name = {c.qualname: c for c in miner.mine(repo)}
    for qn in (
        "com.example.Kernels.dot",
        "com.example.Kernels.matMul",
        "com.example.Kernels.sieve",
        "com.example.Kernels.fib",
    ):
        assert by_name[qn].score > 0, qn
    # impure methods get score 0 (never suggested for porting)
    assert by_name["com.example.Kernels.noisy"].score == 0.0


def test_mine_top_candidate_is_dot(repo):
    cands = miner.mine(repo)
    assert cands[0].qualname == "com.example.Kernels.dot"


def test_mine_impure_has_reasons(repo):
    by_name = {c.qualname: c for c in miner.mine(repo)}
    noisy = by_name["com.example.Kernels.noisy"]
    assert not noisy.pure
    assert noisy.reasons, "impure candidates must carry reasons"
