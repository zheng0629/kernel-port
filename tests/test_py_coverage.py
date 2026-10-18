"""Tests for coverage-guided fuzzing of the Python->C verify path.

``test_static_manifest_*`` / ``test_merge_*`` / ``test_guided_*`` /
``test_minimize_*`` are pure unit tests (no subprocess). The
``test_run_verify_coverage_*`` tests are end-to-end: they build a real
shared object with gcc and run the driver subprocess, proving the
guided round catches a translation bug that the plain round cannot
see. Builds go to tests/cffi_work/ -- never /tmp (small tmpfs).
"""

import importlib.util
import itertools
import random
import shutil
import sys
from pathlib import Path

import pytest

from kernel_port import cffi_glue, py_ast, py_coverage, py_harness
from kernel_port.cffi_glue import KernelSig, ParamMap, RetMap

FIX = Path(__file__).parent / "fixtures" / "py_kernels"
WORK = Path(__file__).parent / "cffi_work"

RARE_SRC = '''
def rare_branch(x: int) -> int:
    if x == 1234567:
        return x * 2
    return x + 1


def loopy(items: list[int]) -> int:
    s = 0
    for v in items:
        s += v
    return s if s > 0 else -s
'''

_counter = itertools.count()


@pytest.fixture()
def wdir():
    d = WORK / f"pycov{next(_counter)}"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def pyrepo():
    return py_ast.scan_repo(FIX)


@pytest.fixture()
def rare_sig():
    return KernelSig(
        func_name="rare_branch",
        qualname="m.rare_branch",
        cname="kp_rare_branch",
        params=[ParamMap(name="x", pytype="int", kind="int",
                         ctype="int64_t", cname="p_x")],
        ret=RetMap(pytype="int", kind="int", ctype="int64_t"),
        file="m.py",
    )


@pytest.fixture()
def rare_manifest(tmp_path):
    src = tmp_path / "m.py"
    src.write_text(RARE_SRC, encoding="utf-8")
    return py_coverage.static_manifest(src)


def _build(pyrepo, wdir, qualname, body):
    """Glue + fill + compile; returns (sig, env)."""
    fn = pyrepo.functions[qualname]
    sig = cffi_glue.map_signature(fn)
    gdir = wdir / "glue"
    files = cffi_glue.generate([fn], gdir)
    cffi_glue.fill_translation(files["c"], sig.cname, body)
    env = py_harness.build_env(gdir, wdir / "work", sig)
    return sig, env


# Correct / buggy C bodies for cov_kernels.rare_branch (agent stand-ins).
# The else-branch is the identity: the boundary sweep feeds I64_MAX /
# I64_MIN, where Python bignum arithmetic would not be int64-closed.
RARE_OK = (
    "    if (p_x == 1234567) {\n"
    "        return p_x * 2;\n"
    "    }\n"
    "    return p_x;"
)
RARE_WRONG = (
    "    if (p_x == 1234567) {\n"
    "        return p_x * 3;  // BUG: mistranslated constant\n"
    "    }\n"
    "    return p_x;"
)


# ---------------------------------------------------------------------------
# static manifest
# ---------------------------------------------------------------------------


def test_static_manifest_enumerates_edges(rare_manifest):
    m = rare_manifest
    by_func = {}
    for b in m:
        by_func.setdefault(b.func, []).append(b)
    assert set(by_func) == {"rare_branch", "loopy"}
    assert len(by_func["rare_branch"]) == 2  # if/else
    assert len(by_func["loopy"]) == 4  # for body/exit + ternary arms


def test_static_manifest_condition_harvest(rare_manifest):
    edges = [b for b in rare_manifest if b.func == "rare_branch"]
    taken = next(b for b in edges if b.taken)
    fall = next(b for b in edges if not b.taken)
    assert (fall.kind, taken.kind) == ("cond-true", "cond-false")
    assert fall.condition == "x == 1234567"
    assert fall.names == ("x",)
    assert fall.literals == (1234567,)
    assert fall.line == 3  # the `if` line in RARE_SRC


def test_static_manifest_loop_kinds(rare_manifest):
    edges = [b for b in rare_manifest if b.func == "loopy" and b.line == 10]
    kinds = {(b.kind, b.taken) for b in edges}
    assert kinds == {("loop-body", False), ("loop-exit", True)}
    assert edges[0].condition == "items"
    assert edges[0].literals == ()


def test_static_manifest_ids_unique_ordered(rare_manifest):
    ids = [b.id for b in rare_manifest]
    assert ids == list(range(len(rare_manifest)))


def test_static_manifest_missing_file_raises(tmp_path):
    with pytest.raises(OSError):
        py_coverage.static_manifest(tmp_path / "nope.py")


def test_static_manifest_branchless_is_empty(tmp_path):
    src = tmp_path / "plain.py"
    src.write_text("def f(x: int) -> int:\n    return x + 1\n", encoding="utf-8")
    assert py_coverage.static_manifest(src) == []


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def test_merge_coverage_counts(rare_manifest):
    m = rare_manifest
    # hit every edge except the rare true-edge (id 1)
    hits = [(b.func, 1, b.instr, b.fall if not b.taken else b.fall + 999)
            for b in m if b.id != 1]
    cov = py_coverage.merge_coverage(m, hits)
    assert cov.branches_total == len(m)
    assert cov.branches_hit == len(m) - 1
    assert cov.coverage_pct == pytest.approx(100.0 * (len(m) - 1) / len(m))
    assert [b.id for b in cov.uncovered()] == [1]
    assert cov.hits_for(1) == 0
    assert cov.hits_for(0) == 1


def test_merge_coverage_ignores_stale_hits(rare_manifest):
    cov = py_coverage.merge_coverage(
        rare_manifest, [("rare_branch", 1, 99999, 0), ("nope", 1, 10, 12)]
    )
    assert cov.branches_hit == 0
    assert cov.coverage_pct == 0.0


def test_coverage_pct_empty_is_100():
    cov = py_coverage.CoverageData(0, 0, {}, [])
    assert cov.coverage_pct == 100.0
    assert cov.uncovered() == []


# ---------------------------------------------------------------------------
# branch recorder (in-process; exercises the real sys.monitoring path)
# ---------------------------------------------------------------------------


def _load_tmp(tmp_path, name="m.py"):
    src = tmp_path / name
    src.write_text(RARE_SRC, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("cov_tmp_mod", str(src))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return src, mod


def test_branch_recorder_full_coverage(tmp_path):
    src, mod = _load_tmp(tmp_path)
    manifest = py_coverage.static_manifest(src)
    with py_coverage.branch_recorder(src) as rec:
        mod.rare_branch(0)
        mod.rare_branch(1234567)
        mod.loopy([])
        mod.loopy([5])  # s = 5 > 0 takes the ternary's true arm
    cov = py_coverage.merge_coverage(manifest, rec.hits)
    assert cov.coverage_pct == 100.0, [
        (b.func, b.line, b.kind) for b in cov.uncovered()
    ]


def test_branch_recorder_partial_coverage(tmp_path):
    src, mod = _load_tmp(tmp_path)
    manifest = py_coverage.static_manifest(src)
    with py_coverage.branch_recorder(src) as rec:
        mod.rare_branch(0)  # never takes the x == 1234567 arm
    cov = py_coverage.merge_coverage(manifest, rec.hits)
    assert cov.coverage_pct < 100.0
    assert any(
        b.func == "rare_branch" and b.kind == "cond-true"
        for b in cov.uncovered()
    )


def test_branch_recorder_ignores_other_files(tmp_path):
    src, mod = _load_tmp(tmp_path)
    other = tmp_path / "other.py"
    other.write_text(
        "def g(y: int) -> int:\n    return y * 2 if y else 0\n", encoding="utf-8"
    )
    spec = importlib.util.spec_from_file_location("cov_other", str(other))
    omod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(omod)
    with py_coverage.branch_recorder(src) as rec:
        omod.g(5)
        omod.g(0)
        mod.rare_branch(1)
    assert rec.hits, "kernel's own hits must be recorded"
    assert all(h[0] != "g" for h in rec.hits)


def test_branch_recorder_releases_tool_id(tmp_path):
    src, _mod = _load_tmp(tmp_path)
    mon = sys.monitoring
    before = {i for i in range(6) if _tool_in_use(mon, i)}
    with py_coverage.branch_recorder(src):
        pass
    after = {i for i in range(6) if _tool_in_use(mon, i)}
    assert after == before


def _tool_in_use(mon, tid):
    try:
        mon.use_tool_id(tid, "kernel-port-probe")
    except ValueError:
        return True
    mon.free_tool_id(tid)
    return False


def test_branch_recorder_no_monitoring_graceful(tmp_path, monkeypatch):
    monkeypatch.delattr(sys, "monitoring", raising=False)
    with pytest.raises(RuntimeError, match="3.12"):
        with py_coverage.branch_recorder(tmp_path / "m.py"):
            pass  # pragma: no cover


# ---------------------------------------------------------------------------
# guided cases
# ---------------------------------------------------------------------------


def _all_uncovered(sig_func, manifest):
    return py_coverage.CoverageData(
        branches_total=len(manifest),
        branches_hit=0,
        per_branch_hits={},
        branches=[b for b in manifest if b.func == sig_func],
    )


def test_guided_cases_targets_magic_literal(rare_sig, rare_manifest):
    cov = _all_uncovered("rare_branch", rare_manifest)
    cases = py_coverage.guided_cases(rare_sig, cov, random.Random(7), 12)
    assert len(cases) == 12
    xs = {a[0] for a in cases}
    assert 1234567 in xs  # the harvested literal is injected
    assert 1234566 in xs and 1234568 in xs  # +/-1 neighborhood


def test_guided_cases_kernel_edges_first(rare_sig, rare_manifest):
    """loopy's uncovered edges must not starve rare_branch's own edge."""
    cov = py_coverage.CoverageData(
        branches_total=len(rare_manifest),
        branches_hit=0,
        per_branch_hits={},
        branches=list(rare_manifest),
    )
    cases = py_coverage.guided_cases(rare_sig, cov, random.Random(7), 8)
    assert cases[0][0] in (1234566, 1234567, 1234568)


def test_guided_cases_loop_body_nondegenerate(rare_manifest):
    sig = KernelSig(
        func_name="loopy", qualname="m.loopy", cname="kp_loopy",
        params=[ParamMap(name="items", pytype="list[int]", kind="list_int",
                         ctype="int64_t", cname="p_items", is_list=True)],
        ret=RetMap(pytype="int", kind="int", ctype="int64_t"),
        file="m.py",
    )
    cov = _all_uncovered("loopy", rare_manifest)
    cases = py_coverage.guided_cases(sig, cov, random.Random(3), 1)
    assert len(cases) == 1
    assert len(cases[0][0]) > 0, "loop-body steering needs a non-empty list"


def test_guided_cases_no_literals_no_crash(rare_sig):
    branch = py_coverage.BranchInfo(
        id=0, func="rare_branch", line=1, kind="loop-body",
        condition="items", names=("items",), literals=(),
        instr=10, fall=14, taken=False,
    )
    cov = py_coverage.CoverageData(1, 0, {}, [branch])
    cases = py_coverage.guided_cases(rare_sig, cov, random.Random(1), 5)
    assert len(cases) == 5  # falls back to random fill


def test_guided_cases_terminates_for_paramless_sig():
    sig = KernelSig(
        func_name="f", qualname="m.f", cname="kp_f", params=[],
        ret=RetMap(pytype="int", kind="int", ctype="int64_t"), file="m.py",
    )
    cov = py_coverage.CoverageData(0, 0, {}, [])
    cases = py_coverage.guided_cases(sig, cov, random.Random(1), 10)
    assert cases == [[]]  # stall guard: exactly one deduped case


# ---------------------------------------------------------------------------
# minimize
# ---------------------------------------------------------------------------


def test_minimize_shrinks_int_and_list():
    out = py_coverage.minimize([100, [5, 4, 3, 2, 1]], lambda c: c[0] > 10)
    assert out[0] == 12  # halving steps: 100->50->25->12 (6 no longer fails)
    assert out[1] == []  # ddmin: the list is irrelevant to the predicate


def test_minimize_shrinks_string():
    out = py_coverage.minimize(["hello world"], lambda c: len(c[0]) > 3)
    assert out == ["hell"]  # prefix binary search keeps the shortest failure


def test_minimize_requires_failing_case():
    with pytest.raises(ValueError, match="currently fails"):
        py_coverage.minimize([1], lambda c: False)


def test_minimize_rejects_nonpositive_budget():
    with pytest.raises(ValueError, match="max_seconds"):
        py_coverage.minimize([5], lambda c: True, max_seconds=0)


# ---------------------------------------------------------------------------
# run_verify wiring (end-to-end with real gcc + driver subprocess)
# ---------------------------------------------------------------------------


def test_run_verify_coverage_round_passes(pyrepo, wdir):
    sig, env = _build(pyrepo, wdir, "cov_kernels.rare_branch", RARE_OK)
    rep = py_harness.run_verify(
        env, sig, FIX, cases=10, seed=11, timeout=3.0,
        coverage=True, guided_n=20,
    )
    assert rep.all_passed
    assert rep.failed == 0 and rep.boundary_failed == 0
    assert rep.guided_failed == 0
    assert rep.guided_passed > 0
    summary = rep.coverage_summary()
    assert summary is not None
    assert summary["branches_total"] > 0
    assert summary["coverage_pct"] < 100.0  # the rare arm is uncovered in round 1
    guided = [c for c in rep.cases if c.guided]
    assert 0 < len(guided) <= 20
    assert not any(c.boundary for c in guided)


def test_run_verify_coverage_finds_rare_branch_bug(pyrepo, wdir):
    """The money test: round 1 cannot see x == 1234567; the guided round can."""
    sig, env = _build(pyrepo, wdir, "cov_kernels.rare_branch", RARE_WRONG)
    plain = py_harness.run_verify(env, sig, FIX, cases=10, seed=11, timeout=3.0)
    assert plain.all_passed, "plain round-1 must miss the rare-branch bug"
    rep = py_harness.run_verify(
        env, sig, FIX, cases=10, seed=11, timeout=3.0,
        coverage=True, guided_n=20,
    )
    assert not rep.all_passed
    r1_failed = sum(
        1 for c in rep.cases + rep.boundary_cases
        if not c.guided and not c.ok and not c.skipped
    )
    assert r1_failed == 0  # round 1 still blind
    assert rep.guided_failed > 0
    failing = [c for c in rep.cases if c.guided and not c.ok and not c.skipped]
    assert failing
    assert "return mismatch" in failing[0].detail
    assert failing[0].args[0] in (1234566, 1234567, 1234568)


def test_run_verify_coverage_defaults_off(pyrepo, wdir):
    sig, env = _build(pyrepo, wdir, "cov_kernels.rare_branch", RARE_OK)
    rep = py_harness.run_verify(env, sig, FIX, cases=5, seed=11, timeout=3.0)
    assert rep.all_passed
    assert rep.coverage is None
    assert rep.coverage_summary() is None
    assert rep.guided_passed == 0 and rep.guided_failed == 0


def test_coverage_round_degrades_gracefully(pyrepo, wdir, monkeypatch):
    sig, env = _build(pyrepo, wdir, "cffi_kernels.dot",
                      "    double s = 0.0;\n"
                      "    for (int64_t i = 0; i < p_a_n; i++) s += p_a[i] * p_b[i];\n"
                      "    return s;")

    def _boom(_path):
        raise RuntimeError("synthetic manifest failure")

    monkeypatch.setattr(py_coverage, "static_manifest", _boom)
    with pytest.warns(UserWarning, match="coverage-guided round skipped"):
        rep = py_harness.run_verify(
            env, sig, FIX, cases=5, seed=11, timeout=3.0, coverage=True
        )
    assert rep.all_passed  # round-1 results stand
    assert rep.coverage is None
