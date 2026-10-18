"""End-to-end tests for the Python->C cffi verify/bench harness.

Builds real shared objects with gcc and runs the driver subprocess, so
these tests prove the harness catches translation bugs (a planted wrong
translation must FAIL). Builds go to tests/cffi_work/ -- never /tmp
(which is a small tmpfs on this machine).
"""

import itertools
import math
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kernel_port import cffi_glue, py_ast, py_harness
from kernel_port.cli import app

FIX = Path(__file__).parent / "fixtures" / "py_kernels"
WORK = Path(__file__).parent / "cffi_work"
runner = CliRunner()

# Correct C translations, standing in for the calling agent's LLM step.
BODIES = {
    "cffi_kernels.dot": (
        "    double s = 0.0;\n"
        "    for (int64_t i = 0; i < p_a_n; i++) s += p_a[i] * p_b[i];\n"
        "    return s;"
    ),
    "cffi_kernels.norm": (
        "    double s = 0.0;\n"
        "    for (int64_t i = 0; i < p_v_n; i++) s += p_v[i] * p_v[i];\n"
        "    return sqrt(s);"
    ),
    "cffi_kernels.matvec": (
        "    for (int64_t i = 0; i < p_n; i++) {\n"
        "        double s = 0.0;\n"
        "        for (int64_t j = 0; j < p_n; j++) s += p_a[i * p_n + j] * p_x[j];\n"
        "        p_y[i] = s;\n"
        "    }"
    ),
    "cffi_kernels.count_upper": (
        "    int64_t n = 0;\n"
        "    for (const char* p = p_s; *p; p++)\n"
        "        if (*p >= 'A' && *p <= 'Z') n++;\n"
        "    return n;"
    ),
    "cffi_kernels.toggle": "    return !p_b;",
}
# Deliberately wrong translation (off by a constant) for negative testing.
WRONG_DOT = (
    "    double s = 0.0;\n"
    "    for (int64_t i = 0; i < p_a_n; i++) s += p_a[i] * p_b[i];\n"
    "    return s + 1.0;"
)


@pytest.fixture(scope="session", autouse=True)
def _clean_work():
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    yield
    shutil.rmtree(WORK, ignore_errors=True)


_counter = itertools.count()


@pytest.fixture()
def wdir():
    d = WORK / f"case{next(_counter)}"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def pyrepo():
    return py_ast.scan_repo(FIX)


def _build(pyrepo, wdir, qualname, body):
    """Glue + fill + compile; returns (sig, env)."""
    fn = pyrepo.functions[qualname]
    sig = cffi_glue.map_signature(fn)
    gdir = wdir / "glue"
    files = cffi_glue.generate([fn], gdir)
    cffi_glue.fill_translation(files["c"], sig.cname, body)
    env = py_harness.build_env(gdir, wdir / "work", sig)
    return sig, env


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def test_build_compiles_shared_object(pyrepo, wdir):
    sig, env = _build(pyrepo, wdir, "cffi_kernels.dot", BODIES["cffi_kernels.dot"])
    assert env.so_path.exists()
    assert env.so_path.stat().st_size > 0


def test_build_refuses_placeholder_body(pyrepo, wdir):
    fn = pyrepo.functions["cffi_kernels.dot"]
    sig = cffi_glue.map_signature(fn)
    gdir = wdir / "glue"
    cffi_glue.generate([fn], gdir)  # placeholder NOT filled
    with pytest.raises(RuntimeError, match="placeholder"):
        py_harness.build_env(gdir, wdir / "work", sig)


def test_build_missing_c_file_errors(pyrepo, wdir):
    fn = pyrepo.functions["cffi_kernels.dot"]
    sig = cffi_glue.map_signature(fn)
    with pytest.raises(RuntimeError, match="kernels_cffi.c.*not found"):
        py_harness.build_env(wdir / "empty", wdir / "work", sig)


# ---------------------------------------------------------------------------
# verify: correct translations pass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("qualname", sorted(BODIES))
def test_verify_correct_translation_passes(pyrepo, wdir, qualname):
    sig, env = _build(pyrepo, wdir, qualname, BODIES[qualname])
    rep = py_harness.run_verify(env, sig, FIX, cases=25, seed=11, timeout=3.0)
    assert rep.passed == 25, [c.detail for c in rep.cases if not c.ok and not c.skipped]
    assert rep.failed == 0
    assert rep.all_passed
    # boundary sweep ran and (mostly) passed
    assert len(rep.boundary_cases) > 0
    assert rep.boundary_failed == 0


def test_verify_boundary_includes_empty_list(pyrepo, wdir):
    sig, env = _build(pyrepo, wdir, "cffi_kernels.dot", BODIES["cffi_kernels.dot"])
    rep = py_harness.run_verify(env, sig, FIX, cases=5, seed=11, timeout=3.0)
    empties = [c for c in rep.boundary_cases if any(a == [] for a in c.args)]
    assert empties, "boundary sweep must include an empty-list case"
    assert all(c.ok for c in empties)


def test_verify_wrong_translation_fails(pyrepo, wdir):
    """The harness must catch bugs: dot translated with `+ 1.0` fails."""
    sig, env = _build(pyrepo, wdir, "cffi_kernels.dot", WRONG_DOT)
    rep = py_harness.run_verify(env, sig, FIX, cases=10, seed=11, timeout=3.0)
    assert not rep.all_passed
    assert rep.failed > 0
    failing = [c for c in rep.cases if not c.ok and not c.skipped]
    assert failing
    assert "return mismatch" in failing[0].detail


def test_verify_out_param_echo_compared(pyrepo, wdir):
    """matvec mutates y: the echo comparison is what verify checks."""
    sig, env = _build(pyrepo, wdir, "cffi_kernels.matvec", BODIES["cffi_kernels.matvec"])
    rep = py_harness.run_verify(env, sig, FIX, cases=15, seed=11, timeout=3.0)
    assert rep.all_passed
    # now a translation that forgets to write y must fail on the echo
    sig2, env2 = _build(
        pyrepo, wdir, "cffi_kernels.matvec",
        "    for (int64_t i = 0; i < p_n; i++) { (void)i; }",  # writes nothing
    )
    rep2 = py_harness.run_verify(env2, sig2, FIX, cases=10, seed=11, timeout=3.0)
    assert not rep2.all_passed
    assert any("list 'y'" in c.detail for c in rep2.cases if not c.ok)


# ---------------------------------------------------------------------------
# input generation: length coherence (mirrors the Java harness)
# ---------------------------------------------------------------------------


def test_random_lists_share_length(pyrepo):
    import random

    sig = cffi_glue.map_signature(pyrepo.functions["cffi_kernels.dot"])
    rng = random.Random(0)
    for _ in range(20):
        args = py_harness.gen_random_args(sig, rng)
        assert len(args[0]) == len(args[1])


def test_boundary_lists_share_length(pyrepo):
    sig = cffi_glue.map_signature(pyrepo.functions["cffi_kernels.matvec"])
    for args in py_harness.gen_boundary_args(sig):
        lens = {len(a) for a in args if isinstance(a, list)}
        assert len(lens) <= 1, f"incoherent lengths in {args}"


def test_compare_case_nan_equal():
    sig = cffi_glue.KernelSig(
        func_name="f", qualname="m.f", cname="kp_f", ret=cffi_glue.RetMap("float", "float", "double")
    )
    ok, _ = py_harness.compare_case(
        sig,
        {"ret": float("nan"), "lists": {}},
        {"ret": float("nan"), "lists": {}},
        1e-6, 1e-12,
    )
    assert ok
    ok, detail = py_harness.compare_case(
        sig,
        {"ret": 1.0, "lists": {}},
        {"ret": 2.0, "lists": {}},
        1e-6, 1e-12,
    )
    assert not ok and "return mismatch" in detail


def test_compare_case_list_length_change_fails(pyrepo):
    sig = cffi_glue.map_signature(pyrepo.functions["cffi_kernels.matvec"])
    ok, detail = py_harness.compare_case(
        sig,
        {"ret": None, "lists": {"a": [1.0], "x": [2.0], "y": [3.0]}},
        {"ret": None, "lists": {"a": [1.0], "x": [2.0], "y": []}},
        1e-6, 1e-12,
    )
    assert not ok and "length changed" in detail


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------


def test_bench_reports_medians(pyrepo, wdir):
    sig, env = _build(pyrepo, wdir, "cffi_kernels.dot", BODIES["cffi_kernels.dot"])
    rep = py_harness.run_bench(env, sig, FIX, warmup=3, iters=20, repeats=3)
    assert rep.python_per_call_ns > 0
    assert rep.c_per_call_ns > 0
    assert math.isfinite(rep.speedup) and rep.speedup > 0


# ---------------------------------------------------------------------------
# CLI wiring (--lang python)
# ---------------------------------------------------------------------------


def test_cli_glue_python(wdir):
    out = wdir / "glue"
    r = runner.invoke(app, ["glue", "--lang", "python", "--method", "cffi_kernels.dot",
                            "--out", str(out), str(FIX)])
    assert r.exit_code == 0, r.output
    assert (out / "kernels_cffi.c").exists()
    assert (out / "kernels_cffi.py").exists()


def test_cli_glue_python_refuses_impure(wdir):
    r = runner.invoke(app, ["glue", "--lang", "python", "--method", "cffi_kernels.log_sample",
                            "--out", str(wdir / "g"), str(FIX)])
    assert r.exit_code == 2
    assert "not pure" in r.output


def test_cli_glue_python_unannotated_errors(wdir):
    r = runner.invoke(app, ["glue", "--lang", "python", "--method", "cffi_kernels.poly_plain",
                            "--out", str(wdir / "g"), str(FIX)])
    assert r.exit_code == 1
    assert "x" in r.output and "coeffs" in r.output


def test_cli_glue_python_rejects_backend(wdir):
    r = runner.invoke(app, ["glue", "--lang", "python", "--backend", "ffm",
                            "--method", "cffi_kernels.dot",
                            "--out", str(wdir / "g"), str(FIX)])
    assert r.exit_code == 1
    assert "cffi" in r.output


def _cli_verify(wdir, qualname, body, cli_cases="12"):
    out = wdir / "glue"
    r = runner.invoke(app, ["glue", "--lang", "python", "--method", qualname,
                            "--out", str(out), str(FIX)])
    assert r.exit_code == 0, r.output
    fn_name = qualname.split(".")[-1]
    sig = cffi_glue.map_signature(py_ast.scan_repo(FIX).functions[qualname])
    cffi_glue.fill_translation(out / "kernels_cffi.c", sig.cname, body)
    r = runner.invoke(app, ["verify", "--lang", "python", "--method", qualname,
                            "--glue-dir", str(out), "--cases", cli_cases,
                            "--seed", "11", str(FIX)])
    return r


def test_cli_verify_python_pass(wdir):
    r = _cli_verify(wdir, "cffi_kernels.dot", BODIES["cffi_kernels.dot"])
    assert r.exit_code == 0, r.output
    assert "PASS" in r.output


def test_cli_verify_python_fail(wdir):
    r = _cli_verify(wdir, "cffi_kernels.dot", WRONG_DOT, cli_cases="8")
    assert r.exit_code == 1, r.output
    assert "FAIL" in r.output


def test_cli_bench_python(wdir):
    out = wdir / "glue"
    r = runner.invoke(app, ["glue", "--lang", "python", "--method", "cffi_kernels.count_upper",
                            "--out", str(out), str(FIX)])
    assert r.exit_code == 0, r.output
    sig = cffi_glue.map_signature(py_ast.scan_repo(FIX).functions["cffi_kernels.count_upper"])
    cffi_glue.fill_translation(out / "kernels_cffi.c", sig.cname, BODIES["cffi_kernels.count_upper"])
    r = runner.invoke(app, ["bench", "--lang", "python", "--method", "cffi_kernels.count_upper",
                            "--glue-dir", str(out), "--iters", "20", "--warmup", "3",
                            "--repeats", "2", str(FIX)])
    assert r.exit_code == 0, r.output
    assert "C (cffi)" in r.output
