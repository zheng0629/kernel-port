"""Unit tests for the harness value model (no JVM needed)."""

import random

import pytest

from kernel_port import harness as H
from kernel_port import java_ast as J


def _method(repo, qualname):
    m = repo.find_method(qualname)
    assert m is not None
    return m


def test_gen_case_deterministic(repo):
    m = _method(repo, "com.example.Kernels.dot")
    a = H.gen_case(m, random.Random(7))
    b = H.gen_case(m, random.Random(7))
    assert a == b
    c = H.gen_case(m, random.Random(8))
    assert a != c


def test_gen_case_dot_arrays_match(repo):
    m = _method(repo, "com.example.Kernels.dot")
    for seed in range(20):
        args = H.gen_case(m, random.Random(seed))
        assert len(args[0]) == len(args[1])


def test_gen_case_matmul_n_matches(repo):
    m = _method(repo, "com.example.Kernels.matMul")
    for seed in range(50):
        a, b, c, n = H.gen_case(m, random.Random(seed))
        assert len(a) == len(b) == len(c)
        assert n * n == len(a) or n == len(a)


def test_encode_decode_roundtrip():
    assert H.encode_arg("int", 42) == "i:42"
    assert H.encode_arg("double", 1.5) == "d:1.5"
    assert H.encode_arg("boolean", True) == "z:1"
    assert H.encode_arg("int[]", [1, 2]) == "ai:1,2"
    assert H.encode_arg("double[]", []) == "ad:"
    assert H._parse_result_token("i:42") == ("int", 42)
    assert H._parse_result_token("ad:") == ("ad", [])


def test_results_equal_scalars(repo):
    m = _method(repo, "com.example.Kernels.fib")
    ok, _ = H.results_equal(m, "l:55", "l:55")
    assert ok
    ok, detail = H.results_equal(m, "l:55", "l:56")
    assert not ok and detail


def test_results_equal_float_tolerance(repo):
    m = _method(repo, "com.example.Kernels.dot")
    ok, _ = H.results_equal(m, "d:0.30000000000000004", "d:0.3")
    assert ok
    ok, _ = H.results_equal(m, "d:1.0", "d:2.0")
    assert not ok


def test_results_equal_nan(repo):
    m = _method(repo, "com.example.Kernels.dot")
    ok, _ = H.results_equal(m, "d:nan", "d:nan")
    assert ok  # NaN == NaN for differential purposes


def test_results_equal_arrays(repo):
    m = _method(repo, "com.example.Kernels.sieve")
    ok, _ = H.results_equal(m, "ai:2,3,5", "ai:2,3,5")
    assert ok
    ok, _ = H.results_equal(m, "ai:2,3,5", "ai:2,3")
    assert not ok


def test_results_equal_void(repo):
    m = _method(repo, "com.example.Kernels.matMul")
    ok, _ = H.results_equal(m, "v", "v")
    assert ok


def test_harness_render_compiles_standalone(tmp_path, repo):
    """The generated harness must be syntactically valid Java."""
    from kernel_port.harness import compile_java, render_harness

    from .conftest import FIXTURE_REPO

    m = _method(repo, "com.example.Kernels.dot")
    hj = tmp_path / "HarnessJava.java"
    hj.write_text(render_harness(m, "java", "HarnessJava"))
    out = tmp_path / "classes"
    compile_java(sorted(FIXTURE_REPO.rglob("*.java")) + [hj], out)
    assert (out / "HarnessJava.class").exists()
