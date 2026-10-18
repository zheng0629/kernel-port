"""Tests for v0.3 production hardening: fallback glue, fingerprint cache,
multi-candidate selection."""

import json

import pytest

from kernel_port import cache, jni, select
from kernel_port import java_ast as J


def _method(repo, qualname):
    m = repo.find_method(qualname)
    assert m is not None
    return m


def _bad_method():
    return J.MethodInfo(
        qualname="p.T.m",
        class_qualname="p.T",
        name="m",
        # v0.4 supports double[][]/int[][]/String; float[][] stays unsupported
        jtype="float[][]",
        params=[],
        modifiers={"static"},
        is_static=True,
        is_constructor=False,
        file="T.java",
        lineno=1,
        end_lineno=2,
        source="",
    )


# ---------------------------------------------------------------- fallback


def test_fallback_java_is_dispatcher(repo, tmp_path):
    m = _method(repo, "com.example.Kernels.dot")
    files = jni.generate(m, tmp_path / "glue", fallback=True)
    src = files["java"].read_text()
    # switch mechanism
    assert 'System.getProperty("kernelport.native"' in src
    assert "setUseNative" in src
    assert "-Dkernelport.native=false" in src
    # delegates to the original implementation
    assert "com.example.Kernels.dot" in src
    # native entry point renamed
    assert "dotNative" in src
    assert "private static native double dotNative(double[] a, double[] b);" in src
    # public dispatcher keeps the kernel signature (not native itself)
    assert "public static double dot(double[] a, double[] b) {" in src
    assert "public static native double dot(" not in src


def test_fallback_java_void_method(repo, tmp_path):
    m = _method(repo, "com.example.Kernels.matMul")
    files = jni.generate(m, tmp_path / "glue", fallback=True)
    src = files["java"].read_text()
    assert "public static void matMul(" in src
    assert "private static native void matMulNative(" in src
    assert "com.example.Kernels.matMul(" in src
    assert "return matMulNative" not in src


def test_fallback_cpp_renames_jni_symbol(repo, tmp_path):
    m = _method(repo, "com.example.Kernels.dot")
    files = jni.generate(m, tmp_path / "glue", fallback=True)
    cpp = files["cpp"].read_text()
    assert "Java_KernelsNative_dotNative" in cpp
    assert "Java_KernelsNative_dot(" not in cpp
    assert "kernel_dot" in cpp  # helper (translated body) unchanged
    assert "TODO(agent)" in cpp


def test_fallback_cpp_compiles(repo, tmp_path):
    from kernel_port.harness import compile_jni

    m = _method(repo, "com.example.Kernels.dot")
    files = jni.generate(m, tmp_path / "glue", fallback=True)
    compile_jni(files["cpp"], tmp_path / "glue" / "libkernels.so")


def test_fallback_false_matches_v01_byte_identical(repo, tmp_path):
    m = _method(repo, "com.example.Kernels.dot")
    gp = jni.plan(m)
    files = jni.generate(m, tmp_path / "glue", fallback=False)
    assert files["java"].read_text() == jni.render_java(gp)
    assert files["cpp"].read_text() == jni.render_cpp(gp)
    assert files["cmake"].read_text() == jni.render_cmake()


def test_two_arg_generate_matches_fallback_true_default(repo, tmp_path):
    # v0.3: the fallback dispatcher is the default; 2-arg call == fallback=True.
    m = _method(repo, "com.example.Kernels.dot")
    a = jni.generate(m, tmp_path / "a")
    b = jni.generate(m, tmp_path / "b", fallback=True)
    for kind in ("java", "cpp", "cmake"):
        assert a[kind].read_text() == b[kind].read_text()


def test_fallback_rejects_unsupported_type(tmp_path):
    with pytest.raises(ValueError, match=r"unsupported type 'float\[\]\[\]'"):
        jni.generate(_bad_method(), tmp_path, fallback=True)


def test_fallback_rejects_non_static(tmp_path):
    m = J.MethodInfo(
        qualname="p.T.m",
        class_qualname="p.T",
        name="m",
        jtype="int",
        params=[],
        modifiers=set(),
        is_static=False,
        is_constructor=False,
        file="T.java",
        lineno=1,
        end_lineno=2,
        source="",
    )
    with pytest.raises(ValueError, match="static"):
        jni.generate(m, tmp_path, fallback=True)


# ---------------------------------------------------------------- cache


def _extraction(source, closure=(), signature="double dot(double[], double[])"):
    return {
        "method": "com.example.Kernels.dot",
        "signature": signature,
        "source": source,
        "closure": list(closure),
    }


def test_cache_store_lookup_roundtrip(tmp_path):
    fp = "ab" * 32
    record = {
        "method": "com.example.Kernels.dot",
        "signature": "double dot(double[], double[])",
        "translated_body": "double s = 0.0; return s;",
        "verify": {"passed": 100, "failed": 0, "skipped": 0},
        "bench": {"java_per_call_ns": 120.5, "jni_per_call_ns": 80.2, "speedup": 1.5},
        "created": "2026-10-17T22:00:00+00:00",
    }
    path = cache.store(tmp_path, fp, record)
    assert path == tmp_path / f"{fp}.json"
    assert path.exists()
    got = cache.lookup(tmp_path, fp)
    assert got is not None
    assert got["fingerprint"] == fp
    assert got["cache_version"] == cache.CACHE_VERSION
    for k, v in record.items():
        assert got[k] == v
    # caller's record not mutated
    assert "fingerprint" not in record
    # no stray tmp file left behind (atomic write)
    assert list(tmp_path.glob("*.tmp")) == []


def test_cache_lookup_unknown_returns_none(tmp_path):
    assert cache.lookup(tmp_path, "0" * 64) is None


def test_cache_lookup_corrupt_json_returns_none(tmp_path):
    (tmp_path / ("ff" * 32 + ".json")).write_text("{not json", encoding="utf-8")
    assert cache.lookup(tmp_path, "ff" * 32) is None


def test_cache_lookup_version_mismatch_returns_none(tmp_path):
    fp = "ab" * 32
    (tmp_path / f"{fp}.json").write_text(
        json.dumps({"cache_version": 999, "fingerprint": fp}), encoding="utf-8"
    )
    assert cache.lookup(tmp_path, fp) is None


def test_cache_lookup_non_dict_returns_none(tmp_path):
    fp = "ab" * 32
    (tmp_path / f"{fp}.json").write_text("[1, 2]", encoding="utf-8")
    assert cache.lookup(tmp_path, fp) is None


def test_normalize_source_ignores_comments_and_whitespace():
    assert cache.normalize_source("int x; // c\n") == cache.normalize_source("int  x;")
    assert cache.normalize_source("int /* multi\nline */ x;") == "int x;"
    assert cache.normalize_source("  int\n\tx;\n") == "int x;"


def test_fingerprint_stable_and_sensitive_to_body():
    a = _extraction("int f() { return 1; } // one")
    b = _extraction("int   f() {  return  1; }\n// two")
    assert cache.fingerprint(a) == cache.fingerprint(b)
    c = _extraction("int f() { return 2; }")
    assert cache.fingerprint(a) != cache.fingerprint(c)
    d = _extraction("int f() { return 1; }", signature="int g()")
    assert cache.fingerprint(a) != cache.fingerprint(d)


def test_fingerprint_closure_sorted_by_qualname():
    c1 = {"qualname": "p.B", "source": "int b(){return 1;}"}
    c2 = {"qualname": "p.A", "source": "int a(){return 2;}"}
    x = _extraction("int f(){return 1;}", closure=[c1, c2])
    y = _extraction("int f(){return 1;}", closure=[c2, c1])
    assert cache.fingerprint(x) == cache.fingerprint(y)
    z = _extraction("int f(){return 1;}", closure=[c1])
    assert cache.fingerprint(x) != cache.fingerprint(z)


def test_fingerprint_defensive_on_sparse_dict():
    fp = cache.fingerprint({})
    assert isinstance(fp, str) and len(fp) == 64


# ---------------------------------------------------------------- select


def _results():
    return [
        {"name": "cand-broken", "verify_passed": 90, "verify_failed": 10,
         "bench_median_ns": 50.0},
        {"name": "cand-slow", "verify_passed": 100, "verify_failed": 0,
         "bench_median_ns": 200.0},
        {"name": "cand-fast", "verify_passed": 100, "verify_failed": 0,
         "bench_median_ns": 80.0},
    ]


def test_pick_winner_fastest_passing():
    winner = select.pick_winner(_results())
    assert winner["name"] == "cand-fast"


def test_pick_winner_ignores_zero_passed():
    results = [
        {"name": "skipped", "verify_passed": 0, "verify_failed": 0,
         "bench_median_ns": 1.0},
        {"name": "ok", "verify_passed": 5, "verify_failed": 0,
         "bench_median_ns": 9.0},
    ]
    assert select.pick_winner(results)["name"] == "ok"


def test_pick_winner_tie_keeps_list_order():
    results = [
        {"name": "first", "verify_passed": 10, "verify_failed": 0,
         "bench_median_ns": 42.0},
        {"name": "second", "verify_passed": 10, "verify_failed": 0,
         "bench_median_ns": 42.0},
    ]
    assert select.pick_winner(results)["name"] == "first"


def test_pick_winner_all_failing_raises():
    results = [
        {"name": "a", "verify_passed": 0, "verify_failed": 3, "bench_median_ns": 1.0},
        {"name": "b", "verify_passed": 0, "verify_failed": 1, "bench_median_ns": 2.0},
    ]
    with pytest.raises(ValueError, match="no passing candidate"):
        select.pick_winner(results)
    with pytest.raises(ValueError, match="no passing candidate"):
        select.pick_winner([])


def test_compare_table_marks_winner_failed_slower():
    table = select.compare_table(_results())
    assert "WINNER" in table
    assert "cand-fast" in table
    lines = {ln.split()[0]: ln for ln in table.splitlines() if ln.strip()}
    assert "WINNER" in lines["cand-fast"]
    assert "FAILED" in lines["cand-broken"]
    assert "slower" in lines["cand-slow"]
    assert "*" in lines["cand-fast"]
