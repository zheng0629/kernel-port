"""v0.4 item 4: per-type float tolerance in verify (unit + e2e).

Float32's epsilon is ~1.2e-7, so a translation that accumulates in double
and casts back to float is routinely ~1e-7 off the Java float32 result --
the old single 1e-9 threshold flaked on it. Double results keep a tight
1e-12 threshold.
"""

from pathlib import Path

import pytest

from kernel_port import harness as H
from kernel_port import java_ast as J

from .conftest import fill_translation

TYPES_REPO = Path(__file__).parent / "fixtures" / "types_kernels"

e2e = pytest.mark.e2e


def _mi(qualname, name, jtype, params):
    cls = qualname.rsplit(".", 1)[0]
    return J.MethodInfo(
        qualname=qualname,
        class_qualname=cls,
        name=name,
        jtype=jtype,
        params=[J.Param(n, t) for n, t in params],
        modifiers={"public", "static"},
        is_static=True,
        is_constructor=False,
        file="src/com/example/TypeKernels.java",
        lineno=1,
        end_lineno=50,
        source=f"{jtype} {name}(...)",
    )


def fsum_m():
    return _mi(
        "com.example.TypeKernels.fsum", "fsum", "float", [("a", "float[]")]
    )


# ---------------------------------------------------------------------------
# unit: per-type thresholds
# ---------------------------------------------------------------------------


def test_float_allows_1e_minus_7_but_not_1e_minus_5():
    m = fsum_m()
    # ~2.8e-7 relative: the realistic float32-accumulation window
    ok, _ = H.results_equal(m, "f:18.7520409", "f:18.7520447")
    assert ok
    ok, _ = H.results_equal(m, "f:1.0", "f:1.00001")  # 1e-5: genuinely wrong
    assert not ok


def test_float_old_single_threshold_would_flake():
    m = fsum_m()
    # Under the pre-v0.4 single 1e-9 threshold this realistic float32
    # accumulation difference failed; tol_float=1e-6 accepts it.
    ok, _ = H.results_equal(m, "f:18.7520409", "f:18.7520447",
                             tol_float=1e-9, tol_double=1e-9)
    assert not ok


def test_double_keeps_tight_threshold():
    m = _mi("q.C.dot", "dot", "double", [("a", "double[]")])
    ok, _ = H.results_equal(m, "d:1.0", "d:1.0000001")  # 1e-7: too big for double
    assert not ok
    ok, _ = H.results_equal(m, "d:1.0", "d:1.0000000000001")  # 1e-13: fine
    assert ok
    # ...unless the caller explicitly loosens it
    ok, _ = H.results_equal(m, "d:1.0", "d:1.0000001", tol_double=1e-6)
    assert ok


def test_float_array_per_element_tolerance():
    m = fsum_m()
    ok, _ = H.results_equal(m, "af:1.0,2.0", "af:1.0000002,2.0")
    assert ok
    ok, _ = H.results_equal(m, "af:1.0,2.0", "af:1.0000002,2.00001")
    assert not ok


def test_double_2d_array_per_element_tolerance():
    m = _mi("q.C.m", "m", "double[][]", [("a", "double[][]")])
    ok, _ = H.results_equal(m, "a2d:1,2,1.0,2.0", "a2d:1,2,1.0,2.0000000000001")
    assert ok
    ok, _ = H.results_equal(m, "a2d:1,2,1.0,2.0", "a2d:1,2,1.0,2.0000001")
    assert not ok
    # shape mismatches still fail loudly, regardless of tolerance
    ok, detail = H.results_equal(m, "a2d:1,2,1.0,2.0", "a2d:2,1,1.0,2.0")
    assert not ok and "shape" in detail


def test_results_equal_backward_compatible_signature():
    # old 3-arg calls keep working with the v0.4 defaults
    m = fsum_m()
    ok, _ = H.results_equal(m, "f:1.5", "f:1.5")
    assert ok
    m2 = _mi("q.C.fib", "fib", "long", [("n", "int")])
    ok, _ = H.results_equal(m2, "l:55", "l:55")
    assert ok


def test_int_arrays_still_exact():
    m = _mi("q.C.s", "s", "int[]", [("n", "int")])
    ok, _ = H.results_equal(m, "ai:1,2,3", "ai:1,2,4")
    assert not ok


# ---------------------------------------------------------------------------
# e2e: float kernel over live JVMs
# ---------------------------------------------------------------------------

# Plausible agent translation: accumulate in double, cast back to float.
# This is *more* accurate than the Java original -- differential testing
# must accept it via tol_float, not reject it via a 1e-9 threshold.
_FSUM_BODY = """    double s = 0.0;
    for (int i = 0; i < p_a_len; i++) s += p_a[i];
    return (float)s;"""

# Deliberately wrong: truncated loop.
_FSUM_WRONG = """    float s = 0.0f;
    for (int i = 0; i < p_a_len - 1; i++) s += p_a[i];
    return s;"""


def _build(body, tmp_path):
    m = fsum_m()
    glue_dir = tmp_path / "glue"
    from kernel_port import jni

    jni.generate(m, glue_dir)
    fill_translation(glue_dir, m.qualname, body)
    return H.build_env(TYPES_REPO, m, glue_dir, tmp_path / "work")


@e2e
def test_e2e_verify_float_accumulation_passes(tmp_path):
    # boundary=False: the boundary sweep's 1024-element array accumulates
    # ~4e-6 float32 error (error grows with sqrt(n)), legitimately above
    # tol_float -- the ~1e-7 window this tests is the random cases (n<=8).
    env = _build(_FSUM_BODY, tmp_path)
    report = H.run_verify(env, cases=20, seed=42, boundary=False)
    assert report.all_passed, [c.detail for c in report.cases if not c.ok]
    assert report.passed == 20
    # the tolerance must actually have been exercised: at least one case
    # differs textually between Java float32 and the double accumulator
    assert any(c.java_result != c.jni_result for c in report.cases)


@e2e
def test_e2e_verify_float_wrong_translation_still_fails(tmp_path):
    env = _build(_FSUM_WRONG, tmp_path)
    report = H.run_verify(env, cases=20, seed=42)
    assert not report.all_passed
    assert report.failed > 0
