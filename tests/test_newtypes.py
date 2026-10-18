"""v0.4 item 2: 2-D arrays + String support (unit + e2e).

MethodInfo objects are built by hand (see _mi): java_ast.type_text
currently collapses "double[][]" to "double[]" (it counts `dimensions`
nodes instead of bracket pairs), so the e2e tests bypass the parser until
that is fixed. Everything downstream of MethodInfo is exercised for real
(glue -> javac/g++ -> verify over live JVMs).
"""

import random
from pathlib import Path

import pytest

from kernel_port import extractor as E
from kernel_port import harness as H
from kernel_port import java_ast as J
from kernel_port import jni
from kernel_port.purity import PurityReport

from .conftest import fill_translation

TYPES_REPO = Path(__file__).parent / "fixtures" / "types_kernels"

e2e = pytest.mark.e2e


def _mi(qualname, name, jtype, params):
    """MethodInfo exactly as java_ast should produce it (see module note)."""
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


def matmul2d_m():
    return _mi(
        "com.example.TypeKernels.matMul2d",
        "matMul2d",
        "double[][]",
        [("a", "double[][]"), ("b", "double[][]")],
    )


# ---------------------------------------------------------------------------
# unit: type gating
# ---------------------------------------------------------------------------


def test_check_harness_types_accepts_new_types():
    for m in [
        matmul2d_m(),
        _mi("q.C.add2d", "add2d", "int[][]", [("a", "int[][]"), ("b", "int[][]")]),
        _mi("q.C.greet", "greet", "String", [("name", "String")]),
        _mi("q.C.f", "f", "void", [("s", "String"), ("a", "double[][]")]),
    ]:
        H.check_harness_types(m)  # must not raise


def test_check_harness_types_rejects_other_2d_and_objects():
    m = _mi("q.C.f", "f", "float[][]", [])
    with pytest.raises(ValueError, match="float\\[\\]\\[\\]"):
        H.check_harness_types(m)
    m = _mi("q.C.f", "f", "void", [("o", "Object")])
    with pytest.raises(ValueError, match="Object"):
        H.check_harness_types(m)
    m = _mi("q.C.f", "f", "String[]", [])
    with pytest.raises(ValueError, match="String\\[\\]"):
        H.check_harness_types(m)


def test_glue_rejects_unsupported_2d():
    m = _mi("q.C.f", "f", "float[][]", [])
    with pytest.raises(ValueError, match="float\\[\\]\\[\\]"):
        jni.plan(m)


# ---------------------------------------------------------------------------
# unit: wire encoding round-trips
# ---------------------------------------------------------------------------


def test_encode_2d():
    assert H.encode_arg("double[][]", [[1.0, 2.0], [3.0, 4.0]]) == "a2d:2,2,1.0,2.0,3.0,4.0"
    assert H.encode_arg("double[][]", []) == "a2d:0,0"
    assert H.encode_arg("int[][]", [[1, 2], [3, 4]]) == "a2i:2,2,1,2,3,4"
    assert H.encode_arg("int[][]", [[]]) == "a2i:1,0"


def test_encode_string_escapes():
    assert H.encode_arg("String", "hi") == "s:hi"
    assert H.encode_arg("String", "") == "s:"
    assert H.encode_arg("String", "a b\tc\nd\\e") == "s:a\\sb\\tc\\nd\\\\e"


def test_parse_result_token_2d_and_string():
    assert H._parse_result_token("a2d:2,2,1.0,2.0,3.0,4.0") == (
        "a2d",
        (2, 2, [1.0, 2.0, 3.0, 4.0]),
    )
    assert H._parse_result_token("a2d:0,0") == ("a2d", (0, 0, []))
    assert H._parse_result_token("a2i:1,2,3,4") == ("a2i", (1, 2, [3, 4]))
    assert H._parse_result_token("s:a\\sb\\nc") == ("s", "a b\nc")


def test_encode_parse_roundtrip_2d_and_string():
    rng = random.Random(11)
    m = matmul2d_m()
    for _ in range(30):
        args = H.gen_case(m, rng)
        for p, v in zip(m.params, args):
            tok = H.encode_arg(p.jtype, v)
            tag, (rows, cols, flat) = H._parse_result_token(tok)
            assert tag == "a2d"
            assert (rows, cols) == (len(v), len(v[0]) if v else 0)
            assert flat == [x for row in v for x in row]
    ms = _mi("q.C.greet", "greet", "String", [("name", "String")])
    for _ in range(100):
        s = H._rand_string(rng)
        tok = H.encode_arg("String", s)
        assert " " not in tok  # still one token on the wire
        assert H._parse_result_token(tok) == ("s", s)


# ---------------------------------------------------------------------------
# unit: generation shapes
# ---------------------------------------------------------------------------


def test_gen_case_2d_square_and_matching():
    m = matmul2d_m()
    for seed in range(20):
        (a, b) = H.gen_case(m, random.Random(seed))
        assert len(a) == len(b)
        for row in a:
            assert len(row) == len(a)  # square


def test_gen_case_2d_size_param_takes_rows():
    m = _mi("q.C.f", "f", "void", [("a", "double[][]"), ("n", "int")])
    for seed in range(20):
        a, n = H.gen_case(m, random.Random(seed))
        assert n == len(a)


def test_gen_boundary_cases_new_types():
    m = _mi(
        "q.C.f", "f", "void", [("a", "double[][]"), ("s", "String"), ("n", "int")]
    )
    cases = H.gen_boundary_cases(m)
    assert cases  # non-empty, deterministic
    a0, s0, n0 = cases[0]
    assert a0 == [] and s0 == "" and n0 == 0  # all-neutral first case
    shapes = {(len(a), len(a[0]) if a else 0) for a, _s, _n in cases}
    assert (0, 0) in shapes and (1, 1) in shapes and (32, 32) in shapes
    strings = {s for _a, s, _n in cases}
    assert "" in strings and "hello world" in strings


# ---------------------------------------------------------------------------
# unit: glue codegen
# ---------------------------------------------------------------------------


def _gen(m, tmp_path):
    d = tmp_path / ("glue_" + m.name)
    return jni.generate(m, d)


def test_glue_2d_generates_flat_native(tmp_path):
    files = _gen(matmul2d_m(), tmp_path)
    java = files["java"].read_text()
    # public method keeps the natural 2-D signature...
    assert "public static double[][] matMul2d(double[][] a, double[][] b)" in java
    # ...and delegates to a private flat native (one JNI pin, not per-row)
    assert (
        "private static native double[][] matMul2dFlat("
        "double[] a_flat, int a_rows, int a_cols, "
        "double[] b_flat, int b_rows, int b_cols);" in java
    )
    assert "double[] flatten(double[][] m)" in java
    cpp = files["cpp"].read_text()
    assert "JNIEXPORT jobjectArray JNICALL Java_TypeKernelsNative_matMul2dFlat" in cpp
    # exactly one pin per flattened param (no per-row GetObjectArrayElement)
    assert "GetObjectArrayElement" not in cpp
    assert cpp.count("GetDoubleArrayElements(a_flat") == 1
    assert cpp.count("GetDoubleArrayElements(b_flat") == 1
    assert "int* p_out_rows, int* p_out_cols" in cpp
    assert 'FindClass("[D")' in cpp


def test_glue_2d_plain_mode_also_flat(tmp_path):
    d = tmp_path / "glue"
    files = jni.generate(matmul2d_m(), d, fallback=False)
    java = files["java"].read_text()
    assert "private static native double[][] matMul2dFlat(" in java
    assert "USE_NATIVE" not in java  # no dispatcher in plain mode
    cpp = files["cpp"].read_text()
    assert "Java_TypeKernelsNative_matMul2dFlat" in cpp


def test_glue_string_codegen(tmp_path):
    m = _mi("q.C.greet", "greet", "String", [("name", "String")])
    files = _gen(m, tmp_path)
    java = files["java"].read_text()
    assert "private static native String greetNative(String name);" in java
    cpp = files["cpp"].read_text()
    assert "GetStringUTFChars" in cpp
    assert "ReleaseStringUTFChars" in cpp
    assert "NewStringUTF" in cpp
    assert "static std::string kernel_greet(const char* p_name)" in cpp


def test_glue_int2d_codegen(tmp_path):
    m = _mi("q.C.add2d", "add2d", "int[][]", [("a", "int[][]"), ("b", "int[][]")])
    files = _gen(m, tmp_path)
    cpp = files["cpp"].read_text()
    assert "GetIntArrayElements" in cpp
    assert 'FindClass("[I")' in cpp
    assert "const int32_t* p_a, int p_a_rows, int p_a_cols" in cpp


# ---------------------------------------------------------------------------
# unit: extractor prompt material
# ---------------------------------------------------------------------------


def _extraction(m):
    return E.Extraction(method=m, purity=PurityReport(pure=True), closure=[])


def test_extractor_type_mapping_note_for_2d_and_string():
    text = E.render_text(_extraction(matmul2d_m()))
    assert "### Type mapping for the native signature" in text
    assert "flattened row-major" in text
    assert "p_a_rows" in text
    assert "*p_out_rows" in text
    text = E.render_text(
        _extraction(_mi("q.C.greet", "greet", "String", [("name", "String")]))
    )
    assert "GetStringUTFChars" in text
    assert "Modified UTF-8" in text


def test_extractor_no_note_for_plain_types(repo):
    m = repo.find_method("com.example.Kernels.dot")
    text = E.render_text(_extraction(m))
    assert "Type mapping for the native signature" not in text


# ---------------------------------------------------------------------------
# e2e: verify over live JVMs
# ---------------------------------------------------------------------------


def _build(qualname, name, jtype, params, tmp_path, body):
    m = _mi(qualname, name, jtype, params)
    glue_dir = tmp_path / "glue"
    jni.generate(m, glue_dir)
    fill_translation(glue_dir, qualname, body)
    return H.build_env(TYPES_REPO, m, glue_dir, tmp_path / "work")


_MATMUL2D_BODY = """    int n = p_a_rows;
    std::vector<double> c((size_t)n * n, 0.0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) s += p_a[i * n + k] * p_b[k * n + j];
            c[i * n + j] = s;
        }
    *p_out_rows = n;
    *p_out_cols = n;
    return c;"""

_MATMUL2D_WRONG = """    int n = p_a_rows;
    std::vector<double> c((size_t)n * n, 0.0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) s += p_a[i * n + k] * p_b[k * n + j];
            c[i * n + j] = s + 1.0;
        }
    *p_out_rows = n;
    *p_out_cols = n;
    return c;"""

_GREET_BODY = """    return std::string("hello ") + p_name;"""

_SCALE2D_BODY = """    for (int i = 0; i < p_a_rows; i++)
        for (int j = 0; j < p_a_cols; j++)
            const_cast<double*>(p_a)[i * p_a_cols + j] *= p_k;"""


@e2e
def test_e2e_verify_matmul2d_passes(tmp_path):
    env = _build(
        "com.example.TypeKernels.matMul2d", "matMul2d", "double[][]",
        [("a", "double[][]"), ("b", "double[][]")], tmp_path, _MATMUL2D_BODY,
    )
    report = H.run_verify(env, cases=20, seed=5)
    assert report.all_passed, [c.detail for c in report.cases if not c.ok]
    assert report.passed == 20


@e2e
def test_e2e_verify_matmul2d_detects_wrong_translation(tmp_path):
    env = _build(
        "com.example.TypeKernels.matMul2d", "matMul2d", "double[][]",
        [("a", "double[][]"), ("b", "double[][]")], tmp_path, _MATMUL2D_WRONG,
    )
    report = H.run_verify(env, cases=10, seed=5)
    assert not report.all_passed
    assert report.failed > 0


@e2e
def test_e2e_verify_greet_string_passes(tmp_path):
    env = _build(
        "com.example.TypeKernels.greet", "greet", "String",
        [("name", "String")], tmp_path, _GREET_BODY,
    )
    report = H.run_verify(env, cases=20, seed=9)
    assert report.all_passed, [c.detail for c in report.cases if not c.ok]
    assert report.passed == 20
    # boundary sweep exercises the escape protocol (spaces, tabs, newlines)
    assert report.boundary_passed == len(report.boundary) > 0


@e2e
def test_e2e_verify_scale2d_void_inout(tmp_path):
    # void 2-D kernel: the harness echoes the mutated matrix; this checks
    # the JNI copy-back + the Java unflattenInto copy-back chain.
    env = _build(
        "com.example.TypeKernels.scale2d", "scale2d", "void",
        [("a", "double[][]"), ("k", "double")], tmp_path, _SCALE2D_BODY,
    )
    report = H.run_verify(env, cases=10, seed=3, boundary=False)
    assert report.all_passed, [c.detail for c in report.cases if not c.ok]
    assert report.passed == 10
