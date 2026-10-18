"""Tests for the JNI glue safety lint (kernel_port/lint.py) and written_arrays."""

import textwrap
from pathlib import Path

import pytest

from kernel_port import java_ast as J
from kernel_port import jni
from kernel_port import rust_glue
from kernel_port.lint import LintWarning, lint_glue, lint_rust
from kernel_port.purity import check_purity


def _method(repo, qualname):
    m = repo.find_method(qualname)
    assert m is not None, f"method not found: {qualname}"
    return m


# ---------------------------------------------------------------------------
# lint: clean generated glue -> zero warnings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fallback", [True, False])
def test_lint_clean_generated_glue_dot(repo, tmp_path, fallback):
    m = _method(repo, "com.example.Kernels.dot")
    files = jni.generate(m, tmp_path / "glue", fallback=fallback)
    assert lint_glue(files["cpp"].read_text()) == []


def test_lint_clean_generated_glue_void_inout(repo, tmp_path):
    # matMul writes through c (mode 0) and only reads a/b (JNI_ABORT).
    m = _method(repo, "com.example.Kernels.matMul")
    files = jni.generate(m, tmp_path / "glue")
    assert lint_glue(files["cpp"].read_text()) == []


def test_lint_clean_generated_glue_array_return(repo, tmp_path):
    m = _method(repo, "com.example.Kernels.sieve")
    files = jni.generate(m, tmp_path / "glue")
    assert lint_glue(files["cpp"].read_text()) == []


# ---------------------------------------------------------------------------
# lint: hand-crafted bad snippets -> each warning kind fires with the line
# ---------------------------------------------------------------------------

_MISSING_RELEASE = """\
JNIEXPORT void JNICALL Java_X_y(JNIEnv* env, jclass, jdoubleArray a) {
    double* p_a = env->GetDoubleArrayElements(a, nullptr);
    if (p_a == nullptr) return;
    double x = p_a[0];
}
"""


def test_lint_missing_release():
    ws = lint_glue(_MISSING_RELEASE)
    assert [(w.kind, w.line) for w in ws] == [("missing_release", 2)]
    assert "p_a" in ws[0].message and "ReleaseDoubleArrayElements" in ws[0].message


_MISSING_CHECK = """\
JNIEXPORT void JNICALL Java_X_y(JNIEnv* env, jclass, jdoubleArray a) {
    double* p_a = env->GetDoubleArrayElements(a, nullptr);
    double x = p_a[0];
    env->ReleaseDoubleArrayElements(a, p_a, 0);
}
"""


def test_lint_missing_null_check():
    ws = lint_glue(_MISSING_CHECK)
    assert [(w.kind, w.line) for w in ws] == [("missing_null_check", 2)]
    assert "p_a" in ws[0].message


_LOCAL_REF_LEAK = """\
JNIEXPORT jobjectArray JNICALL Java_X_y(JNIEnv* env, jclass, int n, jclass cls) {
    jobjectArray out = env->NewObjectArray(n, cls, nullptr);
    for (int i = 0; i < n; i++) {
        jobject row = env->NewObject(cls, mid);
        env->SetObjectArrayElement(out, i, row);
    }
    return out;
}
"""


def test_lint_local_ref_leak_in_loop():
    ws = lint_glue(_LOCAL_REF_LEAK)
    # the NewObjectArray on line 2 is outside the loop: only the in-loop
    # NewObject on line 4 warns.
    assert [(w.kind, w.line) for w in ws] == [("local_ref_leak", 4)]
    assert "DeleteLocalRef" in ws[0].message


_LOOP_WITH_DELETE_IS_CLEAN = """\
    for (int i = 0; i < n; i++) {
        jobject row = env->NewObject(cls, mid);
        env->SetObjectArrayElement(out, i, row);
        env->DeleteLocalRef(row);
    }
    while (more) {
        jstring s = env->NewStringUTF("x");
        env->DeleteLocalRef(s);
    }
"""


def test_lint_loop_with_delete_local_ref_is_clean():
    assert lint_glue(_LOOP_WITH_DELETE_IS_CLEAN) == []


_NULL_CHECK_FORMS_OK = """\
    double* p_a = env->GetDoubleArrayElements(a, nullptr);
    if (!p_a) return 0;
    double* p_b = env->GetDoubleArrayElements(b, nullptr);
    if (p_b != nullptr) { }
    const char* p_s = env->GetStringUTFChars(s, nullptr);
    if (nullptr == p_s) return 0;
    double x = p_a[0] + p_b[0];
    env->ReleaseDoubleArrayElements(a, p_a, 0);
    env->ReleaseDoubleArrayElements(b, p_b, 0);
    env->ReleaseStringUTFChars(s, p_s);
"""


def test_lint_null_check_forms_accepted():
    # !p_x, != nullptr and nullptr == p_x all count as null checks.
    assert lint_glue(_NULL_CHECK_FORMS_OK) == []


def test_lint_warnings_sorted_by_line():
    src = _LOCAL_REF_LEAK + _MISSING_RELEASE
    ws = lint_glue(src)
    lines = [w.line for w in ws]
    assert lines == sorted(lines)
    assert [w.kind for w in ws] == ["local_ref_leak", "missing_release"]


def test_lint_warning_shape():
    w = LintWarning(kind="missing_release", line=3, message="msg")
    assert (w.kind, w.line, w.message) == ("missing_release", 3, "msg")


# ---------------------------------------------------------------------------
# written_arrays unit tests (temp repo, so the fixture repo is untouched)
# ---------------------------------------------------------------------------

_WRITE_SRC = """\
package t;
import java.util.Arrays;
public class W {
    public static void fill(double[] a) {
        for (int i = 0; i < a.length; i++) a[i] = 1.0;
    }
    public static double wrap(double[] a) { fill(a); return a[0]; }
    public static double readOnly(double[] a) { return a[0] + a[1]; }
    public static double viaScalar(double[] a) { return first(a[0]); }
    static double first(double x) { return x; }
    public static double sorted(double[] a) { Arrays.sort(a); return a[0]; }
    public static double rec(double[] a) {
        if (a.length == 0) return 0;
        return rec(a);
    }
    public static void compound(double[] a) { a[0] += 1; }
    public static void upd(double[] a) { a[0]++; }
    public static double unknownTarget(double[] a) { return Helper.id(a); }
}
"""


@pytest.fixture()
def wrepo(tmp_path):
    pkg = tmp_path / "src" / "t"
    pkg.mkdir(parents=True)
    (pkg / "W.java").write_text(textwrap.dedent(_WRITE_SRC), encoding="utf-8")
    return J.scan_repo(tmp_path / "src")


def _w(wrepo, name):
    return _method(wrepo, f"t.W.{name}")


def test_written_arrays_direct_write(wrepo):
    assert jni.written_arrays(_w(wrepo, "fill"), wrepo) == {"a"}


def test_written_arrays_compound_and_update(wrepo):
    assert jni.written_arrays(_w(wrepo, "compound"), wrepo) == {"a"}
    assert jni.written_arrays(_w(wrepo, "upd"), wrepo) == {"a"}


def test_written_arrays_read_only(wrepo):
    assert jni.written_arrays(_w(wrepo, "readOnly"), wrepo) == set()
    # scalar element passed to a pure callee: not a write-through
    assert jni.written_arrays(_w(wrepo, "viaScalar"), wrepo) == set()


def test_written_arrays_write_via_pure_callee(wrepo):
    # fill() passes the purity gate (param writes are allowed) but still
    # writes through the array: the analysis must see through the call.
    assert check_purity(_w(wrepo, "fill"), wrepo).pure
    assert jni.written_arrays(_w(wrepo, "wrap"), wrepo) == {"a"}


def test_written_arrays_unknown_callee_is_conservative(wrepo):
    # Helper is not in the repo: unresolvable -> assume written.
    assert jni.written_arrays(_w(wrepo, "unknownTarget"), wrepo) == {"a"}


def test_written_arrays_jdk_call_is_conservative(wrepo):
    # Arrays.sort(a) is a JDK call -> assume written.
    assert jni.written_arrays(_w(wrepo, "sorted"), wrepo) == {"a"}


def test_written_arrays_recursion_is_conservative(wrepo):
    assert jni.written_arrays(_w(wrepo, "rec"), wrepo) == {"a"}


def test_written_arrays_no_repo_is_conservative(wrepo):
    # without a repo no callee resolves: wrap's fill(a) counts as written,
    # while a body with no escaping calls still analyzes precisely.
    assert jni.written_arrays(_w(wrepo, "wrap")) == {"a"}
    assert jni.written_arrays(_w(wrepo, "readOnly")) == set()
    assert jni.written_arrays(_w(wrepo, "fill")) == {"a"}


def test_written_arrays_no_body_is_conservative():
    # a MethodInfo with no analyzable body: every array param is written
    # (absence of evidence is not evidence of read-only).
    m = J.MethodInfo(
        qualname="q.C.m",
        class_qualname="q.C",
        name="m",
        jtype="void",
        params=[J.Param("a", "double[]"), J.Param("n", "int")],
        modifiers={"static"},
        is_static=True,
        is_constructor=False,
        file="C.java",
        lineno=1,
        end_lineno=2,
        source="",
        body=None,
    )
    assert jni.written_arrays(m) == {"a"}


def test_written_arrays_fixture_methods(repo):
    assert jni.written_arrays(_method(repo, "com.example.Kernels.dot"), repo) == set()
    assert jni.written_arrays(
        _method(repo, "com.example.Kernels.matMul"), repo
    ) == {"c"}
    assert jni.written_arrays(_method(repo, "com.example.Kernels.norm"), repo) == set()


# ---------------------------------------------------------------------------
# lint_rust: real rust_glue template output -> zero warnings (template drift guard)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "qualname", ["com.example.Kernels.dot", "com.example.Kernels.matMul", "com.example.Kernels.sieve"]
)
def test_lint_rust_clean_generated_template(repo, tmp_path, qualname):
    # dot: scalar kernel; matMul: void with a written array; sieve: array
    # return. If the template ever drifts (drops #[no_mangle], gains an
    # unwrap, changes the ABI...), the lint must catch it here first.
    m = _method(repo, qualname)
    files = rust_glue.generate(m, tmp_path / "glue")
    text = files["rust"].read_text(encoding="utf-8")
    assert lint_rust(text) == []


# ---------------------------------------------------------------------------
# lint_rust: hand-crafted bad snippets -> each warning kind fires with the line
# ---------------------------------------------------------------------------

_RUST_MISSING_NO_MANGLE = """\
use jni::JNIEnv;
pub extern "system" fn Java_X_y<'local>(mut env: JNIEnv<'local>) {
    0
}
"""


def test_lint_rust_missing_no_mangle():
    ws = lint_rust(_RUST_MISSING_NO_MANGLE)
    assert [(w.kind, w.line) for w in ws] == [("missing_no_mangle", 2)]
    assert "#[no_mangle]" in ws[0].message


_RUST_NO_MANGLE_OK = """\
use jni::JNIEnv;

// a comment between the attribute and the fn is tolerated
#[no_mangle]
pub extern "system" fn Java_X_y<'local>(mut env: JNIEnv<'local>) {
    0
}
"""


def test_lint_rust_no_mangle_present_is_clean():
    assert lint_rust(_RUST_NO_MANGLE_OK) == []


_RUST_FFI_PANIC = """\
use jni::JNIEnv;

#[no_mangle]
pub extern "system" fn Java_X_y<'local>(mut env: JNIEnv<'local>) -> jint {
    let v = compute().unwrap();
    let s = other.expect("must exist");
    panic!("oops");
    v
}
"""


def test_lint_rust_ffi_panic():
    ws = lint_rust(_RUST_FFI_PANIC)
    assert [(w.kind, w.line) for w in ws] == [
        ("ffi_panic", 5),
        ("ffi_panic", 6),
        ("ffi_panic", 7),
    ]
    assert "jni_try!" in ws[0].message


_RUST_WRONG_ABI = """\
use jni::JNIEnv;

#[no_mangle]
pub extern "C" fn Java_X_y<'local>(mut env: JNIEnv<'local>) {
}
"""


def test_lint_rust_wrong_abi():
    ws = lint_rust(_RUST_WRONG_ABI)
    assert [(w.kind, w.line) for w in ws] == [("wrong_abi", 4)]
    assert '"C"' in ws[0].message and '"system"' in ws[0].message


_RUST_MISSING_JNI_USE = """\
// someone deleted the imports
#[no_mangle]
pub extern "system" fn Java_X_y<'local>() {
}
"""


def test_lint_rust_missing_jni_use():
    ws = lint_rust(_RUST_MISSING_JNI_USE)
    assert [(w.kind, w.line) for w in ws] == [("missing_jni_use", 1)]
    assert "JNIEnv" in ws[0].message


def test_lint_rust_warnings_sorted_by_line():
    src = _RUST_MISSING_JNI_USE + "\n" + _RUST_FFI_PANIC.replace(
        "use jni::JNIEnv;\n\n", ""
    )
    ws = lint_rust(src)
    lines = [w.line for w in ws]
    assert lines == sorted(lines)
    assert ws[0].kind == "missing_jni_use"
    assert all(w.kind == "ffi_panic" for w in ws[1:])
