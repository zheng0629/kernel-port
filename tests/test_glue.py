"""Tests for `glue`: file generation, JNI correctness, compilability."""

import subprocess

import pytest

from kernel_port import jni

from .conftest import FIXTURE_REPO


def _method(repo, qualname):
    m = repo.find_method(qualname)
    assert m is not None
    return m


def test_glue_generates_three_files(repo, tmp_path):
    m = _method(repo, "com.example.Kernels.dot")
    files = jni.generate(m, tmp_path / "glue")
    assert files["java"].name == "KernelsNative.java"
    assert files["cpp"].name == "kernels_jni.cpp"
    assert files["cmake"].name == "CMakeLists.txt"
    for p in files.values():
        assert p.exists()


def test_glue_java_declares_native(repo, tmp_path):
    # v0.3 default: dispatcher with fallback switch (USE_NATIVE).
    m = _method(repo, "com.example.Kernels.sieve")
    files = jni.generate(m, tmp_path / "glue")
    src = files["java"].read_text()
    assert 'System.getProperty("kernelport.native"' in src
    assert "setUseNative" in src
    assert "com.example.Kernels.sieve" in src  # fallback delegates to original
    assert "private static native int[] sieveNative(int n);" in src
    assert 'System.loadLibrary("kernels")' in src


def test_glue_no_fallback_is_plain_native(repo, tmp_path):
    # fallback=False restores the v0.1 plain native declaration.
    m = _method(repo, "com.example.Kernels.sieve")
    files = jni.generate(m, tmp_path / "glue", fallback=False)
    src = files["java"].read_text()
    assert "public static native int[] sieve(int n);" in src
    assert 'System.loadLibrary("kernels")' in src


def test_glue_cpp_has_jni_symbol_and_todo(repo, tmp_path):
    m = _method(repo, "com.example.Kernels.dot")
    files = jni.generate(m, tmp_path / "glue")
    src = files["cpp"].read_text()
    assert "Java_KernelsNative_dot" in src
    assert "TODO(agent)" in src
    assert "kernel_dot" in src
    assert '#include <jni.h>' in src


def test_glue_cpp_compiles(repo, tmp_path):
    """The skeleton (placeholder body) must compile to a shared lib."""
    from kernel_port.harness import compile_jni

    m = _method(repo, "com.example.Kernels.matMul")
    files = jni.generate(m, tmp_path / "glue")
    compile_jni(files["cpp"], tmp_path / "glue" / "libkernels.so")


def test_glue_cmake_references_jni(repo, tmp_path):
    m = _method(repo, "com.example.Kernels.dot")
    files = jni.generate(m, tmp_path / "glue")
    cmake = files["cmake"].read_text()
    assert "find_package(JNI" in cmake
    assert "kernels_jni.cpp" in cmake


def test_glue_rejects_unsupported_type(tmp_path):
    from kernel_port import java_ast as J

    m = J.MethodInfo(
        qualname="p.T.m",
        class_qualname="p.T",
        name="m",
        jtype="float[][]",  # v0.4 supports double[][]/int[][]/String; float[][] stays out
        params=[],
        modifiers={"static"},
        is_static=True,
        is_constructor=False,
        file="T.java",
        lineno=1,
        end_lineno=2,
        source="",
    )
    with pytest.raises(ValueError, match=r"unsupported type 'float\[\]\[\]'"):
        jni.generate(m, tmp_path / "glue")


def test_glue_rejects_instance_method(repo, tmp_path):
    from kernel_port import java_ast as J

    m = _method(repo, "com.example.Kernels.dot")
    m2 = J.MethodInfo(**{**m.__dict__, "is_static": False})
    with pytest.raises(ValueError, match="static"):
        jni.generate(m2, tmp_path / "glue")


def test_glue_cpp_null_checks_and_release_modes(repo, tmp_path):
    # dot: both arrays are read-only -> nullptr check + JNI_ABORT (no copy-back)
    m = _method(repo, "com.example.Kernels.dot")
    files = jni.generate(m, tmp_path / "glue")
    cpp = files["cpp"].read_text()
    assert "if (p_a == nullptr) return 0;" in cpp
    assert "if (p_b == nullptr) return 0;" in cpp
    assert cpp.count("JNI_ABORT); // read-only: skip copy-back") == 2
    assert "copy back written elements" not in cpp

    # matMul: c is written -> mode 0 (copy back); a, b read-only -> JNI_ABORT
    m = _method(repo, "com.example.Kernels.matMul")
    files = jni.generate(m, tmp_path / "glue2")
    cpp = files["cpp"].read_text()
    assert "if (p_c == nullptr) return;" in cpp  # void kernel: bare return
    assert ", 0); // in/out: copy back written elements" in cpp
    assert cpp.count("JNI_ABORT); // read-only: skip copy-back") == 2


def test_glue_cpp_hardening_plain_mode(repo, tmp_path):
    # fallback=False shares render_cpp: same null checks / release modes.
    m = _method(repo, "com.example.Kernels.dot")
    files = jni.generate(m, tmp_path / "glue", fallback=False)
    cpp = files["cpp"].read_text()
    assert "if (p_a == nullptr) return 0;" in cpp
    assert "JNI_ABORT); // read-only: skip copy-back" in cpp
