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
    m = _method(repo, "com.example.Kernels.sieve")
    files = jni.generate(m, tmp_path / "glue")
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
        jtype="String",
        params=[],
        modifiers={"static"},
        is_static=True,
        is_constructor=False,
        file="T.java",
        lineno=1,
        end_lineno=2,
        source="",
    )
    with pytest.raises(ValueError, match="unsupported type 'String'"):
        jni.generate(m, tmp_path / "glue")


def test_glue_rejects_instance_method(repo, tmp_path):
    from kernel_port import java_ast as J

    m = _method(repo, "com.example.Kernels.dot")
    m2 = J.MethodInfo(**{**m.__dict__, "is_static": False})
    with pytest.raises(ValueError, match="static"):
        jni.generate(m2, tmp_path / "glue")
