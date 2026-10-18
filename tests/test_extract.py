"""Tests for `extract`: closure, purity report, prompt material."""

import pytest

from kernel_port import extractor


def test_extract_closure_includes_helper(repo):
    ext = extractor.extract(repo, "com.example.Kernels.norm")
    assert ext.purity.pure
    kinds = {(c.kind, c.qualname) for c in ext.closure}
    assert ("method", "com.example.Kernels.square") in kinds


def test_extract_self_contained_has_empty_closure(repo):
    ext = extractor.extract(repo, "com.example.Kernels.fib")
    assert ext.purity.pure
    assert ext.closure == []


def test_extract_impure_refuses_material(repo):
    ext = extractor.extract(repo, "com.example.Kernels.noisy")
    assert not ext.purity.pure
    text = extractor.render_text(ext)
    assert "IMPURE" in text
    assert "Refusing to emit translation material" in text
    assert "```java" not in text  # no source emitted for impure methods


def test_extract_text_has_prompt_material(repo):
    ext = extractor.extract(repo, "com.example.Kernels.dot")
    text = extractor.render_text(ext)
    assert "double dot(double[] a, double[] b)" in text
    assert "PURE" in text
    assert "public static double dot" in text
    assert "Target language: C++17" in text


def test_extract_json_shape(repo):
    ext = extractor.extract(repo, "com.example.Kernels.sieve")
    data = extractor.to_json(ext)
    assert data["method"] == "com.example.Kernels.sieve"
    assert data["return_type"] == "int[]"
    assert data["params"] == [{"name": "n", "type": "int"}]
    assert data["pure"] is True
    assert isinstance(data["source"], str) and "sieve" in data["source"]


def test_extract_unknown_method(repo):
    with pytest.raises(ValueError, match="not found"):
        extractor.extract(repo, "com.example.Kernels.nope")


def test_render_text_backend_notes(repo):
    # v0.9 dogfood: the translation notes always said C++17/JNI even when the
    # agent was targeting FFM or Rust.
    from kernel_port import extractor

    ext = extractor.extract(repo, "com.example.Kernels.dot")
    jni_text = extractor.render_text(ext)
    assert "C++17" in jni_text and "JNI glue" in jni_text
    ffm_text = extractor.render_text(ext, "ffm")
    assert "plain C ABI" in ffm_text and "--backend ffm" in ffm_text
    assert "C++17" not in ffm_text
    rust_text = extractor.render_text(ext, "rust")
    assert "Target language: Rust" in rust_text and "--backend rust" in rust_text
    assert "C++17" not in rust_text
