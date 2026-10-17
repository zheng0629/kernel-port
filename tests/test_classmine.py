"""Tests for class-level mining (workstream C2).

Fixture repo: tests/fixtures/class_kernels (package com.example, qualnames
disjoint from tests/fixtures/sample_kernels):

- Util      -- stateless utility class            -> portable / "stateless"
- Point     -- immutable (final fields + ctor)    -> portable / "immutable"
- Counter   -- mutable field written in method    -> rejected
- NoisyBox  -- impure method (System.out.println) -> rejected
"""

from pathlib import Path

import pytest

from kernel_port import classglue, classmine, extractor
from kernel_port import java_ast as J
from kernel_port.purity import check_class_portable

FIXTURES = Path(__file__).parent / "fixtures" / "class_kernels"


@pytest.fixture()
def crepo():
    return J.scan_repo(FIXTURES)


def _by_qualname(cands, qualname):
    for c in cands:
        if c.qualname == qualname:
            return c
    raise AssertionError(f"no candidate {qualname}")


def test_find_class_candidates_classifies_all_four(crepo):
    cands = classmine.find_class_candidates(crepo)
    assert len(cands) == 4

    util = _by_qualname(cands, "com.example.Util")
    assert util.portable and util.kind == "stateless" and util.reasons == []

    point = _by_qualname(cands, "com.example.Point")
    assert point.portable and point.kind == "immutable" and point.reasons == []

    counter = _by_qualname(cands, "com.example.Counter")
    assert not counter.portable and counter.kind == "rejected"
    assert any("count" in r and "increment" in r for r in counter.reasons), (
        counter.reasons
    )

    noisy = _by_qualname(cands, "com.example.NoisyBox")
    assert not noisy.portable and noisy.kind == "rejected"
    assert any("println" in r for r in noisy.reasons), noisy.reasons


def test_portable_candidates_sort_first(crepo):
    cands = classmine.find_class_candidates(crepo)
    seen_rejected = False
    for c in cands:
        if not c.portable:
            seen_rejected = True
        assert not (seen_rejected and c.portable), "portable must sort first"


def test_candidate_fields_and_score(crepo):
    cands = classmine.find_class_candidates(crepo)
    util = _by_qualname(cands, "com.example.Util")
    assert util.methods == [
        "com.example.Util.lerp",
        "com.example.Util.clamp",
        "com.example.Util.goldenScale",
    ]
    assert util.fields == ["GOLDEN"]
    assert util.score > 0
    point = _by_qualname(cands, "com.example.Point")
    assert point.fields == ["x", "y"]
    assert point.score > 0


def test_check_class_portable_point_direct(crepo):
    rep = check_class_portable(crepo.classes["com.example.Point"], crepo)
    assert rep.portable and rep.kind == "immutable"
    assert set(rep.method_reports) >= {
        "com.example.Point.Point",
        "com.example.Point.norm",
        "com.example.Point.distance",
    }
    assert all(r.pure for r in rep.method_reports.values()), {
        qn: r.reasons for qn, r in rep.method_reports.items() if not r.pure
    }


def test_check_class_portable_counter_reasons(crepo):
    rep = check_class_portable(crepo.classes["com.example.Counter"], crepo)
    assert not rep.portable and rep.kind == "rejected"
    assert any("count" in r and "increment" in r for r in rep.reasons)


def test_extract_class_util_emits_all_methods(crepo):
    ext = extractor.extract_class(crepo, "com.example.Util")
    assert ext.class_purity.portable
    assert [e.method.qualname for e in ext.method_extractions] == [
        "com.example.Util.clamp",
        "com.example.Util.goldenScale",
        "com.example.Util.lerp",
    ]
    assert all(e.purity.pure for e in ext.method_extractions)
    assert [f.name for f in ext.fields] == ["GOLDEN"]

    text = extractor.render_text_class(ext)
    assert "PORTABLE" in text
    assert "lerp" in text and "GOLDEN" in text

    js = extractor.to_json_class(ext)
    assert js["class"] == "com.example.Util"
    assert js["kind"] == "stateless"
    assert js["portable"] is True
    assert len(js["methods"]) == 3


def test_extract_class_refuses_counter(crepo):
    ext = extractor.extract_class(crepo, "com.example.Counter")
    assert not ext.class_purity.portable
    assert "Refusing to emit" in extractor.render_text_class(ext)


def test_extract_class_unknown_class(crepo):
    with pytest.raises(ValueError, match="class not found"):
        extractor.extract_class(crepo, "com.example.Missing")


def test_generate_class_util_combined_files(crepo, tmp_path):
    files = classglue.generate_class(crepo, "com.example.Util", tmp_path)
    assert set(files) == {"java", "cpp", "cmake"}

    java = files["java"].read_text(encoding="utf-8")
    assert "class UtilNative" in java
    assert "public static native double lerp(double a, double b, double t);" in java
    assert "public static native double clamp(double v, double lo, double hi);" in java
    assert "public static native double goldenScale(double v);" in java

    cpp = files["cpp"].read_text(encoding="utf-8")
    # one combined translation unit: single preamble + single extern block
    assert cpp.count("#include <jni.h>") == 1
    assert cpp.count('extern "C" {') == 1
    for name in ("lerp", "clamp", "goldenScale"):
        assert f"Java_UtilNative_{name}" in cpp
        assert f"kernel_{name}" in cpp
    assert "TODO(agent)" in cpp

    assert files["cmake"].read_text(encoding="utf-8")


def test_generate_class_skips_instance_methods(crepo, tmp_path):
    # Counter's methods are all instance methods / constructors: nothing
    # gluable, but files are still written with the skips documented.
    files = classglue.generate_class(crepo, "com.example.Counter", tmp_path)
    cpp = files["cpp"].read_text(encoding="utf-8")
    assert "No gluable methods" in cpp
    assert "increment" in cpp
    java = files["java"].read_text(encoding="utf-8")
    assert "class CounterNative" in java


def test_generate_class_unknown_class(crepo, tmp_path):
    with pytest.raises(ValueError, match="class not found"):
        classglue.generate_class(crepo, "com.example.Missing", tmp_path)
