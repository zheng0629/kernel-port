"""Tests for kernel_port.pipeline.run_pipeline.

(a) awaiting_translation path -- no build, fast.
(b) rejected path on the impure fixture method.
(c) full run with a hand-written correct C++ body (needs JDK + g++, like the
    e2e tests) incl. multi-candidate evaluation and a cache round-trip.
(d) graceful fallback when the companion modules (cache/select/report/
    counterex) cannot be imported.
"""

import json
import sys
from pathlib import Path

import pytest

from kernel_port import pipeline as P

from .conftest import FIXTURE_REPO, TRANSLATIONS

DOT_BODY = TRANSLATIONS["com.example.Kernels.dot"]
DOT_WRONG = TRANSLATIONS["com.example.Kernels.dot_WRONG"]


def _run(tmp_path, repo=None, **kwargs):
    glue_out = tmp_path / "glue"
    work_dir = tmp_path / "work"
    cache_dir = tmp_path / "cache"
    kwargs.setdefault("glue_out", glue_out)
    kwargs.setdefault("work_dir", work_dir)
    kwargs.setdefault("cache_dir", cache_dir)
    return P.run_pipeline(FIXTURE_REPO if repo is None else repo, **kwargs)


# ---------------------------------------------------------------------------
# (a) awaiting_translation
# ---------------------------------------------------------------------------


def test_awaiting_translation(tmp_path):
    rep = _run(tmp_path, method="com.example.Kernels.dot")
    assert rep["status"] == "awaiting_translation"
    assert rep["method"] == "com.example.Kernels.dot"
    assert rep["backend"] == "jni"  # default backend is recorded in the report
    assert rep["purity"] == {"pure": True, "reasons": []}
    assert rep["verify"] is None and rep["bench"] is None
    assert rep["cache_hit"] is False
    cpp = Path(rep["glue_dir"]) / "kernels_jni.cpp"
    assert cpp.exists()
    assert "BEGIN TRANSLATED BODY" in rep["instructions"]
    assert "END TRANSLATED BODY" in rep["instructions"]
    assert str(cpp) in rep["instructions"]
    assert (Path(rep["work_dir"]) / "prompt_material.md").exists()
    # whole report must be JSON-serializable
    json.dumps(rep)


def test_method_autoresolve_picks_top_pure(tmp_path):
    rep = _run(tmp_path, method=None)
    assert rep["status"] == "awaiting_translation"
    assert rep["method"] is not None
    assert rep["purity"]["pure"] is True


# ---------------------------------------------------------------------------
# (b) rejected
# ---------------------------------------------------------------------------


def test_rejected_impure(tmp_path):
    rep = _run(tmp_path, method="com.example.Kernels.noisy")
    assert rep["status"] == "rejected"
    assert rep["method"] == "com.example.Kernels.noisy"
    assert rep["purity"]["pure"] is False
    assert rep["purity"]["reasons"], "rejection must carry reasons"
    assert rep["verify"] is None
    # glue must not have been generated for a rejected kernel
    assert not Path(rep["glue_dir"]).exists()
    json.dumps(rep)


# ---------------------------------------------------------------------------
# (c) full run + candidates + cache round-trip
# ---------------------------------------------------------------------------


def _full(tmp_path, **kwargs):
    kwargs.setdefault("cases", 20)
    kwargs.setdefault("iters", 200)
    kwargs.setdefault("warmup", 50)
    kwargs.setdefault("repeats", 2)
    return _run(tmp_path, **kwargs)


def test_full_run_success(tmp_path):
    rep = _full(tmp_path, method="com.example.Kernels.dot",
                translated_body=DOT_BODY)
    assert rep["status"] == "success", rep.get("instructions")
    assert rep["verify"]["passed"] > 0
    assert rep["verify"]["failed"] == 0
    assert rep["bench"]["java_per_call_ns"] > 0
    assert rep["bench"]["jni_per_call_ns"] > 0
    assert rep["bench"]["speedup"] > 0
    assert rep["counterexamples_path"] is None
    json.dumps(rep)


def test_full_run_verify_failed_writes_counterexamples(tmp_path):
    rep = _full(tmp_path, method="com.example.Kernels.dot",
                translated_body=DOT_WRONG)
    assert rep["status"] == "verify_failed"
    assert rep["verify"]["failed"] > 0
    path = rep["counterexamples_path"]
    assert path and Path(path).exists()
    text = Path(path).read_text(encoding="utf-8")
    assert "com.example.Kernels.dot" in text
    json.dumps(rep)


def test_candidates_evaluated_and_winner_picked(tmp_path):
    cand_dir = tmp_path / "candidates"
    cand_dir.mkdir()
    (cand_dir / "v1.cpp").write_text(DOT_BODY, encoding="utf-8")
    # second candidate: correct but written differently
    (cand_dir / "v2.cpp").write_text(
        "double s = 0.0;\n"
        "for (int i = 0; i < p_a_len; ++i) { s += p_a[i] * p_b[i]; }\n"
        "return s;",
        encoding="utf-8",
    )
    rep = _full(tmp_path, method="com.example.Kernels.dot",
                translated_body=DOT_BODY, candidates_dir=cand_dir)
    assert rep["status"] == "success"
    assert rep["candidates"] is not None and len(rep["candidates"]) == 2
    names = {c["name"] for c in rep["candidates"]}
    assert names == {"v1", "v2"}
    for c in rep["candidates"]:
        assert c["verify_failed"] == 0, c
        assert c["bench_median_ns"] > 0
    assert rep["winner"] in ("v1", "v2")
    assert "WINNER" in (rep["candidates_table"] or "")
    json.dumps(rep)


def test_cache_hit_skips_rebuild(tmp_path):
    first = _full(tmp_path, method="com.example.Kernels.dot",
                  translated_body=DOT_BODY)
    assert first["status"] == "success"
    assert first["fingerprint"]
    # second run: no translation supplied, but the cache has it
    second = _run(
        tmp_path,
        method="com.example.Kernels.dot",
        glue_out=tmp_path / "glue2",
        work_dir=tmp_path / "work2",
        cases=20,
    )
    assert second["status"] == "cache_hit"
    assert second["cache_hit"] is True
    assert second["fingerprint"] == first["fingerprint"]
    assert second["verify"] == first["verify"]
    assert second["bench"] == first["bench"]
    json.dumps(second)


def test_splice_body_missing_markers():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        cpp = Path(d) / "kernels_jni.cpp"
        cpp.write_text("int main() {}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="markers not found"):
            P.splice_body(cpp, "int x;")


def test_splice_body_hoists_includes():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        cpp = Path(d) / "kernels_jni.cpp"
        cpp.write_text(
            "#include <jni.h>\n"
            "#include <vector>\n"
            "\n"
            "static void kernel_f() {\n"
            "    // ==== BEGIN TRANSLATED BODY ====\n"
            "    // TODO(agent)\n"
            "    // ==== END TRANSLATED BODY ====\n"
            "}\n",
            encoding="utf-8",
        )
        P.splice_body(cpp, "#include <cmath>\n#include <cmath>\ndouble x = sqrt(4.0);\n")
        src = cpp.read_text(encoding="utf-8")
        assert src.count("#include <cmath>") == 1  # deduped
        # include hoisted above the function, not inside the body region
        assert src.index("#include <cmath>") < src.index("static void kernel_f()")
        assert "double x = sqrt(4.0);" in src
        body_region = src.split("==== BEGIN TRANSLATED BODY ====")[1].split(
            "==== END TRANSLATED BODY ===="
        )[0]
        assert "#include" not in body_region


def test_html_report_written(tmp_path):
    html_path = tmp_path / "report.html"
    rep = _run(tmp_path, method="com.example.Kernels.dot", html=html_path)
    assert rep["status"] == "awaiting_translation"
    assert rep["html_path"] == str(html_path)
    assert html_path.exists()
    assert "<html" in html_path.read_text(encoding="utf-8").lower()


# ---------------------------------------------------------------------------
# (d) companion-module fallback
# ---------------------------------------------------------------------------


@pytest.fixture()
def blocked_companions(monkeypatch):
    """Simulate the other workstreams' modules not existing yet."""
    for mod in ("kernel_port.cache", "kernel_port.select",
                "kernel_port.report", "kernel_port.counterex"):
        monkeypatch.setitem(sys.modules, mod, None)
    return None


def test_pipeline_completes_without_companions(tmp_path, blocked_companions):
    # full success path with cache/select/report/counterex all blocked
    rep = _full(tmp_path, method="com.example.Kernels.dot",
                translated_body=DOT_BODY,
                html=tmp_path / "report.html")
    assert rep["status"] == "success", rep.get("instructions")
    assert rep["verify"]["passed"] > 0
    assert rep["bench"]["speedup"] > 0
    assert rep["fingerprint"] is None  # caching skipped
    assert rep["html_path"] is None  # html skipped
    assert "not available" in rep["instructions"]


def test_awaiting_translation_without_companions(tmp_path, blocked_companions):
    rep = _run(tmp_path, method="com.example.Kernels.dot")
    assert rep["status"] == "awaiting_translation"
    assert rep["purity"]["pure"] is True


# ---------------------------------------------------------------------------
# (e) ffm backend
# ---------------------------------------------------------------------------

FFM_REPO = Path(__file__).parent / "fixtures" / "ffm_kernels"
# Hand-written C translation for com.example.FfmKernels.dot, standing in for
# the calling agent's LLM step. FFM C ABI: arrays arrive as
# (const T* p_x, int64_t p_x_len).
FFM_DOT_BODY = (
    "    double s = 0.0;\n"
    "    for (int64_t i = 0; i < p_a_len; i++) s += p_a[i] * p_b[i];\n"
    "    return s;"
)


def test_ffm_backend_awaiting_translation(tmp_path):
    rep = _run(tmp_path, method="com.example.Kernels.dot", backend="ffm")
    assert rep["status"] == "awaiting_translation"
    assert rep["backend"] == "ffm"
    glue_dir = Path(rep["glue_dir"])
    assert (glue_dir / "kernels_ffm.c").exists()
    ffm_java = list(glue_dir.glob("*Ffm.java"))
    assert len(ffm_java) == 1
    assert "BEGIN TRANSLATED BODY" in rep["instructions"]
    assert str(glue_dir / "kernels_ffm.c") in rep["instructions"]
    assert "The C translation is still missing" in rep["instructions"]
    json.dumps(rep)


def test_invalid_backend_returns_error(tmp_path):
    rep = _run(tmp_path, method="com.example.Kernels.dot", backend="cuda")
    assert rep["status"] == "error"
    assert rep["backend"] == "cuda"
    assert "'jni'" in rep["instructions"] and "'ffm'" in rep["instructions"]
    json.dumps(rep)


def test_full_run_ffm_success(tmp_path):
    # Full e2e through the FFM backend (needs JDK >= 22 + cc).
    rep = _full(
        tmp_path,
        method="com.example.FfmKernels.dot",
        translated_body=FFM_DOT_BODY,
        backend="ffm",
        # point at the FFM fixture repo instead of the JNI one
        repo=FFM_REPO,
    )
    assert rep["status"] == "success", rep.get("instructions")
    assert rep["backend"] == "ffm"
    assert rep["verify"]["passed"] > 0
    assert rep["verify"]["failed"] == 0
    # bench summary keeps the jni_* key names (see pipeline docstring), but
    # for the ffm backend they describe the FFM path.
    assert rep["bench"]["jni_per_call_ns"] > 0
    assert rep["bench"]["speedup"] > 0
    # v0.10: the verdict is honest about the measured speedup, whatever it is
    # on this machine (in-band -> "inconclusive", never a claimed win).
    assert "FFM native" in rep["instructions"]
    assert "noise band" in rep["instructions"]
    assert rep["bench"]["verdict"] in ("faster", "slower", "inconclusive")
    json.dumps(rep)


def test_mcp_pipeline_tool_passes_backend(monkeypatch):
    from kernel_port import mcp_server as M

    seen = {}

    def fake_run_pipeline(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return {"status": "awaiting_translation"}

    monkeypatch.setattr(M.pipeline_mod, "run_pipeline", fake_run_pipeline)
    out = M.pipeline(repo="somewhere", backend="ffm")
    assert out["status"] == "awaiting_translation"
    assert seen["args"][0] == "somewhere"
    assert seen["kwargs"]["backend"] == "ffm"
    # default stays jni
    M.pipeline(repo="somewhere")
    assert seen["kwargs"]["backend"] == "jni"


def test_pipeline_overload_suffix_resolves(tmp_path):
    # v0.9 dogfood: pipeline passed m.qualname (bare) to extract, re-triggering
    # AmbiguityError for overloaded methods. The original suffixed ref must be
    # used so the chosen overload flows through.
    from pathlib import Path as _P

    overload_repo = str(_P(__file__).parent / "fixtures" / "overload_kernels")
    rep = _run(
        tmp_path,
        repo=overload_repo,
        method="com.example.Overloads.add(int,int)",
    )
    assert rep["status"] == "awaiting_translation", rep.get("instructions")
    assert rep["method"] == "com.example.Overloads.add"
    assert rep["purity"] == {"pure": True, "reasons": []}
    # ...and the bare ambiguous name still errors with candidates listed.
    rep2 = _run(tmp_path, repo=overload_repo, method="com.example.Overloads.add")
    assert rep2["status"] == "error"
    assert "Disambiguate" in rep2["instructions"]
