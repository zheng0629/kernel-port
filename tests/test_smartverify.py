"""Tests for C4 smart verification & benchmarking.

Covers: boundary-value generation, boundary failures on a planted bug,
failure-JSON persistence, break-even math, JNI overhead accounting,
--rigorous fallback mode, profile parsing/reranking, counterexample
rendering, and IQR outlier trimming.

The e2e tests need a JDK and g++ on PATH (same as test_e2e.py).
"""

import json
import math
import statistics

import pytest

from kernel_port import counterex as C
from kernel_port import harness as H
from kernel_port import jni
from kernel_port import profile as P

from .conftest import FIXTURE_REPO, TRANSLATIONS, fill_translation

E2E = pytest.mark.e2e  # needs a JDK and g++ on PATH


def _method(repo, qualname):
    m = repo.find_method(qualname)
    assert m is not None
    return m


def _build(repo, qualname: str, tmp_path, translation_body: str):
    """Glue + fill translation + build_env (mirrors tests/test_e2e.py)."""
    m = _method(repo, qualname)
    glue_dir = tmp_path / "glue"
    jni.generate(m, glue_dir)
    fill_translation(glue_dir, qualname, translation_body)
    work = tmp_path / "work"
    return H.build_env(FIXTURE_REPO, m, glue_dir, work)


# ---------------------------------------------------------------------------
# gen_boundary_cases
# ---------------------------------------------------------------------------


class TestGenBoundaryCases:
    def test_empty_array_present_for_array_params(self, repo):
        m = _method(repo, "com.example.Kernels.dot")
        cases = H.gen_boundary_cases(m)
        assert any(c[0] == [] for c in cases)
        assert any(c[1] == [] for c in cases)

    def test_nan_and_infs_present_for_double_params(self, repo):
        m = _method(repo, "com.example.Kernels.square")
        cases = H.gen_boundary_cases(m)
        vals = [c[0] for c in cases]
        assert any(isinstance(v, float) and math.isnan(v) for v in vals)
        assert any(v == float("inf") for v in vals)
        assert any(v == float("-inf") for v in vals)
        assert 0.0 in vals and 1.0 in vals and -1.0 in vals

    def test_int_min_max_present_for_int_params(self, repo):
        m = _method(repo, "com.example.Kernels.fib")
        cases = H.gen_boundary_cases(m)
        vals = [c[0] for c in cases]
        assert -(2**31) in vals
        assert 2**31 - 1 in vals
        assert 0 in vals and 1 in vals and -1 in vals

    def test_deterministic_and_capped(self, repo):
        m = _method(repo, "com.example.Kernels.matMul")
        a = H.gen_boundary_cases(m)
        b = H.gen_boundary_cases(m)
        assert a == b
        assert len(a) <= 24
        # matMul: 1 base + 3 array params x 3 edges + int param x 4 edges
        assert len(a) == 1 + 3 * 3 + 4

    def test_base_case_is_all_neutral(self, repo):
        m = _method(repo, "com.example.Kernels.dot")
        assert H.gen_boundary_cases(m)[0] == [[], []]

    def test_json_safe_nan_inf(self):
        assert H._json_safe(float("nan")) == "nan"
        assert H._json_safe(float("inf")) == "inf"
        assert H._json_safe(float("-inf")) == "-inf"
        assert H._json_safe([1.0, float("nan")]) == [1.0, "nan"]
        # result must be valid strict JSON
        json.dumps({"args": H._json_safe([float("inf"), [float("nan")]])})


def test_run_one_case_marks_java_no_answer(repo):
    """A Java-side timeout is flagged inconclusive (not a translation bug)."""

    class TimeoutSession:
        def ask(self, line, timeout):
            raise TimeoutError("harness did not answer")

    m = _method(repo, "com.example.Kernels.dot")
    case, skipped, java_no_answer = H._run_one_case(
        TimeoutSession(), TimeoutSession(), m, 0, [[], []], 1.0
    )
    assert not skipped
    assert java_no_answer
    assert case is not None and not case.ok
    assert "java harness error" in case.detail


# ---------------------------------------------------------------------------
# planted bug: wrong translation mishandles empty arrays
# ---------------------------------------------------------------------------

_DOT_EMPTY_BUG = """    if (p_a_len == 0) return 0.5;
    double s = 0.0;
    for (int i = 0; i < p_a_len; i++) s += p_a[i] * p_b[i];
    return s;"""


@E2E
def test_verify_boundary_catches_empty_array_bug(repo, tmp_path):
    env = _build(repo, "com.example.Kernels.dot", tmp_path, _DOT_EMPTY_BUG)
    report = H.run_verify(env, cases=5, seed=1)
    assert report.boundary_failed > 0
    bad = [c for c in report.boundary if not c.ok]
    assert any(c.args == [[], []] for c in bad)
    # random-case accounting is untouched by the boundary sweep
    assert len(report.cases) == 5
    assert report.passed + report.failed == 5


@E2E
def test_verify_correct_translation_boundary_passes(repo, tmp_path):
    env = _build(
        repo, "com.example.Kernels.dot", tmp_path,
        TRANSLATIONS["com.example.Kernels.dot"],
    )
    report = H.run_verify(env, cases=5, seed=1)
    assert report.all_passed
    assert report.boundary_failed == 0
    assert report.boundary_passed == len(report.boundary) > 0


# ---------------------------------------------------------------------------
# failure persistence
# ---------------------------------------------------------------------------


@E2E
def test_verify_writes_failures_json(repo, tmp_path):
    env = _build(
        repo, "com.example.Kernels.dot", tmp_path,
        TRANSLATIONS["com.example.Kernels.dot_WRONG"],
    )
    report = H.run_verify(env, cases=5, seed=1)
    assert report.failed > 0
    path = tmp_path / "work" / ".kp-work" / "com.example.Kernels.dot.failures.json"
    assert path.exists(), "failures JSON was not written"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) > 0
    for entry in data:
        assert {"args", "java_result", "jni_result", "detail", "boundary"} <= set(entry)
        assert isinstance(entry["boundary"], bool)
    # round-trips through the counterex loader
    assert C.load_failures(path) == data


def test_load_failures_missing_file(tmp_path):
    assert C.load_failures(tmp_path / "nope.json") == []


# ---------------------------------------------------------------------------
# break-even math + overhead accounting (no JVM needed)
# ---------------------------------------------------------------------------


class TestBenchMath:
    def _report(self, **kw):
        base = dict(method="m", iters=100, warmup=0,
                    java_nanos=[1000] * 3, jni_nanos=[500] * 3)
        base.update(kw)
        return H.BenchReport(**base)

    def test_break_even_picks_first_size_over_threshold(self):
        r = self._report(per_size=[
            {"size": 64, "java_ns": 10.0, "jni_ns": 20.0, "speedup": 0.5},
            {"size": 256, "java_ns": 40.0, "jni_ns": 36.0, "speedup": 40.0 / 36.0},
            {"size": 1024, "java_ns": 160.0, "jni_ns": 100.0, "speedup": 1.6},
        ])
        assert r.break_even_size == 1024

    def test_break_even_threshold_is_strict(self):
        # exactly 1.2x does NOT count (heuristic bar is *more than* 20%)
        r = self._report(per_size=[
            {"size": 8, "java_ns": 12.0, "jni_ns": 10.0, "speedup": 1.2},
        ])
        assert r.break_even_size is None

    def test_break_even_none_when_never_faster(self):
        r = self._report(per_size=[
            {"size": 8, "java_ns": 10.0, "jni_ns": 20.0, "speedup": 0.5},
        ])
        assert r.break_even_size is None

    def test_break_even_none_without_per_size(self):
        assert self._report().break_even_size is None

    def test_compute_ns_subtracts_overhead(self):
        r = self._report(jni_overhead_ns=2.0)
        assert r.compute_ns == pytest.approx(5.0 - 2.0)

    def test_compute_ns_floors_at_zero(self):
        r = self._report(jni_overhead_ns=1000.0)
        assert r.compute_ns == 0.0

    def test_compute_ns_without_overhead_is_plain_median(self):
        r = self._report()
        assert r.compute_ns == pytest.approx(r.jni_per_call_ns)

    def test_old_construction_still_works(self):
        r = H.BenchReport(method="m", iters=10, warmup=1,
                          java_nanos=[100], jni_nanos=[50])
        assert r.jni_overhead_ns is None
        assert r.per_size == []
        assert r.rigorous_note == ""
        assert r.ci95_half_width_ns is None


# ---------------------------------------------------------------------------
# rigorous mode + overhead, end to end (small)
# ---------------------------------------------------------------------------


@E2E
def test_e2e_bench_rigorous_overhead_sizes(repo, tmp_path):
    env = _build(
        repo, "com.example.Kernels.dot", tmp_path,
        TRANSLATIONS["com.example.Kernels.dot"],
    )
    report = H.run_bench(
        env, iters=200, warmup=50, repeats=2,
        sizes=[32, 128], rigorous=True, overhead=True,
    )
    # overhead: empty-JNI-call cost measured via the second tiny .so
    assert report.jni_overhead_ns is not None
    assert report.jni_overhead_ns >= 0
    assert report.compute_ns <= report.jni_per_call_ns
    assert report.compute_ns >= 0
    # rigorous fallback: honest wall-clock, never faked framework numbers
    assert "IQR" in report.rigorous_note
    assert "JMH" in report.rigorous_note
    assert report.ci95_half_width_ns is not None
    assert report.ci95_half_width_ns >= 0
    # per-size sweep
    assert [e["size"] for e in report.per_size] == [32, 128]
    for e in report.per_size:
        assert {"size", "java_ns", "jni_ns", "speedup"} <= set(e)
        assert e["java_ns"] > 0 and e["jni_ns"] > 0 and e["speedup"] > 0
    assert report.break_even_size in (None, 32, 128)


# ---------------------------------------------------------------------------
# IQR trimming (synthetic)
# ---------------------------------------------------------------------------


def test_iqr_trim_drops_outlier():
    samples = [100, 102, 98, 101, 99, 5000]
    trimmed = H._iqr_trim(samples)
    assert 5000 not in trimmed
    assert abs(statistics.median(trimmed) - 100) < 3


def test_iqr_trim_keeps_small_samples():
    assert H._iqr_trim([1, 2, 3]) == [1, 2, 3]
    assert H._iqr_trim([10, 10, 10, 10]) == [10, 10, 10, 10]


# ---------------------------------------------------------------------------
# profile.py
# ---------------------------------------------------------------------------


class TestProfile:
    def _collapsed(self, tmp_path):
        p = tmp_path / "out.collapsed"
        p.write_text(
            "# async-profiler collapsed stacks\n"
            "java/lang/Thread.run;com/example/Kernels.dot 100\n"
            "java/lang/Thread.run;com/example/Kernels.norm 50\n"
            "java/lang/Thread.run;com/example/Kernels.dot;com/example/Kernels.square 30\n"
            "JavaMain;libjvm.so 7\n"  # no Java frame -> skipped
            "bogus line without count\n",  # malformed -> skipped
            encoding="utf-8",
        )
        return p

    def test_parse_collapsed_aggregates_by_class_method(self, tmp_path):
        cpu = P.parse_collapsed(self._collapsed(tmp_path))
        # deepest frame wins: line 3 counts for square, not dot
        assert cpu == {
            "Kernels.dot": 100.0,
            "Kernels.norm": 50.0,
            "Kernels.square": 30.0,
        }

    def test_parse_collapsed_interval_header_converts_to_seconds(self, tmp_path):
        p = tmp_path / "h.collapsed"
        p.write_text(
            "# interval: 10000000 ns\n"
            "com/example/Kernels.dot 4\n",
            encoding="utf-8",
        )
        cpu = P.parse_collapsed(p)
        assert cpu["Kernels.dot"] == pytest.approx(0.04)

    def test_rerank_orders_pure_by_cpu_desc(self, tmp_path):
        from kernel_port import miner

        def cand(qn, pure):
            return miner.Candidate(
                qualname=qn, file="Kernels.java", lineno=1, signature="s",
                pure=pure, reasons=[], calls=1, intensity=1.0, score=1.0,
            )

        cands = [
            cand("com.example.Kernels.square", True),
            cand("com.example.Kernels.noisy", False),  # impure, huge CPU
            cand("com.example.Kernels.norm", True),
            cand("com.example.Kernels.dot", True),
            cand("com.example.Kernels.unseen", True),  # not in profile
        ]
        cpu = {"Kernels.dot": 100.0, "Kernels.norm": 50.0,
               "Kernels.square": 30.0, "Kernels.noisy": 999.0}
        ranked = P.rerank(cands, cpu)
        assert [c.qualname for c in ranked] == [
            "com.example.Kernels.dot",
            "com.example.Kernels.norm",
            "com.example.Kernels.square",
            "com.example.Kernels.unseen",
            "com.example.Kernels.noisy",  # impure sorts after pure
        ]
        by_name = {c.qualname: c for c in ranked}
        assert by_name["com.example.Kernels.dot"].profile_cpu_s == 100.0
        assert by_name["com.example.Kernels.dot"].profile_share == pytest.approx(
            100.0 / 1179.0
        )
        assert by_name["com.example.Kernels.unseen"].profile_cpu_s == 0.0
        assert by_name["com.example.Kernels.unseen"].profile_share == 0.0


# ---------------------------------------------------------------------------
# counterex.py
# ---------------------------------------------------------------------------


class TestCounterex:
    def test_format_contains_failing_values(self):
        failures = [
            {"args": [[], []], "java_result": "d:0.0", "jni_result": "d:0.5",
             "detail": "0.0 != 0.5", "boundary": True},
            {"args": [[float("nan")], [1.0]], "java_result": "d:NaN",
             "jni_result": "d:1.0", "detail": "NaN mismatch", "boundary": True},
        ]
        out = C.format_counterexamples(failures)
        assert "## Counterexamples — the previous translation failed these" in out
        assert "0.0" in out and "0.5" in out
        assert "NaN" in out  # readable NaN rendering
        assert "expected (Java)" in out and "got (JNI)" in out
        assert "Case 1" in out  # per-case fix guidance

    def test_format_renders_repr_string_specials(self):
        # as loaded from JSON: NaN/Infinity arrive as repr strings
        failures = [{"args": ["nan", "inf", "-inf"], "java_result": "d:Infinity",
                     "jni_result": "d:-Infinity", "detail": "x", "boundary": False}]
        out = C.format_counterexamples(failures)
        assert "NaN" in out and "+Inf" in out and "-Inf" in out

    def test_format_pipeline_call_shape(self):
        # sibling pipeline.py calls format_counterexamples(qualname, cases)
        class FakeCase:
            def __init__(self):
                self.args = [[1.0], [2.0]]
                self.detail = "boom"
                self.java_result = "d:3.0"
                self.jni_result = "d:4.0"

        out = C.format_counterexamples("com.example.Kernels.dot", [FakeCase()])
        assert "## Counterexamples — the previous translation failed these" in out
        assert "com.example.Kernels.dot" in out
        assert "3.0" in out and "4.0" in out

    def test_format_empty(self):
        out = C.format_counterexamples([])
        assert "## Counterexamples" in out
