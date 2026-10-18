"""Unit tests for the harness value model (no JVM needed)."""

import random

import pytest

from kernel_port import harness as H
from kernel_port import java_ast as J


def _method(repo, qualname):
    m = repo.find_method(qualname)
    assert m is not None
    return m


def test_gen_case_deterministic(repo):
    m = _method(repo, "com.example.Kernels.dot")
    a = H.gen_case(m, random.Random(7))
    b = H.gen_case(m, random.Random(7))
    assert a == b
    c = H.gen_case(m, random.Random(8))
    assert a != c


def test_gen_case_dot_arrays_match(repo):
    m = _method(repo, "com.example.Kernels.dot")
    for seed in range(20):
        args = H.gen_case(m, random.Random(seed))
        assert len(args[0]) == len(args[1])


def test_gen_case_matmul_n_matches(repo):
    m = _method(repo, "com.example.Kernels.matMul")
    for seed in range(50):
        a, b, c, n = H.gen_case(m, random.Random(seed))
        assert len(a) == len(b) == len(c)
        assert n * n == len(a) or n == len(a)


def test_encode_decode_roundtrip():
    assert H.encode_arg("int", 42) == "i:42"
    assert H.encode_arg("double", 1.5) == "d:1.5"
    assert H.encode_arg("boolean", True) == "z:1"
    assert H.encode_arg("int[]", [1, 2]) == "ai:1,2"
    assert H.encode_arg("double[]", []) == "ad:"
    assert H._parse_result_token("i:42") == ("int", 42)
    assert H._parse_result_token("ad:") == ("ad", [])


def test_results_equal_scalars(repo):
    m = _method(repo, "com.example.Kernels.fib")
    ok, _ = H.results_equal(m, "l:55", "l:55")
    assert ok
    ok, detail = H.results_equal(m, "l:55", "l:56")
    assert not ok and detail


def test_results_equal_float_tolerance(repo):
    m = _method(repo, "com.example.Kernels.dot")
    ok, _ = H.results_equal(m, "d:0.30000000000000004", "d:0.3")
    assert ok
    ok, _ = H.results_equal(m, "d:1.0", "d:2.0")
    assert not ok


def test_results_equal_nan(repo):
    m = _method(repo, "com.example.Kernels.dot")
    ok, _ = H.results_equal(m, "d:nan", "d:nan")
    assert ok  # NaN == NaN for differential purposes


def test_results_equal_arrays(repo):
    m = _method(repo, "com.example.Kernels.sieve")
    ok, _ = H.results_equal(m, "ai:2,3,5", "ai:2,3,5")
    assert ok
    ok, _ = H.results_equal(m, "ai:2,3,5", "ai:2,3")
    assert not ok


def test_results_equal_void(repo):
    m = _method(repo, "com.example.Kernels.matMul")
    ok, _ = H.results_equal(m, "v", "v")
    assert ok


def test_results_equal_void_array_echo(repo):
    m = _method(repo, "com.example.Kernels.matMul")
    ok, _ = H.results_equal(m, "v|ad:1.0,2.0|ad:3.0|ai:1", "v|ad:1.0,2.0|ad:3.0|ai:1")
    assert ok
    # per-type tolerance applies to the echoed arrays (v0.4: double -> 1e-12)
    ok, _ = H.results_equal(m, "v|ad:1.0", "v|ad:1.0000000000001")
    assert ok
    ok, _ = H.results_equal(m, "v|ad:1.0", "v|ad:1.0000000001")
    assert not ok  # 1e-10 exceeds the double threshold
    ok, detail = H.results_equal(m, "v|ad:1.0,2.0", "v|ad:1.0,2.001")
    assert not ok and "mismatch" in detail
    # shape mismatches fail loudly
    ok, _ = H.results_equal(m, "v|ad:1.0", "v")
    assert not ok
    ok, _ = H.results_equal(m, "v|ad:1.0", "v|ad:1.0|ad:2.0")
    assert not ok
    ok, _ = H.results_equal(m, "v|ai:1,2", "v|ai:1,3")
    assert not ok


def test_harness_render_compiles_standalone(tmp_path, repo):
    """The generated harness must be syntactically valid Java."""
    from kernel_port.harness import compile_java, render_harness

    from .conftest import FIXTURE_REPO

    m = _method(repo, "com.example.Kernels.dot")
    hj = tmp_path / "HarnessJava.java"
    hj.write_text(render_harness(m, "java", "HarnessJava"))
    out = tmp_path / "classes"
    compile_java(sorted(FIXTURE_REPO.rglob("*.java")) + [hj], out)
    assert (out / "HarnessJava.class").exists()


# ---------------------------------------------------------------------------
# v0.10: verdict noise band + JIT settling + tolerant line-protocol parsing
# ---------------------------------------------------------------------------


def _report(java_per_call_ns: float, jni_per_call_ns: float) -> H.BenchReport:
    """BenchReport with a known median speedup.

    java_per_call_ns / jni_per_call_ns is the median speedup; values are
    chosen so the interesting ratios (1.2, 1/1.2) land exactly on the float
    band edges.
    """
    iters = 1000
    return H.BenchReport(
        method="com.example.Kernels.dot",
        iters=iters,
        warmup=200,
        java_nanos=[int(java_per_call_ns * iters)] * 5,
        jni_nanos=[int(jni_per_call_ns * iters)] * 5,
    )


def test_verdict_inside_noise_band_is_inconclusive():
    for j, n in [(1000, 952), (952, 1000), (1000, 1000), (1100, 1000), (1000, 1100)]:
        r = _report(j, n)
        assert r.verdict == "inconclusive", (j, n, r.speedup)


def test_verdict_band_edges_are_inclusive():
    # Exactly 1.2x (and 1/1.2x) is still too close to call: the edges are
    # inclusive by design, so a borderline reading can never flip the
    # verdict.
    assert _report(1.2, 1.0).verdict == "inconclusive"
    assert _report(1.0, 1.2).verdict == "inconclusive"
    assert H.verdict_for_speedup(1.2) == "inconclusive"
    assert H.verdict_for_speedup(1.0 / 1.2) == "inconclusive"


def test_verdict_outside_band_faster():
    r = _report(1500, 1000)
    assert r.verdict == "faster"
    assert r.speedup == pytest.approx(1.5)
    assert H.verdict_for_speedup(1.21) == "faster"
    assert H.verdict_for_speedup(2.0) == "faster"


def test_verdict_outside_band_slower():
    r = _report(1000, 2000)
    assert r.verdict == "slower"
    assert r.speedup == pytest.approx(0.5)
    assert H.verdict_for_speedup(1.0 / 1.21) == "slower"
    assert H.verdict_for_speedup(0.5) == "slower"


def test_verdict_text_never_claims_win_inside_band():
    for j, n in [(1000, 952), (952, 1000), (1000, 1000), (1.2, 1.0), (1.0, 1.2)]:
        text = _report(j, n).verdict_text()
        low = text.lower()
        assert "inconclusive" in low, text
        assert "too close to call" in low, text
        assert "faster" not in low, text
        assert "slower" not in low, text


def test_verdict_text_outside_band_names_winner():
    assert "faster" in _report(1500, 1000).verdict_text().lower()
    assert "slower" in _report(1000, 2000).verdict_text().lower()
    # the label parameter flows through to the rendered text
    assert "Rust" in _report(1500, 1000).verdict_text("Rust (JNI)")


def test_verdict_text_inconclusive_names_backend():
    # Regression (v0.12): the inconclusive branch used to drop the backend
    # label, which made test_full_run_ffm_success flaky -- its assertion
    # on "FFM native" only held when the measured speedup fell outside
    # the noise band. All three branches must name the backend.
    text = _report(1000, 1176).verdict_text("FFM native")
    assert "inconclusive" in text.lower()
    assert "FFM native" in text


class _StubSession:
    """LineSession-shaped stub: scripted responses, in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.asked = []

    def ask(self, line, timeout):
        self.asked.append(line)
        if not self._responses:
            raise TimeoutError("script exhausted")
        return self._responses.pop(0)


def test_jit_settle_stops_when_stable():
    # First block slow (uncompiled), next two agree -> settled at block 3.
    s = _StubSession(["OK 5000", "OK 1000", "OK 1000", "OK 1000"])
    blocks = H._jit_settle(s, "BENCH 1000", 30.0)
    assert blocks == 3
    assert s.asked == ["BENCH 1000"] * 3


def test_jit_settle_capped_when_never_stable():
    # Wildly drifting workload: must not loop forever, returns the cap.
    s = _StubSession(["OK 1000", "OK 2000"] * 10)
    blocks = H._jit_settle(s, "BENCH 1000", 30.0, max_blocks=5)
    assert blocks == 5


def test_jit_settle_malformed_protocol_is_error():
    s = _StubSession(["OK not-a-number"])
    with pytest.raises(RuntimeError, match="protocol error"):
        H._jit_settle(s, "BENCH 1000", 30.0)


def test_jit_settle_harness_err_is_error():
    s = _StubSession(["ERR ArrayIndexOutOfBoundsException: 5"])
    with pytest.raises(RuntimeError, match="harness error"):
        H._jit_settle(s, "BENCH 1000", 30.0)


def _session_with_lines(lines):
    """A real LineSession whose stdout queue is pre-filled (no subprocess)."""
    import io
    import queue as _queue
    import types

    s = H.LineSession.__new__(H.LineSession)
    s._q = _queue.Queue()
    for ln in lines:
        s._q.put(ln)
    s.proc = types.SimpleNamespace(stdin=io.StringIO())
    return s


def test_ask_skips_jvm_startup_warnings():
    s = _session_with_lines(
        [
            "Warning: failed to rename stale hsperfdata file",
            "Warning: some other launcher chatter",
            "OK d:1.5",
        ]
    )
    assert s.ask("CALL d:1.5", 5.0) == "OK d:1.5"


def test_ask_logs_skipped_warnings_to_stderr(capsys):
    s = _session_with_lines(["Warning: failed to rename stale hsperfdata file", "OK v"])
    assert s.ask("CALL ", 5.0) == "OK v"
    err = capsys.readouterr().err
    assert "hsperfdata" in err
    assert "ignoring non-protocol harness output" in err


def test_ask_returns_err_lines_untouched():
    s = _session_with_lines(
        ["Warning: failed to rename stale hsperfdata file", "ERR boom: 5"]
    )
    assert s.ask("CALL i:5", 5.0) == "ERR boom: 5"


def test_ask_missing_protocol_line_still_times_out():
    s = _session_with_lines(["Warning: and then nothing"])
    with pytest.raises(TimeoutError, match="did not answer"):
        s.ask("CALL i:5", 0.05)


def test_run_one_case_tolerates_warnings(repo, capsys):
    m = _method(repo, "com.example.Kernels.fib")
    s_java = _session_with_lines(
        ["Warning: failed to rename stale hsperfdata file", "OK l:5"]
    )
    s_jni = _session_with_lines(["OK l:5"])
    case, skipped, no_answer = H._run_one_case(s_java, s_jni, m, 0, [5], 5.0)
    assert not skipped and not no_answer
    assert case is not None and case.ok, case.detail if case else None
    assert case.java_result == "l:5"
    assert case.jni_result == "l:5"
    assert "hsperfdata" in capsys.readouterr().err


def test_bench_samples_skips_warnings_and_settles(repo, capsys):
    """End-to-end through _bench_samples with warning-polluted stdout.

    The probe answer arrives behind a JVM warning, the settle phase sees
    an uncompiled first block then a stable one, and the measured samples
    are the clean protocol values -- never corrupted by the warnings.
    """
    m = _method(repo, "com.example.Kernels.fib")
    s_java = _session_with_lines(
        [
            "Warning: failed to rename stale hsperfdata file",
            "OK l:5",  # probe
            "OK 8000",  # settle block 1 (slow: uncompiled)
            "OK 8000",  # settle block 2 (stable -> stop)
            "OK 200",  # rep 1 warmup
            "OK 2000000",  # rep 1 measured
            "OK 200",  # rep 2 warmup
            "OK 2000000",  # rep 2 measured
        ]
    )
    s_jni = _session_with_lines(
        [
            "OK 6000",  # settle block 1
            "OK 6000",  # settle block 2 (stable -> stop)
            "OK 100",  # rep 1 warmup
            "OK 1000000",  # rep 1 measured
            "OK 100",  # rep 2 warmup
            "OK 1000000",  # rep 2 measured
        ]
    )
    j_ns, n_ns = H._bench_samples(
        s_java, s_jni, m, [5], iters=2000, warmup=200, repeats=2, timeout=5.0
    )
    assert j_ns == [2000000, 2000000]
    assert n_ns == [1000000, 1000000]
    assert "hsperfdata" in capsys.readouterr().err


def test_bench_samples_malformed_bench_is_protocol_error(repo):
    """A genuinely malformed protocol line is still a protocol error."""
    m = _method(repo, "com.example.Kernels.fib")
    s_java = _session_with_lines(
        [
            "OK l:5",  # probe
            "OK 8000",  # settle block 1
            "OK 8000",  # settle block 2 (stable)
            "OK 200",  # rep 1 warmup
            "OK not-a-number",  # rep 1 measured: malformed
        ]
    )
    s_jni = _session_with_lines(["OK 6000", "OK 6000"])
    with pytest.raises(RuntimeError, match="protocol error"):
        H._bench_samples(
            s_java, s_jni, m, [5], iters=2000, warmup=200, repeats=2, timeout=5.0
        )
