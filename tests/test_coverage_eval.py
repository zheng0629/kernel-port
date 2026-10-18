"""v0.7 workstream C: coverage-guided fuzz default-on evaluation.

- Pins the opt-in decision: the coverage round's fixed javac+JVM cost makes
  it >2x a cheap ``verify``, so ``--coverage`` stays opt-in.
- Graceful degradation: coverage-round failures warn and fall back to plain
  verify results instead of crashing the command.
- ``CoveredSession`` (compile once, replay many) matches ``run_covered``.
- ``minimize`` gains a wall-clock budget on top of the call budget.

JVM-backed tests need a JDK on PATH (as in test_coverage.py); the build
helper additionally needs g++ (as in test_e2e.py).
"""

import shutil
import time

import pytest

from kernel_port import cli
from kernel_port import coverage as C
from kernel_port import harness as H
from kernel_port import java_ast as J
from kernel_port import jni

from .conftest import FIXTURE_REPO, TRANSLATIONS, fill_translation
from .test_v06_integration import run_cli

RARE_SRC = """
public class Rare {
    public static int f(int x) {
        int r = 0;
        if (x == 1234567) { r += 100; }
        if (x > 5000) { r += 10; } else { r += 1; }
        switch (x) {
            case 3: r += 1000; break;
            case 5: r += 2000; break;
            default: r += 3;
        }
        return r;
    }
}
"""


def _build_dot(tmp_path):
    repo = J.scan_repo(FIXTURE_REPO)
    m = repo.find_method("com.example.Kernels.dot")
    assert m is not None
    glue_dir = tmp_path / "glue"
    jni.generate(m, glue_dir)
    fill_translation(glue_dir, "com.example.Kernels.dot",
                     TRANSLATIONS["com.example.Kernels.dot"])
    work = tmp_path / "work"
    return H.build_env(FIXTURE_REPO, m, glue_dir, work)


def _src_dir(tmp_path, files, tag="src"):
    d = tmp_path / tag
    d.mkdir(parents=True, exist_ok=True)
    for rel, src in files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src, encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# decision pin: --coverage stays opt-in because the round costs >2x verify
# ---------------------------------------------------------------------------


def test_coverage_round_cost_exceeds_2x_on_cheap_kernel(tmp_path):
    """Pins the v0.7 workstream-C decision: --coverage stays opt-in.

    The coverage round replays every case through a freshly compiled
    instrumented JVM; its fixed javac + JVM-startup cost (~2-4s) alone
    exceeds 2x a cheap verify (dot: ~0.1-0.5s). If a future speedup ever
    brings the round under 2x, this test fails on purpose -- re-evaluate
    making --coverage the default then.
    """
    env = _build_dot(tmp_path)
    t0 = time.perf_counter()
    report = H.run_verify(env, cases=20, seed=42)
    plain_s = time.perf_counter() - t0
    assert report.all_passed

    t0 = time.perf_counter()
    cli._run_coverage_round(
        str(FIXTURE_REPO), env, report, cases=20, seed=42,
        tol_float=1e-6, tol_double=1e-12,
    )
    round_s = time.perf_counter() - t0
    assert report.coverage_pct is not None  # the round ran
    assert report.branches_total > 0
    assert round_s > 2 * plain_s, (
        f"coverage round {round_s:.2f}s not > 2x plain verify {plain_s:.2f}s: "
        "re-evaluate making --coverage the default"
    )


def test_verify_coverage_defaults_off(tmp_path):
    """CLI-level pin: --coverage is opt-in, not the default."""
    glue_dir = tmp_path / "g"
    r = run_cli(
        "glue", str(FIXTURE_REPO), "--method", "com.example.Kernels.dot",
        "--out", str(glue_dir), "--no-fallback",
    )
    assert r.returncode == 0, r.stderr
    import re

    cpp = glue_dir / "kernels_jni.cpp"
    t = cpp.read_text()
    body = (
        "// ==== BEGIN TRANSLATED BODY ====\n"
        "    double s = 0.0;\n"
        "    for (int i = 0; i < p_a_len; i++) s += p_a[i] * p_b[i];\n"
        "    return s;\n"
        "    // ==== END TRANSLATED BODY ===="
    )
    t = re.sub(
        r"// ==== BEGIN TRANSLATED BODY ====.*?// ==== END TRANSLATED BODY ====",
        body, t, flags=re.S,
    )
    cpp.write_text(t)
    r = run_cli(
        "verify", str(FIXTURE_REPO), "--method", "com.example.Kernels.dot",
        "--glue-dir", str(glue_dir), "--cases", "10",
    )
    assert r.returncode == 0, r.stderr + r.stdout
    assert "branch coverage" not in r.stdout  # no coverage round ran


# ---------------------------------------------------------------------------
# graceful degradation: coverage failures never crash verify
# ---------------------------------------------------------------------------


def test_coverage_round_degrades_gracefully_on_instrument_failure(
    tmp_path, monkeypatch, capsys
):
    env = _build_dot(tmp_path)
    report = H.run_verify(env, cases=10, seed=42)
    assert report.all_passed

    def boom(sources, work_dir=None):
        raise RuntimeError("synthetic instrumentation failure")

    monkeypatch.setattr(C, "instrument", boom)
    # Must not raise: falls back to plain verify results with a warning.
    cli._run_coverage_round(
        str(FIXTURE_REPO), env, report, cases=10, seed=42,
        tol_float=1e-6, tol_double=1e-12,
    )
    assert report.coverage_pct is None
    assert report.guided == []
    assert report.branches_total == 0
    out = capsys.readouterr().out
    assert "coverage-guided round skipped" in out
    assert "synthetic instrumentation failure" in out


def test_coverage_round_survives_uncompilable_source(tmp_path, capsys):
    """A repo file that breaks the instrumented compile degrades gracefully.

    Real-world trigger: odd syntax the instrumenter copies verbatim but
    javac rejects. The command must warn, not crash.
    """
    repo2 = tmp_path / "repo2"
    shutil.copytree(FIXTURE_REPO, repo2)
    (repo2 / "src" / "com" / "example" / "Broken.java").write_text(
        "public class Broken { this is not java {{{", encoding="utf-8"
    )
    env = _build_dot(tmp_path)
    report = H.run_verify(env, cases=10, seed=42)
    assert report.all_passed

    cli._run_coverage_round(
        str(repo2), env, report, cases=10, seed=42,
        tol_float=1e-6, tol_double=1e-12,
    )
    assert report.coverage_pct is None
    assert report.guided == []
    assert "coverage-guided round skipped" in capsys.readouterr().out


def test_instrument_never_raises_on_garbage():
    # tree-sitter is error-tolerant; instrumentation must be total over
    # inputs (compile may still fail later -- that degrades gracefully).
    ic = C.instrument({"K.java": "public class K { {{{ not java \x00\x01"})
    assert isinstance(ic, C.InstrumentedCopy)
    ic2 = C.instrument({"K.java": ""})
    assert ic2.branches == []


# ---------------------------------------------------------------------------
# CoveredSession: compile once, replay many == repeated run_covered
# ---------------------------------------------------------------------------


def test_covered_session_matches_repeated_run_covered(tmp_path):
    src_dir = _src_dir(tmp_path, {"Rare.java": RARE_SRC})
    m = J.scan_repo(src_dir).find_method("Rare.f")
    assert m is not None
    inst = C.instrument({"Rare.java": RARE_SRC})

    import random

    rng = random.Random(21)
    batch1 = [H.gen_case(m, rng) for _ in range(15)]
    batch2 = [H.gen_case(m, rng) for _ in range(15)]

    with C.CoveredSession(m, inst) as sess:
        sess.replay(batch1)
        cov1 = sess.coverage()
        sess.replay(batch2)
        cov2 = sess.coverage()

    ref1 = C.run_covered(m, batch1, C.instrument({"Rare.java": RARE_SRC}))
    both = C.run_covered(m, batch1 + batch2, C.instrument({"Rare.java": RARE_SRC}))

    assert cov1.per_branch_hits == ref1.per_branch_hits
    # Counters accumulate across replays: the second dump is the union.
    for b in inst.branches:
        assert cov2.hits_for(b.global_id) == both.hits_for(b.global_id)


def test_covered_session_is_reusable_context_manager(tmp_path):
    src_dir = _src_dir(tmp_path, {"Rare.java": RARE_SRC})
    m = J.scan_repo(src_dir).find_method("Rare.f")
    inst = C.instrument({"Rare.java": RARE_SRC})
    sess = C.CoveredSession(m, inst)
    with sess:
        sess.replay([[1]])
        assert sess.coverage().branches_total == len(inst.branches)
    sess.close()  # idempotent


# ---------------------------------------------------------------------------
# minimize: wall-clock budget on top of the call budget
# ---------------------------------------------------------------------------


def test_minimize_wall_clock_budget_caps_runaway():
    # Each predicate call sleeps 20ms and a 400-element list gives ddmin
    # plenty of work; without a wall-clock budget this would take minutes.
    def fails(case):
        time.sleep(0.02)
        return len(case[0]) >= 150

    t0 = time.perf_counter()
    result = C.minimize([list(range(400))], fails,
                        max_calls=10**9, max_seconds=0.5)
    elapsed = time.perf_counter() - t0
    assert elapsed < 8.0, f"ddmin ran away: {elapsed:.1f}s"
    assert fails(result)  # best-so-far still reproduces the failure


def test_minimize_rejects_nonpositive_wall_clock_budget():
    with pytest.raises(ValueError):
        C.minimize([1], lambda case: True, max_seconds=0)
    with pytest.raises(ValueError):
        C.minimize([1], lambda case: True, max_seconds=-1.0)


def test_minimize_call_budget_still_honored():
    calls = {"n": 0}

    def fails(case):
        calls["n"] += 1
        return len(case[0]) > 0  # never minimizable to []

    result = C.minimize([list(range(50))], fails, max_calls=30,
                        max_seconds=120.0)
    assert calls["n"] <= 31
    assert fails(result)
