"""Tests for the Python pipeline's fingerprint cache and multi-candidate selection.

Covers ``run_python_pipeline``'s ``no_cache``/``cache_dir``/``candidates_dir``
wiring (mirroring the Java pipeline's cache/select semantics) plus the
``pipeline --lang python`` CLI plumbing for those flags.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from kernel_port import cache, py_pipeline, select
from kernel_port import py_ast as P

FIXTURES = Path(__file__).parent / "fixtures" / "py_kernels"

DOT_C = """\
double s = 0.0;
for (int64_t i = 0; i < p_a_n; i++) s += p_a[i] * p_b[i];
return s;
"""

DOT_C_WRONG = """\
double s = 0.0;
for (int64_t i = 0; i < p_a_n; i++) s += p_a[i] + p_b[i];
return s;
"""


def _ext(method="cffi_kernels.dot"):
    r = P.scan_repo(FIXTURES)
    fn = r.functions[method]
    return py_pipeline.extract_python(r, fn)


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_stable_under_whitespace_changes():
    ext = _ext()
    fp1 = cache.fingerprint(ext)
    reformatted = dict(ext)
    reformatted["source"] = "\n\n   " + ext["source"].replace("\n", "  \n") + "\n"
    assert cache.fingerprint(reformatted) == fp1


def test_fingerprint_changes_when_body_changes():
    ext = _ext()
    altered = dict(ext)
    altered["source"] = ext["source"].replace("s += x * y", "s += x + y")
    assert cache.fingerprint(altered) != cache.fingerprint(ext)


def test_fingerprint_changes_with_signature():
    ext = _ext()
    altered = dict(ext)
    altered["signature"] = ext["signature"] + " "
    assert cache.fingerprint(altered) != cache.fingerprint(ext)


def test_fingerprint_includes_closure():
    ext = _ext("cffi_kernels.norm")
    assert ext["closure"], "norm should pull in dot as a pure callee"
    fp1 = cache.fingerprint(ext)
    altered = dict(ext)
    closure = [dict(c) for c in ext["closure"]]
    closure[0]["source"] = closure[0]["source"].replace("x * y", "x + y")
    altered["closure"] = closure
    assert cache.fingerprint(altered) != fp1
    # closure entries sort by qualname: order does not matter
    shuffled = dict(ext)
    shuffled["closure"] = list(reversed(ext["closure"]))
    assert cache.fingerprint(shuffled) == fp1


# ---------------------------------------------------------------------------
# cache module records
# ---------------------------------------------------------------------------


def test_store_lookup_roundtrip(tmp_path):
    record = {"method": "m.f", "verify": {"passed": 10}, "bench": {"speedup": 2.0}}
    cache.store(tmp_path, "abc123", record)
    got = cache.lookup(tmp_path, "abc123")
    assert got["method"] == "m.f"
    assert got["fingerprint"] == "abc123"
    assert got["cache_version"] == cache.CACHE_VERSION
    assert got["verify"]["passed"] == 10


def test_lookup_missing_file_misses(tmp_path):
    assert cache.lookup(tmp_path, "nope") is None


def test_lookup_corrupt_json_misses(tmp_path):
    (tmp_path / "deadbeef.json").write_text("{not valid json", encoding="utf-8")
    assert cache.lookup(tmp_path, "deadbeef") is None


def test_lookup_version_mismatch_misses(tmp_path):
    (tmp_path / "cafe.json").write_text(
        json.dumps({"cache_version": 9999, "method": "m.f"}), encoding="utf-8"
    )
    assert cache.lookup(tmp_path, "cafe") is None


# ---------------------------------------------------------------------------
# pipeline cache behaviour
# ---------------------------------------------------------------------------


def _success_run(tmp_path, cache_dir, **kw):
    return py_pipeline.run_python_pipeline(
        FIXTURES,
        method="cffi_kernels.dot",
        glue_out=tmp_path / "glue",
        work_dir=tmp_path / "work",
        translated_body=DOT_C,
        cache_dir=cache_dir,
        cases=20,
        seed=1,
        iters=200,
        warmup=20,
        repeats=2,
        **kw,
    )


def test_cache_hit_skips_rebuild(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    rep1 = _success_run(tmp_path, cache_dir)
    assert rep1["status"] == "success"
    assert len(list(cache_dir.glob("*.json"))) == 1

    def _boom(*a, **k):
        raise AssertionError("glue must not be regenerated on a cache hit")

    monkeypatch.setattr(py_pipeline.C, "generate", _boom)
    monkeypatch.setattr(py_pipeline.H, "build_env", _boom)
    rep2 = py_pipeline.run_python_pipeline(
        FIXTURES,
        method="cffi_kernels.dot",
        glue_out=tmp_path / "glue2",
        work_dir=tmp_path / "work2",
        cache_dir=cache_dir,
    )
    assert rep2["status"] == "cache_hit"
    assert rep2["cache_hit"] is True
    assert rep2["fingerprint"] == rep1["fingerprint"]
    assert "Cache hit" in rep2["instructions"]


def test_cache_hit_reuses_verify_and_bench(tmp_path):
    cache_dir = tmp_path / "cache"
    rep1 = _success_run(tmp_path, cache_dir)
    rep2 = py_pipeline.run_python_pipeline(
        FIXTURES,
        method="cffi_kernels.dot",
        glue_out=tmp_path / "g2",
        work_dir=tmp_path / "w2",
        translated_body=DOT_C,  # ignored: the hit short-circuits before splice
        cache_dir=cache_dir,
        cases=20,
        seed=1,
        iters=200,
        warmup=20,
        repeats=2,
    )
    assert rep2["status"] == "cache_hit"
    assert rep2["verify"] == rep1["verify"]
    assert rep2["bench"] == rep1["bench"]
    assert rep2["translated_body"] == DOT_C


def test_no_cache_bypasses_hit_and_store(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    rep1 = _success_run(tmp_path, cache_dir)
    assert rep1["fingerprint"]
    assert len(list(cache_dir.glob("*.json"))) == 1

    rep2 = _success_run(tmp_path, cache_dir, no_cache=True)
    assert rep2["status"] == "success"
    assert rep2["cache_hit"] is False
    assert rep2["fingerprint"] is None
    # no_cache also skips the store: still exactly one record from run 1
    assert len(list(cache_dir.glob("*.json"))) == 1


def test_failed_run_is_not_cached(tmp_path):
    cache_dir = tmp_path / "cache"
    rep = py_pipeline.run_python_pipeline(
        FIXTURES,
        method="cffi_kernels.dot",
        glue_out=tmp_path / "glue",
        work_dir=tmp_path / "work",
        translated_body=DOT_C_WRONG,
        cache_dir=cache_dir,
        cases=20,
        seed=1,
    )
    assert rep["status"] == "verify_failed"
    assert list(cache_dir.glob("*.json")) == []


def test_rejected_kernel_is_not_cached(tmp_path):
    cache_dir = tmp_path / "cache"
    rep = py_pipeline.run_python_pipeline(
        FIXTURES,
        method="cffi_kernels.log_sample",
        glue_out=tmp_path / "glue",
        work_dir=tmp_path / "work",
        translated_body=DOT_C,
        cache_dir=cache_dir,
    )
    assert rep["status"] == "rejected"
    assert list(cache_dir.glob("*.json")) == []


# ---------------------------------------------------------------------------
# multi-candidate selection
# ---------------------------------------------------------------------------


class _FakeVerify:
    def __init__(self, ok):
        self.passed = 50 if ok else 0
        self.failed = 0 if ok else 10
        self.skipped = 0
        self.boundary_passed = 5 if ok else 0
        self.boundary_failed = 0
        self.boundary_skipped = 0
        self.all_passed = ok
        self.tol_float = 1e-6
        self.tol_double = 1e-12
        self.cases = []
        self.boundary_cases = []


class _FakeBench:
    def __init__(self, c_ns):
        self.iters = 200
        self.warmup = 20
        self.repeats = 2
        self.python_per_call_ns = 1000.0
        self.c_per_call_ns = float(c_ns)
        self.speedup = 1000.0 / float(c_ns)

    @property
    def verdict(self):
        from kernel_port.harness import verdict_for_speedup

        return verdict_for_speedup(self.speedup)

    def verdict_text(self, native_label="C (cffi)"):
        return f"{native_label} is {self.speedup:.2f}x faster (median)."


@pytest.fixture()
def fake_harness(monkeypatch):
    """Replace gcc/verify/bench with deterministic fakes.

    Candidate outcome is keyed by the candidate dir name: names ending in
    "_bad" fail verification, everything else passes; per-call ns comes
    from the name: "fast" -> 100ns, anything else -> 500ns.
    """
    from kernel_port import cffi_glue as C

    def fake_generate(fns, out):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        sig = C.map_signature(fns[0])
        p = out / "kernels_cffi.c"
        p.write_text(
            f"double {sig.cname}(void) {{\n{C._BEGIN}\n/* TODO */\n    {C._END}\n}}\n",
            encoding="utf-8",
        )
        return {"c": p, "py": out / "kernels_cffi.py"}

    def fake_build_env(glue_dir, work_dir, sig):
        return SimpleNamespace(name=Path(glue_dir).name)

    def fake_run_verify(env, sig, repo_root, **kw):
        return _FakeVerify(not env.name.endswith("_bad"))

    def fake_run_bench(env, sig, repo_root, **kw):
        return _FakeBench(100.0 if env.name == "fast" else 500.0)

    monkeypatch.setattr(py_pipeline.C, "generate", fake_generate)
    monkeypatch.setattr(py_pipeline.H, "build_env", fake_build_env)
    monkeypatch.setattr(py_pipeline.H, "run_verify", fake_run_verify)
    monkeypatch.setattr(py_pipeline.H, "run_bench", fake_run_bench)


def _cand_dir(tmp_path, bodies: dict):
    d = tmp_path / "candidates"
    d.mkdir()
    for name, body in bodies.items():
        (d / f"{name}.c").write_text(body, encoding="utf-8")
    return d


def test_candidates_pick_fastest_passing(tmp_path, fake_harness):
    cand = _cand_dir(
        tmp_path, {"slow": "return 0;", "fast": "return 0;", "broken_bad": "return 0;"}
    )
    rep = py_pipeline.run_python_pipeline(
        FIXTURES,
        method="cffi_kernels.dot",
        glue_out=tmp_path / "glue",
        work_dir=tmp_path / "work",
        candidates_dir=cand,
        cases=20,
        seed=1,
    )
    assert rep["status"] == "success"
    assert rep["winner"] == "fast"
    assert [c["name"] for c in rep["candidates"]] == ["broken_bad", "fast", "slow"]
    table = rep["candidates_table"]
    assert "WINNER *" in table and "FAILED" in table
    assert f"Winner: fast" in rep["instructions"]


def test_candidates_all_failing(tmp_path, fake_harness):
    cand = _cand_dir(tmp_path, {"one_bad": "x", "two_bad": "x"})
    rep = py_pipeline.run_python_pipeline(
        FIXTURES,
        method="cffi_kernels.dot",
        glue_out=tmp_path / "glue",
        work_dir=tmp_path / "work",
        candidates_dir=cand,
        cases=20,
        seed=1,
    )
    assert rep["status"] == "verify_failed"
    assert rep["winner"] is None
    assert "No candidate passed verification" in rep["instructions"]
    assert "comparison" in rep["instructions"]
    assert "FAILED" in rep["candidates_table"]


def test_candidates_empty_dir(tmp_path):
    cand = tmp_path / "candidates"
    cand.mkdir()
    rep = py_pipeline.run_python_pipeline(
        FIXTURES,
        method="cffi_kernels.dot",
        glue_out=tmp_path / "glue",
        work_dir=tmp_path / "work",
        candidates_dir=cand,
        cases=20,
        seed=1,
    )
    assert rep["status"] == "verify_failed"
    assert rep["winner"] is None
    assert rep["candidates"] == []


def test_candidates_combined_with_translated_body(tmp_path, fake_harness):
    # translated body drives verify/bench + cache store; candidates still run
    cache_dir = tmp_path / "cache"
    cand = _cand_dir(tmp_path, {"fast": "return 0;", "slow": "return 0;"})
    rep = py_pipeline.run_python_pipeline(
        FIXTURES,
        method="cffi_kernels.dot",
        glue_out=tmp_path / "glue",
        work_dir=tmp_path / "work",
        translated_body=DOT_C,
        candidates_dir=cand,
        cache_dir=cache_dir,
        cases=20,
        seed=1,
        iters=200,
        warmup=20,
        repeats=2,
    )
    assert rep["status"] == "success"
    assert rep["winner"] == "fast"
    assert rep["verify"]["all_passed"]
    assert rep["bench"]["speedup"] > 0
    assert len(list(cache_dir.glob("*.json"))) == 1


def test_candidate_build_error_recorded_and_excluded(tmp_path, fake_harness):
    cand = _cand_dir(
        tmp_path, {"good": "return 0;", "junk": "this is not C at all ((("}
    )

    # the fake harness from the fixture would "build" anything; make the
    # junk candidate's build fail to exercise the error path instead.
    def raising_build_env(glue_dir, work_dir, sig):
        if Path(glue_dir).name == "junk":
            raise RuntimeError("gcc boom")
        return SimpleNamespace(name=Path(glue_dir).name)

    with mock.patch.object(py_pipeline.H, "build_env", raising_build_env):
        rep = py_pipeline.run_python_pipeline(
            FIXTURES,
            method="cffi_kernels.dot",
            glue_out=tmp_path / "glue",
            work_dir=tmp_path / "work",
            candidates_dir=cand,
            cases=20,
            seed=1,
        )
    junk = next(c for c in rep["candidates"] if c["name"] == "junk")
    assert "error" in junk and "gcc boom" in junk["error"]
    assert rep["winner"] == "good"


def test_candidates_real_end_to_end(tmp_path):
    cand = _cand_dir(tmp_path, {"dot_v1": DOT_C})
    rep = py_pipeline.run_python_pipeline(
        FIXTURES,
        method="cffi_kernels.dot",
        glue_out=tmp_path / "glue",
        work_dir=tmp_path / "work",
        candidates_dir=cand,
        cases=20,
        seed=1,
        iters=200,
        warmup=20,
        repeats=2,
    )
    assert rep["status"] == "success", rep.get("instructions")
    assert rep["winner"] == "dot_v1"
    assert rep["candidates"][0]["verify_passed"] > 0
    assert rep["candidates"][0]["bench_median_ns"] > 0


def test_select_pick_winner_prefers_fastest():
    results = [
        {"name": "slow", "verify_passed": 10, "verify_failed": 0, "bench_median_ns": 900.0},
        {"name": "fast", "verify_passed": 10, "verify_failed": 0, "bench_median_ns": 100.0},
        {"name": "bad", "verify_passed": 0, "verify_failed": 3, "bench_median_ns": 10.0},
    ]
    assert select.pick_winner(results)["name"] == "fast"
    table = select.compare_table(results)
    rows = {line.split()[0]: line for line in table.splitlines()[2:]}
    assert rows["fast"].endswith("WINNER *")
    assert rows["slow"].endswith("slower")
    assert rows["bad"].endswith("FAILED")


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def _invoke(args):
    from typer.testing import CliRunner

    from kernel_port.cli import app

    return CliRunner().invoke(app, args)


def test_cli_python_accepts_cache_flags(tmp_path):
    res = _invoke(
        [
            "pipeline", "--lang", "python", "--method", "cffi_kernels.dot",
            "--no-cache", "--cache-dir", str(tmp_path / "cache"),
            "--glue-out", str(tmp_path / "g"), "--work", str(tmp_path / "w"),
            str(FIXTURES),
        ]
    )
    assert res.exit_code == 0, res.output
    assert "awaiting_translation" in res.output


def test_cli_python_accepts_candidates_dir(tmp_path):
    cand = tmp_path / "candidates"
    cand.mkdir()  # empty: no passing candidate -> verify_failed, exit 1
    res = _invoke(
        [
            "pipeline", "--lang", "python", "--method", "cffi_kernels.dot",
            "--candidates-dir", str(cand),
            "--glue-out", str(tmp_path / "g"), "--work", str(tmp_path / "w"),
            str(FIXTURES),
        ]
    )
    assert res.exit_code == 1, res.output
    assert "verify_failed" in res.output
    assert "does not support" not in res.output


def test_cli_python_still_rejects_java_only_flags(tmp_path):
    res = _invoke(
        [
            "pipeline", "--lang", "python", "--backend", "ffm",
            "--method", "cffi_kernels.dot", str(FIXTURES),
        ]
    )
    assert res.exit_code != 0
    assert "does not support" in res.output


def test_cli_python_cache_roundtrip(tmp_path):
    body = tmp_path / "body.c"
    body.write_text(DOT_C, encoding="utf-8")
    cache_dir = tmp_path / "cache"
    args = [
        "pipeline", "--lang", "python", "--method", "cffi_kernels.dot",
        "--translated-body", str(body), "--cache-dir", str(cache_dir),
        "--cases", "20", "--seed", "1",
        "--glue-out", str(tmp_path / "g"), "--work", str(tmp_path / "w"),
        str(FIXTURES),
    ]
    res = _invoke(args)
    assert res.exit_code == 0, res.output
    assert "success" in res.output
    assert len(list(cache_dir.glob("*.json"))) == 1
    # second run: cache hit, no rebuild
    res2 = _invoke(args)
    assert res2.exit_code == 0, res2.output
    assert "cache_hit" in res2.output or "cache hit" in res2.output
