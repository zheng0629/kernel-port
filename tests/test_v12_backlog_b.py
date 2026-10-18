"""Tests for the v0.12 backlog workstream (part B).

Part 1 -- regression pins for items claimed done in v0.10/v0.11:
- B2: bench noise band (+-20% -> inconclusive) applies to the Python backend.

Part 2 -- new behavior:
- B7: --clusters/--no-clusters/--class warn (stderr) under --lang python
      instead of being silently ignored.
- B5: the Python driver reports ImportError/ModuleNotFoundError as a
      driver/import error, not a "segfault".
- B9: --list-len and --sizes for `bench --lang python`.
"""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from kernel_port import cffi_glue, py_ast, py_harness, py_mine
from kernel_port.cffi_glue import KernelSig, RetMap
from kernel_port.cli import app

from .conftest import FIXTURE_REPO

runner = CliRunner()


@pytest.fixture()
def py_repo(tmp_path):
    (tmp_path / "k.py").write_text(
        "def vsum(xs: list[float]) -> float:\n"
        "    return sum(xs)\n"
    )
    return tmp_path


def _py_sig_and_env(tmp_path):
    (tmp_path / "k.py").write_text(
        "def vsum(xs: list[float]) -> float:\n"
        "    return sum(xs)\n"
    )
    repo = py_ast.scan_repo(tmp_path)
    fn = next(f for f in repo.top_level_functions() if f.qualname == "k.vsum")
    sig = cffi_glue.map_signature(fn)
    env = py_harness.PyBuiltEnv(
        glue_dir=tmp_path, work_dir=tmp_path, so_path=tmp_path / "kernels.so",
        sig=sig,
    )
    return env, sig


# ---------------------------------------------------------------------------
# B2: Python backend honors the noise band
# ---------------------------------------------------------------------------


def _py_bench_report(py_ns: float, c_ns: float):
    return py_harness.PyBenchReport(
        method="k.vsum", iters=10, warmup=2, repeats=3,
        python_ns=[py_ns] * 3, c_ns=[c_ns] * 3,
    )


def test_py_bench_verdict_noise_band_inconclusive():
    # 1.10x is inside the +-20% band -> inconclusive, never "faster".
    rep = _py_bench_report(1100.0, 1000.0)
    assert rep.verdict == "inconclusive"
    assert "inconclusive" in rep.verdict_text()
    # The band edges are inclusive: exactly 1.20x is still too close to call.
    assert _py_bench_report(1200.0, 1000.0).verdict == "inconclusive"
    assert _py_bench_report(1000.0, 1200.0).verdict == "inconclusive"


def test_py_bench_verdict_outside_band():
    assert _py_bench_report(1300.0, 1000.0).verdict == "faster"
    assert "faster" in _py_bench_report(1300.0, 1000.0).verdict_text()
    assert _py_bench_report(1000.0, 1300.0).verdict == "slower"
    assert "SLOWER" in _py_bench_report(1000.0, 1300.0).verdict_text()


# ---------------------------------------------------------------------------
# B5: import errors are not reported as segfaults
# ---------------------------------------------------------------------------


def _run_driver_with_fake_proc(tmp_path, monkeypatch, returncode, stderr):
    """Run _run_driver with subprocess.run stubbed; returns the raised error."""
    env, sig = _py_sig_and_env(tmp_path)

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=returncode, stderr=stderr, stdout="")

    monkeypatch.setattr(py_harness.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as excinfo:
        py_harness._run_driver(env, sig, tmp_path, "bench", {"args": []}, 30.0)
    return excinfo.value


def test_driver_failed_on_import_unit():
    assert py_harness._driver_failed_on_import(
        "Traceback (most recent call last):\n"
        '  File "x.py", line 1, in <module>\n'
        "ModuleNotFoundError: No module named 'cffi'\n"
    )
    assert py_harness._driver_failed_on_import(
        "ImportError: cannot import name 'ffi' from 'cffi'\n"
    )
    assert not py_harness._driver_failed_on_import("")
    assert not py_harness._driver_failed_on_import(None)
    assert not py_harness._driver_failed_on_import(
        "ValueError: bad input\n"
    )


def test_run_driver_import_error_says_driver_not_segfault(tmp_path, monkeypatch):
    err = _run_driver_with_fake_proc(
        tmp_path, monkeypatch, 1,
        "Traceback (most recent call last):\n"
        "ModuleNotFoundError: No module named 'cffi'\n",
    )
    msg = str(err)
    assert "driver import error" in msg
    assert "segfault" not in msg
    assert "No module named 'cffi'" in msg  # the message is preserved


def test_run_driver_generic_nonzero_exit_keeps_crash_report(tmp_path, monkeypatch):
    # A non-import failure (exit 1, arbitrary traceback) is still reported
    # as a crash -- that wording is what a segfaulting C translation shows.
    err = _run_driver_with_fake_proc(
        tmp_path, monkeypatch, 1,
        "Traceback (most recent call last):\n"
        "ValueError: kernel returned out-of-range value\n",
    )
    msg = str(err)
    assert "crashed" in msg
    assert "segfault" in msg


def test_run_driver_signal_kill_keeps_crash_report(tmp_path, monkeypatch):
    # Killed by a signal (negative returncode, e.g. -11/SIGSEGV) is always a
    # real crash, even if the stderr tail happens to mention an import.
    err = _run_driver_with_fake_proc(
        tmp_path, monkeypatch, -11,
        "ModuleNotFoundError: No module named 'cffi'\n",
    )
    assert "crashed" in str(err)
    assert "driver import error" not in str(err)


# ---------------------------------------------------------------------------
# B7: warnings for --lang python
# ---------------------------------------------------------------------------


def test_mine_python_no_clusters_warns_on_stderr(py_repo):
    r = runner.invoke(app, ["mine", str(py_repo), "--lang", "python", "--no-clusters"])
    assert r.exit_code == 0, r.output
    assert "Warning" in r.stderr
    assert "--no-clusters" in r.stderr or "--clusters" in r.stderr


def test_mine_python_clusters_warns_on_stderr(py_repo):
    r = runner.invoke(app, ["mine", str(py_repo), "--lang", "python", "--clusters"])
    assert r.exit_code == 0, r.output
    assert "Warning" in r.stderr


def test_mine_python_default_emits_no_warning(py_repo):
    r = runner.invoke(app, ["mine", str(py_repo), "--lang", "python"])
    assert r.exit_code == 0, r.output
    assert "Warning" not in r.stderr


def test_mine_python_json_stdout_stays_machine_clean(py_repo):
    # The warning must go to stderr so --format json stdout stays parseable.
    r = runner.invoke(
        app,
        ["mine", str(py_repo), "--lang", "python", "--format", "json", "--no-clusters"],
    )
    assert r.exit_code == 0, r.output
    data = json.loads(r.stdout)  # would raise if the warning leaked to stdout
    assert data["lang"] == "python"
    assert "Warning" in r.stderr
    assert "Warning" not in r.stdout


def test_mine_java_default_still_shows_clusters():
    # Regression: the None sentinel must preserve the old default (True).
    r = runner.invoke(app, ["mine", str(FIXTURE_REPO)])
    assert r.exit_code == 0, r.output
    assert "function clusters" in r.output


def test_mine_java_no_clusters_hides_cluster_table():
    r = runner.invoke(app, ["mine", str(FIXTURE_REPO), "--no-clusters"])
    assert r.exit_code == 0, r.output
    assert "function clusters" not in r.output
    assert "Warning" not in r.stderr  # Java path never warns


def test_glue_python_class_warns_on_stderr(py_repo, tmp_path):
    out = tmp_path / "glue"
    r = runner.invoke(
        app,
        ["glue", str(py_repo), "--lang", "python", "--method", "k.vsum",
         "--class", "k.SomeClass", "--out", str(out)],
    )
    assert r.exit_code == 0, r.output
    assert "Warning" in r.stderr
    assert "--class" in r.stderr


def test_glue_python_without_class_emits_no_warning(py_repo, tmp_path):
    out = tmp_path / "glue"
    r = runner.invoke(
        app,
        ["glue", str(py_repo), "--lang", "python", "--method", "k.vsum",
         "--out", str(out)],
    )
    assert r.exit_code == 0, r.output
    assert "Warning" not in r.stderr


# ---------------------------------------------------------------------------
# B9: --list-len / --sizes for bench --lang python
# ---------------------------------------------------------------------------


def _capture_run_bench(monkeypatch, speedups):
    """Stub py_harness.run_bench; record list_len kwargs, serve canned reports."""
    captured = []

    def fake_run_bench(env, sig, repo_root, **kw):
        list_len = kw.get("list_len", 64)
        captured.append(list_len)
        sp = speedups.get(list_len, 1.0)
        return py_harness.PyBenchReport(
            method=sig.qualname,
            iters=kw.get("iters", 200),
            warmup=kw.get("warmup", 20),
            repeats=kw.get("repeats", 5),
            python_ns=[sp * 1000.0] * 3,
            c_ns=[1000.0] * 3,
        )

    monkeypatch.setattr(py_harness, "run_bench", fake_run_bench)
    monkeypatch.setattr(
        py_harness, "build_env",
        lambda gdir, work, sig: py_harness.PyBuiltEnv(
            glue_dir=Path(gdir), work_dir=Path(work),
            so_path=Path(gdir) / "kernels.so", sig=sig,
        ),
    )
    return captured


def test_run_bench_list_len_reaches_generated_args(tmp_path, monkeypatch):
    # Library-level: list_len controls the generated bench input sizes.
    env, sig = _py_sig_and_env(tmp_path)
    seen = {}

    def fake_driver(e, s, root, mode, payload, timeout):
        seen["args"] = payload["args"]
        return {"python_ns": [1.0], "c_ns": [1.0]}

    monkeypatch.setattr(py_harness, "_run_driver", fake_driver)
    py_harness.run_bench(env, sig, tmp_path, iters=4, warmup=1, repeats=1,
                         list_len=37)
    assert len(seen["args"][0]) == 37
    py_harness.run_bench(env, sig, tmp_path, iters=4, warmup=1, repeats=1)
    assert len(seen["args"][0]) == 64  # default preserved


def test_cli_bench_python_list_len_single_bench(py_repo, tmp_path, monkeypatch):
    captured = _capture_run_bench(monkeypatch, {})
    r = runner.invoke(
        app,
        ["bench", str(py_repo), "--lang", "python", "--method", "k.vsum",
         "--glue-dir", str(tmp_path / "g"), "--list-len", "37",
         "--iters", "4", "--warmup", "1", "--repeats", "1"],
    )
    assert r.exit_code == 0, r.output
    assert captured == [37]


def test_cli_bench_python_default_list_len_is_64(py_repo, tmp_path, monkeypatch):
    captured = _capture_run_bench(monkeypatch, {})
    r = runner.invoke(
        app,
        ["bench", str(py_repo), "--lang", "python", "--method", "k.vsum",
         "--glue-dir", str(tmp_path / "g"),
         "--iters", "4", "--warmup", "1", "--repeats", "1"],
    )
    assert r.exit_code == 0, r.output
    assert captured == [64]


def test_cli_bench_python_sizes_reports_per_size_and_break_even(
    py_repo, tmp_path, monkeypatch
):
    captured = _capture_run_bench(monkeypatch, {16: 0.9, 64: 1.1, 256: 2.0})
    r = runner.invoke(
        app,
        ["bench", str(py_repo), "--lang", "python", "--method", "k.vsum",
         "--glue-dir", str(tmp_path / "g"), "--sizes", "16,64,256",
         "--iters", "4", "--warmup", "1", "--repeats", "1"],
    )
    assert r.exit_code == 0, r.output
    assert captured == [16, 64, 256]
    assert "16" in r.output and "256" in r.output
    assert "Break-even" in r.output
    assert ">= 256" in r.output  # 2.0x is the first size above the 1.2x bar


def test_cli_bench_python_sizes_never_pays_off(py_repo, tmp_path, monkeypatch):
    _capture_run_bench(monkeypatch, {16: 0.9, 64: 1.05})
    r = runner.invoke(
        app,
        ["bench", str(py_repo), "--lang", "python", "--method", "k.vsum",
         "--glue-dir", str(tmp_path / "g"), "--sizes", "16,64",
         "--iters", "4", "--warmup", "1", "--repeats", "1"],
    )
    assert r.exit_code == 0, r.output
    assert "never pays off" in r.output


def test_cli_bench_python_rejects_bad_list_len(py_repo, tmp_path):
    r = runner.invoke(
        app,
        ["bench", str(py_repo), "--lang", "python", "--method", "k.vsum",
         "--glue-dir", str(tmp_path / "g"), "--list-len", "0"],
    )
    assert r.exit_code == 1
    assert "--list-len" in r.output


def test_cli_bench_python_rejects_bad_sizes(py_repo, tmp_path):
    for bad in ("16,xx", "16,0", "16,-3"):
        r = runner.invoke(
            app,
            ["bench", str(py_repo), "--lang", "python", "--method", "k.vsum",
             "--glue-dir", str(tmp_path / "g"), "--sizes", bad],
        )
        assert r.exit_code == 1, bad
        assert "--sizes" in r.output, bad
