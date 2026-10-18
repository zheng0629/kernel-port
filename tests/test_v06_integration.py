"""v0.6 integration tests: CLI wiring for the five new workstreams.

Covers: mine --lang python, parse-cache flags, glue/verify/bench --backend rust,
repair command, verify --coverage. Module-level behavior is tested in the
workstreams' own test files; here we pin the wiring.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from kernel_port import harness, parsecache, py_ast, py_mine, repair, rust_glue
from kernel_port import java_ast as J

FIX = Path(__file__).parent / "fixtures" / "sample_kernels"
PYFIX = Path(__file__).parent / "fixtures" / "py_kernels"
RUSTFIX = Path(__file__).parent / "fixtures" / "rust_kernels"

VENV_PY = Path(__file__).parent.parent / ".venv" / "bin" / "python"


def run_cli(*args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ)
    env["PATH"] = f"{Path.home()}/workspace/tools/jdk-25.0.4.1+1/bin:" + env["PATH"]
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [str(VENV_PY), "-m", "kernel_port", *args],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
        env=env,
        timeout=300,
    )


# ---------------------------------------------------------------------------
# mine --lang python
# ---------------------------------------------------------------------------


def test_mine_python_json_lists_pure_and_impure():
    r = run_cli("mine", str(PYFIX), "--lang", "python", "--format", "json")
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["lang"] == "python"
    by_name = {c["function"]: c for c in data["functions"]}
    assert by_name["numeric.dot"]["pure"] is True
    assert by_name["numeric.matmul"]["pure"] is True
    # impure.py functions must be rejected with reasons
    impure = [c for c in data["functions"] if not c["pure"]]
    assert impure, "expected some impure Python functions"
    assert all(c["reasons"] for c in impure)


def test_mine_python_module_api():
    repo = py_ast.scan_repo(PYFIX)
    cands = py_mine.mine(repo)
    pure = [c for c in cands if c.pure]
    assert pure, "expected pure Python candidates"
    assert all(c.score > 0 for c in pure)


def test_mine_lang_rejects_unknown():
    r = run_cli("mine", str(FIX), "--lang", "go")
    assert r.returncode != 0


# ---------------------------------------------------------------------------
# parse cache flags
# ---------------------------------------------------------------------------


def test_mine_parse_cache_flags_accepted(tmp_path):
    # warm the cache, then hit it
    r1 = run_cli("mine", str(FIX), "--format", "json")
    assert r1.returncode == 0, r1.stderr
    r2 = run_cli("mine", str(FIX), "--format", "json")
    assert r2.returncode == 0, r2.stderr
    # json mode stays quiet (stdout must remain pure JSON)
    assert "parse cache:" not in r2.stderr and "parse cache:" not in r2.stdout
    json.loads(r2.stdout)  # still valid JSON
    # table mode reports the cache line on stderr
    r2t = run_cli("mine", str(FIX))
    assert r2t.returncode == 0, r2t.stderr
    assert "parse cache:" in r2t.stderr
    r3 = run_cli("mine", str(FIX), "--format", "json", "--no-parse-cache")
    assert r3.returncode == 0, r3.stderr
    r4 = run_cli("mine", str(FIX), "--format", "json", "--refresh-parse-cache")
    assert r4.returncode == 0, r4.stderr
    # same candidates regardless of cache mode
    assert json.loads(r1.stdout)["methods"] == json.loads(r3.stdout)["methods"]


def test_parsecache_module_api(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNEL_PORT_PARSE_CACHE_DIR", str(tmp_path / "pc"))
    repo = parsecache.load_repo(FIX)
    assert repo.methods, "expected parsed methods"
    stats = parsecache.cache_stats()
    assert stats["files_total"] > 0
    # second load: all hits
    repo2 = parsecache.load_repo(FIX)
    stats2 = parsecache.cache_stats()
    assert stats2["hits"] == stats2["files_total"]
    assert sorted(repo.methods) == sorted(repo2.methods)


# ---------------------------------------------------------------------------
# rust backend
# ---------------------------------------------------------------------------


def test_glue_backend_rust_generates(tmp_path):
    r = run_cli(
        "glue", str(RUSTFIX), "--method", "com.example.RustKernels.dot",
        "--backend", "rust", "--out", str(tmp_path / "g"),
    )
    assert r.returncode == 0, r.stderr
    out = tmp_path / "g"
    assert (out / "Cargo.toml").exists()
    assert (out / "src" / "lib.rs").exists()
    assert (out / "RustKernelsNative.java").exists()
    lib_rs = (out / "src" / "lib.rs").read_text()
    assert "BEGIN TRANSLATED BODY" in lib_rs
    assert "#[no_mangle]" in lib_rs


def test_glue_backend_rust_rejects_bad_backend():
    r = run_cli(
        "glue", str(FIX), "--method", "com.example.Kernels.dot",
        "--backend", "cobol", "--out", "/tmp/never",
    )
    assert r.returncode != 0


def test_rust_glue_module_api():
    repo = J.scan_repo(RUSTFIX)
    m = next(v for (c, n), v in repo.methods.items() if n == "dot")[0]
    plan = rust_glue.plan(m)
    assert "dot" in plan.jni_func
    cargo = rust_glue.render_cargo_toml(plan)
    assert 'crate-type = ["cdylib"]' in cargo
    assert 'jni' in cargo


# ---------------------------------------------------------------------------
# repair command
# ---------------------------------------------------------------------------


def test_repair_command_success(tmp_path):
    script = tmp_path / "propose.sh"
    script.write_text(
        "#!/bin/bash\n"
        'if [ "$2" = "1" ]; then\n'
        "  echo 'double s = 0.0; for (int i = 0; i < p_a_len - 1; i++) s += p_a[i] * p_b[i]; return s;'\n"
        "else\n"
        "  echo 'double s = 0.0; for (int i = 0; i < p_a_len; i++) s += p_a[i] * p_b[i]; return s;'\n"
        "fi\n"
    )
    script.chmod(0o755)
    r = run_cli(
        "repair", str(FIX), "--method", "com.example.Kernels.dot",
        "--propose-cmd", str(script), "--max-rounds", "3",
        "--verify-cases", "10", "--work-dir", str(tmp_path / "w"),
    )
    assert r.returncode == 0, r.stderr + r.stdout
    assert "success" in r.stdout


def test_repair_command_requires_propose_cmd():
    r = run_cli("repair", str(FIX), "--method", "com.example.Kernels.dot")
    assert r.returncode != 0
    assert "propose-cmd" in r.stdout + r.stderr


# ---------------------------------------------------------------------------
# verify --coverage (smoke)
# ---------------------------------------------------------------------------


def test_verify_coverage_smoke(tmp_path):
    pytest.importorskip("kernel_port.coverage")
    glue_dir = tmp_path / "g"
    r = run_cli(
        "glue", str(FIX), "--method", "com.example.Kernels.dot",
        "--out", str(glue_dir), "--no-fallback",
    )
    assert r.returncode == 0, r.stderr
    cpp = glue_dir / "kernels_jni.cpp"
    t = cpp.read_text()
    body = (
        "// ==== BEGIN TRANSLATED BODY ====\n"
        "    double s = 0.0;\n"
        "    for (int i = 0; i < p_a_len; i++) s += p_a[i] * p_b[i];\n"
        "    return s;\n"
        "    // ==== END TRANSLATED BODY ===="
    )
    import re

    t = re.sub(
        r"// ==== BEGIN TRANSLATED BODY ====.*?// ==== END TRANSLATED BODY ====",
        body, t, flags=re.S,
    )
    cpp.write_text(t)
    r = run_cli(
        "verify", str(FIX), "--method", "com.example.Kernels.dot",
        "--glue-dir", str(glue_dir), "--cases", "10", "--coverage",
    )
    assert r.returncode == 0, r.stderr + r.stdout
    assert "branch coverage" in r.stdout
    assert "guided" in r.stdout


def test_verify_report_has_coverage_fields():
    rep = harness.VerifyReport(method="m", cases=[], passed=0, failed=0)
    assert rep.coverage_pct is None
    assert rep.branches_hit == 0
    assert rep.guided == []
    assert rep.all_passed  # empty guided list keeps old semantics
