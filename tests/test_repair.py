"""Tests for the repair-loop driver (kernel_port.repair).

The provider hooks are fakes -- no real LLM is involved. Correct/wrong C++
bodies are hand-written like in test_e2e.py. Round-trip tests need a JDK and
g++ on PATH and take a few seconds per round (javac + g++ + two JVM starts).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from kernel_port import ffm, repair
from kernel_port.repair import RepairError, extract_body_from_output, repair_loop

from .conftest import FIXTURE_REPO, TRANSLATIONS

pytestmark = pytest.mark.e2e

DOT = "com.example.Kernels.dot"
DOT_OK = TRANSLATIONS[DOT]
DOT_WRONG = TRANSLATIONS["com.example.Kernels.dot_WRONG"]
FFM_REPO = Path(__file__).parent / "fixtures" / "ffm_kernels"

_JDK25 = ffm.find_jdk(22)
needs_jdk22 = pytest.mark.skipif(
    _JDK25 is None, reason="e2e FFM tests need a JDK >= 22 (none found)"
)

FFM_DOT_OK = (
    "    double s = 0.0;\n"
    "    for (int64_t i = 0; i < p_a_len; i++) s += p_a[i] * p_b[i];\n"
    "    return s;"
)
FFM_DOT_WRONG = (
    "    double s = 0.0;\n"
    "    for (int64_t i = 0; i < p_a_len; i++) s += p_a[i] * p_b[i];\n"
    "    return s + 1.0;"
)


def _run(repo_root, method, provider, tmp_path, **kw):
    kw.setdefault("verify_cases", 10)
    kw.setdefault("work_dir", tmp_path / "repair-work")
    return repair_loop(repo_root, method, provider, **kw)


# ---------------------------------------------------------------------------
# unit tests (no compilation)
# ---------------------------------------------------------------------------


def test_extract_body_from_output_plain():
    assert extract_body_from_output("  int x = 1;\n") == "int x = 1;"


def test_extract_body_from_output_unwraps_fences():
    out = "Here is the fix:\n```cpp\n    double s = 0.0;\n    return s;\n```\nDone."
    assert extract_body_from_output(out) == "double s = 0.0;\n    return s;"


def test_repair_needs_exactly_one_hook(tmp_path):
    with pytest.raises(ValueError):
        repair_loop(FIXTURE_REPO, DOT, work_dir=tmp_path)
    with pytest.raises(ValueError):
        repair_loop(
            FIXTURE_REPO, DOT, lambda p, r, f: "x",
            propose_cmd="true", work_dir=tmp_path,
        )


def test_repair_unknown_method_is_error(tmp_path):
    rep = _run(FIXTURE_REPO, "com.example.Kernels.nope", lambda p, r, f: "x",
               tmp_path)
    assert rep.status == "error"
    assert rep.rounds_run == 0
    assert "not found" in (rep.error or "")


def test_repair_impure_method_refused(tmp_path):
    rep = _run(FIXTURE_REPO, "com.example.Kernels.noisy",
               lambda p, r, f: "x", tmp_path)
    assert rep.status == "error"
    assert "not pure" in (rep.error or "")
    assert rep.rounds_run == 0


# ---------------------------------------------------------------------------
# round-trip tests (JNI backend)
# ---------------------------------------------------------------------------


def test_repair_round1_pass_is_single_round(tmp_path):
    calls = []

    def provider(prompt, round_no, failures):
        calls.append((round_no, failures))
        return DOT_OK

    rep = _run(FIXTURE_REPO, DOT, provider, tmp_path)
    assert rep.status == "success"
    assert rep.rounds_run == 1
    assert rep.final_body == DOT_OK
    assert calls == [(1, [])]
    r1 = rep.rounds[0]
    assert r1.failed == 0 and r1.passed == 10
    assert r1.build_error is None
    assert rep.total_elapsed_s > 0


def test_repair_wrong_then_correct_succeeds_in_two_rounds(tmp_path):
    seen = {}

    def provider(prompt, round_no, failures):
        seen[round_no] = (prompt, list(failures))
        return DOT_WRONG if round_no == 1 else DOT_OK

    rep = _run(FIXTURE_REPO, DOT, provider, tmp_path)
    assert rep.status == "success"
    assert rep.rounds_run == 2
    assert rep.final_body == DOT_OK
    assert rep.rounds[0].failed > 0
    assert rep.rounds[1].failed == 0
    # round 2 saw the counterexamples from round 1, in both channels
    _prompt2, failures2 = seen[2]
    assert failures2, "provider got no failure dicts on round 2"
    assert failures2[0]["java_result"] != failures2[0]["jni_result"]
    prompt2 = (tmp_path / "repair-work" / "round-2" / "prompt.md").read_text()
    assert "Counterexamples" in prompt2
    assert "previous attempt" in prompt2.lower()


def test_repair_always_wrong_gives_up_at_max_rounds(tmp_path):
    # A *different* wrong body every round: failure sets never repeat, so
    # the loop runs to max_rounds instead of stalling.
    last = {}

    def provider(prompt, round_no, failures):
        body = (
            "    double s = 0.0;\n"
            "    for (int i = 0; i < p_a_len; i++) s += p_a[i] * p_b[i];\n"
            f"    return s + {round_no}.0;"
        )
        last["body"] = body
        return body

    rep = _run(FIXTURE_REPO, DOT, provider, tmp_path, max_rounds=3)
    assert rep.status == "gave_up"
    assert rep.rounds_run == 3
    assert rep.final_body is None
    assert all(r.failed > 0 for r in rep.rounds)
    sigs = [r.failure_signature for r in rep.rounds]
    assert len(set(sigs)) == 3, "expected distinct failure sets per round"
    # hardening: last attempted body + human-readable stop reason
    assert rep.last_body == last["body"]
    assert rep.to_dict()["last_body"] == last["body"]
    assert rep.stop_reason == "max_rounds=3 reached without success"
    assert rep.to_dict()["stop_reason"] == rep.stop_reason


def test_repair_identical_failures_stall_early(tmp_path):
    rep = _run(FIXTURE_REPO, DOT, lambda p, r, f: DOT_WRONG, tmp_path,
               max_rounds=5)
    assert rep.status == "stalled"
    assert rep.rounds_run == 2, "should stop after two identical failure sets"
    assert rep.rounds[0].failure_signature == rep.rounds[1].failure_signature
    assert rep.last_body == DOT_WRONG
    assert "round 1" in (rep.stop_reason or "") and "oscillation" in (
        rep.stop_reason or "")


def test_repair_oscillation_abab_stalls_on_round_3(tmp_path):
    # A/B/A/B oscillation: comparing only against the previous round would
    # never catch this; the loop must remember ALL past signatures.
    wrong_b = DOT_WRONG.replace("s + 1.0", "s + 2.0")

    def provider(prompt, round_no, failures):
        return DOT_WRONG if round_no % 2 == 1 else wrong_b

    rep = _run(FIXTURE_REPO, DOT, provider, tmp_path, max_rounds=5)
    assert rep.status == "stalled"
    assert rep.rounds_run == 3, "round 3 repeats round 1's failure set"
    sigs = [r.failure_signature for r in rep.rounds]
    assert sigs[0] != sigs[1], "A and B must fail differently"
    assert sigs[2] == sigs[0], "round 3 must repeat round 1"
    assert rep.last_body == DOT_WRONG
    d = rep.to_dict()
    assert "round 1" in d["stop_reason"] and "oscillation" in d["stop_reason"]


def test_repair_success_and_error_have_no_last_body(tmp_path):
    ok = _run(FIXTURE_REPO, DOT, lambda p, r, f: DOT_OK, tmp_path)
    assert ok.status == "success"
    assert ok.last_body is None
    assert ok.to_dict()["last_body"] is None
    assert "passed" in (ok.stop_reason or "")
    assert 'round 1' in (ok.stop_reason or "")

    err = _run(FIXTURE_REPO, "com.example.Kernels.nope", lambda p, r, f: "x",
               tmp_path)
    assert err.status == "error"
    assert err.last_body is None
    assert (err.stop_reason or "").startswith("error:")


def test_repair_uncompilable_body_records_build_error(tmp_path):
    rep = _run(FIXTURE_REPO, DOT, lambda p, r, f: "this is not C++ {{{",
               tmp_path, max_rounds=5)
    # identical build failures two rounds in a row -> stalled
    assert rep.status == "stalled"
    assert rep.rounds_run == 2
    assert rep.rounds[0].build_error, "expected a recorded build error"
    assert rep.rounds[0].passed == 0 and rep.rounds[0].failed == 0


def test_repair_report_is_json_serializable(tmp_path):
    rep = _run(FIXTURE_REPO, DOT, lambda p, r, f: DOT_WRONG, tmp_path,
               max_rounds=2)
    data = json.loads(json.dumps(rep.to_dict()))
    assert data["status"] == "stalled"
    assert data["method"] == DOT
    assert data["backend"] == "jni"
    assert len(data["rounds"]) == 2
    r1 = data["rounds"][0]
    assert set(r1) >= {"round", "passed", "failed", "build_error",
                       "failure_summaries", "prompt_path"}
    assert data["final_body"] is None
    assert set(data) >= {"last_body", "stop_reason"}
    assert data["last_body"] == DOT_WRONG
    assert "oscillation" in data["stop_reason"]


# ---------------------------------------------------------------------------
# shell-hook contract
# ---------------------------------------------------------------------------

_PROPOSE_SCRIPT = """\
import json
import sys
from pathlib import Path

WRONG = {wrong!r}
CORRECT = {correct!r}

prompt_file, round_no, failures_file = sys.argv[1], int(sys.argv[2]), sys.argv[3]
prompt = Path(prompt_file).read_text(encoding="utf-8")
failures = json.loads(Path(failures_file).read_text(encoding="utf-8"))
record = {{
    "argv_tail": [Path(prompt_file).name, sys.argv[2], Path(failures_file).name],
    "round": round_no,
    "failures_len": len(failures),
    "prompt_has_counterexamples": "Counterexamples" in prompt,
}}
with open({sidecar!r}, "a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\\n")
print(CORRECT if (round_no >= 2 and failures) else WRONG)
"""


def _write_propose_script(tmp_path, wrong=DOT_WRONG, correct=DOT_OK) -> Path:
    sidecar = tmp_path / "propose_calls.jsonl"
    script = tmp_path / "propose.py"
    script.write_text(
        _PROPOSE_SCRIPT.format(wrong=wrong, correct=correct,
                               sidecar=str(sidecar)),
        encoding="utf-8",
    )
    return script


def test_repair_shell_hook_contract(tmp_path):
    script = _write_propose_script(tmp_path)
    cmd = f"{sys.executable} {script}"
    rep = repair_loop(
        FIXTURE_REPO, DOT, propose_cmd=cmd, max_rounds=5,
        verify_cases=10, work_dir=tmp_path / "repair-work",
    )
    assert rep.status == "success"
    assert rep.rounds_run == 2
    assert rep.rounds[0].body_source == "shell-hook"
    calls = [
        json.loads(line)
        for line in (tmp_path / "propose_calls.jsonl").read_text().splitlines()
    ]
    assert [c["round"] for c in calls] == [1, 2]
    # contract: <cmd> <prompt_file> <round> <failures_json>
    assert calls[1]["argv_tail"] == [
        "prompt_round_2.md", "2", "failures_round_2.json"]
    assert calls[0]["failures_len"] == 0
    assert calls[1]["failures_len"] > 0
    assert not calls[0]["prompt_has_counterexamples"]
    assert calls[1]["prompt_has_counterexamples"]


def test_repair_shell_hook_nonzero_exit_is_error(tmp_path):
    script = tmp_path / "boom.py"
    script.write_text("import sys; sys.exit(3)\n", encoding="utf-8")
    rep = repair_loop(
        FIXTURE_REPO, DOT, propose_cmd=f"{sys.executable} {script}",
        verify_cases=10, work_dir=tmp_path / "repair-work",
    )
    assert rep.status == "error"
    assert "exit" in (rep.error or "") and "3" in (rep.error or "")


def test_run_propose_cmd_unwraps_fenced_body(tmp_path):
    script = tmp_path / "fenced.py"
    script.write_text(
        'print("```cpp\\n    return 1;\\n```")\n', encoding="utf-8")
    body = repair.run_propose_cmd(
        f"{sys.executable} {script}", "# prompt", 1, [],
        scratch_dir=tmp_path / "scratch",
    )
    assert body == "return 1;"


def test_run_propose_cmd_empty_stdout_raises(tmp_path):
    script = tmp_path / "empty.py"
    script.write_text('print("")\n', encoding="utf-8")
    with pytest.raises(RepairError):
        repair.run_propose_cmd(
            f"{sys.executable} {script}", "# prompt", 1, [],
            scratch_dir=tmp_path / "scratch",
        )


# ---------------------------------------------------------------------------
# FFM backend
# ---------------------------------------------------------------------------


@needs_jdk22
def test_repair_ffm_backend_wrong_then_correct(tmp_path):
    def provider(prompt, round_no, failures):
        return FFM_DOT_WRONG if round_no == 1 else FFM_DOT_OK

    rep = _run(FFM_REPO, "com.example.FfmKernels.dot", provider, tmp_path,
               backend="ffm")
    assert rep.status == "success"
    assert rep.backend == "ffm"
    assert rep.rounds_run == 2
    assert rep.final_body == FFM_DOT_OK


# Rust backend
# ---------------------------------------------------------------------------


RUST_REPO = Path(__file__).parent / "fixtures" / "rust_kernels"
RUST_DOT_OK = (
    "    let mut s = 0.0;\n"
    "    for i in 0..p_a.len() {\n"
    "        s += p_a[i] * p_b[i];\n"
    "    }\n"
    "    s"
)
RUST_DOT_WRONG = (
    "    let mut s = 0.0;\n"
    "    for i in 0..p_a.len() {\n"
    "        s += p_a[i] * p_b[i];\n"
    "    }\n"
    "    s + 1.0"
)


def _needs_cargo():
    from kernel_port import rust_glue

    return pytest.mark.skipif(
        rust_glue._find_cargo() is None, reason="e2e Rust tests need cargo"
    )


@_needs_cargo()
def test_repair_rust_backend_wrong_then_correct(tmp_path):
    # v0.9 dogfood: repair rejected backend='rust' outright.
    def provider(prompt, round_no, failures):
        assert "Rust" in prompt  # prompt names the right native language
        return RUST_DOT_WRONG if round_no == 1 else RUST_DOT_OK

    rep = _run(RUST_REPO, "com.example.RustKernels.dot", provider, tmp_path,
               backend="rust")
    assert rep.status == "success"
    assert rep.backend == "rust"
    assert rep.rounds_run == 2


@_needs_cargo()
def test_repair_rejects_unknown_backend(tmp_path):
    def provider(prompt, round_no, failures):
        return RUST_DOT_OK

    with pytest.raises(ValueError, match="unknown backend"):
        _run(RUST_REPO, "com.example.RustKernels.dot", provider, tmp_path,
             backend="go")


# ---------------------------------------------------------------------------
# Python -> C (cffi) mode: lang="python"
# ---------------------------------------------------------------------------

PY_REPO = Path(__file__).parent / "fixtures" / "py_kernels"
PY_DOT = "cffi_kernels.dot"

PY_DOT_C_OK = (
    "double s = 0.0;\n"
    "for (int64_t i = 0; i < p_a_n; i++) s += p_a[i] * p_b[i];\n"
    "return s;"
)
PY_DOT_C_WRONG = (
    "double s = 0.0;\n"
    "for (int64_t i = 0; i < p_a_n; i++) s += p_a[i] * p_b[i];\n"
    "return s + 1.0;"
)


def _run_py(method, provider, tmp_path, **kw):
    kw.setdefault("verify_cases", 8)
    kw.setdefault("work_dir", tmp_path / "repair-work-py")
    kw.setdefault("lang", "python")
    return repair_loop(PY_REPO, method, provider, **kw)


def test_repair_python_wrong_then_correct(tmp_path):
    seen = {}

    def provider(prompt, round_no, failures):
        seen[round_no] = (prompt, list(failures))
        return PY_DOT_C_WRONG if round_no == 1 else PY_DOT_C_OK

    rep = _run_py(PY_DOT, provider, tmp_path)
    assert rep.status == "success"
    assert rep.backend == "cffi"
    assert rep.rounds_run == 2
    assert rep.final_body == PY_DOT_C_OK
    assert rep.last_body is None
    assert rep.rounds[0].failed > 0
    assert rep.rounds[1].failed == 0
    # round-1 prompt is a C-translation brief, not Java
    assert "Python kernel to C" in seen[1][0]
    assert "cffi_kernels.dot" in seen[1][0]
    # round 2 got Python-flavoured counterexamples
    prompt2, failures2 = seen[2]
    assert failures2, "provider got no failure dicts on round 2"
    assert "Counterexamples" in prompt2
    assert "mismatch" in failures2[0]["detail"]
    assert set(failures2[0]) >= {"args", "detail", "boundary", "case"}
    prompt_file = tmp_path / "repair-work-py" / "round-2" / "prompt.md"
    assert "previous attempt" in prompt_file.read_text().lower()


def test_repair_python_shell_hook_end_to_end(tmp_path):
    # Planted wrong C body via --propose-cmd; the hook supplies the fix on
    # round 2 and the loop converges to success.
    script = tmp_path / "propose_py.py"
    script.write_text(
        "import json, sys\n"
        f"WRONG = {PY_DOT_C_WRONG!r}\n"
        f"OK = {PY_DOT_C_OK!r}\n"
        "prompt_file, round_no, failures_file = sys.argv[1], int(sys.argv[2]), sys.argv[3]\n"
        "failures = json.loads(open(failures_file).read())\n"
        "print(OK if (round_no >= 2 and failures) else WRONG)\n",
        encoding="utf-8",
    )
    rep = repair_loop(
        PY_REPO, PY_DOT, propose_cmd=f"{sys.executable} {script}",
        lang="python", max_rounds=5, verify_cases=8,
        work_dir=tmp_path / "repair-work-py",
    )
    assert rep.status == "success"
    assert rep.rounds_run == 2
    assert rep.backend == "cffi"
    assert rep.rounds[0].body_source == "shell-hook"
    assert rep.final_body == PY_DOT_C_OK


def test_repair_python_stalled_on_identical_failures(tmp_path):
    rep = _run_py(PY_DOT, lambda p, r, f: PY_DOT_C_WRONG, tmp_path,
                  max_rounds=5)
    assert rep.status == "stalled"
    assert rep.rounds_run == 2
    assert rep.rounds[0].failure_signature == rep.rounds[1].failure_signature
    assert rep.last_body == PY_DOT_C_WRONG
    assert "oscillation" in (rep.stop_reason or "")


def test_repair_python_oscillation_abab(tmp_path):
    wrong_b = PY_DOT_C_WRONG.replace("s + 1.0", "s + 2.0")

    def provider(prompt, round_no, failures):
        return PY_DOT_C_WRONG if round_no % 2 == 1 else wrong_b

    rep = _run_py(PY_DOT, provider, tmp_path, max_rounds=5)
    assert rep.status == "stalled"
    assert rep.rounds_run == 3
    assert rep.rounds[2].failure_signature == rep.rounds[0].failure_signature
    assert "round 1" in (rep.stop_reason or "")


def test_repair_python_impure_refused(tmp_path):
    rep = _run_py("cffi_kernels.log_sample", lambda p, r, f: "x", tmp_path)
    assert rep.status == "error"
    assert "not pure" in (rep.error or "")
    assert rep.rounds_run == 0
    assert rep.last_body is None


def test_repair_python_unknown_function_is_error(tmp_path):
    rep = _run_py("cffi_kernels.nope", lambda p, r, f: "x", tmp_path)
    assert rep.status == "error"
    assert "not found" in (rep.error or "")


def test_repair_python_unmappable_signature_is_error(tmp_path):
    rep = _run_py("cffi_kernels.poly_plain", lambda p, r, f: "x", tmp_path)
    assert rep.status == "error"
    assert rep.rounds_run == 0


def test_repair_python_forces_cffi_backend(tmp_path):
    rep = _run_py(
        PY_DOT, lambda p, r, f: PY_DOT_C_OK, tmp_path, backend="jni",
    )
    assert rep.status == "success"
    assert rep.backend == "cffi"


def test_repair_python_build_error_round(tmp_path):
    rep = _run_py(PY_DOT, lambda p, r, f: "this is not C {{{", tmp_path,
                  max_rounds=5)
    assert rep.status == "stalled"
    assert rep.rounds_run == 2
    assert rep.rounds[0].build_error, "expected a recorded build error"
    assert "oscillation" in (rep.stop_reason or "")


def test_repair_rejects_unknown_lang(tmp_path):
    with pytest.raises(ValueError, match="unknown lang"):
        _run_py(PY_DOT, lambda p, r, f: "x", tmp_path, lang="go")
