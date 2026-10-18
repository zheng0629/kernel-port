"""Tests for `pipeline --lang python` (py_pipeline.run_python_pipeline)."""

from pathlib import Path

import pytest

from kernel_port import py_pipeline

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

MATVEC_C = """\
for (int64_t i = 0; i < p_n; i++) {
    double s = 0.0;
    for (int64_t j = 0; j < p_n; j++) s += p_a[i * p_n + j] * p_x[j];
    p_y[i] = s;
}
"""


def test_awaiting_translation(tmp_path):
    rep = py_pipeline.run_python_pipeline(
        FIXTURES, method="cffi_kernels.dot",
        glue_out=tmp_path / "glue", work_dir=tmp_path / "work",
    )
    assert rep["status"] == "awaiting_translation"
    assert rep["lang"] == "python" and rep["backend"] == "cffi"
    assert (tmp_path / "glue" / "kernels_cffi.c").exists()
    prompt = tmp_path / "work" / "prompt_material.md"
    assert prompt.exists()
    text = prompt.read_text(encoding="utf-8")
    assert "cffi_kernels.dot" in text
    assert "kp_dot" in text  # native signature present for the LLM


def test_success_end_to_end(tmp_path):
    rep = py_pipeline.run_python_pipeline(
        FIXTURES, method="cffi_kernels.dot",
        glue_out=tmp_path / "glue", work_dir=tmp_path / "work",
        translated_body=DOT_C, cases=20, seed=1, iters=200, warmup=20, repeats=2,
    )
    assert rep["status"] == "success", rep.get("instructions")
    assert rep["verify"]["all_passed"]
    assert rep["verify"]["passed"] > 0
    assert rep["bench"]["speedup"] > 0
    assert "differential cases passed" in rep["instructions"]


def test_verify_failed_wrong_translation(tmp_path):
    rep = py_pipeline.run_python_pipeline(
        FIXTURES, method="cffi_kernels.dot",
        glue_out=tmp_path / "glue", work_dir=tmp_path / "work",
        translated_body=DOT_C_WRONG, cases=20, seed=1,
    )
    assert rep["status"] == "verify_failed"
    assert rep["verify"]["failed"] > 0


def test_void_inout_kernel(tmp_path):
    rep = py_pipeline.run_python_pipeline(
        FIXTURES, method="cffi_kernels.matvec",
        glue_out=tmp_path / "glue", work_dir=tmp_path / "work",
        translated_body=MATVEC_C, cases=15, seed=2, iters=100, warmup=10, repeats=2,
    )
    assert rep["status"] == "success", rep.get("instructions")
    assert rep["verify"]["all_passed"]


def test_rejected_impure(tmp_path):
    rep = py_pipeline.run_python_pipeline(
        FIXTURES, method="cffi_kernels.log_sample",
        glue_out=tmp_path / "glue", work_dir=tmp_path / "work",
    )
    assert rep["status"] == "rejected"
    assert "not pure" in rep["instructions"]


def test_unannotated_is_error(tmp_path):
    rep = py_pipeline.run_python_pipeline(
        FIXTURES, method="cffi_kernels.poly_plain",
        glue_out=tmp_path / "glue", work_dir=tmp_path / "work",
    )
    assert rep["status"] == "error"
    assert "annotation" in rep["instructions"].lower()


def test_unknown_function_is_error(tmp_path):
    rep = py_pipeline.run_python_pipeline(
        FIXTURES, method="cffi_kernels.nope",
        glue_out=tmp_path / "glue", work_dir=tmp_path / "work",
    )
    assert rep["status"] == "error"


def test_default_picks_top_candidate(tmp_path):
    rep = py_pipeline.run_python_pipeline(
        FIXTURES, glue_out=tmp_path / "glue", work_dir=tmp_path / "work",
    )
    assert rep["status"] == "awaiting_translation"
    assert rep["method"] is not None
    assert rep["purity"]["pure"]


def test_html_report(tmp_path):
    html = tmp_path / "rep.html"
    rep = py_pipeline.run_python_pipeline(
        FIXTURES, method="cffi_kernels.dot",
        glue_out=tmp_path / "glue", work_dir=tmp_path / "work",
        html=html,
    )
    assert html.exists()
    assert "cffi_kernels.dot" in html.read_text(encoding="utf-8")
    assert rep["html_path"] == str(html)


def test_cli_pipeline_lang_python(tmp_path):
    from typer.testing import CliRunner

    from kernel_port.cli import app

    runner = CliRunner()
    res = runner.invoke(
        app,
        ["pipeline", "--lang", "python", "--method", "cffi_kernels.dot",
         "--glue-out", str(tmp_path / "g"), "--work", str(tmp_path / "w"),
         str(FIXTURES)],
    )
    assert res.exit_code == 0, res.output
    assert "awaiting_translation" in res.output


def test_cli_pipeline_rejects_java_flags(tmp_path):
    from typer.testing import CliRunner

    from kernel_port.cli import app

    runner = CliRunner()
    res = runner.invoke(
        app,
        ["pipeline", "--lang", "python", "--backend", "ffm",
         "--method", "cffi_kernels.dot", str(FIXTURES)],
    )
    assert res.exit_code != 0
    assert "does not support" in res.output


def test_mcp_pipeline_lang_python(tmp_path):
    pytest.importorskip("mcp")
    from kernel_port import mcp_server

    rep = mcp_server.pipeline(
        repo=str(FIXTURES), method="cffi_kernels.dot",
        glue_out=str(tmp_path / "g"), work_dir=str(tmp_path / "w"),
        lang="python",
    )
    assert rep["status"] == "awaiting_translation"
    assert rep["lang"] == "python"
