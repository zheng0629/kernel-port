"""CLI smoke tests via typer's CliRunner."""

from pathlib import Path

from typer.testing import CliRunner

from kernel_port.cli import app

from .conftest import FIXTURE_REPO

runner = CliRunner()


def test_cli_mine_table():
    r = runner.invoke(app, ["mine", str(FIXTURE_REPO)])
    assert r.exit_code == 0, r.output
    assert "com.example.Kernels.dot" in r.output
    assert "no" in r.output  # impure rows present


def test_cli_mine_json():
    import json

    r = runner.invoke(app, ["mine", str(FIXTURE_REPO), "--format", "json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    by_name = {c["method"]: c for c in data["methods"]}
    assert by_name["com.example.Kernels.dot"]["pure"] is True
    assert by_name["com.example.Kernels.noisy"]["pure"] is False
    assert "clusters" in data and "classes" in data


def test_cli_extract_text():
    r = runner.invoke(
        app, ["extract", str(FIXTURE_REPO), "--method", "com.example.Kernels.fib"]
    )
    assert r.exit_code == 0, r.output
    assert "PURE" in r.output


def test_cli_extract_impure_exits_2():
    r = runner.invoke(
        app, ["extract", str(FIXTURE_REPO), "--method", "com.example.Kernels.noisy"]
    )
    assert r.exit_code == 2
    assert "IMPURE" in r.output


def test_cli_glue(tmp_path):
    out = tmp_path / "glue"
    r = runner.invoke(
        app,
        ["glue", str(FIXTURE_REPO), "--method", "com.example.Kernels.dot",
         "--out", str(out)],
    )
    assert r.exit_code == 0, r.output
    assert (out / "kernels_jni.cpp").exists()
    assert (out / "KernelsNative.java").exists()


def test_cli_glue_refuses_impure(tmp_path):
    r = runner.invoke(
        app,
        ["glue", str(FIXTURE_REPO), "--method", "com.example.Kernels.noisy",
         "--out", str(tmp_path / "glue")],
    )
    assert r.exit_code == 2


def test_cli_verify_e2e(tmp_path):
    from kernel_port import jni
    from .conftest import TRANSLATIONS, fill_translation

    from kernel_port import java_ast as J

    repo = J.scan_repo(FIXTURE_REPO)
    m = repo.find_method("com.example.Kernels.dot")
    glue_dir = tmp_path / "glue"
    jni.generate(m, glue_dir)
    fill_translation(glue_dir, m.qualname, TRANSLATIONS[m.qualname])
    r = runner.invoke(
        app,
        ["verify", str(FIXTURE_REPO), "--method", m.qualname,
         "--glue-dir", str(glue_dir), "--cases", "10"],
    )
    assert r.exit_code == 0, r.output
    assert "PASS" in r.output


def test_extract_counterexamples_suffix_agreement(tmp_path):
    """v0.9 dogfood: verify persists failures under the *bare* qualname, but
    `extract --with-counterexamples` looked them up under the raw CLI ref, so
    a v0.4 `(types)` suffix desynced the filename and counterexamples were
    silently missed."""
    import json
    from pathlib import Path

    from kernel_port.cli import _failures_path

    overload_repo = str(Path(__file__).parent / "fixtures" / "overload_kernels")
    work = tmp_path / "work"
    # Simulate what harness._persist_failures writes (bare qualname).
    fpath = _failures_path(work, "com.example.Overloads.add")
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fpath.write_text(
        json.dumps(
            [{"args": [1, 2], "java_result": "i:3", "jni_result": "i:4",
              "detail": "3 != 4", "boundary": False}]
        ),
        encoding="utf-8",
    )
    r = runner.invoke(
        app,
        ["extract", overload_repo, "--method", "com.example.Overloads.add(int,int)",
         "--with-counterexamples", "--work", str(work)],
    )
    assert r.exit_code == 0, r.output
    assert "no failure records" not in r.output
    assert "Counterexamples" in r.output


# ---------------------------------------------------------------------------
# Rich markup: user-controlled content containing [...] must survive printing
# (console has markup enabled; unescaped [...] is eaten as a style tag)
# ---------------------------------------------------------------------------


def test_cli_markup_escape_idiom_keeps_bracket_literals():
    """The exact idiom cli.py uses: styling kept, user content escaped."""
    import io

    from rich.console import Console
    from rich.markup import escape

    buf = io.StringIO()
    c = Console(file=buf, force_terminal=False, width=120)
    c.print(f"[red]Error:[/red] {escape('bad Sequence[float] arg [weird]')}")
    out = buf.getvalue()
    assert "Sequence[float]" in out
    assert "[weird]" in out


def test_cli_markup_bracketed_repo_path_survives():
    r = runner.invoke(app, ["mine", "/tmp/[weird]-repo"])
    assert r.exit_code == 1, r.output
    assert "[weird]-repo" in r.output  # would vanish as a style tag


def test_cli_markup_backend_notes_keep_brackets():
    r = runner.invoke(
        app,
        ["extract", str(FIXTURE_REPO), "--method", "com.example.Kernels.fib",
         "--backend", "rust"],
    )
    assert r.exit_code == 0, r.output
    # rust translation notes contain "Slices are passed as &[T] ..."
    assert "&[T]" in r.output


# ---------------------------------------------------------------------------
# repair CLI: --lang python, boundary counts, stop_reason
# ---------------------------------------------------------------------------

_PY_REPO = str(Path(__file__).parent / "fixtures" / "py_kernels")
_PY_WRONG = (
    "double s = 0.0;\n"
    "for (int64_t i = 0; i < p_a_n; i++) s += p_a[i] * p_b[i];\n"
    "return s + 1.0;"
)
_PY_OK = (
    "double s = 0.0;\n"
    "for (int64_t i = 0; i < p_a_n; i++) s += p_a[i] * p_b[i];\n"
    "return s;"
)


def _py_propose_script(tmp_path, wrong, correct):
    import sys

    script = tmp_path / "propose_py.py"
    script.write_text(
        "import json, sys\n"
        f"WRONG = {wrong!r}\n"
        f"CORRECT = {correct!r}\n"
        "prompt_file, round_no, failures_file = sys.argv[1], int(sys.argv[2]), sys.argv[3]\n"
        "failures = json.loads(open(failures_file).read())\n"
        "print(CORRECT if (round_no >= 2 and failures) else WRONG)\n",
        encoding="utf-8",
    )
    return f"{sys.executable} {script}"


def test_cli_repair_lang_python_converges(tmp_path):
    r = runner.invoke(
        app,
        ["repair", _PY_REPO, "--lang", "python", "--method", "cffi_kernels.dot",
         "--propose-cmd", _py_propose_script(tmp_path, _PY_WRONG, _PY_OK),
         "--verify-cases", "8", "--work-dir", str(tmp_path / "w")],
    )
    assert r.exit_code == 0, r.output
    assert "repair status: success" in r.output
    assert "(boundary " in r.output  # round rows carry boundary counts
    assert "Translation repaired and verified." in r.output


def test_cli_repair_lang_python_stalled_shows_stop_reason(tmp_path):
    r = runner.invoke(
        app,
        ["repair", _PY_REPO, "--lang", "python", "--method", "cffi_kernels.dot",
         "--propose-cmd", _py_propose_script(tmp_path, _PY_WRONG, _PY_WRONG),
         "--verify-cases", "8", "--max-rounds", "4",
         "--work-dir", str(tmp_path / "w")],
    )
    assert r.exit_code == 1, r.output
    assert "repair status: stalled" in r.output
    assert "oscillation" in r.output
    assert "(boundary " in r.output


def test_cli_repair_lang_python_rejects_backend(tmp_path):
    r = runner.invoke(
        app,
        ["repair", _PY_REPO, "--lang", "python", "--method", "cffi_kernels.dot",
         "--backend", "ffm", "--propose-cmd", "true",
         "--work-dir", str(tmp_path / "w")],
    )
    assert r.exit_code == 1, r.output
    assert "cffi backend" in r.output


def test_cli_repair_bracketed_path_survives_in_stop_reason(tmp_path):
    r = runner.invoke(
        app,
        ["repair", "/tmp/[weird]-repo", "--lang", "python",
         "--method", "cffi_kernels.dot", "--propose-cmd", "true",
         "--work-dir", str(tmp_path / "w")],
    )
    assert r.exit_code == 1, r.output
    assert "[weird]-repo" in r.output
