"""CLI smoke tests via typer's CliRunner."""

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
    by_name = {c["method"]: c for c in data}
    assert by_name["com.example.Kernels.dot"]["pure"] is True
    assert by_name["com.example.Kernels.noisy"]["pure"] is False


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
