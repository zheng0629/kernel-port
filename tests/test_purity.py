"""Tests for the conservative purity checker."""

import textwrap
from pathlib import Path

import pytest

from kernel_port import java_ast as J
from kernel_port.purity import check_purity

from .conftest import FIXTURE_REPO


def _repo_of_source(src: str, tmp_path: Path) -> J.JavaRepo:
    pkg = tmp_path / "src" / "p"
    pkg.mkdir(parents=True)
    (pkg / "T.java").write_text(textwrap.dedent(src), encoding="utf-8")
    return J.scan_repo(tmp_path / "src")


def _method(repo: J.JavaRepo, qualname: str):
    m = repo.find_method(qualname)
    assert m is not None, f"method not found: {qualname}"
    return m


@pytest.fixture()
def repo():
    return J.scan_repo(FIXTURE_REPO)


# -- fixture expectations -------------------------------------------------


@pytest.mark.parametrize(
    "qualname",
    [
        "com.example.Kernels.dot",
        "com.example.Kernels.square",
        "com.example.Kernels.norm",
        "com.example.Kernels.matMul",
        "com.example.Kernels.sieve",
        "com.example.Kernels.fib",
    ],
)
def test_fixture_pure_methods(repo, qualname):
    rep = check_purity(_method(repo, qualname), repo)
    assert rep.pure, f"{qualname} should be pure, got: {rep.reasons}"


def test_fixture_noisy_is_impure_with_reasons(repo):
    rep = check_purity(_method(repo, "com.example.Kernels.noisy"), repo)
    assert not rep.pure
    blob = "\n".join(rep.reasons)
    assert "System.out.println" in blob
    assert "callCount" in blob
    assert "System.currentTimeMillis" in blob


def test_app_main_is_impure(repo):
    rep = check_purity(_method(repo, "com.example.App.main"), repo)
    assert not rep.pure


# -- conservative rejections (false negatives are OK, false positives are not)


def _check(tmp_path: Path, body: str) -> list[str]:
    src = f"""package p;
public class T {{
    static double m(double x) {{
{body}
    }}
}}
"""
    repo = _repo_of_source(src, tmp_path)
    rep = check_purity(_method(repo, "p.T.m"), repo)
    assert not rep.pure
    return rep.reasons


def test_math_random_rejected(tmp_path):
    reasons = _check(tmp_path, "        return Math.random();\n")
    assert any("Math.random" in r for r in reasons)


def test_new_object_rejected(tmp_path):
    reasons = _check(tmp_path, "        java.util.Random r = new java.util.Random();\n        return r.nextDouble();\n")
    assert any("creates object" in r for r in reasons)


def test_synchronized_rejected(tmp_path):
    reasons = _check(tmp_path, "        synchronized (T.class) { return x; }\n")
    assert any("synchronized" in r for r in reasons)


def test_throw_rejected(tmp_path):
    reasons = _check(tmp_path, '        if (x < 0) throw new IllegalArgumentException("neg");\n        return x;\n')
    assert any("throw" in r for r in reasons)


def test_lambda_rejected(tmp_path):
    reasons = _check(tmp_path, "        java.util.function.DoubleUnaryOperator f = v -> v * 2;\n        return f.applyAsDouble(x);\n")
    assert any("lambda" in r for r in reasons)


def test_mutable_static_field_read_rejected(tmp_path):
    src = """package p;
public class T {
    static int counter = 0;
    static int m(int x) { return x + counter; }
}
"""
    repo = _repo_of_source(src, tmp_path)
    rep = check_purity(_method(repo, "p.T.m"), repo)
    assert not rep.pure
    assert any("counter" in r for r in rep.reasons)


def test_static_final_constant_allowed(tmp_path):
    src = """package p;
public class T {
    static final double K = 2.5;
    static double m(double x) { return x * K + Math.sqrt(x); }
}
"""
    repo = _repo_of_source(src, tmp_path)
    rep = check_purity(_method(repo, "p.T.m"), repo)
    assert rep.pure, rep.reasons


def test_recursion_allowed(tmp_path):
    src = """package p;
public class T {
    static long fact(int n) { return n <= 1 ? 1 : n * fact(n - 1); }
}
"""
    repo = _repo_of_source(src, tmp_path)
    rep = check_purity(_method(repo, "p.T.fact"), repo)
    assert rep.pure, rep.reasons


def test_impure_callee_poisoning(tmp_path):
    src = """package p;
public class T {
    static int helper(int x) { System.out.println(x); return x; }
    static int m(int x) { return helper(x) * 2; }
}
"""
    repo = _repo_of_source(src, tmp_path)
    rep = check_purity(_method(repo, "p.T.m"), repo)
    assert not rep.pure
    assert any("helper" in r for r in rep.reasons)
