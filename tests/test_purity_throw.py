"""Tests for the deterministic-throw purity rule (v0.4 item 1).

A `throw` is purity-compatible when it is deterministic and side-effect-free:
`throw new <UncheckedType>(pure args...)` with the type on the allowlist.
Everything else -- pre-built objects, checked types, rethrows, impure args --
is rejected with a stated reason.
"""

import textwrap
from pathlib import Path

import pytest

from kernel_port import java_ast as J
from kernel_port import miner
from kernel_port.purity import check_purity

THROW_FIXTURE = Path(__file__).parent / "fixtures" / "throw_kernels"


def _repo_of_source(src: str, tmp_path: Path) -> J.JavaRepo:
    pkg = tmp_path / "src" / "p"
    pkg.mkdir(parents=True)
    (pkg / "T.java").write_text(textwrap.dedent(src), encoding="utf-8")
    return J.scan_repo(tmp_path / "src")


def _method(repo: J.JavaRepo, qualname: str):
    m = repo.find_method(qualname)
    assert m is not None, f"method not found: {qualname}"
    return m


def _assert_pure(tmp_path: Path, src: str, qualname: str = "p.T.m") -> None:
    repo = _repo_of_source(src, tmp_path)
    rep = check_purity(_method(repo, qualname), repo)
    assert rep.pure, f"{qualname} should be pure, got: {rep.reasons}"


def _assert_impure(tmp_path: Path, src: str, qualname: str = "p.T.m") -> list[str]:
    repo = _repo_of_source(src, tmp_path)
    rep = check_purity(_method(repo, qualname), repo)
    assert not rep.pure, f"{qualname} should be impure"
    return rep.reasons


# -- accepted: validation-style deterministic throws ------------------------


def test_int_validation_throw_accepted(tmp_path):
    _assert_pure(
        tmp_path,
        """package p;
        public class T {
            static long factorial(int n) {
                if (n < 0) throw new IllegalArgumentException("n must be >= 0, got " + n);
                long r = 1;
                for (int i = 2; i <= n; i++) r *= i;
                return r;
            }
        }
        """,
        "p.T.factorial",
    )


def test_double_validation_throw_accepted(tmp_path):
    _assert_pure(
        tmp_path,
        """package p;
        public class T {
            static double inv(double x) {
                if (x == 0.0) throw new ArithmeticException("division by zero");
                return 1.0 / x;
            }
        }
        """,
        "p.T.inv",
    )


def test_qualified_name_and_noarg_throw_accepted(tmp_path):
    _assert_pure(
        tmp_path,
        """package p;
        public class T {
            static int m(int x) {
                if (x < 0) throw new java.lang.IllegalStateException("bad");
                if (x == 0) throw new AssertionError();
                return x;
            }
        }
        """,
    )


def test_throw_with_pure_computed_arg_accepted(tmp_path):
    _assert_pure(
        tmp_path,
        """package p;
        public class T {
            static int at(int[] a, int i) {
                if (i < 0 || i >= a.length)
                    throw new IndexOutOfBoundsException("i=" + i + " len=" + a.length);
                return a[i];
            }
        }
        """,
        "p.T.at",
    )


# -- rejected: everything else ----------------------------------------------


def test_throw_prebuilt_object_rejected(tmp_path):
    reasons = _assert_impure(
        tmp_path,
        """package p;
        public class T {
            static final IllegalArgumentException EX = new IllegalArgumentException("bad");
            static int m(int x) {
                if (x < 0) throw EX;
                return x;
            }
        }
        """,
    )
    assert any("pre-built exception object 'EX'" in r for r in reasons)


def test_rethrow_of_caught_variable_rejected(tmp_path):
    reasons = _assert_impure(
        tmp_path,
        """package p;
        public class T {
            static int m(int x) {
                try {
                    return 10 / x;
                } catch (ArithmeticException e) {
                    throw e;
                }
            }
        }
        """,
    )
    assert any("rethrows caught exception 'e'" in r for r in reasons)


def test_checked_exception_throw_rejected(tmp_path):
    reasons = _assert_impure(
        tmp_path,
        """package p;
        public class T {
            static int m(int x) throws java.io.IOException {
                if (x < 0) throw new java.io.IOException("bad");
                return x;
            }
        }
        """,
    )
    assert any("java.io.IOException" in r for r in reasons)


def test_plain_exception_throw_rejected(tmp_path):
    reasons = _assert_impure(
        tmp_path,
        """package p;
        public class T {
            static int m(int x) throws Exception {
                if (x < 0) throw new Exception("bad");
                return x;
            }
        }
        """,
    )
    assert any("'Exception'" in r for r in reasons)


def test_throw_with_impure_arg_rejected(tmp_path):
    reasons = _assert_impure(
        tmp_path,
        """package p;
        public class T {
            static int m(Obj obj, int x) {
                if (x < 0) throw new IllegalArgumentException(obj.foo());
                return x;
            }
        }
        """,
    )
    assert any("throw argument" in r and "impure" in r for r in reasons)


def test_non_allowlisted_unchecked_throw_rejected(tmp_path):
    # NoSuchElementException is unchecked but NOT on the deterministic
    # allowlist: conservative rejection is the documented behavior.
    reasons = _assert_impure(
        tmp_path,
        """package p;
        public class T {
            static int m(int x) {
                if (x < 0) throw new java.util.NoSuchElementException("bad");
                return x;
            }
        }
        """,
    )
    assert any("NoSuchElementException" in r for r in reasons)


def test_throw_of_method_call_result_rejected(tmp_path):
    reasons = _assert_impure(
        tmp_path,
        """package p;
        public class T {
            static IllegalArgumentException mk() {
                return new IllegalArgumentException("x");
            }
            static int m(int x) {
                if (x < 0) throw mk();
                return x;
            }
        }
        """,
    )
    assert any("non-deterministic exception expression" in r for r in reasons)


# -- fixture: throw_kernels --------------------------------------------------


@pytest.fixture()
def throw_repo():
    return J.scan_repo(THROW_FIXTURE / "src")


def test_fixture_factorial_is_pure(throw_repo):
    rep = check_purity(
        _method(throw_repo, "com.example.ThrowKernels.factorial"), throw_repo
    )
    assert rep.pure, rep.reasons


def test_fixture_sqrtpos_is_pure(throw_repo):
    rep = check_purity(
        _method(throw_repo, "com.example.ThrowKernels.sqrtPos"), throw_repo
    )
    assert rep.pure, rep.reasons


def test_fixture_noisy_is_impure(throw_repo):
    rep = check_purity(
        _method(throw_repo, "com.example.ThrowKernels.noisyFactorial"), throw_repo
    )
    assert not rep.pure
    assert any("calls" in r for r in rep.reasons)


def test_mine_level_accepts_validation_kernels(throw_repo):
    """End-to-end-ish: mine() ranks validation-throw kernels as pure."""
    by_name = {c.qualname: c for c in miner.mine(throw_repo)}
    for qn in (
        "com.example.ThrowKernels.factorial",
        "com.example.ThrowKernels.sqrtPos",
    ):
        cand = by_name[qn]
        assert cand.pure, f"{qn} should be pure: {cand.reasons}"
        assert cand.score > 0, f"{qn} should be suggested for porting"
    noisy = by_name["com.example.ThrowKernels.noisyFactorial"]
    assert not noisy.pure
    assert noisy.score == 0.0
    # non-allowlisted throw type stays impure at mine level too
    assert not by_name["com.example.ThrowKernels.firstPositive"].pure
