"""Tests for the pure-String-method allowlist in the purity checker."""

import textwrap
from pathlib import Path

import pytest

from kernel_port import java_ast as J
from kernel_port.purity import check_purity


def _repo_of_source(src: str, tmp_path: Path) -> J.JavaRepo:
    pkg = tmp_path / "src" / "p"
    pkg.mkdir(parents=True)
    (pkg / "T.java").write_text(textwrap.dedent(src), encoding="utf-8")
    return J.scan_repo(tmp_path / "src")


def _method(repo: J.JavaRepo, qualname: str):
    m = repo.find_method(qualname)
    assert m is not None, f"method not found: {qualname}"
    return m


def _repo(tmp_path: Path) -> J.JavaRepo:
    return _repo_of_source(
        """\
        package p;
        class S {
            // --- pure: direct allowlisted calls ---
            static int len(String s) { return s.length(); }
            static char first(String s) { return s.charAt(0); }
            static String sub(String s, int a, int b) { return s.substring(a, b); }
            static boolean eq(String s, String t) { return s.equals(t); }
            static boolean blank(String s) { return s.isBlank(); }
            static int hash(String s) { return s.hashCode(); }
            static char[] chars(String s) { return s.toCharArray(); }
            // --- pure: chains of allowlisted calls ---
            static String norm(String s) { return s.trim().toLowerCase(); }
            static int chain3(String s) { return s.trim().strip().length(); }
            static String cat(String s, String t) { return s.concat(t).toUpperCase(); }
            // --- pure: String-typed locals / params ---
            static int localLen(String s) { String x = s.trim(); return x.length(); }
            static int finLen(String s) { final String x = s; return x.length(); }
            static int reassign(String s, String t) {
                String x = s;
                x = t;
                return x.length();
            }
            static int totalLen(String[] arr) {
                int n = 0;
                for (String w : arr) { n += w.length(); }
                return n;
            }
            static int qlen(java.lang.String s) { return s.length(); }
            static boolean mixed(String s, int i) {
                return s.length() > i && s.charAt(i) == 'x';
            }
            static int paren(String s) { return (s).length(); }
            // --- impure: non-allowlisted String methods ---
            static byte[] gb(String s) { return s.getBytes(); }
            static String[] sp(String s) { return s.split(","); }
            static String in(String s) { return s.intern(); }
            static boolean mt(String s) { return s.matches("[a-z]+"); }
            static String ra(String s) { return s.replaceAll("a", "b"); }
            static String fr(String s) { return s.frobnicate(); }
            static String chainBad(String s) { return s.trim().split(",")[0]; }
            // --- impure: allowlisted call with an impure argument ---
            static String subBad(String s, Box b) { return s.substring(0, b.length()); }
            // --- non-String receiver: existing rules apply, allowlist ignored ---
            static int boxLen(Box b) { return b.length(); }
        }
        class Box {
            int n;
            int length() { return this.n; }
        }
        """,
        tmp_path,
    )


@pytest.mark.parametrize(
    "qualname",
    [
        "p.S.len",
        "p.S.first",
        "p.S.sub",
        "p.S.eq",
        "p.S.blank",
        "p.S.hash",
        "p.S.chars",
        "p.S.norm",
        "p.S.chain3",
        "p.S.cat",
        "p.S.localLen",
        "p.S.finLen",
        "p.S.reassign",
        "p.S.totalLen",
        "p.S.qlen",
        "p.S.mixed",
        "p.S.paren",
    ],
)
def test_allowlisted_string_calls_are_pure(tmp_path, qualname):
    repo = _repo(tmp_path)
    rep = check_purity(_method(repo, qualname), repo)
    assert rep.pure, f"{qualname} should be pure, got: {rep.reasons}"


@pytest.mark.parametrize(
    "qualname, fragment",
    [
        ("p.S.gb", "getBytes"),
        ("p.S.sp", "split"),
        ("p.S.in", "intern"),
        ("p.S.mt", "matches"),
        ("p.S.ra", "replaceAll"),
        ("p.S.fr", "frobnicate"),
        ("p.S.chainBad", "split"),
    ],
)
def test_non_allowlisted_string_calls_are_impure(tmp_path, qualname, fragment):
    repo = _repo(tmp_path)
    rep = check_purity(_method(repo, qualname), repo)
    assert not rep.pure, f"{qualname} should be impure"
    blob = "\n".join(rep.reasons)
    assert fragment in blob, f"reason should name '{fragment}', got: {rep.reasons}"
    assert "allowlist" in blob, f"reason should mention the allowlist, got: {rep.reasons}"


def test_impure_argument_to_allowlisted_call(tmp_path):
    repo = _repo(tmp_path)
    rep = check_purity(_method(repo, "p.S.subBad"), repo)
    assert not rep.pure
    blob = "\n".join(rep.reasons)
    assert "b.length" in blob, f"arg impurity should surface, got: {rep.reasons}"


def test_allowlisted_name_on_non_string_receiver_uses_existing_rules(tmp_path):
    """A custom class with a length() method must NOT be touched by the
    String allowlist: the call stays under the existing conservative rules."""
    repo = _repo(tmp_path)
    rep = check_purity(_method(repo, "p.S.boxLen"), repo)
    assert not rep.pure
    blob = "\n".join(rep.reasons)
    assert "b.length" in blob
    assert "allowlist" not in blob, f"String allowlist must not apply, got: {rep.reasons}"


def test_repo_defined_string_class_shadows_java_lang_string(tmp_path):
    """When the repo defines its own class named String, a bare `String`
    declared type is ambiguous and must stay conservative."""
    repo = _repo_of_source(
        """\
        package p;
        class Shadow {
            static int k(String s) { return s.length(); }
        }
        class String {
            int v;
            int length() { return this.v; }
        }
        """,
        tmp_path,
    )
    rep = check_purity(_method(repo, "p.Shadow.k"), repo)
    assert not rep.pure, "shadowed String must stay conservative"
    blob = "\n".join(rep.reasons)
    assert "not provably pure" in blob
    assert "allowlist" not in blob
