"""Tests for `overloading.parse_method_ref`: overload resolution."""

from pathlib import Path

import pytest

from kernel_port import java_ast as J
from kernel_port.overloading import AmbiguityError, parse_method_ref

FIXTURE = Path(__file__).parent / "fixtures" / "overload_kernels"


@pytest.fixture()
def repo():
    return J.scan_repo(FIXTURE)


def types_of(m: J.MethodInfo) -> list[str]:
    return [p.jtype for p in m.params]


def test_bare_unique_name_resolves(repo):
    m = parse_method_ref(repo, "com.example.Overloads.only")
    assert m.name == "only"
    assert types_of(m) == ["int"]


def test_bare_overloaded_name_raises_with_all_candidates(repo):
    with pytest.raises(AmbiguityError) as exc:
        parse_method_ref(repo, "com.example.Overloads.add")
    lines = [ln.strip() for ln in str(exc.value).splitlines()]
    assert "com.example.Overloads.add(int,int)" in lines
    assert "com.example.Overloads.add(double,double)" in lines
    assert "com.example.Overloads.add(int,int,int)" in lines
    assert "(types)" in str(exc.value)  # hint to disambiguate
    assert exc.value.ref == "com.example.Overloads.add"
    assert len(exc.value.candidates) == 3


def test_suffix_exact_match_int_int(repo):
    m = parse_method_ref(repo, "com.example.Overloads.add(int,int)")
    assert types_of(m) == ["int", "int"]
    assert m.jtype == "int"


def test_suffix_exact_match_double_double(repo):
    m = parse_method_ref(repo, "com.example.Overloads.add(double,double)")
    assert types_of(m) == ["double", "double"]


def test_suffix_exact_match_arity3(repo):
    m = parse_method_ref(repo, "com.example.Overloads.add(int,int,int)")
    assert types_of(m) == ["int", "int", "int"]


def test_suffix_ambiguous_no_exact_match_two_candidates(repo):
    # add(int,double) matches nothing exactly; arity 2 has two overloads.
    with pytest.raises(AmbiguityError) as exc:
        parse_method_ref(repo, "com.example.Overloads.add(int,double)")
    lines = [ln.strip() for ln in str(exc.value).splitlines()]
    assert "com.example.Overloads.add(int,int)" in lines
    assert "com.example.Overloads.add(double,double)" in lines
    assert "com.example.Overloads.add(int,int,int)" not in lines


def test_suffix_single_arity_fallback_when_types_differ(repo):
    # No overload of 'only' takes long; arity 1 is unique, so pick it.
    m = parse_method_ref(repo, "com.example.Overloads.only(long)")
    assert m.name == "only"
    assert types_of(m) == ["int"]


def test_suffix_wrong_arity_reports_available_arities(repo):
    with pytest.raises(LookupError) as exc:
        parse_method_ref(repo, "com.example.Overloads.add(int)")
    msg = str(exc.value)
    assert "1 parameter" in msg
    assert "available arities: 2, 3" in msg


def test_suffix_empty_parens_means_zero_params(repo):
    with pytest.raises(LookupError) as exc:
        parse_method_ref(repo, "com.example.Overloads.add()")
    assert "takes 0 parameter" in str(exc.value)


def test_string_vs_java_lang_string_equivalence(repo):
    m = parse_method_ref(
        repo, "com.example.Overloads.concat(java.lang.String,String)"
    )
    assert m.name == "concat"
    assert types_of(m) == ["String", "String"]


def test_suffix_whitespace_is_normalized(repo):
    m = parse_method_ref(repo, "com.example.Overloads.add( int ,  int )")
    assert types_of(m) == ["int", "int"]


def test_unknown_name_raises_with_similar_suggestions(repo):
    with pytest.raises(LookupError) as exc:
        parse_method_ref(repo, "com.example.Overloads.ade")
    msg = str(exc.value)
    assert "method not found" in msg
    assert "Did you mean" in msg
    assert "com.example.Overloads.add" in msg or "com.example.Overloads.only" in msg
