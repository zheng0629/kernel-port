"""Tests for Python kernel mining (`py_ast` + `py_mine`, slice 1)."""

from pathlib import Path

import pytest

from kernel_port import py_ast, py_mine

FIX = Path(__file__).parent / "fixtures" / "py_kernels"


@pytest.fixture()
def pyrepo():
    return py_ast.scan_repo(FIX)


@pytest.fixture()
def cands(pyrepo):
    return {c.qualname: c for c in py_mine.mine(pyrepo)}


def test_scan_finds_top_level_functions(pyrepo):
    qns = {f.qualname for f in pyrepo.top_level_functions()}
    assert "numeric.dot" in qns
    assert "numeric.matmul" in qns
    assert "numeric.sieve" in qns
    assert "impure.save_results" in qns
    # nested defs and methods are recorded but never mined
    assert "closures.make_adder.adder" in pyrepo.functions
    assert "closures.make_adder.adder" not in qns


def test_pure_numeric_functions_found(cands):
    for qn in (
        "numeric.dot",
        "numeric.matmul",
        "numeric.sieve",
        "numeric.norm",
        "numeric.fib",
        "numeric.poly",
        "numeric.gsum",
        "numeric.clamp",
        "numeric.hypot2",
        "impure.scaled",
        "closures.multiplies_out",
    ):
        assert cands[qn].pure, f"{qn} should be pure: {cands[qn].reasons}"


def test_genexp_effects_are_checked(cands):
    # v0.9 dogfood (CRITICAL soundness hole): a sole generator expression is
    # a direct child of `call` in tree-sitter-python (no `argument_list`
    # wrapper) and was never walked -- open()/print() inside passed as pure.
    for qn in ("impure.genexp_print", "impure.genexp_open"):
        c = cands[qn]
        assert not c.pure, f"{qn} must be impure"
    # ...but a pure genexp stays pure (loop var is bound, not a free read).
    assert cands["numeric.gsum"].pure, cands["numeric.gsum"].reasons


def test_pure_sorts_before_impure(pyrepo):
    ordered = py_mine.mine(pyrepo)
    pure = [c for c in ordered if c.pure]
    impure = [c for c in ordered if not c.pure]
    assert pure and impure
    first_impure = min(ordered.index(c) for c in impure)
    assert all(ordered.index(c) < first_impure for c in pure)
    assert all(c.score > 0 for c in pure)
    assert all(c.score == 0.0 for c in impure)


def test_impure_file_io(cands):
    c = cands["impure.save_results"]
    assert not c.pure
    assert any("with" in r or "open" in r for r in c.reasons), c.reasons


def test_impure_random(cands):
    c = cands["impure.random_walk"]
    assert not c.pure
    assert any("random" in r for r in c.reasons), c.reasons


def test_impure_global_mutation(cands):
    c = cands["impure.bump"]
    assert not c.pure
    assert any("global" in r and "COUNTER" in r for r in c.reasons), c.reasons


def test_impure_module_level_mutation(cands):
    c = cands["impure.cache_put"]
    assert not c.pure
    assert any("LOG" in r for r in c.reasons), c.reasons


def test_impure_module_level_read(cands):
    c = cands["impure.reads_log"]
    assert not c.pure
    assert any("LOG" in r for r in c.reasons), c.reasons


def test_impure_clock_and_os(cands):
    assert not cands["impure.stamp"].pure
    assert any("time" in r for r in cands["impure.stamp"].reasons)
    assert not cands["impure.env_home"].pure
    assert any("os" in r for r in cands["impure.env_home"].reasons)


def test_impure_decorator(cands):
    c = cands["impure.predict"]
    assert not c.pure
    assert any("decorator" in r for r in c.reasons), c.reasons


def test_impure_dynamic(cands):
    assert not cands["impure.run"].pure
    assert any("eval" in r for r in cands["impure.run"].reasons)
    assert not cands["impure.dynamic"].pure
    assert any("getattr" in r for r in cands["impure.dynamic"].reasons)


def test_impure_raise(cands):
    # v0.9: defensive `raise ValueError(...)` is purity-compatible
    # (deterministic-throw carve-out, mirroring Java v0.4).
    c = cands["impure.raiser"]
    assert c.pure, f"defensive raise should be pure: {c.reasons}"
    # ...but custom exception types, pre-built objects, and bare re-raise
    # are still rejected.
    for qn in ("impure.raiser_custom", "impure.raiser_prebuilt", "impure.raiser_bare"):
        cc = cands[qn]
        assert not cc.pure, f"{qn} should be impure"
        assert any("raise" in r for r in cc.reasons), cc.reasons


def test_nested_function_rejected_with_reason(cands):
    # Documented choice: closures are not portable, so any nested def makes
    # the enclosing function impure -- explicitly, not silently.
    c = cands["closures.make_adder"]
    assert not c.pure
    assert any("nested function" in r for r in c.reasons), c.reasons


def test_lambda_generator_nonlocal_rejected(cands):
    assert not cands["closures.with_lambda"].pure
    assert any("lambda" in r for r in cands["closures.with_lambda"].reasons)
    assert not cands["closures.gen"].pure
    assert any("yield" in r for r in cands["closures.gen"].reasons)
    assert not cands["closures.outer_state"].pure
    assert any("nonlocal" in r or "nested function" in r
               for r in cands["closures.outer_state"].reasons)


def test_call_counts(pyrepo):
    by_name = {c.qualname: c for c in py_mine.mine(pyrepo)}
    assert by_name["numeric.dot"].calls == 2  # norm() and cosine()
    assert by_name["numeric.norm"].calls == 3  # unit() + 2x in cosine()
    assert by_name["numeric.sieve"].calls == 0


def test_ranking_top_is_matmul(pyrepo):
    ordered = py_mine.mine(pyrepo)
    assert ordered[0].qualname == "numeric.matmul"
    by_name = {c.qualname: c for c in ordered}
    # score == (1 + calls) * intensity, per the documented formula
    dot = by_name["numeric.dot"]
    assert dot.calls == 2
    assert dot.score == round((1 + dot.calls) * dot.intensity, 1)
    # ranking is by score descending within the pure block
    pure = [c for c in ordered if c.pure]
    assert [c.score for c in pure] == sorted(
        (c.score for c in pure), reverse=True
    )


def test_unannotated_still_analyzed_but_varargs_rejected(cands):
    # No annotations anywhere: still pure, still ranked.
    assert cands["numeric.poly"].pure
    # ...but *args/**kwargs have no C equivalent: rejected with a clear
    # variadic reason, never a bogus "unknown name" (v0.9 dogfood FN #7).
    c = cands["numeric.vsum"]
    assert not c.pure
    assert any("variadic" in r for r in c.reasons), c.reasons
    assert not any("unknown name" in r for r in c.reasons), c.reasons
    assert cands["numeric.vsum"].signature == "vsum(*args, **kwargs)"


def test_signatures_capture_annotations_defaults(pyrepo):
    fns = {f.qualname: f for f in pyrepo.top_level_functions()}
    assert fns["numeric.typed_dot"].signature == (
        "typed_dot(a: list, b: list) -> float"
    )
    assert "lo" in fns["numeric.clamp"].signature
    assert fns["numeric.clamp"].params[1].default == "-1.0"
    assert fns["numeric.vsum"].params[0].kind == "vararg"
    assert fns["numeric.vsum"].params[1].kind == "kwarg"


def test_from_import_math_alias_pure(cands):
    assert cands["numeric.hypot2"].pure, cands["numeric.hypot2"].reasons


def test_out_param_mutation_allowed_by_convention(cands):
    # Documented choice: mutating a parameter (a[i] = v) is the out-param
    # convention, mirroring Java's array-echo model -- not impurity.
    assert cands["closures.multiplies_out"].pure


def test_check_purity_repo_covers_all_top_level(pyrepo):
    reports = py_mine.check_purity_repo(pyrepo)
    qns = {f.qualname for f in pyrepo.top_level_functions()}
    assert set(reports) == qns
    assert reports["numeric.dot"].pure
    assert not reports["impure.bump"].pure
