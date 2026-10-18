"""Tests for v0.10 workstream A: Python purity false-negative fixes.

Covers, against a synthetic multi-module repo built in tmp_path:

1. `from pkg.mod import func` intra-repo call resolution (v0.9 dogfood FN #3):
   intra-repo callees are purity-checked normally; only genuinely foreign
   modules keep the "not a pure module" rejection.
2. `*rest: int` parameter parsing + clear variadic rejection (FN #7): the
   true name is parsed (no bogus "unknown name '*rest'"), and variadic
   signatures are rejected with a clear "not portable" reason.
3. Purity allowlist gaps: single-argument ``type()``, the Unicode-defined
   ``str`` predicates (``isdigit``/``isalpha``/``isspace``), non-mutating
   set algebra, and the strict locally-created-set rule for ``add``/
   ``discard``.
"""

import pytest

from kernel_port import py_ast, py_mine


@pytest.fixture()
def repo(tmp_path):
    files = {
        "pkg/__init__.py": "",
        "pkg/mod.py": (
            "def helper(x):\n"
            "    return x * 2\n"
            "\n"
            "def noisy(x):\n"
            "    print(x)\n"
            "    return x\n"
        ),
        "pkg/user.py": (
            "from pkg.mod import helper\n"
            "\n"
            "def use_helper(a):\n"
            "    return helper(a) + 1\n"
        ),
        "pkg/baduser.py": (
            "from pkg.mod import noisy\n"
            "\n"
            "def use_noisy(a):\n"
            "    return noisy(a)\n"
        ),
        "pkg/reluser.py": (
            "from .mod import helper\n"
            "\n"
            "def use_rel(a):\n"
            "    return helper(a) * 2\n"
        ),
        "pkg/missing.py": (
            "from pkg.mod import ghost\n"
            "\n"
            "def use_ghost(a):\n"
            "    return ghost(a)\n"
        ),
        "foreign.py": (
            "from os import getcwd\n"
            "\n"
            "def grab():\n"
            "    return getcwd()\n"
        ),
        "variadic.py": (
            "def unused_rest(*rest: int):\n"
            "    return 1\n"
            "\n"
            "def kw_only(**kw):\n"
            "    return 0\n"
        ),
        "builtinsx.py": (
            "def type_of(x):\n"
            "    return type(x)\n"
            "\n"
            "def type_three(a, b, c):\n"
            "    return type(a, b, c)\n"
        ),
        "strings.py": (
            "def is_digits(s):\n"
            "    return s.isdigit()\n"
            "\n"
            "def is_digits_lit():\n"
            '    return "123".isdigit()\n'
            "\n"
            "def is_alpha_space(s):\n"
            "    return s.isalpha() or s.isspace()\n"
        ),
        "sets.py": (
            "def local_add():\n"
            "    s = {1, 2}\n"
            "    s.add(3)\n"
            "    return len(s)\n"
            "\n"
            "def local_discard():\n"
            "    s = set()\n"
            "    s.discard(9)\n"
            "    return len(s)\n"
            "\n"
            "def param_add(s):\n"
            "    s.add(1)\n"
            "    return len(s)\n"
            "\n"
            "def aliased(p):\n"
            "    s = p\n"
            "    s.add(1)\n"
            "    return len(s)\n"
            "\n"
            "def rebound():\n"
            "    s = {1}\n"
            "    s = {2}\n"
            "    s.add(3)\n"
            "    return len(s)\n"
            "\n"
            "def mkglobal():\n"
            "    global gset\n"
            "    gset = {1}\n"
            "    gset.add(2)\n"
            "    return len(gset)\n"
            "\n"
            "def inter(a, b):\n"
            "    return a.intersection(b)\n"
            "\n"
            "def union_sub(a):\n"
            "    u = a.union({1})\n"
            "    return u.issubset({1, 2})\n"
        ),
    }
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return py_ast.scan_repo(tmp_path)


@pytest.fixture()
def reports(repo):
    return py_mine.check_purity_repo(repo)


def _pure(reports, qualname):
    rep = reports[qualname]
    assert rep.pure, f"{qualname} should be pure: {rep.reasons}"
    return rep


def _impure(reports, qualname):
    rep = reports[qualname]
    assert not rep.pure, f"{qualname} should be impure"
    return rep


# -- 1. from-import intra-repo resolution ------------------------------------


def test_from_import_intra_repo_pure_callee(reports):
    _pure(reports, "pkg.user.use_helper")


def test_from_import_foreign_module_still_impure(reports):
    rep = _impure(reports, "foreign.grab")
    assert any("not a pure module" in r for r in rep.reasons), rep.reasons


def test_from_import_intra_repo_impure_callee_names_it(reports):
    rep = _impure(reports, "pkg.baduser.use_noisy")
    assert any("pkg.mod.noisy" in r for r in rep.reasons), rep.reasons


def test_from_import_missing_member_clear_reason(reports):
    rep = _impure(reports, "pkg.missing.use_ghost")
    assert any("has no top-level function 'ghost'" in r for r in rep.reasons), (
        rep.reasons
    )


def test_relative_from_import_resolves_intra_repo(reports, repo):
    # `from .mod import helper` must resolve against the importing
    # package (pkg.mod), never misattribute to an unrelated module.
    assert repo.modules["pkg.reluser"].imports["helper"] == "pkg.mod.helper"
    _pure(reports, "pkg.reluser.use_rel")


# -- 2. *args/**kwargs parsing and rejection ---------------------------------


def test_vararg_param_name_parsed_without_star(repo):
    fns = {f.qualname: f for f in repo.top_level_functions()}
    p = fns["variadic.unused_rest"].params[0]
    assert (p.name, p.kind) == ("rest", "vararg")
    assert fns["variadic.unused_rest"].signature == "unused_rest(*rest: int)"
    q = fns["variadic.kw_only"].params[0]
    assert (q.name, q.kind) == ("kw", "kwarg")


def test_unused_vararg_rejected_with_clear_reason(reports):
    rep = _impure(reports, "variadic.unused_rest")
    assert any("variadic" in r for r in rep.reasons), rep.reasons
    assert not any("unknown name" in r for r in rep.reasons), rep.reasons


def test_kwargs_rejected_with_clear_reason(reports):
    rep = _impure(reports, "variadic.kw_only")
    assert any("variadic" in r and "**kw" in r for r in rep.reasons), rep.reasons


# -- 3a. type() ---------------------------------------------------------------


def test_type_single_arg_pure(reports):
    _pure(reports, "builtinsx.type_of")


def test_type_multi_arg_impure(reports):
    rep = _impure(reports, "builtinsx.type_three")
    assert any("type" in r and "single-argument" in r for r in rep.reasons), (
        rep.reasons
    )


# -- 3b. str predicates --------------------------------------------------------


def test_str_predicates_pure(reports):
    _pure(reports, "strings.is_digits")  # param receiver
    _pure(reports, "strings.is_digits_lit")  # string-literal receiver
    _pure(reports, "strings.is_alpha_space")


# -- 3c. set methods ------------------------------------------------------------


def test_local_set_add_discard_pure(reports):
    _pure(reports, "sets.local_add")
    _pure(reports, "sets.local_discard")


def test_param_set_add_impure(reports):
    rep = _impure(reports, "sets.param_add")
    assert any("locally-created set" in r for r in rep.reasons), rep.reasons


def test_aliased_or_rebound_set_add_impure(reports):
    # `s = p` aliases the parameter: the proof of local creation is void.
    rep = _impure(reports, "sets.aliased")
    assert any("locally-created set" in r for r in rep.reasons), rep.reasons
    # Bound twice: no longer provably the original fresh set.
    rep = _impure(reports, "sets.rebound")
    assert any("locally-created set" in r for r in rep.reasons), rep.reasons


def test_global_set_add_impure(reports):
    rep = _impure(reports, "sets.mkglobal")
    assert rep.reasons, "must carry a stated reason"


def test_set_algebra_methods_pure(reports):
    _pure(reports, "sets.inter")
    _pure(reports, "sets.union_sub")
