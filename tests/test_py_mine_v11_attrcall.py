"""Tests for v0.11 workstream 3: intra-repo attribute-style call resolution.

Sibling gap to the v0.10 from-import fix: ``import pkg.mod`` followed by
``pkg.mod.func(...)`` -- or ``import pkg.mod as pm`` followed by
``pm.func(...)`` -- where ``pkg.mod`` is one of the scanned repo's own
modules. These used to be rejected with a stated-reason false negative
("module 'pkg' is not a pure module" / "computed object"); now the member
resolves as a top-level function of that module and is purity-checked
normally. Truly foreign modules (``import os; os.system(...)``) keep the
conservative rejection, unchanged.
"""

import pytest

from kernel_port import py_ast, py_mine


@pytest.fixture()
def repo(tmp_path):
    files = {
        "pkg/__init__.py": "",
        "pkg/mod.py": (
            "def pure_fn(x):\n"
            "    return x * 2\n"
            "\n"
            "def impure_fn(x):\n"
            "    print(x)\n"
            "    return x\n"
            "\n"
            "class Widget:\n"
            "    pass\n"
        ),
        "pkg/caller.py": (
            "import pkg.mod\n"
            "import pkg.mod as pm\n"
            "import math\n"
            "import os\n"
            "\n"
            "def via_chain(a):\n"
            "    return pkg.mod.pure_fn(a)\n"
            "\n"
            "def via_chain_bad(a):\n"
            "    return pkg.mod.impure_fn(a)\n"
            "\n"
            "def via_chain_missing(a):\n"
            "    return pkg.mod.ghost(a)\n"
            "\n"
            "def via_chain_klass(a):\n"
            "    return pkg.mod.Widget(a)\n"
            "\n"
            "def via_alias(a):\n"
            "    return pm.pure_fn(a)\n"
            "\n"
            "def via_alias_bad(a):\n"
            "    return pm.impure_fn(a)\n"
            "\n"
            "def via_math(x):\n"
            "    return math.sqrt(x)\n"
            "\n"
            "def via_os(cmd):\n"
            "    return os.system(cmd)\n"
            "\n"
            "def shadowed(pkg):\n"
            "    return pkg.mod.pure_fn(1)\n"
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


# -- dotted-chain receiver: `import pkg.mod; pkg.mod.func(...)` ----------------


def test_attr_chain_intra_repo_pure_callee(reports):
    _pure(reports, "pkg.caller.via_chain")


def test_attr_chain_intra_repo_impure_callee_names_it(reports):
    rep = _impure(reports, "pkg.caller.via_chain_bad")
    assert any("pkg.mod.impure_fn" in r for r in rep.reasons), rep.reasons


def test_attr_chain_missing_member_clear_reason(reports):
    rep = _impure(reports, "pkg.caller.via_chain_missing")
    assert any("has no top-level function 'ghost'" in r for r in rep.reasons), (
        rep.reasons
    )


def test_attr_chain_non_function_member_not_guessed(reports):
    # A class is not a top-level function: do not guess, stay impure.
    rep = _impure(reports, "pkg.caller.via_chain_klass")
    assert any("has no top-level function 'Widget'" in r for r in rep.reasons), (
        rep.reasons
    )


# -- alias receiver: `import pkg.mod as pm; pm.func(...)` ----------------------


def test_alias_intra_repo_pure_callee(reports):
    _pure(reports, "pkg.caller.via_alias")


def test_alias_intra_repo_impure_callee_names_it(reports):
    rep = _impure(reports, "pkg.caller.via_alias_bad")
    assert any("pkg.mod.impure_fn" in r for r in rep.reasons), rep.reasons


# -- foreign / stdlib behavior unchanged ----------------------------------------


def test_foreign_module_attr_call_still_rejected(reports):
    rep = _impure(reports, "pkg.caller.via_os")
    assert any("not a pure module" in r for r in rep.reasons), rep.reasons


def test_stdlib_pure_module_attr_call_still_pure(reports):
    _pure(reports, "pkg.caller.via_math")


def test_shadowed_chain_root_not_treated_as_module(reports):
    # `pkg` is a parameter here: the chain must not resolve to the module.
    rep = _impure(reports, "pkg.caller.shadowed")
    assert rep.reasons, "must carry a stated reason"
