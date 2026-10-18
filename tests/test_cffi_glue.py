"""Tests for the cffi glue backend (`cffi_glue`: type mapping + codegen)."""

from pathlib import Path

import pytest

from kernel_port import cffi_glue, py_ast
from kernel_port.cffi_glue import CffiGlueError

FIX = Path(__file__).parent / "fixtures" / "py_kernels"


@pytest.fixture()
def pyrepo():
    return py_ast.scan_repo(FIX)


def _fn(pyrepo, qualname):
    return pyrepo.functions[qualname]


# ---------------------------------------------------------------------------
# type mapping
# ---------------------------------------------------------------------------


def test_scalar_mapping(pyrepo):
    sig = cffi_glue.map_signature(_fn(pyrepo, "cffi_kernels.count_upper"))
    assert sig.cname == "kp_count_upper"
    assert [(p.kind, p.ctype) for p in sig.params] == [("str", "const char*")]
    assert (sig.ret.kind, sig.ret.ctype) == ("int", "int64_t")


def test_list_mapping(pyrepo):
    sig = cffi_glue.map_signature(_fn(pyrepo, "cffi_kernels.dot"))
    assert [(p.kind, p.ctype, p.is_list) for p in sig.params] == [
        ("list_float", "double", True),
        ("list_float", "double", True),
    ]
    assert (sig.ret.kind, sig.ret.ctype) == ("float", "double")


def test_bool_mapping(pyrepo):
    sig = cffi_glue.map_signature(_fn(pyrepo, "cffi_kernels.toggle"))
    assert [(p.kind, p.ctype) for p in sig.params] == [("bool", "int")]
    assert (sig.ret.kind, sig.ret.ctype) == ("bool", "int")


def test_int_mapping(pyrepo):
    sig = cffi_glue.map_signature(_fn(pyrepo, "cffi_kernels.fib"))
    assert [(p.kind, p.ctype) for p in sig.params] == [("int", "int64_t")]
    assert (sig.ret.kind, sig.ret.ctype) == ("int", "int64_t")


def test_void_return_mapping(pyrepo):
    sig = cffi_glue.map_signature(_fn(pyrepo, "cffi_kernels.matvec"))
    assert sig.ret.kind == "void"
    assert sig.ret.ctype == "void"


def test_unannotated_params_error_names_them(pyrepo):
    with pytest.raises(CffiGlueError) as ei:
        cffi_glue.map_signature(_fn(pyrepo, "cffi_kernels.poly_plain"))
    msg = str(ei.value)
    assert "poly_plain" in msg
    assert "x" in msg and "coeffs" in msg  # each offending param named
    assert "Annotate" in msg


def _synthetic_fn(name, params, ret):
    """Build a PyFunction without parsing (for shapes the fixtures lack)."""
    return py_ast.PyFunction(
        qualname=f"syn.{name}",
        module="syn",
        name=name,
        file="syn.py",
        lineno=1,
        end_lineno=5,
        source="",
        params=[py_ast.PyParam(name=n, annotation=a, kind=k) for n, a, k in params],
        return_annotation=ret,
    )


def test_varargs_rejected():
    fn = _synthetic_fn("v", [("args", None, "vararg")], "float")
    with pytest.raises(CffiGlueError, match=r"\*args"):
        cffi_glue.map_signature(fn)


def test_kwargs_rejected():
    fn = _synthetic_fn("v", [("kw", None, "kwarg")], "float")
    with pytest.raises(CffiGlueError, match=r"\*\*kw"):
        cffi_glue.map_signature(fn)


def test_unsupported_annotation_rejected():
    fn = _synthetic_fn("v", [("d", "dict[str, int]", "pos")], "float")
    with pytest.raises(CffiGlueError, match="unsupported annotation"):
        cffi_glue.map_signature(fn)


def test_bare_list_rejected():
    fn = _synthetic_fn("v", [("xs", "list", "pos")], "float")
    with pytest.raises(CffiGlueError, match="unsupported annotation"):
        cffi_glue.map_signature(fn)


def test_list_return_rejected():
    fn = _synthetic_fn("v", [("x", "float", "pos")], "list[float]")
    with pytest.raises(CffiGlueError, match="list returns are not supported"):
        cffi_glue.map_signature(fn)


def test_typing_list_alias_accepted():
    fn = _synthetic_fn("v", [("xs", "List[int]", "pos")], "int")
    sig = cffi_glue.map_signature(fn)
    assert sig.params[0].kind == "list_int"


# ---------------------------------------------------------------------------
# codegen
# ---------------------------------------------------------------------------


def _c_body(cname: str) -> str:
    return f"    /* test body for {cname} */\n    return 0.0;"


def test_generate_writes_c_and_py(pyrepo, tmp_path):
    # tmp_path is fine here: no compilation happens in this module's tests.
    fn = _fn(pyrepo, "cffi_kernels.dot")
    files = cffi_glue.generate([fn], tmp_path)
    c_text = files["c"].read_text(encoding="utf-8")
    py_text = files["py"].read_text(encoding="utf-8")
    assert files["c"].name == "kernels_cffi.c"
    assert files["py"].name == "kernels_cffi.py"
    # header documents the type mapping + out-param convention
    assert "TYPE MAPPING" in c_text
    assert "OUT-PARAM CONVENTION" in c_text
    assert "#include <stdint.h>" in c_text
    # export + marker region
    assert "double kp_dot(double* p_a, int64_t p_a_n, double* p_b, int64_t p_b_n)" in c_text
    assert "BEGIN TRANSLATED BODY (agent fills this)" in c_text
    # binding module mirrors the cdef and loads the .so next to itself
    assert "double kp_dot(double* p_a, int64_t p_a_n, double* p_b, int64_t p_b_n);" in py_text
    assert "libkernels_cffi.so" in py_text
    assert '"dot": "kp_dot"' in py_text


def test_generate_multiple_functions_one_unit(pyrepo, tmp_path):
    fns = [_fn(pyrepo, q) for q in ("cffi_kernels.dot", "cffi_kernels.count_upper")]
    files = cffi_glue.generate(fns, tmp_path)
    c_text = files["c"].read_text(encoding="utf-8")
    assert "kp_dot" in c_text and "kp_count_upper" in c_text
    assert c_text.count("BEGIN TRANSLATED BODY") == 2


def test_export_name_collision_rejected(pyrepo, tmp_path):
    fn = _fn(pyrepo, "cffi_kernels.dot")
    clone = _synthetic_fn("dot", [("a", "list[float]", "pos")], "float")
    clone.qualname = "other.dot"
    with pytest.raises(CffiGlueError, match="collision"):
        cffi_glue.generate([fn, clone], tmp_path)


def test_fill_translation_splices_body(pyrepo, tmp_path):
    fn = _fn(pyrepo, "cffi_kernels.dot")
    files = cffi_glue.generate([fn], tmp_path)
    body = "    double s = 1.0;\n    return s;"
    cffi_glue.fill_translation(files["c"], "kp_dot", body)
    c_text = files["c"].read_text(encoding="utf-8")
    assert body in c_text
    assert "PLACEHOLDER" not in c_text
    assert "BEGIN TRANSLATED BODY" in c_text and "END TRANSLATED BODY" in c_text


def test_name_sanitization():
    fn = _synthetic_fn("weird-name", [("x", "float", "pos")], "float")
    sig = cffi_glue.map_signature(fn)
    assert sig.cname == "kp_weird_name"
    assert sig.params[0].cname == "p_x"
