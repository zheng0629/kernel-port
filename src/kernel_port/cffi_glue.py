"""`glue --lang python`: generate cffi boilerplate for a Python kernel.

Given a pure *module-level* Python function (the same population
:mod:`kernel_port.py_mine` mines), generates into ``--out``:

- ``kernels_cffi.c``: one plain-C-ABI export ``kp_<funcname>`` per kernel,
  with a marked ``BEGIN TRANSLATED BODY (agent fills this)`` region where
  the calling agent pastes the translated C body.
- ``kernels_cffi.py``: a cffi binding module with a matching ``cdef`` that
  loads ``libkernels_cffi.so`` from the same directory (built by
  :mod:`kernel_port.py_harness` via ``gcc -shared -fPIC -O2``).

TYPE MAPPING (also repeated in the generated files' header comments):

==================  =============================  =====================
Python annotation   C parameter type(s)            cffi cdef
==================  =============================  =====================
``int``             ``int64_t``                    ``int64_t``
``float``           ``double``                     ``double``
``bool``            ``int`` (0/1)                  ``int``
``str``             ``const char*`` (UTF-8,        ``const char*``
                    NUL-terminated)
``list[int]``       ``(int64_t* data, int64_t n)`` ``int64_t*, int64_t``
``list[float]``     ``(double* data, int64_t n)``  ``double*, int64_t``
==================  =============================  =====================

Return annotations mirror the mapping, except ``list[...]`` returns which
are rejected (there is no length channel for a returned buffer -- see
below); ``None`` or a missing return annotation means ``void``.

The mapping is deliberately conservative: anything not in the table --
unannotated parameters, ``*args``/``**kwargs``, ``dict``, ``tuple``,
``list`` without an element type, unions, ``list`` *returns* -- raises
:exc:`CffiGlueError` with a clear message. It never guesses a C type.

OUT-PARAM CONVENTION: ``list`` parameters are IN/OUT. The C side receives
a mutable buffer and may write through it; the verify harness echoes
mutated lists back and compares them element-wise against the Python
original's mutated lists (the mirror of the Java harness's array-echo).
Scalar returns compare directly. An empty list is passed as ``NULL``
with ``n == 0`` -- translations must guard on ``n``, not on the pointer.

``str`` parameters arrive as UTF-8, NUL-terminated; embedded NULs are out
of domain (C strings cannot represent them). A ``str`` return is read back
with ``ffi.string`` and decoded as UTF-8.

``bool`` crosses the boundary as ``int`` 0/1; the harness compares
``int(py_bool)`` against the C ``int`` exactly (it does *not* bool() the
C value, so a buggy ``return 2`` is caught).

``int`` is 64-bit two's complement: inputs must be int64-representable
(the harness skips out-of-range inputs as out-of-domain) and the kernel
must be int64-closed on the tested inputs -- Python bignum semantics are
*not* emulated, and overflow mismatches are reported as failures.

Export names are ``kp_<funcname>`` with ``funcname`` sanitized to
``[0-9A-Za-z_]`` (anything else becomes ``_``; a leading digit gets an
extra ``_`` prefix). Parameter names become ``p_<name>``, sanitized the
same way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import py_ast as P


class CffiGlueError(ValueError):
    """The cffi backend cannot map this function; message says why."""


# ---------------------------------------------------------------------------
# signature model
# ---------------------------------------------------------------------------


@dataclass
class ParamMap:
    name: str  # python parameter name
    pytype: str  # normalized annotation, e.g. "list[float]"
    kind: str  # "int" | "float" | "bool" | "str" | "list_int" | "list_float"
    ctype: str  # C type of the value, or element type for lists
    cname: str  # p_<name> (sanitized)
    is_list: bool = False


@dataclass
class RetMap:
    pytype: str  # normalized annotation, "None" for void
    kind: str  # "void" | "int" | "float" | "bool" | "str"
    ctype: str  # "void" | "int64_t" | "double" | "int" | "const char*"


@dataclass
class KernelSig:
    func_name: str  # python function name
    qualname: str
    cname: str  # kp_<sanitized>
    params: list[ParamMap] = field(default_factory=list)
    ret: RetMap | None = None
    file: str = ""  # repo-relative path of the defining module


# normalized annotation -> (kind, c type)
_SCALAR_MAP: dict[str, tuple[str, str]] = {
    "int": ("int", "int64_t"),
    "float": ("float", "double"),
    "bool": ("bool", "int"),
    "str": ("str", "const char*"),
}
# normalized annotation -> (kind, element c type)
_LIST_MAP: dict[str, tuple[str, str]] = {
    "list[int]": ("list_int", "int64_t"),
    "list[float]": ("list_float", "double"),
}
# return annotation -> (kind, c type); "None"/missing handled separately
_RET_MAP: dict[str, tuple[str, str]] = {
    "int": ("int", "int64_t"),
    "float": ("float", "double"),
    "bool": ("bool", "int"),
    "str": ("str", "const char*"),
}

_SUPPORTED = (
    "int, float, bool, str, list[int], list[float] "
    "(return: the same scalars and str, or None)"
)


def _normalize(ann: str | None) -> str:
    if not ann:
        return ""
    s = re.sub(r"\s+", "", ann)
    # typing.List[int] is exactly list[int]; not a guess, just an alias.
    s = re.sub(r"^(typing\.)?List\[(.+)\]$", r"list[\2]", s)
    return s


def _sanitize(name: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if re.match(r"^[0-9]", safe):
        safe = "_" + safe
    return safe or "_"


def map_signature(fn: P.PyFunction) -> KernelSig:
    """Map a Python function signature to C types. Raises CffiGlueError."""
    # *args / **kwargs first: they are never mappable.
    for p in fn.params:
        if p.kind in ("vararg", "kwarg"):
            star = "*" if p.kind == "vararg" else "**"
            raise CffiGlueError(
                f"cffi glue: function '{fn.name}' uses {star}{p.name}; "
                "*args/**kwargs are not supported by the cffi backend."
            )
    missing = [p.name for p in fn.params if not (p.annotation or "").strip()]
    if missing:
        raise CffiGlueError(
            f"cffi glue: function '{fn.name}' has parameters without type "
            f"annotations: {', '.join(missing)}. Annotate every parameter "
            "(e.g. x: float, xs: list[float]) -- the cffi backend never "
            "guesses C types."
        )
    params: list[ParamMap] = []
    for p in fn.params:
        ann = _normalize(p.annotation)
        if ann in _SCALAR_MAP:
            kind, ctype = _SCALAR_MAP[ann]
            params.append(
                ParamMap(
                    name=p.name, pytype=ann, kind=kind,
                    ctype=ctype, cname="p_" + _sanitize(p.name),
                )
            )
        elif ann in _LIST_MAP:
            kind, elem = _LIST_MAP[ann]
            params.append(
                ParamMap(
                    name=p.name, pytype=ann, kind=kind, ctype=elem,
                    cname="p_" + _sanitize(p.name), is_list=True,
                )
            )
        else:
            raise CffiGlueError(
                f"cffi glue: function '{fn.name}': parameter '{p.name}' has "
                f"unsupported annotation '{p.annotation}'. Supported: "
                f"{_SUPPORTED}."
            )
    ret_ann = _normalize(fn.return_annotation)
    if not ret_ann or ret_ann in ("None", "void"):
        ret = RetMap(pytype="None", kind="void", ctype="void")
    elif ret_ann in _RET_MAP:
        kind, ctype = _RET_MAP[ret_ann]
        ret = RetMap(pytype=ret_ann, kind=kind, ctype=ctype)
    elif ret_ann in _LIST_MAP:
        raise CffiGlueError(
            f"cffi glue: function '{fn.name}' returns '{fn.return_annotation}': "
            "list returns are not supported in this slice (no length channel "
            "for a returned buffer). Restructure as an IN/OUT list parameter."
        )
    else:
        raise CffiGlueError(
            f"cffi glue: function '{fn.name}' has unsupported return "
            f"annotation '{fn.return_annotation}'. Supported: {_SUPPORTED}."
        )
    return KernelSig(
        func_name=fn.name,
        qualname=fn.qualname,
        cname="kp_" + _sanitize(fn.name),
        params=params,
        ret=ret,
        file=fn.file,
    )


# ---------------------------------------------------------------------------
# code generation
# ---------------------------------------------------------------------------

_BEGIN = "/* ==== BEGIN TRANSLATED BODY (agent fills this) ==== */"
_END = "/* ==== END TRANSLATED BODY ==== */"
#: Placeholder the agent replaces; py_harness refuses to build while it is
#: still the whole body (clearer than a mysterious all-zero verify).
_PLACEHOLDER = "/* PLACEHOLDER: agent pastes the translated C body here */"

_TYPE_MAPPING_DOC = """\
 * TYPE MAPPING (conservative -- unmappable annotations are a hard error):
 *   Python int         -> C int64_t              (bool -> int, 0/1)
 *   Python float       -> C double
 *   Python str         -> C const char*  (UTF-8, NUL-terminated)
 *   Python list[int]   -> (int64_t* data, int64_t n)
 *   Python list[float] -> (double* data, int64_t n)
 *   return: same scalars/str, or void for None / missing annotation
 *
 * OUT-PARAM CONVENTION: list parameters are IN/OUT -- the C side may write
 * through the buffer; the verify harness echoes mutated lists back and
 * compares them element-wise. Empty list arrives as NULL with n == 0.
 * Scalar returns compare directly. See cffi_glue.py for the full contract.
"""

_C_HEADER = """\
/* kernels_cffi.c -- generated by `kernel-port glue --lang python`.
 * Plain C ABI exports (kp_<name>) for cffi; the calling agent pastes the
 * translated C body into each BEGIN/END TRANSLATED BODY region.
 *
 * Build: gcc -shared -fPIC -O2 -o libkernels_cffi.so kernels_cffi.c -lm
 * (py_harness.build_env does this for you.)
 *
%s */
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <string.h>

""" % _TYPE_MAPPING_DOC

_PY_HEADER = '''\
"""kernels_cffi.py -- generated by `kernel-port glue --lang python`.
cffi bindings for kernels_cffi.c. Loads libkernels_cffi.so from this
directory (build it with `kernel-port verify --lang python`, which runs
`gcc -shared -fPIC -O2` for you). Do not edit by hand.
"""
from pathlib import Path

from cffi import FFI

ffi = FFI()
ffi.cdef("""
%s""")

_SO = Path(__file__).with_name("libkernels_cffi.so")
if not _SO.exists():
    raise RuntimeError(
        f"cffi native library not found: {_SO} -- build it with "
        "`kernel-port verify --lang python ...` (or: "
        "gcc -shared -fPIC -O2 -o libkernels_cffi.so kernels_cffi.c -lm)"
    )
lib = ffi.dlopen(str(_SO))

#: python function name -> C export name
KERNELS = {
%s}
'''


def _default_return(ret: RetMap) -> str:
    return {
        "void": "",
        "int": "    return 0;",
        "float": "    return 0.0;",
        "bool": "    return 0;",
        "str": '    return "";',
    }[ret.kind]


def _c_params(sig: KernelSig) -> str:
    parts: list[str] = []
    for pm in sig.params:
        if pm.is_list:
            parts.append(f"{pm.ctype}* {pm.cname}")
            parts.append(f"int64_t {pm.cname}_n")
        elif pm.kind == "str":
            parts.append(f"const char* {pm.cname}")
        else:
            parts.append(f"{pm.ctype} {pm.cname}")
    return ", ".join(parts) if parts else "void"


def _render_c_function(sig: KernelSig, fn: P.PyFunction) -> str:
    params = _c_params(sig)
    ret_stmt = _default_return(sig.ret)
    body_lines = ["    " + _BEGIN, "    " + _PLACEHOLDER]
    if ret_stmt:
        body_lines.append(ret_stmt)
    body_lines.append("    " + _END)
    body = "\n".join(body_lines)
    pargs = ", ".join(
        p.name + (": " + p.annotation if p.annotation else "") for p in fn.params
    )
    py_sig = fn.name + "(" + pargs + ")"
    if fn.return_annotation:
        py_sig += " -> " + fn.return_annotation
    lines = [
        "/* ---- " + fn.qualname + " ----",
        " * python: " + py_sig,
        " * c:      " + sig.ret.ctype + " " + sig.cname + "(" + params + ")",
        " */",
        sig.ret.ctype + " " + sig.cname + "(" + params + ") {",
        body,
        "}",
    ]
    return "\n".join(lines) + "\n"


def _render_cdef(sig: KernelSig) -> str:
    return f"    {sig.ret.ctype} {sig.cname}({_c_params(sig)});"


def generate(
    funcs: list[P.PyFunction], out_dir: Path | str
) -> dict[str, Path]:
    """Generate ``kernels_cffi.c`` + ``kernels_cffi.py`` for `funcs`.

    Raises :exc:`CffiGlueError` if any signature is unmappable. Returns
    ``{"c": ..., "py": ...}`` paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sigs = [map_signature(fn) for fn in funcs]
    # duplicate C export names (e.g. same basename from two modules) would
    # collide in one shared object -- refuse loudly instead of shadowing.
    seen: dict[str, str] = {}
    for sig in sigs:
        if sig.cname in seen:
            raise CffiGlueError(
                f"cffi glue: export name collision: '{sig.qualname}' and "
                f"'{seen[sig.cname]}' both map to '{sig.cname}'. Glue one "
                "kernel per --out dir in this slice."
            )
        seen[sig.cname] = sig.qualname
    c_path = out / "kernels_cffi.c"
    c_path.write_text(
        _C_HEADER + "\n\n".join(_render_c_function(s, f) for s, f in zip(sigs, funcs)) + "\n",
        encoding="utf-8",
    )
    kernels = "\n".join(f'    "{s.func_name}": "{s.cname}",' for s in sigs)
    py_path = out / "kernels_cffi.py"
    py_path.write_text(
        _PY_HEADER % ("\n".join(_render_cdef(s) for s in sigs), kernels),
        encoding="utf-8",
    )
    return {"c": c_path, "py": py_path}


def fill_translation(c_path: Path | str, cname: str, body: str) -> None:
    """Splice an agent-written `body` into the marked region for `cname`.

    Used by tests (standing in for the calling agent's translation step).
    """
    text = Path(c_path).read_text(encoding="utf-8")
    fn_start = text.index(f"{cname}(")
    b = text.index(_BEGIN, fn_start)
    e = text.index(_END, b) + len(_END)
    Path(c_path).write_text(
        text[:b] + _BEGIN + "\n" + body.rstrip() + "\n    " + text[e:],
        encoding="utf-8",
    )
