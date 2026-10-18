"""Compile-and-run harness for `verify` and `bench`.

Protocol: kernel-port drives two long-lived JVM subprocesses over stdin/stdout
(one calling the Java original, one calling the JNI translation). Each input
line is a command; each output line is a result:

    CALL <arg> <arg> ...          ->  OK <result> | ERR <message>
    BENCH <iters> <arg> <arg> ... ->  OK <elapsed-nanos> | ERR <message>

Argument/result encoding (space-separated tokens):

    i:42  l:123  d:1.5  f:0.5  z:1      scalar int/long/double/float/boolean
    ai:1,2,3  al:..  ad:..  af:..       int[]/long[]/double[]/float[] (ai: = empty)
    a2d:2,2,1.0,2.0,3.0,4.0             double[][] as rows,cols,flat-row-major
    a2i:2,2,1,2,3,4                     int[][] (same shape)
    s:hello\\sworld                     String (\\n \\r \\t \\s \\\\ escapes)
    v                                    void result

Result comparison uses per-type float tolerance (v0.4): float/float[]
compare within tol_float (default 1e-6), double/double[]/double[][] within
tol_double (default 1e-12); everything else compares exactly.

BENCH runs the iteration loop *inside* the JVM (timed with nanoTime) so the
IPC cost is amortized and the numbers reflect the kernel, not the pipe.

Random input generation applies documented heuristics for size-like
parameters (named n/size/len/dim/count): they take the length of the first
array parameter (or its integer square root when every array has a perfect
square length -- the matMul(a,b,c,n) shape). Arrays sharing an element type
share a length, so dot(a,b) style kernels get matching inputs.
"""

from __future__ import annotations

import json
import math
import os
import queue
import random
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import java_ast as J
from . import jni as JNI

# ---------------------------------------------------------------------------
# toolchain discovery
# ---------------------------------------------------------------------------


def _which_or_raise(name: str) -> str:
    p = shutil.which(name)
    if not p:
        raise RuntimeError(
            f"required tool '{name}' not found on PATH "
            "(verify/bench need a JDK and g++)"
        )
    return p


def jni_include_dirs() -> list[Path]:
    """Locate <jni.h> and the platform subdir."""
    import os
    import platform

    cands: list[Path] = []
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        cands.append(Path(java_home) / "include")
    javac = shutil.which("javac")
    if javac:
        cands.append(Path(javac).resolve().parent.parent / "include")
    for base in cands:
        plat = base / {"Linux": "linux", "Darwin": "darwin"}.get(
            platform.system(), "linux"
        )
        if (base / "jni.h").exists() and plat.exists():
            return [base, plat]
    raise RuntimeError(
        "could not locate jni.h (tried JAVA_HOME and javac's parent); "
        "set JAVA_HOME to a JDK"
    )


def kernel_sourcepath_roots(repo_root: Path, method: J.MethodInfo) -> list[Path]:
    """Compute javac ``-sourcepath`` roots for the kernel's file.

    Real-world repos cannot be compiled wholesale: test sources need JUnit,
    some files need annotation-only deps (jspecify, intellij) that are not
    on any classpath. Instead compile just the kernel file and let javac
    pull its transitive dependencies from the sourcepath -- test sources
    are never referenced, so they are never compiled.

    The root is derived from the package: the package directory chain must
    be a suffix of the file's directory, i.e. ``<root>/.../<pkg-dirs>/<File>.java``.
    ``class_qualname`` may include enclosing classes
    (``org.joml.Outer.Inner``), so the longest package prefix matching the
    directory suffix wins.
    """
    repo_root = Path(repo_root)
    kf = repo_root / method.file
    if not kf.exists():
        return [repo_root]
    parent_parts = kf.parent.parts
    parts = method.class_qualname.split(".")
    for drop in range(1, len(parts) + 1):
        pkg_parts = parts[: len(parts) - drop]
        if not pkg_parts:
            # Default package (class_qualname has no dots): the file's own
            # dir is the sourcepath root.
            if len(parts) == 1:
                return [kf.parent]
            continue  # package/dir mismatch -> fallback below
        # The package must be a suffix of the file's directory chain:
        # <root>/.../<pkg-dirs>/<File>.java
        if list(parent_parts[-len(pkg_parts):]) == pkg_parts:
            root = kf.parent
            for _ in pkg_parts:
                root = root.parent
            return [root]
    # package/dir mismatch: best effort -- the file's own dir plus repo root.
    # (Explicitly-listed files compile regardless of layout; only implicit
    # sourcepath resolution may miss here.)
    return [kf.parent, repo_root]


def compile_java(
    sources: list[Path],
    classes_dir: Path,
    sourcepath: list[Path] | None = None,
) -> None:
    javac = _which_or_raise("javac")
    classes_dir.mkdir(parents=True, exist_ok=True)
    cmd = [javac, "-nowarn", "-d", str(classes_dir)]
    if sourcepath:
        import os

        cmd += ["-sourcepath", os.pathsep.join(str(p) for p in sourcepath)]
    cmd += [str(s) for s in sources]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"javac failed:\n{r.stderr}")


def compile_jni(cpp: Path, out_so: Path) -> None:
    gxx = _which_or_raise("g++")
    incs = jni_include_dirs()
    cmd = (
        [gxx, "-O2", "-shared", "-fPIC", "-std=c++17"]
        + [f"-I{d}" for d in incs]
        + [str(cpp), "-o", str(out_so)]
    )
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"g++ failed:\n{r.stderr}")


# ---------------------------------------------------------------------------
# value model: types, random generation, text encoding
# ---------------------------------------------------------------------------

_SIZE_NAMES = {"n", "size", "len", "length", "dim", "count", "rows", "cols"}

# harness-supported types (v0.4: 2-D double/int arrays and String join the
# v0.1 set; float[][]/long[][]/other objects are still rejected)
_HARNESS_TYPES = {
    "int",
    "long",
    "float",
    "double",
    "boolean",
    "String",
    "int[]",
    "long[]",
    "float[]",
    "double[]",
    "int[][]",
    "double[][]",
}


def _is_2d(jtype: str) -> bool:
    return jtype in ("double[][]", "int[][]")


def _is_1d(jtype: str) -> bool:
    return jtype.endswith("[]") and not _is_2d(jtype)


def check_harness_types(method: J.MethodInfo) -> None:
    bad = [p.jtype for p in method.params if p.jtype not in _HARNESS_TYPES]
    if method.jtype != "void" and method.jtype not in _HARNESS_TYPES:
        bad.append(f"return {method.jtype}")
    if bad:
        raise ValueError(
            f"verify/bench v0.4 does not support: {', '.join(bad)} "
            "(supported: int/long/float/double/boolean/String, "
            "their 1-D arrays, and double[][]/int[][])"
        )


def _rand_scalar(rng: random.Random, jtype: str, size_like: bool, bench: bool = False):
    if jtype == "int":
        if size_like:
            return rng.randint(256, 2048) if bench else rng.randint(0, 64)
        return rng.randint(-1000, 1000)
    if jtype == "long":
        return rng.randint(-10**6, 10**6)
    if jtype == "float":
        return rng.uniform(-100.0, 100.0)
    if jtype == "double":
        v = rng.uniform(-100.0, 100.0)
        return 0.0 if rng.random() < 0.1 else v
    if jtype == "boolean":
        return rng.random() < 0.5
    if jtype == "String":
        return _rand_string(rng)
    raise AssertionError(jtype)


def _rand_string(rng: random.Random) -> str:
    """Random ASCII string; occasionally sprinkled with whitespace so the
    escape protocol (``\\s``/``\\t``/``\\n``) gets exercised."""
    n = rng.randint(0, 12)
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    s = "".join(rng.choice(alphabet) for _ in range(n))
    if n and rng.random() < 0.25:
        pos = rng.randrange(n)
        s = s[:pos] + rng.choice([" ", "\t", "\n"]) + s[pos + 1 :]
    return s


def gen_case(
    method: J.MethodInfo,
    rng: random.Random,
    array_len: int = 8,
    bench: bool = False,
) -> list:
    """Generate one random argument list for `method`.

    In bench mode, array lengths are perfect squares when a size-like
    parameter is present (the matMul(a,b,c,n) shape, where n*n == len),
    and size-like scalars are larger, so the single representative input
    is both valid and meaty.
    """
    has_size_param = any(
        p.name.lower() in _SIZE_NAMES and not p.jtype.endswith("[]")
        for p in method.params
    )
    # lengths: arrays of the same element type share a length
    len_by_elem: dict[str, int] = {}
    for p in method.params:
        if _is_1d(p.jtype):
            elem = p.jtype[:-2]
            if elem not in len_by_elem:
                if bench and has_size_param:
                    s = rng.randint(8, 32)
                    len_by_elem[elem] = s * s
                else:
                    len_by_elem[elem] = rng.randint(0, array_len)
    # 2-D shapes: square, sharing the 1-D length when the element type has
    # one (so matVec(double[][], double[]) style kernels get matching
    # inputs); square keeps matMul(double[][], double[][]) in-domain.
    shape_by_elem: dict[str, tuple[int, int]] = {}
    for p in method.params:
        if _is_2d(p.jtype):
            elem = p.jtype[:-4]
            if elem not in shape_by_elem:
                if elem in len_by_elem:
                    L = len_by_elem[elem]
                elif bench:
                    L = rng.randint(8, 32)
                else:
                    L = rng.randint(0, array_len)
                shape_by_elem[elem] = (L, L)
    first_len = next(iter(len_by_elem.values()), 0)
    first_rows = next((r for r, _c in shape_by_elem.values()), 0)
    args: list = []
    for p in method.params:
        if _is_2d(p.jtype):
            elem = p.jtype[:-4]
            R, C = shape_by_elem[elem]
            args.append(
                [[_rand_scalar(rng, elem, False) for _ in range(C)] for _ in range(R)]
            )
        elif _is_1d(p.jtype):
            elem = p.jtype[:-2]
            L = len_by_elem[elem]
            args.append([_rand_scalar(rng, elem, False) for _ in range(L)])
        else:
            size_like = p.name.lower() in _SIZE_NAMES
            v = _rand_scalar(rng, p.jtype, size_like, bench)
            if size_like and p.jtype == "int" and (len_by_elem or shape_by_elem):
                if first_len:
                    # matMul(a,b,c,n) shape: n*n == len  ->  n = sqrt(len)
                    import math

                    s = int(math.isqrt(first_len))
                    v = s if s * s == first_len else first_len
                elif first_rows:
                    v = first_rows
                else:
                    v = 0  # empty arrays: only n = 0 is valid
            args.append(v)
    return args


def gen_bench_args(method: J.MethodInfo, seed: int = 1234, array_len: int = 512) -> list:
    rng = random.Random(seed)
    return gen_case(method, rng, array_len=array_len, bench=True)


# ---------------------------------------------------------------------------
# boundary-value generation
# ---------------------------------------------------------------------------

#: Cap on boundary cases per method (deterministic order; first N are kept).
_BOUNDARY_CAP = 24

_INT_EDGES = [0, 1, -1, -(2**31), 2**31 - 1]
_LONG_EDGES = [0, 1, -1, -(2**63), 2**63 - 1]
_DOUBLE_EDGES = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    float("nan"),
    float("inf"),
    float("-inf"),
    5e-324,  # smallest positive subnormal
    sys.float_info.max,
]
_FLOAT_EDGES = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    float("nan"),
    float("inf"),
    float("-inf"),
    1.1754944e-38,  # FLT_MIN
    3.4028235e38,  # FLT_MAX
]
_LARGE_ARRAY_LEN = 1024


def _boundary_scalar_edges(jtype: str) -> list:
    if jtype == "int":
        return list(_INT_EDGES)
    if jtype == "long":
        return list(_LONG_EDGES)
    if jtype == "float":
        return list(_FLOAT_EDGES)
    if jtype == "double":
        return list(_DOUBLE_EDGES)
    if jtype == "boolean":
        return [False, True]
    raise AssertionError(jtype)


def _boundary_array_edges(jtype: str, rng: random.Random) -> list:
    elem = jtype[:-2]
    zero = _boundary_scalar_edges(elem)[0]
    single = 2.5 if elem in ("float", "double") else 7
    if elem in ("float", "double"):
        big = [rng.uniform(-100.0, 100.0) for _ in range(_LARGE_ARRAY_LEN)]
    else:
        big = [rng.randint(-1000, 1000) for _ in range(_LARGE_ARRAY_LEN)]
    return [[], [zero], [single], big]


#: v0.4: 32x32 == 1024 elements, mirroring _LARGE_ARRAY_LEN.
_LARGE_2D = 32

_STRING_EDGES = [
    "",
    "a",
    "hello world",  # exercises the \s escape on the wire
    "x\ty\nz\r",  # exercises \t \n \r escapes
    "q" * 256,  # long string
]


def _boundary_2d_edges(jtype: str, rng: random.Random) -> list:
    elem = jtype[:-4]
    zero = _boundary_scalar_edges(elem)[0]
    single = 2.5 if elem == "double" else 7
    if elem == "double":
        big = [
            [rng.uniform(-100.0, 100.0) for _ in range(_LARGE_2D)]
            for _ in range(_LARGE_2D)
        ]
    else:
        big = [
            [rng.randint(-1000, 1000) for _ in range(_LARGE_2D)]
            for _ in range(_LARGE_2D)
        ]
    return [
        [],  # 0x0
        [[zero]],  # 1x1
        [[single, single], [single, single]],  # 2x2
        big,  # 32x32
    ]


def gen_boundary_cases(
    method: J.MethodInfo, rng: random.Random | None = None
) -> list[list]:
    """Deterministic boundary-value argument lists for `method`.

    Edge values per parameter type:

    - int/long: 0, 1, -1, MIN, MAX
    - float/double: 0.0, -0.0, 1.0, -1.0, NaN, +Inf, -Inf, tiny, max
    - boolean: False, True
    - String: "", "a", "hello world", whitespace/control chars, long string
    - arrays: [], [zero], [single value], large (1024-element) array
    - 2-D arrays: 0x0, 1x1, 2x2, large (32x32) array

    Combination is deliberately simple and deterministic: one all-neutral
    case (the first edge of every parameter), then each parameter swept
    across its remaining edges while the others stay neutral. Combinations
    that fall outside the kernel's valid domain (mismatched array lengths,
    gigantic allocations, ...) are skipped by the Java-first protocol in
    :func:`run_verify`, so generation stays best-effort. The list is capped
    at ``_BOUNDARY_CAP`` (24) cases in deterministic order.
    """
    r = rng if rng is not None else random.Random(0xC0FFEE)
    edge_lists: list[list] = []
    for p in method.params:
        if _is_2d(p.jtype):
            edge_lists.append(_boundary_2d_edges(p.jtype, r))
        elif p.jtype == "String":
            edge_lists.append(list(_STRING_EDGES))
        elif p.jtype.endswith("[]"):
            edge_lists.append(_boundary_array_edges(p.jtype, r))
        else:
            edge_lists.append(_boundary_scalar_edges(p.jtype))
    if not edge_lists:
        return [[]]
    cases = [[edges[0] for edges in edge_lists]]
    seen = {repr(cases[0])}
    for i, edges in enumerate(edge_lists):
        for v in edges[1:]:
            case = [edge_lists[j][0] for j in range(len(edge_lists))]
            case[i] = v
            key = repr(case)
            if key in seen:
                continue
            seen.add(key)
            cases.append(case)
            if len(cases) >= _BOUNDARY_CAP:
                return cases
    return cases


def _encode_fp(value) -> str:
    """Java-parseable rendering of a float/double.

    Python's repr() spells NaN/Infinity as 'nan'/'inf'/'-inf', which
    Double/Float.parseDouble reject (NumberFormatException). Map them to
    the spellings Java accepts so edge values survive the wire protocol.
    """
    f = float(value)
    if math.isnan(f):
        return "NaN"
    if math.isinf(f):
        return "Infinity" if f > 0 else "-Infinity"
    return repr(f)


def _escape_string(s: str) -> str:
    """Escape a String for the space-separated line protocol.

    Spaces would split the token and raw newlines would split the line, so
    ``\\`` ``\\n`` ``\\r`` ``\\t`` and space are backslash-escaped (decoded
    by the ``unescape`` helper in the generated Java harness).
    """
    return (
        s.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace(" ", "\\s")
    )


def _unescape_string(s: str) -> str:
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            e = s[i + 1]
            out.append({"n": "\n", "r": "\r", "t": "\t", "s": " ", "\\": "\\"}.get(e, e))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def encode_arg(jtype: str, value) -> str:
    if jtype == "int":
        return f"i:{int(value)}"
    if jtype == "long":
        return f"l:{int(value)}"
    if jtype == "float":
        return f"f:{_encode_fp(value)}"
    if jtype == "double":
        return f"d:{_encode_fp(value)}"
    if jtype == "boolean":
        return f"z:{1 if value else 0}"
    if jtype == "String":
        return "s:" + _escape_string(str(value))
    if jtype in ("int[]", "long[]", "float[]", "double[]"):
        tag = {"int[]": "ai", "long[]": "al", "float[]": "af", "double[]": "ad"}[jtype]
        inner = ",".join(
            repr(float(x)) if jtype in ("float[]", "double[]") else str(int(x))
            for x in value
        )
        return f"{tag}:{inner}"
    if jtype in ("double[][]", "int[][]"):
        # rows,cols followed by flat row-major elements, e.g. a2d:2,2,1.0,2.0,3.0,4.0
        tag = {"double[][]": "a2d", "int[][]": "a2i"}[jtype]
        rows = len(value)
        cols = len(value[0]) if rows else 0
        flat = [
            repr(float(x)) if jtype == "double[][]" else str(int(x))
            for row in value
            for x in row
        ]
        inner = ",".join(flat)
        return f"{tag}:{rows},{cols}" + ("," + inner if inner else "")
    raise AssertionError(jtype)


def encode_call(method: J.MethodInfo, args: list) -> str:
    return "CALL " + " ".join(
        encode_arg(p.jtype, v) for p, v in zip(method.params, args)
    )


def _parse_result_token(tok: str):
    if tok == "v":
        return ("void", None)
    tag, val = tok.split(":", 1)
    if tag == "i":
        return ("int", int(val))
    if tag == "l":
        return ("long", int(val))
    if tag == "f":
        return ("float", float(val))
    if tag == "d":
        return ("double", float(val))
    if tag == "z":
        return ("boolean", val == "1")
    if tag in ("ai", "al", "af", "ad"):
        elems = [] if val == "" else val.split(",")
        conv = float if tag in ("af", "ad") else int
        return (tag, [conv(e) for e in elems])
    if tag in ("a2d", "a2i"):
        # rows,cols,flat-row-major -> (rows, cols, [elements])
        parts = val.split(",")
        rows, cols = int(parts[0]), int(parts[1])
        conv = float if tag == "a2d" else int
        flat = [conv(e) for e in parts[2:] if e != ""]
        return (tag, (rows, cols, flat))
    if tag == "s":
        return ("s", _unescape_string(val))
    raise ValueError(f"bad result token: {tok}")


def results_equal(
    method: J.MethodInfo,
    java_tok: str,
    jni_tok: str,
    tol_float: float = 1e-6,
    tol_double: float = 1e-12,
) -> tuple[bool, str]:
    """Compare two result tokens; floats compare with per-type tolerance.

    ``tol_float`` applies to ``float``/``float[]`` results, ``tol_double``
    to ``double``/``double[]``/``double[][]``. Rationale: float32's epsilon
    is ~1.2e-7, so the old single 1e-9 threshold caused flaky false
    failures on float32 accumulation (a translation that accumulates in
    double and casts back is routinely ~1e-7 off the Java float result).
    Everything else (ints, booleans, Strings, shapes) compares exactly.
    The extra keyword args default to the v0.4 thresholds, so old 3-arg
    calls keep working.
    """

    def close(a: float, b: float, tol: float) -> bool:
        if math.isnan(a) and math.isnan(b):
            return True
        if math.isinf(a) or math.isinf(b):
            return a == b
        return abs(a - b) <= tol + tol * max(abs(a), abs(b))

    def tol_for(tag: str) -> float | None:
        """Per-element-type tolerance for a result tag; None means exact."""
        if tag in ("float", "af"):
            return tol_float
        if tag in ("double", "ad", "a2d"):
            return tol_double
        return None

    def lists_close(tag: str, xs: list, ys: list) -> tuple[bool, str]:
        tol = tol_for(tag)
        if len(xs) != len(ys):
            return False, f"array length mismatch: {len(xs)} vs {len(ys)}"
        for a, b in zip(xs, ys):
            if tol is None:
                if a != b:
                    return False, f"array element mismatch: {a} != {b}"
            elif not close(a, b, tol):
                return False, f"array element mismatch: {a} != {b}"
        return True, ""

    if java_tok == "v" or jni_tok == "v":
        return (java_tok == jni_tok, "" if java_tok == jni_tok else "void mismatch")
    if java_tok.startswith("v|") or jni_tok.startswith("v|"):
        # void method with mutated-array echo: "v|ad:1.0,2.0|ai:3"
        if not (java_tok.startswith("v|") and jni_tok.startswith("v|")):
            return False, "void/array-echo shape mismatch"
        jsegs = java_tok[2:].split("|")
        nsegs = jni_tok[2:].split("|")
        if len(jsegs) != len(nsegs):
            return False, f"array count mismatch: {len(jsegs)} vs {len(nsegs)}"
        for js, ns in zip(jsegs, nsegs):
            try:
                tj, vj = _parse_result_token(js)
                tn, vn = _parse_result_token(ns)
            except ValueError as e:
                return False, str(e)
            if tj != tn:
                return False, f"array type mismatch: {js!r} vs {ns!r}"
            if tj in ("a2d", "a2i"):
                (rj, cj, fj), (rn, cn, fn) = vj, vn
                if (rj, cj) != (rn, cn):
                    return False, f"2-D shape mismatch: {rj}x{cj} vs {rn}x{cn}"
                ok, detail = lists_close(tj, fj, fn)
                if not ok:
                    return False, detail
                continue
            if not isinstance(vj, list) or not isinstance(vn, list):
                return False, f"array type mismatch: {js!r} vs {ns!r}"
            ok, detail = lists_close(tj, vj, vn)
            if not ok:
                return False, detail
        return True, ""
    try:
        tj, vj = _parse_result_token(java_tok)
        tn, vn = _parse_result_token(jni_tok)
    except ValueError as e:
        return False, str(e)

    if tj in ("a2d", "a2i"):
        if tn != tj:
            return False, f"type mismatch: {java_tok!r} vs {jni_tok!r}"
        (rj, cj, fj), (rn, cn, fn) = vj, vn
        if (rj, cj) != (rn, cn):
            return False, f"2-D shape mismatch: {rj}x{cj} vs {rn}x{cn}"
        ok, detail = lists_close(tj, fj, fn)
        return ok, detail
    if tj in ("ai", "al", "af", "ad"):
        if tn != tj:
            return False, f"type mismatch: {java_tok!r} vs {jni_tok!r}"
        if not isinstance(vn, list):
            return False, f"type mismatch: {java_tok!r} vs {jni_tok!r}"
        ok, detail = lists_close(tj, vj, vn)
        return ok, "" if ok else f"array mismatch: {detail}"
    if tj in ("float", "double"):
        tol = tol_for(tj)
        assert tol is not None
        ok = close(vj, vn, tol)
        return ok, "" if ok else f"{vj} != {vn}"
    if tj == "s":
        ok = tn == "s" and vj == vn
        return ok, "" if ok else f"{vj!r} != {vn!r}"
    ok = vj == vn
    return ok, "" if ok else f"{vj} != {vn}"


# ---------------------------------------------------------------------------
# Java harness codegen
# ---------------------------------------------------------------------------

_JAVA_DECL = {
    "int": "int",
    "long": "long",
    "float": "float",
    "double": "double",
    "boolean": "boolean",
    "String": "String",
    "int[]": "int[]",
    "long[]": "long[]",
    "float[]": "float[]",
    "double[]": "double[]",
    "int[][]": "int[][]",
    "double[][]": "double[][]",
}


def _decode_java(jtype: str, var: str, tok: str) -> str:
    if jtype == "int":
        return f"int {var} = Integer.parseInt({tok}.substring(2));"
    if jtype == "long":
        return f"long {var} = Long.parseLong({tok}.substring(2));"
    if jtype == "float":
        return f"float {var} = Float.parseFloat({tok}.substring(2));"
    if jtype == "double":
        return f"double {var} = Double.parseDouble({tok}.substring(2));"
    if jtype == "boolean":
        return f"boolean {var} = {tok}.equals(\"z:1\");"
    if jtype == "String":
        return f"String {var} = unescape({tok}.substring(2));"
    if jtype == "int[]":
        return f"int[] {var} = parseIntArray({tok}.substring(3));"
    if jtype == "long[]":
        return f"long[] {var} = parseLongArray({tok}.substring(3));"
    if jtype == "float[]":
        return f"float[] {var} = parseFloatArray({tok}.substring(3));"
    if jtype == "double[]":
        return f"double[] {var} = parseDoubleArray({tok}.substring(3));"
    if jtype == "int[][]":
        return f"int[][] {var} = parseInt2d({tok}.substring(4));"
    if jtype == "double[][]":
        return f"double[][] {var} = parseDouble2d({tok}.substring(4));"
    raise AssertionError(jtype)


def _encode_java(jtype: str, var: str) -> str:
    if jtype == "int":
        return f'"i:" + {var}'
    if jtype == "long":
        return f'"l:" + {var}'
    if jtype == "float":
        return f'"f:" + {var}'
    if jtype == "double":
        return f'"d:" + {var}'
    if jtype == "boolean":
        return f'({var} ? "z:1" : "z:0")'
    if jtype == "String":
        return f'"s:" + escape({var})'
    if jtype == "int[]":
        return f'"ai:" + join({var})'
    if jtype == "long[]":
        return f'"al:" + join({var})'
    if jtype == "float[]":
        return f'"af:" + join({var})'
    if jtype == "double[]":
        return f'"ad:" + join({var})'
    if jtype == "int[][]":
        return f'"a2i:" + join2d({var})'
    if jtype == "double[][]":
        return f'"a2d:" + join2d({var})'
    raise AssertionError(jtype)


def render_harness(
    method: J.MethodInfo, target: str, class_name: str, package: str = ""
) -> str:
    """Render HarnessJava/HarnessJni source.

    target='java' calls com.example.Kernels.dot(...);
    target='jni'  calls KernelsNative.dot(...).
    ``package`` (v0.12) qualifies the glue class when it was generated in
    the kernel's package (non-public kernel); "" keeps the legacy
    default-package reference.
    """
    if target == "java":
        receiver = method.class_qualname
    else:
        simple = method.class_qualname.rsplit(".", 1)[-1]
        glue_cls = f"{simple}Ffm" if target == "ffm" else f"{simple}Native"
        receiver = f"{package}.{glue_cls}" if package else glue_cls
    params = method.params
    is_void = method.jtype == "void"

    decls = []
    for i, p in enumerate(params):
        decls.append("        " + _decode_java(p.jtype, f"b_a{i}", f"tok[off+{i}]"))

    # bench loop: re-clone arrays each iteration so in/out params start clean
    loop_setup = []
    call_args = []
    for i, p in enumerate(params):
        if _is_2d(p.jtype):
            # shallow clone() would alias the rows: deep-clone instead
            loop_setup.append(f"            {_JAVA_DECL[p.jtype]} a{i} = clone2d(b_a{i});")
        elif p.jtype.endswith("[]"):
            loop_setup.append(f"            {_JAVA_DECL[p.jtype]} a{i} = b_a{i}.clone();")
        else:
            loop_setup.append(f"            {_JAVA_DECL[p.jtype]} a{i} = b_a{i};")
        call_args.append(f"a{i}")
    call = f"{receiver}.{method.name}({', '.join(call_args)})"

    if is_void:
        call_stmt = f"{call};"
        result_expr = '"v"'
    else:
        call_stmt = f"{_JAVA_DECL[method.jtype]} r = {call};"
        result_expr = _encode_java(method.jtype, "r")

    # CALL path uses b_a* directly (single shot)
    call_args_direct = ", ".join(f"b_a{i}" for i in range(len(params)))
    call_direct = f"{receiver}.{method.name}({call_args_direct})"
    if is_void:
        # Void methods communicate through mutated array params: echo them
        # back so differential testing actually compares something.
        # Token: "v" (no array params) or "v|ad:1.0,2.0|ai:3" (array echo).
        _echo_parts = []
        for _i, _p in enumerate(params):
            if _p.jtype.endswith("[]"):
                _echo_parts.append(f' + "|" + {_encode_java(_p.jtype, f"b_a{_i}")}')
        _echo = "".join(_echo_parts)
        direct_stmt = f"{call_direct};\n            return \"OK v\"{_echo};"
    else:
        direct_stmt = (
            f"{_JAVA_DECL[method.jtype]} r = {call_direct};\n"
            f'            return "OK " + {result_expr};'
        )

    return f"""import java.io.*;

public class {class_name} {{
    public static void main(String[] args) throws Exception {{
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        PrintWriter out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.out)), true);
        String line;
        while ((line = br.readLine()) != null) {{
            line = line.trim();
            if (line.isEmpty()) continue;
            try {{
                out.println(handle(line));
            }} catch (Throwable t) {{
                String m = String.valueOf(t.getMessage()).replace('\\n', ' ');
                out.println("ERR " + t.getClass().getSimpleName() + ": " + m);
            }}
        }}
    }}

    static String handle(String line) {{
        String[] tok = line.split(" ");
        boolean bench = tok[0].equals("BENCH");
        int iters = 1;
        int off = 1;
        if (bench) {{ iters = Integer.parseInt(tok[1]); off = 2; }}
{chr(10).join(decls)}
        if (!bench) {{
            {direct_stmt}
        }}
        long t0 = System.nanoTime();
        for (int k = 0; k < iters; k++) {{
{chr(10).join(loop_setup)}
            {call_stmt}
        }}
        long t1 = System.nanoTime();
        return "OK " + (t1 - t0);
    }}

    static int[] parseIntArray(String s) {{
        if (s.isEmpty()) return new int[0];
        String[] p = s.split(",");
        int[] r = new int[p.length];
        for (int i = 0; i < p.length; i++) r[i] = Integer.parseInt(p[i]);
        return r;
    }}
    static long[] parseLongArray(String s) {{
        if (s.isEmpty()) return new long[0];
        String[] p = s.split(",");
        long[] r = new long[p.length];
        for (int i = 0; i < p.length; i++) r[i] = Long.parseLong(p[i]);
        return r;
    }}
    static float[] parseFloatArray(String s) {{
        if (s.isEmpty()) return new float[0];
        String[] p = s.split(",");
        float[] r = new float[p.length];
        for (int i = 0; i < p.length; i++) r[i] = Float.parseFloat(p[i]);
        return r;
    }}
    static double[] parseDoubleArray(String s) {{
        if (s.isEmpty()) return new double[0];
        String[] p = s.split(",");
        double[] r = new double[p.length];
        for (int i = 0; i < p.length; i++) r[i] = Double.parseDouble(p[i]);
        return r;
    }}
    static String join(int[] a) {{
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < a.length; i++) {{ if (i > 0) sb.append(','); sb.append(a[i]); }}
        return sb.toString();
    }}
    static String join(long[] a) {{
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < a.length; i++) {{ if (i > 0) sb.append(','); sb.append(a[i]); }}
        return sb.toString();
    }}
    static String join(float[] a) {{
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < a.length; i++) {{ if (i > 0) sb.append(','); sb.append(a[i]); }}
        return sb.toString();
    }}
    static String join(double[] a) {{
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < a.length; i++) {{ if (i > 0) sb.append(','); sb.append(a[i]); }}
        return sb.toString();
    }}
{_HARNESS_JAVA_HELPERS_V04}}}
"""


# ---------------------------------------------------------------------------
# v0.4 Java helpers: 2-D arrays, Strings (raw string: backslashes literal)
# ---------------------------------------------------------------------------

_HARNESS_JAVA_HELPERS_V04 = r"""
    // ---- v0.4: 2-D arrays and Strings ----
    static double[][] parseDouble2d(String s) {
        String[] p = s.split(",");
        int rows = Integer.parseInt(p[0]), cols = Integer.parseInt(p[1]);
        double[][] m = new double[rows][cols];
        int k = 2;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i][j] = Double.parseDouble(p[k++]);
        return m;
    }
    static int[][] parseInt2d(String s) {
        String[] p = s.split(",");
        int rows = Integer.parseInt(p[0]), cols = Integer.parseInt(p[1]);
        int[][] m = new int[rows][cols];
        int k = 2;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i][j] = Integer.parseInt(p[k++]);
        return m;
    }
    static String join2d(double[][] m) {
        int rows = m.length, cols = rows > 0 ? m[0].length : 0;
        StringBuilder sb = new StringBuilder();
        sb.append(rows).append(',').append(cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                sb.append(',').append(m[i][j]);
        return sb.toString();
    }
    static String join2d(int[][] m) {
        int rows = m.length, cols = rows > 0 ? m[0].length : 0;
        StringBuilder sb = new StringBuilder();
        sb.append(rows).append(',').append(cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                sb.append(',').append(m[i][j]);
        return sb.toString();
    }
    static double[][] clone2d(double[][] m) {
        double[][] c = new double[m.length][];
        for (int i = 0; i < m.length; i++) c[i] = m[i].clone();
        return c;
    }
    static int[][] clone2d(int[][] m) {
        int[][] c = new int[m.length][];
        for (int i = 0; i < m.length; i++) c[i] = m[i].clone();
        return c;
    }
    static String escape(String s) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '\\') sb.append("\\\\");
            else if (c == '\n') sb.append("\\n");
            else if (c == '\r') sb.append("\\r");
            else if (c == '\t') sb.append("\\t");
            else if (c == ' ') sb.append("\\s");
            else sb.append(c);
        }
        return sb.toString();
    }
    static String unescape(String s) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '\\' && i + 1 < s.length()) {
                char e = s.charAt(++i);
                switch (e) {
                    case 'n': sb.append('\n'); break;
                    case 'r': sb.append('\r'); break;
                    case 't': sb.append('\t'); break;
                    case 's': sb.append(' '); break;
                    case '\\': sb.append('\\'); break;
                    default: sb.append(e); break;
                }
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }
"""

# ---------------------------------------------------------------------------
# subprocess session with per-line timeout
# ---------------------------------------------------------------------------


#: A protocol line is a harness answer: "OK <payload>" or "ERR <message>".
#: Anything else on stdout (JVM startup warnings, stray chatter) is
#: environmental noise, never an answer.
_PROTOCOL_RE = re.compile(r"^(?:OK|ERR) ")


class LineSession:
    """A subprocess speaking the line protocol, with per-line timeouts.

    ``ask`` is tolerant of non-protocol stdout lines: JVM launchers
    sometimes print startup warnings (e.g. stale ``hsperfdata`` lock
    warnings) before the harness speaks, and those lines must never be
    misparsed as protocol errors. Non-protocol lines are skipped -- and
    logged to stderr for debuggability -- while a genuinely missing
    protocol line still raises ``TimeoutError`` and a malformed protocol
    line is returned for the caller to reject ("protocol error"). A real
    protocol violation is therefore never silently swallowed; it just
    cannot be triggered by environmental noise anymore.
    """

    def __init__(self, cmd: list[str], cwd: Path | None = None):
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            cwd=str(cwd) if cwd else None,
        )
        self._q: queue.Queue[str] = queue.Queue()
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()

    def _reader(self):
        try:
            for line in self.proc.stdout:  # type: ignore[union-attr]
                self._q.put(line.rstrip("\n"))
        except Exception:
            pass

    def ask(self, line: str, timeout: float) -> str:
        """Send a command line; return the next *protocol* line.

        Non-protocol stdout lines are skipped (logged to stderr). The
        timeout is a deadline covering the whole skip-and-wait, so a
        warning flood cannot stretch the wait indefinitely: when no
        protocol line arrives in time, ``TimeoutError`` is raised exactly
        as before.
        """
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError:
            raise RuntimeError("harness process died (broken pipe)")
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"harness did not answer within {timeout}s: {line[:80]}"
                )
            try:
                raw = self._q.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(
                    f"harness did not answer within {timeout}s: {line[:80]}"
                )
            if _PROTOCOL_RE.match(raw):
                return raw
            # Environmental noise (e.g. "Warning: failed to ... hsperfdata
            # ..."): not an answer, not a protocol violation. Log it so a
            # debugging human can see what the launcher said.
            print(
                f"kernel-port: ignoring non-protocol harness output: {raw!r}",
                file=sys.stderr,
            )

    def close(self):
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


# ---------------------------------------------------------------------------
# build + session setup shared by verify and bench
# ---------------------------------------------------------------------------


@dataclass
class BuiltEnv:
    work: Path
    classes: Path
    method: J.MethodInfo
    backend: str = "jni"          # "jni", "ffm" or "rust"
    java_bin: str | None = None   # JVM to run harnesses with (FFM needs JDK 22+)


def _write_minimal_kernel_sources(
    repo: J.JavaRepo, method: J.MethodInfo, work: Path
) -> tuple[Path, list[Path]] | None:
    """Synthesize hermetic kernel sources for the harness to compile.

    Returns ``(synth_dir, files)`` or ``None`` when synthesis is unsafe.
    See :func:`extractor.render_minimal_sources` for rationale.
    """
    from . import extractor as E

    closure = E._extract_closure(repo, method)
    rendered = E.render_minimal_sources(method, closure, repo)
    if not rendered:
        return None
    synth_dir = work / "kernel-src"
    files = []
    for rel, text in rendered.items():
        p = synth_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        files.append(p)
    return synth_dir, files


def _find_glue_java(glue_dir: Path, backend: str) -> tuple[Path | None, str]:
    """Locate the generated glue Java class and read its package.

    Returns ``(path, package)``; ``package`` is ``""`` for the legacy
    default-package glue. Uses a recursive glob because v0.12 may place
    the class in the kernel's package directory.
    """
    pattern = "*Ffm.java" if backend == "ffm" else "*Native.java"
    found = sorted(glue_dir.rglob(pattern))
    if not found:
        return None, ""
    path = found[0]
    package = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*package\s+([\w.]+)\s*;", line)
        if m:
            package = m.group(1)
            break
    return path, package


def build_env(
    repo_root: Path,
    method: J.MethodInfo,
    glue_dir: Path,
    work: Path,
    backend: str = "jni",
    repo: J.JavaRepo | None = None,
) -> BuiltEnv:
    """Compile fixture + harnesses + native lib. Raises RuntimeError on failure.

    ``repo`` (the scanned :class:`JavaRepo`) enables the v0.8 dogfood path:
    the harness compiles a synthesized minimal kernel file (kernel +
    dependency closure) instead of the original repo file, so ``javac``
    never typechecks environmentally-broken siblings (test sources,
    annotation-only deps, ``jdk.internal`` users). When synthesis is
    unsafe or its compile fails, falls back to the original file with
    ``-sourcepath`` dependency resolution.
    """
    check_harness_types(method)
    glue_dir = Path(glue_dir)
    work = Path(work)
    if work.exists():
        shutil.rmtree(work)
    classes = work / "classes"
    classes.mkdir(parents=True)

    # Dogfood fix (v0.8): never compile the whole repo. Real repos have
    # test sources (JUnit) and files needing annotation-only deps that are
    # not on any classpath -- compiling everything fails.
    kernel_file = Path(repo_root) / method.file
    if not kernel_file.exists():
        raise RuntimeError(f"kernel source not found: {kernel_file}")
    repo_sproots = kernel_sourcepath_roots(repo_root, method)

    # Prefer synthesized minimal sources; fall back to the repo file when
    # synthesis is unsafe. Each entry: (sources, sourcepath, label).
    attempts: list[tuple[list[Path], list[Path], str]] = []
    if repo is not None:
        synth = _write_minimal_kernel_sources(repo, method, work)
        if synth is not None:
            synth_dir, synth_files = synth
            attempts.append((synth_files, [synth_dir], "minimal"))
    attempts.append(([kernel_file], repo_sproots, "repo-file"))

    def compile_kernel_java(extra: list[Path], ffm_javac: str | None = None) -> None:
        import sys

        last: RuntimeError | None = None
        for sources, sproots, label in attempts:
            try:
                if ffm_javac is not None:
                    from . import ffm as _ffm

                    _ffm.compile_java(
                        sources + extra, classes,
                        javac=ffm_javac, sourcepath=sproots,
                    )
                else:
                    compile_java(sources + extra, classes, sproots)
                if label == "repo-file" and len(attempts) > 1:
                    print(
                        "kernel-port: minimal kernel sources failed to compile; "
                        "fell back to the repo file",
                        file=sys.stderr,
                    )
                return
            except RuntimeError as exc:
                last = exc
        assert last is not None
        raise last
    hj = work / "HarnessJava.java"
    hn = work / "HarnessJni.java"
    hj.write_text(render_harness(method, "java", "HarnessJava"), encoding="utf-8")

    if backend == "ffm":
        from . import ffm as _ffm

        jdk = _ffm.find_jdk()
        if jdk is None:
            raise RuntimeError(
                "FFM backend requires JDK 22+ (25 LTS recommended); "
                "no suitable JDK found on PATH, in ~/workspace/tools/jdk-*, "
                "or $KERNEL_PORT_JDK."
            )
        ok, msg = _ffm.java_version_ok(str(jdk / "bin" / "javac"))
        if not ok:
            raise RuntimeError(msg)
        c_file = glue_dir / "kernels_ffm.c"
        ffm_java, ffm_pkg = _find_glue_java(glue_dir, "ffm")
        if not c_file.exists():
            raise RuntimeError(f"{c_file} not found -- run `kernel-port glue --backend ffm` first")
        if ffm_java is None:
            raise RuntimeError(f"no *Ffm.java in {glue_dir} -- run `kernel-port glue --backend ffm` first")
        hn.write_text(
            render_harness(method, "ffm", "HarnessJni", package=ffm_pkg),
            encoding="utf-8",
        )
        compile_kernel_java(
            [hj, hn, ffm_java], ffm_javac=str(jdk / "bin" / "javac")
        )
        _ffm.compile_c(c_file, work / "libkernels_ffm.so")
        return BuiltEnv(
            work=work, classes=classes, method=method,
            backend="ffm", java_bin=str(jdk / "bin" / "java"),
        )

    if backend == "rust":
        from . import rust_glue as _rust

        cargo_toml = glue_dir / "Cargo.toml"
        lib_rs = glue_dir / "src" / "lib.rs"
        rust_java, rust_pkg = _find_glue_java(glue_dir, "rust")
        if not cargo_toml.exists() or not lib_rs.exists():
            raise RuntimeError(
                f"{cargo_toml} / {lib_rs} not found -- "
                "run `kernel-port glue --backend rust` first"
            )
        if rust_java is None:
            raise RuntimeError(
                f"no *Native.java in {glue_dir} -- run `kernel-port glue --backend rust` first"
            )
        # The Rust cdylib exports JNI entry points directly (jni crate), so
        # the harness protocol is identical to the JNI backend -- only the
        # build step differs (cargo instead of g++). The cdylib name is
        # per-kernel (see rust_glue.lib_name_for); copy it under its own
        # name so System.loadLibrary("<name>") resolves it.
        hn.write_text(
            render_harness(method, "jni", "HarnessJni", package=rust_pkg),
            encoding="utf-8",
        )
        compile_kernel_java([hj, hn, rust_java])
        cdylib = _rust.compile_rust(glue_dir)
        so = work / cdylib.name
        shutil.copy2(cdylib, so)
        return BuiltEnv(work=work, classes=classes, method=method, backend="rust")

    cpp = glue_dir / "kernels_jni.cpp"
    native_java, jni_pkg = _find_glue_java(glue_dir, "jni")
    if not cpp.exists():
        raise RuntimeError(f"{cpp} not found -- run `kernel-port glue` first")
    if native_java is None:
        raise RuntimeError(f"no *Native.java in {glue_dir} -- run `kernel-port glue` first")

    hn.write_text(
        render_harness(method, "jni", "HarnessJni", package=jni_pkg),
        encoding="utf-8",
    )
    compile_kernel_java([hj, hn, native_java])

    so = work / "libkernels.so"
    compile_jni(cpp, so)
    return BuiltEnv(work=work, classes=classes, method=method)


def open_sessions(env: BuiltEnv) -> tuple[LineSession, LineSession]:
    java = env.java_bin or _which_or_raise("java")
    cp = str(env.classes)
    s_java = LineSession([java, "-cp", cp, "HarnessJava"])
    s_jni = LineSession(
        [java, f"-Djava.library.path={env.work}", "-cp", cp, "HarnessJni"]
    )
    return s_java, s_jni


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


@dataclass
class VerifyCase:
    index: int
    args: list
    ok: bool
    detail: str = ""
    java_result: str = ""  # raw result token from the Java harness ("d:1.5", "v", ...)
    jni_result: str = ""  # raw result token from the JNI harness
    minimized_args: list | None = None  # v0.6: ddmin-shrunk failing input, when computed


@dataclass
class VerifyReport:
    method: str
    cases: list[VerifyCase]
    passed: int
    failed: int
    skipped: int = 0
    boundary: list[VerifyCase] = field(default_factory=list)
    boundary_skipped: int = 0
    # --- v0.6 coverage-guided fuzz (all defaulted; old constructions keep working) ---
    guided: list[VerifyCase] = field(default_factory=list)
    guided_skipped: int = 0
    coverage_pct: float | None = None
    branches_hit: int = 0
    branches_total: int = 0

    @property
    def all_passed(self) -> bool:
        # Boundary mismatches fail the run too: they are real translation
        # bugs caught by the edge-case sweep. With boundary=False the list
        # is empty, so this reduces to the v0.1 definition.
        return self.failed == 0 and self.boundary_failed == 0 and self.guided_failed == 0

    @property
    def boundary_passed(self) -> int:
        return sum(1 for c in self.boundary if c.ok)

    @property
    def boundary_failed(self) -> int:
        return sum(1 for c in self.boundary if not c.ok)

    @property
    def guided_passed(self) -> int:
        return sum(1 for c in self.guided if c.ok)

    @property
    def guided_failed(self) -> int:
        return sum(1 for c in self.guided if not c.ok)


def _run_one_case(
    s_java: LineSession,
    s_jni: LineSession,
    method: J.MethodInfo,
    index: int,
    args: list,
    timeout: float,
    tol_float: float = 1e-6,
    tol_double: float = 1e-12,
) -> tuple[VerifyCase | None, bool, bool]:
    """Run one differential case through both sessions.

    Returns ``(case, skipped, java_no_answer)``:

    - ``skipped`` (case None): the Java original rejected the input as
      out-of-domain -- the caller counts it as skipped without feeding it
      to native code, which has no bounds checking.
    - ``java_no_answer``: the Java side timed out or died, so there is no
      reference answer to compare against. The case is inconclusive (it is
      returned as a failed VerifyCase for the caller to classify).
    """
    line = encode_call(method, args)
    try:
        rj = s_java.ask(line, timeout)
    except (TimeoutError, RuntimeError) as e:
        return VerifyCase(index, args, False, f"java harness error: {e}"), False, True
    if rj.startswith("ERR"):
        return None, True, False
    if not rj.startswith("OK "):
        return VerifyCase(index, args, False, f"protocol error: {rj!r}"), False, False
    try:
        rn = s_jni.ask(line, timeout)
    except (TimeoutError, RuntimeError) as e:
        return (
            VerifyCase(
                index, args, False, f"jni harness error: {e}", java_result=rj[3:]
            ),
            False,
            False,
        )
    if rn.startswith("ERR"):
        return (
            VerifyCase(
                index,
                args,
                False,
                f"jni harness threw: {rn}",
                java_result=rj[3:],
                jni_result=rn,
            ),
            False,
            False,
        )
    if not rn.startswith("OK "):
        return (
            VerifyCase(
                index, args, False, f"protocol error: {rn!r}", java_result=rj[3:]
            ),
            False,
            False,
        )
    ok, detail = results_equal(method, rj[3:], rn[3:], tol_float, tol_double)
    return (
        VerifyCase(index, args, ok, detail, java_result=rj[3:], jni_result=rn[3:]),
        False,
        False,
    )


def _json_safe(value):
    """Recursively convert a value to JSON-safe form.

    NaN/Infinity floats are not valid JSON, so they become their repr()
    strings ("nan", "inf", "-inf"); counterex.py renders those readably.
    """
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return repr(value)
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def _safe_method_filename(qualname: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", qualname)


def _persist_failures(env: BuiltEnv, report: VerifyReport) -> Path | None:
    """Write failing cases to ``<work>/.kp-work/<method>.failures.json``.

    Each entry holds JSON-safe ``args`` (NaN/Infinity as repr strings),
    ``java_result`` / ``jni_result`` tokens, ``detail``, and whether the
    case came from the ``boundary`` sweep. Never raises: when the work dir
    is unavailable the failures are dropped silently (the report itself is
    still returned normally).
    """
    try:
        entries: list[dict] = []
        for c in report.cases:
            if not c.ok:
                entries.append(
                    {
                        "args": _json_safe(c.args),
                        "java_result": c.java_result,
                        "jni_result": c.jni_result,
                        "detail": c.detail,
                        "boundary": False,
                    }
                )
        for c in report.boundary:
            if not c.ok:
                entries.append(
                    {
                        "args": _json_safe(c.args),
                        "java_result": c.java_result,
                        "jni_result": c.jni_result,
                        "detail": c.detail,
                        "boundary": True,
                    }
                )
        if not entries:
            return None
        out_dir = Path(env.work)
        if out_dir.name != ".kp-work":
            # env.work is usually already <glue>/.kp-work (the CLI default);
            # don't double it when a caller passes the .kp-work dir directly.
            out_dir = out_dir / ".kp-work"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{_safe_method_filename(report.method)}.failures.json"
        path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        return path
    except Exception:
        return None


def run_verify(
    env: BuiltEnv,
    cases: int = 100,
    seed: int = 42,
    per_case_timeout: float = 30.0,
    boundary: bool = True,
    boundary_cases: int = 24,
    tol_float: float = 1e-6,
    tol_double: float = 1e-12,
) -> VerifyReport:
    """Differential-test the JNI translation against the Java original.

    ``cases`` random inputs are generated exactly as in v0.1. When
    ``boundary`` is true, up to ``boundary_cases`` deterministic edge-case
    inputs (see :func:`gen_boundary_cases`) run through the same sessions
    afterwards and are reported separately in ``report.boundary``;
    ``report.cases`` keeps holding only the random cases, so existing count
    assertions still hold. Failing cases (random and boundary) are persisted
    to ``<work>/.kp-work/<method>.failures.json`` for counterexample
    feedback (see :func:`_persist_failures`).

    ``tol_float``/``tol_double`` are the per-type float comparison
    thresholds forwarded to :func:`results_equal` (v0.4); the CLI can
    expose them as flags later.
    """
    rng = random.Random(seed)
    s_java, s_jni = open_sessions(env)
    results: list[VerifyCase] = []
    skipped = 0
    b_results: list[VerifyCase] = []
    b_skipped = 0
    attempts = 0
    max_attempts = cases * 3 + 10
    try:
        while len(results) < cases and attempts < max_attempts:
            attempts += 1
            i = attempts - 1
            args = gen_case(env.method, rng)
            case, was_skipped, _java_no_answer = _run_one_case(
                s_java, s_jni, env.method, i, args, per_case_timeout,
                tol_float, tol_double,
            )
            if was_skipped:
                skipped += 1
                continue
            assert case is not None
            results.append(case)
        if boundary:
            for bi, bargs in enumerate(
                gen_boundary_cases(env.method)[:boundary_cases]
            ):
                case, was_skipped, java_no_answer = _run_one_case(
                    s_java, s_jni, env.method, bi, bargs, per_case_timeout,
                    tol_float, tol_double,
                )
                if was_skipped:
                    b_skipped += 1
                    continue
                assert case is not None
                if java_no_answer:
                    # The reference implementation itself did not answer in
                    # time (e.g. fib(INT_MAX) needs billions of loop
                    # iterations): the case is inconclusive, not a
                    # translation bug, so it is skipped rather than failed.
                    # Reopen the sessions -- a timed-out session may be
                    # desynced, with a late answer landing in the next ask.
                    b_skipped += 1
                    s_java.close()
                    s_jni.close()
                    s_java, s_jni = open_sessions(env)
                    continue
                b_results.append(case)
    finally:
        s_java.close()
        s_jni.close()
    passed = sum(1 for c in results if c.ok)
    report = VerifyReport(
        method=env.method.qualname,
        cases=results,
        passed=passed,
        failed=len(results) - passed,
        skipped=skipped,
        boundary=b_results,
        boundary_skipped=b_skipped,
    )
    _persist_failures(env, report)
    return report


def run_cases(
    env: BuiltEnv,
    arg_lists: list[list],
    per_case_timeout: float = 30.0,
    tol_float: float = 1e-6,
    tol_double: float = 1e-12,
) -> tuple[list[VerifyCase], int]:
    """Differentially run explicit argument lists through both harnesses.

    Unlike :func:`run_verify` (which generates its own inputs), this runs
    caller-supplied cases -- used for the coverage-guided second round.
    Returns ``(cases, skipped)``; java-no-answer cases are counted as
    skipped after reopening the sessions, mirroring ``run_verify``.
    """
    s_java, s_jni = open_sessions(env)
    results: list[VerifyCase] = []
    skipped = 0
    try:
        for i, args in enumerate(arg_lists):
            case, was_skipped, java_no_answer = _run_one_case(
                s_java, s_jni, env.method, i, args, per_case_timeout,
                tol_float, tol_double,
            )
            if was_skipped or java_no_answer:
                if java_no_answer:
                    s_java.close()
                    s_jni.close()
                    s_java, s_jni = open_sessions(env)
                skipped += 1
                continue
            assert case is not None
            results.append(case)
    finally:
        s_java.close()
        s_jni.close()
    return results, skipped


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------


#: Heuristic break-even bar: a port counts as "worth it" once JNI beats
#: Java by more than 20%. The headroom absorbs measurement noise and the
#: gap between microbenchmark conditions and real call sites.
_BREAK_EVEN_SPEEDUP = 1.2


def _break_even_size(per_size: list[dict]) -> int | None:
    """Smallest size with speedup > 1.2 (the heuristic bar), else None."""
    for entry in sorted(per_size, key=lambda e: e["size"]):
        if entry.get("speedup", 0) > _BREAK_EVEN_SPEEDUP:
            return entry["size"]
    return None


#: Verdict noise band (v0.10): the bench only claims "faster"/"slower"
#: when the median speedup lies *outside* this band; anything inside is
#: "inconclusive -- too close to call".
#:
#: The 1.2 edges are deliberately the same value as _BREAK_EVEN_SPEEDUP.
#: Wall-clock Java microbenchmarks routinely show 5-15% run-to-run jitter
#: (JIT/C2 compilation timing, GC pauses, CPU frequency scaling), and the
#: v0.2 break-even bar was already chosen at 20% headroom for exactly that
#: noise. Reusing it keeps a single honest threshold: below 1.2x a claimed
#: "win" is indistinguishable from measurement noise (the v0.9 dogfood
#: flip -- 5.40x faster one run, 1.06x slower the next -- was a JIT warmup
#: artifact landing inside any reasonable band), so the tool refuses to
#: claim one. The edges themselves are inclusive: exactly 1.20x is still
#: too close to call.
_VERDICT_NOISE_BAND = 1.2


def verdict_for_speedup(speedup: float) -> str:
    """Classify a median speedup as ``"faster"``/``"slower"``/``"inconclusive"``.

    ``"faster"`` only when ``speedup > _VERDICT_NOISE_BAND``, ``"slower"``
    only when ``speedup < 1 / _VERDICT_NOISE_BAND``; the band edges are
    inclusive, so a 1.20x (or 1/1.2x) reading is still "inconclusive".
    The tool never claims a win on data within measurement noise.
    """
    if speedup > _VERDICT_NOISE_BAND:
        return "faster"
    if speedup < 1.0 / _VERDICT_NOISE_BAND:
        return "slower"
    return "inconclusive"


#: Floor for JIT-settling warmup blocks (v0.10): each settling block runs
#: at least this many kernel iterations. Tiered compilation promotes a hot
#: method to C1 after ~10^3 invocations and to C2 after ~10^4, so 1000
#: iterations per block means a few blocks are enough to reach the
#: compiled steady state on typical kernels -- and the stability check
#: itself (not the count) is what guarantees settledness.
_SETTLE_MIN_BLOCK_ITERS = 1000

#: Two consecutive settling-block times within this relative tolerance
#: count as "JIT settled".
_SETTLE_TOL_REL = 0.10

#: Cap on settling blocks: a pathologically drifting workload can never
#: hang the bench; the measurement proceeds on the (noisy) data instead.
_SETTLE_MAX_BLOCKS = 8


def _parse_ok_nanos(resp: str) -> int:
    """Parse a ``"OK <elapsed-nanos>"`` BENCH answer.

    A genuinely malformed protocol line (not "OK ...", or a non-integer
    payload) raises ``RuntimeError("protocol error: ...")`` -- it is never
    misread as a timing.
    """
    if not resp.startswith("OK "):
        raise RuntimeError(f"protocol error: {resp!r}")
    try:
        return int(resp[3:])
    except ValueError:
        raise RuntimeError(f"protocol error: malformed BENCH result: {resp!r}")


def _jit_settle(
    session,
    warm_line: str,
    timeout: float,
    tol_rel: float = _SETTLE_TOL_REL,
    max_blocks: int = _SETTLE_MAX_BLOCKS,
) -> int:
    """Repeat warmup BENCH blocks until the JIT settles (v0.10).

    Each block runs the kernel ``warmup`` times inside the JVM (timed,
    discarded). When two consecutive block times agree within ``tol_rel``
    relative, the hot methods are assumed compiled and the loop stops;
    the loop always stops after ``max_blocks`` blocks. Returns the number
    of blocks run.

    Rationale: identical back-to-back bench runs used to flip the
    faster/slower verdict (Rust dogfood: 5.40x faster, then 1.06x slower)
    because one side's C2 compilation had not finished when measurement
    started. A fixed warmup count cannot guarantee that -- compilation
    time depends on method size and machine load -- so the bench now
    *measures* settledness instead of assuming it. ``session`` is any
    object with ``ask(line, timeout)`` (a :class:`LineSession` in
    production); harness "ERR" answers and malformed protocol lines raise
    ``RuntimeError``.
    """
    prev: int | None = None
    blocks = 0
    for _ in range(max_blocks):
        r = session.ask(warm_line, timeout)
        if r.startswith("ERR"):
            raise RuntimeError(f"harness error during warmup: {r}")
        nanos = _parse_ok_nanos(r)
        blocks += 1
        if prev is not None and abs(nanos - prev) / max(prev, 1) <= tol_rel:
            return blocks
        prev = nanos
    return blocks


@dataclass
class BenchReport:
    method: str
    iters: int
    warmup: int
    java_nanos: list[int]
    jni_nanos: list[int]
    # --- v0.2 additions (all defaulted; old constructions keep working) ---
    jni_overhead_ns: float | None = None  # median ns per empty JNI call
    # --- v0.10 workstream C: the Rust backend measures the same empty-call
    # cost (the JNI transition via the jni crate); kept as a separate field
    # so reports stay unambiguous about which backend measured it.
    rust_overhead_ns: float | None = None  # median ns per empty JNI call, Rust backend
    per_size: list[dict] = field(
        default_factory=list
    )  # [{size, java_ns, jni_ns, speedup}] (median per-call ns)
    break_even_size: int | None = None  # derived from per_size when not given
    rigorous_note: str = ""
    ci95_half_width_ns: float | None = None  # 95% CI half-width, trimmed JNI per-call

    def __post_init__(self) -> None:
        if self.break_even_size is None and self.per_size:
            self.break_even_size = _break_even_size(self.per_size)

    @property
    def java_per_call_ns(self) -> float:
        return statistics.median(self.java_nanos) / self.iters

    @property
    def jni_per_call_ns(self) -> float:
        return statistics.median(self.jni_nanos) / self.iters

    @property
    def speedup(self) -> float:
        j = self.jni_per_call_ns
        return self.java_per_call_ns / j if j > 0 else float("inf")

    @property
    def call_overhead_ns(self) -> float | None:
        """Measured empty-call overhead, whichever backend measured it.

        The Rust backend crosses the same JNI boundary (via the jni crate),
        so its empty-call cost is directly comparable to the C++ backend's.
        """
        if self.jni_overhead_ns is not None:
            return self.jni_overhead_ns
        return self.rust_overhead_ns

    @property
    def compute_ns(self) -> float:
        """Per-call time attributable to the kernel itself.

        Subtracts the measured empty-call overhead (the native call
        transition cost); floored at 0 so an imprecise overhead estimate
        can never go negative. Without a measured overhead this is just
        jni_per_call_ns.
        """
        base = self.jni_per_call_ns
        overhead = self.call_overhead_ns
        if overhead is None:
            return base
        return max(0.0, base - overhead)

    @property
    def verdict(self) -> str:
        """``"faster"`` / ``"slower"`` / ``"inconclusive"`` (v0.10).

        Applies the noise band (see :data:`_VERDICT_NOISE_BAND`) to the
        median speedup, so the tool never claims a win on data within
        measurement noise. The band is applied to the *reported* medians,
        which are IQR-trimmed under ``--rigorous`` -- plain and rigorous
        mode therefore share one verdict rule.
        """
        return verdict_for_speedup(self.speedup)

    def verdict_text(self, native_label: str = "JNI C++") -> str:
        """Honest one-line verdict for CLI / JSON / HTML rendering.

        Never claims a win inside the noise band: in-band readings get an
        explicit "inconclusive -- too close to call" instead of a
        faster/slower claim.
        """
        sp = self.speedup
        v = self.verdict
        band = f"{_VERDICT_NOISE_BAND:.1f}"
        if v == "faster":
            return (
                f"{native_label} is {sp:.2f}x faster "
                f"(median; outside the {band}x noise band)."
            )
        if v == "slower":
            return (
                f"{native_label} is {1.0 / sp:.2f}x SLOWER "
                f"(median; outside the {band}x noise band) -- not worth porting."
            )
        return (
            f"inconclusive: {native_label} {sp:.2f}x median speedup is inside "
            f"the {band}x noise band -- too close to call; not a porting win."
        )


def _bench_samples(
    s_java: LineSession,
    s_jni: LineSession,
    method: J.MethodInfo,
    args: list,
    iters: int,
    warmup: int,
    repeats: int,
    timeout: float,
    settle_tol_rel: float = _SETTLE_TOL_REL,
    settle_max_blocks: int = _SETTLE_MAX_BLOCKS,
) -> tuple[list[int], list[int]]:
    """Timed BENCH loop for one input; returns (java_nanos, jni_nanos).

    Same contract as v0.1 run_bench: raises RuntimeError when the input is
    invalid for the Java original or a harness misbehaves. v0.10 adds a
    JIT-settling phase (see :func:`_jit_settle`): before the measurement
    loop, each session runs warmup blocks until two consecutive block
    times agree within ``settle_tol_rel`` (capped at ``settle_max_blocks``
    blocks), so measurement never starts on an uncompiled method. The
    settling block runs at least ``_SETTLE_MIN_BLOCK_ITERS`` iterations
    regardless of the caller's ``warmup`` value -- a 200-iteration warmup
    is below the tiered-compilation thresholds and was the v0.9 flakiness
    source. The per-repeat warmup still uses ``warmup`` (cheap, keeps
    caches hot once the JIT has settled).
    """
    # sanity: the representative input must be valid for the Java original
    probe = s_java.ask(encode_call(method, args), timeout)
    if probe.startswith("ERR"):
        raise RuntimeError(
            f"generated benchmark input is invalid for the Java original: {probe}. "
            "This is a v0.1 input-heuristic limitation."
        )
    if not probe.startswith("OK "):
        raise RuntimeError(f"protocol error: {probe!r}")
    arg_str = " ".join(encode_arg(p.jtype, v) for p, v in zip(method.params, args))
    bench_line = f"BENCH {iters} " + arg_str
    settle_line = f"BENCH {max(warmup, _SETTLE_MIN_BLOCK_ITERS)} " + arg_str
    warm_line = f"BENCH {warmup} " + arg_str
    _jit_settle(s_java, settle_line, timeout, settle_tol_rel, settle_max_blocks)
    _jit_settle(s_jni, settle_line, timeout, settle_tol_rel, settle_max_blocks)
    j_ns: list[int] = []
    n_ns: list[int] = []
    for rep in range(repeats):
        for sess, store in ((s_java, j_ns), (s_jni, n_ns)):
            r = sess.ask(warm_line, timeout)  # warmup, discarded
            if r.startswith("ERR"):
                raise RuntimeError(f"harness error during warmup: {r}")
            _parse_ok_nanos(r)  # discarded; validates the protocol line
            r = sess.ask(bench_line, timeout)
            if r.startswith("ERR"):
                raise RuntimeError(f"harness error during bench: {r}")
            store.append(_parse_ok_nanos(r))
    return j_ns, n_ns


def _size_entry(
    size: int,
    java_ns: list[int],
    jni_ns: list[int],
    iters: int,
    trim: bool = False,
) -> dict:
    if trim:
        java_ns = _iqr_trim(java_ns)
        jni_ns = _iqr_trim(jni_ns)
    jm = statistics.median(java_ns) / iters
    nm = statistics.median(jni_ns) / iters
    return {
        "size": size,
        "java_ns": jm,
        "jni_ns": nm,
        "speedup": jm / nm if nm > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# rigorous mode: JMH / Google Benchmark detection + honest wall-clock fallback
# ---------------------------------------------------------------------------


def _find_jmh_jar() -> Path | None:
    """Look for a JMH core jar: ~/.m2, `jmh` on PATH, then a bounded find."""
    try:
        m2 = Path.home() / ".m2" / "repository"
        if m2.is_dir():
            for p in sorted(m2.glob("**/jmh-core-*.jar")):
                return p
        jmh = shutil.which("jmh")
        if jmh:
            return Path(jmh)
        r = subprocess.run(
            ["find", "/", "-maxdepth", "4", "-name", "jmh-core-*.jar"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        for line in r.stdout.splitlines():
            line = line.strip()
            if line:
                return Path(line)
    except Exception:
        pass
    return None


def _google_benchmark_available() -> bool:
    """True when <benchmark/benchmark.h> compiles (header check only)."""
    import tempfile

    try:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "gb_probe.cpp"
            src.write_text(
                "#include <benchmark/benchmark.h>\nint main() { return 0; }\n",
                encoding="utf-8",
            )
            r = subprocess.run(
                [_which_or_raise("g++"), "-fsyntax-only", str(src)],
                capture_output=True,
                timeout=60,
            )
            # NOTE: -fsyntax-only proves the headers exist; linking a real
            # benchmark would additionally need -lbenchmark.
            return r.returncode == 0
    except Exception:
        return False


def _iqr_trim(samples: list) -> list:
    """Drop samples outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

    With fewer than 4 samples there is no meaningful IQR, and the trimmed
    list never shrinks below 3 points (a median of 1-2 survivors is noise,
    not rigor).
    """
    if len(samples) < 4:
        return list(samples)
    q1, _q2, q3 = statistics.quantiles(samples, n=4, method="exclusive")
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    trimmed = [s for s in samples if lo <= s <= hi]
    return trimmed if len(trimmed) >= 3 else list(samples)


def _ci95_half_width(per_call_ns: list[float]) -> float | None:
    """95% CI half-width of the mean via normal approx (1.96*sd/sqrt(n))."""
    if len(per_call_ns) < 2:
        return None
    return 1.96 * statistics.pstdev(per_call_ns) / math.sqrt(len(per_call_ns))


def _apply_rigorous(
    report: BenchReport,
    size_samples: list[tuple[int, list[int], list[int]]],
) -> BenchReport:
    """Enhanced wall-clock mode used whenever rigorous=True is requested.

    JMH / Google Benchmark are *detected* but never faked: when neither is
    installed the report says exactly that and falls back to IQR outlier
    trimming plus a 95% CI half-width. When one IS installed the note names
    it, and the same enhanced wall-clock numbers are still reported --
    presenting wall-clock data as JMH output would be dishonest.
    """
    jmh_jar = _find_jmh_jar()
    gb = _google_benchmark_available()
    # NOTE (future work): when a real framework is present, a later pass
    # could shell out to it instead of the wall-clock loop, e.g.
    #   java -cp <jmh-core.jar>:<classes> org.openjdk.jmh.Main <BenchClass>
    # or compile a google/benchmark driver that calls the JNI wrapper and
    # parse its JSON output. v0.1 deliberately does NOT do that: it reports
    # the enhanced wall-clock numbers below and says so, rather than
    # presenting wall-clock data as framework results.
    if jmh_jar is not None or gb:
        found = []
        if jmh_jar is not None:
            found.append(f"JMH ({jmh_jar})")
        if gb:
            found.append("Google Benchmark headers")
        report.rigorous_note = (
            f"Detected {' and '.join(found)}; v0.1 still uses the enhanced "
            "wall-clock (IQR outlier trimming + 95% CI) -- shelling out to a "
            "real framework is future work (see _apply_rigorous comments)."
        )
    else:
        report.rigorous_note = (
            "JMH/Google Benchmark not available; using wall-clock with "
            "IQR outlier trimming and 95% CI"
        )
    # In rigorous mode the stored sample lists hold the IQR-trimmed samples,
    # so every downstream number (medians, speedup, compute_ns) is computed
    # on cleaned data.
    report.java_nanos = _iqr_trim(report.java_nanos)
    report.jni_nanos = _iqr_trim(report.jni_nanos)
    report.ci95_half_width_ns = _ci95_half_width(
        [ns / report.iters for ns in report.jni_nanos]
    )
    report.per_size = [
        _size_entry(size, sj, sn, report.iters, trim=True)
        for size, sj, sn in sorted(size_samples, key=lambda t: t[0])
    ]
    report.break_even_size = _break_even_size(report.per_size)
    return report


# ---------------------------------------------------------------------------
# JNI overhead: empty-call cost measured with a second tiny .so
# ---------------------------------------------------------------------------

_NOOP_JAVA = """\
public class KpNoop {
    static { System.loadLibrary("kpnoop"); }
    public static native void kpNoop();
    public static void main(String[] args) throws Exception {
        int iters = Integer.parseInt(args[0]);
        for (int w = 0; w < 200000; w++) kpNoop(); // warmup
        long t0 = System.nanoTime();
        for (int i = 0; i < iters; i++) kpNoop();
        long t1 = System.nanoTime();
        System.out.println("OK " + (t1 - t0));
    }
}
"""

_NOOP_CPP = """\
#include <jni.h>
extern "C" {
JNIEXPORT void JNICALL Java_KpNoop_kpNoop(JNIEnv*, jclass) {}
} // extern "C"
"""

_NOOP_RS_CARGO = """\
[package]
name = "kpnoop"
version = "0.1.0"
edition = "2021"

[lib]
name = "kpnoop"
crate-type = ["cdylib"]

[dependencies]
jni = "0.21"
"""

_NOOP_RS_LIB = """\
// Empty JNI entry point for call-overhead measurement: the same JNI
// transition the real kernels cross (via the jni crate), minus any compute.
use jni::objects::JClass;
use jni::JNIEnv;

#[no_mangle]
pub extern "system" fn Java_KpNoop_kpNoop<'local>(
    _env: JNIEnv<'local>,
    _class: JClass<'local>,
) {
}
"""


def _build_noop_jni_lib(env: BuiltEnv) -> None:
    """Compile the empty ``kpNoop`` JNI method to ``work/libkpnoop.so``."""
    work = Path(env.work)
    (work / "kpnoop.cpp").write_text(_NOOP_CPP, encoding="utf-8")
    compile_jni(work / "kpnoop.cpp", work / "libkpnoop.so")


def _build_noop_rust_lib(env: BuiltEnv) -> None:
    """Compile the empty ``kpNoop`` JNI entry point (jni crate) to
    ``work/libkpnoop.so``. Raises on failure; the caller turns that into a
    skipped overhead measurement."""
    from . import rust_glue as _rust

    work = Path(env.work)
    proj = work / "kpnoop-rs"
    (proj / "src").mkdir(parents=True, exist_ok=True)
    (proj / "Cargo.toml").write_text(_NOOP_RS_CARGO, encoding="utf-8")
    (proj / "src" / "lib.rs").write_text(_NOOP_RS_LIB, encoding="utf-8")
    cargo = _rust._find_cargo()
    if cargo is None:
        raise RuntimeError("cargo not found; cannot measure Rust call overhead")
    cenv = dict(os.environ)
    # Same target-dir policy as compile_rust: per-project default, an
    # explicit CARGO_TARGET_DIR honored ("kpnoop" is a unique crate name,
    # so a shared target dir is safe).
    cenv["CARGO_TARGET_DIR"] = str(_rust._target_dir(proj))
    cmd = [
        str(cargo), "build", "--release",
        "--manifest-path", str(proj / "Cargo.toml"),
    ]
    if "CARGO_NET_OFFLINE" not in cenv and _rust._jni_crate_cached():
        cmd.append("--offline")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=cenv)
    if proc.returncode != 0:
        raise RuntimeError(
            f"cargo build of the noop crate failed (exit {proc.returncode}):\n"
            f"{_rust._tail(proc.stderr)}"
        )
    lib = Path(cenv["CARGO_TARGET_DIR"]) / "release" / _rust._cdylib_name("kpnoop")
    if not lib.exists():
        raise RuntimeError(f"cargo build succeeded but {lib} is missing")
    shutil.copy2(lib, work / "libkpnoop.so")


def _measure_call_overhead(
    env: BuiltEnv,
    build_native,
    iters: int = 1_000_000,
    repeats: int = 3,
    timeout: float = 120.0,
) -> float | None:
    """Median nanoseconds per *empty* native call.

    ``build_native(env)`` materializes ``work/libkpnoop.so`` (one empty
    native ``kpNoop`` method); the tiny ``KpNoop`` Java driver is shared.
    Returns None -- letting the bench continue -- when anything fails.
    """
    try:
        work = Path(env.work)
        (work / "KpNoop.java").write_text(_NOOP_JAVA, encoding="utf-8")
        build_native(env)
        compile_java([work / "KpNoop.java"], env.classes)
        java = _which_or_raise("java")
        per_call: list[float] = []
        for _ in range(repeats):
            r = subprocess.run(
                [
                    java,
                    f"-Djava.library.path={work}",
                    "-cp",
                    str(env.classes),
                    "KpNoop",
                    str(iters),
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = r.stdout.strip()
            if r.returncode != 0 or not out.startswith("OK "):
                return None
            per_call.append(int(out[3:]) / iters)
        return statistics.median(per_call)
    except Exception:
        return None


def _measure_jni_overhead(
    env: BuiltEnv,
    iters: int = 1_000_000,
    repeats: int = 3,
    timeout: float = 120.0,
) -> float | None:
    """Median nanoseconds per *empty* JNI call.

    Built at bench time WITHOUT touching jni.py: a minimal extra JNI
    source (one empty native method ``kpNoop``) is generated in the work
    dir, compiled to a second .so (``libkpnoop.so``), loaded from a tiny
    Java driver, and timed in a ``System.nanoTime`` loop. Returns None --
    letting the bench continue -- when anything fails.
    """
    return _measure_call_overhead(env, _build_noop_jni_lib, iters, repeats, timeout)


def _measure_rust_overhead(
    env: BuiltEnv,
    iters: int = 1_000_000,
    repeats: int = 3,
    timeout: float = 300.0,
) -> float | None:
    """Median nanoseconds per *empty* JNI call through the Rust backend.

    Same empty-call protocol as :func:`_measure_jni_overhead`, but the
    noop entry point is a Rust cdylib built with the jni crate -- i.e. the
    exact call boundary the Rust kernels cross. Returns None -- letting
    the bench continue -- when anything fails (no cargo, build error,
    driver failure).
    """
    return _measure_call_overhead(env, _build_noop_rust_lib, iters, repeats, timeout)


def run_bench(
    env: BuiltEnv,
    iters: int = 2000,
    warmup: int = 200,
    repeats: int = 5,
    seed: int = 1234,
    timeout: float = 300.0,
    sizes: list[int] | None = None,
    rigorous: bool = False,
    overhead: bool = True,
    settle_tol_rel: float = _SETTLE_TOL_REL,
    settle_max_blocks: int = _SETTLE_MAX_BLOCKS,
) -> BenchReport:
    """Wall-clock benchmark: Java original vs JNI translation.

    New in v0.2 (all optional; defaults keep v0.1 behavior):

    - ``sizes``: when given, the bench is repeated at each array length
      (``array_len`` is passed through to :func:`gen_bench_args`) and the
      per-size medians recorded in ``report.per_size`` as
      ``{size, java_ns, jni_ns, speedup}``; ``report.break_even_size`` is
      the smallest size beating the 1.2x heuristic bar, else None.
    - ``rigorous``: detect JMH / Google Benchmark. When neither is present
      (the usual case) an *honest* enhanced wall-clock is used instead:
      IQR outlier trimming plus a 95% CI half-width, with
      ``report.rigorous_note`` saying exactly what ran. Framework results
      are never faked (see :func:`_apply_rigorous`).
    - ``overhead``: measure the empty-native-call cost into
      ``report.jni_overhead_ns`` (JNI backend) or
      ``report.rust_overhead_ns`` (Rust backend) -- a second tiny .so
      built at bench time; None when the noop build fails, and the bench
      continues.

    New in v0.10:

    - ``report.verdict`` is ``"faster"`` / ``"slower"`` / ``"inconclusive"``
      (see :func:`verdict_for_speedup`): the tool only claims
      faster/slower when the median speedup lies outside the 1.2x noise
      band, and ``report.verdict_text(label)`` renders the honest one-line
      verdict for CLI/JSON/HTML. The band applies to the reported
      (IQR-trimmed under ``--rigorous``) medians, so ``--rigorous`` and
      plain mode share one verdict rule.
    - ``settle_tol_rel`` / ``settle_max_blocks``: JIT-settling detection
      (see :func:`_jit_settle`). Before measuring, each JVM runs warmup
      blocks until two consecutive block times agree within
      ``settle_tol_rel`` (capped at ``settle_max_blocks`` blocks), so a
      run can no longer measure an uncompiled method. The settling block
      is at least ``_SETTLE_MIN_BLOCK_ITERS`` iterations even when the
      caller's ``warmup`` is smaller.
    """
    s_java, s_jni = open_sessions(env)
    try:
        args = gen_bench_args(env.method, seed=seed)
        j_ns, n_ns = _bench_samples(
            s_java, s_jni, env.method, args, iters, warmup, repeats, timeout,
            settle_tol_rel, settle_max_blocks,
        )
        size_samples: list[tuple[int, list[int], list[int]]] = []
        if sizes:
            # NOTE: for matMul-shaped methods (a size-like scalar parameter)
            # the array length comes from gen_case's sqrt heuristic, so
            # `size` is only an approximate knob there; for plain array
            # kernels it is exact.
            for size in sorted(set(sizes)):
                s_args = gen_bench_args(env.method, seed=seed, array_len=size)
                try:
                    sj, sn = _bench_samples(
                        s_java, s_jni, env.method, s_args,
                        iters, warmup, repeats, timeout,
                        settle_tol_rel, settle_max_blocks,
                    )
                except RuntimeError:
                    continue  # invalid input at this size; skip it
                size_samples.append((size, sj, sn))
    finally:
        s_java.close()
        s_jni.close()
    jni_overhead_ns: float | None = None
    rust_overhead_ns: float | None = None
    if overhead:
        # The Rust backend crosses the same JNI boundary (via the jni
        # crate); its empty-call cost is measured with a tiny Rust noop
        # cdylib so the overhead split is comparable across backends.
        if env.backend == "rust":
            rust_overhead_ns = _measure_rust_overhead(env)
        elif env.backend == "jni":
            jni_overhead_ns = _measure_jni_overhead(env)
    report = BenchReport(
        method=env.method.qualname,
        iters=iters,
        warmup=warmup,
        java_nanos=j_ns,
        jni_nanos=n_ns,
        jni_overhead_ns=jni_overhead_ns,
        rust_overhead_ns=rust_overhead_ns,
        per_size=[_size_entry(size, sj, sn, iters) for size, sj, sn in size_samples],
    )
    if rigorous:
        report = _apply_rigorous(report, size_samples)
    return report


def fmt_ns(ns: float) -> str:
    if ns >= 1e9:
        return f"{ns/1e9:.3f} s"
    if ns >= 1e6:
        return f"{ns/1e6:.3f} ms"
    if ns >= 1e3:
        return f"{ns/1e3:.1f} us"
    return f"{ns:.0f} ns"
