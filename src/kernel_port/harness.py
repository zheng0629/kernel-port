"""Compile-and-run harness for `verify` and `bench`.

Protocol: kernel-port drives two long-lived JVM subprocesses over stdin/stdout
(one calling the Java original, one calling the JNI translation). Each input
line is a command; each output line is a result:

    CALL <arg> <arg> ...          ->  OK <result> | ERR <message>
    BENCH <iters> <arg> <arg> ... ->  OK <elapsed-nanos> | ERR <message>

Argument/result encoding (space-separated tokens):

    i:42  l:123  d:1.5  f:0.5  z:1      scalar int/long/double/float/boolean
    ai:1,2,3  al:..  ad:..  af:..       int[]/long[]/double[]/float[] (ai: = empty)
    v                                    void result

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
import queue
import random
import re
import shutil
import statistics
import subprocess
import sys
import threading
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


def compile_java(sources: list[Path], classes_dir: Path) -> None:
    javac = _which_or_raise("javac")
    classes_dir.mkdir(parents=True, exist_ok=True)
    cmd = [javac, "-nowarn", "-d", str(classes_dir)] + [str(s) for s in sources]
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

# harness-supported types (subset of the glue type map)
_HARNESS_TYPES = {
    "int",
    "long",
    "float",
    "double",
    "boolean",
    "int[]",
    "long[]",
    "float[]",
    "double[]",
}


def check_harness_types(method: J.MethodInfo) -> None:
    bad = [p.jtype for p in method.params if p.jtype not in _HARNESS_TYPES]
    if method.jtype != "void" and method.jtype not in _HARNESS_TYPES:
        bad.append(f"return {method.jtype}")
    if bad:
        raise ValueError(
            f"verify/bench v0.1 does not support: {', '.join(bad)} "
            "(supported: int/long/float/double/boolean and their 1-D arrays)"
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
    raise AssertionError(jtype)


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
        if p.jtype.endswith("[]"):
            elem = p.jtype[:-2]
            if elem not in len_by_elem:
                if bench and has_size_param:
                    s = rng.randint(8, 32)
                    len_by_elem[elem] = s * s
                else:
                    len_by_elem[elem] = rng.randint(0, array_len)
    first_len = next(iter(len_by_elem.values()), 0)
    args: list = []
    for p in method.params:
        if p.jtype.endswith("[]"):
            elem = p.jtype[:-2]
            L = len_by_elem[elem]
            args.append([_rand_scalar(rng, elem, False) for _ in range(L)])
        else:
            size_like = p.name.lower() in _SIZE_NAMES
            v = _rand_scalar(rng, p.jtype, size_like, bench)
            if size_like and p.jtype == "int" and len_by_elem:
                if first_len:
                    # matMul(a,b,c,n) shape: n*n == len  ->  n = sqrt(len)
                    import math

                    s = int(math.isqrt(first_len))
                    v = s if s * s == first_len else first_len
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


def gen_boundary_cases(
    method: J.MethodInfo, rng: random.Random | None = None
) -> list[list]:
    """Deterministic boundary-value argument lists for `method`.

    Edge values per parameter type:

    - int/long: 0, 1, -1, MIN, MAX
    - float/double: 0.0, -0.0, 1.0, -1.0, NaN, +Inf, -Inf, tiny, max
    - boolean: False, True
    - arrays: [], [zero], [single value], large (1024-element) array

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
        if p.jtype.endswith("[]"):
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
    if jtype in ("int[]", "long[]", "float[]", "double[]"):
        tag = {"int[]": "ai", "long[]": "al", "float[]": "af", "double[]": "ad"}[jtype]
        inner = ",".join(
            repr(float(x)) if jtype in ("float[]", "double[]") else str(int(x))
            for x in value
        )
        return f"{tag}:{inner}"
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
    raise ValueError(f"bad result token: {tok}")


def results_equal(method: J.MethodInfo, java_tok: str, jni_tok: str) -> tuple[bool, str]:
    """Compare two result tokens; floats compare with tolerance."""

    def close(a: float, b: float) -> bool:
        import math

        if math.isnan(a) and math.isnan(b):
            return True
        if math.isinf(a) or math.isinf(b):
            return a == b
        return abs(a - b) <= 1e-9 + 1e-9 * max(abs(a), abs(b))

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
            if tj != tn or not isinstance(vj, list) or not isinstance(vn, list):
                return False, f"array type mismatch: {js!r} vs {ns!r}"
            if len(vj) != len(vn):
                return False, f"array length mismatch: {len(vj)} vs {len(vn)}"
            floaty = tj in ("af", "ad")
            for a, b in zip(vj, vn):
                if floaty:
                    if not close(a, b):
                        return False, f"array element mismatch: {a} != {b}"
                elif a != b:
                    return False, f"array element mismatch: {a} != {b}"
        return True, ""
    try:
        tj, vj = _parse_result_token(java_tok)
        tn, vn = _parse_result_token(jni_tok)
    except ValueError as e:
        return False, str(e)

    if tj in ("ai", "al"):
        ok = isinstance(vn, list) and len(vj) == len(vn) and all(x == y for x, y in zip(vj, vn))
        return ok, "" if ok else f"array mismatch: {vj} vs {vn}"
    if tj in ("af", "ad"):
        ok = (
            isinstance(vn, list)
            and len(vj) == len(vn)
            and all(close(x, y) for x, y in zip(vj, vn))
        )
        return ok, "" if ok else f"array mismatch: {vj} vs {vn}"
    if tj in ("float", "double"):
        ok = close(vj, vn)
        return ok, "" if ok else f"{vj} != {vn}"
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
    "int[]": "int[]",
    "long[]": "long[]",
    "float[]": "float[]",
    "double[]": "double[]",
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
    if jtype == "int[]":
        return f"int[] {var} = parseIntArray({tok}.substring(3));"
    if jtype == "long[]":
        return f"long[] {var} = parseLongArray({tok}.substring(3));"
    if jtype == "float[]":
        return f"float[] {var} = parseFloatArray({tok}.substring(3));"
    if jtype == "double[]":
        return f"double[] {var} = parseDoubleArray({tok}.substring(3));"
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
    if jtype == "int[]":
        return f'"ai:" + join({var})'
    if jtype == "long[]":
        return f'"al:" + join({var})'
    if jtype == "float[]":
        return f'"af:" + join({var})'
    if jtype == "double[]":
        return f'"ad:" + join({var})'
    raise AssertionError(jtype)


def render_harness(method: J.MethodInfo, target: str, class_name: str) -> str:
    """Render HarnessJava/HarnessJni source.

    target='java' calls com.example.Kernels.dot(...);
    target='jni'  calls KernelsNative.dot(...).
    """
    receiver = method.class_qualname if target == "java" else f"{method.class_qualname.rsplit('.',1)[-1]}Native"
    params = method.params
    is_void = method.jtype == "void"

    decls = []
    for i, p in enumerate(params):
        decls.append("        " + _decode_java(p.jtype, f"b_a{i}", f"tok[off+{i}]"))

    # bench loop: re-clone arrays each iteration so in/out params start clean
    loop_setup = []
    call_args = []
    for i, p in enumerate(params):
        if p.jtype.endswith("[]"):
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
}}
"""


# ---------------------------------------------------------------------------
# subprocess session with per-line timeout
# ---------------------------------------------------------------------------


class LineSession:
    """A subprocess speaking the line protocol, with per-line timeouts."""

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
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError:
            raise RuntimeError("harness process died (broken pipe)")
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"harness did not answer within {timeout}s: {line[:80]}")

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


def build_env(
    repo_root: Path,
    method: J.MethodInfo,
    glue_dir: Path,
    work: Path,
) -> BuiltEnv:
    """Compile fixture + harnesses + JNI lib. Raises RuntimeError on failure."""
    check_harness_types(method)
    glue_dir = Path(glue_dir)
    cpp = glue_dir / "kernels_jni.cpp"
    native_java = next(glue_dir.glob("*Native.java"), None)
    if not cpp.exists():
        raise RuntimeError(f"{cpp} not found -- run `kernel-port glue` first")
    if native_java is None:
        raise RuntimeError(f"no *Native.java in {glue_dir} -- run `kernel-port glue` first")

    work = Path(work)
    if work.exists():
        shutil.rmtree(work)
    classes = work / "classes"
    classes.mkdir(parents=True)

    java_sources = sorted(repo_root.rglob("*.java"))
    if not java_sources:
        raise RuntimeError(f"no .java files under {repo_root}")
    hj = work / "HarnessJava.java"
    hn = work / "HarnessJni.java"
    hj.write_text(render_harness(method, "java", "HarnessJava"), encoding="utf-8")
    hn.write_text(render_harness(method, "jni", "HarnessJni"), encoding="utf-8")
    compile_java(java_sources + [hj, hn, native_java], classes)

    so = work / "libkernels.so"
    compile_jni(cpp, so)
    return BuiltEnv(work=work, classes=classes, method=method)


def open_sessions(env: BuiltEnv) -> tuple[LineSession, LineSession]:
    java = _which_or_raise("java")
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


@dataclass
class VerifyReport:
    method: str
    cases: list[VerifyCase]
    passed: int
    failed: int
    skipped: int = 0
    boundary: list[VerifyCase] = field(default_factory=list)
    boundary_skipped: int = 0

    @property
    def all_passed(self) -> bool:
        # Boundary mismatches fail the run too: they are real translation
        # bugs caught by the edge-case sweep. With boundary=False the list
        # is empty, so this reduces to the v0.1 definition.
        return self.failed == 0 and self.boundary_failed == 0

    @property
    def boundary_passed(self) -> int:
        return sum(1 for c in self.boundary if c.ok)

    @property
    def boundary_failed(self) -> int:
        return sum(1 for c in self.boundary if not c.ok)


def _run_one_case(
    s_java: LineSession,
    s_jni: LineSession,
    method: J.MethodInfo,
    index: int,
    args: list,
    timeout: float,
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
    ok, detail = results_equal(method, rj[3:], rn[3:])
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
        out_dir = Path(env.work) / ".kp-work"
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
                s_java, s_jni, env.method, i, args, per_case_timeout
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
                    s_java, s_jni, env.method, bi, bargs, per_case_timeout
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


@dataclass
class BenchReport:
    method: str
    iters: int
    warmup: int
    java_nanos: list[int]
    jni_nanos: list[int]
    # --- v0.2 additions (all defaulted; old constructions keep working) ---
    jni_overhead_ns: float | None = None  # median ns per empty JNI call
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
    def compute_ns(self) -> float:
        """JNI per-call time attributable to the kernel itself.

        Subtracts the measured empty-call overhead (the JNI transition
        cost); floored at 0 so an imprecise overhead estimate can never go
        negative. Without a measured overhead this is just jni_per_call_ns.
        """
        base = self.jni_per_call_ns
        if self.jni_overhead_ns is None:
            return base
        return max(0.0, base - self.jni_overhead_ns)


def _bench_samples(
    s_java: LineSession,
    s_jni: LineSession,
    method: J.MethodInfo,
    args: list,
    iters: int,
    warmup: int,
    repeats: int,
    timeout: float,
) -> tuple[list[int], list[int]]:
    """Timed BENCH loop for one input; returns (java_nanos, jni_nanos).

    Same contract as v0.1 run_bench: raises RuntimeError when the input is
    invalid for the Java original or a harness misbehaves.
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
    bench_line = f"BENCH {iters} " + " ".join(
        encode_arg(p.jtype, v) for p, v in zip(method.params, args)
    )
    warm_line = f"BENCH {warmup} " + " ".join(
        encode_arg(p.jtype, v) for p, v in zip(method.params, args)
    )
    j_ns: list[int] = []
    n_ns: list[int] = []
    for rep in range(repeats):
        for sess, store in ((s_java, j_ns), (s_jni, n_ns)):
            r = sess.ask(warm_line, timeout)  # warmup, discarded
            if r.startswith("ERR"):
                raise RuntimeError(f"harness error during warmup: {r}")
            r = sess.ask(bench_line, timeout)
            if r.startswith("ERR"):
                raise RuntimeError(f"harness error during bench: {r}")
            if not r.startswith("OK "):
                raise RuntimeError(f"protocol error: {r!r}")
            store.append(int(r[3:]))
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
    try:
        work = Path(env.work)
        (work / "KpNoop.java").write_text(_NOOP_JAVA, encoding="utf-8")
        (work / "kpnoop.cpp").write_text(_NOOP_CPP, encoding="utf-8")
        compile_java([work / "KpNoop.java"], env.classes)
        compile_jni(work / "kpnoop.cpp", work / "libkpnoop.so")
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
    - ``overhead``: measure the empty-JNI-call cost into
      ``report.jni_overhead_ns`` (second tiny .so built at bench time);
      None when the noop build fails, and the bench continues.
    """
    s_java, s_jni = open_sessions(env)
    try:
        args = gen_bench_args(env.method, seed=seed)
        j_ns, n_ns = _bench_samples(
            s_java, s_jni, env.method, args, iters, warmup, repeats, timeout
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
                    )
                except RuntimeError:
                    continue  # invalid input at this size; skip it
                size_samples.append((size, sj, sn))
    finally:
        s_java.close()
        s_jni.close()
    overhead_ns = _measure_jni_overhead(env) if overhead else None
    report = BenchReport(
        method=env.method.qualname,
        iters=iters,
        warmup=warmup,
        java_nanos=j_ns,
        jni_nanos=n_ns,
        jni_overhead_ns=overhead_ns,
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
