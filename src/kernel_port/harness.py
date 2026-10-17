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

import queue
import random
import shutil
import statistics
import subprocess
import sys
import threading
from dataclasses import dataclass
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


def encode_arg(jtype: str, value) -> str:
    if jtype == "int":
        return f"i:{int(value)}"
    if jtype == "long":
        return f"l:{int(value)}"
    if jtype == "float":
        return f"f:{float(value)!r}"
    if jtype == "double":
        return f"d:{float(value)!r}"
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
    if java_tok == "v" or jni_tok == "v":
        return (java_tok == jni_tok, "" if java_tok == jni_tok else "void mismatch")
    try:
        tj, vj = _parse_result_token(java_tok)
        tn, vn = _parse_result_token(jni_tok)
    except ValueError as e:
        return False, str(e)

    def close(a: float, b: float) -> bool:
        import math

        if math.isnan(a) and math.isnan(b):
            return True
        if math.isinf(a) or math.isinf(b):
            return a == b
        return abs(a - b) <= 1e-9 + 1e-9 * max(abs(a), abs(b))

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
        direct_stmt = f"{call_direct};\n            return \"OK v\";"
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


@dataclass
class VerifyReport:
    method: str
    cases: list[VerifyCase]
    passed: int
    failed: int
    skipped: int = 0

    @property
    def all_passed(self) -> bool:
        return self.failed == 0


def run_verify(
    env: BuiltEnv,
    cases: int = 100,
    seed: int = 42,
    per_case_timeout: float = 30.0,
) -> VerifyReport:
    rng = random.Random(seed)
    s_java, s_jni = open_sessions(env)
    results: list[VerifyCase] = []
    skipped = 0
    attempts = 0
    max_attempts = cases * 3 + 10
    try:
        while len(results) < cases and attempts < max_attempts:
            attempts += 1
            i = attempts - 1
            args = gen_case(env.method, rng)
            line = encode_call(env.method, args)
            try:
                rj = s_java.ask(line, per_case_timeout)
            except (TimeoutError, RuntimeError) as e:
                results.append(VerifyCase(i, args, False, f"java harness error: {e}"))
                continue
            if rj.startswith("ERR"):
                # The Java original itself rejects this input (out of the
                # kernel's valid domain): discard the case without feeding
                # it to native code, which has no bounds checking.
                skipped += 1
                continue
            if not rj.startswith("OK "):
                results.append(VerifyCase(i, args, False, f"protocol error: {rj!r}"))
                continue
            try:
                rn = s_jni.ask(line, per_case_timeout)
            except (TimeoutError, RuntimeError) as e:
                results.append(VerifyCase(i, args, False, f"jni harness error: {e}"))
                continue
            if rn.startswith("ERR"):
                results.append(VerifyCase(i, args, False, f"jni harness threw: {rn}"))
                continue
            if not rn.startswith("OK "):
                results.append(VerifyCase(i, args, False, f"protocol error: {rn!r}"))
                continue
            ok, detail = results_equal(env.method, rj[3:], rn[3:])
            results.append(VerifyCase(i, args, ok, detail))
    finally:
        s_java.close()
        s_jni.close()
    passed = sum(1 for c in results if c.ok)
    return VerifyReport(
        method=env.method.qualname,
        cases=results,
        passed=passed,
        failed=len(results) - passed,
        skipped=skipped,
    )


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------


@dataclass
class BenchReport:
    method: str
    iters: int
    warmup: int
    java_nanos: list[int]
    jni_nanos: list[int]

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


def run_bench(
    env: BuiltEnv,
    iters: int = 2000,
    warmup: int = 200,
    repeats: int = 5,
    seed: int = 1234,
    timeout: float = 300.0,
) -> BenchReport:
    args = gen_bench_args(env.method, seed=seed)
    s_java, s_jni = open_sessions(env)
    j_ns: list[int] = []
    n_ns: list[int] = []
    try:
        # sanity: the representative input must be valid for the Java original
        probe = s_java.ask(encode_call(env.method, args), timeout)
        if probe.startswith("ERR"):
            raise RuntimeError(
                f"generated benchmark input is invalid for the Java original: {probe}. "
                "This is a v0.1 input-heuristic limitation."
            )
        if not probe.startswith("OK "):
            raise RuntimeError(f"protocol error: {probe!r}")
        bench_line = f"BENCH {iters} " + " ".join(
            encode_arg(p.jtype, v) for p, v in zip(env.method.params, args)
        )
        warm_line = f"BENCH {warmup} " + " ".join(
            encode_arg(p.jtype, v) for p, v in zip(env.method.params, args)
        )
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
    finally:
        s_java.close()
        s_jni.close()
    return BenchReport(
        method=env.method.qualname,
        iters=iters,
        warmup=warmup,
        java_nanos=j_ns,
        jni_nanos=n_ns,
    )


def fmt_ns(ns: float) -> str:
    if ns >= 1e9:
        return f"{ns/1e9:.3f} s"
    if ns >= 1e6:
        return f"{ns/1e6:.3f} ms"
    if ns >= 1e3:
        return f"{ns/1e3:.1f} us"
    return f"{ns:.0f} ns"
