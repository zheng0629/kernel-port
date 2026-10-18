"""Differential verify + wall-clock bench for the Python->C cffi backend.

Pipeline: :mod:`kernel_port.cffi_glue` generates ``kernels_cffi.c`` (plain C
ABI, agent-filled bodies) and ``kernels_cffi.py`` (cffi bindings); this
module compiles the C file and differentially tests / benchmarks the
translation against the original Python function.

ARCHITECTURE: test execution happens in a *driver subprocess*
(``python -m kernel_port.py_harness run|bench <spec> <out>``), so a
segfaulting or hung translation cannot take down the caller -- a crashed
driver is reported as a failed run, a hung one trips the parent watchdog.
Inputs are generated in the parent, serialized to JSON (Python-flavoured:
``NaN``/``Infinity`` round-trip because both ends are Python), and results
come back the same way. Per-case wall timeouts use ``SIGALRM`` inside the
driver (Linux-only; without it the timeout is best-effort).

CASE SEMANTICS (documented so failures are interpretable):

- A case where the *original Python* raises or times out is SKIPPED as
  out-of-domain (differential testing is only defined where the original
  is defined). A case where the original returns but the *C side* times
  out or crashes is a FAIL -- that is a broken translation.
- List parameters are IN/OUT (see :mod:`kernel_port.cffi_glue`): mutated
  lists are echoed back from both sides and compared element-wise. The C
  side cannot change a list's length; a length change is a FAIL.
- Scalar comparison: ``int``/``bool`` exact; ``str`` exact; ``float``
  (Python float == C double) with relative tolerance ``--tol-double``
  (default 1e-12); ``--tol-float`` (default 1e-6) is accepted for forward
  compatibility with a future float32 mapping. NaN == NaN counts as equal
  (payload differences ignored); inf must match in sign.
- ``int`` is int64 on the C side: inputs outside int64 range are skipped
  as out-of-domain; results must be int64-closed on the tested inputs --
  Python bignum semantics are NOT emulated, overflow mismatches FAIL.

INPUT GENERATION: random inputs per mapped type (seeded) plus a
deterministic boundary sweep: empty list, 0, negatives, values near the
int64 limits (``2**63-1``, ``-2**63``, ``2**62``), ``inf``/``-inf``/``nan``
and extreme magnitudes for floats, empty/whitespace/unicode strings.
Lists share one length per element kind within a case (the Java harness's
convention), and a size-like int parameter (``n``, ``size``, ...) is
derived from the list length (floor ``isqrt``, so ``n*n <= len`` for
matvec-style shapes) -- so every generated case is in the kernel's
valid domain instead of manufacturing zip-truncation mismatches the C
side cannot see. Infeasible boundary values (e.g. a loop bound of
``2**63-1``) time out on the Python side and are skipped, never failed.

BENCH is honest wall-clock: warmup (discarded) + N timed iterations x
repeats, median per-call comparison Python vs C, remarshalling every
iteration so the C number includes the real cffi call cost. Bench inputs
are small by design (``--list-len``, default 64); ``--sizes`` runs the
bench once per input size and reports the break-even point, mirroring
the Java ``--sizes`` semantics.

LIMITATIONS (v0.7 slice): one kernel per glue dir; no JMH-style rigorous
mode; bench input sizing is fixed, not profile-guided. Coverage-guided
inputs are opt-in since v0.11: ``run_verify(..., coverage=True)`` replays
the round-1 cases under :mod:`kernel_port.py_coverage`'s
``sys.monitoring`` branch recorder and runs a second input round biased
at uncovered edges (see that module for the branch model and the
best-effort degradations).
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .cffi_glue import KernelSig

if TYPE_CHECKING:  # pragma: no cover
    from . import py_coverage

GCC = "/usr/bin/gcc"
SO_NAME = "libkernels_cffi.so"
C_NAME = "kernels_cffi.c"

# int64 limits, shared by the generator and the worker
I64_MIN = -(2**63)
I64_MAX = 2**63 - 1


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


@dataclass
class PyBuiltEnv:
    glue_dir: Path
    work_dir: Path
    so_path: Path
    sig: KernelSig


def build_env(
    glue_dir: Path | str, work_dir: Path | str, sig: KernelSig
) -> PyBuiltEnv:
    """``gcc -shared -fPIC -O2`` the generated C file into the glue dir.

    Refuses to build while the translated body is still the placeholder
    (clearer than a mysterious all-zero verify). Raises RuntimeError.
    """
    from . import cffi_glue as _cg

    gdir = Path(glue_dir)
    work = Path(work_dir)
    c_file = gdir / C_NAME
    if not c_file.exists():
        raise RuntimeError(f"{c_file} not found -- run `kernel-port glue --lang python` first")
    text = c_file.read_text(encoding="utf-8")
    if _cg._PLACEHOLDER in text and _cg._BEGIN in text:
        # placeholder still present AND no agent body spliced in
        body_region = text.split(_cg._BEGIN)[1].split(_cg._END)[0]
        if _cg._PLACEHOLDER in body_region and len(body_region.strip().splitlines()) <= 2:
            raise RuntimeError(
                f"{c_file} still has the placeholder body -- paste the translated "
                "C code into the BEGIN/END TRANSLATED BODY region first"
            )
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    gcc = GCC if Path(GCC).exists() else (shutil.which("gcc") or "gcc")
    so_path = gdir / SO_NAME
    proc = subprocess.run(
        [gcc, "-shared", "-fPIC", "-O2", "-o", str(so_path), str(c_file), "-lm"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gcc failed:\n{proc.stderr.strip() or proc.stdout.strip()}")
    return PyBuiltEnv(glue_dir=gdir, work_dir=work, so_path=so_path, sig=sig)


# ---------------------------------------------------------------------------
# input generation
# ---------------------------------------------------------------------------

_INT_BOUNDARY = [0, 1, -1, 2, -2, 100, -100, 2**31 - 1, -(2**31), 2**62, I64_MAX, I64_MIN]
_FLOAT_BOUNDARY = [
    0.0, -0.0, 1.0, -1.0, 0.5, -2.5, 1e-10, -1e-10, 1e10, -1e10,
    1e308, -1e308, 1e-308, 5e-324, float("inf"), float("-inf"), float("nan"),
    3.141592653589793,
]
_STR_BOUNDARY = [
    "", "a", "Z", "ABC", "abc", "Hello, World!", "12345", "   ",
    "aBcDeFgHiJk", "naive cafe", "A" * 64,
]
_LIST_INT_BOUNDARY = [
    [], [0], [1], [-1], [1, 2, 3], [-5, 0, 5],
    [I64_MAX, I64_MIN, 0], list(range(16)),
]
_LIST_FLOAT_BOUNDARY = [
    [], [0.0], [1.5], [-0.0], [1.0, 2.0, 3.0],
    [float("inf"), float("-inf"), float("nan")],
    [1e308, -1e308, 1e-308], [0.1] * 16,
]

#: hard cap so one kernel's boundary sweep stays fast
_MAX_BOUNDARY_CASES = 48

#: parameter names treated as "size-like": derived from list lengths, in the
#: same spirit as the Java harness's matMul(a,b,c,n) shape (n*n == len).
_SIZE_NAMES = {"n", "size", "len", "length", "dim", "count", "rows", "cols"}

_NEUTRAL = {"int": 0, "float": 0.0, "bool": False, "str": ""}


def _boundary_values(kind: str) -> list:
    return {
        "int": _INT_BOUNDARY,
        "float": _FLOAT_BOUNDARY,
        "bool": [False, True],
        "str": _STR_BOUNDARY,
        "list_int": _LIST_INT_BOUNDARY,
        "list_float": _LIST_FLOAT_BOUNDARY,
    }[kind]


def _is_size_like(pm) -> bool:
    return pm.kind == "int" and pm.name.lower() in _SIZE_NAMES


def _size_from_length(length: int) -> int:
    """Derive a size-like scalar from a list length (matvec/matmul shape).

    Floor ``isqrt`` keeps ``n*n <= length`` so non-square lengths stay
    in-domain (the kernel simply ignores the tail) instead of becoming
    skips. Best-effort: a param that really means "length" will mismatch
    and skip via the Python-first rule.
    """
    return math.isqrt(length)


def gen_boundary_args(sig: KernelSig) -> list[list]:
    """Deterministic, length-coherent boundary sweep.

    Lists always share one length per case (like the Java harness: arrays of
    the same element type share a length), so every case is in the kernel's
    valid domain -- the sweep never manufactures a zip-truncation mismatch
    the C side cannot see. Order: one all-neutral case, then scalar sweeps
    (lists empty), then list sweeps (scalars neutral, every list stretched
    to the swept template's length).
    """
    if not sig.params:
        return [[]]
    scalar_idx = [i for i, p in enumerate(sig.params) if not p.is_list]
    list_idx = [i for i, p in enumerate(sig.params) if p.is_list]
    kinds_present = [p.kind for p in sig.params if p.is_list]

    def neutral_args() -> list:
        args = []
        for p in sig.params:
            args.append([] if p.is_list else _NEUTRAL[p.kind])
        return args

    cases = [neutral_args()]
    # scalar sweeps: lists stay empty (coherent)
    for j in scalar_idx:
        pm = sig.params[j]
        for v in _boundary_values(pm.kind):
            if pm.kind == "float":
                if v == 0.0 and math.copysign(1.0, v) > 0:
                    continue  # +0.0 is the neutral; -0.0 stays a case
            elif v == _NEUTRAL[pm.kind]:
                continue
            args = neutral_args()
            args[j] = v
            cases.append(args)
            if len(cases) >= _MAX_BOUNDARY_CASES:
                return cases
    # list sweeps: every list stretched to the template's length
    for kind in dict.fromkeys(kinds_present):
        for tmpl in _boundary_values(kind)[1:]:  # [0] is [] == neutral
            n = len(tmpl)
            args = neutral_args()
            for j in list_idx:
                q = sig.params[j]
                if q.kind == kind:
                    args[j] = list(tmpl)
                else:
                    elem = 0 if q.kind == "list_int" else 0.0
                    args[j] = [elem] * n
            # size-like scalars follow the length (matvec n)
            for j in scalar_idx:
                if _is_size_like(sig.params[j]):
                    args[j] = _size_from_length(n)
            cases.append(args)
            if len(cases) >= _MAX_BOUNDARY_CASES:
                return cases
    return cases


def _rand_scalar(kind: str, rng: random.Random):
    if kind == "int":
        return rng.randint(-1000, 1000)
    if kind == "float":
        r = rng.random()
        if r < 0.03:
            return rng.choice([float("inf"), float("-inf"), float("nan")])
        if r < 0.08:
            return rng.uniform(-1e6, 1e6)
        return rng.uniform(-100.0, 100.0)
    if kind == "bool":
        return rng.choice([False, True])
    if kind == "str":
        alphabet = "abcXYZ019 "
        return "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
    raise AssertionError(kind)


def gen_random_args(
    sig: KernelSig, rng: random.Random, list_len: tuple[int, int] = (0, 8)
) -> list:
    """One random case: lists share one length per element kind; a size-like
    int parameter (``n``, ``size``, ...) is derived from the list length."""
    len_by_kind: dict[str, int] = {}
    for pm in sig.params:
        if pm.is_list and pm.kind not in len_by_kind:
            len_by_kind[pm.kind] = rng.randint(*list_len)
    first_len = next(iter(len_by_kind.values()), 0)
    args = []
    for pm in sig.params:
        if pm.is_list:
            elem = "int" if pm.kind == "list_int" else "float"
            n = len_by_kind[pm.kind]
            args.append([_rand_scalar(elem, rng) for _ in range(n)])
        elif _is_size_like(pm) and len_by_kind:
            args.append(_size_from_length(first_len))
        else:
            args.append(_rand_scalar(pm.kind, rng))
    return args


def gen_bench_args(sig: KernelSig, list_len: int = 64, seed: int = 1234) -> list:
    """One representative bench input: fixed-size lists, mid-range scalars.

    A size-like parameter derives from the length via floor ``isqrt``, so
    matvec-style ``n*n == len`` shapes stay in-domain for any length.
    """
    rng = random.Random(seed)
    len_by_kind = {p.kind: list_len for p in sig.params if p.is_list}
    first_len = next(iter(len_by_kind.values()), 0)
    args = []
    for pm in sig.params:
        if pm.kind == "list_int":
            args.append([rng.randint(-1000, 1000) for _ in range(len_by_kind[pm.kind])])
        elif pm.kind == "list_float":
            args.append([rng.uniform(-100.0, 100.0) for _ in range(len_by_kind[pm.kind])])
        elif _is_size_like(pm) and len_by_kind:
            args.append(_size_from_length(first_len))
        elif pm.kind == "int":
            args.append(100)
        elif pm.kind == "float":
            args.append(1.5)
        elif pm.kind == "bool":
            args.append(True)
        elif pm.kind == "str":
            args.append("HelloWorld" * 6)
    return args


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


def scalars_equal(kind: str, a, b, tol_float: float, tol_double: float) -> bool:
    """kind is a ParamMap/RetMap kind: int|float|bool|str."""
    if kind in ("int", "bool", "str"):
        return a == b
    if kind == "float":
        # Python float == C double: the double tolerance applies. tol_float
        # is kept for a future float32 mapping.
        tol = tol_double
        if a == b:
            return True
        if isinstance(a, float) and isinstance(b, float):
            if math.isnan(a) and math.isnan(b):
                return True
            if math.isinf(a) or math.isinf(b):
                return False
        try:
            return math.isclose(a, b, rel_tol=tol)
        except (TypeError, ValueError):
            return False
    raise AssertionError(kind)


def _elem_kind(list_kind: str) -> str:
    return {"list_int": "int", "list_float": "float"}[list_kind]


def compare_case(
    sig: KernelSig, py_res: dict, c_res: dict,
    tol_float: float, tol_double: float,
) -> tuple[bool, str]:
    """Compare one case's python vs C results. Returns (ok, detail)."""
    rk = sig.ret.kind
    if rk != "void":
        if not scalars_equal(rk, py_res["ret"], c_res["ret"], tol_float, tol_double):
            return False, (
                f"return mismatch: python {py_res['ret']!r} vs C {c_res['ret']!r}"
            )
    for pm in sig.params:
        if not pm.is_list:
            continue
        pl, cl = py_res["lists"][pm.name], c_res["lists"][pm.name]
        if len(pl) != len(cl):
            return False, (
                f"list '{pm.name}' length changed: python {len(pl)} vs C {len(cl)}"
            )
        ek = _elem_kind(pm.kind)
        for i, (x, y) in enumerate(zip(pl, cl)):
            if not scalars_equal(ek, x, y, tol_float, tol_double):
                return False, (
                    f"list '{pm.name}' element {i}: python {x!r} vs C {y!r}"
                )
    return True, ""


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------


@dataclass
class PyCaseResult:
    index: int
    args: list
    ok: bool
    boundary: bool = False
    skipped: bool = False
    detail: str = ""
    guided: bool = False  # from the coverage-guided second round (v0.11)


@dataclass
class PyVerifyReport:
    method: str  # qualname
    cases: list[PyCaseResult] = field(default_factory=list)
    boundary_cases: list[PyCaseResult] = field(default_factory=list)
    tol_float: float = 1e-6
    tol_double: float = 1e-12
    coverage: "py_coverage.CoverageData | None" = None  # set when coverage=True

    @property
    def passed(self): return sum(1 for c in self.cases if c.ok and not c.skipped)

    @property
    def failed(self): return sum(1 for c in self.cases if not c.ok and not c.skipped)

    @property
    def skipped(self): return sum(1 for c in self.cases if c.skipped)

    @property
    def boundary_passed(self): return sum(1 for c in self.boundary_cases if c.ok and not c.skipped)

    @property
    def boundary_failed(self): return sum(1 for c in self.boundary_cases if not c.ok and not c.skipped)

    @property
    def boundary_skipped(self): return sum(1 for c in self.boundary_cases if c.skipped)

    @property
    def all_passed(self): return self.failed == 0 and self.boundary_failed == 0

    @property
    def guided_passed(self): return sum(1 for c in self.cases if c.guided and c.ok and not c.skipped)

    @property
    def guided_failed(self): return sum(1 for c in self.cases if c.guided and not c.ok and not c.skipped)

    @property
    def guided_skipped(self): return sum(1 for c in self.cases if c.guided and c.skipped)

    def coverage_summary(self) -> dict | None:
        """JSON-friendly coverage rollup, or None when no coverage round ran."""
        cov = self.coverage
        if cov is None:
            return None
        return {
            "branches_total": cov.branches_total,
            "branches_hit": cov.branches_hit,
            "coverage_pct": round(cov.coverage_pct, 1),
        }


@dataclass
class PyBenchReport:
    method: str
    iters: int
    warmup: int
    repeats: int
    python_ns: list[float] = field(default_factory=list)  # per-repeat totals
    c_ns: list[float] = field(default_factory=list)

    @property
    def python_per_call_ns(self): return statistics.median(self.python_ns) / self.iters

    @property
    def c_per_call_ns(self): return statistics.median(self.c_ns) / self.iters

    @property
    def speedup(self):
        c = self.c_per_call_ns
        return (self.python_per_call_ns / c) if c > 0 else float("inf")

    @property
    def verdict(self) -> str:
        """``"faster"`` / ``"slower"`` / ``"inconclusive"`` (v0.10).

        Reuses the shared noise-band rule so the Python backend never
        claims a win inside measurement noise either.
        """
        from .harness import verdict_for_speedup

        return verdict_for_speedup(self.speedup)

    def verdict_text(self, native_label: str = "C (cffi)") -> str:
        """Honest one-line verdict; never claims a win inside the noise band."""
        from .harness import _VERDICT_NOISE_BAND

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
            f"inconclusive: {sp:.2f}x median speedup is inside the {band}x "
            "noise band -- too close to call; not a porting win."
        )


# ---------------------------------------------------------------------------
# driver subprocess (worker)
# ---------------------------------------------------------------------------


def _load_module(repo_root: str, module: str, file: str):
    import importlib.util

    path = Path(repo_root) / file
    spec = importlib.util.spec_from_file_location(module, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, repo_root)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(repo_root)
    return mod


class _Timeout(Exception):
    pass


def _with_timeout(call, args, timeout: float):
    """Run call(*args) with a SIGALRM wall timeout (Linux)."""
    import signal

    if timeout <= 0 or not hasattr(signal, "SIGALRM"):
        return call(*args)

    def _handler(signum, frame):
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return call(*args)
    except _Timeout:
        return ("timeout", "case exceeded %.1fs wall time" % timeout)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _jsonable(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return v  # allow_nan round-trips; both ends are Python
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    raise TypeError(f"unserializable {type(v).__name__}")


def _call_python(fn, params, args):
    call_args, lists = [], {}
    for pm, v in zip(params, args):
        if pm["kind"] in ("list_int", "list_float"):
            cp = list(v)
            call_args.append(cp)
            lists[pm["name"]] = cp
        else:
            call_args.append(v)
    try:
        ret = fn(*call_args)
    except Exception as exc:  # noqa: BLE001 - original raised: out of domain
        return ("raised", f"{type(exc).__name__}: {exc}")
    try:
        return ("ok", _jsonable(ret), {k: _jsonable(lst) for k, lst in lists.items()})
    except TypeError as exc:
        return ("unrep", str(exc))


def _marshal_c(ffi, params, args):
    """Build cffi call args. Returns (c_args, keepalive, lists) or ("unrep", msg)."""
    c_args, keepalive, lists = [], [], {}
    for pm, v in zip(params, args):
        kind = pm["kind"]
        if kind == "int":
            iv = int(v)
            if not I64_MIN <= iv <= I64_MAX:
                return ("unrep", "int input outside int64 range")
            c_args.append(iv)
        elif kind == "float":
            c_args.append(float(v))
        elif kind == "bool":
            c_args.append(1 if v else 0)
        elif kind == "str":
            b = v.encode("utf-8")
            if b"\x00" in b:
                return ("unrep", "NUL byte in str input")
            cdata = ffi.new("char[]", b)
            keepalive.append(cdata)
            c_args.append(cdata)
        elif kind in ("list_int", "list_float"):
            n = len(v)
            ctype = "int64_t" if kind == "list_int" else "double"
            if n == 0:
                c_args += [ffi.NULL, 0]
                lists[pm["name"]] = (None, 0)
            else:
                try:
                    # ffi.new from a python list initializes at C speed;
                    # out-of-int64 elements raise OverflowError here.
                    vals = [int(x) for x in v] if kind == "list_int" else [float(x) for x in v]
                    arr = ffi.new(ctype + "[]", vals)
                except OverflowError:
                    return ("unrep", "list[int] element outside int64 range")
                keepalive.append(arr)
                c_args += [arr, n]
                lists[pm["name"]] = (arr, n)
        else:
            raise AssertionError(kind)
    return (c_args, keepalive, lists)


def _call_c(ffi, cfunc, params, ret_kind, args):
    m = _marshal_c(ffi, params, args)
    if isinstance(m, tuple) and m and m[0] == "unrep":
        return m
    c_args, _keep, lists = m
    try:
        c_ret = cfunc(*c_args)
    except Exception as exc:  # noqa: BLE001 - cffi-level failure
        return ("raised", f"{type(exc).__name__}: {exc}")
    out_lists = {}
    for name, (arr, n) in lists.items():
        out_lists[name] = [arr[i] for i in range(n)] if arr is not None else []
    if ret_kind == "void":
        ret = None
    elif ret_kind == "str":
        if c_ret == ffi.NULL:
            return ("unrep", "C returned NULL for str")
        ret = ffi.string(c_ret).decode("utf-8")
    elif ret_kind in ("int", "bool"):
        ret = int(c_ret)
    elif ret_kind == "float":
        ret = float(c_ret)
    else:
        raise AssertionError(ret_kind)
    try:
        return ("ok", _jsonable(ret), {k: _jsonable(v) for k, v in out_lists.items()})
    except TypeError as exc:
        return ("unrep", str(exc))


def _worker_run(spec: dict, out_path: str) -> None:
    mod = _load_module(spec["repo_root"], spec["module"], spec["file"])
    fn = getattr(mod, spec["func"])
    params, ret_kind = spec["params"], spec["ret"]["kind"]
    timeout = spec.get("timeout", 10.0)

    from cffi import FFI

    ffi = FFI()
    ffi.cdef(spec["cdef"])
    lib = ffi.dlopen(spec["so"])
    cfunc = getattr(lib, spec["cname"])

    results = []
    for i, case in enumerate(spec["cases"]):
        args = case["args"]
        pr = _with_timeout(_call_python, (fn, params, args), timeout)
        if pr[0] != "ok":
            reason = pr[1] if len(pr) > 1 else pr[0]
            results.append({
                "index": i, "boundary": case["boundary"],
                "skipped": True, "skip_reason": f"python side {pr[0]}: {reason}",
            })
            continue
        _, py_ret, py_lists = pr
        cr = _with_timeout(_call_c, (ffi, cfunc, params, ret_kind, args), timeout)
        if cr[0] != "ok":
            if cr[0] == "unrep":
                results.append({
                    "index": i, "boundary": case["boundary"],
                    "skipped": True, "skip_reason": f"c side unrepresentable: {cr[1]}",
                })
            else:
                reason = cr[1] if len(cr) > 1 else cr[0]
                results.append({
                    "index": i, "boundary": case["boundary"],
                    "skipped": False, "ok": False,
                    "detail": f"C side {cr[0]}: {reason}",
                    "py": {"ret": py_ret, "lists": py_lists},
                })
            continue
        _, c_ret, c_lists = cr
        results.append({
            "index": i, "boundary": case["boundary"],
            "skipped": False, "ok": True,  # comparison happens in the parent
            "py": {"ret": py_ret, "lists": py_lists},
            "c": {"ret": c_ret, "lists": c_lists},
        })
    Path(out_path).write_text(json.dumps(results, allow_nan=True), encoding="utf-8")


def _worker_bench(spec: dict, out_path: str) -> None:
    mod = _load_module(spec["repo_root"], spec["module"], spec["file"])
    fn = getattr(mod, spec["func"])
    params, ret_kind = spec["params"], spec["ret"]["kind"]
    args = spec["args"]
    warmup, iters, repeats = spec["warmup"], spec["iters"], spec["repeats"]

    def fresh():
        return [
            list(v) if pm["kind"] in ("list_int", "list_float") else v
            for pm, v in zip(params, args)
        ]

    for _ in range(warmup):
        fn(*fresh())
    python_ns = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        for _ in range(iters):
            fn(*fresh())
        python_ns.append(time.perf_counter_ns() - t0)

    from cffi import FFI

    ffi = FFI()
    ffi.cdef(spec["cdef"])
    lib = ffi.dlopen(spec["so"])
    cfunc = getattr(lib, spec["cname"])

    def c_once(a):
        r = _call_c(ffi, cfunc, params, ret_kind, a)
        if r[0] != "ok":
            raise RuntimeError(f"C bench call failed: {r}")

    for _ in range(warmup):
        c_once(fresh())
    c_ns = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        for _ in range(iters):
            c_once(fresh())
        c_ns.append(time.perf_counter_ns() - t0)

    Path(out_path).write_text(
        json.dumps({"python_ns": python_ns, "c_ns": c_ns}), encoding="utf-8"
    )


def _worker_cov(spec: dict, out_path: str) -> None:
    """Coverage worker: replay cases against the Python original only,
    recording sys.monitoring BRANCH hits for the kernel's source file.

    Never touches cffi -- a broken translation cannot pollute coverage.
    Emits ``{"hits": [[qualname, firstlineno, instr, dst], ...]}``.
    """
    from . import py_coverage as PC

    mod = _load_module(spec["repo_root"], spec["module"], spec["file"])
    fn = getattr(mod, spec["func"])
    params = spec["params"]
    timeout = spec.get("timeout", 10.0)
    # The exact path string the module was loaded from: co_filename
    # matches it verbatim, so the recorder's fast path hits.
    mod_path = os.path.join(spec["repo_root"], spec["file"])
    with PC.branch_recorder(mod_path) as rec:
        for case in spec["cases"]:
            _with_timeout(_call_python, (fn, params, case["args"]), timeout)
    Path(out_path).write_text(json.dumps({"hits": rec.hits}), encoding="utf-8")


def worker_main(argv: list[str]) -> None:
    """Entry point for the driver subprocess: run|bench|cov <spec> <out>."""
    mode, spec_path, out_path = argv[0], argv[1], argv[2]
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    if mode == "run":
        _worker_run(spec, out_path)
    elif mode == "bench":
        _worker_bench(spec, out_path)
    elif mode == "cov":
        _worker_cov(spec, out_path)
    else:
        raise SystemExit(f"unknown worker mode: {mode}")


# ---------------------------------------------------------------------------
# parent-side orchestration
# ---------------------------------------------------------------------------


def _cdef_for(sig: KernelSig) -> str:
    parts = []
    for pm in sig.params:
        if pm.is_list:
            parts += [f"{pm.ctype}* {pm.cname}", f"int64_t {pm.cname}_n"]
        elif pm.kind == "str":
            parts.append(f"const char* {pm.cname}")
        else:
            parts.append(f"{pm.ctype} {pm.cname}")
    params = ", ".join(parts) if parts else "void"
    return f"{sig.ret.ctype} {sig.cname}({params});"


def _run_driver(
    env: PyBuiltEnv, sig: KernelSig, repo_root: Path, mode: str,
    payload: dict, overall_timeout: float,
) -> dict | list:
    spec = {
        "mode": mode,
        "repo_root": str(repo_root),
        "module": sig.qualname.rsplit(".", 1)[0],
        "file": sig.file,
        "func": sig.func_name,
        "cname": sig.cname,
        "params": [
            {"name": p.name, "kind": p.kind} for p in sig.params
        ],
        "ret": {"kind": sig.ret.kind},
        "cdef": _cdef_for(sig),
        "so": str(env.so_path),
        **payload,
    }
    spec_path = env.work_dir / f"spec_{mode}.json"
    out_path = env.work_dir / f"out_{mode}.json"
    spec_path.write_text(json.dumps(spec, allow_nan=True), encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "kernel_port.py_harness", mode,
             str(spec_path), str(out_path)],
            capture_output=True, text=True, timeout=overall_timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "driver subprocess timed out after "
            f"{overall_timeout:.0f}s -- the native translation probably hangs "
            "(infinite loop). This is a FAIL for the translation."
        )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        tail = "\n".join(tail[-15:])
        if proc.returncode > 0 and _driver_failed_on_import(proc.stderr):
            # ImportError / ModuleNotFoundError: the driver died on startup,
            # before it could even load the translation -- an environment
            # problem, not a crash of the C code. Say so explicitly.
            raise RuntimeError(
                "driver subprocess failed to start "
                f"(exit {proc.returncode}; this is a driver import error, "
                f"not a crash of the C translation):\n{tail}"
            )
        raise RuntimeError(
            "driver subprocess crashed "
            f"(exit {proc.returncode}; a segfault in the C translation looks "
            f"exactly like this):\n{tail}"
        )
    return json.loads(out_path.read_text(encoding="utf-8"))


#: Matches a driver-side Python import failure (ModuleNotFoundError /
#: ImportError) in the driver's stderr. Reporting that as a "segfault in the
#: C translation" was misleading -- it is an environment/driver problem.
_IMPORT_ERROR_RE = re.compile(r"^(ModuleNotFoundError|ImportError)\s*:", re.MULTILINE)


def _driver_failed_on_import(stderr: str | None) -> bool:
    """True when the driver died before running -- on a Python import.

    An import error means the *driver* never got to load the translation,
    so this is a driver/environment failure, not a crash of the C code.
    A negative returncode (killed by a signal, e.g. -11/SIGSEGV) is always
    treated as a real crash.
    """
    return bool(stderr and _IMPORT_ERROR_RE.search(stderr))


def _run_and_append(
    env: PyBuiltEnv,
    sig: KernelSig,
    repo_root: Path,
    cases: list[dict],
    report: PyVerifyReport,
    timeout: float,
    overall_timeout: float,
    tol_float: float,
    tol_double: float,
    guided: bool = False,
) -> None:
    """Run ``cases`` through the differential driver, compare, append."""
    raw = _run_driver(
        env, sig, repo_root, "run",
        {"cases": cases, "timeout": timeout},
        overall_timeout,
    )
    for entry, case in zip(raw, cases):
        idx = entry["index"]
        if entry.get("skipped"):
            res = PyCaseResult(
                index=idx, args=case["args"], ok=False,
                boundary=case["boundary"], skipped=True,
                detail=entry.get("skip_reason", "skipped"),
                guided=guided,
            )
        elif not entry.get("ok", True):
            res = PyCaseResult(
                index=idx, args=case["args"], ok=False,
                boundary=case["boundary"], detail=entry.get("detail", "C side failed"),
                guided=guided,
            )
        else:
            ok, detail = compare_case(
                sig, entry["py"], entry["c"], tol_float, tol_double
            )
            res = PyCaseResult(
                index=idx, args=case["args"], ok=ok,
                boundary=case["boundary"], detail=detail,
                guided=guided,
            )
        (report.boundary_cases if case["boundary"] else report.cases).append(res)


def _coverage_guided_round(
    env: PyBuiltEnv,
    sig: KernelSig,
    repo_root: Path,
    round1_cases: list[dict],
    report: PyVerifyReport,
    seed: int,
    guided_n: int,
    timeout: float,
    tol_float: float,
    tol_double: float,
) -> None:
    """Coverage-guided second round: replay round-1 under sys.monitoring,
    steer fresh inputs at uncovered edges, differential-test them.

    Best-effort: any failure here (old Python, unparseable source,
    driver crash) raises, and the caller degrades to round-1 results
    with a warning.
    """
    from . import py_coverage as PC

    manifest = PC.static_manifest(Path(repo_root) / sig.file)
    cov_timeout = max(1.0, min(timeout, 10.0))
    raw = _run_driver(
        env, sig, repo_root, "cov",
        {"cases": round1_cases, "timeout": cov_timeout},
        max(180.0, 1.5 * len(round1_cases) * cov_timeout),
    )
    hits = raw.get("hits", []) if isinstance(raw, dict) else []
    report.coverage = PC.merge_coverage(manifest, [tuple(h) for h in hits])
    rng = random.Random(seed + 1)
    gargs = PC.guided_cases(sig, report.coverage, rng, guided_n)
    gcases = [{"args": a, "boundary": False} for a in gargs]
    _run_and_append(
        env, sig, repo_root, gcases, report, timeout,
        max(180.0, 2.5 * len(gcases) * timeout),
        tol_float, tol_double, guided=True,
    )


def run_verify(
    env: PyBuiltEnv,
    sig: KernelSig,
    repo_root: Path | str,
    cases: int = 100,
    seed: int = 42,
    boundary: bool = True,
    timeout: float = 10.0,
    tol_float: float = 1e-6,
    tol_double: float = 1e-12,
    overall_timeout: float | None = None,
    coverage: bool = False,
    guided_n: int = 50,
) -> PyVerifyReport:
    """Differential-test the C translation against the Python original.

    ``coverage=True`` runs an opt-in coverage-guided second round after
    the plain random+boundary round: the round-1 cases are replayed
    against the Python original under ``sys.monitoring`` branch
    recording (:mod:`kernel_port.py_coverage`), and up to ``guided_n``
    fresh inputs biased at uncovered edges are differential-tested and
    appended to the report with ``guided=True``. The coverage round is
    best-effort -- any failure degrades to the round-1 results with a
    warning, never a crash.
    """
    repo_root = Path(repo_root)
    rng = random.Random(seed)
    rand_cases = [
        {"args": gen_random_args(sig, rng), "boundary": False} for _ in range(cases)
    ]
    bnd_cases = (
        [{"args": a, "boundary": True} for a in gen_boundary_args(sig)]
        if boundary else []
    )
    all_cases = rand_cases + bnd_cases
    if overall_timeout is None:
        overall_timeout = max(180.0, 2.5 * len(all_cases) * timeout)
    report = PyVerifyReport(
        method=sig.qualname, tol_float=tol_float, tol_double=tol_double
    )
    _run_and_append(
        env, sig, repo_root, all_cases, report, timeout, overall_timeout,
        tol_float, tol_double,
    )
    if coverage and guided_n > 0:
        try:
            _coverage_guided_round(
                env, sig, repo_root, all_cases, report, seed, guided_n,
                timeout, tol_float, tol_double,
            )
        except Exception as exc:  # noqa: BLE001 - best-effort by design
            warnings.warn(
                f"coverage-guided round skipped ({exc}); round-1 results stand"
            )
    return report


def run_bench(
    env: PyBuiltEnv,
    sig: KernelSig,
    repo_root: Path | str,
    warmup: int = 20,
    iters: int = 200,
    repeats: int = 5,
    seed: int = 1234,
    list_len: int = 64,
    timeout: float = 300.0,
) -> PyBenchReport:
    """Wall-clock bench: Python original vs C translation (median per call)."""
    repo_root = Path(repo_root)
    args = gen_bench_args(sig, list_len=list_len, seed=seed)
    raw = _run_driver(
        env, sig, repo_root, "bench",
        {"args": args, "warmup": warmup, "iters": iters, "repeats": repeats},
        timeout,
    )
    if isinstance(raw, dict) and raw.get("error"):
        raise RuntimeError(f"bench driver failed: {raw['error']}")
    return PyBenchReport(
        method=sig.qualname, iters=iters, warmup=warmup, repeats=repeats,
        python_ns=raw["python_ns"], c_ns=raw["c_ns"],
    )


if __name__ == "__main__":
    worker_main(sys.argv[1:])
