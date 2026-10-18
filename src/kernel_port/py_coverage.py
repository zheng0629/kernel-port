"""Coverage-guided fuzzing support for the Python->C ``verify`` path.

Mirrors the spirit of :mod:`kernel_port.coverage` (the Java side) at a
Python-appropriate level:

1. :func:`static_manifest` enumerates every conditional-branch *edge* of
   the kernel's source file without importing it -- ``ast`` for the
   decision structure plus ``dis`` for the exact branch instructions.
   Each conditional jump contributes two edges: the fall-through edge
   and the jump-taken edge.
2. :func:`branch_recorder` is a ``sys.monitoring`` (Python 3.12+)
   context manager that records ``BRANCH`` events for one source file
   while the round-1 cases replay against the *Python original*.
   A hit is classified as taken vs fall-through by comparing the
   reported destination with the static fall-through offset, so the
   ``FOR_ITER``/``END_FOR`` destination quirk (the interpreter reports
   two bytes past ``dis``'s ``argval``) cannot desync the merge.
3. :func:`merge_coverage` joins the static manifest with the recorded
   hits into :class:`CoverageData`.
4. :func:`guided_cases` generates a second input round biased at
   uncovered edges: numeric literals harvested from the edge's
   condition (``x == 1234567`` -> ``1234567``) are injected into
   matching parameters, and uncovered loop bodies get non-degenerate
   (non-empty-list) inputs.
5. :func:`minimize` delta-minimizes a failing input case (ddmin on
   lists, shrink-toward-zero on scalars, prefix binary search on
   strings), reusing the shrink primitives from :mod:`kernel_port.coverage`.

Branch model: one edge per conditional-jump destination --
``cond-true``/``cond-false`` for comparisons, ``loop-body``/``loop-exit``
for ``FOR_ITER``. Loops contribute the same "body executed" signal the
Java side uses.

Deliberate simplifications:

- ``try``/``except`` handler entry fires no ``BRANCH`` event on CPython,
  so exception edges are invisible (the Java side counts ``catch``
  blocks; here they are silently skipped).
- ``SEND`` (generators/coroutines) is not treated as a branch; kernels
  are plain functions.
- ``if __name__ == "__main__":`` guards are unsteerable by inputs --
  their condition still yields literals, but targeting them is skipped.
- Coverage is measured on the Python original only (like the Java side,
  which instruments the Java original): the guided cases then run
  through the normal differential driver, so both sides see them.
- On Python < 3.12 (no ``sys.monitoring``) :func:`branch_recorder`
  raises ``RuntimeError``; :func:`run_verify` (``coverage=True``)
  degrades to round-1 results with a warning, never a crash.

Public API
----------
- ``static_manifest(source_file) -> list[BranchInfo]``
- ``branch_recorder(target_file)`` -- context manager yielding ``.hits``
- ``merge_coverage(manifest, hits) -> CoverageData``
- ``guided_cases(sig, coverage, rng, n) -> list[list]``
- ``minimize(case, fails, max_calls=20000, max_seconds=120.0) -> list``
"""

from __future__ import annotations

import ast
import contextlib
import dis
import math
import os
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import random

    from .cffi_glue import KernelSig

__all__ = [
    "BranchInfo",
    "CoverageData",
    "static_manifest",
    "branch_recorder",
    "merge_coverage",
    "guided_cases",
    "minimize",
]


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------


@dataclass
class BranchInfo:
    """One statically-enumerated control-flow edge."""

    id: int = 0
    func: str = ""  # code object qualname, e.g. "rare_branch" / "f.<listcomp>"
    line: int = 0  # 1-based source line of the branch instruction
    kind: str = ""  # "cond-true" | "cond-false" | "loop-body" | "loop-exit"
    condition: str = ""  # source text of the enclosing test, "" if unknown
    names: tuple = ()  # names referenced by the condition
    literals: tuple = ()  # int/float literals in the condition (no bools)
    instr: int = 0  # bytecode offset of the branch instruction
    fall: int = 0  # bytecode offset of the fall-through instruction
    taken: bool = False  # True = the jump-taken edge, False = fall-through


@dataclass
class CoverageData:
    """Branch coverage: static manifest joined with recorded hits."""

    branches_total: int
    branches_hit: int
    per_branch_hits: dict[int, int] = field(default_factory=dict)
    branches: list[BranchInfo] = field(default_factory=list)

    @property
    def coverage_pct(self) -> float:
        """Percentage of edges hit; 100.0 when there are no branches."""
        if self.branches_total == 0:
            return 100.0
        return 100.0 * self.branches_hit / self.branches_total

    def uncovered(self) -> list[BranchInfo]:
        """Edges with a zero hit count, in manifest order."""
        return [
            b
            for b in self.branches
            if self.per_branch_hits.get(b.id, 0) == 0
        ]

    def hits_for(self, branch_id: int) -> int:
        return self.per_branch_hits.get(branch_id, 0)


# ---------------------------------------------------------------------------
# static manifest: ast + dis, no import needed
# ---------------------------------------------------------------------------


def _is_cond_branch(ins: dis.Instruction) -> bool:
    """True for conditional jumps (two destinations) and FOR_ITER.

    Unconditional jumps (JUMP_BACKWARD/JUMP_FORWARD) are excluded, as is
    SEND (generators). A conditional jump always carries its target as
    the oparg; the fall-through is the next instruction.
    """
    if ins.opname == "FOR_ITER":
        return True
    return "IF" in ins.opname and ins.argval is not None


def _kinds_for(opname: str) -> tuple[str, str]:
    """(fall-through kind, jump-taken kind) for a branch opcode."""
    if opname == "FOR_ITER":
        return ("loop-body", "loop-exit")
    if "IF_TRUE" in opname or "NOT_NONE" in opname or "TRUE_OR_POP" in opname:
        # jumps when the condition is truthy: taken = true path
        return ("cond-false", "cond-true")
    # POP_JUMP_IF_FALSE / IF_NONE / FALSE_OR_POP: taken = false path
    return ("cond-true", "cond-false")


_COND_NODES = (ast.If, ast.While, ast.IfExp, ast.For, ast.AsyncFor)


@dataclass
class _Cond:
    text: str
    names: tuple
    literals: tuple


def _condition_at(tree: ast.AST, source: str, line: int) -> _Cond:
    """Innermost If/While/For test enclosing ``line``.

    Returns the condition's source text plus the names and numeric
    literals it references, which is what the guided round aims at
    parameters. ``__name__`` guards keep their text (for reporting) but
    yield no literals, so the guider skips them.
    """
    best = None
    for node in ast.walk(tree):
        if not isinstance(node, _COND_NODES):
            continue
        if node.lineno <= line <= node.end_lineno:
            if best is None or node.lineno > best.lineno:
                best = node
    if best is None:
        return _Cond("", (), ())
    expr = best.iter if isinstance(best, (ast.For, ast.AsyncFor)) else best.test
    text = ast.get_source_segment(source, expr) or ""
    if "__name__" in text:
        return _Cond(text, (), ())
    names = tuple(
        sorted({n.id for n in ast.walk(expr) if isinstance(n, ast.Name)})
    )
    literals = tuple(
        dict.fromkeys(
            c.value
            for c in ast.walk(expr)
            if isinstance(c, ast.Constant)
            and isinstance(c.value, (int, float))
            and not isinstance(c.value, bool)
        )
    )
    return _Cond(text, names, literals)


def _line_for(linestarts: list[tuple[int, int, int | None]], offset: int) -> int:
    for start, end, ln in linestarts:
        if ln is None:
            continue
        if start <= offset < end:
            return ln
        if start > offset:
            break
    return 0


def _manifest_code(
    code: types.CodeType,
    tree: ast.AST,
    source: str,
    out: list[BranchInfo],
) -> None:
    linestarts = [
        (s, e, ln) for s, e, ln in code.co_lines() if ln is not None
    ]
    insns = list(dis.get_instructions(code))
    for idx, ins in enumerate(insns):
        if not _is_cond_branch(ins):
            continue
        if idx + 1 >= len(insns):
            continue  # degenerate; a conditional jump always falls through
        fall = insns[idx + 1].offset
        line = _line_for(linestarts, ins.offset) or code.co_firstlineno
        cond = _condition_at(tree, source, line)
        fall_kind, taken_kind = _kinds_for(ins.opname)
        out.append(
            BranchInfo(
                id=len(out),
                func=code.co_qualname,
                line=line,
                kind=fall_kind,
                condition=cond.text,
                names=cond.names,
                literals=cond.literals,
                instr=ins.offset,
                fall=fall,
                taken=False,
            )
        )
        out.append(
            BranchInfo(
                id=len(out),
                func=code.co_qualname,
                line=line,
                kind=taken_kind,
                condition=cond.text,
                names=cond.names,
                literals=cond.literals,
                instr=ins.offset,
                fall=fall,
                taken=True,
            )
        )


def static_manifest(source_file: str | Path) -> list[BranchInfo]:
    """Enumerate every conditional-branch edge of a Python source file.

    Pure static analysis -- the file is parsed and compiled but never
    imported or executed. Raises ``SyntaxError``/``OSError`` for
    unreadable files; callers treat that as "no coverage data".
    """
    path = Path(source_file)
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    code = compile(text, str(path), "exec")
    out: list[BranchInfo] = []

    def walk(c: types.CodeType) -> None:
        _manifest_code(c, tree, text, out)
        for const in c.co_consts:
            if isinstance(const, types.CodeType):
                walk(const)

    walk(code)
    return out


# ---------------------------------------------------------------------------
# runtime hit recording via sys.monitoring (3.12+)
# ---------------------------------------------------------------------------


class _Hits:
    def __init__(self) -> None:
        # each hit: (qualname, firstlineno, instr_offset, dst_offset)
        self.hits: list[tuple] = []


def _claim_tool_id(mon) -> int:
    for tid in range(6):
        try:
            mon.use_tool_id(tid, "kernel-port")
            return tid
        except ValueError:
            continue
    raise RuntimeError("sys.monitoring: no free tool id")


@contextlib.contextmanager
def branch_recorder(target_file: str | Path):
    """Record BRANCH events for one source file.

    Yields an object with ``.hits`` -- a list of
    ``(qualname, firstlineno, instr_offset, dst_offset)`` tuples in
    firing order. Code objects from other files are DISABLEd on first
    sight, so the callback stays cheap.

    Requires Python 3.12+ (``sys.monitoring``); raises ``RuntimeError``
    otherwise so callers can degrade gracefully.
    """
    mon = getattr(sys, "monitoring", None)
    if mon is None:  # pragma: no cover - the test env is 3.12+
        raise RuntimeError(
            "branch_recorder() needs sys.monitoring (Python 3.12+)"
        )
    tool = _claim_tool_id(mon)
    raw_target = os.fspath(target_file)
    real_target = os.path.realpath(raw_target)
    rec = _Hits()

    def _cb(code, instr_offset, dst_offset):
        try:
            fn = code.co_filename
            if fn != raw_target and fn != real_target:
                try:
                    if os.path.realpath(fn) != real_target:
                        return mon.DISABLE
                except OSError:
                    return mon.DISABLE
        except Exception:  # never let the tracer break the program
            return mon.DISABLE
        rec.hits.append(
            (code.co_qualname, code.co_firstlineno, instr_offset, dst_offset)
        )

    mon.register_callback(tool, mon.events.BRANCH, _cb)
    mon.set_events(tool, mon.events.BRANCH)
    try:
        yield rec
    finally:
        try:
            mon.set_events(tool, 0)
        finally:
            mon.register_callback(tool, mon.events.BRANCH, None)
            mon.free_tool_id(tool)


def merge_coverage(
    manifest: list[BranchInfo], hits: list[tuple]
) -> CoverageData:
    """Join a static manifest with recorded hits into :class:`CoverageData`.

    A hit is classified by instruction offset: destination == the
    static fall-through offset means the fall-through edge, anything
    else the jump-taken edge. Hits for unknown instructions (e.g. a
    stale manifest) are ignored -- best-effort.
    """
    by_instr: dict[tuple[str, int], dict[bool, BranchInfo]] = {}
    for b in manifest:
        by_instr.setdefault((b.func, b.instr), {})[b.taken] = b
    per: dict[int, int] = {}
    for h in hits:
        qualname, _firstlineno, ioff, dst = h[0], h[1], h[2], h[3]
        pair = by_instr.get((qualname, ioff))
        if not pair:
            continue
        edge = pair[True] if dst != pair[False].fall else pair[False]
        per[edge.id] = per.get(edge.id, 0) + 1
    total = len(manifest)
    hit = sum(1 for b in manifest if per.get(b.id, 0) > 0)
    return CoverageData(
        branches_total=total,
        branches_hit=hit,
        per_branch_hits=per,
        branches=list(manifest),
    )


# ---------------------------------------------------------------------------
# guided second-round generation
# ---------------------------------------------------------------------------


def _targeted_cases(
    sig: "KernelSig", branch: BranchInfo, rng: "random.Random", cap: int = 8
) -> list[list]:
    """Cases aiming at one uncovered edge.

    Numeric literals harvested from the edge's condition are injected
    (plus +/-1) into parameters named in the condition, falling back to
    every scalar parameter of a matching kind; then the kind's boundary
    edges are tried on the same parameters. Mirrors
    :func:`kernel_port.coverage._targeted_cases`.
    """
    from . import py_harness as PH

    lits = [v for v in branch.literals][:3]
    if not lits:
        return []
    names = set(branch.names)
    out: list[list] = []
    for lit in lits:
        is_int = isinstance(lit, int)
        want = "int" if is_int else "float"
        named = [
            i
            for i, p in enumerate(sig.params)
            if not p.is_list and p.name in names and p.kind == want
        ]
        idxs = named or [
            i for i, p in enumerate(sig.params)
            if not p.is_list and p.kind == want
        ]
        deltas = (0, 1, -1) if is_int else (0.0, 1.0, -1.0)
        for i in idxs[:3]:
            for d in deltas:
                v = lit + d
                if is_int:
                    v = max(PH.I64_MIN, min(PH.I64_MAX, int(v)))
                else:
                    v = float(v)
                args = PH.gen_random_args(sig, rng)
                args[i] = v
                out.append(args)
                if len(out) >= cap:
                    return out
        for i in idxs[:2]:
            try:
                edges = PH._boundary_values(want)
            except KeyError:  # pragma: no cover - defensive
                continue
            for ev in edges:
                if isinstance(ev, float) and math.isnan(ev):
                    continue
                args = PH.gen_random_args(sig, rng)
                args[i] = ev
                out.append(args)
                if len(out) >= cap:
                    return out
    return out


def _nondegenerate_case(sig: "KernelSig", rng: "random.Random") -> list:
    """A random case with no empty lists, so loop bodies can execute."""
    from . import py_harness as PH

    for _ in range(25):
        args = PH.gen_random_args(sig, rng, list_len=(1, 8))
        if all(
            (not p.is_list) or len(a) > 0
            for p, a in zip(sig.params, args)
        ):
            return args
    return PH.gen_random_args(sig, rng, list_len=(1, 8))


def guided_cases(
    sig: "KernelSig", coverage: CoverageData, rng: "random.Random", n: int
) -> list[list]:
    """Generate a second input round biased at uncovered edges.

    For each uncovered edge, literal/boundary-targeted cases are emitted
    first (see :func:`_targeted_cases`); uncovered loop bodies
    additionally get non-degenerate inputs; remaining slots are filled
    with plain random cases. Returns at most ``n`` cases (deduplicated);
    the fill loop is stall-guarded so degenerate signatures (e.g. no
    parameters at all) terminate instead of spinning on duplicates.
    """
    from . import py_harness as PH

    out: list[list] = []
    seen: set[str] = set()

    def push(args: list) -> None:
        if len(out) >= n:
            return
        key = repr(args)
        if key not in seen:
            seen.add(key)
            out.append(args)

    uncovered = coverage.uncovered()
    # The kernel's own edges (and nested scopes) steer first: on real
    # files the manifest also covers module-level helpers and sibling
    # functions, whose uncovered edges must not starve the kernel's.
    def _prio(b: BranchInfo) -> tuple[int, int]:
        own = b.func == sig.func_name or b.func.startswith(sig.func_name + ".")
        return (0 if own else 1, b.id)

    uncovered = sorted(uncovered, key=_prio)
    for b in uncovered:
        if len(out) >= n:
            break
        if "__name__" in (b.condition or ""):
            continue
        for args in _targeted_cases(sig, b, rng):
            push(args)
    for b in uncovered:
        if len(out) >= n:
            break
        if b.kind == "loop-body":
            push(_nondegenerate_case(sig, rng))
    stall = 0
    while len(out) < n and stall < 100:
        before = len(out)
        push(PH.gen_random_args(sig, rng))
        stall = stall + 1 if len(out) == before else 0
    return out


# ---------------------------------------------------------------------------
# failure minimization (ddmin-style)
# ---------------------------------------------------------------------------


def minimize(
    case: list,
    fails,
    max_calls: int = 20000,
    max_seconds: float = 120.0,
) -> list:
    """Delta-minimize a failing Python-verify input case.

    ``case`` is an argument list as produced by
    :func:`kernel_port.py_harness.gen_random_args`; ``fails(case)``
    must be a deterministic predicate reporting whether the case still
    reproduces the failure (typically: the Python and C sides disagree
    on it). Each argument is shrunk independently -- lists via ddmin,
    scalars toward zero, strings via prefix binary search -- keeping
    only variants that still fail.

    The ddmin/shrink engine is shared with
    :func:`kernel_port.coverage.minimize` (it operates on plain Python
    values); only the docstring is Python-flavoured. Raises
    ``ValueError`` when the input case does not fail, and stops early
    (returning the best-so-far) when either the ``max_calls``
    predicate-call budget or the ``max_seconds`` wall-clock budget is
    exhausted.

    ``max_seconds`` must be positive.
    """
    from .coverage import _BudgetExceeded, _shrink_value

    if max_seconds <= 0:
        raise ValueError("minimize() requires max_seconds > 0")
    calls = {"n": 0}
    deadline = time.monotonic() + max_seconds

    def still_fails(new_case: list) -> bool:
        calls["n"] += 1
        if calls["n"] > max_calls:
            raise _BudgetExceeded
        if time.monotonic() >= deadline:
            raise _BudgetExceeded
        return bool(fails(new_case))

    if not still_fails(list(case)):
        raise ValueError("minimize() requires a case that currently fails")

    cur = list(case)
    try:
        for i in range(len(cur)):
            idx = i

            def try_arg(cand, _i=idx):
                new = list(cur)
                new[_i] = cand
                return still_fails(new)

            cur[i] = _shrink_value(cur[i], try_arg)
    except _BudgetExceeded:
        pass
    return cur
