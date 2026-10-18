"""Coverage-guided fuzzing support for ``verify`` (v0.6 workstream D).

Today's ``verify`` throws random + boundary inputs at the translation and
hopes for branch coverage. This module closes the loop:

1. :func:`instrument` copies the kernel + closure Java sources into a work
   dir and injects an integer hit-counter at every branch point
   (``if``/``else``, ``for``/``while``/``do`` bodies, ``switch`` cases,
   ternary arms, ``catch`` blocks). Counters live in a ``public static
   int[]`` injected into each top-level class that has branches.
2. :func:`run_covered` compiles the instrumented copy, replays cases
   through the standard harness line protocol (plus a ``COV`` command that
   dumps the counters), and returns :class:`CoverageData`.
3. :func:`guided_cases` generates a second input round biased at uncovered
   branches: numeric literals harvested from the branch condition (plus
   boundary edges for the parameter type) are injected into matching
   parameters, and uncovered loop bodies get non-degenerate
   (non-empty-array) inputs.
4. :func:`minimize` delta-minimizes a failing input -- ddmin on arrays,
   shrink-toward-zero on scalars, prefix binary search on strings -- while
   a caller-supplied ``fails`` predicate still reports failure, so
   counterexample feedback stays small.

Branch model: one branch per control-flow *edge* out of a decision point
(then/else arms, each ``switch`` case including a synthesized ``default``,
each ``catch``, each ternary arm). Loops contribute a single "body
executed" edge: for fuzzing, the interesting signal is whether the body
ran at all, and a "loop exited" edge would need a post-loop insertion
that labeled ``break``s can skip.

Semantics preservation: statement branches are wrapped in blocks
(``if (c) s;`` -> ``if (c) { HIT[i]++; s; }``) or get the counter as the
first statement of an existing block; the ``if``/loop conditions themselves
are never touched. Ternary arms go through a type-overloaded ``__kpRec``
recorder -- ``c ? a : b`` becomes
``c ? K.__kpRec(K.__kp_hit,t,(a)) : K.__kpRec(K.__kp_hit,f,(b))`` -- which
evaluates each arm exactly once, preserving laziness, operand types
(overload resolution picks the matching primitive overload, objects go
through the generic one) and short-circuit behavior.

Deliberate simplifications:

- Ternary instrumentation is skipped in constant-expression contexts
  (``static final`` field initializers, annotations, ``case`` labels),
  where a method call would destroy compile-time-constant status and
  break compilation.
- A local variable named exactly like the top-level class would shadow the
  qualified counter reference (``K.__kp_hit``); kernels don't do this and
  it is not handled.
- ``guided_cases`` is heuristic, not concolic: it aims literals and
  boundary edges at parameters named in (or type-compatible with) the
  uncovered branch's condition.

Evaluation: should ``--coverage`` be the default? (v0.7 workstream C)
---------------------------------------------------------------
Measured on ``tests/fixtures/sample_kernels`` (JDK 25, this VM):

- ``com.example.Kernels.dot`` (cheap):   plain verify 0.41s,
  coverage round +4.56s  ->  **x12.2** overhead (cases=100)
- ``com.example.Kernels.fib``:           plain verify 30.2s,
  coverage round +64.8s  ->  **x3.1**  (cases=100; both dominated by
  fib(INT_MAX)-style per-case timeouts, coverage still >2x)
- ``com.example.Kernels.sieve`` (branchy): plain verify 0.39s,
  coverage round +10.1s  ->  **x26.7** (cases=30)
- ``class_kernels`` ``Util.clamp`` (branchless): the coverage round's
  fixed cost alone is ~3.3s for 40 cases -- instrumentation of 4 files is
  0.01s; javac + JVM startup dominate.

The coverage *gain* is real: sieve went 60% -> 100% branch coverage after
the guided round, and a synthetic rare-branch kernel
(``x == 1234567``, a tight switch, a high loop-trip threshold) went
70% -> 90%, with the guided round finding the magic-constant branch that
100 random cases never hit.

Decision: ``--coverage`` stays **opt-in** (default off). The <2x-verify-time
precondition for default-on fails on every measured kernel (best case
x3.1); the fixed javac + JVM-startup cost alone (~2-4s) exceeds a typical
cheap verify. Re-evaluate if the round's fixed cost ever drops near 1x
verify -- ``tests/test_coverage_eval.py::test_coverage_round_cost_exceeds_2x``
pins the current decision so a future speedup forces a re-decision.

Hardening notes:

- The CLI wraps the whole coverage round so instrumentation/compile
  failures degrade gracefully to plain-verify results (a warning, never a
  crash).
- :func:`minimize` has both a call budget (``max_calls``) and a wall-clock
  budget (``max_seconds``); either stops it early with best-so-far.
- :class:`CoveredSession` lets callers compile once and replay several
  case batches through one JVM session instead of paying javac + JVM
  startup per :func:`run_covered` call.

Public API
----------
- ``instrument(sources, work_dir=None) -> InstrumentedCopy``
- ``run_covered(method, cases, instrumented, per_case_timeout=30.0,
  java_bin=None) -> CoverageData``
- ``CoveredSession`` -- compile once, ``replay()`` several case batches,
  ``coverage()`` the cumulative counters (``run_covered`` is one batch
  through it)
- ``guided_cases(method, coverage, rng, n) -> list[list]``
- ``minimize(case, fails, max_calls=20000, max_seconds=120.0) -> list``

Only new code lives here; ``harness.py`` is reused read-only (its
``render_harness`` output is text-patched to add the ``COV`` command --
see :func:`_render_cov_harness`).
"""

from __future__ import annotations

import re
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import harness as H
from . import java_ast as J

__all__ = [
    "BranchInfo",
    "CoverageData",
    "CoveredSession",
    "InstrumentedCopy",
    "instrument",
    "run_covered",
    "guided_cases",
    "minimize",
]


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------


@dataclass
class BranchInfo:
    """One instrumented control-flow edge."""

    global_id: int  # 0-based across the whole instrumented copy
    class_qualname: str  # e.g. "com.example.Kernels"
    kind: str  # "if-then" | "if-else" | "for-body" | "enhanced-for-body" |
    # "while-body" | "do-body" | "ternary-true" | "ternary-false" |
    # "switch-case" | "switch-default" | "catch"
    condition: str  # source text of the condition / case label / catch type
    line: int  # 1-based line in the original source
    method: str  # enclosing method name ("" when outside a method)
    local_id: int = 0  # index into the class's hit array


@dataclass
class InstrumentedCopy:
    """Result of :func:`instrument`."""

    work_dir: Path  # directory holding the instrumented sources
    files: list[str]  # relative paths written, in sorted order
    branches: list[BranchInfo]  # in global_id order
    # (class qualname, hit-field name) in dump order; only classes with >=1
    # branch appear here.
    coverage_classes: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class CoverageData:
    """Aggregate branch coverage from one :func:`run_covered` call."""

    branches_total: int
    branches_hit: int
    per_branch_hits: dict[int, int]  # global branch id -> hit count
    branches: list[BranchInfo] = field(default_factory=list)

    @property
    def coverage_pct(self) -> float:
        """Percentage of branches hit; 100.0 when there are no branches."""
        if self.branches_total == 0:
            return 100.0
        return 100.0 * self.branches_hit / self.branches_total

    def uncovered(self) -> list[BranchInfo]:
        """Branches with a zero hit count, in global_id order."""
        return [
            b
            for b in self.branches
            if self.per_branch_hits.get(b.global_id, 0) == 0
        ]

    def hits_for(self, branch_id: int) -> int:
        return self.per_branch_hits.get(branch_id, 0)


# ---------------------------------------------------------------------------
# instrumentation
# ---------------------------------------------------------------------------

# tree-sitter node type -> branch kind for loop bodies
_LOOP_NODE_KINDS = {
    "for_statement": "for-body",
    "enhanced_for_statement": "enhanced-for-body",
    "while_statement": "while-body",
    "do_statement": "do-body",
}

_TYPE_DECLS = {
    "class_declaration",
    "interface_declaration",
    "enum_declaration",
    "record_declaration",
}

# Parents of switch_expression that make it a *statement* switch (as opposed
# to a switch *expression* producing a value). Matters for arrow-rule bodies
# (expression bodies need `yield` only in expression switches) and for the
# synthesized default (a bare `break;` is illegal in an expression switch).
_SWITCH_STMT_PARENTS = frozenset(
    {
        "block",
        "if_statement",
        "for_statement",
        "enhanced_for_statement",
        "while_statement",
        "do_statement",
        "try_statement",
        "catch_clause",
        "finally_clause",
        "labeled_statement",
        "synchronized_statement",
    }
)

_REC_PRIMITIVES = ("int", "long", "float", "double", "boolean", "char")


def _fresh_names(src_text: str) -> tuple[str, str]:
    """Counter-field / recorder names not colliding with the source text."""
    hit, rec, k = "__kp_hit", "__kpRec", 2
    while hit in src_text or rec in src_text:
        hit, rec = f"__kp_hit{k}", f"__kpRec{k}"
        k += 1
    return hit, rec


@dataclass
class _Ctx:
    top_simple: str | None = None
    top_qual: str | None = None
    top_kind: str | None = None
    const: bool = False  # inside a constant-expression context
    method: str = ""


@dataclass
class _RawBranch:
    class_qualname: str
    kind: str
    condition: str
    line: int
    method: str
    local_id: int


def _is_default_label(label) -> bool:
    return any(
        not c.is_named and c.type == "default" for c in label.children
    )


def _has_static_final(node, src: bytes) -> bool:
    mods = J.child(node, "modifiers")
    if mods is None:
        return False
    names = {J.text(c, src) for c in mods.children if not c.is_named}
    return "static" in names and "final" in names


class _Instrumenter:
    """Collects zero-width text insertions implementing the instrumentation.

    All edits are insertions (no replacements), so nested branch points
    compose: insertions at the same byte offset are ordered deepest-node
    first when applied.
    """

    def __init__(self, rel: str, text: str):
        self.rel = rel
        raw = text.encode("utf-8")
        self.src = raw.decode("utf-8", errors="replace").encode("utf-8")
        self.tree = J.parser().parse(self.src)
        self.hit, self.rec = _fresh_names(
            self.src.decode("utf-8", errors="replace")
        )
        self.insertions: list[tuple[int, int, int, bytes]] = []  # pos,depth,seq,bytes
        self._seq = 0
        self._branches: list[_RawBranch] = []
        self._local: dict[str, int] = {}  # class qualname -> next local id
        self._tops: dict[str, tuple] = {}  # qualname -> (node, kind, simple)
        self._package = ""

    # -- low-level helpers -------------------------------------------------

    def _emit(self, pos: int, text: str, depth: int) -> None:
        self.insertions.append((pos, depth, self._seq, text.encode("utf-8")))
        self._seq += 1

    def _hitref(self, ctx: _Ctx) -> str:
        return f"{ctx.top_simple}.{self.hit}"

    def _recref(self, ctx: _Ctx) -> str:
        return f"{ctx.top_simple}.{self.rec}"

    def _new_branch(self, node, ctx: _Ctx, kind: str, condition: str) -> int | None:
        if ctx.top_qual is None:
            return None
        lid = self._local.get(ctx.top_qual, 0)
        self._local[ctx.top_qual] = lid + 1
        self._branches.append(
            _RawBranch(
                class_qualname=ctx.top_qual,
                kind=kind,
                condition=condition,
                line=node.start_point[0] + 1,
                method=ctx.method,
                local_id=lid,
            )
        )
        return lid

    def _instrument_body(self, stmt, lid: int, ctx: _Ctx, depth: int) -> None:
        """Prepend the counter to a branch body, wrapping in a block if needed."""
        href = self._hitref(ctx)
        if stmt.type == "block":
            self._emit(stmt.start_byte + 1, f" {href}[{lid}]++;", depth)
        else:
            self._emit(stmt.start_byte, f"{{ {href}[{lid}]++; ", depth)
            self._emit(stmt.end_byte, " }", depth)

    # -- branch-point handlers ---------------------------------------------

    def _handle_if(self, node, ctx: _Ctx, depth: int) -> None:
        cond = node.child_by_field_name("condition")
        cons = node.child_by_field_name("consequence")
        alt = node.child_by_field_name("alternative")
        ctext = J.text(cond, self.src) if cond is not None else ""
        href = self._hitref(ctx)
        then_id = self._new_branch(node, ctx, "if-then", ctext)
        if then_id is not None and cons is not None:
            self._instrument_body(cons, then_id, ctx, depth)
        else_id = self._new_branch(node, ctx, "if-else", ctext)
        if else_id is None:
            return
        if alt is None:
            # No else: synthesize an empty else arm so the "condition false"
            # edge is countable. Purely additive.
            self._emit(node.end_byte, f" else {{ {href}[{else_id}]++; }}", depth)
        elif alt.type == "if_statement":
            # else-if chain: brace the nested if under the outer else arm.
            # The original `else` keyword stays; the nested if keeps its own
            # instrumentation when the walk recurses into it.
            self._emit(alt.start_byte, f"{{ {href}[{else_id}]++; ", depth)
            self._emit(alt.end_byte, " }", depth)
        else:
            self._instrument_body(alt, else_id, ctx, depth)

    def _handle_loop(self, node, ctx: _Ctx, depth: int) -> None:
        kind = _LOOP_NODE_KINDS[node.type]
        body = node.child_by_field_name("body")
        if body is None:
            return
        cond = node.child_by_field_name("condition")
        ctext = J.text(cond, self.src) if cond is not None else ""
        lid = self._new_branch(node, ctx, kind, ctext)
        if lid is not None:
            self._instrument_body(body, lid, ctx, depth)

    def _handle_ternary(self, node, ctx: _Ctx, depth: int) -> None:
        cons = node.child_by_field_name("consequence")
        alt = node.child_by_field_name("alternative")
        if cons is None or alt is None:
            return
        cond = node.child_by_field_name("condition")
        ctext = J.text(cond, self.src) if cond is not None else ""
        t_id = self._new_branch(node, ctx, "ternary-true", ctext)
        f_id = self._new_branch(node, ctx, "ternary-false", ctext)
        if t_id is None or f_id is None:
            return
        rec, href = self._recref(ctx), self._hitref(ctx)
        # The condition itself is left untouched; each arm is wrapped in the
        # recording call, which evaluates it exactly once.
        self._emit(cons.start_byte, f"{rec}({href},{t_id},(", depth)
        self._emit(cons.end_byte, "))", depth)
        self._emit(alt.start_byte, f"{rec}({href},{f_id},(", depth)
        self._emit(alt.end_byte, "))", depth)

    def _handle_switch(self, node, ctx: _Ctx, depth: int) -> None:
        blk = J.child(node, "switch_block")
        if blk is None:
            return
        parent = node.parent
        is_stmt = parent is not None and parent.type in _SWITCH_STMT_PARENTS
        href = self._hitref(ctx)
        has_default = False
        for grp in J.named(blk):
            if grp.type == "switch_block_statement_group":
                for label in J.named(grp):
                    if label.type != "switch_label":
                        continue
                    is_def = _is_default_label(label)
                    has_default = has_default or is_def
                    lid = self._new_branch(
                        label,
                        ctx,
                        "switch-default" if is_def else "switch-case",
                        J.text(label, self.src),
                    )
                    if lid is not None:
                        # In this grammar the ':' is an anonymous sibling of
                        # the label (not part of it); count after it.
                        colon = next(
                            (
                                c
                                for c in grp.children
                                if not c.is_named and c.type == ":"
                            ),
                            None,
                        )
                        pos = colon.end_byte if colon is not None else label.end_byte
                        self._emit(pos, f" {href}[{lid}]++;", depth)
            elif grp.type == "switch_rule":
                label = J.child(grp, "switch_label")
                is_def = label is not None and _is_default_label(label)
                has_default = has_default or is_def
                lid = self._new_branch(
                    grp,
                    ctx,
                    "switch-default" if is_def else "switch-case",
                    J.text(label, self.src) if label is not None else "",
                )
                body = next(
                    (c for c in J.named(grp) if c.type != "switch_label"),
                    None,
                )
                if lid is None or body is None:
                    continue
                if body.type == "block":
                    self._emit(body.start_byte + 1, f" {href}[{lid}]++;", depth)
                elif body.type == "throw_statement":
                    self._emit(body.start_byte, f"{{ {href}[{lid}]++; ", depth)
                    self._emit(body.end_byte, " }", depth)
                elif body.type == "expression_statement" and not is_stmt:
                    # Expression switch: the arrow body is a value, so the
                    # counter goes into a block that yields it. The original
                    # trailing ';' becomes the yield's semicolon.
                    self._emit(body.start_byte, f"{{ {href}[{lid}]++; yield ", depth)
                    self._emit(body.end_byte, " }", depth)
                else:
                    # Statement switch (or throw already handled): plain wrap.
                    self._emit(body.start_byte, f"{{ {href}[{lid}]++; ", depth)
                    self._emit(body.end_byte, " }", depth)
        if is_stmt and not has_default:
            # A switch statement with no default silently does nothing on an
            # unmatched value; add a counting default. (Never for expression
            # switches: a valueless `break;` would not compile there.)
            lid = self._new_branch(blk, ctx, "switch-default", "default:")
            if lid is not None:
                self._emit(
                    blk.end_byte - 1,
                    f" default: {href}[{lid}]++; break;",
                    depth,
                )

    def _handle_catch(self, node, ctx: _Ctx, depth: int) -> None:
        body = node.child_by_field_name("body")
        if body is None:
            return
        param = J.child(node, "catch_formal_parameter")
        ctype = ""
        if param is not None:
            t = J.child(param, "catch_type")
            ctype = J.text(t, self.src) if t is not None else ""
        lid = self._new_branch(node, ctx, "catch", ctype)
        if lid is not None:
            # catch bodies are always blocks.
            self._emit(body.start_byte + 1, f" {self._hitref(ctx)}[{lid}]++;", depth)

    # -- tree walk ----------------------------------------------------------

    def _walk(self, node, ctx: _Ctx, depth: int) -> None:
        t = node.type
        if t in _TYPE_DECLS and node.parent is not None and node.parent.type == "program":
            ident = J.child(node, "identifier")
            simple = J.text(ident, self.src) if ident is not None else "?"
            qual = f"{self._package}.{simple}" if self._package else simple
            ctx = _Ctx(
                top_simple=simple,
                top_qual=qual,
                top_kind=t,
                const=ctx.const,
                method=ctx.method,
            )
            self._tops.setdefault(qual, (node, t, simple))
        if t in ("method_declaration", "constructor_declaration"):
            ident = J.child(node, "identifier")
            ctx = _Ctx(
                top_simple=ctx.top_simple,
                top_qual=ctx.top_qual,
                top_kind=ctx.top_kind,
                const=ctx.const,
                method=J.text(ident, self.src) if ident is not None else "",
            )
        if t in ("annotation", "marker_annotation") or t.endswith("_annotation"):
            ctx = _Ctx(
                top_simple=ctx.top_simple,
                top_qual=ctx.top_qual,
                top_kind=ctx.top_kind,
                const=True,
                method=ctx.method,
            )
        if t == "switch_label":
            # case labels must stay compile-time constants.
            ctx = _Ctx(
                top_simple=ctx.top_simple,
                top_qual=ctx.top_qual,
                top_kind=ctx.top_kind,
                const=True,
                method=ctx.method,
            )
        if t == "field_declaration" and _has_static_final(node, self.src):
            ctx = _Ctx(
                top_simple=ctx.top_simple,
                top_qual=ctx.top_qual,
                top_kind=ctx.top_kind,
                const=True,
                method=ctx.method,
            )

        # Handle the node itself before recursing, so that at any shared
        # byte offset the deeper node's insertion sorts first.
        if t == "if_statement":
            self._handle_if(node, ctx, depth)
        elif t in _LOOP_NODE_KINDS:
            self._handle_loop(node, ctx, depth)
        elif t == "ternary_expression":
            if not ctx.const and ctx.top_qual is not None:
                self._handle_ternary(node, ctx, depth)
        elif t == "switch_expression":
            self._handle_switch(node, ctx, depth)
        elif t == "catch_clause":
            self._handle_catch(node, ctx, depth)

        for c in node.children:
            self._walk(c, ctx, depth + 1)

    # -- injection of counter field + recorder --------------------------------

    def _injection_point(self, tnode, kind: str) -> tuple[int, str]:
        """(byte offset, prefix text) for the injected members."""
        if kind == "enum_declaration":
            # Enum constants must precede any member declarations; inject
            # after them (adding the ';' when the source omits it).
            body = J.child(tnode, "enum_body")
            if body is None:  # degenerate; give up on injection here
                return tnode.end_byte - 1, ""
            consts = J.child(body, "enum_constants")
            if consts is None:
                return body.start_byte + 1, ""
            kids = body.children
            idx = kids.index(consts)
            for k in kids[idx + 1 :]:
                if not k.is_named and k.type == ";":
                    return k.end_byte, ""
                if k.is_named:
                    break
            return consts.end_byte, ";"
        body = next(
            (c for c in tnode.children if c.type.endswith("_body")),
            None,
        )
        if body is None:  # should not happen; inject nothing
            return tnode.end_byte - 1, ""
        return body.start_byte + 1, ""

    def _members_text(self, n: int) -> str:
        lines = [f"\n    public static int[] {self.hit} = new int[{n}];"]
        for jt in _REC_PRIMITIVES:
            lines.append(
                f"    public static {jt} {self.rec}(int[] h, int i, {jt} v)"
                f" {{ h[i]++; return v; }}"
            )
        lines.append(
            f"    public static <T> T {self.rec}(int[] h, int i, T v)"
            f" {{ h[i]++; return v; }}\n"
        )
        return "\n".join(lines)

    # -- entry point ----------------------------------------------------------

    def run(self) -> tuple[str, list[_RawBranch], list[tuple[str, str]]]:
        root = self.tree.root_node
        for top in J.named(root):
            if top.type == "package_declaration":
                p = J.child(top, "scoped_identifier", "identifier")
                if p is not None:
                    self._package = J.text(p, self.src).replace(" ", "")
        self._walk(root, _Ctx(), 0)

        for qual, (tnode, kind, _simple) in self._tops.items():
            n = self._local.get(qual, 0)
            if n == 0:
                continue
            pos, prefix = self._injection_point(tnode, kind)
            self._emit(pos, prefix + self._members_text(n), -1)

        ins = sorted(self.insertions, key=lambda e: (e[0], -e[1], e[2]))
        out = bytearray()
        prev = 0
        for pos, _d, _s, text in ins:
            out += self.src[prev:pos]
            out += text
            prev = pos
        out += self.src[prev:]
        new_text = bytes(out).decode("utf-8", errors="replace")
        class_infos = [(q, self.hit) for q in self._local]
        return new_text, self._branches, class_infos


def instrument(
    sources: dict[str, str], work_dir: str | Path | None = None
) -> InstrumentedCopy:
    """Instrument branch counters into copies of the given Java sources.

    ``sources`` maps a repo-relative path (e.g. ``"com/example/K.java"``)
    to its source text -- the kernel plus its closure. Instrumented copies
    are written under ``work_dir`` (a fresh temp dir when omitted),
    preserving relative paths, and can be compiled with ``javac`` as-is.

    Returns :class:`InstrumentedCopy` with the branch manifest. Raises
    ``RuntimeError`` only for IO problems; instrumentation itself is
    best-effort and semantics-preserving (see the module docstring).
    """
    work = Path(work_dir) if work_dir is not None else Path(
        tempfile.mkdtemp(prefix="kp-cov-")
    )
    work.mkdir(parents=True, exist_ok=True)
    branches: list[BranchInfo] = []
    coverage_classes: list[tuple[str, str]] = []
    files: list[str] = []
    gid = 0
    for rel in sorted(sources.keys()):
        rel_posix = Path(rel).as_posix()
        target = work / rel_posix
        target.parent.mkdir(parents=True, exist_ok=True)
        inst = _Instrumenter(rel_posix, sources[rel])
        new_text, raw_branches, class_infos = inst.run()
        target.write_text(new_text, encoding="utf-8")
        files.append(rel_posix)
        for rb in raw_branches:
            branches.append(
                BranchInfo(
                    global_id=gid,
                    class_qualname=rb.class_qualname,
                    kind=rb.kind,
                    condition=rb.condition,
                    line=rb.line,
                    method=rb.method,
                    local_id=rb.local_id,
                )
            )
            gid += 1
        coverage_classes.extend(class_infos)
    return InstrumentedCopy(
        work_dir=work,
        files=files,
        branches=branches,
        coverage_classes=coverage_classes,
    )


# ---------------------------------------------------------------------------
# coverage run: compile instrumented copy, replay cases, dump counters
# ---------------------------------------------------------------------------

# Anchor inside harness.render_harness() output where the COV command hooks
# in. If render_harness() ever changes shape, _render_cov_harness() raises a
# clear RuntimeError naming this constant.
_HANDLE_ANCHOR = (
    '    static String handle(String line) {\n'
    '        String[] tok = line.split(" ");'
)


def _render_cov_harness(method: J.MethodInfo, instrumented: InstrumentedCopy) -> str:
    """Render a coverage variant of the standard Java harness.

    Takes :func:`harness.render_harness` output verbatim and patches in:

    - a ``COV`` command at the top of ``handle()`` that dumps every
      instrumented class's hit array as
      ``OK <qual>:c0,c1,...;<qual>:c0,...;``;
    - a ``dumpCoverage()`` static method appended to the class.

    The CALL/BENCH protocol, argument decoding and result encoding are
    untouched, so coverage runs exercise exactly the same code path as
    ``verify``.
    """
    base = H.render_harness(method, "java", "HarnessCov")
    if _HANDLE_ANCHOR not in base:
        raise RuntimeError(
            "coverage: harness.render_harness() output no longer contains the "
            "expected handle() anchor; update _HANDLE_ANCHOR in coverage.py"
        )
    hooked = base.replace(
        _HANDLE_ANCHOR,
        '    static String handle(String line) {\n'
        '        if (line.equals("COV")) return dumpCoverage();\n'
        '        String[] tok = line.split(" ");',
        1,
    )
    parts = [
        "\n    static String dumpCoverage() {\n",
        "        StringBuilder sb = new StringBuilder();\n",
    ]
    for qual, hit in instrumented.coverage_classes:
        parts.append(f'        sb.append("{qual}").append(\':\');\n')
        parts.append(
            "        { int[] h = "
            + qual
            + "."
            + hit
            + "; for (int i = 0; i < h.length; i++)"
            + " { if (i > 0) sb.append(','); sb.append(h[i]); } }\n"
        )
        parts.append("        sb.append(';');\n")
    parts.append('        return "OK " + sb.toString();\n    }\n')
    dump = "".join(parts)
    stripped = hooked.rstrip("\n")
    if not stripped.endswith("}"):
        raise RuntimeError("coverage: unexpected render_harness() tail")
    return stripped[:-1] + dump + "}\n"


def _parse_coverage(
    payload: str, instrumented: InstrumentedCopy
) -> CoverageData:
    """Parse a ``COV`` dump payload into :class:`CoverageData`."""
    by_class: dict[str, list[BranchInfo]] = {}
    for b in instrumented.branches:
        by_class.setdefault(b.class_qualname, []).append(b)
    segs = [s for s in payload.split(";") if s]
    if len(segs) != len(instrumented.coverage_classes):
        raise RuntimeError(
            f"coverage dump has {len(segs)} classes, manifest has "
            f"{len(instrumented.coverage_classes)}"
        )
    per_branch: dict[int, int] = {}
    for (qual, _hit), seg in zip(instrumented.coverage_classes, segs):
        q, _, counts = seg.partition(":")
        if q != qual:
            raise RuntimeError(
                f"coverage dump class order mismatch: {q!r} vs {qual!r}"
            )
        vals = [int(x) for x in counts.split(",")] if counts else []
        expected = by_class.get(qual, [])
        if len(vals) != len(expected):
            raise RuntimeError(
                f"coverage dump for {qual}: {len(vals)} counters, "
                f"manifest has {len(expected)}"
            )
        for b, v in zip(expected, vals):
            per_branch[b.global_id] = v
    total = len(instrumented.branches)
    hit = sum(1 for b in instrumented.branches if per_branch.get(b.global_id, 0) > 0)
    return CoverageData(
        branches_total=total,
        branches_hit=hit,
        per_branch_hits=per_branch,
        branches=list(instrumented.branches),
    )


def _open_cov_session(classes_dir: Path, java_bin: str | None) -> H.LineSession:
    java = java_bin or shutil.which("java") or "java"
    return H.LineSession([java, "-cp", str(classes_dir), "HarnessCov"])


class CoveredSession:
    """A compiled instrumented copy with an open coverage-harness session.

    Compiling once and replaying several case batches through one JVM
    session is much cheaper than repeated :func:`run_covered` calls, each
    of which pays javac + JVM startup again. Hit counters accumulate
    across :meth:`replay` calls, so :meth:`coverage` dumps the cumulative
    state; callers merge per-batch dumps with a per-branch max when they
    need batch-scoped numbers.

    Usable as a context manager; :meth:`close` is idempotent.
    """

    def __init__(
        self,
        method: J.MethodInfo,
        instrumented: InstrumentedCopy,
        java_bin: str | None = None,
    ):
        work = Path(instrumented.work_dir)
        (work / "HarnessCov.java").write_text(
            _render_cov_harness(method, instrumented), encoding="utf-8"
        )
        java_files = sorted(work.rglob("*.java"))
        self._classes_dir = work / "classes"
        # v0.8: same dogfood fix as build_env -- resolve transitive deps via
        # -sourcepath instead of assuming the instrumented set is closed.
        H.compile_java(
            java_files, self._classes_dir,
            H.kernel_sourcepath_roots(work, method),
        )
        self._method = method
        self._instrumented = instrumented
        self._java_bin = java_bin
        self._sess = _open_cov_session(self._classes_dir, java_bin)
        self._closed = False

    def replay(
        self, cases: list[list], per_case_timeout: float = 30.0
    ) -> None:
        """Send ``cases`` through the standard ``CALL`` protocol.

        Inputs the Java original rejects (``ERR``) contribute no coverage,
        mirroring the Java-first protocol of :func:`harness.run_verify`. A
        hung case may desync the session; it is reopened and the case is
        skipped -- its coverage contribution is lost, nothing else.
        """
        for args in cases:
            line = H.encode_call(self._method, args)
            try:
                resp = self._sess.ask(line, per_case_timeout)
            except (TimeoutError, RuntimeError):
                self._sess.close()
                self._sess = _open_cov_session(self._classes_dir, self._java_bin)
                continue
            if resp.startswith("ERR"):
                continue
            if not resp.startswith("OK"):
                raise RuntimeError(
                    f"coverage harness protocol error: {resp!r}"
                )

    def coverage(self, per_case_timeout: float = 30.0) -> CoverageData:
        """Dump the cumulative hit counters as :class:`CoverageData`."""
        try:
            raw = self._sess.ask("COV", per_case_timeout)
        except (TimeoutError, RuntimeError) as e:
            raise RuntimeError(f"coverage dump failed: {e}") from e
        if not raw.startswith("OK "):
            raise RuntimeError(f"coverage dump failed: {raw!r}")
        return _parse_coverage(raw[3:], self._instrumented)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._sess.close()

    def __enter__(self) -> "CoveredSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def run_covered(
    method: J.MethodInfo,
    cases: list[list],
    instrumented: InstrumentedCopy,
    per_case_timeout: float = 30.0,
    java_bin: str | None = None,
) -> CoverageData:
    """Replay ``cases`` against the instrumented copy; return branch coverage.

    Compiles every ``.java`` file under ``instrumented.work_dir`` (the
    instrumented kernel + closure) together with a coverage harness derived
    from :func:`harness.render_harness`, then sends each case through the
    standard ``CALL`` protocol. Inputs the Java original rejects (``ERR``)
    contribute no coverage, mirroring the Java-first protocol of
    :func:`harness.run_verify`. A single ``COV`` dump at the end yields the
    aggregate hit counts.

    For several case batches, prefer :class:`CoveredSession` directly --
    this wrapper compiles and starts a JVM per call.
    """
    with CoveredSession(method, instrumented, java_bin) as sess:
        sess.replay(cases, per_case_timeout)
        return sess.coverage(per_case_timeout)


# ---------------------------------------------------------------------------
# guided second-round generation
# ---------------------------------------------------------------------------

_INT_LIKE = {"int", "long"}
_FLOAT_LIKE = {"float", "double"}
_LOOP_BRANCH_KINDS = {"for-body", "enhanced-for-body", "while-body", "do-body"}

_NUM_LIT_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")

_INT_RANGES = {
    "int": (-(2**31), 2**31 - 1),
    "long": (-(2**63), 2**63 - 1),
}


def _clamp_scalar(jtype: str, v: float):
    if jtype in _INT_RANGES:
        lo, hi = _INT_RANGES[jtype]
        return min(max(int(v), lo), hi)
    return float(v)


def _targeted_cases(
    method: J.MethodInfo,
    branch: BranchInfo,
    rng,
    cap: int = 8,
) -> list[list]:
    """Cases aiming at one uncovered branch.

    Numeric literals harvested from the branch condition (``x == 1234567``
    -> ``1234567``) plus the type's boundary edges are injected into
    parameters named in the condition, falling back to every scalar
    parameter of a matching type when the condition names none.
    """
    params = method.params
    cond = branch.condition or ""
    idents = set(_IDENT_RE.findall(cond))
    lits = _NUM_LIT_RE.findall(cond)[:3]
    if not lits:
        return []
    named = [i for i, p in enumerate(params) if p.name in idents]
    cands: list[list] = []

    def idxs_for(types: set[str]) -> list[int]:
        hit = [i for i in named if params[i].jtype in types]
        return hit or [i for i, p in enumerate(params) if p.jtype in types]

    for lit in lits:
        is_float = "." in lit or "e" in lit.lower() or "E" in lit
        if is_float:
            fv = float(lit)
            targets = [(i, fv + d) for i in idxs_for(_FLOAT_LIKE) for d in (0.0, 1.0, -1.0)]
        else:
            iv = int(lit)
            targets = [(i, iv + d) for i in idxs_for(_INT_LIKE) for d in (0, 1, -1)]
        for i, v in targets[:3]:
            base = H.gen_case(method, rng)
            base[i] = _clamp_scalar(params[i].jtype, v)
            cands.append(base)
            if len(cands) >= cap:
                return cands
        # Boundary edges for the same parameters ("boundary edges near
        # comparisons"): MIN/MAX/0/1/-1 and friends.
        for i in idxs_for(_INT_LIKE | _FLOAT_LIKE)[:2]:
            try:
                edges = H._boundary_scalar_edges(params[i].jtype)
            except AssertionError:
                continue
            for ev in edges[:5]:
                base = H.gen_case(method, rng)
                base[i] = ev
                cands.append(base)
                if len(cands) >= cap:
                    return cands
    return cands


def _nonempty(v) -> bool:
    if not isinstance(v, list):
        return True
    if not v:
        return False
    return all(_nonempty(x) for x in v)


def _nondegenerate_case(method: J.MethodInfo, rng, tries: int = 25) -> list:
    """A random case with no empty arrays (loops over them can iterate)."""
    for _ in range(tries):
        args = H.gen_case(method, rng)
        if all(_nonempty(v) for v in args):
            return args
    return H.gen_case(method, rng)


def guided_cases(
    method: J.MethodInfo,
    coverage: CoverageData,
    rng,
    n: int,
) -> list[list]:
    """Generate a second input round biased at uncovered branches.

    For each uncovered branch, literal/boundary-targeted cases are emitted
    first (see :func:`_targeted_cases`); uncovered loop bodies additionally
    get non-degenerate inputs; remaining slots are filled with plain random
    cases. Returns exactly ``n`` cases (deduplicated).
    """
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
    for b in uncovered:
        if len(out) >= n:
            break
        for args in _targeted_cases(method, b, rng):
            push(args)
    for b in uncovered:
        if len(out) >= n:
            break
        if b.kind in _LOOP_BRANCH_KINDS:
            push(_nondegenerate_case(method, rng))
    while len(out) < n:
        push(H.gen_case(method, rng))
    return out


# ---------------------------------------------------------------------------
# failure minimization (ddmin-style)
# ---------------------------------------------------------------------------


class _BudgetExceeded(Exception):
    pass


def _ddmin_list(lst: list, still_fails: Callable[[list], bool]) -> list:
    """Zeller ddmin: minimal contiguous sublist that still fails.

    1-minimal: no single chunk removal or chunk isolation keeps the failure.
    """
    cur = list(lst)
    n = 2
    while len(cur) >= 2:
        chunk = max(1, len(cur) // n)
        reduced = False
        start = 0
        while start < len(cur):  # try complements (drop one chunk)
            cand = cur[:start] + cur[start + chunk :]
            if still_fails(cand):
                cur = cand
                n = max(2, n - 1)
                reduced = True
                break
            start += chunk
        if reduced:
            continue
        start = 0
        while start < len(cur):  # try single chunks
            cand = cur[start : start + chunk]
            if still_fails(cand):
                cur = cand
                n = 2
                reduced = True
                break
            start += chunk
        if reduced:
            continue
        if chunk == 1:
            break
        n = min(len(cur), n * 2)
    if cur and still_fails([]):
        return []
    return cur


def _shrink_value(v, try_fn: Callable[[object], bool]):
    """Shrink one argument value while ``try_fn`` (still-fails?) holds."""
    if isinstance(v, bool):
        return False if try_fn(False) else v
    if isinstance(v, int):
        if try_fn(0):
            return 0
        t = v
        while abs(t) > 1:
            t2 = t // 2 if t > 0 else -((-t) // 2)
            if try_fn(t2):
                t = t2
            else:
                break
        return t
    if isinstance(v, float):
        if try_fn(0.0):
            return 0.0
        t = v
        while abs(t) > 1.0 and t == t:  # t == t excludes NaN
            t2 = t / 2.0
            if try_fn(t2):
                t = t2
            else:
                break
        return t
    if isinstance(v, str):
        if try_fn(""):
            return ""
        lo, hi = 0, len(v)  # invariant: prefix[:hi] fails
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if try_fn(v[:mid]):
                hi = mid
            else:
                lo = mid
        return v[:hi]
    if isinstance(v, list):
        if not v:
            return v
        if all(isinstance(x, list) for x in v):
            return _shrink_2d(v, try_fn)
        lst = _ddmin_list(v, try_fn)

        def elem_trier(j):
            def go(cand):
                new = list(lst)
                new[j] = cand
                return try_fn(new)

            return go

        return [_shrink_value(x, elem_trier(j)) for j, x in enumerate(lst)]
    return v


def _shrink_2d(v: list, try_fn: Callable[[object], bool]) -> list:
    """Shrink a 2-D array, keeping it rectangular for the wire protocol."""
    rows = _ddmin_list(v, try_fn)
    if not rows:
        return rows
    widths = [len(r) for r in rows]
    if min(widths) == 0:
        return [r[:0] for r in rows]
    lo, hi = 0, min(widths)  # invariant: width hi fails
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if try_fn([r[:mid] for r in rows]):
            hi = mid
        else:
            lo = mid
    narrow = [r[:hi] for r in rows]

    def cell_trier(ri, ci):
        def go(cand):
            new = [list(r) for r in narrow]
            new[ri][ci] = cand
            return try_fn(new)

        return go

    return [
        [_shrink_value(x, cell_trier(ri, ci)) for ci, x in enumerate(row)]
        for ri, row in enumerate(narrow)
    ]


def minimize(
    case: list,
    fails: Callable[[list], bool],
    max_calls: int = 20000,
    max_seconds: float = 120.0,
) -> list:
    """Delta-minimize a failing input case.

    ``case`` is an argument list as produced by :func:`harness.gen_case`;
    ``fails(case)`` must be a deterministic predicate reporting whether the
    case still reproduces the failure (typically: the Java and JNI harnesses
    disagree on it). Each argument is shrunk independently -- arrays via
    ddmin, scalars toward zero, strings via prefix binary search -- keeping
    only variants that still fail. The result is 1-minimal w.r.t. the
    shrink steps: no single step applied here keeps it failing.

    Raises ``ValueError`` when the input case does not fail, and stops
    early (returning the best-so-far) when either the ``max_calls``
    predicate-call budget or the ``max_seconds`` wall-clock budget is
    exhausted. The wall-clock budget matters because a single ``fails``
    call can block up to the harness timeout (tens of seconds), so the
    call budget alone does not bound ddmin's runtime.

    ``max_seconds`` must be positive; it is checked before each predicate
    call, so a single slow call may overrun it by at most one call.
    """
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
