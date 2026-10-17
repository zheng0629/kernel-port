"""`mine`: find pure-compute kernel candidates in a Java repo.

For every *static* method in the repo:

1. run the conservative purity check (impure methods are listed with reasons,
   never silently dropped);
2. count how often each method is called across the repo (call frequency);
3. score compute intensity with a cheap AST heuristic: loops, arithmetic,
   array accesses, recursion and Math calls;
4. rank by ``(1 + calls) * intensity``.

Only static methods are considered in v0.1: kernels are utility-style
functions, and JNI glue for instance methods needs an object instance, which
the harness does not provide yet.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import java_ast as J
from .purity import PurityReport, check_purity_repo


@dataclass
class Candidate:
    qualname: str
    file: str
    lineno: int
    signature: str
    pure: bool
    reasons: list[str]
    calls: int
    intensity: float
    score: float


def _count_calls(repo: J.JavaRepo) -> dict[str, int]:
    """Count method invocations per callee qualname (best effort)."""
    counts: dict[str, int] = {}
    simple_to_methods: dict[str, list[str]] = {}
    for (cls_qn, name), overloads in repo.methods.items():
        for m in overloads:
            simple_to_methods.setdefault(name, []).append(m.qualname)

    def bump(qn: str) -> None:
        counts[qn] = counts.get(qn, 0) + 1

    for (cls_qn, _name), overloads in repo.methods.items():
        for m in overloads:
            if m.body is None:
                continue
            path = repo.root / m.file
            raw = path.read_bytes()
            src = raw.decode("utf-8", errors="replace").encode("utf-8")

            def visit(node, _self_qn=m.qualname) -> None:
                if node.type == "method_invocation":
                    kids = J.named(node)
                    arg_list = J.child(node, "argument_list")
                    if arg_list is not None and arg_list in kids:
                        idx = kids.index(arg_list)
                        name_node = None
                        receiver = None
                        for j in range(idx - 1, -1, -1):
                            if kids[j].type == "identifier":
                                name_node = kids[j]
                                receiver = kids[j - 1] if j > 0 else None
                                break
                        if name_node is not None:
                            cname = J.text(name_node, src)
                            rtext = J.text(receiver, src) if receiver is not None else None
                            resolved = None
                            if receiver is None or (
                                receiver.type == "this"
                            ):
                                cands = repo.methods.get((cls_qn, cname), [])
                                if len(cands) == 1:
                                    resolved = cands[0].qualname
                            elif receiver.type == "identifier" and rtext:
                                for qn in simple_to_methods.get(cname, []):
                                    if qn.rsplit(".", 1)[0].rsplit(".", 1)[-1] == rtext:
                                        resolved = qn
                                        break
                            if resolved is None:
                                # fall back: attribute to same-class method of that name
                                cands = repo.methods.get((cls_qn, cname), [])
                                if len(cands) == 1:
                                    resolved = cands[0].qualname
                            if resolved is not None and resolved != _self_qn:
                                bump(resolved)
                            elif resolved is None and len(simple_to_methods.get(cname, [])) == 1:
                                qn = simple_to_methods[cname][0]
                                if qn != _self_qn:
                                    bump(qn)
                for c in node.children:
                    if c.is_named:
                        visit(c)

            visit(m.body)
    return counts


def _method_source(repo: J.JavaRepo, m: J.MethodInfo) -> bytes:
    raw = (repo.root / m.file).read_bytes()
    return raw.decode("utf-8", errors="replace").encode("utf-8")


def _intensity(m: J.MethodInfo, src: bytes) -> float:
    """Cheap compute-intensity heuristic from the method body AST."""
    if m.body is None:
        return 0.0
    loops = arith = array_acc = math_calls = 0
    recursive = False

    def visit(node) -> None:
        nonlocal loops, arith, array_acc, math_calls, recursive
        t = node.type
        if t in ("for_statement", "enhanced_for_statement", "while_statement", "do_statement"):
            loops += 1
        elif t == "binary_expression":
            arith += 1
        elif t == "array_access":
            array_acc += 1
        elif t == "method_invocation":
            kids = J.named(node)
            for k in kids:
                if k.type == "identifier" and J.text(k, src) == m.name:
                    # possible self-recursion (name match; good enough for a heuristic)
                    recursive = True
            # Math.* call counts as compute
            for k in kids:
                if k.type == "identifier" and J.text(k, src) in ("Math", "StrictMath"):
                    math_calls += 1
                    break
        for c in node.children:
            if c.is_named:
                visit(c)

    visit(m.body)
    return 1.0 + 2.0 * loops + 0.3 * arith + 0.4 * array_acc + 1.0 * math_calls + (
        4.0 if recursive else 0.0
    )


def mine(repo: J.JavaRepo) -> list[Candidate]:
    """Return ranked kernel candidates (pure and impure alike)."""
    reports: dict[str, PurityReport] = check_purity_repo(repo)
    call_counts = _count_calls(repo)
    cands: list[Candidate] = []
    for (cls_qn, _name), overloads in repo.methods.items():
        for m in overloads:
            if not m.is_static or m.is_constructor:
                continue
            rep = reports[m.qualname]
            calls = call_counts.get(m.qualname, 0)
            intensity = _intensity(m, _method_source(repo, m))
            score = (1 + calls) * intensity if rep.pure else 0.0
            cands.append(
                Candidate(
                    qualname=m.qualname,
                    file=m.file,
                    lineno=m.lineno,
                    signature=m.signature,
                    pure=rep.pure,
                    reasons=rep.reasons,
                    calls=calls,
                    intensity=round(intensity, 1),
                    score=round(score, 1),
                )
            )
    cands.sort(key=lambda c: (c.pure, c.score), reverse=True)
    return cands
