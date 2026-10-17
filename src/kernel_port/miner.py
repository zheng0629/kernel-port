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

``find_clusters`` (v0.2 workstream C1) groups methods one level up: each
cluster is "one entry method + the transitive helper methods it calls",
ranked as a single porting candidate.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import extractor, java_ast as J
from .purity import PurityReport, _PurityChecker, check_purity_repo


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


# ---------------------------------------------------------------------------
# Function-cluster mining (v0.2 workstream C1).
#
# A *cluster* is "one entry method + the pure helper methods it calls",
# treated as a single porting candidate. Lone methods (no repo-defined
# callees) stay method-level and never produce a cluster.
# ---------------------------------------------------------------------------


@dataclass
class ClusterCandidate:
    entry: str  # qualname of the entry (root) method
    members: list[str]  # qualnames, entry first, then sorted
    pure: bool
    reasons: list[str]
    calls: int  # calls to the entry from *outside* the cluster
    score: float
    intensity: float  # sum of member intensities


def _closure(method: J.MethodInfo, repo: J.JavaRepo) -> list[J.MethodInfo]:
    """Transitive repo-method callees of `method` (BFS; cycle-safe)."""
    out: list[J.MethodInfo] = []
    seen = {method.qualname}
    queue = list(extractor._called_repo_methods(method, repo))
    while queue:
        callee = queue.pop(0)
        if callee.qualname in seen:
            continue
        seen.add(callee.qualname)
        out.append(callee)
        queue.extend(extractor._called_repo_methods(callee, repo))
    return out


def _ambiguous_callees(method: J.MethodInfo, repo: J.JavaRepo) -> list[tuple[str, int]]:
    """Invoked names with >1 matching repo overloads: ``(name, overload_count)``.

    Mirrors the candidate lookup in ``extractor._called_repo_methods``: when
    more than one overload matches a call site, the callee is *not* resolved.
    Callers record the ambiguity in ``reasons`` rather than guessing.
    """
    if method.body is None:
        return []
    raw = (repo.root / method.file).read_bytes()
    src = raw.decode("utf-8", errors="replace").encode("utf-8")
    checker = _PurityChecker(repo)  # reuse simple-name resolution
    found: list[tuple[str, int]] = []
    seen_names: set[str] = set()

    def visit(node) -> None:
        if node.type == "method_invocation":
            kids = J.named(node)
            arg_list = J.child(node, "argument_list")
            if arg_list is not None and arg_list in kids:
                idx = kids.index(arg_list)
                name_node, receiver = None, None
                for j in range(idx - 1, -1, -1):
                    if kids[j].type == "identifier":
                        name_node = kids[j]
                        receiver = kids[j - 1] if j > 0 else None
                        break
                if name_node is not None:
                    name = J.text(name_node, src)
                    recv_text = J.text(receiver, src) if receiver is not None else None
                    cands: list[J.MethodInfo] = []
                    if receiver is None or receiver.type == "this":
                        cands = repo.methods.get((method.class_qualname, name), [])
                    elif receiver.type == "identifier" and recv_text:
                        cls_qn = checker.class_by_simple_name(recv_text)
                        if cls_qn is not None:
                            cands = repo.methods.get((cls_qn, name), [])
                    if len(cands) > 1 and name not in seen_names:
                        seen_names.add(name)
                        found.append((name, len(cands)))
        for c in node.children:
            if c.is_named:
                visit(c)

    visit(method.body)
    return found


def _call_sites_to(repo: J.JavaRepo, target_qn: str, skip: set[str]) -> int:
    """Count invocation sites resolving to `target_qn`, skipping callers in `skip`.

    Same best-effort resolution as `_count_calls` (which this mirrors line for
    line); used to count calls to a cluster entry coming from *outside* the
    cluster.
    """
    simple_to_methods: dict[str, list[str]] = {}
    for (cls_qn, name), overloads in repo.methods.items():
        for m in overloads:
            simple_to_methods.setdefault(name, []).append(m.qualname)

    total = 0
    for (cls_qn, _name), overloads in repo.methods.items():
        for m in overloads:
            if m.body is None or m.qualname in skip:
                continue
            path = repo.root / m.file
            raw = path.read_bytes()
            src = raw.decode("utf-8", errors="replace").encode("utf-8")

            def visit(node, _self_qn=m.qualname, _cls_qn=cls_qn) -> None:
                nonlocal total
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
                            if receiver is None or (receiver.type == "this"):
                                cands = repo.methods.get((_cls_qn, cname), [])
                                if len(cands) == 1:
                                    resolved = cands[0].qualname
                            elif receiver.type == "identifier" and rtext:
                                for qn in simple_to_methods.get(cname, []):
                                    if qn.rsplit(".", 1)[0].rsplit(".", 1)[-1] == rtext:
                                        resolved = qn
                                        break
                            if resolved is None:
                                # fall back: attribute to same-class method of that name
                                cands = repo.methods.get((_cls_qn, cname), [])
                                if len(cands) == 1:
                                    resolved = cands[0].qualname
                            if resolved is not None and resolved != _self_qn:
                                if resolved == target_qn:
                                    total += 1
                            elif resolved is None and len(simple_to_methods.get(cname, [])) == 1:
                                qn = simple_to_methods[cname][0]
                                if qn != _self_qn and qn == target_qn:
                                    total += 1
                for c in node.children:
                    if c.is_named:
                        visit(c)

            visit(m.body)
    return total


def _pick_entry(
    seed: J.MethodInfo, members: list[J.MethodInfo], repo: J.JavaRepo
) -> J.MethodInfo:
    """The cluster root: the member not called by any other member.

    Falls back to the seed on cycles (every member has an incoming edge) or
    ties (deterministic: lexicographically smallest qualname).
    """
    by_qn = {m.qualname: m for m in members}
    inside = set(by_qn)
    incoming: dict[str, int] = {qn: 0 for qn in by_qn}
    for m in members:
        for callee in extractor._called_repo_methods(m, repo):
            if callee.qualname in inside:
                incoming[callee.qualname] += 1
    roots = sorted(qn for qn, n in incoming.items() if n == 0)
    if seed.qualname in roots:
        return seed
    if roots:
        return by_qn[roots[0]]
    return seed


def find_clusters(repo: J.JavaRepo) -> list[ClusterCandidate]:
    """Detect function clusters: one entry method + its transitive helpers.

    For every static method, the repo-defined callee closure is computed
    (``extractor._called_repo_methods``, cycle-safe). Seeds whose closure is
    empty stay method-level and produce no cluster. Clusters with an impure
    member are still emitted -- with ``pure=False`` and per-member reasons --
    never silently dropped. Clusters sort by ``(pure, score)`` descending,
    like :func:`mine`.
    """
    reports: dict[str, PurityReport] = check_purity_repo(repo)
    seeds = sorted(
        (
            m
            for (_cls_qn, _name), overloads in repo.methods.items()
            for m in overloads
            if m.is_static and not m.is_constructor
        ),
        key=lambda m: m.qualname,
    )
    clusters: list[ClusterCandidate] = []
    seen_sets: set[frozenset] = set()
    for seed in seeds:
        members = [seed] + _closure(seed, repo)
        if len(members) < 2:
            continue  # lone method: stays method-level
        key = frozenset(m.qualname for m in members)
        if key in seen_sets:
            continue  # same member set from another seed (cycles)
        seen_sets.add(key)
        entry = _pick_entry(seed, members, repo)
        member_qns = {m.qualname for m in members}

        pure = True
        reasons: list[str] = []
        for m in sorted(members, key=lambda x: x.qualname):
            rep = reports.get(m.qualname)
            if rep is not None and not rep.pure:
                pure = False
                detail = "; ".join(rep.reasons) if rep.reasons else "impure"
                reasons.append(f"{m.qualname}: impure ({detail})")
        for m in sorted(members, key=lambda x: x.qualname):
            for name, n_overloads in _ambiguous_callees(m, repo):
                reasons.append(
                    f"{m.qualname}: unresolvable callee '{name}' "
                    f"({n_overloads} overloads in repo; excluded from cluster)"
                )

        calls = _call_sites_to(repo, entry.qualname, skip=member_qns)
        raw_intensity = sum(_intensity(m, _method_source(repo, m)) for m in members)
        score = (1 + calls) * raw_intensity
        clusters.append(
            ClusterCandidate(
                entry=entry.qualname,
                members=[entry.qualname]
                + sorted(qn for qn in member_qns if qn != entry.qualname),
                pure=pure,
                reasons=reasons,
                calls=calls,
                intensity=round(raw_intensity, 1),
                score=round(score, 1),
            )
        )
    clusters.sort(key=lambda c: (c.pure, c.score), reverse=True)
    return clusters
