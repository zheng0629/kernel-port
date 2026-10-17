"""`extract`: pull a kernel plus its dependency closure out of the repo.

Given a qualified method name, emit:

- the purity report (pure, or impure with concrete reasons -- in which case
  extraction still prints the report but refuses to emit prompt material);
- the *dependency closure*: helper methods the kernel calls (transitively)
  and ``static final`` constants it reads, each with source text;
- clean prompt material for the calling agent's LLM step: type signature,
  purity verdict, and the full source of the kernel + closure.

Output is either human-readable text (default) or JSON (``--format json``)
for programmatic agent consumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import java_ast as J
from .purity import ClassPurityReport, PurityReport, _PurityChecker, check_class_portable


@dataclass
class ClosureMember:
    kind: str  # "method" | "constant"
    qualname: str
    source: str
    signature: str = ""


@dataclass
class Extraction:
    method: J.MethodInfo
    purity: PurityReport
    closure: list[ClosureMember] = field(default_factory=list)


def _called_repo_methods(method: J.MethodInfo, repo: J.JavaRepo) -> list[J.MethodInfo]:
    """Direct repo-method callees of `method` (same resolution as purity)."""
    if method.body is None:
        return []
    raw = (repo.root / method.file).read_bytes()
    src = raw.decode("utf-8", errors="replace").encode("utf-8")
    checker = _PurityChecker(repo)  # reuse simple-name resolution
    found: list[J.MethodInfo] = []

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
                    target = None
                    if receiver is None or receiver.type == "this":
                        cands = repo.methods.get((method.class_qualname, name), [])
                        if len(cands) == 1:
                            target = cands[0]
                    elif receiver.type == "identifier" and recv_text:
                        cls_qn = checker.class_by_simple_name(recv_text)
                        if cls_qn is not None:
                            cands = repo.methods.get((cls_qn, name), [])
                            if len(cands) == 1:
                                target = cands[0]
                    if target is not None and target.qualname != method.qualname:
                        found.append(target)
        for c in node.children:
            if c.is_named:
                visit(c)

    visit(method.body)
    # dedupe, keep order
    seen: set[str] = set()
    out: list[J.MethodInfo] = []
    for m in found:
        if m.qualname not in seen:
            seen.add(m.qualname)
            out.append(m)
    return out


def _read_constants(method: J.MethodInfo, repo: J.JavaRepo) -> list[J.FieldInfo]:
    """static final constants read by `method` (transitively via closure)."""
    if method.body is None:
        return []
    raw = (repo.root / method.file).read_bytes()
    src = raw.decode("utf-8", errors="replace").encode("utf-8")
    cls = repo.classes.get(method.class_qualname)
    if cls is None:
        return []
    names: set[str] = set()

    def visit(node) -> None:
        if node.type == "field_access":
            kids = J.named(node)
            if kids:
                fname = J.text(kids[-1], src)
                f = cls.fields.get(fname)
                if f is not None and f.is_constant:
                    names.add(fname)
        elif node.type == "identifier":
            nm = J.text(node, src)
            f = cls.fields.get(nm)
            if f is not None and f.is_constant:
                names.add(nm)
        for c in node.children:
            if c.is_named:
                visit(c)

    visit(method.body)
    return [cls.fields[n] for n in sorted(names)]


def extract(repo: J.JavaRepo, qualname: str) -> Extraction:
    """Extract a kernel and its dependency closure."""
    method = repo.find_method(qualname)
    if method is None:
        overloads = repo.overloads(qualname)
        if overloads:
            raise ValueError(
                f"'{qualname}' is overloaded ({len(overloads)} variants); "
                "v0.1 requires a unique method name"
            )
        raise ValueError(f"method not found: {qualname}")
    return _extract_method(repo, method)


def _extract_method(
    repo: J.JavaRepo, method: J.MethodInfo, purity: PurityReport | None = None
) -> Extraction:
    """Extract a kernel given its MethodInfo.

    An explicit `purity` report may be supplied (class extraction passes the
    class-context report, which understands final instance fields); when
    omitted, the standard per-method check is used.
    """
    checker = _PurityChecker(repo)
    if purity is None:
        purity = checker.check(method)

    closure: list[ClosureMember] = []
    seen: set[str] = {method.qualname}
    stack = _called_repo_methods(method, repo)
    while stack:
        callee = stack.pop(0)
        if callee.qualname in seen:
            continue
        seen.add(callee.qualname)
        closure.append(
            ClosureMember(
                kind="method",
                qualname=callee.qualname,
                source=callee.source,
                signature=callee.signature,
            )
        )
        stack.extend(_called_repo_methods(callee, repo))

    for const in _read_constants(method, repo):
        closure.append(
            ClosureMember(
                kind="constant",
                qualname=const.qualname,
                source=const.source,
                signature=f"{const.jtype} {const.name}",
            )
        )
    return Extraction(method=method, purity=purity, closure=closure)


def render_text(ext: Extraction) -> str:
    """Human/agent-readable prompt material."""
    m = ext.method
    lines = [
        "# kernel-port extraction",
        "",
        f"Method:    {m.qualname}",
        f"Signature: {m.signature}",
        f"Location:  {m.file}:{m.lineno}",
        f"Purity:    {'PURE' if ext.purity.pure else 'IMPURE'}",
    ]
    if not ext.purity.pure:
        lines.append("Reasons:")
        for r in ext.purity.reasons:
            lines.append(f"  - {r}")
        lines += [
            "",
            "Refusing to emit translation material: the method is not pure.",
            "Fix the impurity or pick another candidate from `kernel-port mine`.",
        ]
        return "\n".join(lines)
    if ext.closure:
        lines.append("Closure (translate these too, or inline them):")
        for c in ext.closure:
            lines.append(f"  - ({c.kind}) {c.qualname}")
    else:
        lines.append("Closure:   (none -- self-contained)")
    lines += ["", "## Source: kernel", "", "```java", m.source, "```"]
    for c in ext.closure:
        lines += ["", f"## Source: {c.kind} {c.qualname}", "", "```java", c.source, "```"]
    lines += [
        "",
        "## Translation notes for the LLM",
        "- Target language: C++17. Keep the exact signature semantics.",
        "- Only the method body (and closure bodies) need translation;",
        "  kernel-port generates the JNI glue separately (`kernel-port glue`).",
        "- Preserve floating-point operation order; do not introduce",
        "  reassociation unless the differential test still passes.",
    ]
    return "\n".join(lines)


def to_json(ext: Extraction) -> dict:
    m = ext.method
    return {
        "method": m.qualname,
        "signature": m.signature,
        "params": [{"name": p.name, "type": p.jtype} for p in m.params],
        "return_type": m.jtype,
        "file": m.file,
        "lineno": m.lineno,
        "pure": ext.purity.pure,
        "purity_reasons": ext.purity.reasons,
        "source": m.source,
        "closure": [
            {
                "kind": c.kind,
                "qualname": c.qualname,
                "signature": c.signature,
                "source": c.source,
            }
            for c in ext.closure
        ],
    }


# ---------------------------------------------------------------------------
# Class-level extraction (workstream C2)
# ---------------------------------------------------------------------------


@dataclass
class ClassExtraction:
    class_info: J.ClassInfo
    class_purity: ClassPurityReport
    method_extractions: list[Extraction] = field(default_factory=list)
    fields: list[J.FieldInfo] = field(default_factory=list)


def extract_class(repo: J.JavaRepo, class_qualname: str) -> ClassExtraction:
    """Extract a whole class: every method (constructors first) plus fields.

    Per-method purity reports come from the class-context check, so reads of
    final instance fields in an immutable class count as pure.
    """
    cls = repo.classes.get(class_qualname)
    if cls is None:
        raise ValueError(f"class not found: {class_qualname}")
    purity = check_class_portable(cls, repo)
    methods = sorted(cls.all_methods, key=lambda m: (not m.is_constructor, m.name))
    extractions = [
        _extract_method(repo, m, purity.method_reports.get(m.qualname))
        for m in methods
    ]
    return ClassExtraction(
        class_info=cls,
        class_purity=purity,
        method_extractions=extractions,
        fields=list(cls.fields.values()),
    )


def render_text_class(ext: ClassExtraction) -> str:
    """Human/agent-readable prompt material for a whole class."""
    cls = ext.class_info
    rep = ext.class_purity
    lines = [
        "# kernel-port class extraction",
        "",
        f"Class:     {cls.qualname}",
        f"Kind:      {rep.kind}",
        f"Location:  {cls.file}",
        f"Purity:    {'PORTABLE' if rep.portable else 'NOT PORTABLE'}",
    ]
    if not rep.portable:
        lines.append("Reasons:")
        for r in rep.reasons:
            lines.append(f"  - {r}")
        lines += [
            "",
            "Refusing to emit translation material: the class is not portable",
            "as a unit. Fix the reasons above or pick another candidate.",
        ]
        return "\n".join(lines)
    lines.append("Methods:")
    for e in ext.method_extractions:
        verdict = "PURE" if e.purity.pure else "IMPURE"
        lines.append(f"  - {e.method.signature} [{verdict}]")
    if ext.fields:
        lines.append("Fields:")
        for f in ext.fields:
            mods = " ".join(sorted(f.modifiers))
            lines.append(f"  - {f.jtype} {f.name}" + (f" ({mods})" if mods else ""))
    else:
        lines.append("Fields:    (none)")
    for f in ext.fields:
        lines += ["", f"## Field: {f.qualname}", "", "```java", f.source, "```"]
    for e in ext.method_extractions:
        lines += ["", f"## Source: {e.method.qualname}", "", "```java", e.method.source, "```"]
    lines += [
        "",
        "## Translation notes for the LLM",
        "- Target language: C++17. Keep the exact signature semantics.",
        "- Stateless classes become a C++ namespace or a class of statics;",
        "  immutable classes become a C++ struct/class with const members.",
        "- Only the bodies above need translation; kernel-port generates the",
        "  JNI glue separately.",
        "- Preserve floating-point operation order; do not introduce",
        "  reassociation unless the differential test still passes.",
    ]
    return "\n".join(lines)


def to_json_class(ext: ClassExtraction) -> dict:
    cls = ext.class_info
    rep = ext.class_purity
    return {
        "class": cls.qualname,
        "kind": rep.kind,
        "file": cls.file,
        "portable": rep.portable,
        "portability_reasons": rep.reasons,
        "fields": [
            {
                "name": f.name,
                "type": f.jtype,
                "modifiers": sorted(f.modifiers),
                "source": f.source,
            }
            for f in ext.fields
        ],
        "methods": [to_json(e) for e in ext.method_extractions],
    }
