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

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import java_ast as J
from .purity import ClassPurityReport, PurityReport, _PurityChecker, check_class_portable


@dataclass
class ClosureMember:
    kind: str  # "method" | "constant"
    qualname: str
    source: str
    signature: str = ""
    class_qualname: str = ""  # owning class, for minimal-source synthesis


@dataclass
class Extraction:
    method: J.MethodInfo
    purity: PurityReport
    closure: list[ClosureMember] = field(default_factory=list)


def _called_repo_methods(method: J.MethodInfo, repo: J.JavaRepo) -> list[J.MethodInfo]:
    """Direct repo-method callees of `method` (same resolution as purity)."""
    if method.body is None:
        return []
    src = J.source_bytes(repo.root, method.file)
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
    src = J.source_bytes(repo.root, method.file)
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
    """Extract a kernel and its dependency closure.

    ``qualname`` accepts the v0.4 overload suffix, e.g.
    ``com.example.K.foo(int,double)`` (see :mod:`overloading`).
    """
    from . import overloading

    try:
        method = overloading.parse_method_ref(repo, qualname)
    except (overloading.AmbiguityError, LookupError) as exc:
        raise ValueError(str(exc)) from exc
    return _extract_method(repo, method)


def _extract_closure(repo: J.JavaRepo, method: J.MethodInfo) -> list[ClosureMember]:
    """Dependency closure of `method`: called helpers (transitively) and the
    ``static final`` constants it reads. No purity check -- callers that need
    the verdict use :func:`_extract_method`."""
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
                class_qualname=callee.class_qualname,
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
                class_qualname=const.qualname.rsplit(".", 1)[0],
            )
        )
    return closure


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

    closure = _extract_closure(repo, method)
    return Extraction(method=method, purity=purity, closure=closure)


def _file_imports(repo: J.JavaRepo, rel: str) -> list[str]:
    """Import declaration texts from a repo file (best effort)."""
    try:
        src = J.source_bytes(repo.root, rel)
    except OSError:
        return []
    try:
        tree = J.parser().parse(src)
    except Exception:
        return []
    return [
        J.text(top, src)
        for top in J.named(tree.root_node)
        if top.type == "import_declaration"
    ]


def _file_package(repo: J.JavaRepo, rel: str) -> str:
    """The file's own package (authoritative).

    Nested classes inherit the file's package, but ``ClassInfo.package``
    stores the outer qualname for them (``_parse_class`` recursion), so the
    file itself is the reliable source here.
    """
    try:
        src = J.source_bytes(repo.root, rel)
    except OSError:
        return ""
    try:
        tree = J.parser().parse(src)
    except Exception:
        return ""
    for top in J.named(tree.root_node):
        if top.type == "package_declaration":
            p = J.child(top, "scoped_identifier", "identifier")
            if p is not None:
                return J.text(p, src).replace(" ", "")
    return ""


def _needed_imports(imports: list[str], member_sources: list[str]) -> list[str]:
    """Keep only imports actually referenced by the synthesized members.

    A kernel file may import annotation-only deps (jspecify, intellij) used
    solely by *other* methods; carrying them into the minimal file would
    reintroduce the uncompilable-dependency problem this synthesis exists
    to avoid.
    """
    body = "\n".join(member_sources)
    kept = []
    for imp in imports:
        simple = imp.rstrip(";").split(".")[-1].strip()
        if simple == "*":
            # Wildcard: keep only java.* (always resolvable from the JDK).
            if re.match(r"import\s+(static\s+)?java\.", imp):
                kept.append(imp)
            continue
        if re.search(r"\b" + re.escape(simple) + r"\b", body):
            kept.append(imp)
    return kept


def _widen_visibility(source: str) -> str:
    """Rewrite ``private``/``protected`` member modifiers to ``public``.

    Only the declaration header is touched: scanning stops at the first
    ``{`` at parenthesis depth 0 (the body or initializer start), or at
    the end of the source when there is none. String and char literals
    are left alone, so a ``"private"`` inside an annotation argument or
    initializer is never rewritten.
    """
    n = len(source)
    # 1. Find the end of the declaration header.
    depth = 0
    in_str: str | None = None
    esc = False
    end = n
    i = 0
    while i < n:
        ch = source[i]
        if in_str is not None:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == in_str:
                in_str = None
        elif ch in "\"'":
            in_str = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "{" and depth == 0:
            end = i
            break
        i += 1
    # 2. Rewrite modifiers in the header, skipping string/char literals.
    header = source[:end]
    out: list[str] = []
    i = 0
    in_str = None
    esc = False
    while i < end:
        ch = header[i]
        if in_str is not None:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == in_str:
                in_str = None
            i += 1
            continue
        if ch in "\"'":
            in_str = ch
            out.append(ch)
            i += 1
            continue
        m = re.match(r"(?:private|protected)\b", header[i:])
        if m:
            out.append("public")
            i += m.end()
        else:
            out.append(ch)
            i += 1
    return "".join(out) + source[end:]


def render_minimal_sources(
    method: J.MethodInfo, closure: list[ClosureMember], repo: J.JavaRepo
) -> dict[str, str] | None:
    """Render minimal compilable Java sources for the kernel + its closure.

    Returns ``{repo-relative-path: source}`` -- one synthetic container class
    per distinct owning class -- or ``None`` when synthesis is not safe, in
    which case callers fall back to compiling the original file.

    Rationale (v0.8 dogfood): ``javac`` typechecks the *whole* original file,
    so compiling it drags in environmentally-broken siblings (test sources,
    annotation-only deps, ``jdk.internal`` users). The synthesized file holds
    exactly the extracted kernel, so the harness differentially tests the
    same code the agent was asked to translate.

    Bail-outs (return ``None``): non-``class``/``interface`` kinds,
    constants without initializers (would not compile standalone),
    unresolvable class info.
    """
    # Group members by owning class, kernel first.
    groups: dict[str, list[tuple[str, str]]] = {}
    order: list[str] = []

    def add(class_qn: str, kind: str, source: str) -> None:
        if class_qn not in groups:
            groups[class_qn] = []
            order.append(class_qn)
        groups[class_qn].append((kind, source))

    add(method.class_qualname, "method", method.source)
    for c in closure:
        if c.kind == "constant" and "=" not in c.source:
            # e.g. initialized in a static block -- cannot synthesize safely
            return None
        add(c.class_qualname, c.kind, c.source)

    out: dict[str, str] = {}
    for class_qn in order:
        cls = repo.classes.get(class_qn)
        if cls is None or cls.kind not in ("class", "interface"):
            return None
        pkg = _file_package(repo, cls.file)
        if pkg and class_qn.startswith(pkg + "."):
            nesting = class_qn[len(pkg) + 1 :].split(".")
        elif not pkg and "." not in class_qn:
            nesting = [class_qn]  # default package
        else:
            return None  # cannot determine nesting safely: use repo file
        if not nesting or any(not n.isidentifier() for n in nesting):
            return None
        members = groups[class_qn]
        imports = _needed_imports(
            _file_imports(repo, cls.file), [s for _, s in members]
        )
        lines: list[str] = []
        if pkg:
            lines.append(f"package {pkg};")
        lines.extend(imports)
        if lines:
            lines.append("")
        kw = "interface" if cls.kind == "interface" else "class"
        for i, name in enumerate(nesting):
            ind = "    " * i
            if i == 0:
                lines.append(f"{ind}public {kw} {name} {{")
            else:
                lines.append(f"{ind}public static class {name} {{")
        ind = "    " * len(nesting)
        for _kind, src in members:
            # v0.12 dogfood: the harness calls the kernel from another
            # package, so private/protected members must be widened in the
            # synthesized test-only copy (the class wrapper above is already
            # public). Widening is behavior-preserving for differential
            # testing: accessibility never changes dispatch or semantics.
            src = _widen_visibility(src)
            for ln in src.splitlines():
                lines.append(ind + ln if ln.strip() else "")
            lines.append("")
        for i in reversed(range(len(nesting))):
            lines.append("    " * i + "}")
        rel = (
            "/".join(pkg.split(".")) + "/" + nesting[0] + ".java"
            if pkg
            else nesting[0] + ".java"
        )
        out[rel] = "\n".join(lines)
    return out


def _type_mapping_notes(m: J.MethodInfo) -> list[str]:
    """Native-signature notes for 2-D array / String types (v0.4).

    Tells the translating LLM exactly what C++ signature the generated JNI
    glue presents, so the hand-written body targets the right contract.
    Empty for methods using only v0.1 types.
    """
    notes = []
    for p in m.params:
        if p.jtype in ("double[][]", "int[][]"):
            elem = "double" if p.jtype == "double[][]" else "int32_t"
            notes.append(
                f"- param {p.name}: {p.jtype} arrives flattened row-major as "
                f"`const {elem}* p_{p.name}` with `int p_{p.name}_rows, "
                f"p_{p.name}_cols` (the Java glue flattens before the native "
                f"call; one JNI array pin, rectangular arrays only)."
            )
        elif p.jtype == "String":
            notes.append(
                f"- param {p.name}: String arrives as `const char* p_{p.name}` "
                f"via GetStringUTFChars (Modified UTF-8: NUL -> 0xC0 0x80, "
                f"non-BMP chars -> CESU-8 surrogate pairs; byte-oriented "
                f"logic matches Java semantics on ASCII input)."
            )
    if m.jtype in ("double[][]", "int[][]"):
        elem = "double" if m.jtype == "double[][]" else "int32_t"
        notes.append(
            f"- return {m.jtype}: return the data as a flat row-major "
            f"`std::vector<{elem}>` and set `*p_out_rows` / `*p_out_cols` to "
            f"the result dimensions; the JNI wrapper rebuilds the 2-D array."
        )
    elif m.jtype == "String":
        notes.append(
            "- return String: return a `std::string`; the glue wraps it with "
            "NewStringUTF."
        )
    return notes


_BACKEND_NOTES = {
    "jni": (
        "- Target language: C++17. Keep the exact signature semantics.",
        "- Only the method body (and closure bodies) need translation;",
        "  kernel-port generates the JNI glue separately (`kernel-port glue`).",
    ),
    "ffm": (
        "- Target language: C (plain C ABI, no JNI wrapper). Keep the exact signature semantics.",
        "- Only the function body (and closure bodies) need translation;",
        "  kernel-port generates the Panama FFM glue separately (`kernel-port glue --backend ffm`).",
        "- Use the parameter names from the generated skeleton (e.g. p_<name>, p_<name>_len).",
    ),
    "rust": (
        "- Target language: Rust. Keep the exact signature semantics.",
        "- Only the function body (and closure bodies) need translation;",
        "  kernel-port generates the Rust glue separately (`kernel-port glue --backend rust`).",
        "- Slices are passed as &[T] (read-only) or &mut [T] (written); no length params.",
    ),
}


def render_text(ext: Extraction, backend: str = "jni") -> str:
    """Human/agent-readable prompt material.

    ``backend`` selects the translation-notes block (v0.9 dogfood: the notes
    always said C++17/JNI even when the agent was targeting FFM or Rust).
    """
    m = ext.method
    notes = _BACKEND_NOTES.get(backend, _BACKEND_NOTES["jni"])
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
        *notes,
        "- Preserve floating-point operation order; do not introduce",
        "  reassociation unless the differential test still passes.",
    ]
    type_notes = _type_mapping_notes(m)
    if type_notes:
        lines += ["", "### Type mapping for the native signature"] + type_notes
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
