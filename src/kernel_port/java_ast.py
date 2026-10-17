"""Tree-sitter based Java parsing, focused on method-level analysis.

Unlike a dead-code scanner (which tracks symbols repo-wide), kernel-port needs
per-method detail: signature, modifiers, parameter types, body AST, source text
and every call / field access / allocation inside the body. This module builds
a small :class:`JavaRepo` model: files -> classes -> methods, with helpers to
resolve method calls to their definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tree_sitter import Language, Parser

import tree_sitter_java

_PARSER: Parser | None = None


def parser() -> Parser:
    global _PARSER
    if _PARSER is None:
        _PARSER = Parser(Language(tree_sitter_java.language()))
    return _PARSER


def text(node, src: bytes) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def named(node) -> list:
    return [c for c in node.children if c.is_named]


def child(node, *types):
    for c in node.children:
        if c.is_named and c.type in types:
            return c
    return None


def children(node, *types) -> list:
    return [c for c in node.children if c.is_named and c.type in types]


_TYPE_NODES = frozenset(
    {
        "type_identifier",
        "scoped_type_identifier",
        "generic_type",
        "array_type",
        "primitive_type",  # not in grammar; kept for safety
        "integral_type",
        "floating_point_type",
        "boolean_type",
        "void_type",
    }
)

_PRIMITIVES = {
    "byte",
    "short",
    "int",
    "long",
    "float",
    "double",
    "char",
    "boolean",
    "void",
}


def type_text(type_node, src: bytes) -> str:
    """Normalized Java type text, e.g. 'double[]', 'int', 'String'."""
    t = type_node.type
    if t == "array_type":
        inner = next(
            (c for c in named(type_node) if c.type != "dimensions"), None
        )
        base = type_text(inner, src) if inner is not None else "?"
        dims = sum(
            1 for c in type_node.children if c.type == "dimensions"
        ) or text(type_node, src).count("[")
        return base + "[]" * dims
    if t in ("integral_type", "floating_point_type", "boolean_type", "void_type"):
        return text(type_node, src).strip()
    if t == "generic_type":
        # e.g. List<String> -> keep raw text; unsupported for JNI anyway
        return text(type_node, src).replace(" ", "")
    return text(type_node, src).replace(" ", "").strip()


@dataclass
class Param:
    name: str
    jtype: str


@dataclass
class MethodInfo:
    """One method (or constructor) declaration."""

    qualname: str  # com.example.Kernels.dot
    class_qualname: str  # com.example.Kernels
    name: str
    jtype: str  # return type text
    params: list[Param]
    modifiers: set[str]
    is_static: bool
    is_constructor: bool
    file: str  # repo-relative path
    lineno: int
    end_lineno: int
    source: str  # full declaration source text
    body: object = None  # tree-sitter node of the block (None if abstract/native)
    node: object = None  # tree-sitter node of the declaration

    @property
    def signature(self) -> str:
        ps = ", ".join(f"{p.jtype} {p.name}" for p in self.params)
        return f"{self.jtype} {self.name}({ps})"


@dataclass
class FieldInfo:
    qualname: str
    name: str
    jtype: str
    modifiers: set[str]
    is_constant: bool  # static final
    file: str
    lineno: int
    source: str


@dataclass
class ClassInfo:
    qualname: str
    package: str
    name: str
    file: str
    methods: dict[str, MethodInfo] = field(default_factory=dict)  # name -> (first) method
    all_methods: list[MethodInfo] = field(default_factory=list)
    fields: dict[str, FieldInfo] = field(default_factory=dict)


@dataclass
class JavaRepo:
    root: Path
    files: list[str] = field(default_factory=list)
    classes: dict[str, ClassInfo] = field(default_factory=dict)  # qualname -> ClassInfo
    # (class_qualname, method_name) -> list[MethodInfo] (overloads)
    methods: dict[tuple[str, str], list[MethodInfo]] = field(default_factory=dict)

    def find_method(self, qualname: str) -> MethodInfo | None:
        """Resolve 'com.example.Kernels.dot' (unique overload required)."""
        parts = qualname.rsplit(".", 1)
        if len(parts) != 2:
            return None
        key = (parts[0], parts[1])
        overloads = self.methods.get(key, [])
        if len(overloads) == 1:
            return overloads[0]
        return None

    def overloads(self, qualname: str) -> list[MethodInfo]:
        parts = qualname.rsplit(".", 1)
        if len(parts) != 2:
            return []
        return self.methods.get((parts[0], parts[1]), [])


def _modifiers(node, src: bytes) -> set[str]:
    mods = child(node, "modifiers")
    if mods is None:
        return set()
    return {text(c, src) for c in mods.children if not c.is_named}


def _parse_method(node, src: bytes, class_qualname: str, rel: str) -> MethodInfo:
    name = text(child(node, "identifier"), src)
    mods = _modifiers(node, src)
    type_node = next((c for c in named(node) if c.type in _TYPE_NODES), None)
    jtype = type_text(type_node, src) if type_node is not None else "void"
    params: list[Param] = []
    fp = child(node, "formal_parameters")
    if fp is not None:
        for p in children(fp, "formal_parameter"):
            pname_node = child(p, "identifier")
            ptype_node = next(
                (c for c in named(p) if c.type in _TYPE_NODES), None
            )
            if pname_node is None or ptype_node is None:
                continue
            params.append(
                Param(
                    name=text(pname_node, src),
                    jtype=type_text(ptype_node, src),
                )
            )
    body = child(node, "block")
    qualname = f"{class_qualname}.{name}"
    return MethodInfo(
        qualname=qualname,
        class_qualname=class_qualname,
        name=name,
        jtype=jtype,
        params=params,
        modifiers=mods,
        is_static="static" in mods,
        is_constructor=False,
        file=rel,
        lineno=node.start_point[0] + 1,
        end_lineno=node.end_point[0] + 1,
        source=text(node, src),
        body=body,
        node=node,
    )


def _parse_field(node, src: bytes, class_qualname: str, rel: str) -> list[FieldInfo]:
    mods = _modifiers(node, src)
    type_node = next((c for c in named(node) if c.type in _TYPE_NODES), None)
    jtype = type_text(type_node, src) if type_node is not None else "?"
    out = []
    for decl in children(node, "variable_declarator"):
        name_node = child(decl, "identifier")
        if name_node is None:
            continue
        name = text(name_node, src)
        out.append(
            FieldInfo(
                qualname=f"{class_qualname}.{name}",
                name=name,
                jtype=jtype,
                modifiers=mods,
                is_constant=("static" in mods and "final" in mods),
                file=rel,
                lineno=node.start_point[0] + 1,
                source=text(node, src),
            )
        )
    return out


def _parse_class(node, src: bytes, package: str, rel: str, repo: "JavaRepo") -> None:
    name_node = child(node, "identifier")
    if name_node is None:
        return
    name = text(name_node, src)
    qualname = f"{package}.{name}" if package else name
    cls = ClassInfo(qualname=qualname, package=package, name=name, file=rel)
    repo.classes[qualname] = cls
    body = child(
        node,
        "class_body",
        "interface_body",
        "enum_body",
        "annotation_type_body",
    )
    if body is None:
        return
    for member in named(body):
        t = member.type
        if t == "method_declaration":
            m = _parse_method(member, src, qualname, rel)
            cls.all_methods.append(m)
            cls.methods.setdefault(m.name, m)
            repo.methods.setdefault((qualname, m.name), []).append(m)
        elif t == "field_declaration":
            for f in _parse_field(member, src, qualname, rel):
                cls.fields[f.name] = f
        elif t in (
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
        ):
            # nested classes: recurse with dotted qualname
            _parse_class(member, src, qualname, rel, repo)


def scan_repo(root: Path | str, subdir: str | None = None) -> JavaRepo:
    """Parse every .java file under root (or root/subdir)."""
    root = Path(root)
    base = root / subdir if subdir else root
    repo = JavaRepo(root=root)
    if not base.exists():
        raise FileNotFoundError(f"directory not found: {base}")
    java_files = sorted(base.rglob("*.java"))
    for path in java_files:
        rel = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        src = raw.decode("utf-8", errors="replace").encode("utf-8")
        tree = parser().parse(src)
        root_node = tree.root_node
        package = ""
        for top in named(root_node):
            if top.type == "package_declaration":
                p = child(top, "scoped_identifier", "identifier")
                if p is not None:
                    package = text(p, src).replace(" ", "")
        repo.files.append(rel)
        for top in named(root_node):
            if top.type in (
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
                "record_declaration",
                "annotation_type_declaration",
            ):
                _parse_class(top, src, package, rel, repo)
    return repo
