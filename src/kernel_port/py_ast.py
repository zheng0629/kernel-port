"""Tree-sitter based Python parsing, focused on function-level analysis.

Mirrors ``java_ast.py``: builds a small :class:`PyRepo` model -- files ->
top-level functions (plus class methods, recorded for later slices) -- with
signature (arg names, defaults, annotations), decorators, body AST, source
text and every call / name access inside the body.

Slice 1 (Python kernel mining) only considers *module-level* functions:
nested functions (closures) and class methods are parsed and recorded but
never mined -- see :mod:`kernel_port.py_mine` for the rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tree_sitter import Language, Parser

import tree_sitter_python

_PARSER: Parser | None = None


def parser() -> Parser:
    global _PARSER
    if _PARSER is None:
        _PARSER = Parser(Language(tree_sitter_python.language()))
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


def iter_nodes(node):
    """Yield `node` and every named descendant, depth-first."""
    yield node
    for c in node.children:
        if c.is_named:
            yield from iter_nodes(c)


@dataclass
class PyParam:
    name: str
    annotation: str | None = None
    default: str | None = None
    kind: str = "pos"  # "pos" | "vararg" (*args) | "kwarg" (**kwargs)


@dataclass
class PyFunction:
    """One function definition (module-level, nested, or method)."""

    qualname: str  # "numeric.dot" or "numeric.Outer.inner" (nested)
    module: str  # dotted module path, e.g. "numeric" or "pkg.kernels"
    name: str
    file: str  # repo-relative path
    lineno: int
    end_lineno: int
    source: str  # full definition source text
    params: list[PyParam] = field(default_factory=list)
    return_annotation: str | None = None
    decorators: list[str] = field(default_factory=list)
    nested: bool = False  # defined inside another function body
    is_method: bool = False
    class_qualname: str | None = None
    body: object = None  # tree-sitter node of the block
    node: object = None  # tree-sitter node of the (decorated_)definition

    @property
    def signature(self) -> str:
        parts = []
        for p in self.params:
            s = p.name
            if p.kind == "vararg":
                s = "*" + s
            elif p.kind == "kwarg":
                s = "**" + s
            if p.annotation:
                s += f": {p.annotation}"
            if p.default:
                s += f" = {p.default}"
            parts.append(s)
        sig = f"{self.name}({', '.join(parts)})"
        if self.return_annotation:
            sig += f" -> {self.return_annotation}"
        return sig


@dataclass
class PyClass:
    qualname: str
    module: str
    name: str
    file: str
    methods: dict[str, PyFunction] = field(default_factory=dict)


@dataclass
class PyFile:
    """Per-file model: imports, module-level names, functions."""

    path: str  # repo-relative path
    module: str  # dotted module path
    # alias -> dotted module path, e.g. {"np": "numpy", "sqrt": "math.sqrt"}
    imports: dict[str, str] = field(default_factory=dict)
    # module-level assigned names (RHS not necessarily literal)
    globals_assigned: set[str] = field(default_factory=set)
    # ALL_CAPS module-level names bound to a literal: readable as constants
    module_consts: set[str] = field(default_factory=set)
    functions: list[PyFunction] = field(default_factory=list)


@dataclass
class PyRepo:
    root: Path
    files: list[str] = field(default_factory=list)
    modules: dict[str, PyFile] = field(default_factory=dict)  # module -> PyFile
    functions: dict[str, PyFunction] = field(default_factory=dict)  # qualname
    classes: dict[str, PyClass] = field(default_factory=dict)  # qualname

    def top_level_functions(self) -> list[PyFunction]:
        """Module-level functions only: the mining population for slice 1."""
        return sorted(
            (
                f
                for f in self.functions.values()
                if not f.nested and not f.is_method
            ),
            key=lambda f: f.qualname,
        )


# Only *immutable* literals count as module constants: a module-level list
# (even of literals, even empty) is mutable state that other code may change,
# so reading it is not pure.
_LITERAL_TYPES = frozenset(
    {
        "integer",
        "float",
        "string",
        "true",
        "false",
        "none",
        "unary_operator",  # -1, +2.5
        "tuple",  # (1, 2) of literals -- checked shallowly below
    }
)


def _is_literal(node) -> bool:
    if node.type not in _LITERAL_TYPES:
        return False
    if node.type == "unary_operator":
        kids = named(node)
        return len(kids) == 2 and _is_literal(kids[1])
    if node.type == "tuple":
        return all(_is_literal(c) for c in named(node))
    return True


def _param_name(node, src: bytes) -> str:
    """True parameter name, unwrapping *args/**kwargs splat patterns.

    v0.10: ``*rest: int`` parses as ``typed_parameter`` whose first named
    child is a ``list_splat_pattern`` with text ``"*rest"``; taking that
    text verbatim produced a bogus "unknown name '*rest'" purity failure
    that masked the real issue (v0.9 dogfood FN #7).
    """
    if node.type in ("list_splat_pattern", "dictionary_splat_pattern"):
        ident = next((c for c in named(node) if c.type == "identifier"), None)
        return text(ident, src) if ident is not None else "?"
    if node.type == "identifier":
        return text(node, src)
    return "?"


def _parse_param(node, src: bytes) -> PyParam | None:
    t = node.type
    if t == "identifier":
        return PyParam(name=text(node, src))
    raw = text(node, src)
    kind = "pos"
    if raw.startswith("**"):
        kind = "kwarg"
    elif raw.startswith("*"):
        kind = "vararg"
    if t == "typed_parameter":
        kids = named(node)
        name = _param_name(kids[0], src) if kids else "?"
        ann = text(kids[1], src) if len(kids) > 1 else None
        return PyParam(name=name, annotation=ann, kind=kind)
    if t == "default_parameter":
        kids = named(node)
        name = _param_name(kids[0], src) if kids else "?"
        default = text(kids[-1], src) if len(kids) > 1 else None
        return PyParam(name=name, default=default, kind=kind)
    if t == "typed_default_parameter":
        kids = named(node)
        name = _param_name(kids[0], src) if kids else "?"
        ann = text(kids[1], src) if len(kids) > 1 else None
        default = text(kids[-1], src) if len(kids) > 2 else None
        return PyParam(name=name, annotation=ann, default=default, kind=kind)
    if kind != "pos":
        # list_splat_parameter / dictionary_splat_parameter (or the
        # *_pattern variants the grammar reuses): name is the identifier child
        ident = next((c for c in named(node) if c.type == "identifier"), None)
        name = text(ident, src) if ident is not None else "?"
        return PyParam(name=name, kind=kind)
    return None


def _parse_function(
    def_node, src: bytes, module: str, rel: str, *, nested: bool = False,
    is_method: bool = False, class_qualname: str | None = None,
    outer: str | None = None,
) -> PyFunction:
    name_node = child(def_node, "identifier")
    name = text(name_node, src) if name_node is not None else "?"
    params: list[PyParam] = []
    pnode = child(def_node, "parameters")
    if pnode is not None:
        for p in named(pnode):
            if p.type == "keyword_separator":
                continue
            pp = _parse_param(p, src)
            if pp is not None:
                params.append(pp)
    ret_node = child(def_node, "type")
    ret_ann = text(ret_node, src).strip() if ret_node is not None else None
    body = child(def_node, "block")
    if outer:
        qualname = f"{outer}.{name}"
    elif class_qualname:
        qualname = f"{class_qualname}.{name}"
    else:
        qualname = f"{module}.{name}"
    return PyFunction(
        qualname=qualname,
        module=module,
        name=name,
        file=rel,
        lineno=def_node.start_point[0] + 1,
        end_lineno=def_node.end_point[0] + 1,
        source=text(def_node, src),
        params=params,
        return_annotation=ret_ann,
        nested=nested,
        is_method=is_method,
        class_qualname=class_qualname,
        body=body,
        node=def_node,
    )


def _decorator_texts(decorated_node, src: bytes) -> list[str]:
    out = []
    for dec in children(decorated_node, "decorator"):
        # decorator node: "@" then the expression; keep the expression text
        kids = [c for c in dec.children if c.is_named]
        out.append(text(kids[0], src) if kids else "?")
    return out


def _relative_base(module: str, level: int) -> str:
    """Containing package of `module`, walked up ``level - 1`` parents.

    ``from .mod import f`` in module ``"pkg.user"`` (level 1) resolves
    against ``"pkg"``; ``from ..mod import f`` (level 2) against ``""``.
    """
    base = module.rpartition(".")[0]
    for _ in range(level - 1):
        base = base.rpartition(".")[0]
    return base


def _import_aliases(node, src: bytes, module: str = "") -> dict[str, str]:
    """alias -> dotted path for one import statement.

    Relative from-imports (``from .mod import func``) are resolved to
    their absolute dotted path against the importing `module`. Without
    this, the recorded path (e.g. ``"func.func"``) could misresolve to
    an unrelated same-named top-level module during intra-repo call
    resolution in :mod:`kernel_port.py_mine` -- a soundness hole.
    """
    out: dict[str, str] = {}
    if node.type == "import_statement":
        for c in named(node):
            if c.type == "dotted_name":
                dotted = text(c, src).replace(" ", "")
                out[dotted.split(".")[0]] = dotted
            elif c.type == "aliased_import":
                kids = named(c)
                dotted = text(kids[0], src).replace(" ", "") if kids else "?"
                alias = text(kids[-1], src) if len(kids) > 1 else dotted.split(".")[0]
                out[alias] = dotted
    elif node.type == "import_from_statement":
        rel = next(
            (c for c in node.children if c.type == "relative_import"), None
        )
        mod = ""
        if rel is not None:
            # relative_import text is e.g. ".mod", ".", "..pkg": leading
            # dots are the level, the rest (if any) is the module part.
            # (The module name lives *inside* the relative_import node in
            # this grammar; there is no separate dotted_name for it.)
            rel_text = text(rel, src)
            level = len(rel_text) - len(rel_text.lstrip("."))
            modpart = rel_text[level:]
            base = _relative_base(module, level)
            mod = f"{base}.{modpart}" if modpart else base
        else:
            for c in node.children:
                if c.type == "dotted_name":
                    mod = text(c, src).replace(" ", "")
                    break
        # names after the "import" keyword
        seen_import_kw = False
        for c in node.children:
            if c.type == "import":
                seen_import_kw = True
                continue
            if not seen_import_kw or not c.is_named:
                continue
            if c.type == "dotted_name":
                out[text(c, src).replace(" ", "")] = (
                    f"{mod}.{text(c, src).replace(' ', '')}" if mod else text(c, src)
                )
            elif c.type == "aliased_import":
                kids = named(c)
                orig = text(kids[0], src).replace(" ", "") if kids else "?"
                alias = text(kids[-1], src) if len(kids) > 1 else orig
                out[alias] = f"{mod}.{orig}" if mod else orig
            elif c.type == "wildcard_import":
                pass  # `from x import *`: unresolvable; handled conservatively
    return out


def _module_level_assignment(node, src: bytes) -> tuple[str | None, object | None]:
    """(assigned bare name, rhs node) for a top-level assignment statement."""
    # expression_statement -> assignment, possibly with an annotated ":" child
    for c in named(node):
        if c.type == "assignment":
            kids = named(c)
            if kids and kids[0].type == "identifier":
                rhs = kids[-1] if len(kids) > 1 else None
                return text(kids[0], src), rhs
    return None, None


def scan_repo(root: Path | str, subdir: str | None = None) -> PyRepo:
    """Parse every .py file under root (or root/subdir)."""
    root = Path(root)
    base = root / subdir if subdir else root
    repo = PyRepo(root=root)
    if not base.exists():
        raise FileNotFoundError(f"directory not found: {base}")
    for path in sorted(base.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        src = raw.decode("utf-8", errors="replace").encode("utf-8")
        module = path.relative_to(root).with_suffix("").as_posix().replace("/", ".")
        tree = parser().parse(src)
        pyfile = PyFile(path=rel, module=module)
        repo.files.append(rel)
        repo.modules[module] = pyfile

        def register(fn: PyFunction) -> None:
            repo.functions[fn.qualname] = fn
            pyfile.functions.append(fn)

        def scan_nested(node, enclosing_qn: str) -> None:
            # nested defs (at any depth): recorded with nested=True, correct
            # outer qualname, but never mined -- see py_mine.
            for st in named(node):
                if st.type == "function_definition":
                    fn = _parse_function(
                        st, src, module, rel, nested=True, outer=enclosing_qn
                    )
                    register(fn)
                    if fn.body is not None:
                        scan_nested(fn.body, fn.qualname)
                elif st.type == "decorated_definition":
                    inner = child(st, "function_definition")
                    if inner is not None:
                        fn = _parse_function(
                            inner, src, module, rel, nested=True,
                            outer=enclosing_qn,
                        )
                        fn.decorators = _decorator_texts(st, src)
                        register(fn)
                        if fn.body is not None:
                            scan_nested(fn.body, fn.qualname)
                    else:
                        scan_nested(st, enclosing_qn)
                else:
                    scan_nested(st, enclosing_qn)

        for top in named(tree.root_node):
            t = top.type
            if t in ("import_statement", "import_from_statement"):
                pyfile.imports.update(_import_aliases(top, src, module))
            elif t == "expression_statement":
                name, rhs = _module_level_assignment(top, src)
                if name:
                    pyfile.globals_assigned.add(name)
                    if (
                        name.isupper()
                        and rhs is not None
                        and _is_literal(rhs)
                    ):
                        pyfile.module_consts.add(name)
            elif t == "function_definition":
                fn = _parse_function(top, src, module, rel)
                register(fn)
                if fn.body is not None:
                    scan_nested(fn.body, fn.qualname)
            elif t == "decorated_definition":
                inner = child(top, "function_definition")
                if inner is not None:
                    fn = _parse_function(inner, src, module, rel)
                    fn.decorators = _decorator_texts(top, src)
                    register(fn)
                    if fn.body is not None:
                        scan_nested(fn.body, fn.qualname)
            elif t == "class_definition":
                cname_node = child(top, "identifier")
                if cname_node is None:
                    continue
                cname = text(cname_node, src)
                cqn = f"{module}.{cname}"
                cls = PyClass(
                    qualname=cqn, module=module, name=cname, file=rel
                )
                repo.classes[cqn] = cls
                cbody = child(top, "block")
                if cbody is not None:
                    for member in named(cbody):
                        mnode = member
                        decorators: list[str] = []
                        if member.type == "decorated_definition":
                            mnode = child(member, "function_definition")
                            if mnode is None:
                                continue
                            decorators = _decorator_texts(member, src)
                        if mnode.type == "function_definition":
                            fn = _parse_function(
                                mnode, src, module, rel,
                                is_method=True, class_qualname=cqn,
                            )
                            fn.decorators = decorators
                            cls.methods[fn.name] = fn
                            register(fn)
    return repo


# ---------------------------------------------------------------------------
# Body query helpers used by py_mine.
# ---------------------------------------------------------------------------


def call_func_text(call_node, src: bytes) -> str:
    """Full text of the callee expression, e.g. 'print', 'os.path.join'."""
    fn = next((c for c in named(call_node)), None)
    return text(fn, src) if fn is not None else "?"


def call_root_name(call_node, src: bytes) -> str | None:
    """First dotted component of the callee, e.g. 'os' for os.path.join()."""
    fn = next((c for c in named(call_node)), None)
    if fn is None:
        return None
    if fn.type == "identifier":
        return text(fn, src)
    if fn.type == "attribute":
        obj = next((c for c in named(fn)), None)
        return text(obj, src).split(".")[0] if obj is not None else None
    return None


def call_attr_name(call_node, src: bytes) -> str | None:
    """Last dotted component of an attribute callee, e.g. 'join'."""
    fn = next((c for c in named(call_node)), None)
    if fn is not None and fn.type == "attribute":
        kids = named(fn)
        return text(kids[-1], src) if kids else None
    return None
