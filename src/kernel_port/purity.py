"""Conservative purity analysis for Java methods.

A method is *pure* when its body provably has no observable side effects and
its result depends only on its arguments:

- no I/O, network, or framework calls (only an allowlist of known-pure JDK
  calls such as ``Math.*`` / ``StrictMath.*``, never ``Math.random()``)
- no object allocation except array creation (``new int[n]`` is fine,
  ``new Random()`` is not)
- no reads/writes of mutable state: only ``static final`` constants may be
  read; any field write is impure
- no ``synchronized``, ``throw``, lambdas, method references,
  try-with-resources, or ``super`` calls
- calls to other repo methods are allowed only when the callee is itself
  pure (computed coinductively, so direct recursion is fine)

The checker is deliberately conservative: a *false negative* (rejecting a
pure method with a stated reason) is acceptable in v0.1; a *false positive*
(claiming impure code is pure) is a bug. Every rejection carries a concrete
reason string, e.g. ``"calls System.currentTimeMillis()"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .java_ast import (
    MethodInfo,
    children,
    child,
    named,
    text,
)


@dataclass
class PurityReport:
    pure: bool
    reasons: list[str] = field(default_factory=list)
    # qualnames of repo methods that were (transitively) proven pure
    pure_callees: list[str] = field(default_factory=list)


# Known-pure JDK static calls. Deliberately small: Math.random() is NOT pure
# (hidden global Random), and neither is anything on System.
_PURE_MATH = frozenset(
    "abs min max sqrt cbrt pow exp expm1 log log10 log1p "
    "sin cos tan asin acos atan atan2 sinh cosh tanh hypot "
    "floor ceil rint round signum copySign getExponent nextUp nextAfter ulp "
    "IEEEremainder scalb".split()
)
_PURE_RECEIVERS = {"Math": _PURE_MATH, "StrictMath": _PURE_MATH}


class _PurityWalk:
    """Walk one method body, collecting impurity reasons."""

    def __init__(self, method: MethodInfo, src: bytes, checker: "_PurityChecker"):
        self.method = method
        self.src = src
        self.checker = checker
        self.reasons: list[str] = []
        self.locals: set[str] = set()  # params + declared local names
        for p in method.params:
            self.locals.add(p.name)

    # -- helpers --------------------------------------------------------

    def _t(self, node) -> str:
        return text(node, self.src)

    def _impure(self, reason: str) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)

    def _field_of(self, name: str):
        cls = self.checker.repo.classes.get(self.method.class_qualname)
        if cls is None:
            return None
        return cls.fields.get(name)

    def _is_constant_field(self, name: str) -> bool:
        f = self._field_of(name)
        return f is not None and f.is_constant

    def _resolve_call(self, name: str, receiver) -> MethodInfo | None:
        """Resolve a call to a repo MethodInfo, or None when not resolvable."""
        recv_text = self._t(receiver) if receiver is not None else None
        if receiver is None or (receiver.type == "this"):
            cls_qn = self.method.class_qualname
        elif receiver.type == "identifier":
            # ClassName.staticCall(...) or localVar.call(...)
            simple = recv_text
            if simple in self.locals:
                return None  # call on a local object: not statically resolvable
            cls_qn = self.checker.class_by_simple_name(simple)
            if cls_qn is None:
                return None
        else:
            return None  # e.g. System.out.println: complex receiver
        cands = self.checker.repo.methods.get((cls_qn, name), [])
        if len(cands) == 1:
            return cands[0]
        if len(cands) > 1:
            # Overloads: conservatively require a unique match by arity is
            # overkill for v0.1; treat as unresolvable (false negative OK).
            return None
        return None

    def _check_invocation(self, node) -> None:
        kids = named(node)
        arg_list = child(node, "argument_list")
        name_node, receiver = None, None
        if arg_list is not None and arg_list in kids:
            idx = kids.index(arg_list)
            for j in range(idx - 1, -1, -1):
                if kids[j].type == "identifier":
                    name_node = kids[j]
                    receiver = kids[j - 1] if j > 0 else None
                    break
        if name_node is None:
            for c in kids:
                self.walk(c)
            return
        name = self._t(name_node)
        recv_text = self._t(receiver) if receiver is not None else None

        # 1. allowlisted JDK calls: Math.sqrt(x), StrictMath.pow(a, b)
        if receiver is not None and receiver.type == "identifier":
            allowed = _PURE_RECEIVERS.get(recv_text)
            if allowed is not None:
                if name in allowed:
                    for c in kids:
                        if c is not name_node and c is not receiver:
                            self.walk(c)
                    return
                self._impure(f"calls {recv_text}.{name}(...) (not a pure JDK call)")
                return

        # 2. repo-local calls (bare, this., or ClassName.)
        target = self._resolve_call(name, receiver)
        if target is not None:
            sub = self.checker.check(target)
            if not sub.pure:
                self._impure(
                    f"calls impure method '{target.qualname}': {sub.reasons[0]}"
                    if sub.reasons
                    else f"calls impure method '{target.qualname}'"
                )
            else:
                if target.qualname not in self.checker.pure_callees_seen:
                    self.checker.pure_callees_seen.append(target.qualname)
            for c in kids:
                if c is not name_node and c is not receiver:
                    self.walk(c)
            return

        # 3. anything else is conservatively impure
        dotted = f"{recv_text}.{name}" if recv_text else name
        self._impure(f"calls {dotted}(...) (not provably pure)")

    def _check_field_access(self, node, *, is_write: bool) -> None:
        kids = named(node)
        if not kids:
            return
        fname = self._t(kids[-1])
        receiver = kids[-2] if len(kids) >= 2 else None
        # array.length is pure
        if fname == "length" and receiver is not None:
            self.walk(receiver)
            return
        # Foo.class literal: pure
        if fname == "class":
            return
        if receiver is None or receiver.type == "this" or (
            receiver.type == "identifier" and self._t(receiver) not in self.locals
        ):
            # likely a field of this class (or ClassName.CONSTANT)
            if self._is_constant_field(fname):
                return
            kind = "writes to" if is_write else "reads"
            self._impure(f"{kind} mutable field '{fname}'")
            return
        # field of some other object: walk receiver, conservatively impure read
        self.walk(receiver)
        if not is_write:
            self._impure(f"reads field '{fname}' of an object")

    # -- main walk ------------------------------------------------------

    def walk(self, node) -> None:
        if node is None:
            return
        t = node.type

        if t == "method_invocation":
            self._check_invocation(node)
            return
        if t == "object_creation_expression":
            # `new X(...)` -- only array creation (a different node type) is OK
            tn = next(
                (c for c in named(node) if c.type not in ("argument_list",)),
                None,
            )
            tname = self._t(tn) if tn is not None else "?"
            self._impure(f"creates object of type '{tname}'")
            return
        if t == "explicit_constructor_invocation":
            self._impure("calls this(...)/super(...) constructor")
            return
        if t == "field_access":
            self._check_field_access(node, is_write=False)
            return
        if t == "assignment_expression":
            kids = named(node)
            if kids:
                lhs = kids[0]
                if lhs.type == "field_access":
                    self._check_field_access(lhs, is_write=True)
                elif lhs.type == "identifier":
                    lname = self._t(lhs)
                    if lname not in self.locals and self._field_of(lname) is not None:
                        if not self._is_constant_field(lname):
                            self._impure(f"writes to mutable field '{lname}'")
                        else:
                            self._impure(f"writes to constant field '{lname}'")
                    # else: local variable assignment -- fine
                else:
                    self.walk(lhs)
                for c in kids[1:]:
                    self.walk(c)
            return
        if t == "update_expression":
            # i++ / counter++ / --x
            operand = next((c for c in named(node)), None)
            if operand is not None:
                if operand.type == "field_access":
                    self._check_field_access(operand, is_write=True)
                elif operand.type == "identifier":
                    oname = self._t(operand)
                    if oname not in self.locals and self._field_of(oname) is not None:
                        self._impure(f"mutates field '{oname}'")
                else:
                    self.walk(operand)
            return
        if t == "synchronized_statement":
            self._impure("uses synchronized")
            return
        if t == "throw_statement":
            self._impure("throws an exception")
            return
        if t == "lambda_expression":
            self._impure("uses a lambda")
            return
        if t == "method_reference":
            self._impure("uses a method reference")
            return
        if t == "resource":
            # try-with-resources allocates a Closeable
            self._impure("uses try-with-resources")
            return
        if t == "local_variable_declaration":
            for decl in children(node, "variable_declarator"):
                nm = child(decl, "identifier")
                if nm is not None:
                    self.locals.add(self._t(nm))
                for c in named(decl):
                    if c.type != "identifier":
                        self.walk(c)
            return
        if t == "enhanced_for_statement":
            nm = child(node, "identifier")
            if nm is not None:
                self.locals.add(self._t(nm))
            for c in named(node):
                if c is not nm:
                    self.walk(c)
            return
        if t == "catch_clause":
            for c in named(node):
                if c.type == "catch_formal_parameter":
                    nm = child(c, "identifier")
                    if nm is not None:
                        self.locals.add(self._t(nm))
                else:
                    self.walk(c)
            return
        if t == "identifier":
            name = self._t(node)
            if name not in self.locals and self._field_of(name) is not None:
                if not self._is_constant_field(name):
                    self._impure(f"reads mutable field '{name}'")
            return
        for c in named(node):
            self.walk(c)


class _PurityChecker:
    """Memoized, coinductive purity checker over a JavaRepo."""

    def __init__(self, repo):
        self.repo = repo
        self._cache: dict[str, PurityReport] = {}
        self._in_progress: set[str] = set()
        self.pure_callees_seen: list[str] = []
        # simple class name -> qualname (first wins; ambiguity -> None)
        self._simple: dict[str, str | None] = {}
        for qn in repo.classes:
            simple = qn.rsplit(".", 1)[-1]
            if simple in self._simple:
                self._simple[simple] = None
            else:
                self._simple[simple] = qn

    def class_by_simple_name(self, simple: str) -> str | None:
        return self._simple.get(simple)

    def check(self, method: MethodInfo) -> PurityReport:
        if method.qualname in self._cache:
            return self._cache[method.qualname]
        if method.qualname in self._in_progress:
            # Recursive edge: coinductively assume pure; the outer frame
            # decides based on everything else.
            return PurityReport(pure=True)
        if method.body is None:
            rep = PurityReport(
                pure=False, reasons=["method has no body (abstract or native)"]
            )
            self._cache[method.qualname] = rep
            return rep
        self._in_progress.add(method.qualname)
        try:
            src = self._source_of(method)
            walk = _PurityWalk(method, src, self)
            walk.walk(method.body)
            rep = PurityReport(
                pure=not walk.reasons,
                reasons=walk.reasons,
                pure_callees=list(self.pure_callees_seen),
            )
        finally:
            self._in_progress.discard(method.qualname)
        self._cache[method.qualname] = rep
        return rep

    def _source_of(self, method: MethodInfo) -> bytes:
        path = self.repo.root / method.file
        raw = path.read_bytes()
        return raw.decode("utf-8", errors="replace").encode("utf-8")


def check_purity(method: MethodInfo, repo) -> PurityReport:
    """Public entry: purity report for one method."""
    return _PurityChecker(repo).check(method)


def check_purity_repo(repo) -> dict[str, PurityReport]:
    """Purity reports for every method in the repo (shared checker/cache)."""
    checker = _PurityChecker(repo)
    out: dict[str, PurityReport] = {}
    for (cls_qn, name), overloads in repo.methods.items():
        for m in overloads:
            out[m.qualname] = checker.check(m)
    return out
