"""Conservative purity analysis for Java methods.

A method is *pure* when its body provably has no observable side effects and
its result depends only on its arguments:

- no I/O, network, or framework calls (only an allowlist of known-pure JDK
  calls such as ``Math.*`` / ``StrictMath.*``, plus allowlisted pure ``String``
  instance methods like ``length()`` / ``charAt()`` / ``substring()`` on
  String-typed values, never ``Math.random()``)
- no object allocation except array creation (``new int[n]`` is fine,
  ``new Random()`` is not)
- no reads/writes of mutable state: only ``static final`` constants may be
  read; any field write is impure
- no ``synchronized``, lambdas, method references, try-with-resources, or
  ``super`` calls
- ``throw`` is allowed only in the *deterministic-throw* form:
  ``throw new <UncheckedType>(args...)`` where ``<UncheckedType>`` is one of
  ``IllegalArgumentException``, ``IllegalStateException``,
  ``ArithmeticException``, ``NullPointerException``,
  ``IndexOutOfBoundsException``, ``ArrayIndexOutOfBoundsException``,
  ``StringIndexOutOfBoundsException``, ``UnsupportedOperationException``,
  ``RuntimeException``, ``AssertionError`` (simple or qualified name), and
  every argument is a pure expression (constants, locals/params, pure
  arithmetic or string concatenation -- unknown method calls in the args
  are impure, as everywhere else). Still rejected: ``throw someField;``
  (a pre-built object may capture mutable state),
  ``throw new Exception(...)`` / ``throw new IOException(...)`` (checked
  exceptions are out of scope), rethrow of a caught variable
  (``catch (E e) { ... throw e; }`` -- unknown provenance), and any throw
  whose argument expression is impure.
- calls to other repo methods are allowed only when the callee is itself
  pure (computed coinductively, so direct recursion is fine)

The checker is deliberately conservative: a *false negative* (rejecting a
pure method with a stated reason) is acceptable in v0.1; a *false positive*
(claiming impure code is pure) is a bug. Every rejection carries a concrete
reason string, e.g. ``"calls System.currentTimeMillis()"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import java_ast as J
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

# Provably-pure, deterministic String instance methods. Instance calls on a
# String-typed value (param, local, or a chain like s.trim().toLowerCase())
# are pure, which lets real String kernels (hashing, parsing) be mined.
# Deliberately EXCLUDED:
# - getBytes: charset-dependent (uses the platform default charset)
# - split: regex semantics; kept conservative for now (regex compilation and
#   edge-case behavior differ across ports)
# - replaceAll / replaceFirst / matches: regex-based, same as split
# - intern: touches the global string pool (observable side effect)
# - formatted / format: locale-sensitive
# Note: toLowerCase/toUpperCase are allowlisted in their no-arg form, which
# uses the default locale -- deterministic for a given run, matching the
# same-locale differential testing the verify step performs.
_PURE_STRING_METHODS = frozenset(
    "length isEmpty isBlank charAt codePointAt codePointBefore codePointCount "
    "equals equalsIgnoreCase compareTo compareToIgnoreCase hashCode "
    "indexOf lastIndexOf startsWith endsWith contains "
    "substring subSequence toLowerCase toUpperCase trim strip stripLeading "
    "stripTrailing concat replace toCharArray toString".split()
)

# Unchecked exception types whose construction is deterministic and
# side-effect-free, so `throw new <T>(pure args...)` stays pure. Checked
# exceptions (Exception, IOException, ...) are deliberately excluded: they
# belong to a different portability story and are rejected with a reason.
_DETERMINISTIC_THROW_TYPES = frozenset(
    {
        "IllegalArgumentException",
        "IllegalStateException",
        "ArithmeticException",
        "NullPointerException",
        "IndexOutOfBoundsException",
        "ArrayIndexOutOfBoundsException",
        "StringIndexOutOfBoundsException",
        "UnsupportedOperationException",
        "RuntimeException",
        "AssertionError",
    }
)


class _PurityWalk:
    """Walk one method body, collecting impurity reasons."""

    def __init__(
        self,
        method: MethodInfo,
        src: bytes,
        checker: "_PurityChecker",
        final_fields: frozenset = frozenset(),
        in_constructor: bool = False,
    ):
        self.method = method
        self.src = src
        self.checker = checker
        # Class context for class-level portability checks: names of final
        # *instance* fields of the enclosing class. Reads of those fields are
        # pure (a final field never changes after construction); writes are
        # pure only inside a constructor. Empty by default, so plain
        # per-method checks behave exactly as before.
        self.final_fields = final_fields
        self.in_constructor = in_constructor
        self.reasons: list[str] = []
        self.locals: set[str] = set()  # params + declared local names
        # declared simple type of params/locals, for String-receiver checks.
        # Reassignment never changes the declared type, so this is only ever
        # added to, never updated or removed.
        self.local_types: dict[str, str] = {}
        for p in method.params:
            self.locals.add(p.name)
            if self._declared_string_type(p.jtype):
                self.local_types[p.name] = "String"
        # names bound by catch formal parameters, for rethrow detection
        self.caught_exceptions: set[str] = set()

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

    def _declared_string_type(self, jtype: str | None) -> bool:
        """True when a declared type text provably means java.lang.String.

        Bare ``String`` counts only when the repo does not define its own
        class named ``String`` (which would shadow java.lang.String); a
        fully-qualified ``java.lang.String`` always counts.
        """
        if not jtype:
            return False
        if jtype == "java.lang.String":
            return True
        return (
            jtype == "String"
            and self.checker.class_by_simple_name("String") is None
        )

    def _string_typed(self, node) -> bool:
        """True when the expression provably has type java.lang.String.

        Covers identifiers with a String declared type (params and locals)
        and chains of allowlisted String calls (``s.trim().toLowerCase()``:
        the result of an allowlisted call on a String-typed receiver is
        itself a String). Everything else is conservatively False.
        """
        t = node.type
        if t == "identifier":
            return self.local_types.get(self._t(node)) == "String"
        if t == "method_invocation":
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
            if (
                name_node is not None
                and self._t(name_node) in _PURE_STRING_METHODS
                and receiver is not None
            ):
                return self._string_typed(receiver)
            return False
        if t == "parenthesized_expression":
            inner = next((c for c in named(node)), None)
            return inner is not None and self._string_typed(inner)
        return False

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

        # 1b. allowlisted pure String instance methods: s.length(), s.charAt(i)
        #     on a String-typed receiver (params, locals, or a chain like
        #     s.trim().toLowerCase()). Argument expressions must be pure, as
        #     with Math.*.
        if (
            receiver is not None
            and name in _PURE_STRING_METHODS
            and self._string_typed(receiver)
        ):
            for c in kids:
                if c is not name_node:
                    self.walk(c)
            return

        # 1c. String-typed receiver but a non-allowlisted method (getBytes,
        #     split, intern, ...): conservatively impure, with a message that
        #     names the allowlist so users know why.
        if receiver is not None and self._string_typed(receiver):
            self._impure(
                f"calls {recv_text}.{name}(...) "
                f"(String method '{name}' not on the pure-method allowlist)"
            )
            return

        # 2. repo-local calls (bare, this., or ClassName.)
        target = self._resolve_call(name, receiver)
        if target is not None:
            sub = self.checker.check_callee(target, self)
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

    def _final_read_ok(self, receiver) -> bool:
        """True when a read of a final instance field provably targets this class.

        A bare ``x`` / ``this.x`` is always this class's field. ``other.x``
        is only accepted when ``other`` is a parameter whose declared type is
        this class (e.g. ``Point other``); anything else stays conservative.
        """
        if receiver is None:
            return True
        if receiver.type == "this":
            return True
        if receiver.type == "identifier":
            rtext = self._t(receiver)
            for p in self.method.params:
                if p.name == rtext:
                    own = self.method.class_qualname.rsplit(".", 1)[-1]
                    return p.jtype.rsplit(".", 1)[-1] == own
            return False
        return False

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
        # final instance fields of the class under analysis: reads are pure
        # (the field cannot change after construction); writes are pure only
        # inside a constructor. Falls through to the conservative path when
        # the receiver provably isn't this class.
        if fname in self.final_fields:
            if not is_write and self._final_read_ok(receiver):
                return
            if is_write and self.in_constructor and self._final_read_ok(receiver):
                return
            if is_write:
                self._impure(f"writes to final field '{fname}' outside a constructor")
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

    def _pure_expression_reasons(self, expr) -> list[str]:
        """Impurity reasons for one expression, via a scratch walk.

        Reuses the checker's existing pure-expression logic (literals,
        locals/params, pure arithmetic and string concatenation, allowlisted
        ``Math.*`` calls; unknown method calls are impure). The scratch walk
        inherits this walk's locals and final-field context, so anything the
        method itself may read is allowed here too. Conservative: any reason
        means the expression is impure.
        """
        sub = _PurityWalk(
            self.method,
            self.src,
            self.checker,
            final_fields=self.final_fields,
            in_constructor=self.in_constructor,
        )
        sub.locals = set(self.locals)
        sub.local_types = dict(self.local_types)
        sub.caught_exceptions = set(self.caught_exceptions)
        sub.walk(expr)
        return sub.reasons

    def _check_throw(self, node) -> None:
        """Decide whether a throw statement is purity-compatible.

        Accepted: ``throw new <UncheckedType>(pure args...)`` with the type
        on the allowlist. Everything else is rejected with a stated reason:
        throwing a pre-built object (may capture mutable state), a checked
        exception type, a rethrow of a caught variable (unknown provenance),
        or a throw whose argument expression is impure.
        """
        expr = next((c for c in named(node)), None)
        if expr is None:
            self._impure("throw with no expression")
            return
        if expr.type != "object_creation_expression":
            ename = self._t(expr)
            if expr.type == "identifier" and ename in self.caught_exceptions:
                self._impure(
                    f"rethrows caught exception '{ename}' (unknown provenance)"
                )
            elif expr.type == "identifier":
                self._impure(
                    f"throws a pre-built exception object '{ename}' "
                    "(may capture mutable state)"
                )
            else:
                self._impure(
                    f"throws a non-deterministic exception expression '{ename}'"
                )
            return
        # throw new Type(args...)
        tn = next(
            (c for c in named(expr) if c.type not in ("argument_list",)),
            None,
        )
        tname = self._t(tn) if tn is not None else "?"
        simple = tname.rsplit(".", 1)[-1]
        if simple not in _DETERMINISTIC_THROW_TYPES:
            self._impure(
                f"throws exception of type '{tname}' "
                "(only deterministic unchecked types are allowed)"
            )
            return
        arg_list = child(expr, "argument_list")
        if arg_list is not None:
            for arg in named(arg_list):
                bad = self._pure_expression_reasons(arg)
                if bad:
                    self._impure(
                        f"throw argument '{self._t(arg)}' is impure: {bad[0]}"
                    )

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
                        if lname in self.final_fields:
                            # bare `x = ...` targets this.x: pure only in a ctor
                            if not self.in_constructor:
                                self._impure(
                                    f"writes to final field '{lname}' outside a constructor"
                                )
                        elif not self._is_constant_field(lname):
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
                        if oname in self.final_fields and self.in_constructor:
                            pass  # `x++` on a final field inside a ctor: pure
                        elif oname in self.final_fields:
                            self._impure(
                                f"writes to final field '{oname}' outside a constructor"
                            )
                        else:
                            self._impure(f"mutates field '{oname}'")
                else:
                    self.walk(operand)
            return
        if t == "synchronized_statement":
            self._impure("uses synchronized")
            return
        if t == "throw_statement":
            self._check_throw(node)
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
            # First type-like named child is the declared type (a `modifiers`
            # node may precede it, e.g. `final String x`).
            tnode = next(
                (
                    c
                    for c in named(node)
                    if c.type in ("type_identifier", "scoped_type_identifier")
                    or c.type.endswith("_type")
                ),
                None,
            )
            ttext = self._t(tnode) if tnode is not None else ""
            for decl in children(node, "variable_declarator"):
                nm = child(decl, "identifier")
                if nm is not None:
                    lname = self._t(nm)
                    self.locals.add(lname)
                    if self._declared_string_type(ttext):
                        self.local_types[lname] = "String"
                for c in named(decl):
                    if c.type != "identifier":
                        self.walk(c)
            return
        if t == "enhanced_for_statement":
            nm = child(node, "identifier")
            if nm is not None:
                lname = self._t(nm)
                self.locals.add(lname)
                # Declared type is the named child right before the loop var.
                kn = named(node)
                ttext = ""
                if nm in kn:
                    idx = kn.index(nm)
                    if idx > 0:
                        ttext = self._t(kn[idx - 1])
                if self._declared_string_type(ttext):
                    self.local_types[lname] = "String"
            for c in named(node):
                if c is not nm:
                    self.walk(c)
            return
        if t == "catch_clause":
            for c in named(node):
                if c.type == "catch_formal_parameter":
                    nm = child(c, "identifier")
                    if nm is not None:
                        pname = self._t(nm)
                        self.locals.add(pname)
                        self.caught_exceptions.add(pname)
                else:
                    self.walk(c)
            return
        if t == "identifier":
            name = self._t(node)
            if name not in self.locals and self._field_of(name) is not None:
                if name in self.final_fields:
                    return  # bare read of a final instance field: pure
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
        # separate cache for class-context checks (final-field allowances);
        # keyed (qualname, in_constructor) so plain-check results are never
        # contaminated by class-context results or vice versa.
        self._class_cache: dict[tuple[str, bool], PurityReport] = {}
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
        return J.source_bytes(self.repo.root, method.file)

    def check_callee(self, target: MethodInfo, walk: "_PurityWalk") -> PurityReport:
        """Check a call target, inheriting the caller's class context.

        When the caller is being checked with a final-field context and the
        target is a method of the same class, the target is checked with the
        same context (its reads of final fields are pure). Otherwise this is
        exactly ``check(target)``.
        """
        if walk.final_fields and target.class_qualname == walk.method.class_qualname:
            return self.check_in_class(target, walk.final_fields, in_constructor=False)
        return self.check(target)

    def check_in_class(
        self,
        method: MethodInfo,
        final_fields: frozenset,
        in_constructor: bool = False,
    ) -> PurityReport:
        """Purity check with class context: final instance fields are readable
        (and writable inside constructors). Used by class-level portability."""
        key = (method.qualname, in_constructor)
        if key in self._class_cache:
            return self._class_cache[key]
        if method.qualname in self._in_progress:
            # Recursive edge: coinductively assume pure, as in check().
            return PurityReport(pure=True)
        if method.body is None:
            rep = PurityReport(
                pure=False, reasons=["method has no body (abstract or native)"]
            )
            self._class_cache[key] = rep
            return rep
        self._in_progress.add(method.qualname)
        try:
            src = self._source_of(method)
            walk = _PurityWalk(
                method,
                src,
                self,
                final_fields=final_fields,
                in_constructor=in_constructor,
            )
            walk.walk(method.body)
            rep = PurityReport(
                pure=not walk.reasons,
                reasons=walk.reasons,
                pure_callees=list(self.pure_callees_seen),
            )
        finally:
            self._in_progress.discard(method.qualname)
        self._class_cache[key] = rep
        return rep


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


# ---------------------------------------------------------------------------
# Class-level portability (workstream C2)
# ---------------------------------------------------------------------------


@dataclass
class ClassPurityReport:
    portable: bool
    kind: str  # "stateless" | "immutable" | "rejected"
    reasons: list[str] = field(default_factory=list)
    method_reports: dict[str, PurityReport] = field(default_factory=dict)


def _instance_field_writes(
    cls: J.ClassInfo, method: MethodInfo, src: bytes
) -> list[str]:
    """Names of instance fields written by `method` (conservative).

    Covers ``this.x = ...``, bare ``x = ...`` (resolving to an instance
    field), and ``x++``/``x--``. Any ambiguous write counts as a write:
    a false negative (rejecting the class) is acceptable, a false positive
    is not.
    """
    if method.body is None:
        return []
    instance = {f.name for f in cls.fields.values() if "static" not in f.modifiers}
    if not instance:
        return []
    locals_: set[str] = {p.name for p in method.params}
    written: list[str] = []

    def note(name: str) -> None:
        if name in instance and name not in written:
            written.append(name)

    def declare(node) -> None:
        nm = child(node, "identifier")
        if nm is not None:
            locals_.add(text(nm, src))

    def visit(node) -> None:
        t = node.type
        if t == "local_variable_declaration":
            for decl in children(node, "variable_declarator"):
                declare(decl)
            for c in named(node):
                visit(c)
            return
        if t == "enhanced_for_statement":
            declare(node)
            for c in named(node):
                if c.type != "identifier":
                    visit(c)
            return
        if t == "catch_clause":
            for c in named(node):
                if c.type == "catch_formal_parameter":
                    declare(c)
                else:
                    visit(c)
            return
        if t == "assignment_expression":
            kids = named(node)
            if kids:
                lhs = kids[0]
                if lhs.type == "field_access":
                    ks = named(lhs)
                    if ks:
                        note(text(ks[-1], src))
                elif lhs.type == "identifier":
                    nm = text(lhs, src)
                    if nm not in locals_:
                        note(nm)
                else:
                    visit(lhs)
                for c in kids[1:]:
                    visit(c)
            return
        if t == "update_expression":
            operand = next((c for c in named(node)), None)
            if operand is not None:
                if operand.type == "field_access":
                    ks = named(operand)
                    if ks:
                        note(text(ks[-1], src))
                elif operand.type == "identifier":
                    nm = text(operand, src)
                    if nm not in locals_:
                        note(nm)
                else:
                    visit(operand)
            return
        for c in named(node):
            visit(c)

    visit(method.body)
    return written


def check_class_portable(cls: J.ClassInfo, repo) -> ClassPurityReport:
    """Decide whether a whole class is portable as a unit.

    - *stateless*: no instance fields, no explicit constructors, every method
      static (``static final`` constants allowed), every method pure.
    - *immutable*: every instance field is ``final``, no method writes an
      instance field outside a constructor, every method pure (constructors
      may assign final fields).
    - otherwise *rejected*, with concrete reasons.
    """
    method_reports: dict[str, PurityReport] = {}
    reasons: list[str] = []
    checker = _PurityChecker(repo)

    instance_fields = [f for f in cls.fields.values() if "static" not in f.modifiers]
    static_fields = [f for f in cls.fields.values() if "static" in f.modifiers]
    final_instance = frozenset(
        f.name for f in instance_fields if "final" in f.modifiers
    )
    nonfinal_instance = [f.name for f in instance_fields if "final" not in f.modifiers]
    mutable_statics = [f.name for f in static_fields if "final" not in f.modifiers]
    simple = cls.name

    # 1. every method (constructors included) must be pure, with class context:
    #    reads of final instance fields are pure; constructors may assign them.
    for m in cls.all_methods:
        rep = checker.check_in_class(m, final_instance, in_constructor=m.is_constructor)
        method_reports[m.qualname] = rep
        if not rep.pure:
            first = rep.reasons[0] if rep.reasons else "unknown reason"
            reasons.append(f"method '{m.qualname}' is impure: {first}")

    # 2. structural rules
    for fname in mutable_statics:
        reasons.append(f"mutable static field '{fname}'")

    constructors = [m for m in cls.all_methods if m.is_constructor]
    non_ctor = [m for m in cls.all_methods if not m.is_constructor]

    kind = "rejected"
    if not instance_fields:
        if constructors:
            reasons.append(
                f"has {len(constructors)} explicit constructor(s) but no instance state"
            )
        elif all(m.is_static for m in non_ctor):
            kind = "stateless"
        else:
            inst = sorted(m.name for m in non_ctor if not m.is_static)
            reasons.append(f"has instance methods {inst} but no instance fields")
    else:
        for fname in nonfinal_instance:
            reasons.append(f"instance field '{fname}' is not final")
        writes_found = False
        for m in non_ctor:
            src = checker._source_of(m)
            for fname in _instance_field_writes(cls, m, src):
                writes_found = True
                reasons.append(
                    f"mutable field '{fname}' written in '{simple}.{m.name}()'"
                )
        if not nonfinal_instance and not writes_found:
            kind = "immutable"

    portable = kind in ("stateless", "immutable") and not reasons
    if not portable:
        kind = "rejected"
    return ClassPurityReport(
        portable=portable, kind=kind, reasons=reasons, method_reports=method_reports
    )
