"""`mine` for Python: find pure-compute function candidates.

For every *module-level* function in the repo:

1. run the conservative purity check (impure functions are listed with
   reasons, never silently dropped);
2. count how often each function is called across the repo (call frequency);
3. score compute intensity with a cheap AST heuristic: loops,
   comprehensions, arithmetic, subscripts, recursion and ``math`` calls;
4. rank by ``(1 + calls) * intensity``.

Only module-level functions are mined. Nested functions (closures) and
class methods are parsed by :mod:`kernel_port.py_ast` but never mined:
a closure captures its enclosing scope, which has no direct C equivalent,
and porting one would require closure conversion -- out of scope for the
Python->C epic. A function that merely *defines* a nested function is
rejected as impure (with a reason), so the choice is explicit, not silent.

Purity rules for Python (conservative; mirrors ``purity.py``'s philosophy:
a *false negative* -- rejecting a pure function with a stated reason -- is
acceptable; a *false positive* -- claiming impure code is pure -- is a bug):

- **decorators**: any decorator (``@app.route``, ``@lru_cache``, ...) makes
  the function impure -- decorators imply framework/runtime behavior the
  checker cannot see through.
- **I/O builtins**: ``print``, ``open``, ``input`` are impure.
- **dynamic features**: ``eval``, ``exec``, ``compile``, ``getattr``,
  ``setattr``, ``delattr``, ``globals``, ``locals``, ``vars``,
  ``__import__``, ``breakpoint``, ``exit``/``quit`` are impure.
- **modules**: only the ``math`` module is allowlisted (pure functions such
  as ``math.sqrt``/``math.sin`` and constants such as ``math.pi``). Any
  other imported module -- ``os``, ``sys``, ``socket``, ``requests``,
  ``random``, ``time``, ``pathlib``, ``subprocess``, ... -- makes any use
  impure (``random.random()`` is nondeterministic, ``time.time()`` reads
  the clock, ``os.*`` touches the OS).
- **control effects**: ``raise`` is impure (no deterministic-throw
  carve-out in slice 1); ``assert`` is allowed with its condition checked;
  ``yield``/``yield from`` (generators are lazy) and ``with`` (resource
  management) are impure; ``try``/``except`` is allowed (the handler type
  expression is not evaluated for purity).
- **state**: ``nonlocal`` is impure; assigning to a ``global``-declared
  name is impure; mutating a module-level name (``LOG.append(...)``,
  ``CONFIG[k] = v``) is impure. Reading a module-level name is impure
  *unless* it is an ``ALL_CAPS`` literal constant.
  Mutation of **parameters and locals** (``a[i] = v``, ``out.append(x)``)
  is *allowed*: this is the out-parameter convention, mirroring how the
  Java pipeline treats array arguments (the future Python harness will
  have to echo mutated arguments, exactly like the Java harness does).
- **definitions in the body**: nested ``def``/``class``, ``lambda`` and
  body-level ``import`` are impure.
- **calls**: calls to other repo functions are allowed only when the
  callee is itself pure (computed coinductively, so direct recursion is
  fine); a name imported via ``from pkg.mod import func`` is resolved
  against the scanned repo's own modules first (``pkg.mod`` -> the repo
  file) and purity-checked as a normal callee -- only genuinely foreign
  modules fall back to the "not a pure module" rejection. Ambiguous
  (multi-module same-name) or unresolvable callees are impure.
  Allowlisted pure builtins (``len``, ``range``, ``abs``, ``min``/``max``,
  ``sum``, ``sorted``, ``enumerate``, ``zip``, ..., single-argument
  ``type()``) and allowlisted ``math.*`` calls are pure. Method calls on
  local/parameter receivers are pure only for a small allowlist: pure
  accessors (``get``/``keys``/``join``/``split``/``strip``/...), the
  Unicode-defined ``str`` predicates (``isdigit``/``isalpha``/``isspace``/
  ``isdecimal``/``isnumeric``/``isalnum`` -- locale-independent in
  Python 3, total on ``str``, non-mutating) and the non-mutating set
  algebra (``intersection``/``union``/``difference``/
  ``symmetric_difference``/``issubset``/``issuperset``/``isdisjoint``).
  Mutating methods (``append``/``extend``/``sort``/...) on locals/params
  are allowed by the out-param convention -- *except* the set-only
  mutating methods ``add``/``discard``, which are allowed only on a
  provably locally-created set (sets have no FFI representation, so a
  mutated set parameter could never be echoed back to the caller);
  anything else is impure.
- **annotations are optional**: unannotated functions are analyzed
  exactly like annotated ones. ``*args``/``**kwargs`` are rejected with
  a clear "variadic ... not portable" reason (variadic signatures have
  no C equivalent); the parameter names themselves are parsed
  correctly, so the rejection never surfaces as a bogus "unknown name".

Slice boundary: this module only *mines and ranks*. cffi glue generation
and the Python differential-test harness are a later slice of the
Python->C epic and are deliberately not started here.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import py_ast as P
from .purity import PurityReport

# -- allowlists -------------------------------------------------------------

_PURE_MATH_FUNCS = frozenset(
    "acos acosh asin asinh atan atan2 atanh cbrt ceil comb copysign cos cosh "
    "degrees dist erf erfc exp expm1 fabs factorial floor fmod frexp fsum "
    "gamma gcd hypot isclose isfinite isinf isnan isqrt lcm ldexp lgamma log "
    "log10 log1p modf nextafter pow prod radians remainder sin sinh sqrt tan "
    "tanh trunc ulp".split()
)
# module first-component -> allowlisted attribute names
_PURE_MODULES = {"math": _PURE_MATH_FUNCS}

_PURE_BUILTINS = frozenset(
    "len range abs min max sum round pow divmod int float complex str bool "
    "bytes list tuple dict set frozenset enumerate zip reversed sorted any "
    "all isinstance issubclass type ord chr hex oct bin".split()
)
# NOTE: `type` is allowlisted as a *reference*, but *calls* go through an
# arity check in _check_bare_call: only single-argument type(x) (pure
# introspection) is allowed; the multi-argument form defines a new class
# (a definition site with arbitrary metaclass behavior) and is rejected.

_IMPURE_BUILTINS = {
    "print": "console I/O",
    "open": "file I/O",
    "input": "console I/O",
    "eval": "dynamic code execution",
    "exec": "dynamic code execution",
    "compile": "dynamic code execution",
    "getattr": "dynamic attribute access",
    "setattr": "dynamic attribute access",
    "delattr": "dynamic attribute access",
    "globals": "dynamic scope access",
    "locals": "dynamic scope access",
    "vars": "dynamic scope access",
    "__import__": "dynamic import",
    "breakpoint": "debugger interaction",
    "exit": "interpreter exit",
    "quit": "interpreter exit",
}

# method names that mutate the receiver; allowed only on locals/params
# (the out-param convention: the future harness echoes mutated arguments,
# mirroring the Java array model).
_MUTATING_METHODS = frozenset(
    "append extend insert remove pop clear update setdefault "
    "popitem sort reverse".split()
)
# Set-only mutating methods, handled strictly (see _check_attr_call).
# Unlike the methods above, add/discard prove the receiver is a set --
# and a set has no representation at the C FFI boundary, so a mutated
# set *parameter* could never be echoed back to the caller. They are
# therefore allowed ONLY on a provably locally-created set (bound exactly
# once in the body to `set(...)` / `{...}` / a set comprehension, never a
# parameter or global): the mutation is then unobservable outside the
# function. On anything else they are rejected with a stated reason.
# (`update` is deliberately NOT here: dict.update exists too, so the
# receiver type is ambiguous and the out-param convention still covers
# it -- a false negative with a stated reason beats a behavior change.)
_SET_LOCAL_ONLY_MUTATING = frozenset({"add", "discard"})

# method names provably pure on any receiver (local/param)
_PURE_METHODS = frozenset(
    "get keys values items copy index count join split rsplit strip lstrip "
    "rstrip upper lower title capitalize casefold startswith endswith find "
    "rfind replace zfill partition rpartition "
    # str predicates: in Python 3 these are defined by the Unicode standard
    # (unicodedata), NOT by the C locale -- unlike <ctype.h> they cannot
    # change behavior under setlocale(). Audited 2026-10-18: each is total
    # on str (never raises), non-mutating and deterministic. Siblings such
    # as isupper/islower/istitle share these properties but are left out
    # conservatively until needed.
    "isdigit isalpha isspace isdecimal isnumeric isalnum "
    # non-mutating set algebra: each returns a new set (or bool), never
    # modifies the receiver, and is deterministic.
    "intersection union difference symmetric_difference "
    "issubset issuperset isdisjoint".split()
)

# builtin type names that may appear as bare references (isinstance args,
# except types are skipped separately); mentioning them is pure.
_KNOWN_TYPE_NAMES = frozenset(
    "ArithmeticError AssertionError AttributeError BaseException BytesWarning "
    "DeprecationWarning Exception FloatingPointError FutureWarning GeneratorExit "
    "ImportError ImportWarning IndentationError IndexError KeyError KeyboardInterrupt "
    "LookupError MemoryError NameError NotImplementedError OSError OverflowError "
    "PendingDeprecationWarning RecursionError ResourceWarning RuntimeError "
    "RuntimeWarning StopAsyncIteration StopIteration SyntaxError SyntaxWarning "
    "SystemError SystemExit TabError TypeError UnboundLocalError "
    "UnicodeDecodeError UnicodeEncodeError UnicodeError UnicodeWarning "
    "UserWarning ValueError Warning ZeroDivisionError".split()
)

_PATTERN_TYPES = frozenset(
    {"tuple_pattern", "list_pattern", "pattern_list"}
)

# Builtin exception types whose construction is deterministic and
# side-effect-free, so `raise <T>(pure args...)` stays pure (v0.9 dogfood:
# 65% of Python impure rejections were defensive `raise`s -- the same
# carve-out Java got in v0.4). Custom exception classes are excluded: their
# __init__ may do anything. Bare `raise` (re-raise) and `raise` of a
# variable are still rejected: unknown provenance.
_DETERMINISTIC_RAISE_TYPES = frozenset(
    "ValueError TypeError KeyError IndexError AttributeError RuntimeError "
    "ArithmeticError ZeroDivisionError OverflowError LookupError "
    "NotImplementedError StopIteration AssertionError".split()
)


@dataclass
class PyCandidate:
    qualname: str
    file: str
    lineno: int
    signature: str
    pure: bool
    reasons: list[str]
    calls: int
    intensity: float
    score: float


class _PyPurityWalk:
    """Walk one function body, collecting impurity reasons."""

    def __init__(self, func: P.PyFunction, src: bytes, checker: "_PyPurityChecker"):
        self.func = func
        self.src = src
        self.checker = checker
        self.pyfile = checker.repo.modules[func.module]
        self.reasons: list[str] = []
        self.locals: set[str] = {p.name for p in func.params}
        self.globals_declared: set[str] = set()
        # Body-binding bookkeeping for the "provably locally-created set"
        # rule (_SET_LOCAL_ONLY_MUTATING): every name bound in the body is
        # counted, and single-identifier targets bound to a fresh set
        # construction are noted. A name qualifies only if bound exactly
        # once, to a set construction, and is neither a parameter nor a
        # declared global (any rebinding, aliasing or global voids it).
        self._bind_counts: dict[str, int] = {}
        self._set_ctor_names: set[str] = set()
        self._collect_locals(func.body)
        param_names = {p.name for p in func.params}
        self.local_sets: set[str] = {
            n
            for n in self._set_ctor_names
            if self._bind_counts.get(n, 0) == 1
            and n not in param_names
            and n not in self.globals_declared
        }
        # Variadic signatures have no C equivalent: reject with a clear
        # reason rather than a confusing "unknown name '*rest'" -- the
        # parameter names themselves parse correctly (py_ast._param_name),
        # so a *used* variadic still reads as a plain local.
        for p in func.params:
            if p.kind == "vararg":
                self._impure(
                    f"has variadic parameter '*{p.name}' "
                    "(variadic signatures are not portable to C)"
                )
            elif p.kind == "kwarg":
                self._impure(
                    f"has variadic parameter '**{p.name}' "
                    "(variadic signatures are not portable to C)"
                )
        # names of nested defs in this body: calls/references to them must
        # not resolve to same-named module-level functions.
        self.nested_defs: set[str] = set()
        if func.body is not None:
            for n in P.iter_nodes(func.body):
                if n.type == "function_definition":
                    nm = P.child(n, "identifier")
                    if nm is not None:
                        self.nested_defs.add(self._t(nm))

    # -- helpers ----------------------------------------------------------

    def _t(self, node) -> str:
        return P.text(node, self.src)

    def _impure(self, reason: str) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)

    def _bind_pattern(self, node, into: set[str]) -> None:
        if node.type == "identifier":
            name = self._t(node)
            into.add(name)
            # Count every body binding: a "provably locally-created set"
            # must be bound exactly once (any rebinding voids the proof).
            self._bind_counts[name] = self._bind_counts.get(name, 0) + 1
        elif node.type in _PATTERN_TYPES:
            for c in P.named(node):
                self._bind_pattern(c, into)

    def _is_set_construction(self, node) -> bool:
        """True if `node` provably builds a fresh set.

        Accepts ``set(...)`` (unless ``set`` is shadowed by an import, in
        which case the call means something else entirely -- and is
        checked impure on its own), a set display ``{...}`` and a set
        comprehension. A shadowed-by-local ``set()`` call is likewise
        impure on its own, so misclassification there is moot.
        """
        t = node.type
        if t in ("set", "set_comprehension"):
            return True
        if t == "call":
            kids = P.named(node)
            return (
                bool(kids)
                and kids[0].type == "identifier"
                and self._t(kids[0]) == "set"
                and self._module_of("set") is None
            )
        return False

    def _collect_locals(self, node) -> None:
        """Pre-pass: every name bound anywhere in the body (own scope only)."""
        if node is None:
            return
        t = node.type
        if t in ("function_definition", "class_definition", "lambda"):
            return  # separate scope (nested defs are impure anyway)
        if t in ("assignment", "augmented_assignment"):
            kids = P.named(node)
            if kids:
                self._bind_pattern(kids[0], self.locals)
                if (
                    t == "assignment"
                    and kids[0].type == "identifier"
                    and len(kids) > 1
                    and self._is_set_construction(kids[-1])
                ):
                    self._set_ctor_names.add(self._t(kids[0]))
        elif t == "for_statement":
            kids = P.named(node)
            if kids:
                self._bind_pattern(kids[0], self.locals)
        elif t == "for_in_clause":
            kids = P.named(node)
            if kids:
                self._bind_pattern(kids[0], self.locals)
        elif t == "named_expression":
            kids = P.named(node)
            if kids:
                self._bind_pattern(kids[0], self.locals)
        elif t == "as_pattern_target":
            for c in P.named(node):
                self._bind_pattern(c, self.locals)
        elif t == "global_statement":
            for c in P.children(node, "identifier"):
                self.globals_declared.add(self._t(c))
        for c in P.named(node):
            self._collect_locals(c)

    def _target_root(self, target):
        """Innermost base identifier of a subscript/attribute target chain."""
        n = target
        while n.type in ("subscript", "attribute"):
            kids = P.named(n)
            if not kids:
                return None
            n = kids[0]
        return n if n.type == "identifier" else None

    def _resolve_bare(self, name: str):
        """Resolve a bare called name to a repo function.

        Returns (PyFunction | None, match_count).
        """
        same = self.checker.repo.functions.get(f"{self.func.module}.{name}")
        if (
            same is not None
            and not same.nested
            and not same.is_method
        ):
            return same, 1
        cands = self.checker.by_simple.get(name, [])
        if len(cands) == 1:
            return cands[0], 1
        return None, len(cands)

    def _module_of(self, alias: str) -> str | None:
        """Dotted module path for an import alias, e.g. 'numpy' for 'np'."""
        return self.pyfile.imports.get(alias)

    def _attr_chain_root(self, node) -> str | None:
        """Leftmost identifier of an attribute chain, e.g. 'pkg' for `pkg.mod`."""
        cur = node
        while cur is not None and cur.type == "attribute":
            kids = P.named(cur)
            cur = kids[0] if kids else None
        if cur is not None and cur.type == "identifier":
            return self._t(cur)
        return None

    # -- calls ------------------------------------------------------------

    def _check_raise_carveout(self, node) -> bool:
        """True if a `raise` is purity-compatible (deterministic-throw carve-out).

        Accepted: ``raise <BuiltinExc>(pure args...)`` or ``raise <BuiltinExc>``
        where the exception type is in :data:`_DETERMINISTIC_RAISE_TYPES`.
        Constructor args (and any `from` cause) are walked for purity; an
        impure arg is still recorded by the walk. Rejected: bare ``raise``,
        ``raise`` of a variable/pre-built object, and custom exception types
        (unknown __init__ side effects) -- mirroring the Java v0.4 rule.
        """
        kids = P.named(node)
        if not kids:
            return False  # bare `raise`: re-raise, unknown provenance
        exc = kids[0]
        if exc.type == "call":
            fn_kids = P.named(exc)
            fn_node = fn_kids[0] if fn_kids else None
            if fn_node is None or fn_node.type != "identifier":
                return False
            if self._t(fn_node) not in _DETERMINISTIC_RAISE_TYPES:
                return False
        elif exc.type == "identifier":
            if self._t(exc) not in _DETERMINISTIC_RAISE_TYPES:
                return False
        else:
            return False
        # Walk the remaining children (constructor args, `from` cause) so any
        # impurity inside them is still recorded.
        for c in kids[1:]:
            self.walk(c)
        if exc.type == "call":
            arg_list = P.child(exc, "argument_list")
            if arg_list is not None:
                for a in P.named(arg_list):
                    self.walk(a)
        return True

    def _check_call(self, node) -> None:
        kids = P.named(node)
        fn_node = kids[0] if kids else None
        arg_list = P.child(node, "argument_list")
        args = P.named(arg_list) if arg_list is not None else []
        # tree-sitter-python puts a sole generator-expression argument directly
        # under `call` (no `argument_list` wrapper). It must still be
        # purity-checked: v0.9 dogfood found `open()`/`print()` inside a genexp
        # silently passing as pure. (Genexp loop variables are already bound in
        # self.locals by the _collect_locals pre-pass.)
        genexp = P.child(node, "generator_expression")
        if genexp is not None:
            self.walk(genexp)
        if fn_node is None:
            return
        if fn_node.type == "identifier":
            self._check_bare_call(self._t(fn_node), args)
        elif fn_node.type == "attribute":
            self._check_attr_call(fn_node, args)
        else:
            # f()(), d[k]() -- calling the result of an expression
            self.walk(fn_node)
            for a in args:
                self.walk(a)
            self._impure(
                f"calls the result of '{self._t(fn_node)}' (not provably pure)"
            )

    def _check_resolved_callee(self, target: P.PyFunction) -> None:
        """Purity-check a resolved repo callee.

        Shared by the bare-name path and the from-import intra-repo path:
        an impure callee is reported naming it; a pure one is recorded for
        the report's pure-callee list.
        """
        sub = self.checker.check_callee(target)
        if not sub.pure:
            first = sub.reasons[0] if sub.reasons else "impure"
            self._impure(f"calls impure function '{target.qualname}': {first}")
        elif target.qualname not in self.checker.pure_callees_seen:
            self.checker.pure_callees_seen.append(target.qualname)

    def _check_intra_repo_member(self, dotted: str, meth: str, full: str) -> bool:
        """Resolve `dotted.meth` as a top-level function of an intra-repo module.

        Shared by the two attribute-call receiver shapes in
        :meth:`_check_attr_call`: a plain import alias (``pm.func(...)`` after
        ``import pkg.mod as pm``) and a dotted module chain
        (``pkg.mod.func(...)`` after ``import pkg.mod``).

        Returns True when handled: a resolved pure callee is checked normally
        via :meth:`_check_resolved_callee` (an impure callee is rejected
        naming it), and an unresolvable member keeps an impure verdict with a
        clear reason (never guessed). Returns False when `dotted` is not one
        of the scanned repo's own modules, so the caller keeps its existing
        foreign-module treatment.
        """
        if dotted not in self.checker.repo.modules:
            return False
        target = self.checker.repo.functions.get(f"{dotted}.{meth}")
        if target is not None and not target.nested and not target.is_method:
            self._check_resolved_callee(target)
        else:
            self._impure(
                f"calls {full}(...) (module '{dotted}' is in the repo "
                f"but has no top-level function '{meth}')"
            )
        return True

    def _check_bare_call(self, name: str, args) -> None:
        for a in args:
            self.walk(a)
        if name in self.locals:
            self._impure(f"calls local variable '{name}(...)' (not provably pure)")
            return
        if name in _IMPURE_BUILTINS:
            self._impure(f"calls {name}(...) ({_IMPURE_BUILTINS[name]})")
            return
        dotted = self._module_of(name)
        if dotted is not None:
            # `from X.Y import name` records dotted == "X.Y.name" (!= name).
            # If X.Y is one of the scanned repo's own modules, resolve the
            # callee there and check it normally instead of misclassifying
            # the import as a foreign module (v0.9 dogfood FN #3). Anything
            # that does not resolve intra-repo keeps the conservative
            # foreign treatment below (impure, with a clear reason).
            if "." in dotted and dotted != name:
                mod_path, _, member = dotted.rpartition(".")
                if mod_path in self.checker.repo.modules:
                    target = self.checker.repo.functions.get(
                        f"{mod_path}.{member}"
                    )
                    if (
                        target is not None
                        and not target.nested
                        and not target.is_method
                    ):
                        self._check_resolved_callee(target)
                        return
                    self._impure(
                        f"calls '{dotted}(...)' (module '{mod_path}' is in "
                        f"the repo but has no top-level function '{member}')"
                    )
                    return
            first = dotted.split(".")[0]
            rest = dotted[len(first) + 1 :] if "." in dotted else ""
            if first in _PURE_MODULES and (not rest or rest in _PURE_MODULES[first]):
                return  # e.g. `from math import sqrt; sqrt(x)`
            self._impure(
                f"calls {dotted}(...) (module '{first}' is not a pure module)"
            )
            return
        if name == "type":
            # type(x): pure introspection. type(name, bases, dict) defines
            # a new class -- a definition site with arbitrary metaclass
            # behavior -- and is rejected.
            if len(args) == 1:
                return
            self._impure(
                f"calls type(...) with {len(args)} arguments "
                "(only the single-argument form is pure)"
            )
            return
        if name in _PURE_BUILTINS:
            return
        if name in self.nested_defs:
            self._impure(
                f"calls nested function '{name}(...)' (closures are not portable)"
            )
            return
        target, n = self._resolve_bare(name)
        if target is not None:
            self._check_resolved_callee(target)
            return
        if n > 1:
            self._impure(f"calls '{name}(...)' (ambiguous: {n} matches in repo)")
        elif name in self.pyfile.globals_assigned:
            self._impure(f"calls module-level '{name}(...)' (not provably pure)")
        else:
            self._impure(f"calls {name}(...) (not provably pure)")

    def _check_attr_call(self, attr_node, args) -> None:
        kids = P.named(attr_node)
        obj, meth = (kids[0], self._t(kids[-1])) if kids else (None, "?")
        full = self._t(attr_node)
        for a in args:
            self.walk(a)
        if obj is not None and obj.type == "identifier":
            root = self._t(obj)
            if root in self.locals:
                if meth in _SET_LOCAL_ONLY_MUTATING:
                    # add/discard prove a set receiver. On a parameter (or
                    # an alias of one) the mutation is caller-visible, and
                    # the C harness could never echo it back (no set type
                    # at the FFI boundary); only a provably locally-created
                    # set keeps the mutation unobservable outside the
                    # function.
                    if root in self.local_sets:
                        return
                    self._impure(
                        f"calls {root}.{meth}(...) (mutates a set that may be "
                        "caller-visible; only allowed on a locally-created set)"
                    )
                    return
                if meth in _MUTATING_METHODS:
                    return  # local/param mutation: allowed (out-param convention)
                if meth in _PURE_METHODS:
                    return
                self._impure(
                    f"calls {root}.{meth}(...) (method not provably pure)"
                )
                return
            dotted = self._module_of(root)
            if dotted is not None:
                # v0.11 workstream 3: `import pkg.mod [as pm]; pm.func(...)`
                # where pkg.mod is one of the scanned repo's own modules --
                # resolve the member as a top-level function of that module
                # and check it like a from-import callee (v0.10
                # _check_bare_call), so a pure callee keeps the caller pure.
                # Truly foreign modules keep the conservative treatment
                # below, unchanged.
                if self._check_intra_repo_member(dotted, meth, full):
                    return
                first = dotted.split(".")[0]
                if first in _PURE_MODULES and meth in _PURE_MODULES[first]:
                    return  # math.sqrt(x), math.sin(x), ...
                self._impure(
                    f"calls {full}(...) (module '{first}' is not a pure module)"
                )
                return
            if root in self.pyfile.globals_assigned or root in self.globals_declared:
                self._impure(f"calls {full}(...) (mutates/uses module-level '{root}')")
                return
            self._impure(f"calls {full}(...) (receiver '{root}' not provably pure)")
            return
        # computed receiver, e.g. f().x() or "sep".join(xs)
        # module-chain receiver, e.g. `pkg.mod.func(...)` after
        # `import pkg.mod`: the chain root is an import alias whose dotted
        # path exactly matches the receiver text, and that module is one of
        # the scanned repo's own modules -- same intra-repo resolution as the
        # plain-alias branch above. (The root-not-in-locals guard keeps a
        # shadowing local binding from being mistaken for the module.)
        if obj is not None and obj.type == "attribute":
            root = self._attr_chain_root(obj)
            dotted = self._module_of(root) if root is not None else None
            if (
                root is not None
                and root not in self.locals
                and dotted is not None
                and self._t(obj).replace(" ", "") == dotted
                and self._check_intra_repo_member(dotted, meth, full)
            ):
                return
        if obj is not None:
            self.walk(obj)
        if obj is not None and obj.type == "string" and meth in _PURE_METHODS:
            return  # " ".join(xs)
        self._impure(f"calls method '{meth}' on a computed object (not provably pure)")

    # -- names and attributes ----------------------------------------------

    def _check_name_read(self, node) -> None:
        name = self._t(node)
        if name in self.locals:
            return
        if name in ("True", "False", "None"):
            return
        if name in _PURE_BUILTINS or name in _IMPURE_BUILTINS or name in _KNOWN_TYPE_NAMES:
            return
        dotted = self._module_of(name)
        if dotted is not None:
            first = dotted.split(".")[0]
            if first in _PURE_MODULES:
                return
            self._impure(f"references module '{dotted}' (not a pure module)")
            return
        if name in self.pyfile.module_consts:
            return
        if name in self.nested_defs:
            self._impure(
                f"references nested function '{name}' (closures are not portable)"
            )
            return
        target, n = self._resolve_bare(name)
        if target is not None or n > 1:
            return  # bare function reference (not called): no effect
        if name in self.pyfile.globals_assigned or name in self.globals_declared:
            self._impure(f"reads module-level variable '{name}'")
            return
        self._impure(f"reads unknown name '{name}'")

    def _check_attr_read(self, node) -> None:
        kids = P.named(node)
        if not kids:
            return
        obj, meth = kids[0], self._t(kids[-1])
        full = self._t(node)
        if obj.type == "identifier":
            root = self._t(obj)
            dotted = self._module_of(root)
            if dotted is not None:
                first = dotted.split(".")[0]
                if first in _PURE_MODULES:
                    return  # math.pi
                self._impure(f"reads {full} (module '{first}' is not a pure module)")
                return
            if root in self.locals:
                self._impure(
                    f"reads attribute '{meth}' of '{root}' (object state not analyzable)"
                )
                return
            self._impure(f"reads attribute '{meth}' of unknown '{root}'")
            return
        self.walk(obj)
        self._impure(f"reads attribute '{meth}' of a computed value (not provably pure)")

    # -- assignments ---------------------------------------------------------

    def _check_target_write(self, target, kind: str) -> None:
        """Decide whether writing to `target` is purity-compatible.

        `kind` is "assign", "aug-assign" or "delete", used in messages.
        """
        t = target.type
        if t == "identifier":
            name = self._t(target)
            if name in self.globals_declared:
                self._impure(f"{kind} global '{name}'")
            elif (
                name not in self.locals
                and name in self.pyfile.globals_assigned
            ):
                self._impure(f"{kind} module-level variable '{name}'")
            # else: local/param write -- fine
            return
        if t in ("subscript", "attribute"):
            root = self._target_root(target)
            rname = self._t(root) if root is not None else "?"
            if root is not None and (
                rname in self.locals
                or rname in ("True", "False", "None")
            ):
                pass  # out-param / local mutation: allowed by convention
            elif root is not None and (
                rname in self.globals_declared
                or rname in self.pyfile.globals_assigned
            ):
                self._impure(f"{kind} module-level '{rname}'")
            else:
                self._impure(f"{kind} '{rname}[...]' (target not provably local)")
            # walk index/key/value sub-expressions, skipping the root name
            # itself (already classified above)
            for c in P.named(target):
                if c is not root:
                    self.walk(c)
            return
        if t in _PATTERN_TYPES:
            for c in P.named(target):
                self._check_target_write(c, kind)
            return
        # anything else (starred, ...): walk conservatively
        self.walk(target)

    def _check_assignment(self, node, kind: str = "assign") -> None:
        kids = P.named(node)
        if not kids:
            return
        self._check_target_write(kids[0], kind)
        for c in kids[1:]:
            self.walk(c)

    # -- main walk -----------------------------------------------------------

    def walk(self, node) -> None:
        if node is None:
            return
        t = node.type

        if t == "call":
            self._check_call(node)
            return
        if t in ("import_statement", "import_from_statement"):
            self._impure("imports inside function body")
            return
        if t == "global_statement":
            for c in P.children(node, "identifier"):
                self.globals_declared.add(self._t(c))
            return
        if t == "nonlocal_statement":
            self._impure("uses 'nonlocal' (closure state)")
            return
        if t == "function_definition":
            nm = P.child(node, "identifier")
            self._impure(
                f"defines nested function '{self._t(nm) if nm is not None else '?'}' "
                "(closures are not portable)"
            )
            return
        if t == "class_definition":
            self._impure("defines a class inside the function body")
            return
        if t == "lambda":
            self._impure("uses a lambda")
            return
        if t == "yield":
            self._impure("is a generator (uses 'yield')")
            return
        if t == "with_statement":
            self._impure("uses 'with' (resource management)")
            return
        if t == "raise_statement":
            if not self._check_raise_carveout(node):
                self._impure(f"raises an exception ({self._t(node).strip()})")
            return
        if t == "delete_statement":
            kids = P.named(node)
            for target in kids:
                # delete targets bind nothing; check the write itself
                if target.type == "identifier" and self._t(target) in self.locals:
                    continue
                self._check_target_write(target, "deletes")
            return
        if t == "assignment":
            self._check_assignment(node, "assign")
            return
        if t == "augmented_assignment":
            self._check_assignment(node, "aug-assign")
            return
        if t == "assert_statement":
            for c in P.named(node):
                self.walk(c)
            return
        if t == "keyword_argument":
            # f(key=value): the keyword name is not a variable read
            for c in P.named(node)[1:]:
                self.walk(c)
            return
        if t == "except_clause":
            # the handler type expression is a static reference; only the
            # handler body can have effects.
            block = P.child(node, "block")
            self.walk(block)
            return
        if t == "identifier":
            self._check_name_read(node)
            return
        if t == "attribute":
            self._check_attr_read(node)
            return
        for c in P.named(node):
            self.walk(c)


class _PyPurityChecker:
    """Memoized, coinductive purity checker over a PyRepo."""

    def __init__(self, repo: P.PyRepo):
        self.repo = repo
        self._cache: dict[str, PurityReport] = {}
        self._in_progress: set[str] = set()
        self.pure_callees_seen: list[str] = []
        self.by_simple: dict[str, list[P.PyFunction]] = {}
        for f in repo.functions.values():
            if not f.nested and not f.is_method:
                self.by_simple.setdefault(f.name, []).append(f)

    def check(self, func: P.PyFunction) -> PurityReport:
        if func.qualname in self._cache:
            return self._cache[func.qualname]
        if func.qualname in self._in_progress:
            # Recursive edge: coinductively assume pure; the outer frame
            # decides based on everything else.
            return PurityReport(pure=True)
        if func.decorators:
            decs = ", ".join(f"'@{d}'" for d in func.decorators)
            rep = PurityReport(
                pure=False,
                reasons=[
                    f"has decorator {decs} "
                    "(decorators imply framework behavior; not analyzable)"
                ],
            )
            self._cache[func.qualname] = rep
            return rep
        if func.body is None:
            rep = PurityReport(pure=False, reasons=["function has no body"])
            self._cache[func.qualname] = rep
            return rep
        self._in_progress.add(func.qualname)
        try:
            src = self._source_of(func)
            walk = _PyPurityWalk(func, src, self)
            walk.walk(func.body)
            rep = PurityReport(
                pure=not walk.reasons,
                reasons=walk.reasons,
                pure_callees=list(self.pure_callees_seen),
            )
        finally:
            self._in_progress.discard(func.qualname)
        self._cache[func.qualname] = rep
        return rep

    def _source_of(self, func: P.PyFunction) -> bytes:
        raw = (self.repo.root / func.file).read_bytes()
        return raw.decode("utf-8", errors="replace").encode("utf-8")

    def check_callee(self, target: P.PyFunction) -> PurityReport:
        return self.check(target)


def check_purity(func: P.PyFunction, repo: P.PyRepo) -> PurityReport:
    """Public entry: purity report for one Python function."""
    return _PyPurityChecker(repo).check(func)


def check_purity_repo(repo: P.PyRepo) -> dict[str, PurityReport]:
    """Purity reports for every top-level function in the repo."""
    checker = _PyPurityChecker(repo)
    return {f.qualname: checker.check(f) for f in repo.top_level_functions()}


# ---------------------------------------------------------------------------
# Mining: call counts, intensity, ranking (mirrors miner.py).
# ---------------------------------------------------------------------------


def _count_calls(repo: P.PyRepo) -> dict[str, int]:
    """Count function invocations per callee qualname (best effort)."""
    counts: dict[str, int] = {}

    def bump(qn: str) -> None:
        counts[qn] = counts.get(qn, 0) + 1

    for func in repo.top_level_functions():
        raw = (repo.root / func.file).read_bytes()
        src = raw.decode("utf-8", errors="replace").encode("utf-8")

        def visit(node, _self=func) -> None:
            if node.type == "call":
                kids = P.named(node)
                if kids and kids[0].type == "identifier":
                    name = P.text(kids[0], src)
                    if name != _self.name:
                        same = repo.functions.get(f"{_self.module}.{name}")
                        target = None
                        if (
                            same is not None
                            and not same.nested
                            and not same.is_method
                        ):
                            target = same
                        else:
                            cands = [
                                f
                                for f in repo.functions.values()
                                if f.name == name
                                and not f.nested
                                and not f.is_method
                            ]
                            if len(cands) == 1:
                                target = cands[0]
                        if target is not None:
                            bump(target.qualname)
            for c in node.children:
                if c.is_named:
                    visit(c)

        if func.body is not None:
            visit(func.body)
    return counts


def _source_of(repo: P.PyRepo, func: P.PyFunction) -> bytes:
    raw = (repo.root / func.file).read_bytes()
    return raw.decode("utf-8", errors="replace").encode("utf-8")


def _intensity(func: P.PyFunction, src: bytes, repo: P.PyRepo) -> float:
    """Cheap compute-intensity heuristic from the function body AST."""
    if func.body is None:
        return 0.0
    loops = comps = arith = subscripts = math_calls = 0
    recursive = False
    pyfile = repo.modules[func.module]

    def visit(node) -> None:
        nonlocal loops, comps, arith, subscripts, math_calls, recursive
        t = node.type
        if t in ("function_definition", "lambda", "class_definition"):
            return  # nested scopes don't count toward the outer intensity
        if t in ("for_statement", "while_statement"):
            loops += 1
        elif t in (
            "list_comprehension",
            "set_comprehension",
            "dictionary_comprehension",
            "generator_expression",
        ):
            comps += 1
        elif t == "binary_operator":
            arith += 1
        elif t == "subscript":
            subscripts += 1
        elif t == "call":
            kids = P.named(node)
            if kids:
                fn = kids[0]
                if fn.type == "identifier" and P.text(fn, src) == func.name:
                    recursive = True
                elif fn.type == "attribute":
                    attr_kids = P.named(fn)
                    if attr_kids and attr_kids[0].type == "identifier":
                        root = P.text(attr_kids[0], src)
                        dotted = (pyfile.imports.get(root) or "")
                        if dotted.split(".")[0] in _PURE_MODULES:
                            math_calls += 1
        for c in node.children:
            if c.is_named:
                visit(c)

    visit(func.body)
    return (
        1.0
        + 2.0 * loops
        + 1.5 * comps
        + 0.3 * arith
        + 0.4 * subscripts
        + 1.0 * math_calls
        + (4.0 if recursive else 0.0)
    )


def mine(repo: P.PyRepo) -> list[PyCandidate]:
    """Return ranked kernel candidates (pure and impure alike).

    Mirrors :func:`kernel_port.miner.mine`: pure candidates sort before
    impure ones, each side ranked by ``(1 + calls) * intensity``. Impure
    candidates get score 0.0 and carry their rejection reasons.
    """
    reports = check_purity_repo(repo)
    call_counts = _count_calls(repo)
    cands: list[PyCandidate] = []
    for func in repo.top_level_functions():
        rep = reports[func.qualname]
        calls = call_counts.get(func.qualname, 0)
        intensity = _intensity(func, _source_of(repo, func), repo)
        score = (1 + calls) * intensity if rep.pure else 0.0
        cands.append(
            PyCandidate(
                qualname=func.qualname,
                file=func.file,
                lineno=func.lineno,
                signature=func.signature,
                pure=rep.pure,
                reasons=rep.reasons,
                calls=calls,
                intensity=round(intensity, 1),
                score=round(score, 1),
            )
        )
    cands.sort(key=lambda c: (c.pure, c.score), reverse=True)
    return cands
