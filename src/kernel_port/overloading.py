"""Overload resolution for method references.

CLI-facing helper so commands accept either a bare qualified name
('com.example.K.foo') or a name with an explicit signature suffix
('com.example.K.foo(int,double)') and get back one unambiguous
:class:`~kernel_port.java_ast.MethodInfo`.
"""

from __future__ import annotations

import difflib
import re

from kernel_port import java_ast as J


class AmbiguityError(Exception):
    """Raised when a method reference matches more than one overload."""

    def __init__(self, ref: str, candidates: list[str]) -> None:
        self.ref = ref
        self.candidates = list(candidates)
        lines = "\n".join(f"  {c}" for c in self.candidates)
        super().__init__(
            f"'{ref}' matches {len(self.candidates)} overloads:\n"
            f"{lines}\n"
            "Disambiguate with a (types) suffix, e.g. "
            "'com.example.K.foo(int,double)'."
        )


def signature_of(m: J.MethodInfo) -> str:
    """Render a qualified signature string: 'com.example.K.foo(int,double)'."""
    types = ",".join(p.jtype for p in m.params)
    return f"{m.qualname}({types})"


# Simple names that java_ast may leave unqualified; treated as java.lang.*.
_JAVA_LANG_SHORTHAND = {"String": "java.lang.String"}


def _canonical_type(t: str) -> str:
    """Normalize a Java type text for comparison.

    - removes all whitespace
    - preserves array dimensions ('int[]', 'double[][]')
    - 'String' == 'java.lang.String'
    """
    t = re.sub(r"\s+", "", t)
    m = re.match(r"^(.*?)((?:\[\])*)$", t)
    base, dims = m.group(1), m.group(2)
    return _JAVA_LANG_SHORTHAND.get(base, base) + dims


def _split_ref(ref: str) -> tuple[str, str | None]:
    """Split 'qual.name.foo(int,double)' -> ('qual.name.foo', 'int,double').

    Returns (ref, None) for a bare name without a suffix.
    """
    ref = ref.strip()
    i = ref.find("(")
    if i == -1:
        return ref, None
    if not ref.endswith(")"):
        raise LookupError(
            f"malformed method reference '{ref}': expected ')' at the end"
        )
    return ref[:i].strip(), ref[i + 1 : -1]


def _similar_names(repo: J.JavaRepo, qualname: str, limit: int = 5) -> list[str]:
    known = sorted({m.qualname for ms in repo.methods.values() for m in ms})
    if not known:
        return []
    out: list[str] = []
    cls = qualname.rsplit(".", 1)[0]
    # Prefer other methods of the same class.
    for q in known:
        if q.rsplit(".", 1)[0] == cls and q != qualname:
            out.append(q)
    for q in difflib.get_close_matches(qualname, known, n=limit, cutoff=0.6):
        if q not in out:
            out.append(q)
    return out[:limit]


def parse_method_ref(repo: J.JavaRepo, ref: str) -> J.MethodInfo:
    """Resolve 'com.example.K.foo' or 'com.example.K.foo(int,double)'.

    - Bare name with a unique match -> return it.
    - Bare name with no match -> raise LookupError, listing similar names.
    - Bare name matching several overloads -> raise AmbiguityError with
      candidate signatures.
    - With a '(...)' suffix: filter overloads by arity first, then pick the
      exact parameter-type match; if no exact type match but exactly one
      overload has that arity, pick it; otherwise raise AmbiguityError
      listing the arity-matching candidates.
    """
    ref = (ref or "").strip()
    qualname, types_str = _split_ref(ref)
    overloads = repo.overloads(qualname)

    if not overloads:
        msg = f"method not found: '{ref}'"
        similar = _similar_names(repo, qualname)
        if similar:
            msg += "\nDid you mean:\n" + "\n".join(f"  {s}" for s in similar)
        raise LookupError(msg)

    if types_str is None:
        if len(overloads) == 1:
            return overloads[0]
        raise AmbiguityError(ref, [signature_of(m) for m in overloads])

    wanted_raw = (
        [t for t in (p.strip() for p in types_str.split(",")) if t]
        if types_str.strip()
        else []
    )
    n = len(wanted_raw)
    by_arity = [m for m in overloads if len(m.params) == n]
    if not by_arity:
        arities = ", ".join(str(a) for a in sorted({len(m.params) for m in overloads}))
        raise LookupError(
            f"'{ref}': no overload of '{qualname}' takes {n} parameter(s); "
            f"available arities: {arities}."
        )

    wanted = [_canonical_type(t) for t in wanted_raw]
    exact = [
        m
        for m in by_arity
        if [_canonical_type(p.jtype) for p in m.params] == wanted
    ]
    if exact:
        if len(exact) == 1:
            return exact[0]
        raise AmbiguityError(ref, [signature_of(m) for m in exact])
    if len(by_arity) == 1:
        # Exactly one overload at this arity: pick it even though the
        # spelled types don't match exactly.
        return by_arity[0]
    raise AmbiguityError(ref, [signature_of(m) for m in by_arity])
