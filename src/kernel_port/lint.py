"""Safety lint for generated JNI glue (``kernels_jni.cpp``) and Rust glue
(``src/lib.rs``).

Line/regex-based checks over the generated C++ text for three classic JNI
bug shapes:

- ``missing_release``: every ``Get<..>ArrayElements`` / ``GetStringUTFChars``
  pin of a variable ``p_x`` has a matching ``Release<..>`` referencing the
  same variable;
- ``missing_null_check``: after each Get there is a nullptr check on the
  variable (``== nullptr``, ``!= nullptr`` or ``!p_x``) before its first use;
- ``local_ref_leak``: ``NewObjectArray`` / ``NewObject`` / ``NewStringUTF``
  inside a ``for``/``while`` loop body with no ``DeleteLocalRef`` in that
  body (local-reference accumulation).

and, for the Rust glue (see :func:`lint_rust`):

- ``missing_no_mangle``: a ``pub extern "system" fn`` entry point without a
  preceding ``#[no_mangle]`` (``System.loadLibrary`` would fail to find it);
- ``ffi_panic``: ``panic!(`` / ``.unwrap()`` / ``.expect(`` anywhere in the
  file (panicking across the FFI boundary is UB; the template uses
  ``jni_try!`` instead);
- ``wrong_abi``: an exported entry fn declared with an ABI other than
  ``"system"`` (e.g. ``"C"``), which is wrong for JNI on most platforms;
- ``missing_jni_use``: no ``use jni::JNIEnv`` (template broken or
  hand-edited badly).

Pure functions over the generated text: no I/O. Warnings are sorted by
line; clean input yields an empty list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class LintWarning:
    kind: str  # "missing_release" | "missing_null_check" | "local_ref_leak"
    #        | "missing_no_mangle" | "ffi_panic" | "wrong_abi" | "missing_jni_use"
    line: int  # 1-based line number
    message: str


# `double* p_a = reinterpret_cast<double*>(\n env->GetDoubleArrayElements(a, nullptr));`
# or `const char* p_s = env->GetStringUTFChars(s, nullptr);`
_ACQUIRE_RE = re.compile(
    r"(?P<var>[A-Za-z_]\w*)\s*=\s*"
    r"(?:reinterpret_cast<[^;]*?>\(\s*)?"
    r"env->(?P<get>Get\w+ArrayElements|GetStringUTFChars)\(",
    re.DOTALL,
)

_NULL_CHECK_RES = [
    r"!\s*{v}\b",
    r"\b{v}\s*==\s*nullptr",
    r"nullptr\s*==\s*{v}",
    r"\b{v}\s*!=\s*nullptr",
    r"nullptr\s*!=\s*{v}",
]

_NEW_RE = re.compile(r"env->(NewObjectArray|NewObject|NewStringUTF)\s*\(")
_LOOP_RE = re.compile(r"^\s*(for|while)\s*\(.*\)\s*(\{)?\s*(?://.*)?$")


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _null_check_re(var: str) -> re.Pattern:
    return re.compile("|".join(p.format(v=re.escape(var)) for p in _NULL_CHECK_RES))


def _loop_body(lines: list[str], header: int) -> tuple[int, int] | None:
    """Return (start, end) 0-based line range of a loop body, end exclusive.

    ``header`` is the 0-based line of the ``for``/``while`` header. With a
    ``{`` on the header line the body runs to the matching ``}``; otherwise
    the body is the single following line.
    """
    m = _LOOP_RE.match(lines[header])
    if m is None:
        return None
    if m.group(2) != "{":
        nxt = header + 1
        return (nxt, nxt + 1) if nxt < len(lines) else None
    depth = 0
    opened = False
    for k in range(header, len(lines)):
        for ch in lines[k]:
            if ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
        if opened and depth == 0:
            return (header + 1, k)
    return (header + 1, len(lines))


def lint_glue(cpp_text: str) -> list[LintWarning]:
    """Lint generated JNI C++ text; return warnings sorted by line."""
    warnings: list[LintWarning] = []
    lines = cpp_text.splitlines()

    # ---- (a) release pairing + (b) null check, per pin ----
    for m in _ACQUIRE_RE.finditer(cpp_text):
        var = m.group("var")
        get = m.group("get")
        line = _line_of(cpp_text, m.start())
        # end of the acquire statement: first ';' at/after the Get line
        stmt_end = line
        for i in range(line - 1, len(lines)):
            if ";" in lines[i]:
                stmt_end = i + 1
                break

        # (a) the matching Release<..> must reference the same variable
        expected_rel = "Release" + get[len("Get") :]
        rel_re = re.compile(
            r"env->" + re.escape(expected_rel) + r"\([^;]*?\b" + re.escape(var) + r"\b",
            re.DOTALL,
        )
        if not rel_re.search(cpp_text):
            warnings.append(
                LintWarning(
                    kind="missing_release",
                    line=line,
                    message=(
                        f"{get} pins '{var}' but no {expected_rel} "
                        f"references the same variable"
                    ),
                )
            )

        # (b) a nullptr check must come before the variable's first use
        null_re = _null_check_re(var)
        use_re = re.compile(r"\b" + re.escape(var) + r"\b")
        checked = False
        first_use: int | None = None
        for i in range(stmt_end, len(lines)):
            text = lines[i]
            if null_re.search(text):
                checked = True
                break
            if use_re.search(text):
                first_use = i + 1
                break
        if not checked:
            where = (
                f"before its first use (line {first_use})"
                if first_use is not None
                else "before first use"
            )
            warnings.append(
                LintWarning(
                    kind="missing_null_check",
                    line=line,
                    message=f"pin '{var}' from {get} is not null-checked {where}",
                )
            )

    # ---- (c) local references created inside loops ----
    for h, text in enumerate(lines):
        if not _LOOP_RE.match(text):
            continue
        body = _loop_body(lines, h)
        if body is None:
            continue
        start, end = body
        body_text = "\n".join(lines[start:end])
        if "DeleteLocalRef" in body_text:
            continue
        for b in range(start, end):
            for nm in _NEW_RE.finditer(lines[b]):
                warnings.append(
                    LintWarning(
                        kind="local_ref_leak",
                        line=b + 1,
                        message=(
                            f"env->{nm.group(1)} inside a loop without "
                            f"DeleteLocalRef: local references accumulate"
                        ),
                    )
                )

    warnings.sort(key=lambda w: w.line)
    return warnings


# ---------------------------------------------------------------------------
# Rust glue lint (src/lib.rs)
# ---------------------------------------------------------------------------

#: ``pub extern "<abi>" fn`` -- the shape of a JNI entry point.
_ENTRY_FN_RE = re.compile(r'^\s*pub\s+extern\s+"(?P<abi>[^"]+)"\s+fn\b')

_JNI_USE_RE = re.compile(r"^\s*use\s+jni::JNIEnv\s*;")

_FFI_PANIC_RES = [
    re.compile(r"\bpanic!\s*\("),
    re.compile(r"\.unwrap\s*\("),
    re.compile(r"\.expect\s*\("),
]


def _has_no_mangle(lines: list[str], fn_idx: int) -> bool:
    """True when the entry-point fn on ``lines[fn_idx]`` (0-based) has a
    ``#[no_mangle]`` attribute immediately before it, skipping blank lines
    and ``//`` comments (and tolerating other ``#[...]`` attributes in
    between)."""
    i = fn_idx - 1
    while i >= 0:
        s = lines[i].strip()
        if not s or s.startswith("//"):
            i -= 1
            continue
        if s.startswith("#["):
            if s == "#[no_mangle]":
                return True
            i -= 1
            continue
        return False
    return False


def lint_rust(text: str) -> list[LintWarning]:
    """Lint generated Rust glue (``src/lib.rs``); return warnings sorted by
    line.

    Cheap, high-value static checks over the generated wrapper plus the
    agent-pasted translated body:

    - ``missing_no_mangle``: a ``pub extern "system" fn`` JNI entry point
      without a preceding ``#[no_mangle]`` -- ``System.loadLibrary``
      would fail to find it;
    - ``ffi_panic``: ``panic!(`` / ``.unwrap()`` / ``.expect(`` in code
      (``//`` comments stripped, so the ``TODO(agent)`` markers never
      fire) -- panicking across the FFI boundary is UB, which the JVM
      cannot catch; the template's ``jni_try!`` macro is the intended
      pattern (return a sensible default instead);
    - ``wrong_abi``: an exported entry fn declared with an ABI other than
      ``"system"`` (e.g. ``"C"``) -- wrong for JNI on most platforms;
    - ``missing_jni_use``: no ``use jni::JNIEnv`` -- the template is
      broken or was hand-edited badly (the entry point signature needs
      ``JNIEnv``).

    Deliberately NOT checked: null checks on JNI handles. The generated
    entry points use the jni crate's safe wrapper types (``JObject``,
    ``JString``, ...) which are non-null by construction -- there are no
    raw JNI pointers to null-check in generated code, and flagging the
    agent's translated body for idioms it may not use would be noise.
    """
    warnings: list[LintWarning] = []
    lines = text.splitlines()

    # ---- (d) the entry point needs the jni crate's JNIEnv type ----
    if not any(_JNI_USE_RE.match(line) for line in lines):
        warnings.append(
            LintWarning(
                kind="missing_jni_use",
                line=1,
                message="no `use jni::JNIEnv;`: the generated JNI entry "
                "point signature needs JNIEnv",
            )
        )

    # ---- (a)/(c) per exported entry-point fn ----
    for i, line in enumerate(lines):
        m = _ENTRY_FN_RE.match(line)
        if not m:
            continue
        abi = m.group("abi")
        fn_line = i + 1
        if abi != "system":
            warnings.append(
                LintWarning(
                    kind="wrong_abi",
                    line=fn_line,
                    message=(
                        f'JNI entry point declared as extern "{abi}"; '
                        'JNI on most platforms requires extern "system"'
                    ),
                )
            )
        elif not _has_no_mangle(lines, i):
            warnings.append(
                LintWarning(
                    kind="missing_no_mangle",
                    line=fn_line,
                    message="JNI entry point lacks #[no_mangle]: "
                    "System.loadLibrary would fail to find the symbol",
                )
            )

    # ---- (b) panicking constructs in code (not `//` comments) ----
    for i, line in enumerate(lines):
        code = line.split("//", 1)[0]
        for cre in _FFI_PANIC_RES:
            if cre.search(code):
                warnings.append(
                    LintWarning(
                        kind="ffi_panic",
                        line=i + 1,
                        message="panics across the FFI boundary (UB, the "
                        "JVM cannot catch it): return a default via "
                        "jni_try! instead",
                    )
                )
                break  # one finding per line keeps the output readable

    warnings.sort(key=lambda w: w.line)
    return warnings
