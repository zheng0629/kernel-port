"""Counterexample persistence and LLM-facing rendering.

When ``verify`` finds mismatches, :func:`harness._persist_failures` writes
them to ``.kp-work/<method>.failures.json``. This module reads that file
back (:func:`load_failures`) and renders it as a markdown prompt section
(:func:`format_counterexamples`) that the calling agent feeds to its LLM
so the next translation fixes exactly the failing inputs.

``format_counterexamples`` accepts two call shapes:

- ``format_counterexamples(failures)`` -- ``failures`` is a list of dicts
  as written by the harness (``args``, ``java_result``, ``jni_result``,
  ``detail``, ``boundary``), e.g. via :func:`load_failures`;
- ``format_counterexamples(qualname, failing_cases)`` -- the shape used by
  the pipeline companion, where ``failing_cases`` holds VerifyCase-like
  objects (duck-typed: ``args``, ``detail``, ``java_result``,
  ``jni_result``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

_READABLE_SPECIAL = {"nan": "NaN", "inf": "+Inf", "-inf": "-Inf"}


def load_failures(path: str | Path) -> list[dict]:
    """Read a ``.failures.json`` file; [] when missing or invalid."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    return []


def _pretty_scalar(v) -> str:
    if isinstance(v, str) and v in _READABLE_SPECIAL:
        return _READABLE_SPECIAL[v]
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "+Inf" if v > 0 else "-Inf"
        return repr(v)
    return str(v)


def _pretty_value(v) -> str:
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_pretty_scalar(x) for x in v) + "]"
    return _pretty_scalar(v)


def _pretty_token(tok: str) -> str:
    """Render a harness result token readably (``d:NaN`` -> ``NaN``)."""
    if not tok:
        return "(no result)"
    if tok == "v":
        return "void"
    if ":" not in tok:
        return tok
    tag, val = tok.split(":", 1)
    if tag in ("ai", "al", "af", "ad"):
        elems = [] if val == "" else val.split(",")
        conv = _pretty_scalar(float(e)) if tag in ("af", "ad") else _pretty_scalar
        return "[" + ", ".join(conv(e) for e in elems) + "]"
    if tag in ("d", "f", "i", "l", "z"):
        if val in _READABLE_SPECIAL:
            return _READABLE_SPECIAL[val]
        return val
    return tok


def _case_to_dict(c) -> dict:
    """Duck-typed conversion of a VerifyCase-like object."""
    return {
        "args": getattr(c, "args", []),
        "java_result": getattr(c, "java_result", ""),
        "jni_result": getattr(c, "jni_result", ""),
        "detail": getattr(c, "detail", ""),
        "boundary": bool(getattr(c, "boundary", False)),
    }


def _esc(cell: str) -> str:
    return cell.replace("|", "\\|").replace("\n", " ")


def format_counterexamples(failures, qualname: str | None = None) -> str:
    """Render failing cases as a markdown prompt section for the LLM.

    Call as ``format_counterexamples(failures: list[dict])`` or, as the
    pipeline companion does, ``format_counterexamples(qualname,
    failing_cases)`` with VerifyCase-like objects.
    """
    if isinstance(failures, str):
        # Pipeline call shape: format_counterexamples(qualname, failing_cases).
        qualname, cases = failures, qualname
        failures = [_case_to_dict(c) for c in (cases or [])]
    failures = list(failures or [])

    lines = ["## Counterexamples — the previous translation failed these", ""]
    if qualname:
        lines.append(f"Method: `{qualname}`")
        lines.append("")
    if not failures:
        lines.append("No counterexamples recorded.")
        return "\n".join(lines) + "\n"

    lines.append("| # | inputs | expected (Java) | got (JNI) | boundary | detail |")
    lines.append("|---|--------|-----------------|-----------|----------|--------|")
    for i, f in enumerate(failures, 1):
        args = ", ".join(_pretty_value(a) for a in f.get("args", []))
        java_r = _pretty_token(str(f.get("java_result", "")))
        jni_r = _pretty_token(str(f.get("jni_result", "")))
        b = "yes" if f.get("boundary") else "no"
        detail = _esc(str(f.get("detail", "")))
        lines.append(
            f"| {i} | `{_esc(args)}` | `{_esc(java_r)}` | `{_esc(jni_r)}` | {b} | {detail} |"
        )
    lines.append("")
    lines.append(
        "The C++ translation is wrong on exactly the inputs above: for each "
        "case it must produce the Java result instead of the JNI result."
    )
    for i, f in enumerate(failures, 1):
        args = ", ".join(_pretty_value(a) for a in f.get("args", []))
        java_r = _pretty_token(str(f.get("java_result", "")))
        jni_r = _pretty_token(str(f.get("jni_result", "")))
        lines.append(
            f"- Case {i}: with inputs ({args}), return {java_r} instead of {jni_r}."
        )
    lines.append(
        "Do not change behavior on any input that currently passes; "
        "re-run `kernel-port verify` to confirm every case passes."
    )
    return "\n".join(lines) + "\n"
