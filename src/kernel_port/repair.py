"""Deterministic repair loop: verify -> counterexamples -> re-translate -> re-verify.

``repair_loop`` automates the manual loop that v0.3 left to the calling
agent: build the glue with the current translated body, run differential
verification, and on failure feed the counterexamples (via
:mod:`kernel_port.counterex`) back into the translator for another round.

kernel-port stays LLM-agnostic: translation proposals come from a
caller-provided hook, either a Python callable
(:data:`BodyProvider`) or a shell command (``propose_cmd``) with the agent /
LLM sitting behind it. The shell contract is::

    <propose-cmd> <prompt_material_file> <round_no> <failures_json_file>

The command reads the two input files and its round number from argv and
prints the translated native body (C++ for the ``jni`` backend, C for the
``ffm`` backend, Rust for the ``rust`` backend) to stdout; fenced code blocks are unwrapped automatically.

Stopping conditions
-------------------
- every verify case passes -> ``status="success"``
- ``max_rounds`` reached without success -> ``status="gave_up"``
- the current failure signature matches ANY earlier round's signature
  (A/B/A/B oscillation included) -> ``status="stalled"``
  (no progress; continuing would just burn tokens)
- setup problems (unknown/impure method, glue generation failure, hook
  crash) -> ``status="error"`` with a message, no rounds recorded

The returned :class:`RepairReport` is JSON-serializable via
:meth:`RepairReport.to_dict` for later ``--json`` CLI output.

Python mode
-----------
``repair_loop(..., lang="python")`` runs the same loop against the
Python->C (cffi) pipeline instead of the JVM harness: the function is
resolved with :mod:`kernel_port.py_ast`, prompt material comes from
:func:`kernel_port.py_pipeline.render_python_extraction`, and each round
generates cffi glue, splices the C body, gcc-builds, and differential-
verifies with :mod:`kernel_port.py_harness`. Failure signatures are
computed from the Python verify report; the stop conditions are the same.
``backend`` is forced to ``"cffi"`` in Python mode.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import math
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

from . import counterex, extractor, harness, jni
from . import java_ast as J
from .pipeline import splice_body

__all__ = [
    "BodyProvider",
    "RepairError",
    "RepairReport",
    "RoundResult",
    "REPAIR_STATUSES",
    "extract_body_from_output",
    "run_propose_cmd",
    "repair_loop",
]

REPAIR_STATUSES = ("success", "stalled", "gave_up", "error")

#: A translation-proposal hook. ``prompt_material`` is the markdown brief
#: for this round (translation brief on round 1, brief + counterexamples +
#: previous body on later rounds); ``round_no`` starts at 1; ``failures`` is
#: a list of dicts with keys ``args`` / ``java_result`` / ``jni_result`` /
#: ``detail`` / ``boundary`` (or ``kind="build_error"`` when the previous
#: round failed to build). In Python mode (``lang="python"``) the failure
#: dicts instead carry ``args`` / ``detail`` / ``boundary`` (the detail
#: already records the Python-vs-C mismatch) and the brief asks for a C
#: body. Returns the translated native body as a string.
BodyProvider = Callable[[str, int, list[dict]], str]


class RepairError(Exception):
    """The propose hook (or the repair setup) failed deterministically."""


# ---------------------------------------------------------------------------
# report types
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RoundResult:
    round: int
    body_source: str  # "initial" | "provider" | "shell-hook"
    passed: int
    failed: int
    skipped: int
    boundary_passed: int
    boundary_failed: int
    boundary_skipped: int
    build_error: str | None
    elapsed_s: float
    failure_signature: str  # sha256 hex of the normalized failure set
    failure_summaries: list[str]
    prompt_path: str
    failures_path: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class RepairReport:
    status: str  # one of REPAIR_STATUSES
    method: str
    signature: str
    backend: str
    rounds: list[RoundResult]
    final_body: str | None  # the passing body on success, else None
    total_elapsed_s: float
    work_dir: str
    error: str | None = None  # set when status == "error"
    last_body: str | None = None  # last attempted body on stalled/gave_up, else None
    stop_reason: str | None = None  # human-readable reason the loop stopped

    @property
    def rounds_run(self) -> int:
        return len(self.rounds)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "method": self.method,
            "signature": self.signature,
            "backend": self.backend,
            "rounds": [r.to_dict() for r in self.rounds],
            "final_body": self.final_body,
            "total_elapsed_s": self.total_elapsed_s,
            "work_dir": self.work_dir,
            "error": self.error,
            "last_body": self.last_body,
            "stop_reason": self.stop_reason,
            "generated_at": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(timespec="seconds"),
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _jsonable(value):
    """Make ``value`` JSON-safe (non-finite floats become strings)."""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "+Inf" if value > 0 else "-Inf"
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _native_lang(backend: str) -> str:
    return {"ffm": "C", "rust": "Rust"}.get(backend, "C++")


def _failure_dicts(vrep: harness.VerifyReport) -> list[dict]:
    """JSON-safe failure entries for the provider and the failures file."""
    out: list[dict] = []
    for is_boundary, cases in ((False, vrep.cases), (True, vrep.boundary)):
        for c in cases:
            if c.ok:
                continue
            out.append(
                {
                    "kind": "verify_failure",
                    "case": c.index,
                    "boundary": is_boundary,
                    "args": _jsonable(c.args),
                    "java_result": c.java_result,
                    "jni_result": c.jni_result,
                    "detail": c.detail,
                }
            )
    return out


def _failure_summaries(vrep: harness.VerifyReport, limit: int = 8) -> list[str]:
    lines: list[str] = []
    failing = [c for c in vrep.cases if not c.ok]
    failing += [c for c in vrep.boundary if not c.ok]
    for c in failing[:limit]:
        lines.append(
            f"case {c.index}: args={c.args!r} "
            f"java={c.java_result!r} native={c.jni_result!r} {c.detail}".strip()
        )
    if len(failing) > limit:
        lines.append(f"... and {len(failing) - limit} more failing cases")
    return lines


def _failure_signature(
    vrep: harness.VerifyReport | None, build_error: str | None, work_dir: Path
) -> str:
    """Normalized identity of a round's failure set, for stall detection.

    Build-error text is normalized against ``work_dir`` so the deterministic
    per-round directory names (``round-1``, ``round-2``, ...) do not make
    identical failures look different.
    """
    if build_error is not None:
        norm = build_error.replace(str(work_dir), "<work>")
        # Per-round directory names (round-1, round-2, ...) are
        # deterministic but differ across rounds; normalize them so an
        # identical build failure compares equal.
        norm = re.sub(r"round-\d+", "round-N", norm)
        payload = ("build", norm.strip())
    else:
        assert vrep is not None
        items = []
        for is_boundary, cases in ((False, vrep.cases), (True, vrep.boundary)):
            for c in cases:
                if c.ok:
                    continue
                items.append(
                    (
                        is_boundary,
                        c.index,
                        json.dumps(_jsonable(c.args), sort_keys=True),
                        c.java_result,
                        c.jni_result,
                    )
                )
        payload = ("verify", sorted(items))
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()[:16]


def extract_body_from_output(text: str) -> str:
    """Extract the translated body from hook stdout.

    Plain output is used as-is (stripped). When the output contains a
    markdown fenced code block (an LLM habit), the first block's contents
    are used and a leading language tag (``cpp``/``c``/...) is dropped.
    """
    text = text.strip()
    lines = text.splitlines()
    fence = next(
        (i for i, ln in enumerate(lines) if ln.strip().startswith("```")), None
    )
    if fence is not None:
        rest = lines[fence + 1 :]
        # drop a trailing closing fence if present
        if rest and rest[-1].strip().startswith("```"):
            rest = rest[:-1]
        # also drop an inner closing fence (block ends mid-output)
        end = next(
            (i for i, ln in enumerate(rest) if ln.strip().startswith("```")),
            None,
        )
        if end is not None:
            rest = rest[:end]
        text = "\n".join(rest).strip()
    return text


def run_propose_cmd(
    cmd: str,
    prompt_material: str,
    round_no: int,
    failures: list[dict],
    timeout: float = 300.0,
    scratch_dir: str | Path | None = None,
) -> str:
    """Run the ``--propose-cmd`` shell hook and return the translated body.

    Contract: ``<cmd> <prompt_material_file> <round_no> <failures_json_file>``
    is executed (no shell); the translated body is read from stdout.
    ``RepairError`` is raised when the command exits nonzero, times out, or
    prints an empty body.
    """
    scratch = Path(scratch_dir) if scratch_dir is not None else Path(
        tempfile.mkdtemp(prefix="kp-propose-")
    )
    scratch.mkdir(parents=True, exist_ok=True)
    prompt_file = scratch / f"prompt_round_{round_no}.md"
    failures_file = scratch / f"failures_round_{round_no}.json"
    prompt_file.write_text(prompt_material, encoding="utf-8")
    failures_file.write_text(
        json.dumps(_jsonable(failures), indent=2), encoding="utf-8"
    )
    argv = shlex.split(cmd) + [str(prompt_file), str(round_no), str(failures_file)]
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise RepairError(
            f"propose command timed out after {timeout}s: {cmd}"
        ) from exc
    except OSError as exc:
        raise RepairError(f"could not run propose command {cmd!r}: {exc}") from exc
    if proc.returncode != 0:
        raise RepairError(
            f"propose command exited with code {proc.returncode}: "
            f"{proc.stderr.strip()[:2000]}"
        )
    body = extract_body_from_output(proc.stdout)
    if not body:
        raise RepairError("propose command printed an empty body on stdout")
    return body


# ---------------------------------------------------------------------------
# prompt material
# ---------------------------------------------------------------------------


def _initial_prompt(method: J.MethodInfo, ext, backend: str) -> str:
    lang = _native_lang(backend)
    return "\n".join(
        [
            f"# Translate this Java kernel to {lang} (kernel-port repair, round 1)",
            "",
            f"Method: `{method.qualname}`",
            f"Signature: `{method.signature}`",
            "",
            extractor.render_text(ext, backend),
            "",
            "## What to return",
            f"Return ONLY the translated {lang} body of `{method.name}` -- "
            "plain code, no markdown fences, no explanations. It will be "
            "spliced into the generated glue file and verified against the "
            "Java original with differential tests.",
            "",
        ]
    )


def _repair_prompt(
    method: J.MethodInfo,
    ext,
    backend: str,
    round_no: int,
    prev_body: str,
    vrep: harness.VerifyReport | None,
    build_error: str | None,
    failure_dicts: list[dict],
) -> str:
    lang = _native_lang(backend)
    lines = [
        f"# Fix this {lang} translation (kernel-port repair, round {round_no})",
        "",
        f"Method: `{method.qualname}`",
        f"Signature: `{method.signature}`",
        "",
        extractor.render_text(ext, backend),
        "",
    ]
    if build_error is not None:
        lines += [
            "## Build failed",
            "",
            "The previous body did not compile. Compiler output:",
            "",
            "```",
            build_error.strip()[:4000],
            "```",
            "",
        ]
    else:
        assert vrep is not None
        lines.append(
            counterex.format_counterexamples(failure_dicts).rstrip("\n")
        )
        lines.append("")
    lines += [
        "## Your previous attempt",
        "",
        f"```{lang.lower()}",
        prev_body.rstrip("\n"),
        "```",
        "",
        "Fix the body so every case passes. Return ONLY the corrected "
        f"{lang} body -- plain code, no markdown fences, no explanations.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Python -> C (cffi) path: prompt material, build, failure extraction
# ---------------------------------------------------------------------------


def _py_initial_prompt(fn, pext: dict) -> str:
    from . import py_pipeline as _pp

    return "\n".join(
        [
            "# Translate this Python kernel to C (kernel-port repair, round 1)",
            "",
            f"Function: `{fn.qualname}`",
            f"Signature: `{fn.signature}`",
            "",
            _pp.render_python_extraction(pext),
            "",
            "## What to return",
            f"Return ONLY the translated C body of `{fn.qualname}` -- "
            "plain code, no markdown fences, no explanations. It will be "
            "spliced into the generated cffi glue file and verified against "
            "the Python original with differential tests.",
            "",
        ]
    )


def _format_py_counterexamples(failures: list[dict]) -> str:
    """Markdown counterexample section for Python->C repair prompts.

    ``failures`` are :func:`_py_failure_dicts` entries (keys ``case`` /
    ``boundary`` / ``args`` / ``detail``); the detail already records the
    Python-vs-C mismatch, so no separate result columns are needed.
    """
    lines = ["## Counterexamples — the previous C translation failed these", ""]
    if not failures:
        lines.append("No counterexamples recorded.")
        return "\n".join(lines) + "\n"
    for i, f in enumerate(failures, 1):
        args = ", ".join(repr(a) for a in f.get("args", []))
        b = "boundary" if f.get("boundary") else "random"
        lines.append(f"- Case {i} [{b}] inputs ({args}): {f.get('detail', '')}")
    lines.append("")
    lines.append(
        "The C translation is wrong on exactly the inputs above: fix the "
        "body so it matches the Python result on every case. Do not change "
        "behavior on any input that currently passes."
    )
    return "\n".join(lines) + "\n"


def _py_repair_prompt(
    fn,
    pext: dict,
    round_no: int,
    prev_body: str,
    build_error: str | None,
    failure_dicts: list[dict],
) -> str:
    from . import py_pipeline as _pp

    lines = [
        f"# Fix this C translation (kernel-port repair, round {round_no})",
        "",
        f"Function: `{fn.qualname}`",
        f"Signature: `{fn.signature}`",
        "",
        _pp.render_python_extraction(pext),
        "",
    ]
    if build_error is not None:
        lines += [
            "## Build failed",
            "",
            "The previous body did not compile. Compiler output:",
            "",
            "```",
            build_error.strip()[:4000],
            "```",
            "",
        ]
    else:
        lines.append(_format_py_counterexamples(failure_dicts).rstrip("\n"))
        lines.append("")
    lines += [
        "## Your previous attempt",
        "",
        "```c",
        prev_body.rstrip("\n"),
        "```",
        "",
        "Fix the body so every case passes. Return ONLY the corrected "
        "C body -- plain code, no markdown fences, no explanations.",
        "",
    ]
    return "\n".join(lines)


def _py_build_round(fn, sig, body: str, round_dir: Path):
    """Splice ``body`` into fresh cffi glue and gcc-build.

    Returns ``(env, build_error)`` mirroring :func:`_build_round`.
    """
    from . import cffi_glue as _cg
    from . import py_harness as _ph

    glue_dir = round_dir / "glue"
    try:
        files = _cg.generate([fn], glue_dir)
        native_path = files["c"]  # kernels_cffi.c
    except _cg.CffiGlueError as exc:
        return None, f"glue generation failed: {exc}"
    try:
        _cg.fill_translation(native_path, sig.cname, body)
    except (ValueError, KeyError) as exc:
        return None, f"could not splice translated body: {exc}"
    try:
        env = _ph.build_env(glue_dir, round_dir / "build", sig)
    except RuntimeError as exc:
        return None, str(exc)
    return env, None


def _py_failure_dicts(vrep) -> list[dict]:
    """JSON-safe failure entries for the provider and the failures file."""
    out: list[dict] = []
    for is_boundary, cases in ((False, vrep.cases), (True, vrep.boundary_cases)):
        for c in cases:
            if c.ok or c.skipped:
                continue
            out.append(
                {
                    "kind": "verify_failure",
                    "case": c.index,
                    "boundary": is_boundary,
                    "args": _jsonable(c.args),
                    "detail": c.detail,
                }
            )
    return out


def _py_failure_summaries(vrep, limit: int = 8) -> list[str]:
    lines: list[str] = []
    failing = [
        c
        for c in vrep.cases + vrep.boundary_cases
        if not c.ok and not c.skipped
    ]
    for c in failing[:limit]:
        lines.append(f"case {c.index}: args={c.args!r} {c.detail}".strip())
    if len(failing) > limit:
        lines.append(f"... and {len(failing) - limit} more failing cases")
    return lines


def _py_failure_signature(vrep, build_error: str | None, work_dir: Path) -> str:
    """Normalized identity of a Python round's failure set, for stall detection.

    Mirrors :func:`_failure_signature`: build-error text is normalized
    against ``work_dir`` and per-round directory names; verify failures are
    keyed by (boundary, index, args, detail) since the Python case results
    carry the mismatch in ``detail`` rather than separate result fields.
    """
    if build_error is not None:
        norm = build_error.replace(str(work_dir), "<work>")
        norm = re.sub(r"round-\d+", "round-N", norm)
        payload = ("build", norm.strip())
    else:
        assert vrep is not None
        items = []
        for is_boundary, cases in ((False, vrep.cases), (True, vrep.boundary_cases)):
            for c in cases:
                if c.ok or c.skipped:
                    continue
                items.append(
                    (
                        is_boundary,
                        c.index,
                        json.dumps(_jsonable(c.args), sort_keys=True),
                        c.detail,
                    )
                )
        payload = ("verify", sorted(items))
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# glue + build + verify for one round
# ---------------------------------------------------------------------------


def _generate_glue(
    method: J.MethodInfo, glue_dir: Path, backend: str, repo=None
) -> Path:
    """Generate fresh glue; return the native source file path."""
    if backend == "ffm":
        from . import ffm as _ffm

        files = _ffm.generate(method, glue_dir, package=jni.glue_package(repo, method))
        return files["c"]  # kernels_ffm.c
    if backend == "rust":
        from . import rust_glue as _rust

        files = _rust.generate(
            method, glue_dir, package=jni.glue_package(repo, method)
        )
        return files["rust"]  # src/lib.rs
    files = jni.generate(method, glue_dir, package=jni.glue_package(repo, method))
    return files["cpp"]  # kernels_jni.cpp


def _build_round(
    repo_root: Path,
    method: J.MethodInfo,
    body: str,
    round_dir: Path,
    backend: str,
    repo=None,
) -> tuple[harness.BuiltEnv | None, str | None]:
    """Splice ``body`` into fresh glue and compile. Returns (env, build_error)."""
    glue_dir = round_dir / "glue"
    try:
        native_path = _generate_glue(method, glue_dir, backend, repo=repo)
    except ValueError as exc:
        return None, f"glue generation failed: {exc}"
    try:
        splice_body(native_path, body)
    except ValueError as exc:
        return None, f"could not splice translated body: {exc}"
    try:
        env = harness.build_env(
            repo_root, method, glue_dir, round_dir / "build", backend=backend
        )
    except (RuntimeError, ValueError) as exc:
        return None, str(exc)
    return env, None


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def repair_loop(
    repo,
    method,
    translated_body_provider: BodyProvider | None = None,
    propose_cmd: str | None = None,
    max_rounds: int = 5,
    verify_cases: int = 50,
    seed: int = 42,
    boundary: bool = True,
    boundary_cases: int = 24,
    tol_float: float = 1e-6,
    tol_double: float = 1e-12,
    backend: str = "jni",
    work_dir=None,
    initial_body: str | None = None,
    propose_timeout: float = 300.0,
    subdir=None,
    lang: str = "java",
) -> RepairReport:
    """Run the verify -> counterexamples -> re-translate loop to completion.

    ``repo`` is the repo root; ``method`` is a qualified method name
    (a :class:`java_ast.MethodInfo` / :class:`py_ast.PyFunction` is also
    accepted). Exactly one of ``translated_body_provider`` / ``propose_cmd``
    must be given (unless ``initial_body`` is combined with one of them --
    a provider or command is still required for round 2+).

    ``lang="java"`` (default) drives the JVM pipeline with one of the
    ``backend`` native targets (``"jni"`` / ``"ffm"`` / ``"rust"``).
    ``lang="python"`` drives the Python->C (cffi) pipeline instead
    (:mod:`kernel_port.py_pipeline` / :mod:`kernel_port.py_harness`);
    ``backend`` is forced to ``"cffi"`` and ``boundary_cases`` is ignored
    (the Python harness caps its boundary sweep internally).

    Each round regenerates the glue from scratch into
    ``<work_dir>/round-<n>/`` (rounds are hermetic), splices in the proposed
    body, builds, and runs differential verification with the fixed
    ``seed`` so failure sets are comparable across rounds. The loop stops
    with ``status="success"`` as soon as every case passes, with
    ``status="stalled"`` when the current failure signature matches any
    earlier round's (oscillation), and with ``status="gave_up"`` when
    ``max_rounds`` is reached.
    """
    started = time.perf_counter()
    if lang not in ("java", "python"):
        raise ValueError(f"unknown lang {lang!r}: expected 'java' or 'python'")
    if lang == "python":
        backend = "cffi"  # the only backend for Python kernels
    elif backend not in ("jni", "ffm", "rust"):
        raise ValueError(f"unknown backend {backend!r}: expected 'jni', 'ffm' or 'rust'")
    if (translated_body_provider is None) == (propose_cmd is None):
        raise ValueError(
            "exactly one of translated_body_provider / propose_cmd is required"
        )
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1")

    repo_root = Path(repo)
    work = (
        Path(work_dir)
        if work_dir is not None
        else Path(tempfile.mkdtemp(prefix="kp-repair-"))
    )
    work.mkdir(parents=True, exist_ok=True)

    def _error(msg: str, mname="", sig="") -> RepairReport:
        return RepairReport(
            status="error",
            method=mname,
            signature=sig,
            backend=backend,
            rounds=[],
            final_body=None,
            total_elapsed_s=time.perf_counter() - started,
            work_dir=str(work),
            error=msg,
            stop_reason=f"error: {msg}",
        )

    # --- resolve method + purity gate (same as pipeline) ---
    if lang == "python":
        from . import py_ast as _P
        from . import cffi_glue as _C
        from . import py_harness as _H
        from . import py_pipeline as _PP

        try:
            pr = _P.scan_repo(repo_root, subdir)
        except Exception as exc:
            return _error(f"could not scan repo {repo_root}: {exc}")
        try:
            fn = (
                method
                if isinstance(method, _P.PyFunction)
                else pr.functions.get(method)
            )
            if fn is None or fn.nested or fn.is_method:
                return _error(
                    f"module-level Python function {method!r} not found in "
                    f"{repo_root}"
                )
        except Exception as exc:
            return _error(f"method resolution failed: {exc}")
        try:
            pext = _PP.extract_python(pr, fn)
        except ValueError as exc:
            return _error(str(exc), fn.qualname, fn.signature)
        if not pext["pure"]:
            return _error(
                f"{fn.qualname} is not pure; the repair loop refuses to port it. "
                f"Reasons: {'; '.join(pext['purity_reasons'])}",
                fn.qualname,
                fn.signature,
            )
        if pext["map_error"] is not None:
            return _error(
                f"glue generation failed for {fn.qualname}: {pext['map_error']}",
                fn.qualname,
                fn.signature,
            )
        psig = _C.map_signature(fn)
        qualname, method_sig = fn.qualname, fn.signature

        def _prompt1() -> str:
            return _py_initial_prompt(fn, pext)

        def _promptN(
            round_no: int,
            prev_body: str,
            _prev_vrep,
            build_error: str | None,
            failure_dicts: list[dict],
        ) -> str:
            return _py_repair_prompt(
                fn, pext, round_no, prev_body, build_error, failure_dicts
            )

        def _build(body: str, round_dir: Path):
            return _py_build_round(fn, psig, body, round_dir)

        def _verify(env):
            return _H.run_verify(
                env, psig, repo_root, cases=verify_cases, seed=seed,
                boundary=boundary, tol_float=tol_float, tol_double=tol_double,
            )

        _fdicts = _py_failure_dicts
        _fsums = _py_failure_summaries

        def _fsig(vrep, build_error: str | None) -> str:
            return _py_failure_signature(vrep, build_error, work)
    else:
        try:
            r = J.scan_repo(repo_root, subdir)
        except Exception as exc:
            return _error(f"could not scan repo {repo_root}: {exc}")
        try:
            m = method if isinstance(method, J.MethodInfo) else r.find_method(method)
            if m is None:
                return _error(f"method {method!r} not found in {repo_root}")
        except Exception as exc:
            return _error(f"method resolution failed: {exc}")
        try:
            ext = extractor.extract(r, m.qualname)
        except ValueError as exc:
            return _error(str(exc), m.qualname, m.signature)
        if not ext.purity.pure:
            return _error(
                f"{m.qualname} is not pure; the repair loop refuses to port it. "
                f"Reasons: {'; '.join(ext.purity.reasons)}",
                m.qualname,
                m.signature,
            )
        qualname, method_sig = m.qualname, m.signature

        def _prompt1() -> str:
            return _initial_prompt(m, ext, backend)

        def _promptN(
            round_no: int,
            prev_body: str,
            prev_vrep,
            build_error: str | None,
            failure_dicts: list[dict],
        ) -> str:
            return _repair_prompt(
                m, ext, backend, round_no, prev_body,
                prev_vrep, build_error, failure_dicts,
            )

        def _build(body: str, round_dir: Path):
            return _build_round(repo_root, m, body, round_dir, backend, repo=r)

        def _verify(env):
            return harness.run_verify(
                env, cases=verify_cases, seed=seed,
                boundary=boundary, boundary_cases=boundary_cases,
                tol_float=tol_float, tol_double=tol_double,
            )

        _fdicts = _failure_dicts
        _fsums = _failure_summaries

        def _fsig(vrep, build_error: str | None) -> str:
            return _failure_signature(vrep, build_error, work)

    def _propose(prompt_material: str, round_no: int, failures: list[dict],
                 round_dir: Path) -> tuple[str, str]:
        """Return (body, source).

        Shell-hook failures raise RepairError (handled by the caller as an
        ``error`` report); exceptions from a Python provider propagate --
        a crashing provider is a caller bug, not a translation failure.
        """
        if round_no == 1 and initial_body is not None:
            return initial_body, "initial"
        if translated_body_provider is not None:
            body = translated_body_provider(prompt_material, round_no, failures)
            return body, "provider"
        assert propose_cmd is not None
        body = run_propose_cmd(
            propose_cmd, prompt_material, round_no, failures,
            timeout=propose_timeout, scratch_dir=round_dir / "propose",
        )
        return body, "shell-hook"

    rounds: list[RoundResult] = []
    # (round_no, signature) for every completed round: a repeated signature
    # from ANY earlier round (not just the previous one) means the loop is
    # oscillating instead of converging.
    past_signatures: list[tuple[int, str]] = []
    prev_body: str | None = None
    prev_vrep = None
    prev_build_error: str | None = None
    prev_failure_dicts: list[dict] = []

    for round_no in range(1, max_rounds + 1):
        round_started = time.perf_counter()
        round_dir = work / f"round-{round_no}"
        round_dir.mkdir(parents=True, exist_ok=True)

        if round_no == 1:
            prompt_material = _prompt1()
            failures: list[dict] = []
        else:
            assert prev_body is not None
            prompt_material = _promptN(
                round_no, prev_body, prev_vrep,
                prev_build_error, prev_failure_dicts,
            )
            failures = prev_failure_dicts
        prompt_path = round_dir / "prompt.md"
        prompt_path.write_text(prompt_material, encoding="utf-8")
        failures_path = round_dir / "failures.json"
        failures_path.write_text(
            json.dumps(_jsonable(failures), indent=2), encoding="utf-8"
        )

        try:
            body, source = _propose(prompt_material, round_no, failures, round_dir)
        except RepairError as exc:
            rounds.append(
                RoundResult(
                    round=round_no, body_source="shell-hook",
                    passed=0, failed=0, skipped=0,
                    boundary_passed=0, boundary_failed=0, boundary_skipped=0,
                    build_error=None,
                    elapsed_s=time.perf_counter() - round_started,
                    failure_signature="", failure_summaries=[],
                    prompt_path=str(prompt_path), failures_path=str(failures_path),
                )
            )
            return RepairReport(
                status="error", method=qualname, signature=method_sig,
                backend=backend, rounds=rounds, final_body=None,
                total_elapsed_s=time.perf_counter() - started,
                work_dir=str(work), error=str(exc),
                stop_reason=f"error: {exc}",
            )
        if not isinstance(body, str) or not body.strip():
            return _error(
                f"the proposal hook returned an empty/non-string body "
                f"on round {round_no}",
                qualname, method_sig,
            )

        env, build_error = _build(body, round_dir)
        vrep = None
        all_passed = False
        if build_error is None:
            assert env is not None
            vrep = _verify(env)
            failure_dicts = _fdicts(vrep)
            summaries = _fsums(vrep)
            passed, failed, skipped = vrep.passed, vrep.failed, vrep.skipped
            b_passed = vrep.boundary_passed
            b_failed = vrep.boundary_failed
            b_skipped = vrep.boundary_skipped
            all_passed = vrep.all_passed
        else:
            failure_dicts = [
                {"kind": "build_error", "detail": build_error}
            ]
            summaries = [f"build failed: {build_error[:300]}"]
            passed = failed = skipped = 0
            b_passed = b_failed = b_skipped = 0

        signature = _fsig(vrep, build_error)
        rounds.append(
            RoundResult(
                round=round_no, body_source=source,
                passed=passed, failed=failed, skipped=skipped,
                boundary_passed=b_passed, boundary_failed=b_failed,
                boundary_skipped=b_skipped,
                build_error=build_error,
                elapsed_s=time.perf_counter() - round_started,
                failure_signature=signature,
                failure_summaries=summaries,
                prompt_path=str(prompt_path),
                failures_path=str(failures_path),
            )
        )

        if build_error is None and all_passed:
            return RepairReport(
                status="success", method=qualname, signature=method_sig,
                backend=backend, rounds=rounds, final_body=body,
                total_elapsed_s=time.perf_counter() - started,
                work_dir=str(work),
                stop_reason=(
                    f"all {passed + b_passed} cases passed "
                    f"(round {round_no})"
                ),
            )
        repeated_from = next(
            (r for r, s in past_signatures if s == signature), None
        )
        if repeated_from is not None:
            return RepairReport(
                status="stalled", method=qualname, signature=method_sig,
                backend=backend, rounds=rounds, final_body=None,
                total_elapsed_s=time.perf_counter() - started,
                work_dir=str(work),
                last_body=body,
                stop_reason=(
                    f"failure signature repeated from round {repeated_from} "
                    f"(oscillation); round {round_no} failed identically"
                ),
            )
        past_signatures.append((round_no, signature))
        prev_body = body
        prev_vrep = vrep
        prev_build_error = build_error
        prev_failure_dicts = failure_dicts

    return RepairReport(
        status="gave_up", method=qualname, signature=method_sig,
        backend=backend, rounds=rounds, final_body=None,
        total_elapsed_s=time.perf_counter() - started,
        work_dir=str(work),
        last_body=body,
        stop_reason=f"max_rounds={max_rounds} reached without success",
    )
