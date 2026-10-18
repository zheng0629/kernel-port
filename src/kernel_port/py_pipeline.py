"""`pipeline --lang python`: one-shot mine -> extract -> glue -> verify -> bench.

Python->C (cffi) counterpart of :func:`pipeline.run_pipeline`. Stages:

1. scan the Python repo (tree-sitter) and resolve the target function
   (explicit ``--method`` or the top-ranked pure candidate);
2. purity gate (conservative: impure kernels are refused with reasons);
3. fingerprint cache lookup (a stored verified translation short-circuits
   the whole run with ``status="cache_hit"``);
4. cffi glue generation (``kernels_cffi.c`` + ``kernels_cffi.py``);
5. without a translation: stop after glue with ``awaiting_translation``
   and write the prompt material;
6. with ``translated_body``: splice it in, gcc-build, differential verify,
   wall-clock bench; on verify failure write counterexamples for the
   repair loop;
7. with ``candidates_dir``: verify+bench every ``*.c`` body in the dir and
   pick the fastest passing one (:func:`select.pick_winner`).

The report dict mirrors the Java pipeline's shape (``status``,
``method``, ``verify``/``bench`` summaries, ``instructions``,
``cache_hit``/``fingerprint``, ``candidates``/``winner``) so the same
HTML renderer and MCP surface can consume it.
``lang`` is always ``"python"`` and ``backend`` always ``"cffi"`` here.

The fingerprint cache (``kernel_port.cache``) and multi-candidate
selection (``kernel_port.select``) share record shapes and CLI semantics
with the Java pipeline: ``--no-cache`` skips lookup/store,
``--cache-dir`` overrides the default ``~/.cache/kernel-port``, and a
cache hit reuses the stored translation + verify/bench summaries without
rebuilding or re-running anything.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from . import cache
from . import cffi_glue as C
from . import py_ast as P
from . import py_harness as H
from . import py_mine as PM
from . import select


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def extract_python(repo: P.PyRepo, fn: P.PyFunction) -> dict:
    """Build the extraction record for a Python kernel.

    Returns a JSON-able dict: qualname, signature, purity verdict + reasons,
    dependency closure (pure callees with source), and the full source text.
    Raises :class:`ValueError` when the function cannot be resolved.
    """
    purity = PM.check_purity(fn, repo)
    closure = []
    for qn in purity.pure_callees:
        if qn == fn.qualname:
            continue
        callee = repo.functions.get(qn)
        if callee is None:
            continue
        closure.append(
            {
                "kind": "function",
                "qualname": qn,
                "signature": callee.signature,
                "source": callee.source,
            }
        )
    try:
        sig = C.map_signature(fn)
        c_signature = f"{sig.ret.ctype} {sig.cname}({C._c_params(sig)})"
        map_error = None
    except C.CffiGlueError as exc:
        sig = None
        c_signature = None
        map_error = str(exc)
    return {
        "function": fn.qualname,
        "signature": fn.signature,
        "file": fn.file,
        "lineno": fn.lineno,
        "pure": purity.pure,
        "purity_reasons": list(purity.reasons),
        "source": fn.source,
        "closure": closure,
        "c_signature": c_signature,
        "map_error": map_error,
    }


def render_python_extraction(ext: dict) -> str:
    """Human/agent-readable prompt material for the C translation step."""
    lines = [
        "# kernel-port extraction (Python -> C via cffi)",
        "",
        f"Function:  {ext['function']}",
        f"Signature: {ext['signature']}",
        f"Location:  {ext['file']}:{ext['lineno']}",
        f"Purity:    {'PURE' if ext['pure'] else 'IMPURE'}",
    ]
    if not ext["pure"]:
        lines.append("Reasons:")
        for r in ext["purity_reasons"]:
            lines.append(f"  - {r}")
        lines += [
            "",
            "Refusing to emit translation material: the function is not pure.",
            "Fix the impurity or pick another candidate from `kernel-port mine --lang python`.",
        ]
        return "\n".join(lines)
    if ext["closure"]:
        lines.append("Closure (translate these too, or inline them):")
        for c in ext["closure"]:
            lines.append(f"  - (function) {c['qualname']}")
    else:
        lines.append("Closure:   (none -- self-contained)")
    lines += ["", "## Source: kernel", "", "```python", ext["source"], "```"]
    for c in ext["closure"]:
        lines += ["", f"## Source: function {c['qualname']}", "", "```python", c["source"], "```"]
    lines += [
        "",
        "## Translation notes for the LLM",
        "- Target language: C (compiled with gcc, loaded via cffi). Keep the exact signature semantics.",
        "- Only the function body (and closure bodies) need translation;",
        "  kernel-port generates the cffi glue separately (`kernel-port glue --lang python`).",
    ]
    if ext["c_signature"]:
        lines += [
            "",
            "### Native signature the glue presents",
            f"```c\n{ext['c_signature']}\n```",
            "- Your C body receives exactly these parameters; do not change the signature.",
            "- Type mapping: Python `int` -> `int64_t`, `float` -> `double`,",
            "  `bool` -> `int` (0/1), `str` -> `const char*` (UTF-8, NUL-terminated),",
            "  `list[int]` -> `(int64_t* p_x, int64_t p_x_n)`,",
            "  `list[float]` -> `(double* p_x, int64_t p_x_n)` (row-major, length n).",
            "- List parameters are IN/OUT: mutate them in place exactly as the",
            "  Python code does; an empty list arrives as `NULL` with `n == 0`.",
            "- Python `float` is C `double`: preserve operation order; do not",
            "  reassociate unless the differential test still passes.",
        ]
    else:
        lines += ["", f"WARNING: signature not mappable to C: {ext['map_error']}"]
    return "\n".join(lines)


def _verify_summary(vrep: H.PyVerifyReport) -> dict:
    return {
        "passed": vrep.passed,
        "failed": vrep.failed,
        "skipped": vrep.skipped,
        "boundary_passed": vrep.boundary_passed,
        "boundary_failed": vrep.boundary_failed,
        "boundary_skipped": vrep.boundary_skipped,
        "all_passed": vrep.all_passed,
        "tol_float": vrep.tol_float,
        "tol_double": vrep.tol_double,
    }


def _bench_summary(brep: H.PyBenchReport) -> dict:
    return {
        "iters": brep.iters,
        "warmup": brep.warmup,
        "repeats": brep.repeats,
        "python_per_call_ns": brep.python_per_call_ns,
        "c_per_call_ns": brep.c_per_call_ns,
        "speedup": brep.speedup,
    }


def run_python_pipeline(
    repo,
    method=None,
    subdir=None,
    glue_out=None,
    work_dir=None,
    translated_body=None,
    candidates_dir=None,
    html=None,
    no_cache=False,
    cache_dir=None,
    cases=100,
    seed=42,
    iters=2000,
    warmup=20,
    repeats=3,
    tol_float=1e-6,
    tol_double=1e-12,
) -> dict:
    """Run mine -> extract -> glue -> verify -> bench for one Python kernel.

    ``method`` is the qualified function name (``module.func``); when None the
    top-ranked pure candidate from :func:`py_mine.mine` is used.
    ``translated_body`` is the C translation text produced by the calling
    agent; without it (and without ``candidates_dir`` or a cache hit) the
    pipeline stops after glue with ``status="awaiting_translation"``.
    ``candidates_dir`` points at a dir of ``*.c`` candidate bodies: each is
    spliced into a fresh glue copy, built, verified and benched, and the
    fastest passing one wins (:func:`select.pick_winner`).
    ``no_cache`` skips the fingerprint cache; ``cache_dir`` overrides the
    default cache location (same semantics as the Java pipeline).
    """
    from .pipeline import _maybe_write_html

    def err(instructions: str, meth=None) -> dict:
        return {
            "status": "error",
            "lang": "python",
            "backend": "cffi",
            "method": meth,
            "instructions": instructions,
            "generated_at": _utcnow(),
        }

    repo_root = Path(repo)
    if glue_out is None and work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="kp-pipeline-py-"))
        glue_out = work_dir / "glue"
    elif work_dir is None:
        work_dir = Path(glue_out) / ".kp-work"
    elif glue_out is None:
        glue_out = Path(work_dir) / "glue"
    glue_dir = Path(glue_out)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. scan + resolve function
    try:
        r = P.scan_repo(repo_root, subdir)
    except Exception as exc:
        return err(f"Could not scan Python repo {repo_root}: {exc}", method)
    try:
        if method is not None:
            fn = r.functions.get(method)
            if fn is None or fn.nested or fn.is_method:
                return err(
                    f"module-level Python function not found: {method}",
                    method,
                )
        else:
            tried = []
            fn = None
            for c in PM.mine(r):
                if not c.pure:
                    continue
                cand = r.functions.get(c.qualname)
                if cand is None or cand.nested or cand.is_method:
                    continue
                try:
                    C.map_signature(cand)
                except C.CffiGlueError as exc:
                    tried.append(f"{c.qualname} ({exc})")
                    continue
                fn = cand
                break
            if fn is None:
                return err(
                    "No pure Python kernel candidates with C-mappable signatures "
                    f"found in {repo_root}. "
                    + (
                        "Skipped: " + "; ".join(tried) + ". "
                        if tried
                        else ""
                    )
                    + "Annotate parameters (int/float/bool/str/list[int]/list[float]) "
                    "or pass an explicit --method.",
                    None,
                )
    except Exception as exc:  # pragma: no cover - defensive
        return err(f"Function resolution failed: {exc}", method)

    report: dict = {
        "status": "rejected",
        "lang": "python",
        "backend": "cffi",
        "method": fn.qualname,
        "signature": fn.signature,
        "cache_hit": False,
        "fingerprint": None,
        "verify": None,
        "bench": None,
        "candidates": None,
        "candidates_table": None,
        "winner": None,
        "generated_at": _utcnow(),
    }

    # 2. extract + purity gate
    ext = extract_python(r, fn)
    report["purity"] = {"pure": ext["pure"], "reasons": ext["purity_reasons"]}
    if not ext["pure"]:
        report["instructions"] = (
            f"{fn.qualname} is not pure; the pipeline refuses to port it. "
            f"Reasons: {'; '.join(ext['purity_reasons'])}. "
            "Fix the impurity or pick another candidate from "
            "`kernel-port mine --lang python`."
        )
        return report
    if ext["map_error"] is not None:
        report["status"] = "error"
        report["instructions"] = (
            f"Glue generation failed for {fn.qualname}: {ext['map_error']}"
        )
        return report
    sig = C.map_signature(fn)

    # 2b. fingerprint + cache lookup (same semantics as the Java pipeline)
    fingerprint = None
    if not no_cache:
        try:
            fingerprint = cache.fingerprint(ext)
            cdir = Path(cache_dir) if cache_dir is not None else cache.default_cache_dir()
            record = cache.lookup(cdir, fingerprint)
        except Exception:
            record = None
        report["fingerprint"] = fingerprint
        if record:
            report["status"] = "cache_hit"
            report["cache_hit"] = True
            report["translated_body"] = record.get("translated_body")
            report["verify"] = dict(record.get("verify") or {})
            report["bench"] = dict(record.get("bench") or {})
            report["instructions"] = (
                f"Cache hit for {fn.qualname} (fingerprint {fingerprint[:12]}...): "
                "reused the stored translation, verification and benchmark "
                "results; nothing was rebuilt or re-run."
            )
            _maybe_write_html(report, html)
            return report

    # 3. glue
    try:
        files = C.generate([fn], glue_dir)
        native_path = files["c"]  # kernels_cffi.c
    except C.CffiGlueError as exc:
        report["status"] = "error"
        report["instructions"] = f"Glue generation failed for {fn.qualname}: {exc}"
        return report

    # 4. no translation yet -> stop after glue
    if translated_body is None and candidates_dir is None:
        prompt_path = work_dir / "prompt_material.md"
        prompt_path.write_text(render_python_extraction(ext), encoding="utf-8")
        report["status"] = "awaiting_translation"
        report["instructions"] = (
            f"Glue generated in {glue_dir}. The C translation is still missing.\n"
            f"1. Read the translation brief: {prompt_path}\n"
            f"2. Paste the translated C body of {fn.qualname} into {native_path},\n"
            "   replacing everything between the markers\n"
            "   `/* ==== BEGIN TRANSLATED BODY ==== */` and\n"
            "   `/* ==== END TRANSLATED BODY ==== */`.\n"
            "3. Re-run with translated_body=<the C code>, or run "
            "`kernel-port verify --lang python` / `kernel-port bench --lang python` "
            f"with --glue-dir {glue_dir}."
        )
        _maybe_write_html(report, html)
        return report

    if translated_body is not None:
        # 5. splice the translation into the glue
        if isinstance(translated_body, Path):
            translated_body = translated_body.read_text(encoding="utf-8")
        try:
            C.fill_translation(native_path, sig.cname, str(translated_body))
        except (ValueError, KeyError) as exc:
            report["status"] = "error"
            report["instructions"] = f"Could not splice translation: {exc}"
            return report

        # 6. build + verify + bench
        try:
            env = H.build_env(glue_dir, work_dir / "build", sig)
        except RuntimeError as exc:
            report["status"] = "error"
            report["instructions"] = (
                f"Build failed for {fn.qualname}: {exc}\n"
                "The translated body probably does not compile; fix it and retry."
            )
            return report

        vrep = H.run_verify(
            env, sig, repo_root, cases=cases, seed=seed,
            tol_float=tol_float, tol_double=tol_double,
        )
        report["verify"] = _verify_summary(vrep)
        if not vrep.all_passed:
            report["status"] = "verify_failed"
            failing = [c for c in vrep.cases + vrep.boundary_cases if not c.ok and not c.skipped]
            report["instructions"] = (
                f"Verification FAILED: {vrep.failed + vrep.boundary_failed} cases "
                "mismatched. Failing args (first few): "
                + "; ".join(str(c.args) for c in failing[:5])
                + ". Fix the C body and re-run the pipeline."
            )
        else:
            brep = H.run_bench(env, sig, repo_root, iters=iters, warmup=warmup, repeats=repeats)
            report["bench"] = _bench_summary(brep)
            report["status"] = "success"
            verdict = brep.verdict_text("C")
            report["instructions"] = (
                f"Success: {vrep.passed + vrep.boundary_passed} differential cases "
                f"passed; {verdict}."
            )

    # 7. multi-candidate evaluation (mirrors the Java pipeline)
    if candidates_dir is not None:
        report["candidates"] = _run_python_candidates(
            repo_root, fn, sig, glue_dir, work_dir, Path(candidates_dir),
            cases=cases, seed=seed, iters=iters, warmup=warmup, repeats=repeats,
            tol_float=tol_float, tol_double=tol_double,
        )
        try:
            winner = select.pick_winner(report["candidates"])
            winner_name = winner.get("name")
        except ValueError:
            winner_name = None
        table = select.compare_table(report["candidates"])
        report["winner"] = winner_name
        report["candidates_table"] = table
        if translated_body is None:
            # candidates-only run: status derives from the selection
            if winner_name:
                report["status"] = "success"
                report["instructions"] = (
                    f"Winner: {winner_name} passed all verification cases; "
                    f"comparison:\n{table}"
                )
            else:
                report["status"] = "verify_failed"
                report["instructions"] = (
                    "No candidate passed verification; "
                    f"comparison:\n{table}"
                )

    # 8. cache the successful record (same semantics as the Java pipeline)
    if (
        report["status"] == "success"
        and fingerprint
        and translated_body is not None
    ):
        try:
            cdir = Path(cache_dir) if cache_dir is not None else cache.default_cache_dir()
            cache.store(
                cdir,
                fingerprint,
                {
                    "fingerprint": fingerprint,
                    "method": fn.qualname,
                    "signature": fn.signature,
                    "translated_body": str(translated_body),
                    "verify": report["verify"],
                    "bench": report["bench"],
                    "created": _utcnow(),
                },
            )
        except Exception:
            pass  # caching must never fail the pipeline

    _maybe_write_html(report, html)
    return report


def _run_python_candidates(
    repo_root: Path,
    fn,
    sig,
    glue_dir: Path,
    work_dir: Path,
    candidates_dir: Path,
    cases: int,
    seed: int,
    iters: int,
    warmup: int,
    repeats: int,
    tol_float: float = 1e-6,
    tol_double: float = 1e-12,
) -> list[dict]:
    """Build + verify + bench every candidate ``*.c`` file in candidates_dir.

    Each file is treated as an alternative translated C *body*: it is
    spliced into a fresh copy of the cffi glue (unless the file already
    contains the translation markers, in which case it is used as the full
    native source file), gcc-built via :func:`py_harness.build_env`, and
    differentially verified + benched. Returns per-candidate result dicts
    shaped for :func:`select.pick_winner`.
    """
    results: list[dict] = []
    cand_root = work_dir / "candidates"
    cand_files = sorted(candidates_dir.glob("*.c"))
    for cand_file in cand_files:
        name = cand_file.stem
        res: dict = {"name": name}
        cand_glue = cand_root / name
        try:
            files = C.generate([fn], cand_glue)
            native_path = files["c"]  # kernels_cffi.c
            text = cand_file.read_text(encoding="utf-8")
            if C._BEGIN in text and C._END in text:
                native_path.write_text(text, encoding="utf-8")
            else:
                C.fill_translation(native_path, sig.cname, text)
            env = H.build_env(cand_glue, cand_root / name / "build", sig)
        except Exception as exc:
            res["error"] = f"{type(exc).__name__}: {exc}"
            results.append(res)
            continue
        vrep = H.run_verify(
            env, sig, repo_root, cases=cases, seed=seed,
            tol_float=tol_float, tol_double=tol_double,
        )
        res["verify_passed"] = vrep.passed
        res["verify_failed"] = vrep.failed
        res["verify_skipped"] = vrep.skipped
        if vrep.all_passed and vrep.passed > 0:
            brep = H.run_bench(
                env, sig, repo_root, iters=iters, warmup=warmup, repeats=repeats
            )
            res["python_per_call_ns"] = float(brep.python_per_call_ns)
            res["c_per_call_ns"] = float(brep.c_per_call_ns)
            res["speedup"] = float(brep.speedup)
            res["bench_median_ns"] = float(brep.c_per_call_ns)
        results.append(res)
    return results
