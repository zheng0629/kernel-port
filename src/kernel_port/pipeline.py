"""One-shot pipeline: mine -> extract -> glue -> (translate) -> verify -> bench.

``run_pipeline`` drives the whole kernel-port toolchain for one kernel and
returns a single JSON-serializable report dict. The LLM translation step
stays with the calling agent: when no translated C++ body is available the
pipeline stops after glue generation with ``status="awaiting_translation"``
and instructions for the agent.

Optional companion modules (built by parallel workstreams) are imported
lazily with graceful fallback, so the pipeline works whether or not they
exist:

- ``kernel_port.cache``     -- ``fingerprint`` / ``lookup`` / ``store``
- ``kernel_port.select``     -- ``pick_winner`` / ``compare_table``
- ``kernel_port.report``     -- ``render_html``
- ``kernel_port.counterex``  -- ``format_counterexamples``

Report schema
-------------
``run_pipeline(...)`` returns a plain ``dict`` shaped like this (every value
is JSON-serializable)::

    {
        "status": str,            # success | rejected | awaiting_translation
                                  # | verify_failed | cache_hit | error
        "method": str,            # qualified method name
        "signature": str,         # Java signature
        "purity": {"pure": bool, "reasons": [str]},
        "cache_hit": bool,
        "fingerprint": str | None,
        "verify": {"passed": int, "failed": int, "skipped": int} | None,
        "bench": {"java_per_call_ns": float,
                  "jni_per_call_ns": float,
                  "speedup": float} | None,
        "candidates": [           # per-candidate results, or None
            {"name": str,
             "verify_passed": int, "verify_failed": int, "verify_skipped": int,
             "java_per_call_ns": float, "jni_per_call_ns": float,
             "speedup": float, "bench_median_ns": float,
             "error": str}        # only present when that candidate failed
        ],
        "candidates_table": str | None,
        "winner": str | None,     # name of the winning candidate
        "counterexamples_path": str | None,  # prompt_with_counterexamples.md
        "html_path": str | None,
        "work_dir": str,
        "glue_dir": str,
        "instructions": str,      # what the calling agent should do next
        "generated_at": str,     # ISO-8601 timestamp
    }
"""

from __future__ import annotations

import datetime
import importlib
import tempfile
from pathlib import Path

from . import extractor, harness, jni, miner
from . import java_ast as J

__all__ = ["run_pipeline", "splice_body", "PIPELINE_STATUSES"]

PIPELINE_STATUSES = (
    "success",
    "rejected",
    "awaiting_translation",
    "verify_failed",
    "cache_hit",
    "error",
)

_BEGIN_MARKER = "==== BEGIN TRANSLATED BODY ===="
_END_MARKER = "==== END TRANSLATED BODY ===="


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _optional_module(name: str):
    """Import ``kernel_port.<name>`` if it exists, else return None.

    Used for companion modules that parallel workstreams may not have
    created yet; every call site treats ``None`` as "feature skipped".
    """
    try:
        mod = importlib.import_module(f"kernel_port.{name}")
    except ImportError:
        return None
    return mod


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def splice_body(cpp_path: str | Path, body: str) -> None:
    """Splice a translated C++ body into ``kernels_jni.cpp``.

    Everything between the ``BEGIN TRANSLATED BODY`` and
    ``END TRANSLATED BODY`` markers is replaced with ``body``; the markers
    themselves are kept. Raises ``ValueError`` when a marker is missing.

    ``#include`` lines in ``body`` are hoisted: they are inserted after the
    last existing ``#include`` of the generated file instead of the body
    region, so translations may pull in extra headers (e.g. ``<cmath>``).
    """
    cpp_path = Path(cpp_path)
    text = cpp_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    begin = end = None
    for i, line in enumerate(lines):
        if begin is None and _BEGIN_MARKER in line:
            begin = i
        elif begin is not None and _END_MARKER in line:
            end = i
            break
    if begin is None or end is None:
        raise ValueError(
            f"translated-body markers not found in {cpp_path} "
            f"(looked for '{_BEGIN_MARKER}' / '{_END_MARKER}'); "
            "regenerate the glue with `kernel-port glue` first"
        )
    body_lines = body.strip().splitlines()
    includes = sorted(
        {ln.strip() for ln in body_lines if ln.strip().startswith("#include")}
    )
    code_lines = [ln for ln in body_lines if not ln.strip().startswith("#include")]
    if includes:
        last_inc = max(
            (i for i, ln in enumerate(lines) if ln.strip().startswith("#include")),
            default=None,
        )
        at = (last_inc + 1) if last_inc is not None else 0
        lines[at:at] = includes
        # markers shifted down by the inserted lines
        begin += len(includes)
        end += len(includes)
    indent = lines[begin][: len(lines[begin]) - len(lines[begin].lstrip())]
    new_body = [
        (indent + ln.rstrip() if ln.strip() else "") for ln in code_lines
    ]
    lines[begin + 1 : end] = new_body
    out = "\n".join(lines)
    if text.endswith("\n"):
        out += "\n"
    cpp_path.write_text(out, encoding="utf-8")


def _format_counterexamples_md(qualname: str, failing) -> str:
    """Fallback counterexamples prompt when kernel_port.counterex is absent."""
    lines = [
        f"# Counterexamples: `{qualname}`",
        "",
        "The JNI translation below disagrees with the Java original on these "
        "differential-test inputs. Fix the translation so every case matches, "
        "then re-run verification.",
        "",
    ]
    for c in failing:
        lines.append(f"## case {c.index}")
        lines.append(f"- args: `{c.args}`")
        lines.append(f"- detail: {c.detail}")
        lines.append("")
    return "\n".join(lines)


def _write_counterexamples(work_dir: Path, qualname: str, failing_cases) -> str:
    """Write prompt_with_counterexamples.md; return its path as str."""
    counterex = _optional_module("counterex")
    text = None
    if counterex is not None:
        fmt = getattr(counterex, "format_counterexamples", None)
        if callable(fmt):
            try:
                text = fmt(qualname, failing_cases)
            except Exception:
                text = None
    if text is None:
        text = _format_counterexamples_md(qualname, failing_cases)
    path = work_dir / "prompt_with_counterexamples.md"
    work_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text), encoding="utf-8")
    return str(path)


def _pick_winner_and_table(results: list[dict]) -> tuple[str | None, str]:
    select = _optional_module("select")
    if select is not None and hasattr(select, "pick_winner"):
        try:
            winner = select.pick_winner(results)
            winner_name = winner.get("name")
        except Exception:
            winner_name = None
        table_fn = getattr(select, "compare_table", None)
        if callable(table_fn):
            try:
                return winner_name, str(table_fn(results))
            except Exception:
                pass
    # Fallback: replicate select.py semantics (fastest passing candidate).
    eligible = [
        r
        for r in results
        if (r.get("verify_failed") or 0) == 0 and (r.get("verify_passed") or 0) > 0
    ]
    winner_name = None
    if eligible:
        winner_name = min(
            eligible, key=lambda r: r.get("bench_median_ns", float("inf"))
        ).get("name")
    header = f"{'name':<24}{'verify':<14}{'median_ns':>12}  verdict"
    rows = [header, "-" * len(header)]
    for r in results:
        name = str(r.get("name", "?"))
        if name == winner_name:
            verdict = "WINNER *"
        elif (r.get("verify_failed") or 0) > 0 or r.get("error"):
            verdict = "FAILED"
        else:
            verdict = "slower"
        rows.append(
            f"{name:<24}{r.get('verify_passed', 0)}-{r.get('verify_failed', 0):<11}"
            f"{str(r.get('bench_median_ns', '-')):>12}  {verdict}"
        )
    return winner_name, "\n".join(rows)


def _base_report(status: str, method: J.MethodInfo, ext, work_dir: Path, glue_dir: Path) -> dict:
    return {
        "status": status,
        "method": method.qualname,
        "signature": method.signature,
        "purity": {
            "pure": bool(ext.purity.pure),
            "reasons": list(ext.purity.reasons),
        },
        "cache_hit": False,
        "fingerprint": None,
        "verify": None,
        "bench": None,
        "candidates": None,
        "candidates_table": None,
        "winner": None,
        "counterexamples_path": None,
        "html_path": None,
        "work_dir": str(work_dir),
        "glue_dir": str(glue_dir),
        "instructions": "",
        "generated_at": _utcnow(),
    }


def _verify_summary(vrep) -> dict:
    return {
        "passed": vrep.passed,
        "failed": vrep.failed,
        "skipped": vrep.skipped,
    }


def _bench_summary(brep) -> dict:
    return {
        "java_per_call_ns": float(brep.java_per_call_ns),
        "jni_per_call_ns": float(brep.jni_per_call_ns),
        "speedup": float(brep.speedup),
    }


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
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
    cases=50,
    seed=42,
    iters=2000,
    warmup=200,
    repeats=3,
    rigorous=False,
) -> dict:
    """Run mine -> extract -> glue -> verify -> bench for one kernel.

    Parameters mirror the CLI stages: ``repo`` is the Java repo root,
    ``method`` the qualified method name (when ``None`` the top-ranked pure
    candidate from :func:`miner.mine` is used). ``translated_body`` is the
    C++ translation text produced by the calling agent; without it (and
    without a cache hit) the pipeline stops after glue with
    ``status="awaiting_translation"``.

    ``rigorous=True`` requests JMH-style benchmarking; the v0.1 harness only
    does wall-clock timing, so the report records that the request fell back.
    """
    repo_root = Path(repo)

    # Resolve dirs: work_dir defaults under glue_out; everything str()-able.
    if glue_out is None and work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="kp-pipeline-"))
        glue_out = work_dir / "glue"
    elif work_dir is None:
        work_dir = Path(glue_out) / ".kp-work"
    elif glue_out is None:
        glue_out = Path(work_dir) / "glue"
    glue_dir = Path(glue_out)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. scan + resolve method
    try:
        r = J.scan_repo(repo_root, subdir)
    except Exception as exc:
        return {
            "status": "error",
            "method": method,
            "instructions": f"Could not scan repo {repo_root}: {exc}",
            "generated_at": _utcnow(),
        }
    try:
        if method is not None:
            m = r.find_method(method)
            if m is None:
                return {
                    "status": "error",
                    "method": method,
                    "instructions": f"Method not found in repo: {method}",
                    "generated_at": _utcnow(),
                }
        else:
            pure = [c for c in miner.mine(r) if c.pure]
            if not pure:
                return {
                    "status": "error",
                    "method": None,
                    "instructions": (
                        f"No pure kernel candidates found in {repo_root}. "
                        "Pass an explicit method name or fix purity."
                    ),
                    "generated_at": _utcnow(),
                }
            top = pure[0]
            m = r.find_method(top.qualname)
            assert m is not None  # mined from this repo
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "status": "error",
            "method": method,
            "instructions": f"Method resolution failed: {exc}",
            "generated_at": _utcnow(),
        }

    # 2. extract + purity gate
    try:
        ext = extractor.extract(r, m.qualname)
    except ValueError as exc:
        return {
            "status": "error",
            "method": m.qualname,
            "instructions": str(exc),
            "generated_at": _utcnow(),
        }
    report = _base_report("rejected", m, ext, work_dir, glue_dir)
    if not ext.purity.pure:
        report["instructions"] = (
            f"{m.qualname} is not pure; the pipeline refuses to port it. "
            f"Reasons: {'; '.join(ext.purity.reasons)}. "
            "Fix the impurity or pick another candidate from `kernel-port mine`."
        )
        return report

    # 3. fingerprint + cache lookup (graceful when kernel_port.cache absent)
    cache_mod = _optional_module("cache")
    fingerprint = None
    if cache_mod is not None and not no_cache:
        try:
            fingerprint = cache_mod.fingerprint(extractor.to_json(ext))
            cdir = (
                Path(cache_dir)
                if cache_dir is not None
                else cache_mod.default_cache_dir()
            )
            record = cache_mod.lookup(cdir, fingerprint)
        except Exception:
            record = None
        report["fingerprint"] = fingerprint
        if record:
            report["status"] = "cache_hit"
            report["cache_hit"] = True
            report["verify"] = dict(record.get("verify") or {})
            report["bench"] = dict(record.get("bench") or {})
            report["instructions"] = (
                f"Cache hit for {m.qualname} (fingerprint {fingerprint[:12]}...): "
                "reused the stored translation, verification and benchmark "
                "results; nothing was rebuilt or re-run."
            )
            _maybe_write_html(report, html)
            return report

    # 4. glue
    try:
        files = jni.generate(m, glue_dir)
    except ValueError as exc:
        report["status"] = "error"
        report["instructions"] = f"Glue generation failed for {m.qualname}: {exc}"
        return report
    cpp_path = files["cpp"]

    # 5. no translation yet -> stop after glue
    if translated_body is None and candidates_dir is None:
        prompt_path = work_dir / "prompt_material.md"
        prompt_path.write_text(extractor.render_text(ext), encoding="utf-8")
        report["status"] = "awaiting_translation"
        report["instructions"] = (
            f"Glue generated in {glue_dir}. The C++ translation is still missing.\n"
            f"1. Read the translation brief: {prompt_path}\n"
            f"2. Paste the translated body of {m.qualname} into {cpp_path},\n"
            "   replacing everything between the markers\n"
            "   `// ==== BEGIN TRANSLATED BODY ====` and\n"
            "   `// ==== END TRANSLATED BODY ====` (the `TODO(agent)` placeholder).\n"
            "   `#include` lines in your translation are hoisted into the\n"
            "   file's include block automatically.\n"
            "3. Re-run run_pipeline with translated_body=<the C++ code>, "
            "or run `kernel-port verify` / `kernel-port bench` "
            f"with --glue-dir {glue_dir}."
        )
        _maybe_write_html(report, html)
        return report

    if translated_body is not None:
        # 6. splice the translation into the glue
        if isinstance(translated_body, Path):
            translated_body = translated_body.read_text(encoding="utf-8")
        try:
            splice_body(cpp_path, str(translated_body))
        except ValueError as exc:
            report["status"] = "error"
            report["instructions"] = str(exc)
            return report

        # 7. build + verify + bench
        try:
            env = harness.build_env(repo_root, m, glue_dir, work_dir / "build")
        except (RuntimeError, ValueError) as exc:
            report["status"] = "error"
            report["instructions"] = (
                f"Build failed for {m.qualname}: {exc}\n"
                "The translated body probably does not compile; fix it and retry."
            )
            return report

        vrep = harness.run_verify(env, cases=cases, seed=seed)
        report["verify"] = _verify_summary(vrep)

        if not vrep.all_passed:
            report["status"] = "verify_failed"
            failing = [c for c in vrep.cases if not c.ok]
            report["counterexamples_path"] = _write_counterexamples(
                work_dir, m.qualname, failing
            )
            report["instructions"] = (
                f"Verification FAILED: {vrep.failed}/{vrep.passed + vrep.failed} cases "
                f"mismatched. Counterexamples were written to "
                f"{report['counterexamples_path']} -- feed that file back to the "
                "translator, fix the C++ body, and re-run the pipeline."
            )
        else:
            brep = harness.run_bench(env, iters=iters, warmup=warmup, repeats=repeats)
            report["bench"] = _bench_summary(brep)
            report["status"] = "success"
            speed = brep.speedup
            verdict = (
                f"JNI is {speed:.2f}x faster (median per call)"
                if speed >= 1
                else f"JNI is {1/speed:.2f}x slower (median per call)"
            )
            note = (
                " Note: rigorous=True was requested but the v0.1 harness only "
                "supports wall-clock timing; used wall-clock median instead."
                if rigorous
                else ""
            )
            report["instructions"] = (
                f"Success: {vrep.passed} differential cases passed; {verdict}.{note}"
            )

    # 8. multi-candidate evaluation
    if candidates_dir is not None:
        report["candidates"] = _run_candidates(
            repo_root, m, glue_dir, work_dir, Path(candidates_dir),
            cases=cases, seed=seed, iters=iters, warmup=warmup, repeats=repeats,
        )
        winner, table = _pick_winner_and_table(report["candidates"])
        report["winner"] = winner
        report["candidates_table"] = table
        if translated_body is None:
            # candidates-only run: status derives from the selection
            if winner:
                report["status"] = "success"
                report["instructions"] = (
                    f"Winner: {winner} passed all verification cases; "
                    f"comparison:\n{table}"
                )
            else:
                report["status"] = "verify_failed"
                report["instructions"] = (
                    "No candidate passed verification; "
                    f"comparison:\n{table}"
                )

    # 10. cache the successful record
    if report["status"] == "success" and cache_mod is not None and fingerprint:
        try:
            cdir = (
                Path(cache_dir)
                if cache_dir is not None
                else cache_mod.default_cache_dir()
            )
            cache_mod.store(
                cdir,
                fingerprint,
                {
                    "fingerprint": fingerprint,
                    "method": m.qualname,
                    "signature": m.signature,
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


def _run_candidates(
    repo_root: Path,
    method: J.MethodInfo,
    glue_dir: Path,
    work_dir: Path,
    candidates_dir: Path,
    cases: int,
    seed: int,
    iters: int,
    warmup: int,
    repeats: int,
) -> list[dict]:
    """Build + verify + bench every *.cpp in candidates_dir.

    Each file is treated as an alternative translated *body*: it is spliced
    into a fresh copy of the glue (unless the file already contains the
    translation markers, in which case it is used as the full kernels_jni.cpp).
    """
    results: list[dict] = []
    cand_root = work_dir / "candidates"
    for cand_file in sorted(candidates_dir.glob("*.cpp")):
        name = cand_file.stem
        res: dict = {"name": name}
        cand_glue = cand_root / name
        try:
            files = jni.generate(method, cand_glue)
            text = cand_file.read_text(encoding="utf-8")
            if _BEGIN_MARKER in text and _END_MARKER in text:
                files["cpp"].write_text(text, encoding="utf-8")
            else:
                splice_body(files["cpp"], text)
            env = harness.build_env(
                repo_root, method, cand_glue, cand_root / name / ".kp-work"
            )
        except Exception as exc:
            res["error"] = f"{type(exc).__name__}: {exc}"
            results.append(res)
            continue
        vrep = harness.run_verify(env, cases=cases, seed=seed)
        res["verify_passed"] = vrep.passed
        res["verify_failed"] = vrep.failed
        res["verify_skipped"] = vrep.skipped
        if vrep.all_passed and vrep.passed > 0:
            brep = harness.run_bench(env, iters=iters, warmup=warmup, repeats=repeats)
            res["java_per_call_ns"] = float(brep.java_per_call_ns)
            res["jni_per_call_ns"] = float(brep.jni_per_call_ns)
            res["speedup"] = float(brep.speedup)
            res["bench_median_ns"] = float(brep.jni_per_call_ns)
        results.append(res)
    return results


def _maybe_write_html(report: dict, html) -> None:
    if html is None:
        return
    report_mod = _optional_module("report")
    if report_mod is None or not hasattr(report_mod, "render_html"):
        report["instructions"] += (
            "\nHTML report was requested but kernel_port.report is not available; skipped."
        )
        return
    try:
        html_text = report_mod.render_html(report)
    except Exception as exc:  # pragma: no cover - defensive
        report["instructions"] += f"\nHTML rendering failed: {exc}"
        return
    html_path = Path(html)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_text, encoding="utf-8")
    report["html_path"] = str(html_path)
