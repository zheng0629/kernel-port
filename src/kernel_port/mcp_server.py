"""MCP server exposing kernel-port as tools over stdio.

Built on the ``mcp`` Python SDK (v2.x, ``MCPServer``). Each tool is a thin
wrapper around the toolchain that returns a JSON-serializable dict -- the
handlers never print rich console output; rendering is left to the caller.

Tools
-----
- ``mine(repo, dir=None, min_score=0.0)``
- ``extract(repo, method, dir=None)``
- ``glue(repo, method, out_dir, dir=None)``
- ``verify(repo, method, glue_dir, cases=50)``
- ``bench(repo, method, glue_dir)``
- ``pipeline(repo, method=None, translated_body=None, ...)``
- ``repair(repo, method, propose_cmd, ...)``

Every tool returns ``{"ok": True, ...}`` on success or
``{"ok": False, "error": "..."}`` on failure.

Run with ``python -m kernel_port.mcp_server`` or ``serve()``.
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.mcpserver import MCPServer

from . import extractor, harness, jni, miner, parsecache, pipeline as pipeline_mod
from . import java_ast as J
from .purity import check_purity

__all__ = ["server", "serve", "TOOL_NAMES"]

TOOL_NAMES = ("mine", "extract", "glue", "verify", "bench", "pipeline", "repair")

server = MCPServer(
    name="kernel-port",
    title="kernel-port",
    description=(
        "Mine pure-compute Java kernels and port them to native code "
        "(C++ via JNI, C via Panama FFM, or Rust): "
        "mine candidates, extract prompt material, generate FFI glue, "
        "differentially verify translations, and benchmark them."
    ),
)


def _load_repo(repo: str, subdir: str | None):
    root = Path(repo)
    if not root.exists():
        raise ValueError(f"path not found: {repo}")
    return parsecache.load_repo(root, subdir)


def _resolve_method(repo, qualname: str):
    from . import overloading

    try:
        return overloading.parse_method_ref(repo, qualname)
    except (overloading.AmbiguityError, LookupError) as exc:
        raise ValueError(str(exc)) from exc


def _check_pure(repo, m):
    rep = check_purity(m, repo)
    if not rep.pure:
        raise ValueError(f"{m.qualname} is not pure: {'; '.join(rep.reasons)}")
    return rep


def _err(exc: Exception) -> dict:
    return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


@server.tool()
def mine(repo: str, dir: str | None = None, min_score: float = 0.0) -> dict:
    """Scan a Java repo for pure-compute kernel candidates, ranked by value."""
    try:
        r = _load_repo(repo, dir)
        cands = [c for c in miner.mine(r) if c.score >= min_score or not c.pure]
        return {
            "ok": True,
            "repo": repo,
            "candidates": [
                {
                    "method": c.qualname,
                    "signature": c.signature,
                    "file": c.file,
                    "line": c.lineno,
                    "pure": c.pure,
                    "reasons": c.reasons,
                    "calls": c.calls,
                    "intensity": c.intensity,
                    "score": c.score,
                }
                for c in cands
            ],
        }
    except Exception as exc:
        return _err(exc)


@server.tool()
def extract(repo: str, method: str, dir: str | None = None) -> dict:
    """Extract a kernel: purity report, dependency closure, LLM prompt material."""
    try:
        r = _load_repo(repo, dir)
        ext = extractor.extract(r, method)
        data = extractor.to_json(ext)
        data["ok"] = True
        return data
    except Exception as exc:
        return _err(exc)


@server.tool()
def glue(
    repo: str, method: str, out_dir: str, dir: str | None = None,
    backend: str = "jni",
) -> dict:
    """Generate FFI glue (native Java class, native skeleton, build files) for a kernel.

    backend: 'jni' (C++), 'ffm' (C via Panama FFM), or 'rust' (Rust cdylib via the jni crate).
    """
    try:
        if backend not in ("jni", "ffm", "rust"):
            raise ValueError(f"unknown backend {backend!r}: choose 'jni', 'ffm' or 'rust'")
        r = _load_repo(repo, dir)
        m = _resolve_method(r, method)
        _check_pure(r, m)
        if backend == "ffm":
            from . import ffm as _ffm
            files = _ffm.generate(m, Path(out_dir), package=jni.glue_package(r, m))
        elif backend == "rust":
            from . import rust_glue as _rust
            files = _rust.generate(
                m, Path(out_dir), package=jni.glue_package(r, m)
            )
        else:
            files = jni.generate(m, Path(out_dir), package=jni.glue_package(r, m))
        return {
            "ok": True,
            "method": m.qualname,
            "out_dir": str(out_dir),
            "files": {kind: str(p) for kind, p in files.items()},
            "todo_marker": (
                f"{files['cpp']}: between "
                "'// ==== BEGIN TRANSLATED BODY ====' and "
                "'// ==== END TRANSLATED BODY ===='"
            ),
        }
    except Exception as exc:
        return _err(exc)


def _build_env(repo: str, method: str, glue_dir: str, subdir: str | None, backend: str = "jni"):
    if backend not in ("jni", "ffm", "rust"):
        raise ValueError(f"unknown backend {backend!r}: choose 'jni', 'ffm' or 'rust'")
    r = _load_repo(repo, subdir)
    m = _resolve_method(r, method)
    _check_pure(r, m)
    gdir = Path(glue_dir)
    work = gdir / ".kp-work"
    env = harness.build_env(Path(repo), m, gdir, work, backend=backend, repo=r)
    return env


@server.tool()
def verify(
    repo: str, method: str, glue_dir: str, cases: int = 50, dir: str | None = None,
    backend: str = "jni",
) -> dict:
    """Differentially test the native translation in glue_dir against the Java original."""
    try:
        env = _build_env(repo, method, glue_dir, dir, backend)
        rep = harness.run_verify(env, cases=cases, seed=42)
        failing = [
            {"index": c.index, "args": c.args, "detail": c.detail}
            for c in rep.cases
            if not c.ok
        ][:10]
        return {
            "ok": rep.all_passed,
            "method": rep.method,
            "passed": rep.passed,
            "failed": rep.failed,
            "skipped": rep.skipped,
            "all_passed": rep.all_passed,
            "failing_cases": failing,
        }
    except Exception as exc:
        return _err(exc)


@server.tool()
def bench(
    repo: str, method: str, glue_dir: str, dir: str | None = None,
    backend: str = "jni",
) -> dict:
    """Wall-clock benchmark: Java original vs the native translation in glue_dir."""
    try:
        env = _build_env(repo, method, glue_dir, dir, backend)
        rep = harness.run_bench(env, iters=2000, warmup=200, repeats=3)
        return {
            "ok": True,
            "method": rep.method,
            "iters": rep.iters,
            "java_per_call_ns": float(rep.java_per_call_ns),
            "jni_per_call_ns": float(rep.jni_per_call_ns),
            "speedup": float(rep.speedup),
            "verdict": rep.verdict,
        }
    except Exception as exc:
        return _err(exc)


@server.tool()
def pipeline(
    repo: str,
    method: str | None = None,
    translated_body: str | None = None,
    dir: str | None = None,
    glue_out: str | None = None,
    work_dir: str | None = None,
    candidates_dir: str | None = None,
    no_cache: bool = False,
    cache_dir: str | None = None,
    cases: int = 50,
    seed: int = 42,
    iters: int = 2000,
    warmup: int = 200,
    repeats: int = 3,
    backend: str = "jni",
    lang: str = "java",
) -> dict:
    """One-shot mine -> extract -> glue -> verify -> bench for one kernel.

    With method=None the top pure candidate is chosen automatically. Without
    translated_body the run stops after glue with status 'awaiting_translation'.
    backend selects the glue/verify/bench path: 'jni' (default), 'ffm'
    (Panama Foreign Function & Memory), or 'rust'. lang='python' runs the
    Python->C (cffi) pipeline instead (backend must stay 'jni' there; it is
    ignored for Python).
    """
    try:
        if lang == "python":
            from . import py_pipeline

            if backend != "jni":
                return _err(ValueError(
                    "lang='python' does not take a backend (Java-pipeline-only option)."
                ))
            return py_pipeline.run_python_pipeline(
                repo,
                method=method,
                subdir=dir,
                glue_out=glue_out,
                work_dir=work_dir,
                translated_body=translated_body,
                candidates_dir=candidates_dir,
                no_cache=no_cache,
                cache_dir=cache_dir,
                cases=cases,
                seed=seed,
                iters=iters,
                warmup=warmup,
                repeats=repeats,
            )
        if lang != "java":
            return _err(ValueError(f"unknown lang {lang!r}: choose 'java' or 'python'"))
        return pipeline_mod.run_pipeline(
            repo,
            method=method,
            subdir=dir,
            glue_out=glue_out,
            work_dir=work_dir,
            translated_body=translated_body,
            candidates_dir=candidates_dir,
            no_cache=no_cache,
            cache_dir=cache_dir,
            cases=cases,
            seed=seed,
            iters=iters,
            warmup=warmup,
            repeats=repeats,
            backend=backend,
        )
    except Exception as exc:  # pragma: no cover - run_pipeline is defensive
        return _err(exc)


@server.tool()
def repair(
    repo: str,
    method: str,
    propose_cmd: str,
    dir: str | None = None,
    max_rounds: int = 5,
    verify_cases: int = 50,
    backend: str = "jni",
) -> dict:
    """Auto-repair loop: verify -> re-translate (via propose_cmd hook) -> re-verify.

    propose_cmd is a shell command run as
    '<cmd> <prompt_material_file> <round_no> <failures_json>'; the translated
    body is read from its stdout. backend: 'jni' or 'ffm'.
    """
    try:
        from . import repair as _repair

        rep = _repair.repair_loop(
            repo, method, propose_cmd=propose_cmd, max_rounds=max_rounds,
            verify_cases=verify_cases, backend=backend, subdir=dir,
        )
        d = rep.to_dict()
        d["ok"] = d.get("status") == "success"
        return d
    except Exception as exc:
        return _err(exc)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def serve() -> None:
    """Run the MCP server over stdio."""
    import anyio

    anyio.run(server.run_stdio_async)


if __name__ == "__main__":
    serve()
