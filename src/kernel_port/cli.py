"""Command-line interface for kernel-port."""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import __version__, classmine, extractor, ffm, harness, jni, lint, miner, overloading
from . import java_ast as J
from . import parsecache, py_ast, py_mine
from . import repair as repair_mod
from .purity import check_purity

app = typer.Typer(
    help="Mine pure-compute kernels from Java repos; extract, glue (JNI/FFM/Rust), verify and benchmark native translations.",
    no_args_is_help=True,
)
console = Console()


def main() -> None:
    app()


def _warn_lang_ignored(option: str, given: bool, *, detail: str) -> None:
    """Warn (stderr) that a CLI flag is ignored under --lang python.

    Several Java-centric flags (``--clusters``/``--no-clusters``/``--class``)
    used to be silently dropped on the Python path; an explicit flag should
    never vanish without a word. stderr keeps ``--format json`` stdout pure.
    """
    if not given:
        return
    import sys

    print(
        f"Warning: {option} has no effect with --lang python ({detail}); ignoring.",
        file=sys.stderr,
    )


def _load_repo(
    repo: str,
    subdir: str | None,
    use_parse_cache: bool = True,
    refresh_parse_cache: bool = False,
    quiet: bool = False,
) -> J.JavaRepo:
    root = Path(repo)
    if not root.exists():
        console.print(f"[red]Error:[/red] path not found: {escape(repo)}")
        raise typer.Exit(code=1)
    try:
        if use_parse_cache:
            loaded = parsecache.load_repo(root, subdir, refresh=refresh_parse_cache)
            stats = parsecache.cache_stats()
            if stats["files_total"] and not quiet:
                # stderr: keep stdout pure for --format json consumers
                import sys

                print(
                    f"parse cache: {stats['hits']} hit / {stats['misses']} miss "
                    f"({stats['files_total']} files)",
                    file=sys.stderr,
                )
            return loaded
        return J.scan_repo(root, subdir)
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1)


def _resolve_method(repo: J.JavaRepo, qualname: str) -> J.MethodInfo:
    """Resolve '--method', accepting an optional '(types)' overload suffix (v0.4)."""
    try:
        return overloading.parse_method_ref(repo, qualname)
    except overloading.AmbiguityError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1)
    except LookupError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1)


def _failures_path(work: Path, method_qn: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", method_qn)
    return work / ".kp-work" / f"{safe}.failures.json"


def _run_coverage_round(
    repo: str,
    env: harness.BuiltEnv,
    report: harness.VerifyReport,
    cases: int,
    seed: int,
    tol_float: float,
    tol_double: float,
) -> None:
    """Coverage-guided second verify round (v0.6): instrument, measure, guide, merge.

    Mutates ``report`` in place: sets ``guided``/``guided_skipped`` and the
    ``coverage_pct``/``branches_hit``/``branches_total`` fields. Failing
    guided cases are delta-minimized (best-effort, capped) before the
    report is persisted.

    Best-effort by design: any failure in the coverage machinery
    (instrumentation of odd syntax, instrumented-compile errors, harness
    drift) degrades gracefully to the plain verify results with a warning,
    never a crashed command.
    """
    import random as _random

    from . import coverage as covmod

    try:
        _coverage_round_inner(
            repo, env, report, cases, seed, tol_float, tol_double,
            covmod, _random,
        )
    except Exception as exc:  # noqa: BLE001 -- best-effort, see docstring
        console.print(
            "[yellow]Warning:[/yellow] coverage-guided round skipped "
            f"({escape(str(exc))}); falling back to plain verify results."
        )


def _coverage_round_inner(
    repo: str,
    env: harness.BuiltEnv,
    report: harness.VerifyReport,
    cases: int,
    seed: int,
    tol_float: float,
    tol_double: float,
    covmod,
    _random,
) -> None:
    repo_root = Path(repo)
    sources = {
        str(p.relative_to(repo_root)): p.read_text(encoding="utf-8")
        for p in sorted(repo_root.rglob("*.java"))
    }
    if not sources:
        return
    with console.status("Instrumenting for branch coverage..."):
        instrumented = covmod.instrument(sources)
    ran_args = [c.args for c in report.cases + report.boundary]
    # Compile once; replay both case batches through one JVM session (v0.7:
    # this halves the round's fixed javac + JVM-startup cost).
    with covmod.CoveredSession(env.method, instrumented) as sess:
        with console.status("Measuring branch coverage..."):
            sess.replay(ran_args)
            cov = sess.coverage()
        rng = _random.Random(seed + 1)
        guided_args = covmod.guided_cases(env.method, cov, rng, n=max(cases // 2, 10))
        with console.status(f"Running {len(guided_args)} coverage-guided cases..."):
            guided, gskipped = harness.run_cases(
                env, guided_args, tol_float=tol_float, tol_double=tol_double
            )
        # Merge coverage from both rounds (per-branch max), scoped to the
        # tested method's own branches (repo-wide counts would be misleading).
        with console.status("Merging coverage..."):
            sess.replay(guided_args)
            cov2 = sess.coverage()
    merged_hits: dict[int, int] = {}
    for b in cov.branches:
        merged_hits[b.global_id] = max(
            cov.hits_for(b.global_id), cov2.hits_for(b.global_id)
        )
    own = [b for b in cov.branches if b.method == env.method.name]
    total = len(own)
    hit = sum(1 for b in own if merged_hits.get(b.global_id, 0) > 0)
    report.guided = guided
    report.guided_skipped = gskipped
    report.branches_total = total
    report.branches_hit = hit
    report.coverage_pct = (100.0 * hit / total) if total else 100.0
    # Delta-minimize failing guided cases (best-effort, call + wall-clock
    # budgets: each _fails call can block up to the harness timeout).
    to_min = [c for c in guided if not c.ok][:3]
    if to_min:
        with console.status(f"Minimizing {len(to_min)} failing guided case(s)..."):
            s_java, s_jni = harness.open_sessions(env)
            try:
                for c in to_min:
                    def _fails(args, _c=c):
                        line = harness.encode_call(env.method, args)
                        try:
                            rj = s_java.ask(line, 30.0)
                            rn = s_jni.ask(line, 30.0)
                        except (TimeoutError, RuntimeError):
                            return False
                        if not rj.startswith("OK ") or not rn.startswith("OK "):
                            return True
                        ok, _ = harness.results_equal(
                            env.method, rj[3:], rn[3:], tol_float, tol_double
                        )
                        return not ok

                    try:
                        c.minimized_args = covmod.minimize(
                            c.args, _fails, max_calls=200, max_seconds=60.0
                        )
                    except Exception:
                        c.minimized_args = None
            finally:
                s_java.close()
                s_jni.close()


def _mine_python(repo: str, subdir: str | None, fmt: str, min_score: float) -> None:
    """Python kernel mining (v0.6 slice 1: mine/purity only; glue/verify come later)."""
    root = Path(repo)
    if not root.exists():
        console.print(f"[red]Error:[/red] path not found: {escape(repo)}")
        raise typer.Exit(code=1)
    prepo = py_ast.scan_repo(root, subdir)
    cands = [c for c in py_mine.mine(prepo) if c.score >= min_score or not c.pure]
    if fmt == "json":
        print(
            json.dumps(
                {
                    "lang": "python",
                    "functions": [
                        {
                            "function": c.qualname,
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
                },
                indent=2,
            )
        )
        return
    table = Table(title=f"python kernel candidates in {repo}")
    table.add_column("Function", style="cyan")
    table.add_column("Location", style="dim")
    table.add_column("Pure", justify="center")
    table.add_column("Calls", justify="right")
    table.add_column("Score", justify="right")
    for c in cands:
        pure = "[green]yes[/green]" if c.pure else "[red]no[/red]"
        table.add_row(c.qualname, f"{c.file}:{c.lineno}", pure, str(c.calls), f"{c.score:.1f}")
    console.print(table)
    impure = [c for c in cands if not c.pure]
    if impure:
        console.print("\n[dim]Rejected (impure) -- first reason shown:[/dim]")
        for c in impure[:10]:
            reason = c.reasons[0] if c.reasons else "unknown"
            console.print(f"  [dim]{escape(c.qualname)}: {escape(reason)}[/dim]")
    console.print(
        "\n[dim]Next: [cyan]kernel-port glue --lang python --method <func> --out <dir>[/cyan], "
        "fill the translated C body, then [cyan]verify --lang python[/cyan] / "
        "[cyan]bench --lang python[/cyan] with [cyan]--glue-dir <dir>[/cyan].[/dim]"
    )


# ---------------------------------------------------------------------------
# v0.7 python->c slice (cffi backend): thin CLI dispatch.
# All logic lives in cffi_glue.py / py_harness.py; these helpers only scan,
# resolve, purity-check and print. The Java paths below are untouched.
# ---------------------------------------------------------------------------


def _scan_python_repo(repo: str, subdir: str | None):
    """Scan a Python repo dir (tree-sitter); exit 1 if unreadable."""
    from . import py_ast as _pyast

    root = Path(repo)
    if not root.exists():
        console.print(f"[red]Error:[/red] path not found: {escape(repo)}")
        raise typer.Exit(code=1)
    try:
        return _pyast.scan_repo(root, subdir)
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1)


def _resolve_py_function(prepo, method: str):
    """Resolve --method to a module-level Python function; exit 1 if missing."""
    fn = prepo.functions.get(method)
    if fn is None or fn.nested or fn.is_method:
        console.print(
            f"[red]Error:[/red] module-level Python function not found: {escape(method)}"
        )
        raise typer.Exit(code=1)
    return fn


def _check_py_pure(prepo, fn) -> None:
    """Refuse impure Python kernels (exit 2), mirroring the Java path."""
    from . import py_mine

    rep = py_mine.check_purity(fn, prepo)
    if not rep.pure:
        console.print(f"[red]Refusing:[/red] {escape(fn.qualname)} is not pure:")
        for reason in rep.reasons:
            console.print(f"  - {reason}", markup=False)
        raise typer.Exit(code=2)


def _glue_python(repo: str, method: str | None, out: str, subdir: str | None) -> None:
    from . import cffi_glue

    if not method:
        console.print(
            "[red]Error:[/red] --method is required with --lang python "
            "(e.g. --method cffi_kernels.dot)."
        )
        raise typer.Exit(code=1)
    prepo = _scan_python_repo(repo, subdir)
    fn = _resolve_py_function(prepo, method)
    _check_py_pure(prepo, fn)
    try:
        files = cffi_glue.generate([fn], Path(out))
    except cffi_glue.CffiGlueError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1)
    console.print(f"[green]cffi glue generated for {escape(fn.qualname)}:[/green]")
    for kind, path in files.items():
        console.print(f"  {kind:5s} {path}", markup=False)
    console.print(
        "\nNext: the agent pastes the translated C body into the "
        "[cyan]BEGIN TRANSLATED BODY[/cyan] spot in kernels_cffi.c, then runs "
        "[cyan]kernel-port verify --lang python[/cyan] / [cyan]bench --lang python[/cyan] "
        f"with [cyan]--glue-dir {out}[/cyan]."
    )


def _verify_python(
    repo: str,
    method: str,
    glue_dir: str,
    cases: int,
    seed: int,
    boundary: bool,
    work: str | None,
    subdir: str | None,
    tol_float: float,
    tol_double: float,
    coverage: bool = False,
) -> None:
    from . import cffi_glue, py_harness

    prepo = _scan_python_repo(repo, subdir)
    fn = _resolve_py_function(prepo, method)
    _check_py_pure(prepo, fn)
    try:
        sig = cffi_glue.map_signature(fn)
    except cffi_glue.CffiGlueError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1)
    gdir = Path(glue_dir)
    work_dir = Path(work) if work else gdir / ".kp-work"
    try:
        with console.status("Compiling C (gcc -shared -fPIC -O2)..."):
            env = py_harness.build_env(gdir, work_dir, sig)
    except RuntimeError as exc:
        console.print(f"[red]Build failed:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1)
    with console.status(f"Running {cases} differential cases..."):
        try:
            report = py_harness.run_verify(
                env, sig, Path(repo), cases=cases, seed=seed,
                boundary=boundary, tol_float=tol_float, tol_double=tol_double,
                coverage=coverage,
            )
        except RuntimeError as exc:
            console.print(f"[red]Verify failed:[/red] {escape(str(exc))}")
            raise typer.Exit(code=1)
    skip_note = f" ({report.skipped} out-of-domain inputs skipped)" if report.skipped else ""
    bskip = f", {report.boundary_skipped} boundary skipped" if report.boundary_skipped else ""
    n_guided = report.guided_passed + report.guided_failed + report.guided_skipped
    guided_note = (
        f", guided {report.guided_passed}/{n_guided}" if coverage and n_guided else ""
    )
    if report.all_passed:
        console.print(
            f"[green]PASS[/green] random {report.passed - report.guided_passed}/"
            f"{len(report.cases) - n_guided}"
            + (f", boundary {report.boundary_passed}/{len(report.boundary_cases)}" if boundary else "")
            + guided_note
            + f"{skip_note}{bskip} -- {escape(report.method)} cffi translation matches Python."
        )
        if coverage and report.coverage is not None:
            console.print(
                f"[dim]branch coverage of Python original: {report.coverage.coverage_pct:.1f}% "
                f"({report.coverage.branches_hit}/{report.coverage.branches_total} branches)[/dim]"
            )
        return
    console.print(
        f"[red]FAIL[/red] {report.passed + report.boundary_passed} passed, "
        f"{report.failed + report.boundary_failed} mismatches "
        f"(random {report.failed - report.guided_failed}, boundary {report.boundary_failed}"
        + (f", guided {report.guided_failed}" if coverage and n_guided else "")
        + "):"
    )
    shown = 0
    for c in report.cases + report.boundary_cases:
        if not c.ok and not c.skipped:
            tag = "guided" if c.guided else ("boundary" if c.boundary else "random")
            console.print(escape(f"  [{tag}] case {c.index}: {c.detail}"))
            shown += 1
            if shown >= 10:
                break
    raise typer.Exit(code=1)


def _bench_python(
    repo: str,
    method: str,
    glue_dir: str,
    iters: int,
    warmup: int,
    repeats: int,
    work: str | None,
    subdir: str | None,
    list_len: int = 64,
    sizes: list[int] | None = None,
) -> None:
    from . import cffi_glue, py_harness

    prepo = _scan_python_repo(repo, subdir)
    fn = _resolve_py_function(prepo, method)
    _check_py_pure(prepo, fn)
    try:
        sig = cffi_glue.map_signature(fn)
    except cffi_glue.CffiGlueError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1)
    gdir = Path(glue_dir)
    work_dir = Path(work) if work else gdir / ".kp-work"
    try:
        with console.status("Compiling C (gcc -shared -fPIC -O2)..."):
            env = py_harness.build_env(gdir, work_dir, sig)
    except RuntimeError as exc:
        console.print(f"[red]Build failed:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1)

    def fmt_ns(ns: float) -> str:
        if ns >= 1e9:
            return f"{ns / 1e9:.2f}s"
        if ns >= 1e6:
            return f"{ns / 1e6:.2f}ms"
        if ns >= 1e3:
            return f"{ns / 1e3:.2f}us"
        return f"{ns:.0f}ns"

    def bench_failed(exc: Exception) -> None:
        console.print(f"[red]Benchmark failed:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1)

    if sizes:
        # --sizes: mirror the Java break-even sweep -- one bench per input
        # size, then a per-size table plus the break-even verdict.
        rows: list[tuple[int, object]] = []
        with console.status("Benchmarking across sizes..."):
            for size in sorted(set(sizes)):
                try:
                    rep = py_harness.run_bench(
                        env, sig, Path(repo), iters=iters, warmup=warmup,
                        repeats=repeats, list_len=size,
                    )
                except RuntimeError as exc:
                    console.print(
                        f"[yellow]size {size}: skipped "
                        f"({escape(str(exc))})[/yellow]"
                    )
                    continue
                rows.append((size, rep))
        if not rows:
            console.print("[red]Benchmark failed:[/red] no input size produced a result.")
            raise typer.Exit(code=1)
        stable = Table(
            title=f"benchmark by input size: {sig.qualname} "
                  f"({iters} iters x {repeats} repeats)"
        )
        stable.add_column("Size", justify="right")
        stable.add_column("Python/call", justify="right")
        stable.add_column("C (cffi)/call", justify="right")
        stable.add_column("Speedup", justify="right")
        per_size: list[dict] = []
        for size, rep in rows:
            sp = rep.speedup
            sp_s = f"[green]{sp:.2f}x[/green]" if sp >= 1 else f"[red]{sp:.2f}x[/red]"
            stable.add_row(
                str(size),
                fmt_ns(rep.python_per_call_ns),
                fmt_ns(rep.c_per_call_ns),
                sp_s,
            )
            per_size.append({"size": size, "speedup": sp})
        console.print(stable)
        break_even = harness._break_even_size(per_size)
        if break_even is not None:
            console.print(
                f"[green]Break-even:[/green] porting pays off at input size "
                f">= {break_even} (speedup > 1.2x)."
            )
        else:
            console.print(
                "[yellow]Break-even:[/yellow] native never pays off in the measured sizes."
            )
        return

    with console.status("Benchmarking..."):
        try:
            report = py_harness.run_bench(
                env, sig, Path(repo), iters=iters, warmup=warmup, repeats=repeats,
                list_len=list_len,
            )
        except RuntimeError as exc:
            bench_failed(exc)

    table = Table(title=f"benchmark: {report.method} ({report.iters} iters x {report.repeats} repeats)")
    table.add_column("Side")
    table.add_column("Median/call", justify="right")
    table.add_row("Python", fmt_ns(report.python_per_call_ns))
    table.add_row("C (cffi)", fmt_ns(report.c_per_call_ns))
    console.print(table)
    speed = report.speedup
    vtext = report.verdict_text()
    if report.verdict == "faster":
        console.print(f"[green]{vtext}[/green]")
    elif report.verdict == "slower":
        console.print(f"[yellow]{vtext}[/yellow]")
    else:
        console.print(f"[dim]{vtext}[/dim]")


# ---------------------------------------------------------------------------
# mine
# ---------------------------------------------------------------------------


@app.command()
def mine(
    repo: str = typer.Argument(..., help="Path to the Java repo (or module) to scan."),
    dir: str | None = typer.Option(
        None, "--dir", help="Only scan this subdirectory of the repo."
    ),
    format: str = typer.Option("table", "--format", help="'table' or 'json'."),
    min_score: float = typer.Option(0.0, "--min-score", help="Hide candidates below this score."),
    clusters: bool | None = typer.Option(
        None, "--clusters/--no-clusters", help="Show function-cluster candidates (entry + pure helpers). Java only."
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="async-profiler collapsed-stack file; re-rank by measured CPU time.",
    ),
    lang: str = typer.Option(
        "java", "--lang", help="'java' (default) or 'python' (tree-sitter Python mining).",
    ),
    no_parse_cache: bool = typer.Option(
        False, "--no-parse-cache", help="Bypass the incremental parse cache (Java only)."
    ),
    refresh_parse_cache: bool = typer.Option(
        False, "--refresh-parse-cache", help="Re-parse everything and rewrite the parse cache (Java only)."
    ),
) -> None:
    """Scan for pure-compute kernel candidates, ranked by value."""
    fmt = format.lower()
    if fmt not in ("table", "json"):
        console.print("[red]Error:[/red] --format must be 'table' or 'json'")
        raise typer.Exit(code=1)
    lang = lang.lower()
    if lang not in ("java", "python"):
        console.print("[red]Error:[/red] --lang must be 'java' or 'python'")
        raise typer.Exit(code=1)
    if lang == "python":
        _warn_lang_ignored(
            "--clusters/--no-clusters",
            clusters is not None,
            detail="function clusters are only mined from Java call graphs",
        )
        _mine_python(repo, dir, fmt, min_score)
        return
    r = _load_repo(repo, dir, use_parse_cache=not no_parse_cache,
                   refresh_parse_cache=refresh_parse_cache,
                   quiet=(fmt == "json"))

    cands = miner.mine(r)
    show_clusters = True if clusters is None else clusters
    cluster_list = miner.find_clusters(r) if show_clusters else []
    # Members of pure clusters are represented by the cluster row; members of
    # impure clusters stay listed individually (still individually portable).
    hidden = {m for cl in cluster_list if cl.pure for m in cl.members} if show_clusters else set()
    cands = [c for c in cands if c.qualname not in hidden and (c.score >= min_score or not c.pure)]
    cluster_list = [cl for cl in cluster_list if cl.score >= min_score or not cl.pure]
    class_cands = classmine.find_class_candidates(r)

    cpu_map: dict[str, float] = {}
    if profile:
        from . import profile as prof

        try:
            cpu_map = prof.parse_collapsed(profile)
        except (OSError, ValueError) as exc:
            console.print(f"[red]Error:[/red] cannot parse profile: {escape(str(exc))}")
            raise typer.Exit(code=1)
        cands = prof.rerank(cands, cpu_map)
        cluster_list = prof.rerank(cluster_list, cpu_map)

    if fmt == "json":
        # plain print: rich console would wrap long lines and corrupt JSON
        print(
            json.dumps(
                {
                    "methods": [
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
                            "profile_cpu_s": getattr(c, "profile_cpu_s", None),
                        }
                        for c in cands
                    ],
                    "clusters": [
                        {
                            "entry": cl.entry,
                            "members": cl.members,
                            "member_count": len(cl.members),
                            "pure": cl.pure,
                            "reasons": cl.reasons,
                            "calls": cl.calls,
                            "score": cl.score,
                            "profile_cpu_s": getattr(cl, "profile_cpu_s", None),
                        }
                        for cl in cluster_list
                    ],
                    "classes": [
                        {
                            "class": cc.qualname,
                            "kind": cc.kind,
                            "portable": cc.portable,
                            "reasons": cc.reasons,
                            "methods": cc.methods,
                            "score": cc.score,
                        }
                        for cc in class_cands
                    ],
                },
                indent=2,
            )
        )
        return

    # -- method table -------------------------------------------------------
    table = Table(title=f"kernel candidates in {repo}")
    table.add_column("Method", style="cyan")
    table.add_column("Location", style="dim")
    table.add_column("Pure", justify="center")
    table.add_column("Calls", justify="right")
    if cpu_map:
        table.add_column("CPU s", justify="right")
    table.add_column("Score", justify="right")
    for c in cands:
        pure = "[green]yes[/green]" if c.pure else "[red]no[/red]"
        row = [c.qualname, f"{c.file}:{c.lineno}", pure, str(c.calls)]
        if cpu_map:
            row.append(f"{getattr(c, 'profile_cpu_s', 0.0):.2f}")
        row.append(f"{c.score:.1f}")
        table.add_row(*row)
    console.print(table)
    impure = [c for c in cands if not c.pure]
    if impure:
        console.print("\n[dim]Rejected (impure) -- first reason shown:[/dim]")
        for c in impure[:10]:
            reason = c.reasons[0] if c.reasons else "unknown"
            console.print(f"  [dim]{escape(c.qualname)}: {escape(reason)}[/dim]")
        if len(impure) > 10:
            console.print(f"  [dim]... and {len(impure) - 10} more[/dim]")

    # -- cluster table ------------------------------------------------------
    if show_clusters and cluster_list:
        ctable = Table(title="function clusters (entry + pure helpers, ported as one unit)")
        ctable.add_column("Entry", style="cyan")
        ctable.add_column("Members", justify="right")
        ctable.add_column("Pure", justify="center")
        ctable.add_column("Calls", justify="right")
        ctable.add_column("Score", justify="right")
        for cl in cluster_list:
            pure = "[green]yes[/green]" if cl.pure else "[red]no[/red]"
            ctable.add_row(cl.entry, str(len(cl.members)), pure, str(cl.calls), f"{cl.score:.1f}")
        console.print()
        console.print(ctable)
        console.print("[dim]Members: " + escape("; ".join(
            f"{cl.entry} <- {', '.join(m for m in cl.members[1:])}" for cl in cluster_list[:5]
        )) + "[/dim]")

    # -- class table --------------------------------------------------------
    if class_cands:
        ktable = Table(title="class candidates (ported as a unit)")
        ktable.add_column("Class", style="cyan")
        ktable.add_column("Kind", justify="center")
        ktable.add_column("Portable", justify="center")
        ktable.add_column("Methods", justify="right")
        ktable.add_column("Score", justify="right")
        for cc in class_cands:
            port = "[green]yes[/green]" if cc.portable else "[red]no[/red]"
            ktable.add_row(cc.qualname, cc.kind, port, str(len(cc.methods)), f"{cc.score:.1f}")
        console.print()
        console.print(ktable)
        rejected = [cc for cc in class_cands if not cc.portable]
        if rejected:
            console.print("\n[dim]Rejected classes:[/dim]")
            for cc in rejected[:5]:
                reason = cc.reasons[0] if cc.reasons else "unknown"
                console.print(f"  [dim]{escape(cc.qualname)}: {escape(reason)}[/dim]")


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


@app.command()
def extract(
    repo: str = typer.Argument(..., help="Path to the Java repo."),
    method: str | None = typer.Option(None, "--method", help="Qualified method name, e.g. com.example.Kernels.dot."),
    class_qn: str | None = typer.Option(None, "--class", help="Qualified class name (class-level extraction)."),
    dir: str | None = typer.Option(None, "--dir", help="Only scan this subdirectory of the repo."),
    format: str = typer.Option("text", "--format", help="'text' or 'json'."),
    with_counterexamples: bool = typer.Option(
        False,
        "--with-counterexamples",
        help="Append failing verify cases (from --work) to the LLM prompt material.",
    ),
    work: str | None = typer.Option(
        None, "--work", help="Work dir used by verify (to find failure records)."
    ),
    glue_dir: str | None = typer.Option(
        None, "--glue-dir", help="Glue dir; implies --work <glue-dir>/.kp-work for counterexamples."
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ignore the kernel fingerprint cache."),
    cache_dir: str | None = typer.Option(None, "--cache-dir", help="Cache dir (default ~/.cache/kernel-port)."),
    no_parse_cache: bool = typer.Option(
        False, "--no-parse-cache", help="Bypass the incremental parse cache."
    ),
    refresh_parse_cache: bool = typer.Option(
        False, "--refresh-parse-cache", help="Re-parse everything and rewrite the parse cache."
    ),
    backend: str = typer.Option(
        "jni", "--backend", help="Target backend for the translation notes: 'jni', 'ffm' or 'rust'."
    ),
) -> None:
    """Extract a kernel + dependency closure + purity report (LLM prompt material)."""
    fmt = format.lower()
    if fmt not in ("text", "json"):
        console.print("[red]Error:[/red] --format must be 'text' or 'json'")
        raise typer.Exit(code=1)
    if bool(method) == bool(class_qn):
        console.print("[red]Error:[/red] pass exactly one of --method or --class.")
        raise typer.Exit(code=1)
    r = _load_repo(repo, dir, use_parse_cache=not no_parse_cache,
                       refresh_parse_cache=refresh_parse_cache,
                       quiet=(fmt == "json"))

    if class_qn:
        try:
            cext = extractor.extract_class(r, class_qn)
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {escape(str(exc))}")
            raise typer.Exit(code=1)
        if fmt == "json":
            print(json.dumps(extractor.to_json_class(cext), indent=2))
        else:
            # soft_wrap: the extraction is preformatted prompt material; Rich
            # must not reflow its lines (v0.12 dogfood: a wrapped Location:
            # line looked like an empty field when redirected to a file).
            console.print(
                extractor.render_text_class(cext), markup=False, soft_wrap=True
            )
        if not cext.class_purity.portable:
            raise typer.Exit(code=2)
        return

    try:
        ext = extractor.extract(r, method)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1)

    counterex_text = ""
    counterex_data: list[dict] = []
    if with_counterexamples:
        w = work or (str(Path(glue_dir) / ".kp-work") if glue_dir else None)
        if not w:
            console.print("[red]Error:[/red] --with-counterexamples needs --work or --glue-dir.")
            raise typer.Exit(code=1)
        from . import counterex

        # The failures file is written by harness._persist_failures under the
        # *resolved* (bare) qualname; use ext.method.qualname so a v0.4
        # `(types)` suffix on the CLI ref can't desync the filename (v0.9).
        fpath = _failures_path(Path(w), ext.method.qualname)
        counterex_data = counterex.load_failures(fpath)
        if counterex_data:
            counterex_text = "\n\n" + counterex.format_counterexamples(counterex_data, method)
        else:
            console.print(f"[yellow]Note:[/yellow] no failure records at {escape(str(fpath))}.")

    cache_note = ""
    cache_hit: dict | None = None
    if not no_cache:
        from . import cache as kcache

        fp = kcache.fingerprint(extractor.to_json(ext))
        cache_hit = kcache.lookup(Path(cache_dir) if cache_dir else kcache.default_cache_dir(), fp)
        if cache_hit:
            v = cache_hit.get("verify", {})
            b = cache_hit.get("bench", {})
            cache_note = (
                "\n\n## Cache hit — previously verified translation\n"
                f"A verified translation for this kernel is cached "
                f"(verify {v.get('passed', '?')}/{v.get('passed', 0) + v.get('failed', 0)} passed, "
                f"bench speedup {b.get('speedup', '?')}). "
                "Reuse the cached body below instead of re-translating:\n\n"
                "```cpp\n" + str(cache_hit.get("translated_body", "")) + "\n```"
            )

    if fmt == "json":
        data = extractor.to_json(ext)
        data["counterexamples"] = counterex_data
        data["cache_hit"] = bool(cache_hit)
        if cache_hit:
            data["cached"] = {
                "verify": cache_hit.get("verify"),
                "bench": cache_hit.get("bench"),
                "translated_body": cache_hit.get("translated_body"),
            }
        print(json.dumps(data, indent=2))
    else:
        if backend not in ("jni", "ffm", "rust"):
            console.print(f"[red]Error:[/red] unknown backend '{escape(backend)}' (choose 'jni', 'ffm' or 'rust').")
            raise typer.Exit(code=1)
        console.print(
            extractor.render_text(ext, backend) + counterex_text + cache_note,
            markup=False,
            # soft_wrap: the extraction is preformatted prompt material; Rich
            # must not reflow its lines (v0.12 dogfood: a wrapped Location:
            # line looked like an empty field when redirected to a file).
            soft_wrap=True,
        )
    if not ext.purity.pure:
        raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# glue
# ---------------------------------------------------------------------------


@app.command()
def glue(
    repo: str = typer.Argument(..., help="Path to the repo (Java sources; Python package dir with --lang python)."),
    method: str | None = typer.Option(None, "--method", help="Qualified method name."),
    class_qn: str | None = typer.Option(None, "--class", help="Qualified class name (class-level glue)."),
    out: str = typer.Option(..., "--out", help="Directory for generated glue files."),
    dir: str | None = typer.Option(None, "--dir", help="Only scan this subdirectory of the repo."),
    fallback: bool = typer.Option(
        True,
        "--fallback/--no-fallback",
        help="Keep the Java original callable behind a runtime switch (-Dkernelport.native=false).",
    ),
    backend: str = typer.Option(
        "jni", "--backend", help="'jni' (default), 'ffm' (Panama FFM, needs JDK 22+), or 'rust' (needs cargo).",
    ),
    lang: str = typer.Option(
        "java", "--lang", help="'java' (default) or 'python' (Python->C cffi backend).",
    ),
    no_parse_cache: bool = typer.Option(
        False, "--no-parse-cache", help="Bypass the incremental parse cache."
    ),
    refresh_parse_cache: bool = typer.Option(
        False, "--refresh-parse-cache", help="Re-parse everything and rewrite the parse cache."
    ),
) -> None:
    """Generate FFI boilerplate: native Java class, native skeleton, build files."""
    # --- v0.7 python->c slice: dispatch before the Java path (untouched) ---
    lang = lang.lower()
    if lang not in ("java", "python"):
        console.print("[red]Error:[/red] --lang must be 'java' or 'python'")
        raise typer.Exit(code=1)
    if lang == "python":
        if backend != "jni":
            console.print("[red]Error:[/red] --lang python implies the cffi backend; drop --backend.")
            raise typer.Exit(code=1)
        _warn_lang_ignored(
            "--class",
            class_qn is not None,
            detail="Python glue is function-level only; use --method",
        )
        _glue_python(repo, method, out, dir)
        return
    if backend not in ("jni", "ffm", "rust"):
        console.print(f"[red]Error:[/red] unknown backend '{escape(backend)}' (choose 'jni', 'ffm' or 'rust').")
        raise typer.Exit(code=1)
    if bool(method) == bool(class_qn):
        console.print("[red]Error:[/red] pass exactly one of --method or --class.")
        raise typer.Exit(code=1)
    r = _load_repo(repo, dir, use_parse_cache=not no_parse_cache,
                       refresh_parse_cache=refresh_parse_cache)

    if class_qn:
        if backend != "jni":
            console.print(f"[red]Error:[/red] --backend {backend} currently supports --method only.")
            raise typer.Exit(code=1)
        from . import classglue
        from .purity import check_class_portable

        cls = r.classes.get(class_qn)
        if cls is None:
            console.print(f"[red]Error:[/red] class not found: {escape(class_qn)}")
            raise typer.Exit(code=1)
        rep = check_class_portable(cls, r)
        if not rep.portable:
            console.print(f"[red]Refusing:[/red] {escape(class_qn)} is not portable as a unit:")
            for reason in rep.reasons:
                console.print(f"  - {reason}", markup=False)
            raise typer.Exit(code=2)
        try:
            files = classglue.generate_class(r, class_qn, Path(out))
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {escape(str(exc))}")
            raise typer.Exit(code=1)
        console.print(f"[green]Class glue generated for {escape(class_qn)}:[/green]")
        for kind, path in files.items():
            console.print(f"  {kind:5s} {path}", markup=False)
        return

    m = _resolve_method(r, method)
    rep = check_purity(m, r)
    if not rep.pure:
        console.print(f"[red]Refusing:[/red] {escape(m.qualname)} is not pure:")
        for reason in rep.reasons:
            console.print(f"  - {reason}", markup=False)
        raise typer.Exit(code=2)
    if backend == "ffm":
        jdk = ffm.find_jdk()
        if jdk is None:
            ok, msg = ffm.java_version_ok()
            console.print(f"[red]Error:[/red] {escape(str(msg))}")
            raise typer.Exit(code=1)
        console.print(f"[dim]Using JDK {escape(str(jdk))} for the FFM backend.[/dim]")
        try:
            files = ffm.generate(m, Path(out), package=jni.glue_package(r, m))
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {escape(str(exc))}")
            raise typer.Exit(code=1)
        console.print(f"[green]FFM glue generated for {escape(m.qualname)}:[/green]")
        for kind, path in files.items():
            console.print(f"  {kind:5s} {path}", markup=False)
        console.print(
            "\n[dim]Panama FFM backend: no JNI wrapper, the C file exports a plain C ABI "
            "function. Needs JDK 22+ (25 LTS recommended) to compile and run.[/dim]"
        )
        console.print(
            "\nNext: the agent pastes the translated C body into the "
            "[cyan]TODO(agent)[/cyan] spot in kernels_ffm.c, then runs "
            "[cyan]kernel-port verify --backend ffm[/cyan] / [cyan]bench --backend ffm[/cyan] with "
            f"[cyan]--glue-dir {escape(str(out))}[/cyan]."
        )
        return
    if backend == "rust":
        from . import rust_glue

        try:
            files = rust_glue.generate(
                m, Path(out), fallback=fallback, package=jni.glue_package(r, m)
            )
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {escape(str(exc))}")
            raise typer.Exit(code=1)
        except RuntimeError as exc:
            console.print(f"[red]Error:[/red] {escape(str(exc))}")
            raise typer.Exit(code=1)
        console.print(f"[green]Rust glue generated for {escape(m.qualname)}:[/green]")
        for kind, path in files.items():
            console.print(f"  {kind:5s} {path}", markup=False)
        rust_warnings = rust_glue.lint_lib_rs(files["rust"].read_text(encoding="utf-8"))
        if rust_warnings:
            console.print(f"[yellow]Rust FFI lint: {len(rust_warnings)} warning(s):[/yellow]")
            for w in rust_warnings:
                console.print(f"  [yellow]{escape(w.kind)}[/yellow] line {w.line}: {escape(w.message)}")
        else:
            console.print("[dim]Rust FFI lint: clean.[/dim]")
        console.print(
            "\n[dim]Rust backend: the cdylib exports JNI entry points directly "
            "(jni crate), so it loads via the standard JNI harness -- build with "
            "[cyan]cargo build --release[/cyan] (see build_rust.sh), no C++ wrapper needed.[/dim]"
        )
        if fallback:
            console.print(
                "\n[dim]Fallback enabled: the Java original stays callable. "
                "Roll back with -Dkernelport.native=false.[/dim]"
            )
        console.print(
            "\nNext: the agent pastes the translated Rust body into the "
            "[cyan]TODO(agent)[/cyan] spot in src/lib.rs, then runs "
            "[cyan]kernel-port verify --backend rust[/cyan] / [cyan]bench --backend rust[/cyan] with "
            f"[cyan]--glue-dir {escape(str(out))}[/cyan]."
        )
        return
    try:
        files = jni.generate(
            m, Path(out), fallback=fallback, package=jni.glue_package(r, m)
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1)
    console.print(f"[green]Glue generated for {escape(m.qualname)}:[/green]")
    for kind, path in files.items():
        console.print(f"  {kind:5s} {path}", markup=False)
    warnings = lint.lint_glue(files["cpp"].read_text(encoding="utf-8"))
    if warnings:
        console.print(f"[yellow]Glue lint: {len(warnings)} warning(s):[/yellow]")
        for w in warnings:
            console.print(f"  [yellow]{escape(w.kind)}[/yellow] line {w.line}: {escape(w.message)}")
    else:
        console.print("[dim]Glue lint: clean.[/dim]")
    if fallback:
        console.print(
            "\n[dim]Fallback enabled: the Java original stays callable. "
            "Roll back with -Dkernelport.native=false or "
            f"{files['java'].stem}.setUseNative(false).[/dim]"
        )
    console.print(
        "\nNext: the agent pastes the translated C++ body into the "
        "[cyan]TODO(agent)[/cyan] spot in kernels_jni.cpp, then runs "
        "[cyan]kernel-port verify[/cyan] / [cyan]bench[/cyan] with "
        f"[cyan]--glue-dir {escape(str(out))}[/cyan]."
    )


# ---------------------------------------------------------------------------
# verify / bench
# ---------------------------------------------------------------------------


def _build(repo: str, method_qn: str, glue_dir: str, work: str | None, subdir: str | None, backend: str = "jni") -> harness.BuiltEnv:
    r = _load_repo(repo, subdir)
    m = _resolve_method(r, method_qn)
    rep = check_purity(m, r)
    if not rep.pure:
        console.print(f"[red]Refusing:[/red] {escape(m.qualname)} is not pure:")
        for reason in rep.reasons:
            console.print(f"  - {reason}", markup=False)
        raise typer.Exit(code=2)
    gdir = Path(glue_dir)
    work_dir = Path(work) if work else gdir / ".kp-work"
    try:
        with console.status(f"Compiling Java + {backend.upper()}..."):
            env = harness.build_env(Path(repo), m, gdir, work_dir, backend=backend, repo=r)
    except RuntimeError as exc:
        console.print(f"[red]Build failed:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1)
    return env


@app.command()
def verify(
    repo: str = typer.Argument(..., help="Path to the repo (Java sources; Python package dir with --lang python)."),
    method: str = typer.Option(..., "--method", help="Qualified method name."),
    glue_dir: str = typer.Option(..., "--glue-dir", help="Dir with the filled native glue (*Native.java + native sources)."),
    cases: int = typer.Option(100, "--cases", help="Number of random differential cases."),
    seed: int = typer.Option(42, "--seed", help="RNG seed for reproducibility."),
    boundary: bool = typer.Option(
        True, "--boundary/--no-boundary", help="Also run boundary-value cases (empty arrays, NaN, min/max...)."
    ),
    work: str | None = typer.Option(None, "--work", help="Work dir (default: <glue-dir>/.kp-work)."),
    dir: str | None = typer.Option(None, "--dir", help="Only scan this subdirectory of the repo."),
    backend: str = typer.Option("jni", "--backend", help="'jni' (default), 'ffm' (needs JDK 22+) or 'rust' (needs cargo)."),
    tol_float: float = typer.Option(1e-6, "--tol-float", help="Relative tolerance for float comparisons."),
    tol_double: float = typer.Option(1e-12, "--tol-double", help="Relative tolerance for double comparisons."),
    lang: str = typer.Option(
        "java", "--lang", help="'java' (default) or 'python' (Python->C cffi backend).",
    ),
    coverage: bool = typer.Option(
        False, "--coverage/--no-coverage",
        help="Measure branch coverage of the original and run a coverage-guided second input round. "
        "Opt-in: Java replays cases through an instrumented JVM; Python uses sys.monitoring on the "
        "original function (adds seconds; see README).",
    ),
) -> None:
    """Differential-test the native translation against the Java original."""
    # --- v0.7 python->c slice: dispatch before the Java path (untouched) ---
    lang = lang.lower()
    if lang not in ("java", "python"):
        console.print("[red]Error:[/red] --lang must be 'java' or 'python'")
        raise typer.Exit(code=1)
    if lang == "python":
        if backend != "jni":
            console.print("[red]Error:[/red] --lang python implies the cffi backend; drop --backend.")
            raise typer.Exit(code=1)
        _verify_python(repo, method, glue_dir, cases, seed, boundary, work, dir,
                       tol_float, tol_double, coverage)
        return
    if backend not in ("jni", "ffm", "rust"):
        console.print(f"[red]Error:[/red] unknown backend '{escape(backend)}' (choose 'jni', 'ffm' or 'rust').")
        raise typer.Exit(code=1)
    env = _build(repo, method, glue_dir, work, dir, backend)
    with console.status(f"Running {cases} differential cases..."):
        report = harness.run_verify(env, cases=cases, seed=seed, boundary=boundary,
                                    tol_float=tol_float, tol_double=tol_double)
    if coverage:
        _run_coverage_round(repo, env, report, cases=cases, seed=seed,
                            tol_float=tol_float, tol_double=tol_double)
    total_pass = report.passed + report.boundary_passed + report.guided_passed
    total_fail = report.failed + report.boundary_failed + report.guided_failed
    skip_note = f" ({report.skipped} out-of-domain inputs skipped)" if report.skipped else ""
    bskip = f", {report.boundary_skipped} boundary skipped" if report.boundary_skipped else ""
    if report.all_passed:
        console.print(
            f"[green]PASS[/green] random {report.passed}/{len(report.cases)}"
            + (f", boundary {report.boundary_passed}/{len(report.boundary)}" if boundary else "")
            + (f", guided {report.guided_passed}/{len(report.guided)}" if coverage else "")
            + f"{skip_note}{bskip} -- {escape(report.method)} {backend.upper()} translation matches Java."
        )
    else:
        console.print(
            f"[red]FAIL[/red] {total_pass} passed, {total_fail} mismatches "
            f"(random {report.failed}, boundary {report.boundary_failed}"
            + (f", guided {report.guided_failed}" if coverage else "")
            + "):"
        )
        shown = 0
        for c in report.cases + report.boundary + report.guided:
            if not c.ok:
                if c in report.guided:
                    tag = "guided"
                elif c in report.boundary:
                    tag = "boundary"
                else:
                    tag = "random"
                line = f"  [{tag}] case {c.index}: {c.detail}"
                if c.minimized_args is not None:
                    line += f" [minimized: {c.minimized_args!r}]"
                console.print(escape(line))
                shown += 1
                if shown >= 10:
                    break
        fpath = _failures_path(env.work, method)
        if fpath.exists():
            console.print(
                f"\n[dim]Failing cases saved to {escape(str(fpath))} -- re-run "
                f"[cyan]extract --method {escape(method)} --with-counterexamples --work {escape(str(env.work))}[/cyan] "
                "to feed them back into the translation prompt.[/dim]"
            )
        raise typer.Exit(code=1)
    if coverage and report.coverage_pct is not None:
        console.print(
            f"[dim]branch coverage of Java original: {report.coverage_pct:.1f}% "
            f"({report.branches_hit}/{report.branches_total} branches)[/dim]"
        )


@app.command()
def bench(
    repo: str = typer.Argument(..., help="Path to the repo (Java sources; Python package dir with --lang python)."),
    method: str = typer.Option(..., "--method", help="Qualified method name."),
    glue_dir: str = typer.Option(..., "--glue-dir", help="Dir with the filled native glue (*Native.java + native sources)."),
    iters: int = typer.Option(2000, "--iters", help="Timed iterations per repeat."),
    warmup: int = typer.Option(200, "--warmup", help="Warmup iterations (discarded)."),
    repeats: int = typer.Option(5, "--repeats", help="Measurement repeats (median taken)."),
    sizes: str | None = typer.Option(
        None, "--sizes", help="Comma-separated input sizes for break-even analysis, e.g. '16,64,256,1024'."
    ),
    list_len: int = typer.Option(
        64, "--list-len", help="Python bench (--lang python): input list length when --sizes is not given."
    ),
    rigorous: bool = typer.Option(
        False, "--rigorous", help="Use JMH/Google Benchmark when available (falls back to enhanced wall-clock)."
    ),
    html: str | None = typer.Option(None, "--html", help="Write a self-contained HTML report."),
    work: str | None = typer.Option(None, "--work", help="Work dir (default: <glue-dir>/.kp-work)."),
    dir: str | None = typer.Option(None, "--dir", help="Only scan this subdirectory of the repo."),
    backend: str = typer.Option("jni", "--backend", help="'jni' (default), 'ffm' (needs JDK 22+) or 'rust' (needs cargo)."),
    lang: str = typer.Option(
        "java", "--lang", help="'java' (default) or 'python' (Python->C cffi backend).",
    ),
) -> None:
    """Benchmark: Java original vs native translation, with call-overhead split."""
    # --- v0.7 python->c slice: dispatch before the Java path (untouched) ---
    lang = lang.lower()
    if lang not in ("java", "python"):
        console.print("[red]Error:[/red] --lang must be 'java' or 'python'")
        raise typer.Exit(code=1)
    if lang == "python":
        if backend != "jni":
            console.print("[red]Error:[/red] --lang python implies the cffi backend; drop --backend.")
            raise typer.Exit(code=1)
        if rigorous:
            console.print("[red]Error:[/red] --rigorous is Java-only in this slice.")
            raise typer.Exit(code=1)
        if html:
            console.print("[red]Error:[/red] --html is Java-only in this slice.")
            raise typer.Exit(code=1)
        size_list: list[int] | None = None
        if sizes:
            try:
                size_list = [int(s) for s in sizes.split(",")]
            except ValueError:
                console.print("[red]Error:[/red] --sizes must be comma-separated integers, e.g. '16,64,256'.")
                raise typer.Exit(code=1)
            if any(s < 1 for s in size_list):
                console.print("[red]Error:[/red] --sizes entries must be >= 1.")
                raise typer.Exit(code=1)
        if list_len < 1:
            console.print("[red]Error:[/red] --list-len must be >= 1.")
            raise typer.Exit(code=1)
        _bench_python(repo, method, glue_dir, iters, warmup, repeats, work, dir,
                      list_len=list_len, sizes=size_list)
        return
    if list_len != 64:
        import sys

        print(
            "Warning: --list-len has no effect with --lang java (use --sizes); ignoring.",
            file=sys.stderr,
        )
    if backend not in ("jni", "ffm", "rust"):
        console.print(f"[red]Error:[/red] unknown backend '{escape(backend)}' (choose 'jni', 'ffm' or 'rust').")
        raise typer.Exit(code=1)
    env = _build(repo, method, glue_dir, work, dir, backend)
    native_label = {"jni": "JNI C++", "ffm": "FFM C", "rust": "Rust (JNI)"}[backend]
    # Short per-backend labels for the overhead split and per-size tables.
    native_short = {"jni": "JNI", "ffm": "FFM", "rust": "Rust"}[backend]
    native_lang = {"jni": "C++", "ffm": "C", "rust": "Rust"}[backend]
    size_list = [int(s) for s in sizes.split(",")] if sizes else None
    with console.status("Benchmarking..."):
        try:
            report = harness.run_bench(
                env, iters=iters, warmup=warmup, repeats=repeats,
                sizes=size_list, rigorous=rigorous,
            )
        except RuntimeError as exc:
            console.print(f"[red]Benchmark failed:[/red] {escape(str(exc))}")
            raise typer.Exit(code=1)
    import statistics

    table = Table(title=f"benchmark: {report.method} ({report.iters} iters x {repeats} repeats)")
    table.add_column("Side")
    table.add_column("Median/call", justify="right")
    table.add_column("Mean/call", justify="right")
    j_med = report.java_per_call_ns
    n_med = report.jni_per_call_ns
    j_mean = statistics.mean(report.java_nanos) / report.iters
    n_mean = statistics.mean(report.jni_nanos) / report.iters
    table.add_row("Java", harness.fmt_ns(j_med), harness.fmt_ns(j_mean))
    table.add_row(native_label, harness.fmt_ns(n_med), harness.fmt_ns(n_mean))
    console.print(table)

    if report.call_overhead_ns is not None:
        otable = Table(title=f"{native_short} overhead split")
        otable.add_column("Component")
        otable.add_column("ns/call", justify="right")
        otable.add_row(
            "Call overhead (empty native call)",
            harness.fmt_ns(report.call_overhead_ns),
        )
        otable.add_row(
            f"Pure compute in {native_lang} (native - overhead)",
            harness.fmt_ns(report.compute_ns),
        )
        otable.add_row("Java original", harness.fmt_ns(j_med))
        console.print(otable)
        if report.compute_ns > j_med and n_med > j_med:
            console.print(
                f"[yellow]The {native_lang} compute itself is slower than Java -- "
                "the translation, not the call boundary, is the problem.[/yellow]"
            )
        elif n_med > j_med:
            console.print(
                "[yellow]Call overhead dominates -- the kernel is too small; "
                "porting only pays off at larger input sizes (see break-even).[/yellow]"
            )

    if report.per_size:
        stable = Table(title="speedup by input size")
        stable.add_column("Size", justify="right")
        stable.add_column("Java/call", justify="right")
        stable.add_column(f"{native_short}/call", justify="right")
        stable.add_column("Speedup", justify="right")
        for ps in report.per_size:
            sp = ps["speedup"]
            sp_s = f"[green]{sp:.2f}x[/green]" if sp >= 1 else f"[red]{sp:.2f}x[/red]"
            stable.add_row(
                str(ps["size"]),
                harness.fmt_ns(ps["java_ns"]),
                harness.fmt_ns(ps["jni_ns"]),
                sp_s,
            )
        console.print(stable)
        if report.break_even_size is not None:
            console.print(
                f"[green]Break-even:[/green] porting pays off at input size "
                f">= {report.break_even_size} (speedup > 1.2x)."
            )
        else:
            console.print("[yellow]Break-even:[/yellow] native never pays off in the measured sizes.")

    if report.rigorous_note:
        console.print(f"[dim]{escape(report.rigorous_note)}[/dim]")

    speed = report.speedup
    vtext = report.verdict_text(native_label)
    if report.verdict == "faster":
        console.print(f"[green]{vtext}[/green]")
    elif report.verdict == "slower":
        console.print(f"[yellow]{vtext}[/yellow]")
    else:
        console.print(f"[dim]{vtext}[/dim]")

    if html:
        from . import report as kreport

        verdict = report.verdict_text(native_label)
        doc = kreport.render_html(
            {
                "title": "kernel-port bench report",
                "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "method": report.method,
                "status": "success",
                "bench": {
                    "backend": backend,
                    "java_per_call_ns": j_med,
                    "jni_per_call_ns": n_med,
                    "speedup": speed,
                    "jni_overhead_ns": report.jni_overhead_ns,
                    "rust_overhead_ns": report.rust_overhead_ns,
                    "per_size": [
                        {"size": ps["size"], "speedup": ps["speedup"]}
                        for ps in report.per_size
                    ],
                },
                "verdict": verdict,
            }
        )
        Path(html).write_text(doc, encoding="utf-8")
        console.print(f"[green]HTML report written to {escape(str(html))}[/green]")


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


def _pipeline_python(repo, method, dir, translated_body, glue_out, work, html,
                     json_out, cases, seed, tol_float, tol_double,
                     candidates_dir=None, no_cache=False, cache_dir=None,
                     java_only_flags: dict | None = None) -> None:
    """`pipeline --lang python`: one-shot Python->C (cffi) pipeline."""
    from . import py_pipeline

    java_only_flags = java_only_flags or {}
    bad = [
        k for k, v in java_only_flags.items()
        if v not in (None, False) and not (k == "backend" and v == "jni")
    ]
    if bad:
        console.print(
            "[red]Error:[/red] --lang python does not support "
            + ", ".join(f"--{k.replace('_', '-')}" for k in bad)
            + " (Java-pipeline-only options)."
        )
        raise typer.Exit(code=1)
    work_dir = Path(work) if work else Path.cwd() / ".kp-work"
    gout = Path(glue_out) if glue_out else work_dir / "glue"
    with console.status("Running kernel-port pipeline (--lang python)..."):
        try:
            rep = py_pipeline.run_python_pipeline(
                repo,
                method=method,
                subdir=dir,
                glue_out=gout,
                work_dir=work_dir,
                translated_body=Path(translated_body) if translated_body else None,
                candidates_dir=Path(candidates_dir) if candidates_dir else None,
                html=Path(html) if html else None,
                no_cache=no_cache,
                cache_dir=Path(cache_dir) if cache_dir else None,
                cases=cases,
                seed=seed,
                tol_float=tol_float,
                tol_double=tol_double,
            )
        except Exception as exc:  # noqa: BLE001 - report pipeline errors cleanly
            console.print(f"[red]Pipeline failed:[/red] {escape(str(exc))}")
            raise typer.Exit(code=1)

    status = rep.get("status", "error")
    console.print(f"[bold]pipeline status:[/bold] {status}")
    if rep.get("method"):
        console.print(f"  function: {escape(str(rep['method']))}")
    purity = rep.get("purity", {})
    if purity:
        console.print(f"  purity:  {'PURE' if purity.get('pure') else 'IMPURE'}")
    if rep.get("cache_hit"):
        console.print("  [cyan]cache hit:[/cyan] reused verified translation + results.")
    v = rep.get("verify") or {}
    if v:
        console.print(
            f"  verify:  {v.get('passed', '?')} passed / {v.get('failed', '?')} failed"
            f" (skipped {v.get('skipped', 0)}, boundary "
            f"{v.get('boundary_passed', '?')}/{v.get('boundary_failed', '?')})"
        )
    b = rep.get("bench") or {}
    if b and b.get("speedup") is not None:
        sp = b["speedup"]
        console.print(
            f"  bench:   C {sp:.2f}x vs Python "
            f"({'faster' if sp >= 1 else 'SLOWER'})"
        )
    if rep.get("winner"):
        console.print(f"  winner:  [green]{escape(str(rep['winner']))}[/green]")
    if rep.get("html_path"):
        console.print(f"  html:    {escape(str(rep['html_path']))}")
    instructions = rep.get("instructions")
    if instructions:
        console.print(f"\n{instructions}", markup=False)

    if json_out:
        Path(json_out).write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
        console.print(f"[dim]JSON report written to {escape(str(json_out))}[/dim]")

    if status in ("success", "cache_hit", "awaiting_translation"):
        raise typer.Exit(code=0)
    if status == "verify_failed":
        raise typer.Exit(code=1)
    raise typer.Exit(code=2)


@app.command()
def pipeline(
    repo: str = typer.Argument(..., help="Path to the Java repo."),
    method: str | None = typer.Option(None, "--method", help="Qualified method name (default: top pure candidate)."),
    dir: str | None = typer.Option(None, "--dir", help="Only scan this subdirectory of the repo."),
    translated_body: str | None = typer.Option(
        None, "--translated-body", help="File with the agent's native translation (C++ for JNI, C for FFM, Rust for rust) spliced into the glue."
    ),
    candidates_dir: str | None = typer.Option(
        None, "--candidates-dir", help="Dir of candidate bodies (*.cpp for JNI, *.c for FFM, *.rs for rust); verify+bench each, pick the winner."
    ),
    glue_out: str | None = typer.Option(None, "--glue-out", help="Glue output dir (default: <work>/glue)."),
    work: str | None = typer.Option(None, "--work", help="Work dir (default: ./.kp-work)."),
    html: str | None = typer.Option(None, "--html", help="Write a self-contained HTML report."),
    json_out: str | None = typer.Option(None, "--json", help="Write the machine-readable report as JSON."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Ignore the kernel fingerprint cache."),
    cache_dir: str | None = typer.Option(None, "--cache-dir", help="Cache dir (default ~/.cache/kernel-port)."),
    cases: int = typer.Option(50, "--cases", help="Differential cases per candidate."),
    seed: int = typer.Option(42, "--seed", help="RNG seed."),
    rigorous: bool = typer.Option(False, "--rigorous", help="Rigorous bench mode when frameworks are available."),
    tol_float: float = typer.Option(1e-6, "--tol-float", help="Relative tolerance for float comparisons."),
    tol_double: float = typer.Option(1e-12, "--tol-double", help="Relative tolerance for double comparisons."),
    backend: str = typer.Option("jni", "--backend", help="'jni' (default), 'ffm' (Panama FFM, needs JDK 22+) or 'rust' (needs cargo)."),
    lang: str = typer.Option("java", "--lang", help="'java' (default) or 'python' (Python->C via cffi)."),
) -> None:
    """One-shot mine -> extract -> glue -> verify -> bench for a kernel.

    The native translation (C++/C/Rust) itself stays with the calling
    agent: pass it via --translated-body, or run once without it to get
    the glue + prompt material, translate, then re-run.
    """
    from . import pipeline as kpipeline

    if lang not in ("java", "python"):
        console.print(f"[red]Error:[/red] unknown --lang '{escape(lang)}' (choose 'java' or 'python').")
        raise typer.Exit(code=1)
    if lang == "python":
        _pipeline_python(
            repo, method, dir, translated_body, glue_out, work, html, json_out,
            cases, seed, tol_float, tol_double,
            candidates_dir=candidates_dir, no_cache=no_cache, cache_dir=cache_dir,
            java_only_flags={
                "rigorous": rigorous, "backend": backend,
            },
        )
        return

    if backend not in ("jni", "ffm", "rust"):
        console.print(f"[red]Error:[/red] unknown backend '{escape(backend)}' (choose 'jni', 'ffm' or 'rust').")
        raise typer.Exit(code=1)
    work_dir = Path(work) if work else Path.cwd() / ".kp-work"
    gout = Path(glue_out) if glue_out else work_dir / "glue"
    with console.status("Running kernel-port pipeline..."):
        try:
            rep = kpipeline.run_pipeline(
                repo,
                method=method,
                subdir=dir,
                glue_out=gout,
                work_dir=work_dir,
                translated_body=Path(translated_body) if translated_body else None,
                candidates_dir=Path(candidates_dir) if candidates_dir else None,
                html=Path(html) if html else None,
                no_cache=no_cache,
                cache_dir=Path(cache_dir) if cache_dir else None,
                cases=cases,
                seed=seed,
                rigorous=rigorous,
                tol_float=tol_float,
                tol_double=tol_double,
                backend=backend,
            )
        except Exception as exc:  # noqa: BLE001 - report pipeline errors cleanly
            console.print(f"[red]Pipeline failed:[/red] {escape(str(exc))}")
            raise typer.Exit(code=1)

    status = rep.get("status", "error")
    console.print(f"[bold]pipeline status:[/bold] {status}")
    if rep.get("method"):
        console.print(f"  method:  {escape(str(rep['method']))}")
    purity = rep.get("purity", {})
    if purity:
        console.print(f"  purity:  {'PURE' if purity.get('pure') else 'IMPURE'}")
    if rep.get("cache_hit"):
        console.print("  [cyan]cache hit:[/cyan] reused verified translation + results.")
    v = rep.get("verify") or {}
    if v:
        console.print(
            f"  verify:  {v.get('passed', '?')} passed / {v.get('failed', '?')} failed"
            f" (skipped {v.get('skipped', 0)})"
        )
    b = rep.get("bench") or {}
    if b and b.get("speedup") is not None:
        sp = b["speedup"]
        console.print(
            f"  bench:   native {sp:.2f}x vs Java "
            f"({'faster' if sp >= 1 else 'SLOWER'})"
        )
    if rep.get("winner"):
        console.print(f"  winner:  [green]{escape(str(rep['winner']))}[/green]")
    if rep.get("html_path"):
        console.print(f"  html:    {escape(str(rep['html_path']))}")
    if rep.get("counterexamples_path"):
        console.print(f"  counterexamples: {escape(str(rep['counterexamples_path']))}")
    instructions = rep.get("instructions")
    if instructions:
        console.print(f"\n{instructions}", markup=False)

    if json_out:
        Path(json_out).write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
        console.print(f"[dim]JSON report written to {escape(str(json_out))}[/dim]")

    if status in ("success", "cache_hit", "awaiting_translation"):
        raise typer.Exit(code=0)
    if status == "verify_failed":
        raise typer.Exit(code=1)
    raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# repair
# ---------------------------------------------------------------------------


@app.command()
def repair(
    repo: str = typer.Argument(..., help="Path to the repo (Java sources; Python package dir with --lang python)."),
    method: str = typer.Option(..., "--method", help="Qualified method name."),
    propose_cmd: str | None = typer.Option(
        None,
        "--propose-cmd",
        help="Shell command producing a translated body: "
        "kernel-port runs '<cmd> <prompt_material_file> <round_no> <failures_json>' "
        "and reads the body from stdout. The agent's LLM sits behind this hook.",
    ),
    max_rounds: int = typer.Option(5, "--max-rounds", help="Give up after this many rounds."),
    verify_cases: int = typer.Option(50, "--verify-cases", help="Differential cases per round."),
    seed: int = typer.Option(42, "--seed", help="RNG seed for reproducibility."),
    boundary: bool = typer.Option(
        True, "--boundary/--no-boundary", help="Run boundary-value cases each round."
    ),
    lang: str = typer.Option("java", "--lang", help="'java' (default) or 'python' (Python->C via cffi)."),
    backend: str = typer.Option("jni", "--backend", help="'jni' (default), 'ffm' (needs JDK 22+) or 'rust' (needs cargo)."),
    work_dir: str | None = typer.Option(None, "--work-dir", help="Work dir (default: ./.kp-repair)."),
    dir: str | None = typer.Option(None, "--dir", help="Only scan this subdirectory of the repo."),
    initial_body_file: str | None = typer.Option(
        None, "--initial-body-file", help="File with the round-1 body (skips the first hook call)."
    ),
    propose_timeout: float = typer.Option(300.0, "--propose-timeout", help="Seconds per hook call."),
    json_out: str | None = typer.Option(None, "--json", help="Write the machine-readable report as JSON."),
) -> None:
    """Auto-repair loop: verify -> re-translate (via hook) -> re-verify until it passes.

    kernel-port stays LLM-agnostic: translation proposals come from
    --propose-cmd. The loop stops on success, when a failure signature
    repeats an earlier round's (stalled/oscillating), or at --max-rounds.
    With --lang python the loop drives the Python->C (cffi) pipeline
    instead of the JVM harness.
    """
    lang = lang.lower()
    if lang not in ("java", "python"):
        console.print("[red]Error:[/red] --lang must be 'java' or 'python'.")
        raise typer.Exit(code=1)
    if lang == "python":
        if backend != "jni":
            console.print(
                "[red]Error:[/red] --lang python implies the cffi backend; drop --backend."
            )
            raise typer.Exit(code=1)
        backend = "cffi"
    elif backend not in ("jni", "ffm"):
        console.print(f"[red]Error:[/red] unknown backend '{escape(backend)}' (choose 'jni' or 'ffm').")
        raise typer.Exit(code=1)
    if not propose_cmd:
        console.print(
            "[red]Error:[/red] --propose-cmd is required: it is the hook that "
            "produces translated bodies (your LLM sits behind it)."
        )
        raise typer.Exit(code=1)
    wdir = Path(work_dir) if work_dir else Path.cwd() / ".kp-repair"
    initial_body = Path(initial_body_file).read_text(encoding="utf-8") if initial_body_file else None
    with console.status("Running repair loop..."):
        try:
            rep = repair_mod.repair_loop(
                repo,
                method,
                propose_cmd=propose_cmd,
                max_rounds=max_rounds,
                verify_cases=verify_cases,
                seed=seed,
                boundary=boundary,
                backend=backend,
                work_dir=wdir,
                subdir=dir,
                initial_body=initial_body,
                propose_timeout=propose_timeout,
                lang=lang,
            )
        except repair_mod.RepairError as exc:
            console.print(f"[red]Repair failed:[/red] {escape(str(exc))}")
            raise typer.Exit(code=1)
    d = rep.to_dict()
    status = d.get("status", "error")
    n_rounds = len(d.get("rounds", []))
    console.print(f"[bold]repair status:[/bold] {status} "
                  f"({n_rounds} rounds, {d.get('total_elapsed_s', 0):.1f}s)")
    for rnd in d.get("rounds", []):
        console.print(
            f"  round {rnd.get('round')}: "
            f"{rnd.get('passed', '?')} passed / {rnd.get('failed', '?')} failed"
            f" (boundary {rnd.get('boundary_passed', '?')}/{rnd.get('boundary_failed', '?')})"
            + (" (build error)" if rnd.get("build_error") else "")
        )
    stop_reason = d.get("stop_reason")
    if status != "success" and stop_reason:
        console.print(f"[yellow]Stopped:[/yellow] {escape(stop_reason)}")
    if json_out:
        Path(json_out).write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")
        console.print(f"[dim]JSON report written to {escape(str(json_out))}[/dim]")
    if status == "success":
        console.print("[green]Translation repaired and verified.[/green]")
        raise typer.Exit(code=0)
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# mcp
# ---------------------------------------------------------------------------


@app.command()
def mcp() -> None:
    """Start the MCP server (stdio) exposing mine/extract/glue/verify/bench/pipeline as tools."""
    try:
        from .mcp_server import serve
    except ImportError as exc:
        console.print(
            f"[red]Error:[/red] MCP SDK not installed ({escape(str(exc))}). "
            "Install it with: pip install 'kernel-port[mcp]'"
        )
        raise typer.Exit(code=1)
    serve()
