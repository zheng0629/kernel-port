"""Command-line interface for kernel-port."""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__, classmine, extractor, harness, jni, miner
from . import java_ast as J
from .purity import check_purity

app = typer.Typer(
    help="Mine pure-compute kernels from Java repos; extract, glue (JNI), verify and benchmark C++ translations.",
    no_args_is_help=True,
)
console = Console()


def main() -> None:
    app()


def _load_repo(repo: str, subdir: str | None) -> J.JavaRepo:
    root = Path(repo)
    if not root.exists():
        console.print(f"[red]Error:[/red] path not found: {repo}")
        raise typer.Exit(code=1)
    try:
        return J.scan_repo(root, subdir)
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)


def _resolve_method(repo: J.JavaRepo, qualname: str) -> J.MethodInfo:
    m = repo.find_method(qualname)
    if m is None:
        overloads = repo.overloads(qualname)
        if overloads:
            console.print(
                f"[red]Error:[/red] '{qualname}' matches {len(overloads)} overloads; "
                "v0.3 needs a unique method name."
            )
        else:
            console.print(f"[red]Error:[/red] method not found: {qualname}")
        raise typer.Exit(code=1)
    return m


def _failures_path(work: Path, method_qn: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", method_qn)
    return work / ".kp-work" / f"{safe}.failures.json"


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
    clusters: bool = typer.Option(
        True, "--clusters/--no-clusters", help="Show function-cluster candidates (entry + pure helpers)."
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="async-profiler collapsed-stack file; re-rank by measured CPU time.",
    ),
) -> None:
    """Scan for pure-compute kernel candidates, ranked by value."""
    fmt = format.lower()
    if fmt not in ("table", "json"):
        console.print("[red]Error:[/red] --format must be 'table' or 'json'")
        raise typer.Exit(code=1)
    r = _load_repo(repo, dir)

    cands = miner.mine(r)
    cluster_list = miner.find_clusters(r) if clusters else []
    # Members of pure clusters are represented by the cluster row; members of
    # impure clusters stay listed individually (still individually portable).
    hidden = {m for cl in cluster_list if cl.pure for m in cl.members} if clusters else set()
    cands = [c for c in cands if c.qualname not in hidden and (c.score >= min_score or not c.pure)]
    cluster_list = [cl for cl in cluster_list if cl.score >= min_score or not cl.pure]
    class_cands = classmine.find_class_candidates(r)

    cpu_map: dict[str, float] = {}
    if profile:
        from . import profile as prof

        try:
            cpu_map = prof.parse_collapsed(profile)
        except (OSError, ValueError) as exc:
            console.print(f"[red]Error:[/red] cannot parse profile: {exc}")
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
            console.print(f"  [dim]{c.qualname}: {reason}[/dim]")
        if len(impure) > 10:
            console.print(f"  [dim]... and {len(impure) - 10} more[/dim]")

    # -- cluster table ------------------------------------------------------
    if clusters and cluster_list:
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
        console.print("[dim]Members: " + "; ".join(
            f"{cl.entry} <- {', '.join(m for m in cl.members[1:])}" for cl in cluster_list[:5]
        ) + "[/dim]")

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
                console.print(f"  [dim]{cc.qualname}: {reason}[/dim]")


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
) -> None:
    """Extract a kernel + dependency closure + purity report (LLM prompt material)."""
    fmt = format.lower()
    if fmt not in ("text", "json"):
        console.print("[red]Error:[/red] --format must be 'text' or 'json'")
        raise typer.Exit(code=1)
    if bool(method) == bool(class_qn):
        console.print("[red]Error:[/red] pass exactly one of --method or --class.")
        raise typer.Exit(code=1)
    r = _load_repo(repo, dir)

    if class_qn:
        try:
            cext = extractor.extract_class(r, class_qn)
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1)
        if fmt == "json":
            print(json.dumps(extractor.to_json_class(cext), indent=2))
        else:
            console.print(extractor.render_text_class(cext))
        if not cext.class_purity.portable:
            raise typer.Exit(code=2)
        return

    try:
        ext = extractor.extract(r, method)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    counterex_text = ""
    counterex_data: list[dict] = []
    if with_counterexamples:
        w = work or (str(Path(glue_dir) / ".kp-work") if glue_dir else None)
        if not w:
            console.print("[red]Error:[/red] --with-counterexamples needs --work or --glue-dir.")
            raise typer.Exit(code=1)
        from . import counterex

        fpath = _failures_path(Path(w), method)
        counterex_data = counterex.load_failures(fpath)
        if counterex_data:
            counterex_text = "\n\n" + counterex.format_counterexamples(counterex_data, method)
        else:
            console.print(f"[yellow]Note:[/yellow] no failure records at {fpath}.")

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
        console.print(extractor.render_text(ext) + counterex_text + cache_note)
    if not ext.purity.pure:
        raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# glue
# ---------------------------------------------------------------------------


@app.command()
def glue(
    repo: str = typer.Argument(..., help="Path to the Java repo."),
    method: str | None = typer.Option(None, "--method", help="Qualified method name."),
    class_qn: str | None = typer.Option(None, "--class", help="Qualified class name (class-level glue)."),
    out: str = typer.Option(..., "--out", help="Directory for generated JNI files."),
    dir: str | None = typer.Option(None, "--dir", help="Only scan this subdirectory of the repo."),
    fallback: bool = typer.Option(
        True,
        "--fallback/--no-fallback",
        help="Keep the Java original callable behind a runtime switch (-Dkernelport.native=false).",
    ),
) -> None:
    """Generate JNI boilerplate: native Java class, C++ skeleton, CMakeLists."""
    if bool(method) == bool(class_qn):
        console.print("[red]Error:[/red] pass exactly one of --method or --class.")
        raise typer.Exit(code=1)
    r = _load_repo(repo, dir)

    if class_qn:
        from . import classglue
        from .purity import check_class_portable

        cls = r.classes.get(class_qn)
        if cls is None:
            console.print(f"[red]Error:[/red] class not found: {class_qn}")
            raise typer.Exit(code=1)
        rep = check_class_portable(cls, r)
        if not rep.portable:
            console.print(f"[red]Refusing:[/red] {class_qn} is not portable as a unit:")
            for reason in rep.reasons:
                console.print(f"  - {reason}")
            raise typer.Exit(code=2)
        try:
            files = classglue.generate_class(r, class_qn, Path(out))
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1)
        console.print(f"[green]Class glue generated for {class_qn}:[/green]")
        for kind, path in files.items():
            console.print(f"  {kind:5s} {path}")
        return

    m = _resolve_method(r, method)
    rep = check_purity(m, r)
    if not rep.pure:
        console.print(f"[red]Refusing:[/red] {m.qualname} is not pure:")
        for reason in rep.reasons:
            console.print(f"  - {reason}")
        raise typer.Exit(code=2)
    try:
        files = jni.generate(m, Path(out), fallback=fallback)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)
    console.print(f"[green]Glue generated for {m.qualname}:[/green]")
    for kind, path in files.items():
        console.print(f"  {kind:5s} {path}")
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
        f"[cyan]--glue-dir {out}[/cyan]."
    )


# ---------------------------------------------------------------------------
# verify / bench
# ---------------------------------------------------------------------------


def _build(repo: str, method_qn: str, glue_dir: str, work: str | None, subdir: str | None) -> harness.BuiltEnv:
    r = _load_repo(repo, subdir)
    m = _resolve_method(r, method_qn)
    rep = check_purity(m, r)
    if not rep.pure:
        console.print(f"[red]Refusing:[/red] {m.qualname} is not pure:")
        for reason in rep.reasons:
            console.print(f"  - {reason}")
        raise typer.Exit(code=2)
    gdir = Path(glue_dir)
    work_dir = Path(work) if work else gdir / ".kp-work"
    try:
        with console.status("Compiling Java + JNI..."):
            env = harness.build_env(Path(repo), m, gdir, work_dir)
    except RuntimeError as exc:
        console.print(f"[red]Build failed:[/red] {exc}")
        raise typer.Exit(code=1)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)
    return env


@app.command()
def verify(
    repo: str = typer.Argument(..., help="Path to the Java repo."),
    method: str = typer.Option(..., "--method", help="Qualified method name."),
    glue_dir: str = typer.Option(..., "--glue-dir", help="Dir with filled kernels_jni.cpp + *Native.java."),
    cases: int = typer.Option(100, "--cases", help="Number of random differential cases."),
    seed: int = typer.Option(42, "--seed", help="RNG seed for reproducibility."),
    boundary: bool = typer.Option(
        True, "--boundary/--no-boundary", help="Also run boundary-value cases (empty arrays, NaN, min/max...)."
    ),
    work: str | None = typer.Option(None, "--work", help="Work dir (default: <glue-dir>/.kp-work)."),
    dir: str | None = typer.Option(None, "--dir", help="Only scan this subdirectory of the repo."),
) -> None:
    """Differential-test the JNI translation against the Java original."""
    env = _build(repo, method, glue_dir, work, dir)
    with console.status(f"Running {cases} differential cases..."):
        report = harness.run_verify(env, cases=cases, seed=seed, boundary=boundary)
    total_pass = report.passed + report.boundary_passed
    total_fail = report.failed + report.boundary_failed
    skip_note = f" ({report.skipped} out-of-domain inputs skipped)" if report.skipped else ""
    bskip = f", {report.boundary_skipped} boundary skipped" if report.boundary_skipped else ""
    if report.all_passed:
        console.print(
            f"[green]PASS[/green] random {report.passed}/{len(report.cases)}"
            + (f", boundary {report.boundary_passed}/{len(report.boundary)}" if boundary else "")
            + f"{skip_note}{bskip} -- {report.method} JNI translation matches Java."
        )
    else:
        console.print(
            f"[red]FAIL[/red] {total_pass} passed, {total_fail} mismatches "
            f"(random {report.failed}, boundary {report.boundary_failed}):"
        )
        shown = 0
        for c in report.cases + report.boundary:
            if not c.ok:
                tag = "boundary" if c in report.boundary else "random"
                console.print(f"  [{tag}] case {c.index}: {c.detail}")
                shown += 1
                if shown >= 10:
                    break
        fpath = _failures_path(env.work, method)
        if fpath.exists():
            console.print(
                f"\n[dim]Failing cases saved to {fpath} -- re-run "
                f"[cyan]extract --method {method} --with-counterexamples --work {env.work}[/cyan] "
                "to feed them back into the translation prompt.[/dim]"
            )
        raise typer.Exit(code=1)


@app.command()
def bench(
    repo: str = typer.Argument(..., help="Path to the Java repo."),
    method: str = typer.Option(..., "--method", help="Qualified method name."),
    glue_dir: str = typer.Option(..., "--glue-dir", help="Dir with filled kernels_jni.cpp + *Native.java."),
    iters: int = typer.Option(2000, "--iters", help="Timed iterations per repeat."),
    warmup: int = typer.Option(200, "--warmup", help="Warmup iterations (discarded)."),
    repeats: int = typer.Option(5, "--repeats", help="Measurement repeats (median taken)."),
    sizes: str | None = typer.Option(
        None, "--sizes", help="Comma-separated input sizes for break-even analysis, e.g. '16,64,256,1024'."
    ),
    rigorous: bool = typer.Option(
        False, "--rigorous", help="Use JMH/Google Benchmark when available (falls back to enhanced wall-clock)."
    ),
    html: str | None = typer.Option(None, "--html", help="Write a self-contained HTML report."),
    work: str | None = typer.Option(None, "--work", help="Work dir (default: <glue-dir>/.kp-work)."),
    dir: str | None = typer.Option(None, "--dir", help="Only scan this subdirectory of the repo."),
) -> None:
    """Benchmark: Java original vs JNI translation, with JNI-overhead split."""
    env = _build(repo, method, glue_dir, work, dir)
    size_list = [int(s) for s in sizes.split(",")] if sizes else None
    with console.status("Benchmarking..."):
        try:
            report = harness.run_bench(
                env, iters=iters, warmup=warmup, repeats=repeats,
                sizes=size_list, rigorous=rigorous,
            )
        except RuntimeError as exc:
            console.print(f"[red]Benchmark failed:[/red] {exc}")
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
    table.add_row("JNI C++", harness.fmt_ns(n_med), harness.fmt_ns(n_mean))
    console.print(table)

    if report.jni_overhead_ns is not None:
        otable = Table(title="JNI overhead split")
        otable.add_column("Component")
        otable.add_column("ns/call", justify="right")
        otable.add_row("JNI call overhead (empty call)", harness.fmt_ns(report.jni_overhead_ns))
        otable.add_row("Pure compute in C++ (jni - overhead)", harness.fmt_ns(report.compute_ns))
        otable.add_row("Java original", harness.fmt_ns(j_med))
        console.print(otable)
        if report.compute_ns > j_med and n_med > j_med:
            console.print(
                "[yellow]The C++ compute itself is slower than Java -- "
                "the translation, not the JNI boundary, is the problem.[/yellow]"
            )
        elif n_med > j_med:
            console.print(
                "[yellow]JNI call overhead dominates -- the kernel is too small; "
                "porting only pays off at larger input sizes (see break-even).[/yellow]"
            )

    if report.per_size:
        stable = Table(title="speedup by input size")
        stable.add_column("Size", justify="right")
        stable.add_column("Java/call", justify="right")
        stable.add_column("JNI/call", justify="right")
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
            console.print("[yellow]Break-even:[/yellow] JNI never pays off in the measured sizes.")

    if report.rigorous_note:
        console.print(f"[dim]{report.rigorous_note}[/dim]")

    speed = report.speedup
    if speed >= 1:
        console.print(f"[green]JNI is {speed:.2f}x faster[/green] (median).")
    else:
        console.print(f"[yellow]JNI is {1/speed:.2f}x SLOWER[/yellow] (median) -- not worth porting.")

    if html:
        from . import report as kreport

        verdict = (
            f"worth porting: {speed:.2f}x faster"
            if speed >= 1
            else f"not worth porting: {1/speed:.2f}x slower"
        )
        doc = kreport.render_html(
            {
                "title": "kernel-port bench report",
                "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "method": report.method,
                "status": "success",
                "bench": {
                    "java_per_call_ns": j_med,
                    "jni_per_call_ns": n_med,
                    "speedup": speed,
                    "jni_overhead_ns": report.jni_overhead_ns,
                    "per_size": [
                        {"size": ps["size"], "speedup": ps["speedup"]}
                        for ps in report.per_size
                    ],
                },
                "verdict": verdict,
            }
        )
        Path(html).write_text(doc, encoding="utf-8")
        console.print(f"[green]HTML report written to {html}[/green]")


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


@app.command()
def pipeline(
    repo: str = typer.Argument(..., help="Path to the Java repo."),
    method: str | None = typer.Option(None, "--method", help="Qualified method name (default: top pure candidate)."),
    dir: str | None = typer.Option(None, "--dir", help="Only scan this subdirectory of the repo."),
    translated_body: str | None = typer.Option(
        None, "--translated-body", help="File with the agent's C++ translation (spliced into the glue)."
    ),
    candidates_dir: str | None = typer.Option(
        None, "--candidates-dir", help="Dir of *.cpp candidate bodies; verify+bench each, pick the winner."
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
) -> None:
    """One-shot mine -> extract -> glue -> verify -> bench for a kernel.

    The C++ translation itself stays with the calling agent: pass it via
    --translated-body, or run once without it to get the glue + prompt
    material, translate, then re-run.
    """
    from . import pipeline as kpipeline

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
            )
        except Exception as exc:  # noqa: BLE001 - report pipeline errors cleanly
            console.print(f"[red]Pipeline failed:[/red] {exc}")
            raise typer.Exit(code=1)

    status = rep.get("status", "error")
    console.print(f"[bold]pipeline status:[/bold] {status}")
    if rep.get("method"):
        console.print(f"  method:  {rep['method']}")
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
            f"  bench:   JNI {sp:.2f}x vs Java "
            f"({'faster' if sp >= 1 else 'SLOWER'})"
        )
    if rep.get("winner"):
        console.print(f"  winner:  [green]{rep['winner']}[/green]")
    if rep.get("html_path"):
        console.print(f"  html:    {rep['html_path']}")
    if rep.get("counterexamples_path"):
        console.print(f"  counterexamples: {rep['counterexamples_path']}")
    instructions = rep.get("instructions")
    if instructions:
        console.print(f"\n{instructions}")

    if json_out:
        Path(json_out).write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
        console.print(f"[dim]JSON report written to {json_out}[/dim]")

    if status in ("success", "cache_hit", "awaiting_translation"):
        raise typer.Exit(code=0)
    if status == "verify_failed":
        raise typer.Exit(code=1)
    raise typer.Exit(code=2)


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
            f"[red]Error:[/red] MCP SDK not installed ({exc}). "
            "Install it with: pip install 'kernel-port[mcp]'"
        )
        raise typer.Exit(code=1)
    serve()
