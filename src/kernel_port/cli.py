"""Command-line interface for kernel-port."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__, extractor, harness, jni, miner
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
                "v0.1 needs a unique method name."
            )
        else:
            console.print(f"[red]Error:[/red] method not found: {qualname}")
        raise typer.Exit(code=1)
    return m


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
) -> None:
    """Scan for pure-compute kernel candidates, ranked by value."""
    fmt = format.lower()
    if fmt not in ("table", "json"):
        console.print("[red]Error:[/red] --format must be 'table' or 'json'")
        raise typer.Exit(code=1)
    r = _load_repo(repo, dir)
    cands = [c for c in miner.mine(r) if c.score >= min_score or not c.pure]
    if fmt == "json":
        console.print(
            json.dumps(
                [
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
                indent=2,
            )
        )
        return
    table = Table(title=f"kernel candidates in {repo}")
    table.add_column("Method", style="cyan")
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
            console.print(f"  [dim]{c.qualname}: {reason}[/dim]")
        if len(impure) > 10:
            console.print(f"  [dim]... and {len(impure) - 10} more[/dim]")


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


@app.command()
def extract(
    repo: str = typer.Argument(..., help="Path to the Java repo."),
    method: str = typer.Option(..., "--method", help="Qualified method name, e.g. com.example.Kernels.dot."),
    dir: str | None = typer.Option(None, "--dir", help="Only scan this subdirectory of the repo."),
    format: str = typer.Option("text", "--format", help="'text' or 'json'."),
) -> None:
    """Extract a kernel + dependency closure + purity report (LLM prompt material)."""
    fmt = format.lower()
    if fmt not in ("text", "json"):
        console.print("[red]Error:[/red] --format must be 'text' or 'json'")
        raise typer.Exit(code=1)
    r = _load_repo(repo, dir)
    try:
        ext = extractor.extract(r, method)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)
    if fmt == "json":
        console.print(json.dumps(extractor.to_json(ext), indent=2))
    else:
        console.print(extractor.render_text(ext))
    if not ext.purity.pure:
        raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# glue
# ---------------------------------------------------------------------------


@app.command()
def glue(
    repo: str = typer.Argument(..., help="Path to the Java repo."),
    method: str = typer.Option(..., "--method", help="Qualified method name."),
    out: str = typer.Option(..., "--out", help="Directory for generated JNI files."),
    dir: str | None = typer.Option(None, "--dir", help="Only scan this subdirectory of the repo."),
) -> None:
    """Generate JNI boilerplate: native Java class, C++ skeleton, CMakeLists."""
    r = _load_repo(repo, dir)
    m = _resolve_method(r, method)
    rep = check_purity(m, r)
    if not rep.pure:
        console.print(f"[red]Refusing:[/red] {m.qualname} is not pure:")
        for reason in rep.reasons:
            console.print(f"  - {reason}")
        raise typer.Exit(code=2)
    try:
        files = jni.generate(m, Path(out))
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)
    console.print(f"[green]Glue generated for {m.qualname}:[/green]")
    for kind, path in files.items():
        console.print(f"  {kind:5s} {path}")
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
    work: str | None = typer.Option(None, "--work", help="Work dir (default: <glue-dir>/.kp-work)."),
    dir: str | None = typer.Option(None, "--dir", help="Only scan this subdirectory of the repo."),
) -> None:
    """Differential-test the JNI translation against the Java original."""
    env = _build(repo, method, glue_dir, work, dir)
    with console.status(f"Running {cases} differential cases..."):
        report = harness.run_verify(env, cases=cases, seed=seed)
    if report.all_passed:
        skip_note = f" ({report.skipped} out-of-domain inputs skipped)" if report.skipped else ""
        console.print(
            f"[green]PASS[/green] {report.passed}/{len(report.cases)} cases{skip_note} -- "
            f"{report.method} JNI translation matches Java."
        )
    else:
        console.print(
            f"[red]FAIL[/red] {report.passed}/{len(report.cases)} passed; "
            f"{report.failed} mismatches:"
        )
        for c in report.cases:
            if not c.ok:
                console.print(f"  case {c.index}: {c.detail}")
                if len([x for x in report.cases if not x.ok]) > 10:
                    break
        raise typer.Exit(code=1)


@app.command()
def bench(
    repo: str = typer.Argument(..., help="Path to the Java repo."),
    method: str = typer.Option(..., "--method", help="Qualified method name."),
    glue_dir: str = typer.Option(..., "--glue-dir", help="Dir with filled kernels_jni.cpp + *Native.java."),
    iters: int = typer.Option(2000, "--iters", help="Timed iterations per repeat."),
    warmup: int = typer.Option(200, "--warmup", help="Warmup iterations (discarded)."),
    repeats: int = typer.Option(5, "--repeats", help="Measurement repeats (median taken)."),
    work: str | None = typer.Option(None, "--work", help="Work dir (default: <glue-dir>/.kp-work)."),
    dir: str | None = typer.Option(None, "--dir", help="Only scan this subdirectory of the repo."),
) -> None:
    """Wall-clock benchmark: Java original vs JNI translation."""
    env = _build(repo, method, glue_dir, work, dir)
    with console.status("Benchmarking..."):
        try:
            report = harness.run_bench(env, iters=iters, warmup=warmup, repeats=repeats)
        except RuntimeError as exc:
            console.print(f"[red]Benchmark failed:[/red] {exc}")
            raise typer.Exit(code=1)
    table = Table(title=f"benchmark: {report.method} ({report.iters} iters x {repeats} repeats)")
    table.add_column("Side")
    table.add_column("Median/call", justify="right")
    table.add_column("Mean/call", justify="right")
    import statistics

    j_med = report.java_per_call_ns
    n_med = report.jni_per_call_ns
    j_mean = statistics.mean(report.java_nanos) / report.iters
    n_mean = statistics.mean(report.jni_nanos) / report.iters
    table.add_row("Java", harness.fmt_ns(j_med), harness.fmt_ns(j_mean))
    table.add_row("JNI C++", harness.fmt_ns(n_med), harness.fmt_ns(n_mean))
    console.print(table)
    speed = report.speedup
    if speed >= 1:
        console.print(f"[green]JNI is {speed:.2f}x faster[/green] (median).")
    else:
        console.print(f"[yellow]JNI is {1/speed:.2f}x SLOWER[/yellow] (median) -- not worth porting.")
