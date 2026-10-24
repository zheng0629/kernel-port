---
name: kernel-port
description: Port performance-critical pure functions across languages with proof. Mine pure-compute kernels from Java or Python repos, extract them with static purity reports, generate FFI glue (JNI, Panama FFM, Rust, cffi), then differential-test and benchmark the translation. Use when asked to translate hot numeric code to C++/C/Rust, to generate JNI/FFM glue, or to verify a native translation is correct and actually faster.
---

# kernel-port

`kernel-port` (PyPI: `kernel-port-agent`) is a deterministic toolchain for cross-language performance porting. It does NOT call an LLM: you (the agent) do the translation with your own model; the tool handles everything around it that is deterministic.

Pipeline: `mine` → `extract` → (you translate) → `glue` → `verify` → `bench`, or one-shot `pipeline`.

## Install

- Python 3.10+: `pip install 'kernel-port-agent[mcp]'` (the `[mcp]` extra is only needed for the MCP server).
- Per backend: a JDK (Panama FFM needs 22+), a C++ toolchain (JNI), `cargo` (Rust), `cffi` (Python→C).

## Two ways to call it

**MCP server (recommended).** `kernel-port mcp` speaks stdio. Register it:

```json
{ "mcpServers": { "kernel-port": { "command": "kernel-port", "args": ["mcp"] } } }
```

It exposes 7 tools: `mine`, `extract`, `glue`, `verify`, `bench`, `pipeline`, `repair`.

**CLI.** Same capabilities via shell, e.g. `kernel-port mine /path/to/repo`.

## Workflow

1. **mine** — scan a repo for pure-compute kernel candidates, ranked by value:
   `kernel-port mine <repo> --clusters` (add `--lang python` for Python repos).
2. **extract** — get the translation prompt material: type signatures, dependency-closure source, purity report:
   `kernel-port extract <repo> --method com.example.Kernels.dot` (`--format json` for machines).
3. **translate (you)** — translate the method body with your own LLM into the target language; save it as a file (e.g. `out.cpp`).
4. **pipeline** — one shot: generates FFI glue, compiles, differential-tests (default 50 random + boundary cases), benchmarks Java vs native, writes an HTML report:
   `kernel-port pipeline <repo> --method com.example.Kernels.dot --translated-body out.cpp --html report.html`
   Omit `--method` to take the top-ranked pure candidate. If `verify` fails, `repair` runs a verify → re-translate → re-verify loop.
5. **Read the verdict.** `bench` reports speedup honestly, including the JNI call-overhead split and a break-even analysis. Small kernels often get SLOWER through JNI; the tool says so. Trust the verdict.

## Backends

- `--backend jni` (default): Java → C++ via JNI.
- `--backend ffm`: Java → C via Panama FFM (needs JDK 22+).
- `--backend rust`: Java → Rust via the `jni` crate (needs cargo).
- `--lang python`: Python → C via cffi (`pipeline --lang python`).

## Rules

- Never skip `verify`: differential testing (float 1e-6, double 1e-12 relative tolerance) is what makes a port trustworthy.
- `glue` output is scaffolding with your translation spliced in; read the JNI safety notes it emits.
- Full docs: `README.md` (English), `README.zh-CN.md` (中文).
