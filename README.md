# kernel-port

**Mine pure-compute "kernels" from a Java repo, and port them to C++ via JNI — with proof.**

`kernel-port` is an **agent-callable toolchain** for cross-language performance porting. It does not call an LLM itself. Instead, it handles every *deterministic* step around the translation, so a coding agent (like Muse) can run the whole pipeline:

```
mine → extract → (agent translates with its LLM) → glue → verify → bench
```

| Step | Command | What it does |
|------|---------|--------------|
| 1. Mine | `kernel-port mine <repo>` | Scan Java sources for pure-compute methods, **function clusters** (entry + pure helpers) and **portable classes**; rank by calls × compute intensity (or measured CPU with `--profile`) |
| 2. Extract | `kernel-port extract --method <qualname>` | Pull the method + its dependency closure, emit a purity report and clean prompt material for the LLM step |
| 3. Translate | *(the calling agent's LLM)* | The tool deliberately stops here — the agent translates the body to C++ |
| 4. Glue | `kernel-port glue --method … --out <dir>` | Generate JNI boilerplate: `*Native.java` (with **fallback switch**), `kernels_jni.cpp` skeleton with a marked fill-in spot, `CMakeLists.txt` |
| 5. Verify | `kernel-port verify --glue-dir <dir> …` | Differential-test the JNI build against the Java original on random **and boundary** inputs |
| 6. Bench | `kernel-port bench --glue-dir <dir> …` | Wall-clock benchmark with **JNI-overhead split** and **break-even analysis**; `--rigorous` uses JMH/Google Benchmark when available |
| 1–6 in one | `kernel-port pipeline …` | One-shot mine → extract → glue → verify → bench with a consolidated report (console + JSON + HTML) |

Existing tools (CodeMorph, SwapCode, …) are paste-code-in / get-translation-out web pages: no repository mining, no purity proof, no FFI glue, no differential testing, no benchmark. kernel-port is the pipeline *around* the translation — that's the moat.

## Requirements

- Python ≥ 3.10
- A JDK (17+) and `g++` on `PATH` — needed for `verify` / `bench` (they compile and run both sides)
- `mcp` Python package — only needed for `kernel-port mcp` (`pip install 'kernel-port[mcp]'`)

## Install

```bash
pip install kernel-port
# or from source:
pip install -e .
```

## Quick start

```bash
# 1. Find candidates (methods + clusters + classes)
kernel-port mine ./my-java-project

# 2. Extract one (purity report + LLM prompt material)
kernel-port extract ./my-java-project --method com.example.Kernels.matMul

# 3. Generate JNI glue (fallback switch included by default)
kernel-port glue ./my-java-project --method com.example.Kernels.matMul --out ./port/matmul

# 4. (agent) translate the body to C++; #include lines in the translation
#    are hoisted into the generated file automatically.

# 5. Prove the translation is correct (random + boundary cases)
kernel-port verify ./my-java-project --method com.example.Kernels.matMul \
    --glue-dir ./port/matmul --cases 200

# 6. Prove it's worth it (overhead split + break-even)
kernel-port bench ./my-java-project --method com.example.Kernels.matMul \
    --glue-dir ./port/matmul --sizes 16,64,256,1024 --html bench.html
```

Or do it all in one shot (the agent still supplies the translation):

```bash
kernel-port pipeline ./my-java-project --method com.example.Kernels.matMul \
    --translated-body ./matmul.cpp --html report.html --json report.json
```

Example output of `mine`:

```
kernel candidates in ./my-java-project
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ Method                  ┃ Location ┃ Pure ┃ Calls ┃ Score ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━━┩
│ com.example.Kernels.dot │ …:12     │ yes  │     3 │  17.6 │
│ …                       │          │      │       │       │
└─────────────────────────┴──────────┴──────┴───────┴───────┘

function clusters (entry + pure helpers, ported as one unit)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━━━┓
┃ Entry                    ┃ Members ┃ Pure ┃ Calls ┃ Score ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━━┩
│ com.example.Kernels.norm │       3 │ yes  │     0 │  21.4 │
└──────────────────────────┴─────────┴──────┴───────┴───────┘

class candidates (ported as a unit)
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┓
┃ Class              ┃ Kind      ┃ Portable ┃ Methods ┃ Score ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━┩
│ com.example.Util   │ stateless │ yes      │       4 │  19.2 │
│ com.example.Counter│ rejected  │ no       │       2 │   0.0 │
└────────────────────┴───────────┴──────────┴─────────┴───────┘
```

Impure methods/classes are never silently dropped — they're listed with concrete reasons (`calls System.out.println(...)`, `mutable field 'count' written in 'Counter.increment()'`, …).

## How the agent loop works

1. `mine` ranks candidates so the agent ports the highest-value kernel first. Beyond lone methods it finds **function clusters** (an entry method plus the pure helpers it calls — translated as one unit) and **portable classes** (stateless utility classes, or immutable classes with all-`final` fields; mutable classes are rejected with reasons). `--no-clusters` restores pure method-level listing; `--profile <collapsed.txt>` re-ranks by measured CPU time from an async-profiler collapsed-stack file instead of static heuristics.
2. `extract` gives the LLM exactly what it needs: the type signature, the purity verdict, and the full source of the method plus its dependency closure (helpers, constants). Impure methods are refused with reasons. `--class` extracts a whole class. When a previous `verify` failed, `--with-counterexamples` appends the failing input/output pairs to the prompt material so the next translation targets the failure. Verified translations are fingerprinted (normalized-AST hash) and cached in `~/.cache/kernel-port` — `extract` reuses them instead of re-translating (`--no-cache` to skip).
3. `glue` means the LLM only translates the *algorithm*, never JNI boilerplate. Array parameters are marshalled as in/out (copied back) — safe by default. The generated `*Native.java` is a **dispatcher**: the Java original stays callable behind `-Dkernelport.native=false` (or `setUseNative(false)`), so a bad native port rolls back instantly. `--no-fallback` restores the plain native declaration. `--class` generates glue for every method of a portable class.
4. `verify` generates random inputs **plus boundary cases** (empty arrays, 0, NaN/±Infinity, `Integer.MIN_VALUE/MAX_VALUE`, …) — the corners where LLM translations most often break — runs both sides, and compares with float tolerance. Inputs the Java original itself rejects are skipped (out of domain); a JNI crash or mismatch fails loudly. Failing cases are persisted for `--with-counterexamples`.
5. `bench` reports median per-call time for both sides, **splits out the JNI call overhead** (measured with an empty native call) from pure C++ compute time, and tells you *why* a port is slower — overhead-dominated (kernel too small) vs. compute-dominated (bad translation). `--sizes 16,64,256,1024` runs a **break-even analysis**: the input size at which porting starts paying off. `--rigorous` uses JMH / Google Benchmark when installed, otherwise an enhanced wall-clock (IQR outlier trimming + 95% CI) with an honest notice. `--html` writes a self-contained report (inline SVG charts, no external deps).
6. `pipeline` chains it all: mine → extract → glue → (agent's `--translated-body`, `#include`s hoisted automatically) → verify → bench, with `--candidates-dir` to try N translations and auto-pick the fastest fully-passing one, `--cache-dir`/`--no-cache` for the fingerprint cache, and `--html`/`--json` reports. Without `--translated-body` it stops after glue and prints exact next steps for the agent.
7. `kernel-port mcp` starts a **stdio MCP server** exposing `mine`, `extract`, `glue`, `verify`, `bench`, `pipeline` as tools, so any MCP-capable agent can call the toolchain natively.

## Design notes

- **Purity is conservative.** False negatives (rejecting a pure method with a stated reason) are acceptable; false positives (claiming impure code is pure) are treated as bugs. The allowlist is deliberately small: `Math.*`/`StrictMath.*` minus `random()`, array creation, `static final` constants, reads of `final` instance fields (immutable-class context).
- **Scope:** static methods; `void`, primitives and 1-D primitive arrays. `String`, objects, 2-D arrays and generics are rejected with a clear error. Classes: only stateless or immutable ones are portable as a unit.
- Parsing is tree-sitter based (same infrastructure family as `deadcode-hunter`).

## License

MIT
