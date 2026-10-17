# kernel-port

**Mine pure-compute "kernels" from a Java repo, and port them to C++ via JNI — with proof.**

`kernel-port` is an **agent-callable toolchain** for cross-language performance porting. It does not call an LLM itself. Instead, it handles every *deterministic* step around the translation, so a coding agent (like Muse) can run the whole pipeline:

```
mine → extract → (agent translates with its LLM) → glue → verify → bench
```

| Step | Command | What it does |
|------|---------|--------------|
| 1. Mine | `kernel-port mine <repo>` | Scan Java sources for pure-compute methods (no I/O, no network, no mutable global state), rank by calls × compute intensity |
| 2. Extract | `kernel-port extract --method <qualname>` | Pull the method + its dependency closure, emit a purity report and clean prompt material for the LLM step |
| 3. Translate | *(the calling agent's LLM)* | The tool deliberately stops here — the agent translates the body to C++ |
| 4. Glue | `kernel-port glue --method … --out <dir>` | Generate JNI boilerplate: `*Native.java`, `kernels_jni.cpp` skeleton with a marked fill-in spot, `CMakeLists.txt` |
| 5. Verify | `kernel-port verify --glue-dir <dir> …` | Differential-test the JNI build against the Java original on random inputs |
| 6. Bench | `kernel-port bench --glue-dir <dir> …` | Wall-clock benchmark (warmup + N timed runs, median) with a speedup verdict |

Existing tools (CodeMorph, SwapCode, …) are paste-code-in / get-translation-out web pages: no repository mining, no purity proof, no FFI glue, no differential testing, no benchmark. kernel-port is the pipeline *around* the translation — that's the moat.

## Requirements

- Python ≥ 3.10
- A JDK (17+) and `g++` on `PATH` — needed for `verify` / `bench` (they compile and run both sides)

## Install

```bash
pip install kernel-port
# or from source:
pip install -e .
```

## Quick start

```bash
# 1. Find candidates
kernel-port mine ./my-java-project

# 2. Extract one (purity report + LLM prompt material)
kernel-port extract ./my-java-project --method com.example.Kernels.matMul

# 3. Generate JNI glue
kernel-port glue ./my-java-project --method com.example.Kernels.matMul --out ./port/matmul

# 4. (agent) paste the translated C++ body into the TODO(agent) spot in
#    ./port/matmul/kernels_jni.cpp

# 5. Prove the translation is correct
kernel-port verify ./my-java-project --method com.example.Kernels.matMul \
    --glue-dir ./port/matmul --cases 200

# 6. Prove it's worth it
kernel-port bench ./my-java-project --method com.example.Kernels.matMul \
    --glue-dir ./port/matmul
```

Example output of `mine`:

```
com.example.Kernels.dot    src/com/example/Kernels.java:12   yes   3    17.6
com.example.Kernels.sieve  src/com/example/Kernels.java:48   yes   0    13.1
com.example.Kernels.noisy  src/com/example/Kernels.java:91   no    1     0.0
```

Impure methods are never silently dropped — they're listed with concrete reasons (`calls System.out.println(...)`, `mutates field 'callCount'`, …).

## How the agent loop works

1. `mine` ranks candidates so the agent ports the highest-value kernel first.
2. `extract` gives the LLM exactly what it needs: the type signature, the purity verdict, and the full source of the method plus its dependency closure (helpers, constants). Impure methods are refused with reasons.
3. `glue` means the LLM only translates the *algorithm*, never JNI boilerplate. Array parameters are marshalled as in/out (copied back) — safe by default.
4. `verify` generates random inputs, runs both sides, and compares with float tolerance. Inputs the Java original itself rejects are skipped (out of domain); a JNI crash or mismatch fails loudly.
5. `bench` reports median per-call time for both sides and a verdict like `JNI is 1.88x faster` — or honestly `1.49x SLOWER -- not worth porting`.

## Design notes

- **Purity is conservative.** False negatives (rejecting a pure method with a stated reason) are acceptable; false positives (claiming impure code is pure) are treated as bugs. The allowlist is deliberately small: `Math.*`/`StrictMath.*` minus `random()`, array creation, `static final` constants.
- **v0.1 scope:** static methods only; `void`, primitives and 1-D primitive arrays. `String`, objects, 2-D arrays and generics are rejected with a clear error.
- Parsing is tree-sitter based (same infrastructure family as `deadcode-hunter`).

## Roadmap (v0.2)

- `--rigorous` benchmark mode (JMH on the Java side, Google Benchmark on the C++ side)
- Overloaded-method disambiguation (`--signature`)
- Instance methods and more types (`String`, 2-D arrays)

## License

MIT
