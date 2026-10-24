# kernel-port

[中文](README.zh-CN.md) | **English**

**Mine pure-compute "kernels" from a Java repo, and port them to C++ via JNI — with proof.**

`kernel-port` is an **agent-callable toolchain** for cross-language performance porting. It does not call an LLM itself. Instead, it handles every *deterministic* step around the translation, so a coding agent (like Muse) can run the whole pipeline:

```
mine → extract → (agent translates with its LLM) → glue → verify → bench
```

| Step | Command | What it does |
|------|---------|--------------|
| 1. Mine | `kernel-port mine <repo>` | Scan Java sources for pure-compute methods, **function clusters** (entry + pure helpers) and **portable classes**; rank by calls × compute intensity (or measured CPU with `--profile`) |
| 2. Extract | `kernel-port extract --method <qualname>` | Pull the method + its dependency closure, emit a purity report and clean prompt material for the LLM step (overloads: `--method com.example.K.foo(int,double)`) |
| 3. Translate | *(the calling agent's LLM)* | The tool deliberately stops here — the agent translates the body to C++/C/Rust (per the chosen backend) |
| 4. Glue | `kernel-port glue --method … --out <dir>` | Generate FFI boilerplate: `*Native.java` (with **fallback switch**), `kernels_jni.cpp` skeleton with a marked fill-in spot, `CMakeLists.txt`; `--backend ffm` generates Panama FFM bindings + a pure-C export instead (JDK 22+); `--backend rust` generates a Rust cdylib via the jni crate (needs cargo) |
| 5. Verify | `kernel-port verify --glue-dir <dir> …` | Differential-test the native build against the original on random **and boundary** inputs; per-type float tolerance (`--tol-float` 1e-6, `--tol-double` 1e-12); `--coverage` (opt-in) adds a coverage-guided input round |
| 6. Bench | `kernel-port bench --glue-dir <dir> …` | Wall-clock benchmark with **call-overhead split** and **break-even analysis**; faster/slower claimed only outside a ±20% noise band ("inconclusive" inside); `--rigorous` uses JMH/Google Benchmark when available |
| 1–6 in one | `kernel-port pipeline …` | One-shot mine → extract → glue → verify → bench with a consolidated report (console + JSON + HTML) |

Existing tools (CodeMorph, SwapCode, …) are paste-code-in / get-translation-out web pages: no repository mining, no purity proof, no FFI glue, no differential testing, no benchmark. kernel-port is the pipeline *around* the translation — that's the moat.

## What's new in v0.12

- **Fresh-repo dogfood (apache/commons-numbers).** A second real-repo validation round — every fix below was found on non-fixture code and has a regression test (`tests/test_dogfood_v12.py`).
- **Example/benchmark sources are skipped in mining.** `java_ast.is_skipped_source` now also excludes example/benchmark dirs (exact names like `examples`, `samples`, `jmh`, `benchmark`, plus hyphenated suffixes like `-examples`, `-jmh`, so Maven modules such as `commons-numbers-examples` are skipped while a `com.example` package dir is not) — on top of the v0.8 test-dir exclusion.
- **Package-aware glue + reflective fallback.** Glue is generated in the kernel's package when the class or method isn't public (the JNI export symbol carries the package-mangled name); a private kernel method gets a `setAccessible` reflective handle on the fallback path instead of a direct call.
- **Minimal-synthesis visibility widening.** The synthesized harness copy now widens members to `public`, so a private kernel method is callable from the harness's foreign package instead of silently falling back to the repo file.
- **`extract` no longer reflows output.** Preformatted extraction lines (e.g. `Location:`) print with `soft_wrap`, so redirecting to a file can't wrap mid-path.
- **`--lang python` flag warnings.** `mine --lang python` with `--clusters`/`--no-clusters`, and `glue --lang python` with `--class`, now warn on stderr and ignore the flag instead of silently dropping it.
- **Honest Python crash reporting.** When the verify/bench driver subprocess dies on a Python import (`ModuleNotFoundError`/`ImportError`), it is reported as a driver import error — an environment problem — instead of being mislabeled a segfault in the C translation. Real crashes (signal kills) still report as crashes.
- **Python bench: `--sizes` and `--list-len`.** `bench --lang python` accepts `--list-len` (default 64: the list length when `--sizes` is absent) and `--sizes` (per-size table + break-even analysis, mirroring the Java semantics).

## What's new in v0.11

- **Repair loop hardening.** Stalled detection now compares the failure signature against *all* previous rounds, so A/B/A/B oscillation is caught (previously only adjacent rounds were compared). `stalled`/`gave_up` reports carry the last attempted body (`last_body`) and a human-readable stop reason (`"failure signature repeated from round 2 (oscillation)"`, `"max_rounds=5 reached without success"`), printed by the CLI. Round rows now show boundary counts (`round 1: 3 passed / 5 failed (boundary 2/6)`).
- **`repair --lang python`.** The auto-repair loop now drives the Python→C (cffi) pipeline too: failure signatures from the Python verify report, C-oriented prompt material, same stop conditions (success / stalled incl. oscillation / max-rounds). Java behavior unchanged.
- **Rust FFI lint, consolidated.** `lint.lint_rust` is now the canonical Rust lint with four checks — `missing_no_mangle` (entry point without `#[no_mangle]`), `ffi_panic` (`panic!`/`.unwrap()`/`.expect(` in code, `//` comments stripped), `wrong_abi` (non-`"system"` ABI), `missing_jni_use`. `glue --backend rust` reports all four; `rust_glue.lint_lib_rs` is kept as a legacy alias.
- **Console output no longer eats `[...]`.** User-controlled content (signatures like `Sequence[float]`, paths, reasons) is escaped from Rich markup; the `[random]`/`[boundary]` case tags that were previously swallowed now render literally.
- **Python purity: attribute-style intra-repo calls.** `import pkg.mod` + `pkg.mod.func(...)` (and `import pkg.mod as pm`) now resolves the callee inside the scanned repo like the v0.10 from-import fix did; genuinely foreign modules keep the conservative verdict.
- **Coverage-guided fuzz for Python** (`verify --lang python --coverage`, opt-in). `sys.monitoring`-based branch recording on the Python original, static branch manifest via `ast`+`dis`, literal-harvesting guided inputs, ddmin minimization — mirroring the Java `--coverage` contract including graceful degradation.

## What's new in v0.10

- **Python purity false negatives fixed.** `from pkg.mod import func` where `pkg.mod` is inside the scanned repo now resolves the callee and purity-checks it (previously: "foreign module" → impure). Relative from-imports (`from .mod import func`) resolve to absolute paths correctly (previously mis-recorded — a latent soundness hole). `*rest: int` now parses to the name `rest`; variadic signatures are rejected with a clear reason instead of a bogus "unknown name". Allowlist: single-arg `type(x)`, `str.isdigit/isalpha/isspace/isdecimal/isnumeric/isalnum`, non-mutating set algebra; `set.add`/`discard` are allowed only on provably locally-created sets.
- **Honest bench verdicts.** A ±20% noise band: the tool claims "faster"/"slower" only outside the band, otherwise reports "inconclusive — too close to call; not a porting win". Warmup now waits for measured JIT settling (capped) instead of a fixed iteration count. Applies to all backends (JNI, FFM, Rust, Python/cffi) and to console, JSON, and HTML output alike.
- **Tolerant verify protocol.** JVM launcher warnings on stdout (e.g. stale `hsperfdata` locks) are skipped with an stderr note instead of failing verification as protocol errors. Genuine protocol violations still fail loudly.
- **Rust backend parity.** `bench --backend rust` now reports the call-overhead split and break-even analysis, matching JNI. Fixed the shared-`CARGO_TARGET_DIR` + fixed crate-name foot-gun: crate names are unique per glue dir, so stale/wrong `.so` artifacts can't leak across kernels. Backend-aware labels ("Rust is X faster", "fix the Rust body").
- **FFM codegen fixes.** Generated wrapper locals are collision-proof (`kp_` prefix, freshened against parameter names — a parameter literally named `r` no longer breaks the build). `build_ffm.sh` resolves its output dir from its own location instead of a baked-in relative name. The fallback dispatcher now lazy-loads the native library: a missing `.so` falls back to the Java original (one stderr note) instead of failing class initialization.

## What's new in v0.9

- **Dogfood on all backends (real repos).** v0.8 proved JNI on JOML; v0.9 ran the full agent chain on real code for the other two backends: `--backend ffm` and `--backend rust` on JOML's `org.joml.Intersectiond.testRayTriangle` (verify passed on both; both benches honestly said "not worth porting" for the tiny kernel), and `--lang python` on [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python) (`maths`/`sorts`: mine → hand-written C → glue → verify → bench). Every bug below was found this way and has a regression test.
- **Parse cache survives scan-root changes.** Cached per-file paths were stored relative to the root used at write time; re-scanning the same files under a different root (repo root vs `src/main/java`) doubled the path and crashed with `FileNotFoundError`. Cached records are now re-stamped to the current scan root on load.
- **Pipeline keeps the overload suffix.** `pipeline --method 'K.foo(int,double)'` resolved the overload, then dropped the `(types)` suffix before extraction and died with `AmbiguityError`. The original ref is now passed through.
- **Counterexample filename agreement.** `verify` persists failures under the *bare* qualname, but `extract --with-counterexamples` looked them up under the raw CLI ref — a `(types)` suffix silently missed the file. Both sides now use the resolved qualname.
- **Python purity: soundness + parity.** Two dogfood findings: (1) **CRITICAL** — a sole generator-expression argument sits directly under `call` in tree-sitter-python (no `argument_list` wrapper) and was never walked, so `open()`/`print()` inside a genexp passed as pure; genexps are now checked. (2) Defensive `raise ValueError(...)` no longer rejects the function — the deterministic-throw carve-out Java got in v0.4 now applies to Python (`raise` of builtin exceptions with pure args; custom types, pre-built objects and bare re-raise still rejected).
- **`repair` supports `--backend rust`.** The repair loop rejected the Rust backend outright; it now runs verify→re-translate→re-verify through the Rust toolchain like JNI/FFM.
- **Backend-aware extract notes.** `extract --backend ffm|rust` (new flag) tailors the LLM translation notes to the target (plain-C ABI / Rust conventions) instead of always saying C++17/JNI; `pipeline` and `repair` pass their backend through automatically.
- **MCP `pipeline` accepts cache options for Python.** The MCP tool rejected `candidates_dir`/`no_cache`/`cache_dir` with `lang='python'` although the CLI (and `run_python_pipeline`) support them; the validation now matches the CLI.
- **`.kp-work` no longer doubles.** `_persist_failures` appended `/.kp-work` even when the work dir already was one.

## What's new in v0.8

- **Real-repo dogfood (JOML).** First validation on non-fixture code: mined [JOML](https://github.com/JOML-CI/JOML) (121 files, 662 methods) and ran the full agent workflow end-to-end on `org.joml.Intersectiond.testRayTriangle` (Möller–Trumbore): extract → hand-written C++ → glue → verify **100/100 random + 24/24 boundary** → bench. The bench told the honest story: **2.47x slower, not worth porting** — with the overhead split correctly blaming the C++ compute itself, not the JNI boundary. See [DOGFOOD.md](DOGFOOD.md).
- **`build_env` no longer compiles the whole repo.** It used to run `javac` over every `*.java` under the repo root — real repos break that (test sources need JUnit, files need annotation-only deps). Now it compiles a **synthesized minimal kernel file** (kernel + dependency closure + only the imports they reference; `extractor.render_minimal_sources`), with `-sourcepath` fallback to the repo file when synthesis is unsafe. Bonus: the harness now tests exactly the extracted kernel — the same code the agent translated.
- **`mine` is ~2-3x faster.** File bytes were re-read once per method (11,541 reads for 121 JOML files); now `java_ast.source_bytes` (mtime+size-validated cache) is shared by miner/purity/extractor/jni. Full JOML `mine`: 84–165s → ~53s.
- **Test sources excluded from mining.** `src/test`/`test`/`tests` directories are never port candidates (JOML's `TestUtil` was ranked before this fix).
- **Enum static methods are now parsed.** tree-sitter nests them under `enum_body_declarations`; they were invisible to mine/extract entirely.
- **Python pipeline parity.** The fingerprint cache (`--no-cache`/`--cache-dir`, `cache_hit` short-circuit) and multi-candidate selection (`--candidates-dir`, fastest passing wins) now work for `pipeline --lang python`, same semantics as Java.

## What's new in v0.7

- **Python→C backend slice: cffi glue + verify/bench** (`glue/verify/bench --lang python`). Generates `kernels_cffi.c` (plain-C-ABI `kp_<name>` exports with a marked body region) plus a `kernels_cffi.py` cffi binding; `gcc -shared -fPIC -O2` builds it. Conservative type mapping — `int`→`int64_t`, `float`→`double`, `bool`→`int`, `str`→`const char*`, `list[int]`/`list[float]`→ pointer+length — and anything unmappable (unannotated params, `*args`, `list` returns) is a clear error, never a guess. Verify runs in a driver subprocess, so a segfaulting translation is reported as a failed run instead of killing the caller; boundary cases (empty list, int64 limits, inf/nan) run separately from random ones, and mutated list params are echoed back and compared element-wise. The bench is honestly wall-clock and will tell you tiny kernels are *slower* in C (cffi call overhead dominates) — same honest story as the JNI bench.
- **`pipeline --lang python`: end-to-end Python→C one-shot.** mine → extract (prompt material with the exact C signature the glue presents) → glue → verify → bench in a single report (console + JSON + HTML); without a translation it stops after glue with exact next steps. The MCP `pipeline` tool accepts `lang="python"` too. Default candidate selection skips pure-but-unmappable functions and picks the first C-mappable one. The fingerprint cache works here too: a verified translation is stored under `~/.cache/kernel-port` (override with `--cache-dir`, bypass with `--no-cache`) and a later run on the same kernel short-circuits with `cache_hit` — no rebuild, no re-verify. `--candidates-dir` takes a dir of `*.c` bodies, verifies + benches each in a fresh glue copy, and reports the fastest passing one as the winner.
- **Rust backend hardening.** The `--backend rust` Java side now has the same fallback dispatcher as JNI/FFM (`-Dkernelport.native=false` / `setUseNative(false)` rolls back to the Java original — e2e-proven). `CARGO_TARGET_DIR` defaults under the work dir (never `/tmp`), `--offline` is used when the pinned `jni` 0.21 crate is already cached, and cargo failures surface the exit code plus the last 30 lines of stderr instead of a raw dump.
- **Coverage stays opt-in — by measurement, not by default.** Evaluated making `verify --coverage` the default: overhead is 3–27x a plain verify on every measured kernel (the <2x precondition fails), so it stays opt-in, documented with numbers. Hardened anyway: coverage-round failures degrade gracefully to plain verify results (warning, never a crash), ddmin has a wall-clock budget on top of the call budget, and the round now compiles once and replays case batches through a single JVM session (`CoveredSession`).

## What's new in v0.6

- **Rust glue backend** (`glue/verify/bench --backend rust`). Generates a Rust cdylib via the `jni` crate — the Rust code exports the JNI entry points directly, so it loads through the existing JNI harness with no C++ wrapper at all. Needs `cargo` on `PATH`; the agent fills the `TODO(agent)` spot in `src/lib.rs` exactly like the C++ skeleton.
- **`kernel-port repair`: the auto-repair loop.** `verify` fails → failing cases feed back into the prompt → a caller-provided hook (`--propose-cmd`, your LLM behind it) proposes a new body → re-verify, until it passes, stalls (same failures twice), or hits `--max-rounds`. kernel-port stays LLM-agnostic; the loop is deterministic.
- **Coverage-guided fuzzing** (`verify --coverage`, opt-in). The Java original is instrumented at the source level (branch counters for if/loops/switch/ternary/catch); after the random + boundary rounds, a second input round targets uncovered branches (literal/boundary injection near comparisons). The report shows branch coverage of the tested method, and failing guided cases are delta-minimized (ddmin, call + wall-clock budgets) before they reach counterexample feedback. Deliberately opt-in, not the default: measured overhead is 3–27x a plain `verify` (dot x12.2, fib x3.1, sieve x26.7 on the sample fixtures) because the round recompiles the instrumented sources and replays every case through a fresh JVM — the ~2–4s javac + JVM-startup fixed cost alone exceeds a typical cheap verify. Coverage gain is real (sieve 60% → 100%; a magic-constant branch random testing never hit gets found), but the <2x precondition for default-on fails everywhere. Coverage-round failures degrade gracefully to plain verify results (warning, never a crash).
- **Incremental parse cache.** `mine`/`extract`/`glue` now reuse tree-sitter parse results across runs (keyed by path + mtime + size under `~/.cache/kernel-port`, `$KERNEL_PORT_PARSE_CACHE_DIR` override). `--no-parse-cache` / `--refresh-parse-cache` to control it; a cache line on stderr reports hit/miss (stdout stays pure for `--format json`).
- **Python kernel mining (slice 1 of the Python→C epic).** `mine --lang python` finds pure-compute Python functions with a conservative purity checker (no IO/`random`/global mutation/`eval`/framework decorators; `math.*` allowlisted). cffi glue + verify/bench arrived in v0.7.

## What's new in v0.5

- **JNI safety lint.** `glue` now lints the generated `kernels_jni.cpp` after generation: every `Get*` pin must have a matching `Release*`, every pin must be null-checked, and `NewObjectArray`/`NewObject` inside loops without `DeleteLocalRef` is flagged. Warnings print inline; kernel-port's own generated glue is lint-clean.
- **Hardened JNI codegen.** Pinned arrays/strings are null-checked on acquisition, and read-only array params are released with `JNI_ABORT` (skipping the copy-back) — decided statically by `jni.written_arrays()`; the FFM backend likewise skips its copy-back for read-only arrays.
- **`pipeline --backend ffm`.** The one-shot pipeline (and the MCP `pipeline` tool) now accept `--backend ffm`, running mine → extract → glue → verify → bench end-to-end through the Panama backend.
- **`String` method purity whitelist.** 28 provably-pure `String` methods (`length`, `charAt`, `substring`, `equals`, `hashCode`, …) no longer disqualify a kernel. Regex/charset/locale-sensitive ones (`split`, `getBytes`, `intern`, `matches`, …) stay rejected with explicit reasons.
- **FFM fallback dispatcher.** `--backend ffm` glue now keeps the original Java implementation behind the same runtime switch as JNI (`-Dkernelport.native=false` or `setUseNative(false)`). The native library loads lazily on the first native-call attempt (never in a static initializer), so a *missing* `libkernels_ffm.so` falls back to the Java original with a one-line stderr note instead of failing class initialization.

## What's new in v0.4

- **Deterministic `throw` is purity-compatible.** Argument-validation guards like `if (n < 0) throw new IllegalArgumentException(...)` no longer disqualify a kernel. The rule: `throw new <unchecked-type>(pure args)` is fine; throwing pre-built objects, rethrowing caught exceptions, or checked exceptions is still rejected with reasons.
- **`double[][]` / `int[][]` and `String` are supported** in glue/verify/bench. 2-D arrays are flattened row-major to one contiguous array (+ rows/cols) before the native call — a single JNI pin instead of per-row marshalling; the flattening contract is spelled out in the LLM prompt material.
- **Overload resolution.** `--method` accepts an optional signature suffix, e.g. `--method com.example.K.foo(int,double)`; ambiguous bare names now list the available overloads instead of just erroring out.
- **Per-type float tolerance in verify.** `float` compares at 1e-6 relative, `double` at 1e-12 (configurable: `--tol-float`, `--tol-double`). No more flaky false failures on float32 accumulation — verify stays the trust anchor.
- **Panama FFM glue backend** (`glue --backend ffm`, also on `verify`/`bench`). Generates a pure-C ABI export plus `java.lang.foreign` bindings — no hand-written JNI wrapper, no local-ref table, no manual exception checks. Requires JDK 22+ (25 LTS recommended); refuses clearly on older JDKs. The generated `build_ffm.sh` resolves paths from its own location (safe to rename/move the glue dir); the dispatcher loads the native library lazily and falls back to the Java original if it is missing.

## Requirements

- Python ≥ 3.10
- A JDK (17+) and `g++` on `PATH` — needed for `verify` / `bench` (they compile and run both sides)
- `--backend ffm` needs JDK 22+ (25 LTS recommended); it refuses with a clear error on older JDKs
- `mcp` Python package — only needed for `kernel-port mcp` (`pip install 'kernel-port-agent[mcp]'`)
- `cargo` — only needed for `--backend rust`

## Install

```bash
pip install kernel-port-agent
# or from source:
pip install -e .
```

**For agents:** this repo ships a [SKILL.md](SKILL.md) (Agent Skills format) so skill-compatible agents can discover and drive the toolchain directly.

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

Overloads are resolved by suffix; the FFM backend skips JNI entirely:

```bash
# disambiguate an overload
kernel-port extract ./my-java-project --method 'com.example.K.foo(int,double)'

# Panama FFM backend (JDK 22+, 25 LTS recommended)
kernel-port glue ./my-java-project --method com.example.Kernels.dot \
    --backend ffm --out ./port/dot-ffm
# ... agent fills kernels_ffm.c ...
kernel-port verify ./my-java-project --method com.example.Kernels.dot \
    --backend ffm --glue-dir ./port/dot-ffm --cases 100
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

1. `mine` ranks candidates so the agent ports the highest-value kernel first. Beyond lone methods it finds **function clusters** (an entry method plus the pure helpers it calls — translated as one unit) and **portable classes** (stateless utility classes, or immutable classes with all-`final` fields; mutable classes are rejected with reasons). `--no-clusters` restores pure method-level listing; `--profile <collapsed.txt>` re-ranks by measured CPU time from an async-profiler collapsed-stack file instead of static heuristics. (`--lang python` has no clusters: `--clusters`/`--no-clusters` warn on stderr and are ignored.)
2. `extract` gives the LLM exactly what it needs: the type signature, the purity verdict, and the full source of the method plus its dependency closure (helpers, constants). Impure methods are refused with reasons. `--class` extracts a whole class. When a previous `verify` failed, `--with-counterexamples` appends the failing input/output pairs to the prompt material so the next translation targets the failure. Verified translations are fingerprinted (normalized-AST hash) and cached in `~/.cache/kernel-port` — `extract` reuses them instead of re-translating (`--no-cache` to skip).
3. `glue` means the LLM only translates the *algorithm*, never the FFI boilerplate. Array parameters the kernel may write are copied back after the call, while read-only arrays skip the copy-back (`JNI_ABORT`) — safe by default. The generated `*Native.java` is a **dispatcher**: the Java original stays callable behind `-Dkernelport.native=false` (or `setUseNative(false)`), so a bad native port rolls back instantly. `--no-fallback` restores the plain native declaration. `--class` generates glue for every method of a portable class (with `--lang python` glue is function-level: `--class` warns on stderr and is ignored).
4. `verify` generates random inputs **plus boundary cases** (empty arrays, 0, NaN/±Infinity, `Integer.MIN_VALUE/MAX_VALUE`, …) — the corners where LLM translations most often break — runs both sides, and compares with float tolerance. Inputs the Java original itself rejects are skipped (out of domain); a native crash or mismatch fails loudly. Failing cases are persisted for `--with-counterexamples`.
5. `bench` reports median per-call time for both sides, **splits out the call overhead** (measured with an empty native call) from **pure native compute** time, and tells you *why* a port is slower — overhead-dominated (kernel too small) vs. compute-dominated (bad translation). `--sizes 16,64,256,1024` runs a **break-even analysis**: the input size at which porting starts paying off (`--lang python` too: per-size table + break-even, with `--list-len` (default 64) setting the list length when `--sizes` is absent). `--rigorous` uses JMH / Google Benchmark when installed, otherwise an enhanced wall-clock (IQR outlier trimming + 95% CI) with an honest notice (`--rigorous` and `--html` are Java-only in the `--lang python` bench path). `--html` writes a self-contained report (inline SVG charts, no external deps).
6. `pipeline` chains it all: mine → extract → glue → (agent's `--translated-body`, `#include`s hoisted automatically) → verify → bench, with `--candidates-dir` to try N translations and auto-pick the fastest fully-passing one, `--cache-dir`/`--no-cache` for the fingerprint cache, `--backend jni|ffm|rust` to choose the glue backend, and `--html`/`--json` reports. Without `--translated-body` it stops after glue and prints exact next steps for the agent.
7. `kernel-port mcp` starts a **stdio MCP server** exposing `mine`, `extract`, `glue`, `verify`, `bench`, `pipeline` as tools, so any MCP-capable agent can call the toolchain natively.

## Design notes

- **Purity is conservative.** False negatives (rejecting a pure method with a stated reason) are acceptable; false positives (claiming impure code is pure) are treated as bugs. The allowlist is deliberately small: `Math.*`/`StrictMath.*` minus `random()`, array creation, `static final` constants, reads of `final` instance fields (immutable-class context). Since v0.4, **deterministic throws** are allowed: `throw new IllegalArgumentException(...)` with pure arguments doesn't break purity; throwing pre-built exception objects, rethrowing caught exceptions, and checked exceptions still do.
- **Scope:** static methods; `void`, primitives, 1-D primitive arrays, plus (v0.4) `double[][]`/`int[][]` (flattened row-major) and `String` (UTF-8 via `GetStringUTFChars`/`NewStringUTF`; Modified-UTF-8 caveat documented). Objects, other 2-D types and generics are rejected with a clear error. Classes: only stateless or immutable ones are portable as a unit.
- Overloaded methods are resolved by `--method` suffix, e.g. `com.example.K.foo(int,double)` (arity first, then exact type match).
- Parsing is tree-sitter based (same infrastructure family as `deadcode-hunter`).

## License

MIT
