# kernel-port dogfood: JOML (v0.8)

First real-repository validation. Until v0.8 the tool had only ever run on
hand-made fixtures. This note records what broke on real code, what was
fixed, and the end-to-end numbers.

## Repo

[JOML](https://github.com/JOML-CI/JOML) (Java OpenGL Math Library), cloned
2026-10-18 (`--depth 1`). 121 Java files, 662 methods, real-world numeric
code: vectors, matrices, intersection tests, noise. Maven layout
(`src/main/java`), dependency-free (pure JDK).

## What `mine` found

```
mine ~/joml/src
# 662 methods, 27 function clusters, 136 classes -- 53s (was 84-165s before
# the v0.8 file-bytes cache)
```

Top pure primitive-only candidates are exactly the right shape: static
intersection routines in `org.joml.Intersectiond` / `Intersectionf`
(`testRayTriangle`, `intersectRayTriangle`, `testCircleTriangle`, …).
Notable observations:

- **Zero portable classes.** JOML is a mutable linear-algebra library, so
  every class is correctly rejected (e.g. `Matrix4f`: mutable fields;
  `org.joml.Math`: reads a `static final` lookup table that is
  conditionally initialized -- the conservative rejection is correct).
- **Test sources were mined as candidates** (`org.joml.test.TestUtil`).
  Fixed: `src/test`/`test`/`tests` dirs are now excluded from scanning.
- **Overloads handled.** `findClosestPointOnTriangle` has 4 overloads;
  each is listed with its full signature and resolvable via
  `--method 'K.m(int,double)'`.

## End-to-end run

Candidate: `org.joml.Intersectiond.testRayTriangle` (Möller–Trumbore
ray-triangle test, 16 `double` params → `boolean`, pure, self-contained).

```
extract  -> PURE, closure none, prompt material emitted
glue     -> JNI skeleton + lint clean + fallback dispatcher
verify   -> PASS random 100/100, boundary 24/24
bench    -> Java 270ns vs JNI C++ 668ns median
```

The bench is the honest story the tool exists to tell: **2.47x SLOWER --
not worth porting**, with the overhead split correctly diagnosing that the
C++ compute itself (636ns) is slower than Java (270ns), not the JNI
boundary (32ns). The translation was a mechanical 1:1 port; the JIT wins
on this small kernel. A tool that only ever says "faster!" would be lying.

## Bugs found and fixed (each with regression tests)

1. **`build_env` compiled the whole repo.** `verify`/`bench` ran
   `javac` over every `*.java` under the repo root: JOML's `src/test`
   needs JUnit and several files need annotation-only deps
   (`org.jspecify.annotations`) -- 898 errors. Fixed in two layers:
   compile just the kernel file with `-sourcepath` (javac pulls only
   referenced deps), and prefer a **synthesized minimal kernel file**
   (kernel + dependency closure + needed imports only) so `javac` never
   typechecks environmentally-broken siblings at all
   (`extractor.render_minimal_sources`; falls back to the repo file when
   synthesis is unsafe). Bonus: the harness now differentially tests
   exactly the extracted kernel -- the same code the agent translated.
2. **File bytes re-read per method.** `mine` read each file ~95x
   (11,541 reads / 121 files). Fixed: `java_ast.source_bytes`
   (mtime+size-validated cache) shared by miner/purity/extractor/jni;
   full `mine` on JOML went from 84-165s to ~53s.
3. **Enum methods invisible.** tree-sitter nests them under
   `enum_body_declarations`; `_parse_class` never descended, so enum
   static methods were missing from mine/extract entirely. Fixed.
4. **Test sources mined.** `scan_repo` and the parse-cache loop now skip
   `test`/`tests` directories.

`--profile` was not exercised: there is no representative JOML workload
harness handy, and profile-guided mining is documented as opt-in for when
the caller has one. Static ranking remains the default path.

## Environment note

The dogfood run needs a JDK 22+ for the FFM backend only; JNI/verify/bench
work on any JDK 17+. This VM's system JDK vanished after a redeploy; the
run used a local Temurin 25 (`~/workspace/tools/jdk-25.0.4.1+1`).
