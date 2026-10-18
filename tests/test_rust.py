"""Tests for the Rust (cdylib) glue backend.

Unit tests assert the structure of the generated ``Cargo.toml`` /
``src/lib.rs``. End-to-end tests build the real cdylib with cargo and
drive it through the EXISTING JNI line-protocol harness: the Rust crate
exports the same ``Java_<Cls>Native_<name>Native`` symbols as the C++
backend, so ``verify``/``bench`` need zero harness changes -- only the
build step differs (``cargo build --release`` instead of ``g++``).

The e2e tests need cargo (found at ``~/.cargo/bin`` on this dev
machine); they are gated with pytest.mark.skipif so the suite stays
green on cargo-less machines.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from kernel_port import harness, jni
from kernel_port import java_ast as J
from kernel_port import rust_glue

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "rust_kernels"

# Hand-written Rust translations, standing in for the calling agent's LLM
# step. Helper signatures follow the generated conventions: T[] params are
# &[T] (read-only) or &mut [T] (written through, copied back after the
# call); slices carry their length, so there is no p_<name>_len.
RUST_TRANSLATIONS = {
    "com.example.RustKernels.dot": (
        "    let mut s = 0.0;\n"
        "    for i in 0..p_a.len() {\n"
        "        s += p_a[i] * p_b[i];\n"
        "    }\n"
        "    s"
    ),
    "com.example.RustKernels.scale": (
        "    for i in 0..p_a.len() {\n"
        "        p_a[i] *= p_f;\n"
        "    }"
    ),
}

_ALL_METHODS = [
    "com.example.RustKernels.dot",
    "com.example.RustKernels.scale",
    "com.example.RustKernels.greet",
    "com.example.RustKernels.isPositive",
    "com.example.RustKernels.countTrue",
    "com.example.RustKernels.matAdd",
    "com.example.RustKernels.iota",
]

needs_cargo = pytest.mark.skipif(
    rust_glue._find_cargo() is None, reason="e2e Rust tests need cargo"
)


@pytest.fixture()
def repo():
    return J.scan_repo(FIXTURE_REPO)


def _fill_rust_translation(lib_rs: Path, body: str) -> None:
    """Replace the TODO(agent) block in src/lib.rs with `body`."""
    text = lib_rs.read_text(encoding="utf-8")
    start_marker = "    // ==== BEGIN TRANSLATED BODY ===="
    end_marker = "    // ==== END TRANSLATED BODY ===="
    start = text.index(start_marker)
    end = text.index(end_marker) + len(end_marker)
    lib_rs.write_text(
        text[:start] + body.rstrip() + "\n" + text[end:], encoding="utf-8"
    )


def _method_info(jtype: str, params: list[tuple[str, str]], is_static: bool = True):
    return J.MethodInfo(
        qualname="com.example.X.m",
        class_qualname="com.example.X",
        name="m",
        jtype=jtype,
        params=[J.Param(name=n, jtype=t) for n, t in params],
        modifiers={"static"} if is_static else set(),
        is_static=is_static,
        is_constructor=False,
        file="X.java",
        lineno=1,
        end_lineno=2,
        source="",
    )


# --------------------------------------------------------------------------
# plan()
# --------------------------------------------------------------------------


def test_plan_names(repo):
    m = repo.find_method("com.example.RustKernels.dot")
    rp = rust_glue.plan(m)
    assert rp.native_class == "RustKernelsNative"
    # default: dispatcher mode -> <name>Native entry point
    assert rp.jni_func == "Java_RustKernelsNative_dotNative"
    assert rp.helper_name == "kernel_dot"
    assert rp.flat is False


def test_plan_fallback_false_symbol(repo):
    m = repo.find_method("com.example.RustKernels.dot")
    rp = rust_glue.plan(m, fallback=False)
    assert rp.jni_func == "Java_RustKernelsNative_dot"


def test_plan_flat_2d(repo):
    m = repo.find_method("com.example.RustKernels.matAdd")
    rp = rust_glue.plan(m)
    assert rp.flat is True
    assert rp.jni_func == "Java_RustKernelsNative_matAddFlat"


def test_plan_rejects_non_static():
    with pytest.raises(ValueError, match="static"):
        rust_glue.plan(_method_info("void", [], is_static=False))


def test_plan_rejects_unsupported_type():
    with pytest.raises(ValueError, match="Rust glue"):
        rust_glue.plan(_method_info("void", [("o", "Object")]))


# --------------------------------------------------------------------------
# render_lib_rs()
# --------------------------------------------------------------------------


def test_lib_rs_mangled_symbol_and_markers(repo):
    m = repo.find_method("com.example.RustKernels.dot")
    src = rust_glue.render_lib_rs(rust_glue.plan(m))
    assert "#[no_mangle]" in src
    assert 'pub extern "system" fn Java_RustKernelsNative_dotNative' in src
    assert "fn kernel_dot(" in src
    assert "TODO(agent)" in src
    assert "// ==== BEGIN TRANSLATED BODY ====" in src
    assert "// ==== END TRANSLATED BODY ====" in src


@pytest.mark.parametrize(
    ("qualname", "signature"),
    [
        ("com.example.RustKernels.dot",
         "fn kernel_dot(p_a: &[f64], p_b: &[f64]) -> f64"),
        ("com.example.RustKernels.scale",
         "fn kernel_scale(p_a: &mut [f64], p_f: f64)"),
        ("com.example.RustKernels.greet",
         "fn kernel_greet(p_name: &str) -> String"),
        ("com.example.RustKernels.isPositive",
         "fn kernel_isPositive(p_x: i32) -> bool"),
        ("com.example.RustKernels.countTrue",
         "fn kernel_countTrue(p_flags: &[bool]) -> i32"),
        ("com.example.RustKernels.iota",
         "fn kernel_iota(p_n: i32) -> Vec<i32>"),
        ("com.example.RustKernels.matAdd",
         "fn kernel_matAdd(p_a: &[f64], p_a_rows: i32, p_a_cols: i32, "
         "p_b: &[f64], p_b_rows: i32, p_b_cols: i32, "
         "p_out_rows: &mut i32, p_out_cols: &mut i32) -> Vec<f64>"),
    ],
)
def test_lib_rs_signature_mapping(repo, qualname, signature):
    m = repo.find_method(qualname)
    src = rust_glue.render_lib_rs(rust_glue.plan(m, fallback=False))
    assert signature in src


def test_lib_rs_uses_jni_crate_idioms(repo):
    m = repo.find_method("com.example.RustKernels.dot")
    src = rust_glue.render_lib_rs(rust_glue.plan(m))
    assert "use jni::objects::" in src
    assert "use jni::sys::" in src
    assert "use jni::JNIEnv;" in src
    assert "macro_rules! jni_try" in src
    # no C/C++ idioms: this is pure Rust, the jni crate does the marshaling
    assert "#include <jni.h>" not in src
    assert "JNIEXPORT" not in src
    assert "JNIEnv *" not in src


def test_lib_rs_written_array_copied_back(repo):
    # scale() mutates a[]: the entry point must copy the buffer back.
    m = repo.find_method("com.example.RustKernels.scale")
    src = rust_glue.render_lib_rs(rust_glue.plan(m, fallback=False))
    assert "set_double_array_region" in src


def test_lib_rs_readonly_array_not_copied_back(repo):
    # dot() never writes its arrays: no copy-back region calls.
    m = repo.find_method("com.example.RustKernels.dot")
    src = rust_glue.render_lib_rs(rust_glue.plan(m, fallback=False))
    assert "set_double_array_region" not in src
    assert "get_double_array_region" in src


# --------------------------------------------------------------------------
# render_cargo_toml() / render_build_script()
# --------------------------------------------------------------------------


def test_cargo_toml(repo):
    m = repo.find_method("com.example.RustKernels.dot")
    rp = rust_glue.plan(m)
    toml = rust_glue.render_cargo_toml(rp)
    # per-kernel crate name (foot-gun fix), in both [package] and [lib]
    assert f'name = "{rp.lib_name}"' in toml
    assert toml.count(f'name = "{rp.lib_name}"') == 2
    assert 'crate-type = ["cdylib"]' in toml
    assert 'jni = "0.21"' in toml


def test_render_build_script(repo, tmp_path):
    m = repo.find_method("com.example.RustKernels.dot")
    rp = rust_glue.plan(m)
    sh = rust_glue.render_build_script(rp, tmp_path / "out")
    assert "build --release" in sh  # via "$CARGO_BIN" (CARGO env override)
    assert "CARGO_BIN" in sh
    assert f"lib{rp.lib_name}.so" in sh
    assert "RustKernelsNative.java" in sh


# --------------------------------------------------------------------------
# generate()
# --------------------------------------------------------------------------


def test_generate_writes_four_files(repo, tmp_path):
    m = repo.find_method("com.example.RustKernels.dot")
    files = rust_glue.generate(m, tmp_path / "glue")
    assert set(files) == {"java", "rust", "cargo", "build"}
    assert files["java"].name == "RustKernelsNative.java"
    assert files["rust"] == tmp_path / "glue" / "src" / "lib.rs"
    assert files["cargo"].name == "Cargo.toml"
    assert files["build"].name == "build_rust.sh"
    for p in files.values():
        assert p.exists()
    assert files["build"].stat().st_mode & 0o111  # executable


def test_generate_java_differs_from_jni_only_in_loadlibrary(repo, tmp_path):
    # The Rust backend reuses the JNI backend's Java dispatcher verbatim
    # EXCEPT the System.loadLibrary line, which carries the per-kernel
    # cdylib name (foot-gun fix): the harness line protocol is unchanged.
    m = repo.find_method("com.example.RustKernels.dot")
    rp = rust_glue.plan(m)
    rust_files = rust_glue.generate(m, tmp_path / "rust")
    jni_files = jni.generate(m, tmp_path / "jni")
    rust_java = rust_files["java"].read_text(encoding="utf-8")
    jni_java = jni_files["java"].read_text(encoding="utf-8")
    assert f'System.loadLibrary("{rp.lib_name}")' in rust_java
    assert 'System.loadLibrary("kernels")' in jni_java
    rust_norm = rust_java.replace(
        f'System.loadLibrary("{rp.lib_name}")', 'System.loadLibrary("kernels")'
    )
    assert rust_norm == jni_java


def test_generate_fallback_flag_is_wired(repo, tmp_path):
    m = repo.find_method("com.example.RustKernels.dot")
    default_files = rust_glue.generate(m, tmp_path / "fb")  # 2-arg call
    bare_files = rust_glue.generate(m, tmp_path / "bare", fallback=False)
    assert "USE_NATIVE" in default_files["java"].read_text(encoding="utf-8")
    assert "USE_NATIVE" not in bare_files["java"].read_text(encoding="utf-8")
    default_rs = default_files["rust"].read_text(encoding="utf-8")
    bare_rs = bare_files["rust"].read_text(encoding="utf-8")
    assert "fn Java_RustKernelsNative_dotNative" in default_rs
    assert "fn Java_RustKernelsNative_dot<" in bare_rs
    assert "dotNative" not in bare_rs


def test_generate_java_has_fallback_switch(repo, tmp_path):
    """Parity with the JNI/FFM backends: the generated Java dispatcher
    rolls back to the original Java kernel via -Dkernelport.native=false
    or setUseNative(false)."""
    m = repo.find_method("com.example.RustKernels.dot")
    files = rust_glue.generate(m, tmp_path / "fb")
    src = files["java"].read_text(encoding="utf-8")
    assert (
        'Boolean.parseBoolean(System.getProperty("kernelport.native", "true"))'
        in src
    )
    assert "public static void setUseNative(boolean useNative)" in src
    assert "java -Dkernelport.native=false" in src
    assert "RustKernelsNative.setUseNative(false)" in src
    # loadLibrary carries the per-kernel cdylib name (foot-gun fix)
    rp = rust_glue.plan(m)
    assert f'System.loadLibrary("{rp.lib_name}")' in src


# --------------------------------------------------------------------------
# compile_rust()
# --------------------------------------------------------------------------


def test_compile_rust_missing_cargo(monkeypatch):
    monkeypatch.setattr(rust_glue, "_find_cargo", lambda: None)
    with pytest.raises(RuntimeError, match="cargo not found"):
        rust_glue.compile_rust("/tmp/does-not-matter")


def _fake_cargo_run(monkeypatch, *, returncode=0, stderr="", raise_timeout=False):
    """Patch subprocess.run and _find_cargo; return captured kwargs.

    Materialize the expected cdylib with _materialize_cdylib so
    compile_rust's post-build existence check passes.
    """
    captured = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["env"] = kw.get("env")
        if raise_timeout:
            raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))
        return subprocess.CompletedProcess(
            cmd, returncode, stdout="", stderr=stderr
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(rust_glue, "_find_cargo", lambda: Path("/fake/cargo"))
    monkeypatch.setattr(rust_glue, "_jni_crate_cached", lambda: False)
    return captured


def _materialize_cdylib(target_dir: Path) -> Path:
    lib = target_dir / "release" / rust_glue._cdylib_name()
    lib.parent.mkdir(parents=True, exist_ok=True)
    lib.write_bytes(b"fake-cdylib")
    return lib


def test_compile_rust_pins_target_dir_next_to_build(monkeypatch, tmp_path):
    """CARGO_TARGET_DIR defaults to <out_dir>/target for the cargo run, so
    artifacts live next to the build and never touch an unrelated dir."""
    captured = _fake_cargo_run(monkeypatch)
    monkeypatch.delenv("CARGO_TARGET_DIR", raising=False)
    out = tmp_path / "out"
    lib = _materialize_cdylib(out / "target")
    assert rust_glue.compile_rust(out) == lib
    assert captured["env"]["CARGO_TARGET_DIR"] == str(out / "target")


def test_compile_rust_respects_existing_target_dir(monkeypatch, tmp_path):
    """An explicitly set CARGO_TARGET_DIR (e.g. a shared target dir across
    crates) is respected, not overridden."""
    shared = tmp_path / "shared-target"
    monkeypatch.setenv("CARGO_TARGET_DIR", str(shared))
    captured = _fake_cargo_run(monkeypatch)
    lib = _materialize_cdylib(shared)
    assert rust_glue.compile_rust(tmp_path / "out") == lib
    assert captured["env"]["CARGO_TARGET_DIR"] == str(shared)


def test_compile_rust_empty_target_dir_treated_as_unset(monkeypatch, tmp_path):
    """An empty CARGO_TARGET_DIR is treated as unset (cargo does the same),
    so artifacts still land in <out_dir>/target."""
    monkeypatch.setenv("CARGO_TARGET_DIR", "")
    captured = _fake_cargo_run(monkeypatch)
    out = tmp_path / "out"
    lib = _materialize_cdylib(out / "target")
    assert rust_glue.compile_rust(out) == lib
    assert captured["env"]["CARGO_TARGET_DIR"] == str(out / "target")


def test_compile_rust_offline_when_cache_warm(monkeypatch, tmp_path):
    """--offline is passed when the jni 0.21 crate is cached, so a warm
    machine never stalls on the network."""
    captured = _fake_cargo_run(monkeypatch)
    monkeypatch.delenv("CARGO_TARGET_DIR", raising=False)
    monkeypatch.delenv("CARGO_NET_OFFLINE", raising=False)
    monkeypatch.setattr(rust_glue, "_jni_crate_cached", lambda: True)
    out = tmp_path / "out"
    _materialize_cdylib(out / "target")
    rust_glue.compile_rust(out)
    assert "--offline" in captured["cmd"]


def test_compile_rust_online_when_cache_cold(monkeypatch, tmp_path):
    """When the crate cache is cold, cargo runs online so it can download
    the jni dependency."""
    captured = _fake_cargo_run(monkeypatch)
    monkeypatch.delenv("CARGO_TARGET_DIR", raising=False)
    out = tmp_path / "out"
    _materialize_cdylib(out / "target")
    rust_glue.compile_rust(out)
    assert "--offline" not in captured["cmd"]


def test_compile_rust_respects_cargo_net_offline_env(monkeypatch, tmp_path):
    """An explicit CARGO_NET_OFFLINE overrides the cache-based --offline
    choice; the env value is passed through to cargo."""
    monkeypatch.setenv("CARGO_NET_OFFLINE", "false")
    captured = _fake_cargo_run(monkeypatch)
    monkeypatch.setattr(rust_glue, "_jni_crate_cached", lambda: True)
    monkeypatch.delenv("CARGO_TARGET_DIR", raising=False)
    out = tmp_path / "out"
    _materialize_cdylib(out / "target")
    rust_glue.compile_rust(out)
    assert "--offline" not in captured["cmd"]
    assert captured["env"]["CARGO_NET_OFFLINE"] == "false"


def test_jni_crate_cached_matches_021_only(tmp_path, monkeypatch):
    """_jni_crate_cached matches the pinned 0.21 series, not 0.22, in both
    the .crate cache and the extracted src cache."""
    home = tmp_path / "cargo-home"
    cache = home / "registry" / "cache" / "index.crates.io-6faca5"
    src = home / "registry" / "src" / "index.crates.io-6faca5"
    cache.mkdir(parents=True)
    src.mkdir(parents=True)
    monkeypatch.setenv("CARGO_HOME", str(home))
    (cache / "jni-0.22.4.crate").write_bytes(b"x")
    (src / "jni-0.22.4").mkdir()
    assert rust_glue._jni_crate_cached() is False
    (cache / "jni-0.21.1.crate").write_bytes(b"x")
    assert rust_glue._jni_crate_cached() is True
    (cache / "jni-0.21.1.crate").unlink()
    (src / "jni-0.21.1").mkdir()
    assert rust_glue._jni_crate_cached() is True


def test_compile_rust_error_shows_tail_not_dump(monkeypatch, tmp_path):
    """A failed build raises with the tail of cargo's stderr (where cargo
    prints the actionable error), not the full multi-page dump."""
    stderr = "\n".join(f"warning: noise {i}" for i in range(100))
    stderr += "\nerror[E0308]: mismatched types\nnote: boom"
    _fake_cargo_run(monkeypatch, returncode=101, stderr=stderr)
    with pytest.raises(RuntimeError) as excinfo:
        rust_glue.compile_rust(tmp_path / "out")
    msg = str(excinfo.value)
    assert "exit 101" in msg
    assert "error[E0308]: mismatched types" in msg
    assert "note: boom" in msg
    assert "warning: noise 0" not in msg  # head of the dump is cut


def test_compile_rust_timeout_surfaces_readably(monkeypatch, tmp_path):
    _fake_cargo_run(monkeypatch, raise_timeout=True)
    with pytest.raises(RuntimeError, match="timed out after"):
        rust_glue.compile_rust(tmp_path / "out", timeout=1)


# --------------------------------------------------------------------------
# end-to-end: cargo build + JNI harness verify/bench
# --------------------------------------------------------------------------


@needs_cargo
def test_e2e_skeletons_compile(repo, tmp_path, monkeypatch):
    """Every fixture kernel's placeholder lib.rs must compile to a cdylib.

    A shared CARGO_TARGET_DIR keeps the jni dependency build to a single
    compilation across the per-method crates.
    """
    monkeypatch.setenv("CARGO_TARGET_DIR", str(tmp_path / "cargo-target"))
    for qualname in _ALL_METHODS:
        m = repo.find_method(qualname)
        rp = rust_glue.plan(m)
        out = tmp_path / "crates" / m.name
        files = rust_glue.generate(m, out, fallback=False)
        lib = rust_glue.compile_rust(out)
        assert lib.exists()
        assert lib.name == f"lib{rp.lib_name}.so"


def _e2e_env(repo, qualname: str, tmp_path: Path, translation: str):
    """Generate (dispatcher mode), fill the Rust translation, cargo-build,
    and assemble a harness BuiltEnv whose ``libkernels.so`` is the Rust
    cdylib. Returns (method, env)."""
    m = repo.find_method(qualname)
    assert m is not None
    out = tmp_path / "rust"
    files = rust_glue.generate(m, out)  # fallback=True default
    _fill_rust_translation(files["rust"], translation)
    so = rust_glue.compile_rust(out)
    work = tmp_path / "work"
    classes = work / "classes"
    classes.mkdir(parents=True)
    java_sources = sorted(FIXTURE_REPO.rglob("*.java"))
    hj = work / "HarnessJava.java"
    hn = work / "HarnessJni.java"
    hj.write_text(
        harness.render_harness(m, "java", "HarnessJava"), encoding="utf-8"
    )
    hn.write_text(
        harness.render_harness(m, "jni", "HarnessJni"), encoding="utf-8"
    )
    harness.compile_java(java_sources + [hj, hn, files["java"]], classes)
    shutil.copy(so, work / so.name)  # per-kernel cdylib name (lib_name_for)
    env = harness.BuiltEnv(work=work, classes=classes, method=m, backend="jni")
    return m, env


@needs_cargo
def test_e2e_dot_verify_and_bench(repo, tmp_path):
    m, env = _e2e_env(
        repo, "com.example.RustKernels.dot", tmp_path,
        RUST_TRANSLATIONS["com.example.RustKernels.dot"],
    )
    report = harness.run_verify(env, cases=30, boundary=False)
    assert report.all_passed, [
        (c.index, c.detail, c.java_result, c.jni_result)
        for c in report.cases if not c.ok
    ]
    assert report.passed > 0
    bench = harness.run_bench(
        env, iters=300, warmup=100, repeats=2, overhead=False
    )
    assert bench.java_per_call_ns > 0
    assert bench.jni_per_call_ns > 0
    assert bench.speedup > 0


@needs_cargo
def test_e2e_scale_verify(repo, tmp_path):
    # void kernel: differential testing compares the mutated array, so
    # this exercises the &mut [f64] copy-back path end to end.
    m, env = _e2e_env(
        repo, "com.example.RustKernels.scale", tmp_path,
        RUST_TRANSLATIONS["com.example.RustKernels.scale"],
    )
    report = harness.run_verify(env, cases=30, boundary=False)
    assert report.all_passed, [
        (c.index, c.detail, c.java_result, c.jni_result)
        for c in report.cases if not c.ok
    ]
    assert report.passed > 0


@needs_cargo
def test_e2e_fallback_toggle_native_on_and_off(repo, tmp_path, monkeypatch):
    """Dispatcher toggle end to end: the unfilled skeleton's placeholder
    (0.0) distinguishes the native Rust path from the Java rollback, so
    native-on must print 0.0 while ``-Dkernelport.native=false`` and
    ``setUseNative(false)`` must print the original Java result 32.0."""
    m = repo.find_method("com.example.RustKernels.dot")
    out = tmp_path / "rust"
    files = rust_glue.generate(m, out)  # fallback=True (default)
    monkeypatch.setenv("CARGO_TARGET_DIR", str(tmp_path / "cargo-target"))
    so = rust_glue.compile_rust(out)  # skeleton: placeholder returns 0.0
    work = tmp_path / "work"
    classes = work / "classes"
    classes.mkdir(parents=True)
    java_sources = sorted(FIXTURE_REPO.rglob("*.java"))
    runner = work / "RunDotToggle.java"
    runner.write_text(
        "public class RunDotToggle {\n"
        "    public static void main(String[] a) {\n"
        '        if (a.length > 0 && a[0].equals("off")) {\n'
        "            RustKernelsNative.setUseNative(false);\n"
        "        }\n"
        "        double r = RustKernelsNative.dot(\n"
        "            new double[]{1.0, 2.0, 3.0}, new double[]{4.0, 5.0, 6.0});\n"
        '        System.out.println("DOT " + r);\n'
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    harness.compile_java(java_sources + [runner, files["java"]], classes)
    shutil.copy(so, work / so.name)  # per-kernel cdylib name (lib_name_for)

    java = shutil.which("java")
    assert java is not None

    def run_dot(*args, jvm_args=()):
        return subprocess.run(
            [
                java, f"-Djava.library.path={work}", "-cp", str(classes),
                *jvm_args, "RunDotToggle", *args,
            ],
            capture_output=True, text=True, timeout=120,
        )

    # native on (default): the Rust placeholder runs -> 0.0. This proves
    # the dispatcher actually reaches the cdylib.
    native = run_dot()
    assert native.returncode == 0, native.stderr
    assert "DOT 0.0" in native.stdout

    # -Dkernelport.native=false: rolls back to the original Java kernel.
    flagged = run_dot(jvm_args=["-Dkernelport.native=false"])
    assert flagged.returncode == 0, flagged.stderr
    assert "DOT 32.0" in flagged.stdout

    # setUseNative(false): the programmatic rollback path.
    programmatic = run_dot("off")
    assert programmatic.returncode == 0, programmatic.stderr
    assert "DOT 32.0" in programmatic.stdout


# --------------------------------------------------------------------------
# v0.10 workstream C: unique per-kernel crate names (stale-.so foot-gun fix)
# --------------------------------------------------------------------------


def test_lib_name_for_is_deterministic_and_sanitized():
    n1 = rust_glue.lib_name_for("com.example.RustKernels.dot")
    n2 = rust_glue.lib_name_for("com.example.RustKernels.dot")
    assert n1 == n2  # deterministic: rebuilds reproduce the name
    assert n1.startswith("kp_")
    assert all(c in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in n1)
    assert len(n1) <= 64


def test_lib_name_for_unique_across_sanitization_collisions():
    # "com.Foo.bar" and "com.Foo_bar" sanitize identically; the sha1
    # suffix must still keep them distinct.
    a = rust_glue.lib_name_for("com.Foo.bar")
    b = rust_glue.lib_name_for("com.Foo_bar")
    assert a != b
    assert rust_glue.lib_name_for("com.example.A.m") != rust_glue.lib_name_for(
        "com.example.B.m"
    )


def test_lib_name_for_truncates_long_qualnames():
    long_qn = "com.example." + "VeryLongPackageName" * 10 + ".method"
    name = rust_glue.lib_name_for(long_qn)
    assert len(name) <= 64
    assert name.startswith("kp_")


def test_plan_carries_lib_name(repo):
    m = repo.find_method("com.example.RustKernels.dot")
    rp = rust_glue.plan(m)
    assert rp.lib_name == rust_glue.lib_name_for("com.example.RustKernels.dot")


def test_crate_name_from_toml_old_and_new(tmp_path):
    # old (pre-unique-name) glue dirs said name = "kernels"
    old = tmp_path / "old.toml"
    old.write_text('[package]\nname = "kernels"\nversion = "0.1.0"\n', encoding="utf-8")
    assert rust_glue.crate_name_from_toml(old) == "kernels"
    new = tmp_path / "new.toml"
    new.write_text(
        '[package]\nname = "kp_com_example_x_m_12345678"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    assert rust_glue.crate_name_from_toml(new) == "kp_com_example_x_m_12345678"
    # missing file -> legacy default
    assert rust_glue.crate_name_from_toml(tmp_path / "nope.toml") == "kernels"


def test_generate_loadlibrary_matches_crate_name_all_cases(repo, tmp_path):
    """System.loadLibrary must name the generated crate in every render
    mode (dispatcher on/off): the JVM resolves lib<name>.so, so any
    mismatch is an UnsatisfiedLinkError at best, a wrong .so at worst."""
    for qualname in ("com.example.RustKernels.dot", "com.example.RustKernels.matAdd"):
        m = repo.find_method(qualname)
        for fallback in (True, False):
            rp = rust_glue.plan(m, fallback=fallback)
            files = rust_glue.generate(m, tmp_path / m.name / str(fallback), fallback=fallback)
            java = files["java"].read_text(encoding="utf-8")
            cargo = files["cargo"].read_text(encoding="utf-8")
            # [package] name, [lib] name, and loadLibrary all agree
            assert cargo.count(f'name = "{rp.lib_name}"') == 2
            assert f'System.loadLibrary("{rp.lib_name}")' in java
            # and the cdylib file name compile_rust will produce
            assert rust_glue._cdylib_name(rp.lib_name) == f"lib{rp.lib_name}.so"


def test_compile_rust_reads_unique_name_from_manifest(monkeypatch, tmp_path, repo):
    """compile_rust resolves the cdylib name from the glue Cargo.toml, so
    it builds/copies the per-kernel artifact, not a fixed libkernels.so."""
    m = repo.find_method("com.example.RustKernels.dot")
    rp = rust_glue.plan(m)
    out = tmp_path / "glue"
    rust_glue.generate(m, out)
    captured = _fake_cargo_run(monkeypatch)
    monkeypatch.delenv("CARGO_TARGET_DIR", raising=False)
    expected = out / "target" / "release" / f"lib{rp.lib_name}.so"
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.write_bytes(b"fake-cdylib")
    assert rust_glue.compile_rust(out) == expected


# --------------------------------------------------------------------------
# v0.10 workstream C: cheap Rust FFI lint
# --------------------------------------------------------------------------


def test_lint_lib_rs_flags_unwrap_expect_panic():
    src = (
        "use jni::JNIEnv;\n"
        "fn kernel_m(p_a: &[f64]) -> f64 {\n"
        "    let v = p_a.get(0).unwrap();\n"
        "    let w = p_a.get(1).expect(\"need two\");\n"
        "    if v < 0.0 { panic!(\"neg\"); }\n"
        "    v + w\n"
        "}\n"
    )
    findings = rust_glue.lint_lib_rs(src)
    kinds = [w.kind for w in findings]
    assert kinds == ["ffi_panic", "ffi_panic", "ffi_panic"]
    assert [w.line for w in findings] == [3, 4, 5]
    assert all("FFI boundary" in w.message for w in findings)


def test_lint_lib_rs_ignores_comments():
    src = (
        "use jni::JNIEnv;\n"
        "// .unwrap() in a comment is fine\n"
        "fn kernel_m() {\n"
        "    // TODO(agent): paste the translated body here\n"
        "}\n"
    )
    assert rust_glue.lint_lib_rs(src) == []


def test_lint_lib_rs_generated_skeleton_clean(repo):
    m = repo.find_method("com.example.RustKernels.dot")
    src = rust_glue.render_lib_rs(rust_glue.plan(m))
    assert rust_glue.lint_lib_rs(src) == []


# --------------------------------------------------------------------------
# v0.10 workstream C: bench overhead split for the Rust backend
# --------------------------------------------------------------------------


def _bench_report(**kw):
    base = dict(
        method="com.example.RustKernels.dot",
        iters=1000,
        warmup=100,
        java_nanos=[1_000_000] * 5,
        jni_nanos=[1_200_000] * 5,
    )
    base.update(kw)
    return harness.BenchReport(**base)


def test_bench_report_rust_overhead_feeds_compute_ns():
    r = _bench_report(rust_overhead_ns=200.0)
    assert r.call_overhead_ns == 200.0
    # jni_per_call_ns = 1200; 1200 - 200 = 1000
    assert r.compute_ns == pytest.approx(1000.0)


def test_bench_report_jni_overhead_takes_precedence():
    r = _bench_report(jni_overhead_ns=100.0, rust_overhead_ns=200.0)
    assert r.call_overhead_ns == 100.0
    assert r.compute_ns == pytest.approx(1100.0)


def test_bench_report_no_overhead_is_plain_median():
    r = _bench_report()
    assert r.call_overhead_ns is None
    assert r.compute_ns == pytest.approx(1200.0)


def test_bench_report_rust_overhead_clamped_at_zero():
    r = _bench_report(rust_overhead_ns=10_000.0)
    assert r.compute_ns == 0.0


# --------------------------------------------------------------------------
# v0.10 workstream C: backend-aware bench CLI labels
# --------------------------------------------------------------------------


def _fake_bench_cli(monkeypatch, report):
    """Patch cli._build and harness.run_bench; return the CliRunner."""
    from typer.testing import CliRunner

    from kernel_port import cli as _cli

    monkeypatch.setattr(_cli, "_build", lambda *a, **k: object())
    monkeypatch.setattr(harness, "run_bench", lambda env, **k: report)
    return CliRunner()


def _rust_bench_report():
    return harness.BenchReport(
        method="com.example.RustKernels.dot",
        iters=1000,
        warmup=100,
        java_nanos=[1_000_000] * 5,
        jni_nanos=[1_300_000] * 5,
        rust_overhead_ns=150.0,
        per_size=[
            {"size": 64, "java_ns": 1000.0, "jni_ns": 800.0, "speedup": 1.25}
        ],
    )


def test_cli_bench_rust_labels(monkeypatch, tmp_path):
    runner = _fake_bench_cli(monkeypatch, _rust_bench_report())
    from kernel_port.cli import app

    r = runner.invoke(
        app,
        ["bench", str(FIXTURE_REPO), "--method", "com.example.RustKernels.dot",
         "--glue-dir", str(tmp_path), "--backend", "rust"],
    )
    assert r.exit_code == 0, r.output
    assert "Rust overhead split" in r.output
    assert "Pure compute in Rust (native - overhead)" in r.output
    assert "Rust/call" in r.output
    # fabricated numbers: compute (1150ns) > java (1000ns) -> the
    # backend-aware "translation is the problem" warning fires
    assert "The Rust compute itself is slower than Java" in r.output
    assert "JNI is " not in r.output
    assert "C++" not in r.output


def test_cli_bench_jni_labels_unchanged_in_spirit(monkeypatch, tmp_path):
    jni_report = _rust_bench_report()
    jni_report.jni_overhead_ns = 150.0
    jni_report.rust_overhead_ns = None
    runner = _fake_bench_cli(monkeypatch, jni_report)
    from kernel_port.cli import app

    r = runner.invoke(
        app,
        ["bench", str(FIXTURE_REPO), "--method", "com.example.RustKernels.dot",
         "--glue-dir", str(tmp_path), "--backend", "jni"],
    )
    assert r.exit_code == 0, r.output
    assert "JNI overhead split" in r.output
    assert "Pure compute in C++ (native - overhead)" in r.output
    assert "JNI/call" in r.output
    assert "The C++ compute itself is slower than Java" in r.output
    assert "Rust overhead split" not in r.output
    assert "Pure compute in Rust" not in r.output


# --------------------------------------------------------------------------
# v0.10 workstream C: end-to-end (cargo + JVM)
# --------------------------------------------------------------------------

#: Shared cargo target dir for the e2e tests: under ~/workspace (NOT the
#: /tmp tmpfs, which is only 512MB) and shared across crates so the jni
#: dependency compiles once.
_KP_CARGO_TARGET = Path.home() / "workspace" / "kernel-port" / ".kp-cargo-target"


@needs_cargo
def test_e2e_measure_rust_overhead(tmp_path, monkeypatch):
    """The Rust empty-call overhead measurement builds a tiny noop cdylib
    and times it via the shared KpNoop driver."""
    monkeypatch.setenv("CARGO_TARGET_DIR", str(_KP_CARGO_TARGET))
    work = tmp_path / "work"
    classes = work / "classes"
    classes.mkdir(parents=True)
    env = harness.BuiltEnv(work=work, classes=classes, method=None, backend="rust")
    ns = harness._measure_rust_overhead(env, iters=50_000, repeats=1)
    assert ns is not None
    assert ns >= 0


@needs_cargo
def test_e2e_two_kernels_shared_target_dir_no_stale_so(repo, tmp_path, monkeypatch):
    """Foot-gun regression test: two different kernels built sequentially
    with a SHARED CARGO_TARGET_DIR must produce distinct, correctly-loading
    artifacts. (With the old fixed crate name "kernels", the second build
    reused the first kernel's libkernels.so -> UnsatisfiedLinkError or
    silent wrong results.)"""
    monkeypatch.setenv("CARGO_TARGET_DIR", str(_KP_CARGO_TARGET))
    built = {}
    for qualname in (
        "com.example.RustKernels.dot",
        "com.example.RustKernels.scale",
    ):
        m = repo.find_method(qualname)
        glue = tmp_path / "glue" / m.name
        files = rust_glue.generate(m, glue)
        _fill_rust_translation(
            files["rust"], RUST_TRANSLATIONS[qualname]
        )
        env = harness.build_env(
            FIXTURE_REPO, m, glue, tmp_path / "work" / m.name,
            backend="rust", repo=repo,
        )
        built[qualname] = (m, env)
    (m_dot, env_dot) = built["com.example.RustKernels.dot"]
    (m_scale, env_scale) = built["com.example.RustKernels.scale"]
    so_dot = next(env_dot.work.glob("lib*.so"))
    so_scale = next(env_scale.work.glob("lib*.so"))
    # distinct artifacts, each named after its own crate
    assert so_dot.name != so_scale.name
    assert so_dot.name == f"lib{rust_glue.lib_name_for(m_dot.qualname)}.so"
    assert so_scale.name == f"lib{rust_glue.lib_name_for(m_scale.qualname)}.so"
    # each .so loads under its own System.loadLibrary name and computes
    # correctly -- a stale/wrong .so would fail to link or mismatch here
    vdot = harness.run_verify(env_dot, cases=20, boundary=False)
    assert vdot.all_passed, [
        (c.index, c.detail) for c in vdot.cases if not c.ok
    ]
    vscale = harness.run_verify(env_scale, cases=20, boundary=False)
    assert vscale.all_passed, [
        (c.index, c.detail) for c in vscale.cases if not c.ok
    ]


@needs_cargo
def test_e2e_bench_rust_reports_overhead_split(repo, tmp_path, monkeypatch):
    """run_bench with the Rust backend measures the empty-call overhead
    into report.rust_overhead_ns and derives compute_ns from it."""
    monkeypatch.setenv("CARGO_TARGET_DIR", str(_KP_CARGO_TARGET))
    m = repo.find_method("com.example.RustKernels.dot")
    glue = tmp_path / "glue"
    files = rust_glue.generate(m, glue)
    _fill_rust_translation(
        files["rust"], RUST_TRANSLATIONS["com.example.RustKernels.dot"]
    )
    env = harness.build_env(
        FIXTURE_REPO, m, glue, tmp_path / "work",
        backend="rust", repo=repo,
    )
    bench = harness.run_bench(
        env, iters=300, warmup=100, repeats=2, overhead=True
    )
    assert bench.rust_overhead_ns is not None
    assert bench.rust_overhead_ns >= 0
    assert bench.jni_overhead_ns is None
    assert bench.compute_ns == pytest.approx(
        max(0.0, bench.jni_per_call_ns - bench.rust_overhead_ns)
    )
    assert bench.compute_ns <= bench.jni_per_call_ns
