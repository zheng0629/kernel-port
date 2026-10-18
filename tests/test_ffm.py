"""Tests for the FFM (Panama Foreign Function & Memory API) glue backend.

Unit tests assert the structure of the generated C/Java text. End-to-end
tests build the real C library and run the generated Java bindings, but
need a JDK >= 22; on this dev machine a Temurin 25 lives in
~/workspace/tools/jdk-25.0.4.1+1 (see ffm.find_jdk). The e2e tests are
gated with pytest.mark.skipif so the suite stays green on JDK-21-only
machines.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from kernel_port import ffm
from kernel_port import java_ast as J

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "ffm_kernels"

# Hand-written C translations, standing in for the calling agent's LLM step.
# Params follow the FFM C ABI: arrays arrive as (const T* p_x, int64_t p_x_len).
FFM_TRANSLATIONS = {
    "com.example.FfmKernels.dot": (
        "    double s = 0.0;\n"
        "    for (int64_t i = 0; i < p_a_len; i++) s += p_a[i] * p_b[i];\n"
        "    return s;"
    ),
    # scale() mutates its array param in place, so the const pointer from the
    # generated signature is cast away with a plain C cast (same in/out
    # convention as the JNI backend); the Java wrapper copies the array
    # back after the call.
    "com.example.FfmKernels.scale": (
        "    for (int64_t i = 0; i < p_a_len; i++)\n"
        "        ((double*)p_a)[i] *= p_f;"
    ),
}

_JDK25 = ffm.find_jdk(22)
needs_jdk22 = pytest.mark.skipif(
    _JDK25 is None, reason="e2e FFM tests need a JDK >= 22 (none found)"
)


@pytest.fixture()
def repo():
    return J.scan_repo(FIXTURE_REPO)


def _fill_ffm_translation(c_path: Path, body: str) -> None:
    """Replace the TODO(agent) block in kernels_ffm.c with `body`."""
    text = c_path.read_text(encoding="utf-8")
    start_marker = "    // ==== BEGIN TRANSLATED BODY ===="
    end_marker = "    // ==== END TRANSLATED BODY ===="
    start = text.index(start_marker)
    end = text.index(end_marker) + len(end_marker)
    c_path.write_text(text[:start] + body.rstrip() + "\n" + text[end:], encoding="utf-8")


# --------------------------------------------------------------------------
# plan()
# --------------------------------------------------------------------------


def test_plan_maps_scalar_and_array_types(repo):
    m = repo.find_method("com.example.FfmKernels.dot")
    fp = ffm.plan(m)
    assert fp.qualname == "com.example.FfmKernels.dot"
    assert fp.class_name == "FfmKernelsFfm"
    assert fp.c_name == "kernel_dot"
    assert fp.java_return == "double"
    assert fp.java_return_layout == "JAVA_DOUBLE"
    assert fp.java_param_layouts == [
        "ADDRESS",
        "JAVA_LONG",
        "ADDRESS",
        "JAVA_LONG",
    ]
    assert (
        fp.c_signature
        == "double kernel_dot(const double* p_a, int64_t p_a_len, "
        "const double* p_b, int64_t p_b_len)"
    )


def test_plan_void_kernel(repo):
    m = repo.find_method("com.example.FfmKernels.scale")
    fp = ffm.plan(m)
    assert fp.java_return == "void"
    assert fp.java_return_layout == ""
    assert fp.java_param_layouts == ["ADDRESS", "JAVA_LONG", "JAVA_DOUBLE"]
    assert (
        fp.c_signature
        == "void kernel_scale(const double* p_a, int64_t p_a_len, double p_f)"
    )


def _method_info(jtype: str, params: list[tuple[str, str]], is_static: bool = True):
    return J.MethodInfo(
        qualname=f"com.example.X.m",
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


def test_plan_rejects_non_static():
    with pytest.raises(ValueError, match="static"):
        ffm.plan(_method_info("void", [], is_static=False))


def test_plan_rejects_unsupported_type():
    with pytest.raises(ValueError, match="unsupported type 'String'"):
        ffm.plan(_method_info("void", [("s", "String")]))


# --------------------------------------------------------------------------
# render_c()
# --------------------------------------------------------------------------


def test_render_c_is_pure_c_abi(repo):
    fp = ffm.plan(repo.find_method("com.example.FfmKernels.dot"))
    c = ffm.render_c(fp)
    assert "#include <jni.h>" not in c
    assert "JNIEnv" not in c
    assert '__attribute__((visibility("default")))' in c
    assert "KP_API double kernel_dot(" in c
    assert "const double* p_a, int64_t p_a_len" in c
    assert "TODO(agent)" in c
    assert "kp_free" in c  # ownership helper for array-returning kernels
    assert "#include <stdint.h>" in c


# --------------------------------------------------------------------------
# render_java()
# --------------------------------------------------------------------------


def test_render_java_uses_foreign_api_not_jni(repo):
    fp = ffm.plan(repo.find_method("com.example.FfmKernels.dot"))
    src = ffm.render_java(fp)
    assert "import java.lang.foreign.*;" in src
    assert "Linker.nativeLinker()" in src
    assert "SymbolLookup" in src
    assert "FunctionDescriptor.of(" in src
    assert "MemorySegment" in src
    assert "Arena.ofConfined()" in src
    assert "allocateFrom" in src
    assert 'findOrThrow("kernel_dot")' in src
    assert "public static double dot(double[] a, double[] b)" in src
    # no JNI idioms: no `native` method *declaration* anywhere (the word
    # "native" only appears inside comments and the Linker.nativeLinker()
    # API name).
    assert not re.search(r"(?m)^\s*(?:public|private|protected)?\s*(?:static\s+)?native\s", src)
    # no JNI API usage: no jni.h, no JNIEnv (the word "JNI" only appears in
    # explanatory comments contrasting with the JNI backend).
    assert "jni.h" not in src
    assert "JNIEnv" not in src


def test_render_java_carries_jdk22_requirement_comment(repo):
    fp = ffm.plan(repo.find_method("com.example.FfmKernels.dot"))
    src = ffm.render_java(fp)
    assert "REQUIRES JDK 22+" in src
    assert "25 LTS" in src
    assert "java_version_ok" in src


def test_render_java_void_kernel_copies_arrays_back(repo):
    fp = ffm.plan(repo.find_method("com.example.FfmKernels.scale"))
    src = ffm.render_java(fp)
    assert "public static void scale(double[] a, double f)" in src
    assert "FunctionDescriptor.ofVoid(" in src
    # in/out arrays are copied back so differential testing sees mutations
    assert "MemorySegment.copy(" in src


# --------------------------------------------------------------------------
# fallback dispatcher (render_java fallback=True/False)
# --------------------------------------------------------------------------


def test_render_java_fallback_is_default_and_dispatches(repo):
    fp = ffm.plan(repo.find_method("com.example.FfmKernels.dot"))
    src = ffm.render_java(fp)  # default: fallback=True
    # USE_NATIVE switch, mirroring the JNI dispatcher
    assert "private static boolean USE_NATIVE" in src
    assert (
        'Boolean.parseBoolean(System.getProperty("kernelport.native", "true"))'
        in src
    )
    assert "public static void setUseNative(boolean useNative)" in src
    # public method keeps the kernel signature and dispatches
    assert "public static double dot(double[] a, double[] b)" in src
    assert "if (FfmKernelsFfm.USE_NATIVE && FfmKernelsFfm.ensureNative()) {" in src
    assert "return dotFfm(a, b);" in src
    assert "return com.example.FfmKernels.dot(a, b);" in src
    # the FFM downcall moved into the private method
    assert "private static double dotFfm(double[] a, double[] b)" in src
    assert 'findOrThrow("kernel_dot")' in src
    # both switch mechanisms documented in the header comments
    assert "-Dkernelport.native=false" in src
    assert "FfmKernelsFfm.setUseNative(false)" in src


def test_render_java_fallback_void_kernel_dispatches(repo):
    fp = ffm.plan(repo.find_method("com.example.FfmKernels.scale"))
    src = ffm.render_java(fp)
    assert "public static void scale(double[] a, double f)" in src
    assert "private static void scaleFfm(double[] a, double f)" in src
    assert "scaleFfm(a, f);" in src
    assert "com.example.FfmKernels.scale(a, f);" in src
    assert "USE_NATIVE" in src


def test_render_java_fallback_false_renders_bare_form(repo):
    fp = ffm.plan(repo.find_method("com.example.FfmKernels.dot"))
    src = ffm.render_java(fp, fallback=False)
    assert "USE_NATIVE" not in src
    assert "setUseNative" not in src
    assert "dotFfm" not in src
    # no fallback delegation to the original kernel
    assert "FfmKernels.dot(a, b)" not in src
    assert "public static double dot(double[] a, double[] b)" in src
    # the downcall still lives directly in the public method
    assert "MH_DOT.invokeExact" in src


# --------------------------------------------------------------------------
# v0.10 workstream D: lazy native loading (fix 3)
# --------------------------------------------------------------------------


def test_render_java_lazy_loads_instead_of_static_block(repo):
    fp = ffm.plan(repo.find_method("com.example.FfmKernels.dot"))
    src = ffm.render_java(fp)
    # no eager static { System.loadLibrary(..) } anymore
    assert "static {" not in src
    # lazy loader with the documented contract
    assert "private static synchronized boolean ensureNative()" in src
    assert "catch (UnsatisfiedLinkError e)" in src
    assert 'System.err.println("kernel-port FFM:' in src
    assert "falling back to the Java implementation" in src
    # one-line note: the message literal contains no newline
    note = [l for l in src.splitlines() if "kernel-port FFM:" in l]
    assert len(note) == 1
    # sticky failure state, re-armed by setUseNative(true)
    assert "private static volatile boolean nativeReady" in src
    assert "private static volatile boolean nativeFailed" in src
    assert "nativeFailed = false;" in src
    # dispatcher consults the loader only while the switch is on
    assert (
        "if (FfmKernelsFfm.USE_NATIVE && FfmKernelsFfm.ensureNative()) {"
        in src
    )


def test_render_java_documents_lazy_loading(repo):
    fp = ffm.plan(repo.find_method("com.example.FfmKernels.dot"))
    src = ffm.render_java(fp)
    assert "Lazy native loading" in src
    assert "UnsatisfiedLinkError" in src
    assert "never in a static initializer" in src
    # same switch semantics as the JNI backend
    assert "-Dkernelport.native=false" in src
    assert "setUseNative(false)" in src


def test_render_java_fallback_false_keeps_eager_load(repo):
    # The bare form has no Java original to fall back to, so it keeps the
    # eager static-block load (and documents nothing about laziness).
    fp = ffm.plan(repo.find_method("com.example.FfmKernels.dot"))
    src = ffm.render_java(fp, fallback=False)
    assert "static {" in src
    assert "ensureNative" not in src
    assert "Lazy native loading" not in src


@needs_jdk22
def test_e2e_lazy_fallback_missing_so_returns_correct_result(repo, tmp_path):
    """Missing libkernels_ffm.so: class init must succeed and the call
    must fall back to the Java original (one stderr note, sticky)."""
    m = repo.find_method("com.example.FfmKernels.dot")
    out = tmp_path / "noso"
    files = ffm.generate(m, out)  # .so deliberately NOT built
    runner = out / "RunNoSo.java"
    runner.write_text(
        "public class RunNoSo {\n"
        "    public static void main(String[] a) {\n"
        "        double r1 = FfmKernelsFfm.dot(\n"
        "            new double[]{1.0, 2.0}, new double[]{3.0, 4.0});\n"
        "        double r2 = FfmKernelsFfm.dot(\n"
        "            new double[]{1.0, 2.0}, new double[]{3.0, 4.0});\n"
        "        System.out.println(\"NOSO \" + r1 + \"/\" + r2);\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    ffm.compile_java(
        [runner, files["java"], ORIG_KERNEL], out / "classes",
        javac=_javac(),
    )
    proc = ffm.run_java(
        _java(), out / "classes", "RunNoSo",
        lib_dir=tmp_path / "empty-lib-dir",  # no .so anywhere near
    )
    assert proc.returncode == 0, proc.stderr
    # Java original: 1*3 + 2*4 = 11
    assert "NOSO 11.0/11.0" in proc.stdout
    # exactly one stderr note (sticky failure: not retried per call)
    assert proc.stderr.count("kernel-port FFM:") == 1
    assert "falling back to the Java implementation" in proc.stderr


@needs_jdk22
def test_e2e_lazy_fallback_native_path_used_when_lib_present(
    repo, tmp_path
):
    """A deliberately WRONG C translation must surface when the .so loads,
    proving the native path (not the Java fallback) is taken."""
    m = repo.find_method("com.example.FfmKernels.dot")
    out = tmp_path / "wrong"
    files = ffm.generate(m, out)
    _fill_ffm_translation(files["c"], "    return 999.0; // wrong on purpose")
    ffm.compile_c(out / "kernels_ffm.c", out / f"lib{ffm.LIB_NAME}.so")
    runner = out / "RunWrong.java"
    runner.write_text(
        "public class RunWrong {\n"
        "    public static void main(String[] a) {\n"
        "        double r = FfmKernelsFfm.dot(\n"
        "            new double[]{1.0}, new double[]{2.0});\n"
        "        System.out.println(\"WRONG \" + r);\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    ffm.compile_java(
        [runner, files["java"], ORIG_KERNEL], out / "classes",
        javac=_javac(),
    )
    proc = ffm.run_java(
        _java(), out / "classes", "RunWrong", lib_dir=out
    )
    assert proc.returncode == 0, proc.stderr
    assert "WRONG 999.0" in proc.stdout  # native, not Java (2.0)
    assert "kernel-port FFM:" not in proc.stderr  # no fallback note


@needs_jdk22
def test_e2e_native_disabled_skips_library_load(repo, tmp_path):
    """-Dkernelport.native=false with a missing .so: no load attempt at
    all (no stderr note), correct result from the Java original."""
    m = repo.find_method("com.example.FfmKernels.dot")
    out = tmp_path / "disabled"
    files = ffm.generate(m, out)  # .so deliberately NOT built
    runner = out / "RunDisabled.java"
    runner.write_text(
        "public class RunDisabled {\n"
        "    public static void main(String[] a) {\n"
        "        double r = FfmKernelsFfm.dot(\n"
        "            new double[]{1.0, 2.0}, new double[]{3.0, 4.0});\n"
        "        System.out.println(\"DISABLED \" + r);\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    ffm.compile_java(
        [runner, files["java"], ORIG_KERNEL], out / "classes",
        javac=_javac(),
    )
    proc = ffm.run_java(
        _java(),
        out / "classes",
        "RunDisabled",
        lib_dir=tmp_path / "empty-lib-dir",
        extra_jvm_args=["-Dkernelport.native=false"],
    )
    assert proc.returncode == 0, proc.stderr
    assert "DISABLED 11.0" in proc.stdout
    assert "kernel-port FFM:" not in proc.stderr


def test_generate_fallback_flag_is_wired(repo, tmp_path):
    m = repo.find_method("com.example.FfmKernels.dot")
    default_files = ffm.generate(m, tmp_path / "fb")  # positional 2-arg call
    bare_files = ffm.generate(m, tmp_path / "bare", fallback=False)
    assert "USE_NATIVE" in default_files["java"].read_text(encoding="utf-8")
    assert "USE_NATIVE" not in bare_files["java"].read_text(encoding="utf-8")
    # build script is the same either way (it compiles whatever *Ffm.java exists)
    assert bare_files["build"].read_text(encoding="utf-8") == (
        default_files["build"].read_text(encoding="utf-8").replace(
            str(tmp_path / "fb"), str(tmp_path / "bare")
        )
    )


# --------------------------------------------------------------------------
# read-only array copy-back skip (jni.written_arrays)
# --------------------------------------------------------------------------


def test_readonly_array_skips_copy_back_but_keeps_in_copy(repo):
    # dot() never writes its arrays: the OUT copy is provably useless.
    fp = ffm.plan(repo.find_method("com.example.FfmKernels.dot"))
    src = ffm.render_java(fp, fallback=False)
    assert "allocateFrom" in src  # IN copy always emitted
    assert "MemorySegment.copy(" not in src  # OUT copy-back skipped
    # the skip is recorded in the generated code
    assert "Copy-back skipped" in src
    assert "'a'" in src and "'b'" in src


def test_written_array_keeps_copy_back(repo):
    # scale() mutates a[] in place: copy-back must stay.
    fp = ffm.plan(repo.find_method("com.example.FfmKernels.scale"))
    src = ffm.render_java(fp, fallback=False)
    assert "MemorySegment.copy(" in src
    assert "Copy-back skipped" not in src


# --------------------------------------------------------------------------
# render_build_script()
# --------------------------------------------------------------------------


def test_render_build_script(repo, tmp_path):
    fp = ffm.plan(repo.find_method("com.example.FfmKernels.dot"))
    sh = ffm.render_build_script(fp, tmp_path / "out")
    assert "-shared -fPIC -O2" in sh
    assert "libkernels_ffm.so" in sh
    assert "kernels_ffm.c" in sh
    assert "FfmKernelsFfm.java" in sh
    assert "JAVAC" in sh and "JDK 22+" in sh


# --------------------------------------------------------------------------
# java_version_ok()
# --------------------------------------------------------------------------


def test_java_version_ok_refuses_default_jdk21():
    javac = shutil.which("javac")
    assert javac is not None
    major = ffm._parse_major(
        __import__("subprocess").run(
            [javac, "-version"], capture_output=True, text=True
        ).stdout
    )
    if major is not None and major >= 22:
        pytest.skip("default javac is already >= 22; refusal path not exercised")
    ok, msg = ffm.java_version_ok(javac)
    assert ok is False
    assert "22" in msg
    assert "25" in msg  # recommends the LTS


def test_java_version_ok_accepts_jdk25():
    if _JDK25 is None:
        pytest.skip("no JDK >= 22 found")
    ok, msg = ffm.java_version_ok(str(_JDK25 / "bin" / "javac"))
    assert ok is True
    assert ">= 22" in msg


def test_java_version_ok_missing_binary():
    ok, msg = ffm.java_version_ok("/nonexistent/javac-xyz")
    assert ok is False
    assert "JDK 22+" in msg


# --------------------------------------------------------------------------
# end-to-end: build C lib + generated Java, run on JDK 25
# --------------------------------------------------------------------------


# The original kernel class: the fallback dispatcher delegates to it, so
# javac must compile it together with the generated binding.
ORIG_KERNEL = FIXTURE_REPO / "src" / "com" / "example" / "FfmKernels.java"


def _e2e_build(repo, qualname: str, tmp_path: Path):
    """Generate glue (fallback=True default) and build the C library.

    Java compilation is left to the caller: the generated ``*Ffm.java`` and
    the original kernel class must be compiled in one javac call so the
    fallback delegation resolves. Callers pass
    ``[runner, paths["java_src"], ORIG_KERNEL]`` to ``ffm.compile_java``.
    Returns ``(method, paths)``.
    """
    m = repo.find_method(qualname)
    assert m is not None
    out = tmp_path / "ffm"
    files = ffm.generate(m, out)
    _fill_ffm_translation(files["c"], FFM_TRANSLATIONS[qualname])
    javac = str(_JDK25 / "bin" / "javac")  # type: ignore[union-attr]
    ffm.compile_c(out / "kernels_ffm.c", out / f"lib{ffm.LIB_NAME}.so")
    return m, {
        "c_src": files["c"],
        "java_src": files["java"],
        "lib": out / f"lib{ffm.LIB_NAME}.so",
        "classes": out / "classes",
        "lib_dir": out,
    }


def _javac():
    return str(_JDK25 / "bin" / "javac")  # type: ignore[union-attr]


def _java():
    return str(_JDK25 / "bin" / "java")  # type: ignore[union-attr]


@needs_jdk22
def test_e2e_dot_product(repo, tmp_path):
    m, paths = _e2e_build(repo, "com.example.FfmKernels.dot", tmp_path)
    runner = tmp_path / "ffm" / "RunDot.java"
    runner.write_text(
        "public class RunDot {\n"
        "    public static void main(String[] a) {\n"
        "        double r = FfmKernelsFfm.dot(\n"
        "            new double[]{1.0, 2.0, 3.0}, new double[]{4.0, 5.0, 6.0});\n"
        "        System.out.println(\"DOT \" + r);\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    ffm.compile_java(
        [runner, paths["java_src"], ORIG_KERNEL], paths["classes"],
        javac=_javac(),
    )
    proc = ffm.run_java(
        _java(),
        paths["classes"],
        "RunDot",
        lib_dir=paths["lib_dir"],
    )
    assert proc.returncode == 0, proc.stderr
    assert "DOT 32.0" in proc.stdout  # 1*4 + 2*5 + 3*6


@needs_jdk22
def test_e2e_void_kernel_mutates_array(repo, tmp_path):
    m, paths = _e2e_build(repo, "com.example.FfmKernels.scale", tmp_path)
    runner = tmp_path / "ffm" / "RunScale.java"
    runner.write_text(
        "public class RunScale {\n"
        "    public static void main(String[] a) {\n"
        "        double[] v = new double[]{1.0, 2.0, 3.0};\n"
        "        FfmKernelsFfm.scale(v, 2.0);\n"
        "        System.out.println(\"SCALE \" + v[0] + \",\" + v[1] + \",\" + v[2]);\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    ffm.compile_java(
        [runner, paths["java_src"], ORIG_KERNEL], paths["classes"],
        javac=_javac(),
    )
    proc = ffm.run_java(
        _java(),
        paths["classes"],
        "RunScale",
        lib_dir=paths["lib_dir"],
    )
    assert proc.returncode == 0, proc.stderr
    # in/out array copy-back: the Java array must show the mutation
    assert "SCALE 2.0,4.0,6.0" in proc.stdout


@needs_jdk22
def test_e2e_fallback_off_uses_original_java(repo, tmp_path):
    """-Dkernelport.native=false rolls back to the original Java kernel."""
    m, paths = _e2e_build(repo, "com.example.FfmKernels.dot", tmp_path)
    runner = tmp_path / "ffm" / "RunDotFallback.java"
    runner.write_text(
        "public class RunDotFallback {\n"
        "    public static void main(String[] a) {\n"
        "        double r = FfmKernelsFfm.dot(\n"
        "            new double[]{1.0, 2.0, 3.0}, new double[]{4.0, 5.0, 6.0});\n"
        "        System.out.println(\"FALLBACK \" + r);\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    ffm.compile_java(
        [runner, paths["java_src"], ORIG_KERNEL], paths["classes"],
        javac=_javac(),
    )
    proc = ffm.run_java(
        _java(),
        paths["classes"],
        "RunDotFallback",
        lib_dir=paths["lib_dir"],
        extra_jvm_args=["-Dkernelport.native=false"],
    )
    assert proc.returncode == 0, proc.stderr
    # 1*4 + 2*5 + 3*6 computed by the ORIGINAL Java kernel, not the C lib
    assert "FALLBACK 32.0" in proc.stdout


@needs_jdk22
def test_e2e_fallback_false_java_compiles(repo, tmp_path):
    """The bare (fallback=False) form must still compile with the JDK."""
    m = repo.find_method("com.example.FfmKernels.dot")
    files = ffm.generate(m, tmp_path / "bare", fallback=False)
    ffm.compile_java([files["java"]], tmp_path / "bare" / "classes",
                     javac=_javac())


# --------------------------------------------------------------------------
# v0.10 workstream D: collision-proof generated locals (fix 1)
# --------------------------------------------------------------------------


COLLISION_KERNEL = FIXTURE_REPO / "src" / "com" / "example" / "CollisionKernels.java"

# Hand-written C translations for the collision fixtures (same convention
# as FFM_TRANSLATIONS above).
COLLISION_TRANSLATIONS = {
    "com.example.CollisionKernels.weird": (
        "    double s = p_r;\n"
        "    for (int64_t i = 0; i < p_data_len; i++) s += p_data[i] * p_n;\n"
        "    return s;"
    ),
    "com.example.CollisionKernels.make": (
        "    double* out = (double*)malloc(p_n * sizeof(double));\n"
        "    for (int32_t i = 0; i < p_n; i++) out[i] = p_r * i;\n"
        "    *out_len = p_n;\n"
        "    return out;"
    ),
    "com.example.CollisionKernels.fill": (
        "    for (int64_t i = 0; i < p_data_len; i++)\n"
        "        ((double*)p_data)[i] = p_r;"
    ),
}


def _hostile_method():
    """A method whose params collide with every generated local name."""
    return _method_info(
        "double[]",
        [
            ("r", "double[]"),
            ("n", "double"),
            ("data", "long"),
            ("outLen", "int"),
            ("ptr", "boolean"),
            ("arena", "char"),
            ("kp_r", "float"),  # even the prefixed fallback name is taken
            ("kp_data", "byte"),
            ("t", "short"),
        ],
    )


def test_generated_locals_never_collide_with_params():
    fp = ffm.plan(_hostile_method())
    for fallback in (True, False):
        src = ffm.render_java(fp, fallback=fallback)
        declared = set(
            re.findall(
                r"(?:MemorySegment|Arena|long|double\[\]|double|int|boolean)"
                r"\s+([A-Za-z_]\w*)\s*=",
                src,
            )
        )
        declared |= set(
            re.findall(r"catch\s*\(\s*Throwable\s+([A-Za-z_]\w*)\)", src)
        )
        assert declared, "expected generated locals in the wrapper body"
        params = {p.name for p in fp.method.params}
        assert not (declared & params), (
            f"generated locals collide with params: {declared & params}"
        )
        # the kp_ prefix is in use, and the taken kp_r was freshened
        assert "kp_r_2" in src


def _e2e_collision_build(qualname: str, tmp_path: Path, tag: str):
    """Generate + build a collision-fixture kernel; return run paths."""
    import kernel_port.java_ast as _J

    repo = _J.scan_repo(FIXTURE_REPO)
    m = repo.find_method(qualname)
    assert m is not None
    out = tmp_path / tag
    files = ffm.generate(m, out)
    _fill_ffm_translation(files["c"], COLLISION_TRANSLATIONS[qualname])
    ffm.compile_c(out / "kernels_ffm.c", out / f"lib{ffm.LIB_NAME}.so")
    return {
        "java_src": files["java"],
        "classes": out / "classes",
        "lib_dir": out,
    }


@needs_jdk22
def test_e2e_collision_scalar_kernel_compiles_and_verifies(tmp_path):
    """Params literally named r, n, data: the old `double r = ...` local
    made javac fail with 'variable r is already defined'."""
    paths = _e2e_collision_build(
        "com.example.CollisionKernels.weird", tmp_path, "weird"
    )
    out = paths["lib_dir"]
    runner = out / "RunWeird.java"
    runner.write_text(
        "public class RunWeird {\n"
        "    public static void main(String[] a) {\n"
        "        double r = CollisionKernelsFfm.weird(\n"
        "            new double[]{1.0, 2.0, 3.0}, 10.0, 2L);\n"
        "        double ref = com.example.CollisionKernels.weird(\n"
        "            new double[]{1.0, 2.0, 3.0}, 10.0, 2L);\n"
        "        System.out.println(\"WEIRD \" + r + \"/\" + ref);\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    # javac success is the regression test (old code did not compile)
    ffm.compile_java(
        [runner, paths["java_src"], COLLISION_KERNEL], paths["classes"],
        javac=_javac(),
    )
    proc = ffm.run_java(
        _java(), paths["classes"], "RunWeird", lib_dir=paths["lib_dir"]
    )
    assert proc.returncode == 0, proc.stderr
    # 10 + (1+2+3)*2 = 22, native and Java original must agree
    assert "WEIRD 22.0/22.0" in proc.stdout


@needs_jdk22
def test_e2e_collision_array_return_kernel(tmp_path):
    """Array-returning kernel with params n, r: exercises the
    kp_outLen/kp_ptr/kp_n/kp_data/kp_r locals."""
    paths = _e2e_collision_build(
        "com.example.CollisionKernels.make", tmp_path, "make"
    )
    out = paths["lib_dir"]
    runner = out / "RunMake.java"
    runner.write_text(
        "public class RunMake {\n"
        "    public static void main(String[] a) {\n"
        "        double[] v = CollisionKernelsFfm.make(4, 2.5);\n"
        "        StringBuilder sb = new StringBuilder(\"MAKE\");\n"
        "        for (double x : v) sb.append(' ').append(x);\n"
        "        System.out.println(sb);\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    ffm.compile_java(
        [runner, paths["java_src"], COLLISION_KERNEL], paths["classes"],
        javac=_javac(),
    )
    proc = ffm.run_java(
        _java(), paths["classes"], "RunMake", lib_dir=paths["lib_dir"]
    )
    assert proc.returncode == 0, proc.stderr
    assert "MAKE 0.0 2.5 5.0 7.5" in proc.stdout


@needs_jdk22
def test_e2e_collision_void_kernel_copyback(tmp_path):
    """Void kernel with an in/out array param named data."""
    paths = _e2e_collision_build(
        "com.example.CollisionKernels.fill", tmp_path, "fill"
    )
    out = paths["lib_dir"]
    runner = out / "RunFill.java"
    runner.write_text(
        "public class RunFill {\n"
        "    public static void main(String[] a) {\n"
        "        double[] v = new double[]{0.0, 0.0, 0.0};\n"
        "        CollisionKernelsFfm.fill(v, 7.0);\n"
        "        System.out.println(\"FILL \" + v[0] + \",\" + v[1] + \",\" + v[2]);\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    ffm.compile_java(
        [runner, paths["java_src"], COLLISION_KERNEL], paths["classes"],
        javac=_javac(),
    )
    proc = ffm.run_java(
        _java(), paths["classes"], "RunFill", lib_dir=paths["lib_dir"]
    )
    assert proc.returncode == 0, proc.stderr
    assert "FILL 7.0,7.0,7.0" in proc.stdout


# --------------------------------------------------------------------------
# v0.10 workstream D: build_ffm.sh that survives renames/cwd (fix 2)
# --------------------------------------------------------------------------


@needs_jdk22
def test_render_build_script_resolves_from_script_dir(repo, tmp_path):
    """The script must not bake in a directory name: renaming the glue dir
    or invoking from another cwd must keep working."""
    fp = ffm.plan(repo.find_method("com.example.FfmKernels.dot"))
    sh = ffm.render_build_script(fp, tmp_path / "out")
    assert 'OUT_DIR="$(cd "$(dirname "$0")" && pwd)"' in sh
    assert str(tmp_path) not in sh  # no baked-in absolute path either


@needs_jdk22
def test_build_script_end_to_end_from_foreign_cwd(repo, tmp_path):
    """Run the generated build_ffm.sh for real: from a foreign cwd, after
    renaming the glue dir (the old script baked OUT_DIR="glue" and broke
    here), then run the built artifacts."""
    import os
    import subprocess

    m = repo.find_method("com.example.FfmKernels.dot")
    glue = tmp_path / "glue"
    files = ffm.generate(m, glue)
    _fill_ffm_translation(
        files["c"], FFM_TRANSLATIONS["com.example.FfmKernels.dot"]
    )
    renamed = tmp_path / "glue-renamed"
    glue.rename(renamed)  # old script would now look for "glue/" and fail
    script = renamed / "build_ffm.sh"
    assert os.access(script, os.X_OK)
    env = {
        "PATH": os.environ["PATH"],
        "CC": "cc",
        "JAVAC": str(_JDK25 / "bin" / "javac"),  # type: ignore[union-attr]
        # the fallback dispatcher references the original kernel class
        "JAVACFLAGS": f"-sourcepath {FIXTURE_REPO / 'src'}",
    }
    proc = subprocess.run(
        ["sh", str(script)],
        cwd=tmp_path,  # foreign cwd: NOT the glue dir
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert (renamed / f"lib{ffm.LIB_NAME}.so").exists()
    assert (renamed / "classes" / "FfmKernelsFfm.class").exists()
    # the script-built artifacts actually work
    runner = renamed / "RunScript.java"
    runner.write_text(
        "public class RunScript {\n"
        "    public static void main(String[] a) {\n"
        "        double r = FfmKernelsFfm.dot(\n"
        "            new double[]{1.0, 2.0, 3.0}, new double[]{4.0, 5.0, 6.0});\n"
        "        System.out.println(\"SCRIPT \" + r);\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    ffm.compile_java(
        [runner, renamed / "FfmKernelsFfm.java"],
        renamed / "classes",
        javac=str(_JDK25 / "bin" / "javac"),  # type: ignore[union-attr]
        sourcepath=[FIXTURE_REPO / "src"],
    )
    proc = ffm.run_java(
        _java(), renamed / "classes", "RunScript", lib_dir=renamed
    )
    assert proc.returncode == 0, proc.stderr
    assert "SCRIPT 32.0" in proc.stdout


# --------------------------------------------------------------------------
# v0.10 workstream D: full CLI verify + bench through the FFM backend
# --------------------------------------------------------------------------


@needs_jdk22
def test_e2e_cli_verify_and_bench_ffm(repo, tmp_path):
    """Real end-to-end: build the glue dir with the fixed build_ffm.sh,
    then drive `kernel-port verify` and `bench --backend ffm` on it."""
    import os
    import subprocess
    import sys

    m = repo.find_method("com.example.FfmKernels.dot")
    glue = tmp_path / "cli-glue"
    files = ffm.generate(m, glue)
    _fill_ffm_translation(
        files["c"], FFM_TRANSLATIONS["com.example.FfmKernels.dot"]
    )
    env = {
        "PATH": f"{_JDK25 / 'bin'}{os.pathsep}{os.environ['PATH']}",  # type: ignore[union-attr]
        "CC": "cc",
        "JAVAC": str(_JDK25 / "bin" / "javac"),  # type: ignore[union-attr]
        "JAVACFLAGS": f"-sourcepath {FIXTURE_REPO / 'src'}",
    }
    proc = subprocess.run(
        ["sh", str(glue / "build_ffm.sh")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr

    def cli(*args: str) -> "subprocess.CompletedProcess[str]":
        return subprocess.run(
            [sys.executable, "-m", "kernel_port", *args],
            cwd=tmp_path,
            env={**os.environ, "PATH": env["PATH"]},
            capture_output=True,
            text=True,
            timeout=600,
        )

    verify = cli(
        "verify", str(FIXTURE_REPO),
        "--method", "com.example.FfmKernels.dot",
        "--glue-dir", str(glue),
        "--backend", "ffm",
        "--cases", "20",
    )
    assert verify.returncode == 0, verify.stderr + verify.stdout
    assert "PASS" in verify.stdout

    bench = cli(
        "bench", str(FIXTURE_REPO),
        "--method", "com.example.FfmKernels.dot",
        "--glue-dir", str(glue),
        "--backend", "ffm",
        "--iters", "100",
        "--warmup", "20",
        "--repeats", "2",
    )
    assert bench.returncode == 0, bench.stderr + bench.stdout
    assert "FFM C" in bench.stdout or "benchmark" in bench.stdout
