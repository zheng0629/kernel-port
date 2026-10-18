"""v0.12 dogfood regressions (apache/commons-numbers, fresh repo).

1. ``extract`` text output was reflowed by Rich when redirected: a long
   ``Location:`` line wrapped mid-path and looked like an empty field.
   Fixed with ``soft_wrap=True`` on the ``console.print`` calls.
2. Example/benchmark sources were mined as port candidates (370 of 880
   methods came from commons-numbers' ``examples-jmh`` module, including
   a top-5 candidate). ``java_ast.is_skipped_source`` now excludes
   example/benchmark directories as well as test directories.
3. ``render_minimal_sources`` pasted members verbatim, so a private kernel
   method stayed private in the synthesized class and the harness (in
   another package) could not call it -- the minimal attempt failed and
   the build fell back to the repo file. Members are now widened to
   ``public`` in the synthesized test-only copy.
4. Glue assumed a public kernel class accessed from the default package:
   a package-private class (commons-numbers' ``BoostErf``) or a private
   method (``BoostErf.erfInvImp``) broke ``verify``/``bench`` with
   "not public ... cannot be accessed from outside package". Glue is now
   generated in the kernel's package when the class or method is not
   public (:func:`kernel_port.jni.glue_package`), with a setAccessible
   reflective fallback for private methods.
"""

import pytest

from typer.testing import CliRunner

from kernel_port import extractor as E
from kernel_port import ffm
from kernel_port import harness as H
from kernel_port import java_ast as J
from kernel_port import jni
from kernel_port import rust_glue
from kernel_port.cli import app

from .conftest import fill_translation

runner = CliRunner()


def _write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


_PKG_PRIVATE_REPO = (
    "package com.ex;\n"
    "class K {\n"  # package-private, like org.apache.commons.numbers.gamma.BoostErf
    "    private static double f(double x) {\n"  # private, like erfInvImp
    "        return x * x + 1.0;\n"
    "    }\n"
    "    public static double api(double x) { return f(x); }\n"
    "}\n"
)


def _pkg_private_repo(root):
    _write(root, "src/main/java/com/ex/K.java", _PKG_PRIVATE_REPO)
    return J.scan_repo(root)


# ---------------------------------------------------------------------------
# 1. extract output must not be reflowed
# ---------------------------------------------------------------------------


def test_extract_text_does_not_wrap_long_location_line(tmp_path):
    # A repo-relative path long enough that "Location:  <path>:<line>"
    # exceeds Rich's 80-column non-tty width -- it used to wrap mid-path.
    deep = "src/" + "d" * 60
    _write(
        tmp_path / "repo",
        f"{deep}/com/ex/K.java",
        "package com.ex;\n"
        "public class K {\n"
        "    public static double f(double x) { return x * 2.0; }\n"
        "}\n",
    )
    r = runner.invoke(
        app, ["extract", str(tmp_path / "repo"), "--method", "com.ex.K.f"]
    )
    assert r.exit_code == 0, r.output
    loc = [ln for ln in r.output.splitlines() if ln.startswith("Location:")]
    assert len(loc) == 1
    assert loc[0].rstrip().endswith("K.java:3"), loc[0]


# ---------------------------------------------------------------------------
# 2. example/benchmark sources are skipped
# ---------------------------------------------------------------------------


def test_is_skipped_source():
    # test dirs still skipped
    assert J.is_skipped_source("src/test/java/com/foo/BarTest.java")
    assert J.is_skipped_source("tests/com/foo/Bar.java")
    # example/benchmark dirs skipped (v0.12 dogfood)
    assert J.is_skipped_source(
        "commons-numbers-examples/examples-jmh/src/main/java/org/apache/commons/numbers/examples/jmh/X.java"
    )
    assert J.is_skipped_source("mylib-examples/src/main/java/com/foo/E.java")
    assert J.is_skipped_source("examples/src/main/java/com/foo/E.java")
    assert J.is_skipped_source("project/jmh/src/main/java/com/foo/B.java")
    # ... but a com.example *package* directory is not an examples dir
    assert not J.is_skipped_source("src/com/example/Kernels.java")
    assert not J.is_skipped_source("src/main/java/com/foo/Bar.java")


def test_scan_repo_skips_example_dirs(tmp_path):
    _write(
        tmp_path / "repo",
        "mod-examples/src/main/java/com/ex/E.java",
        "package com.ex;\npublic class E {\n"
        "    public static int f(int x) { return x; }\n}\n",
    )
    _write(
        tmp_path / "repo",
        "mod/src/main/java/com/ex/K.java",
        "package com.ex;\npublic class K {\n"
        "    public static int g(int x) { return x; }\n}\n",
    )
    repo = J.scan_repo(tmp_path / "repo")
    assert repo.find_method("com.ex.K.g") is not None
    assert repo.find_method("com.ex.E.f") is None


# ---------------------------------------------------------------------------
# 3. minimal synthesis widens member visibility
# ---------------------------------------------------------------------------


def test_widen_visibility_basic():
    assert E._widen_visibility(
        "private static double f(double x) { return x; }"
    ).startswith("public static double f(")
    assert E._widen_visibility("protected int g() { return 1; }").startswith(
        "public int g()"
    )
    assert E._widen_visibility("public static int h() { return 1; }").startswith(
        "public static int h()"
    )
    # package-private stays package-private (the harness only needs the
    # kernel method callable; widening the minimum keeps the diff honest)
    out = E._widen_visibility("static int k() { return 1; }")
    assert out.startswith("static int k()")


def test_widen_visibility_leaves_body_and_strings_alone():
    src = 'private static String s() { return "private"; }'
    out = E._widen_visibility(src)
    assert out.startswith("public static String s()")
    assert '"private"' in out  # string literal untouched
    body = out[out.index("{") :]
    assert "private" not in body.replace('"private"', "")


def test_widen_visibility_annotation_with_braces():
    src = "@Anno({1, 2})\nprivate static int a() { return 1; }"
    out = E._widen_visibility(src)
    assert out.startswith("@Anno({1, 2})\npublic static int a()")


def test_render_minimal_sources_widens_private_kernel(tmp_path):
    repo = _pkg_private_repo(tmp_path / "repo")
    m = repo.find_method("com.ex.K.f")
    assert m is not None
    assert "private" in m.modifiers
    rendered = E.render_minimal_sources(m, E._extract_closure(repo, m), repo)
    assert rendered is not None
    src = rendered["com/ex/K.java"]
    assert "public static double f(double x)" in src
    assert "private static double f" not in src


@pytest.mark.e2e  # needs a JDK on PATH
def test_synthesized_private_kernel_compiles_for_foreign_caller(tmp_path):
    # The harness scenario: a caller in another package must be able to
    # call the synthesized kernel.
    repo = _pkg_private_repo(tmp_path / "repo")
    m = repo.find_method("com.ex.K.f")
    rendered = E.render_minimal_sources(m, E._extract_closure(repo, m), repo)
    assert rendered is not None
    synth = tmp_path / "synth"
    srcs = []
    for rel, text in rendered.items():
        p = synth / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        srcs.append(p)
    caller = tmp_path / "Caller.java"
    caller.write_text(
        "public class Caller {\n"
        "    public static void main(String[] a) {\n"
        "        System.out.println(com.ex.K.f(2.0));\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    srcs.append(caller)
    H.compile_java(srcs, tmp_path / "classes", [synth])
    assert (tmp_path / "classes" / "com" / "ex" / "K.class").exists()


# ---------------------------------------------------------------------------
# 4. package-aware glue
# ---------------------------------------------------------------------------


def test_glue_package_public_kernel_is_default_package(repo):
    m = repo.find_method("com.example.Kernels.dot")
    assert m is not None
    assert jni.glue_package(repo, m) == ""


def test_glue_package_package_private_class(tmp_path):
    repo = _pkg_private_repo(tmp_path / "repo")
    m = repo.find_method("com.ex.K.api")  # public method, package-private class
    assert jni.glue_package(repo, m) == "com.ex"


def test_glue_package_private_method(tmp_path):
    repo = _pkg_private_repo(tmp_path / "repo")
    m = repo.find_method("com.ex.K.f")  # private method
    assert jni.glue_package(repo, m) == "com.ex"


def test_glue_package_no_repo_falls_back_to_default():
    # glue_package must tolerate repo=None (pipeline multi-candidate path).
    assert jni.glue_package(None, _FakeMethod()) == ""


class _FakeMethod:
    class_qualname = "com.ex.K"
    modifiers = {"public"}


def test_jni_generate_packaged_layout(tmp_path):
    repo = _pkg_private_repo(tmp_path / "repo")
    m = repo.find_method("com.ex.K.api")
    files = jni.generate(m, tmp_path / "glue", package="com.ex")
    assert files["java"] == tmp_path / "glue" / "com" / "ex" / "KNative.java"
    src = files["java"].read_text(encoding="utf-8")
    assert "package com.ex;" in src
    assert "public class KNative" in src
    cpp = files["cpp"].read_text(encoding="utf-8")
    assert "Java_com_ex_KNative_apiNative" in cpp


def test_jni_generate_default_package_unchanged(repo, tmp_path):
    # Legacy output is byte-stable for public kernels.
    m = repo.find_method("com.example.Kernels.dot")
    files = jni.generate(m, tmp_path / "glue")
    assert files["java"].parent == tmp_path / "glue"
    src = files["java"].read_text(encoding="utf-8")
    assert "package " not in src
    assert "Java_KernelsNative_dotNative" in files["cpp"].read_text(
        encoding="utf-8"
    )


def test_jni_dispatcher_private_method_uses_reflection(tmp_path):
    repo = _pkg_private_repo(tmp_path / "repo")
    m = repo.find_method("com.ex.K.f")
    gp = jni.plan(m, package="com.ex")
    gp = jni._dispatcher_plan(gp)
    src = jni.render_java_dispatcher(gp)
    assert "package com.ex;" in src
    assert "FALLBACK_M" in src
    assert "setAccessible(true)" in src
    assert '.getDeclaredMethod("f", double.class)' in src
    assert "return (Double) FALLBACK_M.invoke(null, x);" in src


def test_jni_dispatcher_public_method_direct_call(repo):
    m = repo.find_method("com.example.Kernels.dot")
    gp = jni._dispatcher_plan(jni.plan(m))
    src = jni.render_java_dispatcher(gp)
    assert "FALLBACK_M" not in src
    assert "com.example.Kernels.dot(" in src


def test_ffm_generate_packaged(tmp_path):
    repo = _pkg_private_repo(tmp_path / "repo")
    m = repo.find_method("com.ex.K.api")
    files = ffm.generate(m, tmp_path / "glue", package="com.ex")
    assert files["java"] == tmp_path / "glue" / "com" / "ex" / "KFfm.java"
    assert "package com.ex;" in files["java"].read_text(encoding="utf-8")


def test_rust_generate_packaged(tmp_path):
    repo = _pkg_private_repo(tmp_path / "repo")
    m = repo.find_method("com.ex.K.api")
    files = rust_glue.generate(m, tmp_path / "glue", package="com.ex")
    assert files["java"] == tmp_path / "glue" / "com" / "ex" / "KNative.java"
    assert "package com.ex;" in files["java"].read_text(encoding="utf-8")


def test_render_harness_qualifies_packaged_glue(repo):
    m = repo.find_method("com.example.Kernels.dot")
    h = H.render_harness(m, "jni", "HarnessJni", package="com.ex")
    assert "com.ex.KernelsNative.dot(" in h
    h = H.render_harness(m, "ffm", "HarnessJni", package="com.ex")
    assert "com.ex.KernelsFfm.dot(" in h
    # legacy: unqualified
    h = H.render_harness(m, "jni", "HarnessJni")
    assert "KernelsNative.dot(" in h
    assert "com.ex.KernelsNative" not in h


def test_find_glue_java_discovers_package(tmp_path):
    (tmp_path / "glue" / "com" / "ex").mkdir(parents=True)
    (tmp_path / "glue" / "com" / "ex" / "KNative.java").write_text(
        "package com.ex;\npublic class KNative {}\n", encoding="utf-8"
    )
    path, pkg = H._find_glue_java(tmp_path / "glue", "jni")
    assert path is not None and path.name == "KNative.java"
    assert pkg == "com.ex"
    (tmp_path / "glue2").mkdir()
    (tmp_path / "glue2" / "KNative.java").write_text(
        "public class KNative {}\n", encoding="utf-8"
    )
    path, pkg = H._find_glue_java(tmp_path / "glue2", "jni")
    assert pkg == ""


@pytest.mark.e2e  # needs a JDK and g++ on PATH
def test_e2e_package_private_kernel_verify(tmp_path):
    """Full chain on a package-private class with a private static kernel.

    Mirrors the commons-numbers BoostErf.erfInvImp dogfood case: the
    synthesized minimal sources must compile, the packaged glue must
    compile, and verify must pass.
    """
    root = tmp_path / "repo"
    repo = _pkg_private_repo(root)
    m = repo.find_method("com.ex.K.f")
    assert m is not None
    glue_dir = tmp_path / "glue"
    jni.generate(m, glue_dir, package=jni.glue_package(repo, m))
    fill_translation(
        glue_dir,
        "com.ex.K.f",
        "    double r = p_x * p_x + 1.0;\n    return r;",
    )
    env = H.build_env(root, m, glue_dir, tmp_path / "work", repo=repo)
    report = H.run_verify(env, cases=20, seed=1)
    assert report.all_passed, [c.detail for c in report.cases if not c.ok]
    assert report.passed == 20


# ---------------------------------------------------------------------------
# 5. parse cache round-trips class modifiers (regression for bug 4's
#    first broken attempt: cache-hit ClassInfos lost `modifiers`, so
#    glue_package() packaged every kernel)
# ---------------------------------------------------------------------------


def test_parse_cache_round_trips_class_modifiers(tmp_path, monkeypatch):
    from kernel_port import parsecache

    monkeypatch.setenv(
        "KERNEL_PORT_PARSE_CACHE_DIR", str(tmp_path / "pcache")
    )
    root = tmp_path / "repo"
    _write(
        root,
        "src/main/java/com/ex/Pub.java",
        "package com.ex;\n"
        "public class Pub {\n"
        "    public static double f(double x) { return x; }\n"
        "}\n",
    )
    _write(
        root,
        "src/main/java/com/ex/Hidden.java",
        "package com.ex;\n"
        "class Hidden {\n"  # package-private
        "    public static double g(double x) { return x; }\n"
        "}\n",
    )
    cold = parsecache.load_repo(root)  # miss: parses and stores
    assert cold.classes["com.ex.Pub"].modifiers == {"public"}
    assert cold.classes["com.ex.Hidden"].modifiers == set()
    warm = parsecache.load_repo(root)  # hit: rebuilt from cache
    assert warm.classes["com.ex.Pub"].modifiers == {"public"}
    assert warm.classes["com.ex.Hidden"].modifiers == set()
    # ... and the glue decision agrees on both
    assert jni.glue_package(cold, cold.find_method("com.ex.Pub.f")) == ""
    assert jni.glue_package(warm, warm.find_method("com.ex.Pub.f")) == ""
    assert (
        jni.glue_package(cold, cold.find_method("com.ex.Hidden.g"))
        == "com.ex"
    )
    assert (
        jni.glue_package(warm, warm.find_method("com.ex.Hidden.g"))
        == "com.ex"
    )
