"""Dogfood regression (v0.8): build_env must not compile the whole repo.

On the real JOML repo, ``build_env`` compiled every ``*.java`` under the
repo root, which failed: ``src/test`` needs JUnit and several main-scope
files need annotation-only deps (``org.jspecify.annotations``,
``org.intellij.lang.annotations``) absent from any classpath.

The fix: compile just the kernel file and let javac resolve its transitive
dependencies via ``-sourcepath`` (see ``harness.kernel_sourcepath_roots``).
These tests pin that behavior: unit tests for root derivation plus an
integration test with a fake repo that reproduces the JOML failure mode
(uncompilable test sources + uncompilable annotation-dependent file) and
asserts the build now succeeds.
"""

from pathlib import Path

import pytest

from kernel_port import harness as H
from kernel_port import java_ast as J
from kernel_port import jni

pytestmark = pytest.mark.e2e  # needs JDK + g++ on PATH


def _mi(**kw):
    base = dict(
        qualname="com.ex.K.add",
        class_qualname="com.ex.K",
        name="add",
        jtype="int",
        params=[],
        modifiers=set(),
        is_static=True,
        is_constructor=False,
        file="src/main/java/com/ex/K.java",
        lineno=1,
        end_lineno=5,
        source="int add(int a, int b) { return a + b; }",
    )
    base.update(kw)
    return J.MethodInfo(**base)


def _touch(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# kernel_sourcepath_roots: root derivation
# ---------------------------------------------------------------------------


def test_roots_maven_layout(tmp_path):
    _touch(tmp_path, "main/java/org/joml/Intersectiond.java", "package org.joml; class Intersectiond {}")
    m = _mi(
        class_qualname="org.joml.Intersectiond",
        file="main/java/org/joml/Intersectiond.java",
    )
    assert H.kernel_sourcepath_roots(tmp_path, m) == [tmp_path / "main" / "java"]


def test_roots_flat_src_layout(tmp_path):
    _touch(tmp_path, "src/com/example/Kernels.java", "package com.example; class Kernels {}")
    m = _mi(
        class_qualname="com.example.Kernels",
        file="src/com/example/Kernels.java",
    )
    assert H.kernel_sourcepath_roots(tmp_path, m) == [tmp_path / "src"]


def test_roots_nested_class(tmp_path):
    # class_qualname includes enclosing classes; the package is the longest
    # prefix whose directory layout matches the file on disk.
    _touch(
        tmp_path,
        "src/main/java/org/joml/sampling/BestCandidateSampling.java",
        "package org.joml.sampling; class BestCandidateSampling { static class Sphere { static class Node {} } }",
    )
    m = _mi(
        qualname="org.joml.sampling.BestCandidateSampling.Sphere.Node.isPointOnSphericalTriangle",
        class_qualname="org.joml.sampling.BestCandidateSampling.Sphere.Node",
        file="src/main/java/org/joml/sampling/BestCandidateSampling.java",
    )
    assert H.kernel_sourcepath_roots(tmp_path, m) == [tmp_path / "src" / "main" / "java"]


def test_roots_default_package(tmp_path):
    _touch(tmp_path, "Kernels.java", "class Kernels {}")
    m = _mi(class_qualname="Kernels", file="Kernels.java")
    assert H.kernel_sourcepath_roots(tmp_path, m) == [tmp_path]


def test_roots_package_dir_mismatch_falls_back(tmp_path):
    # Weird layout: package does not match directories. Best effort: the
    # file's own dir plus the repo root (explicitly-listed files compile
    # regardless of layout).
    _touch(tmp_path, "weird/Place.java", "package com.ex; class K {}")
    m = _mi(file="weird/Place.java")
    roots = H.kernel_sourcepath_roots(tmp_path, m)
    assert roots == [tmp_path / "weird", tmp_path]


def test_roots_missing_file_falls_back_to_repo_root(tmp_path):
    m = _mi(file="nope/Missing.java")
    assert H.kernel_sourcepath_roots(tmp_path, m) == [tmp_path]


# ---------------------------------------------------------------------------
# build_env integration: the JOML failure mode
# ---------------------------------------------------------------------------


def _dogfood_repo(root: Path) -> Path:
    """Fake repo reproducing the JOML build failure.

    - ``K.add``: the pure kernel, references a helper in another main-scope
      file (must be resolved via -sourcepath).
    - ``KTest``: uncompilable test source (JUnit not on classpath).
    - ``UsesAnn``: uncompilable main file (annotation-only dep missing).
    """
    _touch(
        root,
        "src/main/java/com/ex/K.java",
        "package com.ex;\n"
        "public class K {\n"
        "    public static int add(int a, int b) { return Helper.twice(a) + b; }\n"
        "}\n",
    )
    _touch(
        root,
        "src/main/java/com/ex/Helper.java",
        "package com.ex;\n"
        "public class Helper {\n"
        "    public static int twice(int x) { return 2 * x; }\n"
        "}\n",
    )
    _touch(
        root,
        "src/test/java/com/ex/KTest.java",
        "package com.ex;\n"
        "import org.junit.jupiter.api.Test;\n"
        "public class KTest {\n"
        "    @Test public void t() { assert K.add(1, 2) == 4; }\n"
        "}\n",
    )
    _touch(
        root,
        "src/main/java/com/ex/UsesAnn.java",
        "package com.ex;\n"
        "import org.jspecify.annotations.NullMarked;\n"
        "@NullMarked public class UsesAnn {}\n",
    )
    return root


def test_build_env_skips_uncompilable_test_and_ann_files(tmp_path):
    repo_root = _dogfood_repo(tmp_path / "repo")
    repo = J.scan_repo(repo_root)
    m = repo.find_method("com.ex.K.add")
    assert m is not None

    glue_dir = tmp_path / "glue"
    jni.generate(m, glue_dir)
    cpp = glue_dir / "kernels_jni.cpp"
    text = cpp.read_text(encoding="utf-8")
    text = text.replace("return 0; // placeholder", "return p_a + p_b;")
    cpp.write_text(text, encoding="utf-8")

    # Old code compiled every *.java under the repo and died on KTest /
    # UsesAnn. The fix compiles just K.java; Helper.java is pulled in via
    # -sourcepath because K references it.
    env = H.build_env(repo_root, m, glue_dir, tmp_path / "work")
    assert (env.classes / "com" / "ex" / "K.class").exists()
    assert (env.classes / "com" / "ex" / "Helper.class").exists()
    # Test sources and the annotation-dependent file must NOT be compiled.
    assert not (env.classes / "com" / "ex" / "KTest.class").exists()
    assert not (env.classes / "com" / "ex" / "UsesAnn.class").exists()


def test_build_env_still_fails_loudly_on_broken_kernel(tmp_path):
    # The scoped compile must not swallow real errors in the kernel file.
    repo_root = tmp_path / "repo"
    _touch(
        repo_root,
        "src/main/java/com/ex/K.java",
        "package com.ex;\n"
        "public class K {\n"
        "    public static int add(int a, int b) { return ; }\n"  # syntax error
        "}\n",
    )
    repo = J.scan_repo(repo_root)
    m = repo.find_method("com.ex.K.add")
    glue_dir = tmp_path / "glue"
    jni.generate(m, glue_dir)
    with pytest.raises(RuntimeError, match="javac failed"):
        H.build_env(repo_root, m, glue_dir, tmp_path / "work")
