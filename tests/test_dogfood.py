"""v0.8 dogfood regressions.

1. ``build_env`` compiled the whole repo (JOML: test sources need JUnit,
   main files need annotation-only deps) -- fixed by compiling a synthesized
   minimal kernel file (``extractor.render_minimal_sources``) with
   ``-sourcepath`` fallback to the repo file.
2. File bytes were re-read per method (11,541 reads / 121 files on JOML) --
   fixed by ``java_ast.source_bytes`` (mtime+size validated cache).
3. Test sources (``src/test``) were mined as port candidates -- now excluded
   from scanning.
"""

import pytest

from kernel_port import extractor as E
from kernel_port import harness as H
from kernel_port import java_ast as J

pytestmark = pytest.mark.e2e  # some tests need a JDK on PATH


def _write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def _repo_with_kernel(root):
    _write(
        root,
        "src/main/java/com/ex/K.java",
        "package com.ex;\n"
        "import java.util.Arrays;\n"
        "import org.jspecify.annotations.Nullable;\n"
        "public class K {\n"
        "    static final double FACTOR = 2.0;\n"
        "    public static double scaled(double x) { return Helper.twice(x) * FACTOR; }\n"
        "    public static int countPos(double[] a) {\n"
        "        int n = 0;\n"
        "        for (double v : a) if (v > 0) n++;\n"
        "        return n;\n"
        "    }\n"
        "    @Nullable public static String tag() { return \"k\"; }\n"
        "}\n",
    )
    _write(
        root,
        "src/main/java/com/ex/Helper.java",
        "package com.ex;\n"
        "public class Helper {\n"
        "    public static double twice(double x) { return 2 * x; }\n"
        "}\n",
    )
    return J.scan_repo(root)


# ---------------------------------------------------------------------------
# render_minimal_sources
# ---------------------------------------------------------------------------


def test_render_minimal_sources_basic(tmp_path):
    repo = _repo_with_kernel(tmp_path / "repo")
    m = repo.find_method("com.ex.K.scaled")
    assert m is not None
    rendered = E.render_minimal_sources(m, E._extract_closure(repo, m), repo)
    assert rendered is not None
    assert list(rendered) == ["com/ex/K.java", "com/ex/Helper.java"]
    ksrc = rendered["com/ex/K.java"]
    # package kept, needed import kept, unneeded annotation import dropped
    assert "package com.ex;" in ksrc
    assert "import java.util.Arrays;" not in ksrc  # not referenced by members
    assert "jspecify" not in ksrc
    assert "static final double FACTOR = 2.0;" in ksrc
    assert "public static double scaled(double x)" in ksrc
    assert "countPos" not in ksrc  # not in the closure -- left out
    assert "tag()" not in ksrc
    hsrc = rendered["com/ex/Helper.java"]
    assert "public static double twice(double x)" in hsrc


def test_render_minimal_sources_compiles(tmp_path):
    repo = _repo_with_kernel(tmp_path / "repo")
    m = repo.find_method("com.ex.K.scaled")
    rendered = E.render_minimal_sources(m, E._extract_closure(repo, m), repo)
    assert rendered is not None
    out = tmp_path / "classes"
    srcs = []
    for rel, text in rendered.items():
        p = tmp_path / "synth" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        srcs.append(p)
    H.compile_java(srcs, out, [tmp_path / "synth"])
    assert (out / "com" / "ex" / "K.class").exists()
    assert (out / "com" / "ex" / "Helper.class").exists()


def test_render_minimal_sources_nested_class(tmp_path):
    _write(
        tmp_path / "repo",
        "src/o/Outer.java",
        "package o;\n"
        "public class Outer {\n"
        "    public static class Inner {\n"
        "        public static int f(int x) { return x + 1; }\n"
        "    }\n"
        "}\n",
    )
    repo = J.scan_repo(tmp_path / "repo")
    m = repo.find_method("o.Outer.Inner.f")
    assert m is not None
    rendered = E.render_minimal_sources(m, E._extract_closure(repo, m), repo)
    assert rendered is not None
    assert list(rendered) == ["o/Outer.java"]
    src = rendered["o/Outer.java"]
    assert "public static class Inner {" in src
    assert "public static int f(int x)" in src


def test_render_minimal_sources_bails_safely(tmp_path):
    # enum kind -> None (fall back to repo-file compile)
    _write(
        tmp_path / "repo",
        "src/e/E.java",
        "package e;\n"
        "public enum E { A, B;\n"
        "    public static int f(int x) { return x; } }\n",
    )
    repo = J.scan_repo(tmp_path / "repo")
    m = repo.find_method("e.E.f")
    assert E.render_minimal_sources(m, E._extract_closure(repo, m), repo) is None

    # constant without initializer -> None
    _write(
        tmp_path / "repo2",
        "src/c/C.java",
        "package c;\n"
        "public class C {\n"
        "    static final int N;\n"
        "    static { N = 3; }\n"
        "    public static int f(int x) { return x + N; }\n"
        "}\n",
    )
    repo2 = J.scan_repo(tmp_path / "repo2")
    m2 = repo2.find_method("c.C.f")
    assert E.render_minimal_sources(m2, E._extract_closure(repo2, m2), repo2) is None


# ---------------------------------------------------------------------------
# source_bytes cache
# ---------------------------------------------------------------------------


def test_source_bytes_caches_and_invalidates(tmp_path):
    p = _write(tmp_path, "a/F.java", "class F {}")
    J._source_bytes_cache.clear()
    first = J.source_bytes(tmp_path, "a/F.java")
    assert first == b"class F {}"
    # second read served from cache (same object)
    assert J.source_bytes(tmp_path, "a/F.java") is first
    # modify (size change forces invalidation via mtime+size)
    p.write_text("class F { int x; }", encoding="utf-8")
    third = J.source_bytes(tmp_path, "a/F.java")
    assert third == b"class F { int x; }"
    assert third is not first


# ---------------------------------------------------------------------------
# test-source exclusion
# ---------------------------------------------------------------------------


def test_scan_repo_skips_test_sources(tmp_path):
    _write(tmp_path, "src/main/java/com/ex/K.java", "package com.ex; public class K {}")
    _write(
        tmp_path,
        "src/test/java/com/ex/KTest.java",
        "package com.ex; public class KTest {}",
    )
    repo = J.scan_repo(tmp_path)
    assert "com.ex.K" in repo.classes
    assert "com.ex.KTest" not in repo.classes


def test_parsecache_skips_test_sources(tmp_path):
    from kernel_port import parsecache

    _write(tmp_path, "src/main/java/com/ex/K.java", "package com.ex; public class K {}")
    _write(
        tmp_path,
        "src/test/java/com/ex/KTest.java",
        "package com.ex; public class KTest {}",
    )
    repo = parsecache.load_repo(tmp_path, None, use_cache=False)
    assert "com.ex.K" in repo.classes
    assert "com.ex.KTest" not in repo.classes


def test_enum_body_declarations_parsed(tmp_path):
    # v0.8 dogfood: tree-sitter nests enum methods under
    # enum_body_declarations; they were invisible to mine before.
    _write(
        tmp_path,
        "src/e/E.java",
        "package e;\n"
        "public enum E { A, B;\n"
        "    public static int f(int x) { return x * 2; } }\n",
    )
    repo = J.scan_repo(tmp_path)
    m = repo.find_method("e.E.f")
    assert m is not None
    assert m.jtype == "int"
