"""Tests for kernel_port.parsecache (incremental parse cache).

Every test isolates the cache via KERNEL_PORT_PARSE_CACHE_DIR pointed at a
tmp dir, and works on a writable copy of the sample_kernels fixture.  No JDK
needed: all assertions run on the parsed model plus pure-Python analyses.
"""

import json
import os
import shutil
from pathlib import Path

import pytest

from kernel_port import extractor, java_ast as J, parsecache
from kernel_port.purity import check_purity_repo

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_kernels"


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "KERNEL_PORT_PARSE_CACHE_DIR", str(tmp_path / "pcache")
    )
    yield tmp_path


@pytest.fixture()
def repo_dir(tmp_path):
    target = tmp_path / "repo"
    shutil.copytree(FIXTURE_REPO, target)
    return target


def _norm_src_bytes(repo: J.JavaRepo, rel: str) -> bytes:
    raw = (repo.root / rel).read_bytes()
    return raw.decode("utf-8", errors="replace").encode("utf-8")


def _tree_key(node):
    if node is None:
        return None
    return (
        node.type,
        node.is_named,
        node.start_byte,
        node.end_byte,
        (node.start_point[0], node.start_point[1]),
        (node.end_point[0], node.end_point[1]),
        tuple(_tree_key(c) for c in node.children),
    )


def _method_key(m: J.MethodInfo):
    return (
        m.qualname,
        m.class_qualname,
        m.name,
        m.jtype,
        tuple((p.name, p.jtype) for p in m.params),
        sorted(m.modifiers),
        m.is_static,
        m.is_constructor,
        m.file,
        m.lineno,
        m.end_lineno,
        m.source,
    )


def assert_repos_equal(a: J.JavaRepo, b: J.JavaRepo):
    """Deep equality: model facts, ordering, and body trees (offsets + text)."""
    assert a.root == b.root
    assert a.files == b.files
    assert list(a.classes) == list(b.classes)
    for qn, ca in a.classes.items():
        cb = b.classes[qn]
        assert (ca.qualname, ca.package, ca.name, ca.file, ca.kind) == (
            cb.qualname,
            cb.package,
            cb.name,
            cb.file,
            cb.kind,
        )
        assert ca.modifiers == cb.modifiers  # v0.12: cache round-trips these
        assert list(ca.methods) == list(cb.methods)
        assert [_method_key(m) for m in ca.all_methods] == [
            _method_key(m) for m in cb.all_methods
        ]
        assert list(ca.fields) == list(cb.fields)
        for fname, fa in ca.fields.items():
            fb = cb.fields[fname]
            assert (
                fa.qualname,
                fa.name,
                fa.jtype,
                sorted(fa.modifiers),
                fa.is_constant,
                fa.file,
                fa.lineno,
                fa.source,
            ) == (
                fb.qualname,
                fb.name,
                fb.jtype,
                sorted(fb.modifiers),
                fb.is_constant,
                fb.file,
                fb.lineno,
                fb.source,
            )
        # Body trees: structurally identical and slice the same source text.
        for ma, mb in zip(ca.all_methods, cb.all_methods):
            assert _tree_key(ma.body) == _tree_key(mb.body)
            if ma.body is not None:
                src = _norm_src_bytes(a, ma.file)
                assert J.text(ma.body, src) == J.text(mb.body, src)
    assert set(a.methods) == set(b.methods)
    for key in a.methods:
        assert [m.signature for m in a.methods[key]] == [
            m.signature for m in b.methods[key]
        ]


def _entry_file(abs_path: Path) -> Path:
    return parsecache._entry_path(
        parsecache.cache_dir(), str(Path(os.path.realpath(abs_path)))
    )


# ---------------------------------------------------------------------------
# Core: cold/warm equality and stats
# ---------------------------------------------------------------------------


def test_cold_warm_deep_equal(repo_dir):
    cold = parsecache.load_repo(repo_dir)
    warm = parsecache.load_repo(repo_dir)
    assert_repos_equal(cold, warm)


def test_stats_cold_then_warm(repo_dir):
    parsecache.load_repo(repo_dir)
    assert parsecache.cache_stats() == {
        "files_total": 2,
        "hits": 0,
        "misses": 2,
    }
    parsecache.load_repo(repo_dir)
    assert parsecache.cache_stats() == {
        "files_total": 2,
        "hits": 2,
        "misses": 0,
    }


def test_cache_files_land_in_versioned_dir(repo_dir, tmp_path):
    parsecache.load_repo(repo_dir)
    v = parsecache.CACHE_VERSION
    entries = list((tmp_path / "pcache" / "parse" / f"v{v}").glob("*.json"))
    assert len(entries) == 2
    payload = json.loads(entries[0].read_text(encoding="utf-8"))
    assert payload["v"] == v
    assert {"path", "mtime_ns", "size", "classes"} <= set(payload)


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------


def test_touch_mtime_reparses_only_touched(repo_dir):
    parsecache.load_repo(repo_dir)  # cold: populate
    parsecache.load_repo(repo_dir)  # warm
    assert parsecache.cache_stats()["hits"] == 2

    app = repo_dir / "src" / "com" / "example" / "App.java"
    st = app.stat()
    os.utime(app, (st.st_atime, st.st_mtime + 5))  # content unchanged, mtime bumped

    warm = parsecache.load_repo(repo_dir)
    assert parsecache.cache_stats() == {
        "files_total": 2,
        "hits": 1,
        "misses": 1,
    }
    assert_repos_equal(parsecache.load_repo(repo_dir, use_cache=False), warm)


def test_content_change_reparses_and_reflected(repo_dir):
    cold = parsecache.load_repo(repo_dir)
    app = repo_dir / "src" / "com" / "example" / "App.java"
    app.write_text(
        app.read_text(encoding="utf-8")
        + "\nclass Extra {\n    public static int one() { return 1; }\n}\n",
        encoding="utf-8",
    )
    warm = parsecache.load_repo(repo_dir)
    stats = parsecache.cache_stats()
    assert stats["misses"] == 1 and stats["hits"] == 1
    assert "com.example.Extra" in warm.classes
    assert "com.example.Extra" not in cold.classes
    assert_repos_equal(parsecache.load_repo(repo_dir, use_cache=False), warm)


def test_new_file(repo_dir):
    parsecache.load_repo(repo_dir)
    new = repo_dir / "src" / "com" / "example" / "BrandNew.java"
    new.write_text(
        "package com.example;\n"
        "public class BrandNew {\n"
        "    public static long twice(long x) { return 2 * x; }\n"
        "}\n",
        encoding="utf-8",
    )
    warm = parsecache.load_repo(repo_dir)
    stats = parsecache.cache_stats()
    assert stats == {"files_total": 3, "hits": 2, "misses": 1}
    assert "com.example.BrandNew" in warm.classes
    m = warm.find_method("com.example.BrandNew.twice")
    assert m is not None and m.signature == "long twice(long x)"


def test_deleted_file_dropped_and_pruned(repo_dir):
    parsecache.load_repo(repo_dir)
    app = repo_dir / "src" / "com" / "example" / "App.java"
    entry = _entry_file(app)
    assert entry.exists()
    app.unlink()

    warm = parsecache.load_repo(repo_dir)
    assert parsecache.cache_stats()["files_total"] == 1
    assert not any("App.java" in f for f in warm.files)
    assert "com.example.App" not in warm.classes
    assert not entry.exists(), "stale cache entry should be pruned"


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_corrupt_cache_recovers(repo_dir):
    parsecache.load_repo(repo_dir)
    kernels = repo_dir / "src" / "com" / "example" / "Kernels.java"
    entry = _entry_file(kernels)
    entry.write_text("this is not json {{{", encoding="utf-8")

    warm = parsecache.load_repo(repo_dir)  # must not crash
    assert parsecache.cache_stats() == {
        "files_total": 2,
        "hits": 1,
        "misses": 1,
    }
    assert_repos_equal(parsecache.load_repo(repo_dir, use_cache=False), warm)
    # Healed: entry rewritten as valid JSON on the miss path.
    assert json.loads(entry.read_text(encoding="utf-8"))["v"] == parsecache.CACHE_VERSION


def test_corrupt_entry_structure_recovers(repo_dir):
    """Valid JSON but garbage model content -> treated as miss, never crash."""
    parsecache.load_repo(repo_dir)
    kernels = repo_dir / "src" / "com" / "example" / "Kernels.java"
    entry = _entry_file(kernels)
    payload = json.loads(entry.read_text(encoding="utf-8"))
    payload["classes"] = [{"bogus": True}]
    entry.write_text(json.dumps(payload), encoding="utf-8")

    warm = parsecache.load_repo(repo_dir)
    assert parsecache.cache_stats()["misses"] == 1
    assert_repos_equal(parsecache.load_repo(repo_dir, use_cache=False), warm)


def test_use_cache_false_bypasses_and_matches_scan_repo(repo_dir, tmp_path):
    repo = parsecache.load_repo(repo_dir, use_cache=False)
    assert parsecache.cache_stats() == {
        "files_total": 2,
        "hits": 0,
        "misses": 2,
    }
    # Untouched cache dir: bypass must not read or write anything.
    assert not (tmp_path / "pcache").exists()
    assert_repos_equal(J.scan_repo(repo_dir), repo)


def test_refresh_reparses_all(repo_dir):
    parsecache.load_repo(repo_dir)
    parsecache.load_repo(repo_dir)
    assert parsecache.cache_stats()["hits"] == 2

    warm = parsecache.load_repo(repo_dir, refresh=True)
    assert parsecache.cache_stats() == {
        "files_total": 2,
        "hits": 0,
        "misses": 2,
    }
    assert_repos_equal(parsecache.load_repo(repo_dir, use_cache=False), warm)


def test_missing_root_raises_like_scan_repo(tmp_path):
    with pytest.raises(FileNotFoundError):
        parsecache.load_repo(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        J.scan_repo(tmp_path / "nope")


# ---------------------------------------------------------------------------
# Behavioral parity: cached bodies must analyze identically
# ---------------------------------------------------------------------------


def test_purity_reports_identical(repo_dir):
    cold = parsecache.load_repo(repo_dir, use_cache=False)
    parsecache.load_repo(repo_dir)  # populate cache
    warm = parsecache.load_repo(repo_dir)

    def summarize(repo):
        return {
            qn: (rep.pure, sorted(rep.reasons), sorted(rep.pure_callees))
            for qn, rep in check_purity_repo(repo).items()
        }

    cold_reps = summarize(cold)
    assert len(cold_reps) > 0  # sanity: fixture has methods
    assert summarize(warm) == cold_reps
    # Sanity: fixture contains both pure and impure methods.
    assert any(pure for pure, _, _ in cold_reps.values())
    assert any(not pure for pure, _, _ in cold_reps.values())


def test_extractor_output_identical(repo_dir):
    cold = parsecache.load_repo(repo_dir, use_cache=False)
    parsecache.load_repo(repo_dir)
    warm = parsecache.load_repo(repo_dir)

    for qn in ("com.example.Kernels.dot", "com.example.Kernels.matMul"):
        assert extractor.to_json(extractor.extract(warm, qn)) == extractor.to_json(
            extractor.extract(cold, qn)
        )


# ---------------------------------------------------------------------------
# Misc API
# ---------------------------------------------------------------------------


def test_cache_dir_resolution(monkeypatch):
    v = parsecache.CACHE_VERSION
    monkeypatch.delenv("KERNEL_PORT_PARSE_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/xdg-test")
    assert str(parsecache.cache_dir()) == \
        f"/tmp/xdg-test/kernel-port/parse/v{v}"
    monkeypatch.setenv("KERNEL_PORT_PARSE_CACHE_DIR", "/tmp/override")
    assert str(parsecache.cache_dir()) == \
        f"/tmp/override/parse/v{v}"


def test_clear_cache(repo_dir):
    parsecache.load_repo(repo_dir)
    cdir = parsecache.cache_dir()
    assert len(list(cdir.glob("*.json"))) == 2
    assert parsecache.clear_cache(repo_dir) == 2
    assert list(cdir.glob("*.json")) == []
    # After clearing, next load is a full miss again.
    parsecache.load_repo(repo_dir)
    assert parsecache.cache_stats() == {
        "files_total": 2,
        "hits": 0,
        "misses": 2,
    }


def test_cache_survives_scan_root_change(repo_dir):
    """Regression (v0.9 dogfood): cached `file` rels were stored relative to
    the root used when the entry was written. Re-scanning the same files
    under a different root reused the stale rel -> `root/file` doubled and
    source_bytes crashed with FileNotFoundError."""
    # Warm the cache scanning at the repo root...
    r1 = parsecache.load_repo(repo_dir)
    assert parsecache.cache_stats()["hits"] == 0
    # ...then scan the src/ subtree as its own root: all hits, but the rels
    # must be re-stamped to the new root.
    sub = repo_dir / "src"
    r2 = parsecache.load_repo(sub)
    stats = parsecache.cache_stats()
    assert stats["hits"] == stats["files_total"] > 0
    assert r2.root == sub
    for rel in r2.files:
        assert not rel.startswith("src/"), f"stale rel not re-stamped: {rel}"
    # Every method's file must resolve under the new root (this crashed).
    for key, methods in r2.methods.items():
        for m in methods:
            src = J.source_bytes(r2.root, m.file)
            assert len(src) > 0
    # And the model must equal a fresh no-cache scan of the subtree.
    r3 = parsecache.load_repo(sub, use_cache=False)
    assert_repos_equal(r2, r3)
    # Sanity: the original root scan still works too.
    r4 = parsecache.load_repo(repo_dir)
    assert_repos_equal(r1, r4)
