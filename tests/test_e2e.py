"""End-to-end: glue -> (simulated agent translation) -> verify -> bench.

These tests need a JDK and g++ on PATH and take a few seconds each (two JVM
starts + javac + g++ per test).
"""

import pytest

from kernel_port import harness as H
from kernel_port import jni

from .conftest import FIXTURE_REPO, TRANSLATIONS, fill_translation

pytestmark = pytest.mark.e2e


def _build(repo, qualname: str, tmp_path, translation_key: str | None = None):
    m = repo.find_method(qualname)
    assert m is not None
    glue_dir = tmp_path / "glue"
    jni.generate(m, glue_dir)
    key = translation_key or qualname
    fill_translation(glue_dir, qualname, TRANSLATIONS[key])
    work = tmp_path / "work"
    return H.build_env(FIXTURE_REPO, m, glue_dir, work)


def test_e2e_verify_dot_passes(repo, tmp_path):
    env = _build(repo, "com.example.Kernels.dot", tmp_path)
    report = H.run_verify(env, cases=20, seed=1)
    assert report.all_passed, [c.detail for c in report.cases if not c.ok]
    assert report.passed == 20


def test_e2e_verify_detects_wrong_translation(repo, tmp_path):
    env = _build(
        repo, "com.example.Kernels.dot", tmp_path,
        translation_key="com.example.Kernels.dot_WRONG",
    )
    report = H.run_verify(env, cases=10, seed=1)
    assert not report.all_passed
    assert report.failed > 0


def test_e2e_verify_fib_passes(repo, tmp_path):
    env = _build(repo, "com.example.Kernels.fib", tmp_path)
    report = H.run_verify(env, cases=20, seed=3)
    assert report.all_passed, [c.detail for c in report.cases if not c.ok]


def test_e2e_bench_smoke(repo, tmp_path):
    env = _build(repo, "com.example.Kernels.dot", tmp_path)
    report = H.run_bench(env, iters=200, warmup=50, repeats=2)
    assert report.java_per_call_ns > 0
    assert report.jni_per_call_ns > 0
    assert report.speedup > 0
