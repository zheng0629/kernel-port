"""Tests for kernel_port.coverage (v0.6 workstream D).

JVM-backed tests compile small fixtures with javac and drive them through
the harness line protocol; pure-Python tests cover guided-case heuristics
and ddmin directly.
"""

import contextlib
import random
import shutil

import pytest

from kernel_port import coverage as C
from kernel_port import harness as H
from kernel_port import java_ast as J

SRC_BRANCHY = """
public class K {
    public static int f(int x) {
        int r;
        if (x > 0) { r = 1; } else if (x == 0) { r = 2; } else { r = -1; }
        r = x > 10 ? 100 : r + 200;
        for (int i = 0; i < x; i++) { r += i; }
        while (r > 5000) { r -= 1000; }
        switch (x) {
            case 1: r += 1; break;
            case 2: r += 2; break;
            default: r += 3;
        }
        try { r = r / x; } catch (ArithmeticException e) { r = -999; }
        return r;
    }
}
"""


def _src_dir(tmp_path, files, tag="src"):
    d = tmp_path / tag
    d.mkdir(parents=True, exist_ok=True)
    for rel, src in files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src, encoding="utf-8")
    return d


def _method(src_dir, qualname):
    m = J.scan_repo(src_dir).find_method(qualname)
    assert m is not None, qualname
    return m


def _compile_plain(src_dir, method, harness_cls="HarnessJava"):
    """Compile original sources + a standard harness; return classes dir."""
    hp = src_dir / f"{harness_cls}.java"
    hp.write_text(H.render_harness(method, "java", harness_cls), encoding="utf-8")
    classes = src_dir / f"classes_{harness_cls}"
    H.compile_java(sorted(src_dir.rglob("*.java")), classes)
    return classes


@contextlib.contextmanager
def _session(classes_dir, cls="HarnessJava"):
    s = H.LineSession([shutil.which("java"), "-cp", str(classes_dir), cls])
    try:
        yield s
    finally:
        s.close()


def _kinds(ic):
    counts = {}
    for b in ic.branches:
        counts[b.kind] = counts.get(b.kind, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# instrumentation
# ---------------------------------------------------------------------------


def test_instrument_branch_inventory(tmp_path):
    ic = C.instrument({"K.java": SRC_BRANCHY}, work_dir=tmp_path / "cov")
    assert (tmp_path / "cov" / "K.java").exists()
    assert _kinds(ic) == {
        "if-then": 2,
        "if-else": 2,
        "ternary-true": 1,
        "ternary-false": 1,
        "for-body": 1,
        "while-body": 1,
        "switch-case": 2,
        "switch-default": 1,
        "catch": 1,
    }
    assert [b.global_id for b in ic.branches] == list(range(12))
    assert all(b.class_qualname == "K" for b in ic.branches)
    assert all(b.method == "f" for b in ic.branches)
    assert ic.coverage_classes == [("K", "__kp_hit")]
    conds = {(b.kind, b.condition) for b in ic.branches}
    assert ("if-then", "(x > 0)") in conds
    assert ("ternary-false", "x > 10") in conds
    assert ("switch-case", "case 1") in conds
    assert ("catch", "ArithmeticException") in conds


def test_instrumented_semantics_match_original(tmp_path):
    files = {"K.java": SRC_BRANCHY}
    src_dir = _src_dir(tmp_path, files)
    m = _method(src_dir, "K.f")
    orig_classes = _compile_plain(src_dir, m)

    ic = C.instrument(files, work_dir=tmp_path / "cov")
    hp = ic.work_dir / "HarnessJava.java"
    hp.write_text(H.render_harness(m, "java", "HarnessJava"), encoding="utf-8")
    inst_classes = ic.work_dir / "classes_ab"
    H.compile_java(sorted(ic.work_dir.rglob("*.java")), inst_classes)

    rng = random.Random(1234)
    with _session(orig_classes) as s1, _session(inst_classes) as s2:
        for _ in range(60):
            args = H.gen_case(m, rng)
            line = H.encode_call(m, args)
            r1 = s1.ask(line, 10)
            r2 = s2.ask(line, 10)
            # Identical control flow => identical outcomes, including
            # out-of-domain rejections (x == 0 hits the catch in both).
            assert r1 == r2, (args, r1, r2)


def test_instrument_name_collision_avoided(tmp_path):
    src = """
public class K {
    int __kp_hit = 5;
    public static int f(int x) {
        if (x > 0) { return 1; }
        return 0;
    }
}
"""
    ic = C.instrument({"K.java": src}, work_dir=tmp_path / "cov")
    text = (tmp_path / "cov" / "K.java").read_text(encoding="utf-8")
    assert "__kp_hit2" in text  # mangled away from the existing field
    assert ic.coverage_classes == [("K", "__kp_hit2")]
    H.compile_java([tmp_path / "cov" / "K.java"], tmp_path / "classes")
    assert len(ic.branches) == 2  # then + synthesized else


# ---------------------------------------------------------------------------
# run_covered
# ---------------------------------------------------------------------------


def test_run_covered_hit_counts(tmp_path):
    src_dir = _src_dir(tmp_path, {"K.java": SRC_BRANCHY})
    m = _method(src_dir, "K.f")
    ic = C.instrument({"K.java": SRC_BRANCHY}, work_dir=tmp_path / "cov")
    cov = C.run_covered(m, [[1]], ic)
    assert cov.branches_total == 12
    assert cov.branches_hit == 4
    assert cov.coverage_pct == pytest.approx(100.0 * 4 / 12)

    by_kind = {}
    for b in ic.branches:
        by_kind.setdefault(b.kind, []).append(cov.hits_for(b.global_id))
    # x = 1: then, ternary-false, for body, case 1.
    assert by_kind["if-then"] == [1, 0]
    assert by_kind["if-else"] == [0, 0]
    assert by_kind["ternary-true"] == [0]
    assert by_kind["ternary-false"] == [1]
    assert by_kind["for-body"] == [1]
    assert by_kind["while-body"] == [0]
    assert by_kind["catch"] == [0]
    # switch-case: case 1 hit, case 2 not; default not hit.
    case_hits = {
        b.condition: cov.hits_for(b.global_id)
        for b in ic.branches
        if b.kind == "switch-case"
    }
    assert case_hits["case 1"] == 1
    assert case_hits["case 2"] == 0
    assert by_kind["switch-default"] == [0]


def test_run_covered_out_of_domain_inputs_skipped(tmp_path):
    # x = 0 -> ArithmeticException -> catch arm; the Java harness answers
    # OK (not ERR) because the exception is caught inside the kernel.
    src_dir = _src_dir(tmp_path, {"K.java": SRC_BRANCHY})
    m = _method(src_dir, "K.f")
    ic = C.instrument({"K.java": SRC_BRANCHY}, work_dir=tmp_path / "cov")
    cov = C.run_covered(m, [[0]], ic)
    catch = next(b for b in ic.branches if b.kind == "catch")
    assert cov.hits_for(catch.global_id) == 1
    inner_then = [
        b for b in ic.branches if b.kind == "if-then" and "(x == 0)" in b.condition
    ][0]
    assert cov.hits_for(inner_then.global_id) == 1


def test_coverage_pct_math_and_empty_method(tmp_path):
    cov = C.CoverageData(
        branches_total=4,
        branches_hit=3,
        per_branch_hits={0: 1, 1: 2, 2: 0, 3: 5},
        branches=[
            C.BranchInfo(i, "K", "if-then", "c", 1, "f", local_id=i)
            for i in range(4)
        ],
    )
    assert cov.coverage_pct == 75.0
    assert [b.global_id for b in cov.uncovered()] == [2]

    src = "public class K6 { public static int id(int x) { return x * 2 + 1; } }"
    src_dir = _src_dir(tmp_path, {"K6.java": src})
    m = _method(src_dir, "K6.id")
    ic = C.instrument({"K6.java": src}, work_dir=tmp_path / "cov")
    assert ic.branches == []
    cov = C.run_covered(m, [[1], [2], [3]], ic)
    assert cov.branches_total == 0
    assert cov.coverage_pct == 100.0


# ---------------------------------------------------------------------------
# guided_cases
# ---------------------------------------------------------------------------


def test_guided_covers_rare_branch(tmp_path):
    src = """
public class K2 {
    public static int g(int x) {
        if (x == 1234567) { return 1; }
        return 0;
    }
}
"""
    src_dir = _src_dir(tmp_path, {"K2.java": src})
    m = _method(src_dir, "K2.g")
    ic = C.instrument({"K2.java": src}, work_dir=tmp_path / "cov")

    rng = random.Random(7)
    cov1 = C.run_covered(m, [H.gen_case(m, rng) for _ in range(30)], ic)
    then = next(b for b in ic.branches if b.kind == "if-then")
    assert cov1.hits_for(then.global_id) == 0  # random never guesses 1234567

    guided = C.guided_cases(m, cov1, random.Random(7), 10)
    assert len(guided) == 10
    cov2 = C.run_covered(m, guided, ic)
    assert cov2.hits_for(then.global_id) > 0
    assert cov2.coverage_pct == 100.0


def test_guided_covers_loop_body_via_literal(tmp_path):
    src = """
public class K7 {
    public static int w(int[] a, int n) {
        int s = 0;
        int i = 0;
        while (n > 100) { s += a[i % a.length]; i++; n--; }
        return s;
    }
}
"""
    src_dir = _src_dir(tmp_path, {"K7.java": src})
    m = _method(src_dir, "K7.w")
    ic = C.instrument({"K7.java": src}, work_dir=tmp_path / "cov")

    rng = random.Random(11)
    cov1 = C.run_covered(m, [H.gen_case(m, rng) for _ in range(30)], ic)
    body = next(b for b in ic.branches if b.kind == "while-body")
    assert cov1.hits_for(body.global_id) == 0  # gen_case keeps n <= 8

    guided = C.guided_cases(m, cov1, random.Random(11), 12)
    cov2 = C.run_covered(m, guided, ic)
    assert cov2.hits_for(body.global_id) > 0


# ---------------------------------------------------------------------------
# minimize
# ---------------------------------------------------------------------------


def test_minimize_shrinks_array():
    fails = lambda case: 999 in case[0]  # noqa: E731
    result = C.minimize([list(range(1000))], fails)
    assert result == [[999]]
    assert fails(result)


def test_minimize_shrinks_int_toward_zero():
    fails = lambda case: case[0] > 100  # noqa: E731
    result = C.minimize([1000], fails)
    assert fails(result)
    assert result[0] < 1000
    # 1-minimal w.r.t. the halving step: one more halving would not fail.
    assert not fails([result[0] // 2 if result[0] > 0 else result[0]])


def test_minimize_string_prefix():
    fails = lambda case: "needle" in case[0]  # noqa: E731
    result = C.minimize(["x" * 500 + "needle" + "y" * 500], fails)
    assert result == ["x" * 500 + "needle"]
    assert fails(result)


def test_minimize_2d_array():
    grid = [[1] * 40 for _ in range(40)]
    grid[37][13] = 7
    fails = lambda case: any(c == 7 for row in case[0] for c in row)  # noqa: E731
    result = C.minimize([grid], fails)
    assert fails(result)
    assert result == [[[0] * 13 + [7]]]


def test_minimize_rejects_non_failing():
    with pytest.raises(ValueError):
        C.minimize([[1, 2, 3]], lambda case: False)


# ---------------------------------------------------------------------------
# tricky instrumentation shapes
# ---------------------------------------------------------------------------


def test_ternary_in_constant_context_not_instrumented(tmp_path):
    # Instrumenting this ternary would destroy N's compile-time-constant
    # status and break `case N:`. The instrumenter must skip it.
    src = """
public class K3 {
    static final int N = true ? 4 : 8;
    public static int h(int x) {
        switch (x) { case N: return 1; default: return 2; }
    }
}
"""
    ic = C.instrument({"K3.java": src}, work_dir=tmp_path / "cov")
    assert not [b for b in ic.branches if b.kind.startswith("ternary")]
    H.compile_java([tmp_path / "cov" / "K3.java"], tmp_path / "classes")

    src_dir = _src_dir(tmp_path, {"K3.java": src})
    m = _method(src_dir, "K3.h")
    cov = C.run_covered(m, [[4], [5]], ic)
    assert cov.coverage_pct == 100.0


def test_switch_arrow_and_fallthrough(tmp_path):
    src = """
public class K4 {
    public static int h(int x) {
        int y = switch (x) {
            case 1 -> 10;
            case 2 -> { int q = 3; yield q * 10; }
            default -> 30;
        };
        switch (x) {
            case 1:
            case 2: y += 1; break;
            default: y += 2;
        }
        return y;
    }
}
"""
    files = {"K4.java": src}
    src_dir = _src_dir(tmp_path, files)
    m = _method(src_dir, "K4.h")
    orig_classes = _compile_plain(src_dir, m)

    ic = C.instrument(files, work_dir=tmp_path / "cov")
    assert _kinds(ic) == {"switch-case": 4, "switch-default": 2}

    # Semantics: A/B compare, including fall-through (x=1 hits both labels).
    hp = ic.work_dir / "HarnessJava.java"
    hp.write_text(H.render_harness(m, "java", "HarnessJava"), encoding="utf-8")
    inst_classes = ic.work_dir / "classes_ab"
    H.compile_java(sorted(ic.work_dir.rglob("*.java")), inst_classes)
    with _session(orig_classes) as s1, _session(inst_classes) as s2:
        for x in (1, 2, 5):
            r1 = s1.ask(H.encode_call(m, [x]), 10)
            r2 = s2.ask(H.encode_call(m, [x]), 10)
            assert r1 == r2 == "OK i:%d" % (11 if x == 1 else 31 if x == 2 else 32)

    # x = 1: arrow "case 1" fires; colon "case 1" AND "case 2" fire via
    # fall-through (counter sits right after each label).
    cov = C.run_covered(m, [[1]], ic)
    # Arrow rules live on lines 5-7 of the fixture, colon labels on 10-12.
    arrow_hits = {
        b.condition: cov.hits_for(b.global_id)
        for b in ic.branches
        if b.kind == "switch-case" and b.line < 9
    }
    colon_hits = {
        b.condition: cov.hits_for(b.global_id)
        for b in ic.branches
        if b.kind == "switch-case" and b.line >= 9
    }
    assert arrow_hits == {"case 1": 1, "case 2": 0}
    assert colon_hits == {"case 1": 1, "case 2": 1}  # fall-through


def test_do_while_and_enhanced_for(tmp_path):
    src = """
public class K5 {
    public static int k(int[] a) {
        int s = 0;
        for (int v : a) { s += v; }
        int i = 0;
        do { s += i; i++; } while (i < 3);
        return s;
    }
}
"""
    files = {"K5.java": src}
    src_dir = _src_dir(tmp_path, files)
    m = _method(src_dir, "K5.k")
    orig_classes = _compile_plain(src_dir, m)

    ic = C.instrument(files, work_dir=tmp_path / "cov")
    assert _kinds(ic) == {"enhanced-for-body": 1, "do-body": 1}

    hp = ic.work_dir / "HarnessJava.java"
    hp.write_text(H.render_harness(m, "java", "HarnessJava"), encoding="utf-8")
    inst_classes = ic.work_dir / "classes_ab"
    H.compile_java(sorted(ic.work_dir.rglob("*.java")), inst_classes)
    rng = random.Random(99)
    with _session(orig_classes) as s1, _session(inst_classes) as s2:
        for _ in range(20):
            args = H.gen_case(m, rng)
            line = H.encode_call(m, args)
            assert s1.ask(line, 10) == s2.ask(line, 10)

    cov = C.run_covered(m, [[[1, 2]]], ic)
    assert cov.coverage_pct == 100.0
