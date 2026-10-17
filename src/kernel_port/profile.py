"""Profile-guided mining input: async-profiler collapsed stacks.

CPU profiles tell the miner *where time actually goes*, so kernel
candidates can be ranked by measured cost instead of static heuristics
alone. Typical flow::

    mine_candidates = miner.mine(repo)          # static ranking
    cpu = profile.parse_collapsed("out.collapsed")
    ranked = profile.rerank(mine_candidates, cpu)  # measured-CPU ranking

JFR is NOT parsed directly: there is no reliable pure-Python JFR parser in
this toolchain. The supported path is async-profiler, which converts a JFR
recording (or profiles live) into the collapsed-stack text format::

    asprof --action collapsed -f out.collapsed <pid>
    # ... or from an existing recording:
    asprof --action collapsed --jfr -f out.collapsed recording.jfr

then :func:`parse_collapsed` reads ``out.collapsed``.
"""

from __future__ import annotations

import re
from pathlib import Path

#: Optional header line carrying the sampling interval, e.g.
#: ``# interval: 10000000 ns``. async-profiler does not emit one by
#: default; when present, sample counts are converted to seconds.
_INTERVAL_RE = re.compile(
    r"(?i)^#?\s*interval[^0-9]*([\d.]+)\s*(ns|us|\u00b5s|ms|s)?\b"
)
_INTERVAL_MULT = {"ns": 1e-9, "us": 1e-6, "\u00b5s": 1e-6, "ms": 1e-3, "s": 1.0}

#: A Java frame after `/` -> `.` normalization, e.g. ``com.example.Kernels.dot``.
_JAVA_FRAME_RE = re.compile(r"^[\w$]+(\.[\w$]+)+$")
#: Native library frames are not Java methods, e.g. ``libjvm.so``.
_NATIVE_LIB_RE = re.compile(r"\.(so|dll|dylib)(\.\d+)*$")
#: async-profiler frame-type annotations, e.g. ``com.example.Kernels.dot_[j]``.
_FRAME_ANN_RE = re.compile(r"_\[[jki]\]$")


def _parse_interval_ns(line: str) -> float | None:
    m = _INTERVAL_RE.match(line.strip())
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    unit = (m.group(2) or "ns").lower()
    return val * _INTERVAL_MULT.get(unit, 1e-9)


def parse_collapsed(path: str | Path) -> dict[str, float]:
    """Parse an async-profiler collapsed-stack file.

    Format: one ``frame;frame;... <samples>`` line per stack, frames
    ordered outermost-first. Comment lines (``#...``) and malformed lines
    are skipped. An optional ``# interval: <n> <unit>`` header converts
    sample counts to seconds; without it the values are raw sample counts
    (still proportional to CPU time).

    Samples are attributed to Java methods by simple name: the DEEPEST
    frame matching the ``pkg.Class.method`` shape wins, and counts are
    aggregated per ``Class.method`` (simple class name + method name).
    """
    interval_s: float | None = None
    agg: dict[str, float] = {}
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        iv = _parse_interval_ns(line)
        if iv is not None and interval_s is None:
            interval_s = iv
            continue
        if line.startswith("#"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        stack_s, count_s = parts
        try:
            count = float(count_s)
        except ValueError:
            continue
        target: str | None = None
        for frame in reversed(stack_s.split(";")):  # deepest first
            f = _FRAME_ANN_RE.sub("", frame.strip().replace("/", "."))
            if _NATIVE_LIB_RE.search(f):
                continue  # e.g. libjvm.so -- not a Java method
            if _JAVA_FRAME_RE.match(f):
                target = f
                break
        if target is None:
            continue
        key = ".".join(target.split(".")[-2:])  # Class.method
        agg[key] = agg.get(key, 0.0) + count
    if interval_s is not None:
        agg = {k: v * interval_s for k, v in agg.items()}
    return agg


def rerank(candidates: list, cpu_map: dict[str, float]) -> list:
    """Attach measured CPU info to candidates and sort pure ones by CPU desc.

    Each candidate gets two new attributes (via setattr, so the
    :class:`miner.Candidate` dataclass is untouched):

    - ``profile_cpu_s``: measured CPU for ``Class.method`` (seconds when
      the collapsed file carried an interval header, else sample counts --
      proportional either way; 0.0 when the method was not observed).
    - ``profile_share``: fraction of the total observed CPU.

    Pure candidates come first, sorted by measured CPU descending; impure
    ones follow, also by CPU descending. Candidates are sorted in place
    semantics (a new list is returned; the objects are mutated).
    """
    total = sum(cpu_map.values())
    for c in candidates:
        # method candidates carry .qualname; cluster candidates carry .entry
        qn = getattr(c, "qualname", None) or getattr(c, "entry", "")
        cls_simple = qn.rsplit(".", 1)[0].rsplit(".", 1)[-1]
        meth = qn.rsplit(".", 1)[-1]
        cpu = float(cpu_map.get(f"{cls_simple}.{meth}", 0.0))
        setattr(c, "profile_cpu_s", cpu)
        setattr(c, "profile_share", cpu / total if total > 0 else 0.0)
    return sorted(
        candidates,
        key=lambda c: (not c.pure, -getattr(c, "profile_cpu_s", 0.0)),
    )
