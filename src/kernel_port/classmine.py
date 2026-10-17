"""Class-level mining: detect whole classes portable as a unit.

Only two shapes are portable:

- *stateless* utility classes: all methods static, no instance state
  (``static final`` constants are fine);
- *immutable* classes: every instance field ``final``, no method writes an
  instance field outside a constructor, every method pure.

Anything else is listed as rejected with concrete reasons -- never silently
dropped. Ranking mirrors method-level mining: portable candidates first,
then by ``(1 + calls) * mean_intensity``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import java_ast as J
from .miner import _count_calls, _intensity, _method_source
from .purity import ClassPurityReport, check_class_portable


@dataclass
class ClassCandidate:
    qualname: str
    kind: str  # "stateless" | "immutable" | "rejected"
    portable: bool
    reasons: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)  # method qualnames
    fields: list[str] = field(default_factory=list)  # field names
    score: float = 0.0


def find_class_candidates(repo: J.JavaRepo) -> list[ClassCandidate]:
    """Return class candidates, portable first, then by score descending."""
    call_counts = _count_calls(repo)
    cands: list[ClassCandidate] = []
    for qualname, cls in repo.classes.items():
        if cls.kind != "class" and not cls.all_methods:
            continue  # skip interfaces/enums/records without methods
        if not cls.all_methods:
            continue  # nothing to port
        report: ClassPurityReport = check_class_portable(cls, repo)
        calls = sum(call_counts.get(m.qualname, 0) for m in cls.all_methods)
        intensities = [_intensity(m, _method_source(repo, m)) for m in cls.all_methods]
        mean_intensity = sum(intensities) / len(intensities)
        score = (1 + calls) * mean_intensity
        cands.append(
            ClassCandidate(
                qualname=qualname,
                kind=report.kind,
                portable=report.portable,
                reasons=report.reasons,
                methods=[m.qualname for m in cls.all_methods],
                fields=[f.name for f in cls.fields.values()],
                score=round(score, 1),
            )
        )
    cands.sort(key=lambda c: (c.portable, c.score), reverse=True)
    return cands
