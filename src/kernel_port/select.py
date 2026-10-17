"""Multi-candidate selection: pick the fastest verified translation.

A calling agent may produce several candidate translations for one kernel
(e.g. different optimization attempts). ``pick_winner`` keeps only the
candidates that passed differential verification and returns the fastest
one; ``compare_table`` renders a plain-text comparison for reports.
"""

from __future__ import annotations


def pick_winner(results: list[dict]) -> dict:
    """Return the winning candidate result dict.

    Each result is ``{name, verify_passed, verify_failed, bench_median_ns}``.
    Eligible = ``verify_failed == 0`` and ``verify_passed > 0``. The winner
    is the eligible candidate with the smallest ``bench_median_ns``; ties
    keep the first candidate in list order (``min`` is stable). Raises
    ``ValueError("no passing candidate")`` when nothing is eligible.
    """
    eligible = [
        r
        for r in results
        if (r.get("verify_failed") or 0) == 0 and (r.get("verify_passed") or 0) > 0
    ]
    if not eligible:
        raise ValueError("no passing candidate")
    return min(eligible, key=lambda r: r.get("bench_median_ns", float("inf")))


def compare_table(results: list[dict]) -> str:
    """Plain-text comparison table of candidates.

    Columns: name / verify passed-failed / median ns / verdict, where the
    verdict is ``WINNER *`` for the picked candidate, ``FAILED`` for
    candidates that failed verification, and ``slower`` for passing
    candidates that lost on speed.
    """
    try:
        winner_name = pick_winner(results).get("name")
    except ValueError:
        winner_name = None
    header = f"{'name':<24}{'verify':<14}{'median_ns':>12}  verdict"
    lines = [header, "-" * len(header)]
    for r in results:
        name = str(r.get("name", "?"))
        passed = r.get("verify_passed", 0)
        failed = r.get("verify_failed", 0)
        median = r.get("bench_median_ns", "-")
        if winner_name is not None and name == winner_name:
            verdict = "WINNER *"
        elif (failed or 0) > 0:
            verdict = "FAILED"
        else:
            verdict = "slower"
        lines.append(f"{name:<24}{passed}-{failed:<12}{median!s:>12}  {verdict}")
    return "\n".join(lines)
