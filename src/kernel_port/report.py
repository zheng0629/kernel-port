"""Self-contained HTML report for kernel-port pipeline results.

Produces a single HTML5 document with inline CSS and hand-rolled inline
SVG charts -- zero external dependencies, no CDN, no external images,
openable by double-clicking. All user-supplied content is passed through
``html.escape``; the output intentionally contains no URLs at all.

The module's only public entry point is :func:`render_html`, a pure
function of a plain ``dict``. It deliberately does NOT import any other
``kernel_port`` submodule (miner, extractor, harness, jni, cli) so it
stays dependency-light (stdlib only) and cannot participate in import
cycles.

Input schema
------------
``render_html(report)`` accepts a plain dict shaped like this::

    {
        "title": str,                      # e.g. "kernel-port pipeline report"
        "generated_at": str,
        "method": str,                     # qualified name under test
        "status": str,                     # success | verify_failed | rejected
                                           # | cache_hit | awaiting_translation
        "purity": {"pure": bool, "reasons": [str]},
        "candidates": [{"name": str, "kind": "method"|"cluster"|"class",
                        "score": float, "pure": bool}],
        "verify": {"passed": int, "failed": int, "skipped": int,
                   "boundary_passed": int, "boundary_failed": int},
        "bench": {"backend": str, "java_per_call_ns": float, "jni_per_call_ns": float,
                  "speedup": float, "jni_overhead_ns": float | None,
                  "rust_overhead_ns": float | None,
                  "per_size": [{"size": int, "speedup": float}]},
        "winner": str | None,
        "verdict": str,                    # free text, e.g. "worth porting: 1.88x faster"
    }

Every lookup uses ``dict.get()`` defensively: missing sections render as
``"n/a"``, missing list fields render as empty tables/charts, and unknown
status values get a neutral gray badge. The report never crashes on a
partial dict.
"""

import html

__all__ = ["render_html"]

_NA = "n/a"


def _e(value) -> str:
    """Escape arbitrary user content for safe HTML embedding."""
    if value is None:
        return _NA
    return html.escape(str(value), quote=True)


def _num(value, fmt="{:.2f}"):
    """Format a number or return n/a when it is missing/not numeric."""
    if value is None or isinstance(value, bool):
        return _NA
    try:
        return fmt.format(float(value))
    except (TypeError, ValueError):
        return _NA


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


_STATUS_STYLES = {
    "success": ("green", "success"),
    "verify_failed": ("red", "verify failed"),
    "rejected": ("red", "rejected"),
    "cache_hit": ("blue", "cache hit"),
    "awaiting_translation": ("gray", "awaiting translation"),
}


def _status_badge(status) -> str:
    css, label = _STATUS_STYLES.get(str(status).lower() if status else "", ("gray", _NA))
    # Re-escape: label comes from our own table except the unknown-status case,
    # which echoes the raw status text.
    if label == _NA and status is not None:
        label = str(status)
    return '<span class="badge badge-{}">{}</span>'.format(css, _e(label))


def _purity_panel(purity) -> str:
    purity = purity if isinstance(purity, dict) else {}
    pure = purity.get("pure")
    reasons = purity.get("reasons") or []
    if pure is True:
        cls, head = "pure", "PURE"
    elif pure is False:
        cls, head = "impure", "NOT PURE"
    else:
        cls, head = "unknown", "UNKNOWN"
    if isinstance(reasons, (list, tuple)) and reasons:
        items = "".join("<li>{}</li>".format(_e(r)) for r in reasons)
        reasons_html = "<ul class=\"reasons\">{}</ul>".format(items)
    else:
        reasons_html = "<p class=\"muted\">{}</p>".format(_NA)
    return (
        '<section class="panel purity-{}">'
        "<h2>Purity verdict</h2>"
        '<div class="purity-head">{}</div>'
        "{}"
        "</section>"
    ).format(cls, head, reasons_html)


def _candidates_table(candidates) -> str:
    if not isinstance(candidates, (list, tuple)) or not candidates:
        rows = '<tr><td colspan="4" class="muted">{}</td></tr>'.format(_NA)
    else:
        rows = ""
        for c in candidates:
            c = c if isinstance(c, dict) else {}
            pure = c.get("pure")
            pure_txt = "yes" if pure is True else ("no" if pure is False else _NA)
            rows += (
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                    _e(c.get("name", _NA)),
                    _e(c.get("kind", _NA)),
                    _num(c.get("score")),
                    pure_txt,
                )
            )
    return (
        '<section class="panel"><h2>Candidates</h2>'
        '<table class="tbl"><thead><tr>'
        "<th>Name</th><th>Kind</th><th>Score</th><th>Pure</th>"
        "</tr></thead><tbody>{}</tbody></table></section>"
    ).format(rows)


def _td(content, cls=""):
    cls_attr = ' class="{}"'.format(cls) if cls else ""
    return "<td{}>{}</td>".format(cls_attr, content)


def _verify_panel(verify) -> str:
    verify = verify if isinstance(verify, dict) else {}

    def positive(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0

    failed = verify.get("failed")
    bfailed = verify.get("boundary_failed")
    cells = "".join(
        [
            _td(_num(verify.get("passed"), "{:.0f}")),
            _td(_num(failed, "{:.0f}"), "fail" if positive(failed) else ""),
            _td(_num(verify.get("skipped"), "{:.0f}")),
            _td(_num(verify.get("boundary_passed"), "{:.0f}")),
            _td(_num(bfailed, "{:.0f}"), "fail" if positive(bfailed) else ""),
        ]
    )
    return (
        '<section class="panel"><h2>Verify</h2>'
        '<table class="tbl"><thead><tr>'
        "<th>Random passed</th><th>Random failed</th><th>Skipped</th>"
        "<th>Boundary passed</th><th>Boundary failed</th>"
        "</tr></thead><tbody><tr>{}</tr></tbody></table></section>"
    ).format(cells)


# ---------------------------------------------------------------------------
# hand-rolled SVG charts
# ---------------------------------------------------------------------------


def _bench_svg(java_ns, jni_ns) -> str:
    """Two horizontal bars comparing Java vs JNI ns/call, linear scale."""
    try:
        j, n = float(java_ns), float(jni_ns)
    except (TypeError, ValueError):
        return ""
    if j < 0 or n < 0:
        return ""
    top = max(j, n)
    if top <= 0:
        return ""
    w, label_w, gap = 560, 130, 8
    bar_h, row_h = 34, 56
    plot_w = w - label_w - 140
    rows = [("Java", j, "#3b82f6"), ("JNI", n, "#f59e0b")]
    parts = [
        '<svg width="{0}" height="{1}" viewBox="0 0 {0} {1}" role="img" '
        'aria-label="Java vs JNI ns per call">'.format(w, row_h * 2 + 34)
    ]
    for i, (name, val, color) in enumerate(rows):
        y = 10 + i * row_h
        bw = max(2.0, val / top * plot_w)
        parts.append(
            '<text x="0" y="{0}" class="svg-label">{1}</text>'.format(y + 23, name)
        )
        parts.append(
            '<rect x="{0}" y="{1}" width="{2:.1f}" height="{3}" rx="4" '
            'fill="{4}"/>'.format(label_w, y, bw, bar_h, color)
        )
        parts.append(
            '<text x="{0:.1f}" y="{1}" class="svg-val">{2} ns</text>'.format(
                label_w + bw + gap, y + 23, _num(val, "{:,.1f}")
            )
        )
    parts.append("</svg>")
    return "".join(parts)


def _speedup_svg(per_size) -> str:
    """Vertical bars of speedup vs input size, with a 1.0x break-even line."""
    points = []
    if isinstance(per_size, (list, tuple)):
        for p in per_size:
            if not isinstance(p, dict):
                continue
            try:
                points.append((int(p["size"]), float(p["speedup"])))
            except (TypeError, ValueError, KeyError):
                continue
    if not points:
        return ""
    w, h = 560, 240
    pad_l, pad_b, pad_t = 48, 44, 18
    plot_w, plot_h = w - pad_l - 24, h - pad_t - pad_b
    vmax = max(max(s for _, s in points), 1.0) * 1.15
    n = len(points)
    slot = plot_w / n
    bar_w = min(slot * 0.55, 64)
    base_y = pad_t + plot_h
    parts = [
        '<svg width="{0}" height="{1}" viewBox="0 0 {0} {1}" role="img" '
        'aria-label="speedup vs input size">'.format(w, h)
    ]
    # break-even line at 1.0x
    y1 = base_y - (1.0 / vmax) * plot_h
    parts.append(
        '<line x1="{0}" y1="{1:.1f}" x2="{2}" y2="{1:.1f}" '
        'class="svg-grid" stroke-dasharray="5,4"/>'.format(pad_l, y1, w - 24)
    )
    parts.append(
        '<text x="{0}" y="{1:.1f}" class="svg-val">1.0x</text>'.format(w - 22, y1 - 5)
    )
    for i, (size, speedup) in enumerate(points):
        x = pad_l + i * slot + (slot - bar_w) / 2
        bh = max(2.0, speedup / vmax * plot_h)
        from .harness import verdict_for_speedup as _vfs
        _v = _vfs(speedup)
        color = {"faster": "#10b981", "slower": "#ef4444"}.get(_v, "#3b82f6")
        parts.append(
            '<rect x="{0:.1f}" y="{1:.1f}" width="{2:.1f}" height="{3:.1f}" '
            'rx="3" fill="{4}"/>'.format(x, base_y - bh, bar_w, bh, color)
        )
        parts.append(
            '<text x="{0:.1f}" y="{1:.1f}" class="svg-val" '
            'text-anchor="middle">{2}x</text>'.format(
                x + bar_w / 2, base_y - bh - 6, _num(speedup)
            )
        )
        parts.append(
            '<text x="{0:.1f}" y="{1}" class="svg-label" '
            'text-anchor="middle">n={2}</text>'.format(
                x + bar_w / 2, base_y + 20, _e(size)
            )
        )
    parts.append("</svg>")
    return "".join(parts)


def _bench_panel(bench) -> str:
    bench = bench if isinstance(bench, dict) else {}
    java_ns = bench.get("java_per_call_ns")
    jni_ns = bench.get("jni_per_call_ns")
    speedup = bench.get("speedup")
    overhead = bench.get("jni_overhead_ns")
    if overhead is None:
        overhead = bench.get("rust_overhead_ns")
    native_short = {"jni": "JNI", "ffm": "FFM", "rust": "Rust"}.get(
        bench.get("backend"), "JNI"
    )
    svg = _bench_svg(java_ns, jni_ns)
    sp_svg = _speedup_svg(bench.get("per_size"))
    stats = (
        "<ul class=\"stats\">"
        "<li>Java: <b>{}</b> ns/call</li>"
        "<li>JNI: <b>{}</b> ns/call</li>"
        "<li>Speedup: <b>{}x</b></li>"
        "<li>{} overhead: <b>{}</b> ns</li>"
        "</ul>"
    ).format(
        _num(java_ns, "{:,.1f}"),
        _num(jni_ns, "{:,.1f}"),
        _num(speedup),
        native_short,
        _num(overhead, "{:,.1f}"),
    )
    sp_section = (
        '<h3>Speedup vs input size</h3>{}'.format(sp_svg) if sp_svg else ""
    )
    chart = svg if svg else '<p class="muted">no benchmark data</p>'
    return (
        '<section class="panel"><h2>Benchmark</h2>'
        '<div class="chart">{}</div>'
        "{}{}{}"
        "</section>"
    ).format(chart, stats, sp_section, "")


def _verdict_banner(report) -> str:
    verdict = report.get("verdict") or ""
    text = str(verdict)
    low = text.lower()
    speedup = report.get("bench", {}).get("speedup") if isinstance(
        report.get("bench"), dict
    ) else None
    try:
        sp = float(speedup) if speedup is not None else None
    except (TypeError, ValueError):
        sp = None
    if "inconclusive" in low:
        cls, prefix = "verdict-warn", "Inconclusive"
    elif "not worth porting" in low:
        cls, prefix = "verdict-bad", "Not worth porting"
    elif "worth porting" in low or (sp is not None and sp > 1.0):
        cls, prefix = "verdict-good", "Worth porting"
    elif sp is not None and sp <= 1.0:
        cls, prefix = "verdict-bad", "Not worth porting"
    else:
        cls, prefix = "verdict-bad", "Not worth porting"
    body = _e(text) if text else _NA
    winner = report.get("winner")
    if winner:
        body = "winner: <b>{}</b> &mdash; {}".format(_e(winner), body)
    return (
        '<section class="verdict {}"><span class="verdict-tag">{}</span>'
        "<p>{}</p></section>"
    ).format(cls, prefix, body)


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------

_CSS = """
:root{--bg:#0f172a;--card:#1e293b;--ink:#e2e8f0;--muted:#94a3b8;
--green:#10b981;--red:#ef4444;--blue:#3b82f6;--yellow:#f59e0b;--gray:#64748b}
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
background:var(--bg);color:var(--ink);line-height:1.5}
main{max-width:900px;margin:0 auto;padding:24px 16px 64px}
header.top{border-bottom:2px solid var(--card);padding-bottom:16px;margin-bottom:20px}
header.top h1{margin:0 0 8px;font-size:1.6rem}
.meta{color:var(--muted);font-size:.9rem}
.meta .method{font-family:ui-monospace,Menlo,Consolas,monospace;color:var(--ink)}
.badge{display:inline-block;padding:2px 12px;border-radius:999px;
font-size:.8rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.badge-green{background:rgba(16,185,129,.18);color:#34d399}
.badge-red{background:rgba(239,68,68,.18);color:#f87171}
.badge-blue{background:rgba(59,130,246,.18);color:#60a5fa}
.badge-yellow{background:rgba(245,158,11,.18);color:#fbbf24}
.badge-gray{background:rgba(100,116,139,.25);color:#cbd5e1}
.panel{background:var(--card);border-radius:10px;padding:18px 20px;margin:16px 0}
.panel h2{margin:0 0 12px;font-size:1.05rem;color:var(--muted);
text-transform:uppercase;letter-spacing:.06em}
.panel h3{margin:16px 0 8px;font-size:.95rem;color:var(--muted)}
.purity-green{border-left:4px solid var(--green)}
.purity-red{border-left:4px solid var(--red)}
.purity-unknown{border-left:4px solid var(--gray)}
.purity-head{font-size:1.4rem;font-weight:700;margin-bottom:8px}
.purity-green .purity-head{color:#34d399}
.purity-red .purity-head{color:#f87171}
.purity-unknown .purity-head{color:#cbd5e1}
.reasons{margin:8px 0;padding-left:20px}
.muted{color:var(--muted);font-style:italic}
table.tbl{width:100%;border-collapse:collapse;font-size:.92rem}
table.tbl th,table.tbl td{padding:8px 10px;text-align:left;
border-bottom:1px solid rgba(148,163,184,.15)}
table.tbl th{color:var(--muted);font-weight:600;font-size:.8rem;
text-transform:uppercase;letter-spacing:.04em}
table.tbl td.fail{background:rgba(239,68,68,.22);color:#fca5a5;font-weight:700}
.stats{list-style:none;padding:0;margin:12px 0 0;display:grid;
grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px}
.stats li{background:rgba(15,23,42,.5);border-radius:6px;padding:8px 12px;
font-size:.9rem}
.chart{overflow-x:auto}
svg text{fill:var(--ink);font-size:12px;font-family:inherit}
svg .svg-label{fill:var(--muted)}
svg .svg-grid{stroke:#475569;stroke-width:1}
.verdict{border-radius:10px;padding:16px 20px;margin:16px 0;display:flex;
gap:14px;align-items:flex-start}
.verdict-good{background:rgba(16,185,129,.14);border:1px solid var(--green)}
.verdict-bad{background:rgba(245,158,11,.12);border:1px solid var(--yellow)}
.verdict-warn{background:rgba(96,165,250,.12);border:1px solid var(--blue)}
.verdict-tag{flex:none;font-weight:700;text-transform:uppercase;font-size:.85rem;
letter-spacing:.04em;padding:2px 10px;border-radius:6px}
.verdict-good .verdict-tag{background:var(--green);color:#052e1b}
.verdict-bad .verdict-tag{background:var(--yellow);color:#451a03}
.verdict-warn .verdict-tag{background:var(--blue);color:#eff6ff}
.verdict p{margin:0}
footer{margin-top:28px;padding-top:14px;border-top:1px solid var(--card);
color:var(--muted);font-size:.8rem;text-align:center}
"""


def render_html(report: dict) -> str:
    """Render a kernel-port pipeline result dict as a self-contained HTML page.

    ``report`` follows the schema documented in this module's docstring.
    Every field is optional; missing sections render as ``"n/a"``.
    Returns a complete HTML5 document as a string.
    """
    if not isinstance(report, dict):
        report = {}
    title = report.get("title") or "kernel-port pipeline report"
    head = (
        '<!DOCTYPE html><html lang="en"><head>'
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>{}</title>"
        "<style>{}</style>"
        "</head><body><main>"
    ).format(_e(title), _CSS)
    header = (
        '<header class="top"><h1>{}</h1>'
        '<div class="meta">method: <span class="method">{}</span>'
        " &middot; generated: {} &middot; {}"
        "</div></header>"
    ).format(
        _e(title),
        _e(report.get("method")),
        _e(report.get("generated_at")),
        _status_badge(report.get("status")),
    )
    body = "".join(
        [
            header,
            _purity_panel(report.get("purity")),
            _candidates_table(report.get("candidates")),
            _verify_panel(report.get("verify")),
            _bench_panel(report.get("bench")),
            _verdict_banner(report),
        ]
    )
    foot = (
        "</main><footer>generated by kernel-port v0.3.0</footer>"
        "</body></html>"
    )
    return head + body + foot
