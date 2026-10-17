"""Fingerprint cache for verified kernel translations.

A kernel is identified by a sha256 fingerprint over its normalized source
(comments stripped, whitespace collapsed) plus its dependency closure, so a
verified translation can be reused across runs instead of re-translating,
re-verifying and re-benchmarking an identical kernel.

Records live as one JSON file per fingerprint under the cache dir
(``~/.cache/kernel-port`` by default). Writes are atomic (tmp file +
``os.replace``). ``lookup`` is defensive: missing files, corrupt JSON and
records written by a different ``CACHE_VERSION`` all yield ``None``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

CACHE_VERSION = 1

# Best-effort comment stripping: `// ...` to end of line, and `/* ... */`.
# Not string-literal aware (a `//` inside a string would be stripped); fine
# for kernel sources, which are pure computation.
_COMMENT_RE = re.compile(r"//.*?$|/\*.*?\*/", re.MULTILINE | re.DOTALL)
_WS_RE = re.compile(r"\s+")


def normalize_source(source: str) -> str:
    """Strip comments and collapse whitespace for fingerprinting.

    Best-effort: removes ``//`` and ``/* */`` comments, collapses every
    whitespace run to a single space, strips leading/trailing whitespace.
    Two sources that differ only in comments/formatting normalize equal.
    """
    no_comments = _COMMENT_RE.sub("", source or "")
    return _WS_RE.sub(" ", no_comments).strip()


def fingerprint(extraction: dict) -> str:
    """sha256 hex fingerprint of an extraction dict.

    The extraction dict has the shape of ``extractor.to_json()``; fields
    are read defensively with ``.get()``. The hashed payload is::

        CACHE_VERSION | signature | normalized kernel source |
        normalized closure sources (sorted by qualname, joined with "|")
    """
    signature = extraction.get("signature") or ""
    kernel_src = normalize_source(extraction.get("source") or "")
    closure = extraction.get("closure") or []
    entries = [c for c in closure if isinstance(c, dict)]
    entries.sort(key=lambda c: c.get("qualname") or "")
    closure_part = "|".join(normalize_source(c.get("source") or "") for c in entries)
    payload = f"{CACHE_VERSION}|{signature}|{kernel_src}|{closure_part}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def default_cache_dir() -> Path:
    """Default cache location: ``~/.cache/kernel-port``."""
    return Path.home() / ".cache" / "kernel-port"


def _record_path(cache_dir: Path | str, fp: str) -> Path:
    return Path(cache_dir) / f"{fp}.json"


def lookup(cache_dir: Path | str, fp: str) -> dict | None:
    """Return the cached record for ``fp``, or ``None``.

    ``None`` is returned when the record file is missing, the JSON is
    corrupt, the decoded value is not a dict, or it was written by a
    different ``CACHE_VERSION``.
    """
    try:
        data = json.loads(_record_path(cache_dir, fp).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("cache_version") != CACHE_VERSION:
        return None
    return data


def store(cache_dir: Path | str, fp: str, record: dict) -> Path:
    """Atomically store ``record`` under fingerprint ``fp``; return the path.

    Expected record keys (all optional except where noted)::

        fingerprint, method, signature, translated_body,
        verify: {passed, failed, skipped},
        bench: {java_per_call_ns, jni_per_call_ns, speedup},
        created: iso8601

    ``fingerprint`` is set to ``fp`` and ``cache_version`` to
    ``CACHE_VERSION`` by this function; ``created`` defaults to now (UTC).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload["fingerprint"] = fp
    payload["cache_version"] = CACHE_VERSION
    payload.setdefault("created", datetime.now(timezone.utc).isoformat())
    path = _record_path(cache_dir, fp)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return path
