"""Incremental parse cache for Java sources.

``mine`` (and friends) re-parse every ``.java`` file on every run, which is
painfully slow on large repos. This module adds a persistent per-file cache so
repeated runs only re-parse files that actually changed.

Layout
------
One JSON file per source file, stored under::

    <base>/parse/v1/<sha256(abs_path)>.json

``<base>`` is ``~/.cache/kernel-port`` by default (``$XDG_CACHE_HOME`` is
honored) and can be overridden repo-locally via the
``KERNEL_PORT_PARSE_CACHE_DIR`` environment variable (``parse/v1`` is still
appended, so the cache stays version-namespaced).

A cache entry is keyed by **absolute path + mtime_ns + size**.  Each entry
holds the file's extracted model (classes, methods, fields — everything
``JavaRepo`` needs) plus a compact frozen copy of every method *declaration*
subtree.  tree-sitter nodes are not serializable, so on a cache hit the
subtrees are rehydrated as :class:`FrozenNode`, a duck-typed stand-in that
implements the small node API surface the rest of kernel-port actually uses
(``type``, ``children``, ``is_named``, ``start_byte``/``end_byte``,
``start_point``/``end_point``).  Byte offsets stay valid because an entry is
only used when mtime_ns and size match, i.e. the file content is unchanged.

Writes are atomic (temp file + ``os.replace``) and corrupt/unreadable cache
files are treated as a miss — the cache never crashes a run.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from . import java_ast as J
from .java_ast import _parse_class  # same package; mirrors scan_repo's per-file loop

CACHE_VERSION = 2  # v2: class records carry modifiers (v0.12 glue_package)

_ENV_VAR = "KERNEL_PORT_PARSE_CACHE_DIR"

# Stats for the most recent load_repo() call.
_STATS: dict[str, int] = {"files_total": 0, "hits": 0, "misses": 0}


def cache_dir() -> Path:
    """Resolved directory holding the ``parse/vN`` cache entries."""
    override = os.environ.get(_ENV_VAR)
    if override:
        base = Path(override).expanduser()
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = (
            Path(xdg).expanduser()
            if xdg
            else Path.home() / ".cache"
        )
        base = base / "kernel-port"
    return base / "parse" / f"v{CACHE_VERSION}"


def cache_stats() -> dict[str, int]:
    """Stats for the most recent :func:`load_repo` call.

    Returns ``{"files_total": N, "hits": H, "misses": M}`` — a copy, safe to
    keep.
    """
    return dict(_STATS)


def _reset_stats() -> None:
    _STATS["files_total"] = 0
    _STATS["hits"] = 0
    _STATS["misses"] = 0


def clear_cache(root: Path | str | None = None) -> int:
    """Delete cache entries; ``root`` limits deletion to one repo tree.

    Returns the number of entry files removed.  Never raises on I/O errors.
    """
    cdir = cache_dir()
    removed = 0
    try:
        files = list(cdir.glob("*.json"))
    except OSError:
        return 0
    prefix = None
    if root is not None:
        prefix = str(Path(os.path.realpath(root))) + os.sep
    for p in files:
        if prefix is not None:
            try:
                with open(p, encoding="utf-8") as fh:
                    entry = json.load(fh)
                epath = entry.get("path")
            except (OSError, ValueError):
                epath = None
            if not (isinstance(epath, str) and epath.startswith(prefix)):
                continue
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    return removed


# ---------------------------------------------------------------------------
# Frozen tree-sitter nodes
# ---------------------------------------------------------------------------


class FrozenNode:
    """Duck-typed replacement for a tree-sitter ``Node``, rehydrated from cache.

    Implements exactly the node API surface used across kernel-port:
    ``type``, ``is_named``, ``start_byte``, ``end_byte``, ``start_point``,
    ``end_point`` and ``children`` (plus a ``child(i)`` convenience).  Helpers
    like ``java_ast.text()`` / ``named()`` / ``child()`` work unchanged because
    they only touch these attributes.
    """

    __slots__ = (
        "type",
        "is_named",
        "start_byte",
        "end_byte",
        "start_point",
        "end_point",
        "_children",
    )

    def __init__(
        self,
        type: str,
        is_named: bool,
        start_byte: int,
        end_byte: int,
        start_point: tuple[int, int],
        end_point: tuple[int, int],
        children: list["FrozenNode"],
    ) -> None:
        self.type = type
        self.is_named = is_named
        self.start_byte = start_byte
        self.end_byte = end_byte
        self.start_point = start_point
        self.end_point = end_point
        self._children = children

    @property
    def children(self) -> list["FrozenNode"]:
        return self._children

    def child(self, index: int) -> "FrozenNode | None":
        if 0 <= index < len(self._children):
            return self._children[index]
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"FrozenNode(type={self.type!r}, named={self.is_named}, "
            f"bytes={self.start_byte}:{self.end_byte})"
        )


def _freeze(node) -> list:
    """Serialize a tree-sitter node subtree to a compact JSON-able nested list.

    Iterative (explicit stack) so deeply nested files can't hit Python's
    recursion limit.  Layout per node::

        [type, is_named, start_byte, end_byte,
         start_row, start_col, end_row, end_col, [children...]]
    """
    root = [
        node.type,
        1 if node.is_named else 0,
        node.start_byte,
        node.end_byte,
        node.start_point[0],
        node.start_point[1],
        node.end_point[0],
        node.end_point[1],
        [],
    ]
    stack = [(node, root)]
    while stack:
        ts_node, frozen = stack.pop()
        kids = frozen[8]
        for c in ts_node.children:
            fc = [
                c.type,
                1 if c.is_named else 0,
                c.start_byte,
                c.end_byte,
                c.start_point[0],
                c.start_point[1],
                c.end_point[0],
                c.end_point[1],
                [],
            ]
            kids.append(fc)
            stack.append((c, fc))
    return root


def _thaw(data: list) -> FrozenNode:
    """Rebuild a :class:`FrozenNode` tree from :func:`_freeze` output."""
    root = FrozenNode(
        data[0],
        bool(data[1]),
        data[2],
        data[3],
        (data[4], data[5]),
        (data[6], data[7]),
        [],
    )
    stack = [(data, root)]
    while stack:
        fdata, fnode = stack.pop()
        kids = []
        fnode._children = kids
        for kd in fdata[8]:
            fc = FrozenNode(
                kd[0], bool(kd[1]), kd[2], kd[3], (kd[4], kd[5]), (kd[6], kd[7]), []
            )
            kids.append(fc)
            stack.append((kd, fc))
    return root


# ---------------------------------------------------------------------------
# Cache entry read/write
# ---------------------------------------------------------------------------


def _entry_path(cdir: Path, abs_path: str) -> Path:
    digest = hashlib.sha256(abs_path.encode("utf-8")).hexdigest()
    return cdir / f"{digest}.json"


def _load_entry(cdir: Path, abs_path: str) -> dict | None:
    """Read and validate a cache entry; None on any problem (treated as miss)."""
    try:
        st = os.stat(abs_path)
    except OSError:
        return None
    try:
        with open(_entry_path(cdir, abs_path), encoding="utf-8") as fh:
            entry = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(entry, dict):
        return None
    if entry.get("v") != CACHE_VERSION:
        return None
    if entry.get("path") != abs_path:
        return None
    if entry.get("mtime_ns") != st.st_mtime_ns or entry.get("size") != st.st_size:
        return None
    if not isinstance(entry.get("classes"), list):
        return None
    return entry


def _store_entry(cdir: Path, abs_path: str, records: list[dict]) -> None:
    """Write a cache entry atomically; never raises."""
    try:
        st = os.stat(abs_path)
        entry = {
            "v": CACHE_VERSION,
            "path": abs_path,
            "mtime_ns": st.st_mtime_ns,
            "size": st.st_size,
            "classes": records,
        }
        cdir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(cdir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(entry, fh)
            os.replace(tmp, _entry_path(cdir, abs_path))
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    except OSError:
        pass


def _prune_stale(root: Path, cdir: Path) -> None:
    """Drop entries for files under ``root`` that no longer exist (best effort)."""
    prefix = str(Path(os.path.realpath(root))) + os.sep
    try:
        files = list(cdir.glob("*.json"))
    except OSError:
        return
    for p in files:
        try:
            with open(p, encoding="utf-8") as fh:
                entry = json.load(fh)
        except (OSError, ValueError):
            # Unreadable entry can never become valid; drop it.
            try:
                p.unlink()
            except OSError:
                pass
            continue
        epath = entry.get("path") if isinstance(entry, dict) else None
        if (
            isinstance(epath, str)
            and epath.startswith(prefix)
            and not os.path.exists(epath)
        ):
            try:
                p.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Model (de)serialization
# ---------------------------------------------------------------------------


def _method_record(m: J.MethodInfo) -> dict:
    return {
        "qualname": m.qualname,
        "class_qualname": m.class_qualname,
        "name": m.name,
        "jtype": m.jtype,
        "params": [{"name": p.name, "jtype": p.jtype} for p in m.params],
        "modifiers": sorted(m.modifiers),
        "is_static": m.is_static,
        "is_constructor": m.is_constructor,
        "file": m.file,
        "lineno": m.lineno,
        "end_lineno": m.end_lineno,
        "source": m.source,
        # Freeze the declaration subtree; the body is re-derived from it on
        # rebuild via child(node, "block", "constructor_body"), exactly as the
        # parser does.  (method.node is never read downstream, so storing the
        # declaration covers both node and body.)
        "decl": _freeze(m.node) if m.node is not None else None,
    }


def _field_record(f: J.FieldInfo) -> dict:
    return {
        "qualname": f.qualname,
        "name": f.name,
        "jtype": f.jtype,
        "modifiers": sorted(f.modifiers),
        "is_constant": f.is_constant,
        "file": f.file,
        "lineno": f.lineno,
        "source": f.source,
    }


def _records_for_file(frag: J.JavaRepo, rel: str) -> list[dict]:
    """Extract per-file class records (in parse order) from a parsed fragment."""
    out = []
    for cls in frag.classes.values():
        if cls.file != rel:
            continue
        out.append(
            {
                "qualname": cls.qualname,
                "package": cls.package,
                "name": cls.name,
                "kind": cls.kind,
                "modifiers": sorted(cls.modifiers),
                "file": cls.file,
                "methods": [_method_record(m) for m in cls.all_methods],
                "fields": [_field_record(f) for f in cls.fields.values()],
            }
        )
    return out


def _rebuild_method(rec: dict) -> J.MethodInfo:
    decl = _thaw(rec["decl"]) if rec.get("decl") else None
    body = J.child(decl, "block", "constructor_body") if decl is not None else None
    return J.MethodInfo(
        qualname=rec["qualname"],
        class_qualname=rec["class_qualname"],
        name=rec["name"],
        jtype=rec["jtype"],
        params=[J.Param(name=p["name"], jtype=p["jtype"]) for p in rec["params"]],
        modifiers=set(rec["modifiers"]),
        is_static=rec["is_static"],
        is_constructor=rec["is_constructor"],
        file=rec["file"],
        lineno=rec["lineno"],
        end_lineno=rec["end_lineno"],
        source=rec["source"],
        body=body,
        node=decl,
    )


def _merge_records(repo: J.JavaRepo, rel: str, records: list[dict]) -> None:
    """Merge one file's class records into ``repo`` (mirrors scan_repo order)."""
    repo.files.append(rel)
    for rec in records:
        cls = J.ClassInfo(
            qualname=rec["qualname"],
            package=rec["package"],
            name=rec["name"],
            file=rec["file"],
            kind=rec["kind"],
            modifiers=set(rec["modifiers"]),
        )
        repo.classes[cls.qualname] = cls
        for mrec in rec["methods"]:
            m = _rebuild_method(mrec)
            cls.all_methods.append(m)
            cls.methods.setdefault(m.name, m)
            repo.methods.setdefault((m.class_qualname, m.name), []).append(m)
        for frec in rec["fields"]:
            f = J.FieldInfo(
                qualname=frec["qualname"],
                name=frec["name"],
                jtype=frec["jtype"],
                modifiers=set(frec["modifiers"]),
                is_constant=frec["is_constant"],
                file=frec["file"],
                lineno=frec["lineno"],
                source=frec["source"],
            )
            cls.fields[f.name] = f


def _parse_file(path: Path, root: Path, rel: str) -> list[dict]:
    """Parse a single file and return its class records.

    Mirrors the per-file logic of :func:`java_ast.scan_repo`.
    """
    raw = path.read_bytes()
    src = raw.decode("utf-8", errors="replace").encode("utf-8")
    tree = J.parser().parse(src)
    root_node = tree.root_node
    package = ""
    for top in J.named(root_node):
        if top.type == "package_declaration":
            p = J.child(top, "scoped_identifier", "identifier")
            if p is not None:
                package = J.text(p, src).replace(" ", "")
    frag = J.JavaRepo(root=root)
    for top in J.named(root_node):
        if top.type in (
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
            "annotation_type_declaration",
        ):
            _parse_class(top, src, package, rel, frag)
    return _records_for_file(frag, rel)


def _normalize_record_files(classes: list[dict], rel: str) -> None:
    """Rewrite cached per-file paths to the current scan root.

    Cache entries are keyed by absolute path, but the stored ``file`` values
    are relative to whichever root was used when the entry was written. A
    later scan of the same file under a different root (e.g. repo root vs
    ``src/main/java``) would otherwise resolve ``root/file`` to a doubled,
    nonexistent path and crash with FileNotFoundError (v0.9 dogfood). The
    ``rel`` computed for the current scan is always correct, so stamp it onto
    the records before merging.
    """
    for cls in classes:
        cls["file"] = rel
        for m in cls.get("methods", []):
            m["file"] = rel
        for f in cls.get("fields", []):
            f["file"] = rel


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_repo(
    root: Path | str,
    subdir: str | None = None,
    use_cache: bool = True,
    refresh: bool = False,
) -> J.JavaRepo:
    """Parse ``root`` (or ``root/subdir``) into a :class:`JavaRepo`, incrementally.

    With ``use_cache=True`` (default), per-file results are reused from the
    parse cache when the file's mtime/size still match; only new or changed
    files are re-parsed.  ``refresh=True`` ignores existing entries (re-parses
    everything) but still rewrites the cache.  ``use_cache=False`` behaves
    exactly like :func:`java_ast.scan_repo` and never touches the cache.

    After the call, :func:`cache_stats` reports
    ``{"files_total", "hits", "misses"}``.
    """
    root = Path(root)
    base = root / subdir if subdir else root
    if not base.exists():
        raise FileNotFoundError(f"directory not found: {base}")
    _reset_stats()
    repo = J.JavaRepo(root=root)
    cdir = cache_dir() if use_cache else None
    java_files = sorted(base.rglob("*.java"))
    for path in java_files:
        rel = path.relative_to(root).as_posix()
        if J.is_skipped_source(rel):
            continue  # test/example sources are never port candidates
        abs_path = str(Path(os.path.realpath(path)))
        _STATS["files_total"] += 1
        records: list[dict] | None = None
        tmp_repo: J.JavaRepo | None = None
        if cdir is not None and not refresh:
            entry = _load_entry(cdir, abs_path)
            if entry is not None:
                try:
                    # Rebuild into a throwaway repo first so a corrupt entry
                    # falls back to a re-parse instead of poisoning the repo.
                    tmp_repo = J.JavaRepo(root=root)
                    # The cached `file` paths are relative to the root used
                    # when the entry was written; re-stamp them to this scan's
                    # root so root/file never doubles (v0.9 dogfood crash).
                    _normalize_record_files(entry["classes"], rel)
                    _merge_records(tmp_repo, rel, entry["classes"])
                except Exception:
                    tmp_repo = None
                else:
                    _STATS["hits"] += 1
        if tmp_repo is None:
            records = _parse_file(path, root, rel)
            _STATS["misses"] += 1
            if cdir is not None:
                _store_entry(cdir, abs_path, records)
            tmp_repo = J.JavaRepo(root=root)
            _merge_records(tmp_repo, rel, records)
        # Splice the file's fragment into the result, preserving scan order.
        repo.files.extend(tmp_repo.files)
        for qn, cls in tmp_repo.classes.items():
            repo.classes[qn] = cls
        for key, ms in tmp_repo.methods.items():
            repo.methods.setdefault(key, []).extend(ms)
    if cdir is not None:
        _prune_stale(root, cdir)
    return repo
