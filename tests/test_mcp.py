"""Smoke tests for the kernel-port MCP server.

Checks the tool registry (exactly the 7 expected tools, correct names)
without launching the stdio loop, plus a light functional check of the
`mine` tool handler.
"""

import asyncio

from kernel_port import mcp_server as M

EXPECTED_TOOLS = ("mine", "extract", "glue", "verify", "bench", "pipeline", "repair")


def _tool_names() -> list[str]:
    tools = asyncio.run(M.server.list_tools())
    return sorted(t.name for t in tools)


def test_registers_exactly_seven_tools():
    assert _tool_names() == sorted(EXPECTED_TOOLS)


def test_tool_names_constant_matches_registry():
    assert tuple(sorted(M.TOOL_NAMES)) == tuple(sorted(EXPECTED_TOOLS))
    assert _tool_names() == sorted(M.TOOL_NAMES)


def test_mine_tool_handler_returns_data():
    from .conftest import FIXTURE_REPO

    result = M.mine(repo=str(FIXTURE_REPO))
    assert result["ok"] is True
    assert result["candidates"], "fixture repo should yield candidates"
    first = result["candidates"][0]
    assert "method" in first and "pure" in first and "score" in first


def test_tool_handlers_return_errors_as_dicts():
    result = M.extract(repo=str("/nonexistent/repo"), method="x.Y.z")
    assert result["ok"] is False
    assert "error" in result


def test_serve_entrypoint_exists():
    assert callable(M.serve)


def test_pipeline_tool_forwards_cache_opts_for_python_lang(monkeypatch, tmp_path):
    """Regression: MCP `pipeline` used to reject candidates_dir/no_cache/cache_dir
    for lang='python' while the CLI supports them. They must be forwarded."""
    from pathlib import Path

    from kernel_port import py_pipeline

    seen: dict = {}

    def fake_run(repo, **kwargs):
        seen.update(kwargs)
        return {"ok": True, "status": "stubbed"}

    monkeypatch.setattr(py_pipeline, "run_python_pipeline", fake_run)
    repo = str(Path(__file__).parent / "fixtures" / "py_kernels")
    result = M.pipeline(
        repo=repo,
        method="cffi_kernels.dot",
        lang="python",
        candidates_dir=str(tmp_path / "cand"),
        no_cache=True,
        cache_dir=str(tmp_path / "cache"),
    )
    assert result["ok"] is True
    assert seen["candidates_dir"] == str(tmp_path / "cand")
    assert seen["no_cache"] is True
    assert seen["cache_dir"] == str(tmp_path / "cache")


def test_pipeline_tool_still_rejects_backend_for_python_lang():
    from pathlib import Path

    repo = str(Path(__file__).parent / "fixtures" / "py_kernels")
    result = M.pipeline(repo=repo, method="cffi_kernels.dot", lang="python", backend="ffm")
    assert result["ok"] is False
    assert "backend" in result["error"]
