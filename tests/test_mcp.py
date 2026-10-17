"""Smoke tests for the kernel-port MCP server.

Checks the tool registry (exactly the 6 expected tools, correct names)
without launching the stdio loop, plus a light functional check of the
`mine` tool handler.
"""

import asyncio

from kernel_port import mcp_server as M

EXPECTED_TOOLS = ("mine", "extract", "glue", "verify", "bench", "pipeline")


def _tool_names() -> list[str]:
    tools = asyncio.run(M.server.list_tools())
    return sorted(t.name for t in tools)


def test_registers_exactly_six_tools():
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
