"""Regression tests for ctxfeed v0.8.0 — the three bug-fix milestones.

Covers:
- fix-mcp-2x-fastmcp-import: the MCP server import is dual-compatible —
  under mcp 1.x it imports FastMCP, and when ``mcp.server.fastmcp`` is
  absent (the mcp 2.x world, where FastMCP was renamed to MCPServer) it
  falls back to ``mcp.server.mcpserver.MCPServer`` and still builds a
  server with all three tools registered. Before the fix, a fresh install
  resolving mcp 2.x crashed on import (ModuleNotFoundError) and CI was red
  on both 3.12 and 3.13.
- fix-for-repo-mutates-shared-config: ``CachePlan.for_repo`` no longer
  mutates a caller-shared ``IngestConfig`` — a deepseek plan used to leave
  DeepSeek's 128k window on the shared object, silently shrinking every
  later glm plan's budget to 12.8% of GLM-5.2's 1M window.
"""

from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path

import pytest

from ctxfeed import mcp_server as ms_module
from ctxfeed.cache_plan import CachePlan
from ctxfeed.ingest import IngestConfig
from ctxfeed.models.deepseek import DEEPSEEK_CONTEXT_WINDOW, DeepSeekConfig
from ctxfeed.models.glm import GLM_CONTEXT_WINDOW, GLMConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_repo(tmp_path: Path) -> Path:
    """A small repo: one body file (a cache_db path is derived from it)."""
    (tmp_path / "app.py").write_text("def main():\n    return 0\n", encoding="utf-8")
    return tmp_path


# ===========================================================================
# fix-mcp-2x-fastmcp-import
# ===========================================================================


class _FakeTool:
    def __init__(self, name: str, fn):
        self.name = name
        self.fn = fn


class _FakeToolManager:
    def __init__(self):
        self._tools: dict[str, _FakeTool] = {}

    def list_tools(self) -> list[_FakeTool]:
        return list(self._tools.values())


class _FakeMCPServer:
    """Minimal stand-in for mcp 2.x's ``mcp.server.mcpserver.MCPServer``.

    Implements only the surfaces ctxfeed's build_server touches (verified
    against the real mcp 2.2.0): __init__(name, instructions=...), the
    @server.tool() decorator, and _tool_manager.list_tools().
    """

    def __init__(self, name: str, instructions: str | None = None, **kwargs):
        self.name = name
        self.instructions = instructions
        self._tool_manager = _FakeToolManager()

    def tool(self, *dargs, **dkwargs):
        def deco(fn):
            self._tool_manager._tools[fn.__name__] = _FakeTool(fn.__name__, fn)
            return fn

        return deco


def test_build_server_falls_back_to_mcp2_mcpserver(tiny_repo: Path, monkeypatch):
    """When ``mcp.server.fastmcp`` is missing (mcp 2.x), build_server must
    import MCPServer from mcp.server.mcpserver and still register the three
    tools. Before the v0.8 fix the 1.x-only import raised ModuleNotFoundError
    — the failure that broke every fresh install resolving mcp 2.x and kept
    CI red on main."""
    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "mcp.server.fastmcp":
            raise ModuleNotFoundError(
                f"No module named {name!r} (simulated mcp 2.x layout)"
            )
        return real_import(name, *args, **kwargs)

    stub = types.ModuleType("mcp.server.mcpserver")
    stub.MCPServer = _FakeMCPServer
    monkeypatch.setitem(sys.modules, "mcp.server.mcpserver", stub)
    monkeypatch.setattr(builtins, "__import__", _import)

    server = ms_module.build_server(str(tiny_repo))

    tool_names = sorted(t.name for t in server._tool_manager.list_tools())
    assert tool_names == ["cost_delta", "list_files", "query_repo"]
    # The mcp 2.x fallback received the same constructor args as the 1.x path.
    assert server.name == "ctxfeed"
    assert "ctxfeed" in (server.instructions or "")


def test_mcp_dependency_bounded_below_3():
    """The dependency spec must carry the evidence-based ``<3`` upper bound
    (mcp 2.0 already broke the 1.x-only import once; the bound stops an
    mcp 3.x from doing it again silently)."""
    from pathlib import Path

    import re

    pyproject = (
        Path(ms_module.__file__).resolve().parents[2] / "pyproject.toml"
    ).read_text(encoding="utf-8")
    match = re.search(r'"mcp>=([^"]+)"', pyproject)
    assert match, "no mcp dependency entry in pyproject.toml"
    assert match.group(1).endswith(",<3"), (
        f"mcp spec must be bounded below 3, got >= {match.group(1)!r}"
    )


# ===========================================================================
# fix-for-repo-mutates-shared-config
# ===========================================================================


def test_for_repo_does_not_mutate_shared_config(tiny_repo: Path):
    """A caller-shared IngestConfig must survive for_repo untouched. Before
    the v0.8 fix, the deepseek branch wrote ``cfg.window = 128_000`` in place,
    so a subsequent glm plan with the SAME config was built against DeepSeek's
    128k window instead of GLM-5.2's 1M — silently moving files into
    ``skipped`` on large repos."""
    shared = IngestConfig(cache_db=str(tiny_repo / "cache.db"))

    with CachePlan.for_repo(tiny_repo, model="deepseek", config=shared) as ds_cp:
        assert ds_cp.config.window == DEEPSEEK_CONTEXT_WINDOW
    # The shared object is untouched by the deepseek plan.
    assert shared.window == GLM_CONTEXT_WINDOW

    # And a glm plan from the same shared config gets the full GLM window.
    with CachePlan.for_repo(tiny_repo, model="glm", config=shared) as glm_cp:
        assert glm_cp.config.window == GLM_CONTEXT_WINDOW


def test_for_repo_mirrors_api_key_without_mutation(tiny_repo: Path):
    """The v0.2 api_key-mirroring behavior is preserved (deepseek/glm keys
    flow into the effective config) while the caller's object stays clean."""
    shared = IngestConfig(cache_db=str(tiny_repo / "cache.db"))

    with CachePlan.for_repo(
        tiny_repo,
        model="deepseek",
        deepseek=DeepSeekConfig(api_key="ds-key"),
        config=shared,
    ) as ds_cp:
        assert ds_cp.config.api_key == "ds-key"
    assert shared.api_key == ""  # untouched

    with CachePlan.for_repo(
        tiny_repo, model="glm", glm=GLMConfig(api_key="glm-key"), config=shared
    ) as glm_cp:
        assert glm_cp.config.api_key == "glm-key"
    assert shared.api_key == ""  # untouched
