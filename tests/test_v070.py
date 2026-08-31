"""Regression tests for ctxfeed v0.7.0 — the two bug-fix milestones.

Covers:
- fix-mcp-query-repo-unhandled-model-error: the MCP ``query_repo`` tool now
  catches ``ModelAPIError`` and returns a structured ``[ctxfeed] error: ...``
  string instead of letting a terminal model failure (bad key -> 401, or
  429/5xx exhausting the retry budget) propagate uncaught through the FastMCP
  tool boundary. The v0.2 changelog + the tool docstring promised this; the
  shipped v0.6.0 code did not deliver it.
- fix-init-redundant-double-scan: ``ctxfeed init`` used to call
  ``cp.ingest()`` then ``cp.files_vs_cap()``, each driving a full
  ``scan_repo`` (scan + tokenize + hash + rank) — the second solely to read a
  file count ``result.files`` already held. Now ``init`` derives the
  files-vs-cap dict from the already-built plan, so the repo is scanned once,
  not twice (mirrors the v0.2 single-pass ``cost_and_files()`` fix).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import ctxfeed.cache_plan as cp_module
import ctxfeed.ingest as ingest_module
from ctxfeed import mcp_server as ms_module
from ctxfeed.cache_plan import CachePlan
from ctxfeed.ingest import IngestConfig
from ctxfeed.models import ModelAPIError, ModelConfig, ModelResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_repo(tmp_path: Path) -> Path:
    """A small repo: stable_prefix (README) + body (1 .py)."""
    (tmp_path / "README.md").write_text("# tiny\nA small test repo.\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("def main():\n    return 0\n", encoding="utf-8")
    return tmp_path


class _AlwaysFailModel:
    """Model client whose every call raises a terminal 401 ModelAPIError.

    Simulates a bad/expired GLM key — the #1 onboarding failure for an
    API-backed MCP tool. Used to drive the query_repo error path.
    """

    def __init__(self, model: str = "glm-5.2"):
        self.config = ModelConfig(model=model, dry_run=True)

    def _fail(self, *args, **kwargs) -> ModelResponse:
        raise ModelAPIError(
            f"{self.config.model} API error 401: invalid api key",
            status_code=401,
            model=self.config.model,
            body='{"error":{"message":"invalid api key"}}',
        )

    def chat(self, messages):  # pragma: no cover — not exercised here
        return self._fail()

    def ingest(self, prompt: str) -> ModelResponse:
        return self._fail()

    def query(self, prompt: str, question: str) -> ModelResponse:
        return self._fail()


def _fail_cache_plan(root: Path, cache_db: str) -> CachePlan:
    """A CachePlan backed by a model that always raises a terminal 401."""
    cfg = IngestConfig(dry_run=True, cache_db=cache_db)
    return CachePlan(root, config=cfg, model=_AlwaysFailModel())


def _server_query_repo_fn(repo_root: Path | str):
    """Build the MCP server and return the query_repo tool's callable."""
    server = ms_module.build_server(repo_root)
    tools = {t.name: t for t in server._tool_manager.list_tools()}
    return tools["query_repo"].fn


# ===========================================================================
# fix-mcp-query-repo-unhandled-model-error
# ===========================================================================


def test_query_repo_returns_structured_error_on_model_failure(tiny_repo: Path, monkeypatch):
    """A terminal ModelAPIError (401 bad key) must surface as a structured
    ``[ctxfeed] error: ...`` tool-result string, NOT propagate uncaught out of
    the query_repo tool. Before the v0.7 fix the exception escaped the tool
    boundary and the structured error (status/model/body) was discarded."""
    cache_db = str(tiny_repo / "cache.db")

    def _patched_for_repo(root, model):
        return _fail_cache_plan(root, cache_db)

    monkeypatch.setattr(ms_module, "_for_repo", _patched_for_repo)

    fn = _server_query_repo_fn(str(tiny_repo))
    out = fn(question="where is auth?", repo=str(tiny_repo), model="")

    assert isinstance(out, str)
    assert out.startswith("[ctxfeed] error:")
    # The structured error carries the model name + status + body excerpt
    # (built by the model layer; surfaced verbatim by the tool).
    assert "glm-5.2" in out
    assert "401" in out
    assert "invalid api key" in out


def test_query_repo_error_does_not_raise(tiny_repo: Path, monkeypatch):
    """The v0.7 fix must catch ModelAPIError — the tool returns a string, never
    raises. (A regression here would re-break the v0.2 changelog promise.)"""
    cache_db = str(tiny_repo / "cache.db")

    def _patched_for_repo(root, model):
        return _fail_cache_plan(root, cache_db)

    monkeypatch.setattr(ms_module, "_for_repo", _patched_for_repo)

    fn = _server_query_repo_fn(str(tiny_repo))
    # Must not raise — the whole point of the fix.
    out = fn(question="anything", repo=str(tiny_repo), model="")
    assert "[ctxfeed] error:" in out


def test_query_repo_success_path_unchanged(tiny_repo: Path, monkeypatch):
    """Regression guard: the try/except must not change the success path. A
    dry-run (no key) query still returns the normal context summary + answer,
    not an error string."""
    for k in ("ZHIPU_API_KEY", "GLM_API_KEY", "DEEPSEEK_API_KEY", "CTXFEED_MODEL"):
        monkeypatch.delenv(k, raising=False)

    fn = _server_query_repo_fn(str(tiny_repo))
    out = fn(question="where is auth?", repo=str(tiny_repo), model="")

    assert isinstance(out, str)
    assert out.startswith("[ctxfeed]")
    assert "files in context" in out
    assert "[ctxfeed] error:" not in out


# ===========================================================================
# fix-init-redundant-double-scan
# ===========================================================================


def test_init_scans_repo_once(tiny_repo: Path, monkeypatch):
    """`ctxfeed init` must scan the repo exactly ONCE. Before the v0.7 fix it
    scanned twice: `cp.ingest()` then `cp.files_vs_cap()` (a redundant
    read-only re-scan for a file count the first pass already produced)."""
    calls = {"n": 0}
    orig = ingest_module.scan_repo

    def _counting_scan(root, config=None):
        calls["n"] += 1
        return orig(root, config)

    # cache_plan does `from .ingest import scan_repo` -> its own binding; patch
    # both module refs so the counter sees every call from the init path.
    monkeypatch.setattr(ingest_module, "scan_repo", _counting_scan)
    monkeypatch.setattr(cp_module, "scan_repo", _counting_scan)
    monkeypatch.chdir(tiny_repo)

    from typer.testing import CliRunner

    from ctxfeed.cli import get_app

    runner = CliRunner()
    result = runner.invoke(get_app(), ["init", "--repo", str(tiny_repo)])
    assert result.exit_code == 0, result.stdout
    assert calls["n"] == 1, (
        f"init should scan the repo once, not {calls['n']}x "
        "(the second scan was the redundant files_vs_cap pass the v0.7 fix removed)"
    )


def test_init_files_accepted_unchanged(tiny_repo: Path, monkeypatch):
    """The single-pass fix must not change the files-accepted banner — the
    count still equals the plan's file count (now derived from result.files
    instead of a second scan)."""
    for k in ("ZHIPU_API_KEY", "GLM_API_KEY", "DEEPSEEK_API_KEY", "CTXFEED_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tiny_repo)

    from typer.testing import CliRunner

    from ctxfeed.cli import get_app

    runner = CliRunner()
    result = runner.invoke(get_app(), ["init", "--repo", str(tiny_repo)])
    assert result.exit_code == 0, result.stdout
    # tiny_repo has README.md + app.py => 2 files accepted, UNDER the 40-cap.
    assert "2 files accepted" in result.stdout
    assert "UNDER" in result.stdout  # over_cap banner (2 < 40)
