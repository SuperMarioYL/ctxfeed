"""Regression tests for ctxfeed v0.10.0.

Covers:
- m1_cache_store_degrade: an unusable (corrupt / non-file) cache DB degrades to
  a rebuilt cold cache instead of raising ``sqlite3.DatabaseError`` through
  every CLI / MCP / library plan path.
- m2_engine_cache_anchor: a reused ``IngestEngine`` / ``ShardPlanBuilder``
  re-anchors its cache per repo-rooted call and never mutates a caller-shared
  ``IngestConfig`` (the v0.3 cwd-cache fix + v0.8 shared-config invariant,
  applied to the lower-level engines).
- m3_prompt_envelope_budget: the per-file prompt wrapper (``<file path=...>`` +
  ``</file>``) is charged to the token budget, so a plan reported as fitting
  does not overflow the model window once the wrapper is rendered.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ctxfeed.cache_plan import CachePlan
from ctxfeed.ingest import (
    CacheStore,
    IngestConfig,
    IngestEngine,
    TokenBudget,
    build_ingest_order,
    count_tokens,
    format_ingest_prompt,
    scan_repo,
)
from ctxfeed.shard import ShardPlanBuilder, build_shard_plan

ENV_KEYS = ("ZHIPU_API_KEY", "GLM_API_KEY", "DEEPSEEK_API_KEY", "CTXFEED_MODEL")


def _write_repo(root: Path) -> Path:
    """Two files whose content is identical across repos (same hashes)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text(
        "# shared\nIdentical content across repos.\n", encoding="utf-8"
    )
    (root / "app.py").write_text("def main():\n    return 0\n", encoding="utf-8")
    return root


@pytest.fixture()
def clean_env(monkeypatch):
    """Force the deterministic dry-run path (no stray provider keys)."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


# ===========================================================================
# m1_cache_store_degrade
# ===========================================================================


def test_corrupt_cache_degrades_to_cold_start(tmp_path: Path, clean_env):
    """A garbage cache.db is quarantined and rebuilt; the plan is a cold start.

    Before the fix, sqlite3.connect + CREATE TABLE raised DatabaseError
    ("file is not a database") out of CacheStore.delta -> build_shard_plan, so
    every plan command for such a repo crashed.
    """
    repo = _write_repo(tmp_path / "repo")
    db = repo / ".ctxfeed" / "cache.db"
    db.parent.mkdir(parents=True)
    db.write_bytes(b"this is not a sqlite database\n" * 4)

    with CachePlan.for_repo(repo) as cp:
        result = cp.plan()

    assert result.plan.cache_hit_rate == 0.0
    # Healed on disk (not the in-memory fallback): the bad file moved aside and
    # a fresh, usable DB now serves the next call.
    quarantined = sorted(db.parent.glob("cache.db.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes().startswith(b"this is not a sqlite")
    with CacheStore(str(db)) as store:
        assert store.delta({"deadbeef"}) == {"deadbeef"}


def test_unusable_cache_path_falls_back_to_memory(tmp_path: Path, clean_env):
    """A cache path that cannot be opened (a directory) degrades to memory."""
    repo = _write_repo(tmp_path / "repo")
    (repo / ".ctxfeed" / "cache.db").mkdir(parents=True)

    with CachePlan.for_repo(repo) as cp:
        result = cp.plan()
        assert cp.cache.degraded is True
        assert cp.cache.degraded_reason

    assert result.plan.cache_hit_rate == 0.0


def test_cli_cost_survives_corrupt_cache(tmp_path: Path, clean_env):
    """The reproduced crash path: `ctxfeed cost` must not raise a traceback."""
    from typer.testing import CliRunner

    from ctxfeed.cli import get_app

    repo = _write_repo(tmp_path / "repo")
    db = repo / ".ctxfeed" / "cache.db"
    db.parent.mkdir(parents=True)
    db.write_bytes(b"garbage")

    result = CliRunner().invoke(get_app(), ["cost", "--repo", str(repo)])

    assert result.exit_code == 0, result.stdout
    assert "cost-delta" in result.stdout


# ===========================================================================
# m2_engine_cache_anchor
# ===========================================================================


def test_engine_two_repos_same_engine_no_cache_collision(tmp_path: Path):
    """One engine over two repos anchors to each in turn (no false hits).

    Before the fix the first absolute path was written back into the config, so
    repo B reused repo A's cache and reported cache_hit_rate 1.0 for files it
    had never ingested.
    """
    repo_a = _write_repo(tmp_path / "repoA")
    repo_b = _write_repo(tmp_path / "repoB")
    shared = IngestConfig(dry_run=True)  # default relative cache_db

    engine = IngestEngine(shared)
    with engine:
        engine.ingest_repo(repo_a)
        assert Path(engine.cache.db_path) == (
            repo_a.resolve() / ".ctxfeed" / "cache.db"
        )
        result_b = engine.ingest_repo(repo_b)

    assert Path(engine.cache.db_path) == (
        repo_b.resolve() / ".ctxfeed" / "cache.db"
    )
    assert (repo_b / ".ctxfeed" / "cache.db").exists()
    assert result_b.cache_hit_rate == 0.0
    assert shared.cache_db == ".ctxfeed/cache.db"  # caller's object untouched


def test_builder_two_repos_same_builder_no_cache_collision(tmp_path: Path):
    """ShardPlanBuilder has the same per-call anchoring contract."""
    repo_a = _write_repo(tmp_path / "repoA")
    repo_b = _write_repo(tmp_path / "repoB")
    shared = IngestConfig()

    builder = ShardPlanBuilder(shared)
    with builder:
        builder.build(repo_a)
        plan_b = builder.build(repo_b)

    assert Path(builder.cache.db_path) == (
        repo_b.resolve() / ".ctxfeed" / "cache.db"
    )
    assert (repo_b / ".ctxfeed" / "cache.db").exists()
    assert plan_b.cache_hit_rate == 0.0
    assert shared.cache_db == ".ctxfeed/cache.db"


def test_shared_config_not_mutated_by_engine_and_builder(tmp_path: Path):
    """Neither engine writes the resolved path into a caller-shared config."""
    repo = _write_repo(tmp_path / "repo")
    shared = IngestConfig(dry_run=True)

    with IngestEngine(shared) as engine:
        engine.ingest_repo(repo)
    assert shared.cache_db == ".ctxfeed/cache.db"

    with ShardPlanBuilder(shared) as builder:
        builder.build(repo)
    assert shared.cache_db == ".ctxfeed/cache.db"


# ===========================================================================
# m3_prompt_envelope_budget
# ===========================================================================


@pytest.fixture()
def envelope_repo(tmp_path: Path) -> Path:
    """40 small distinct files: enough that the wrapper dominates a tight budget."""
    repo = tmp_path / "repo"
    repo.mkdir()
    for i in range(40):
        (repo / f"mod{i:02d}.py").write_text(
            f"VALUE_{i} = {i}\ndef helper_{i}():\n    return VALUE_{i}\n",
            encoding="utf-8",
        )
    return repo


def test_plan_prompt_fits_reserved_window(envelope_repo: Path):
    """A plan whose CONTENT fits exactly must still leave room for wrappers.

    Before the fix every file was packed (content == window - headroom), yet the
    rendered prompt was larger than the available budget once the per-file
    ``<file path=...>`` wrapper was added.
    """
    cfg = IngestConfig()
    chunks = scan_repo(envelope_repo, cfg)
    content = sum(c.tokens for c in chunks)
    assert content > 0

    budget = TokenBudget(window=content + 8_000, headroom=8_000)
    plan = build_shard_plan(chunks, budget)

    assert plan.files < len(chunks)  # at least one file no longer fits
    assert plan.envelope_tokens > 0
    prompt_tokens = count_tokens(plan.to_prompt(cfg))
    assert prompt_tokens <= content  # == window - headroom


def test_ingest_order_prompt_fits_reserved_window(envelope_repo: Path):
    """The m1 IngestEngine ordering loop reserves the same envelope."""
    cfg = IngestConfig()
    chunks = scan_repo(envelope_repo, cfg)
    content = sum(c.tokens for c in chunks)

    budget = TokenBudget(window=content + 8_000, headroom=8_000)
    ordered, skipped = build_ingest_order(chunks, budget)

    assert skipped, "content-only packing fits every file but the prompt would overflow"
    prompt_tokens = count_tokens(format_ingest_prompt(ordered, cfg))
    assert prompt_tokens <= content
