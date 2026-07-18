"""Tests for ctxfeed.shard — ShardPlan, build_shard_plan, ShardPlanBuilder,
centrality scoring, and cache-aware delta behavior.

All tests run in dry-run mode (no GLM API key). They cover the m2
primitive (cache-aware ingest ordering) the MCP server + CLI (polish)
consume.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ctxfeed.ingest import CacheStore, FileChunk, IngestConfig, TokenBudget, scan_repo
from ctxfeed.shard import (
    ShardPlan,
    ShardPlanBuilder,
    build_shard_plan,
    centrality_score,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def repo_with_centrality(tmp_path: Path) -> Path:
    """A repo where `core.py` is referenced by two other files (centrality)."""
    (tmp_path / "README.md").write_text("# repo\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text('[project]\nname="r"\n', encoding="utf-8")
    (tmp_path / "core.py").write_text(
        "def core():\n    return 'core'\n", encoding="utf-8"
    )
    (tmp_path / "a.py").write_text(
        "from core import core\nprint(core())\n", encoding="utf-8"
    )
    (tmp_path / "b.py").write_text(
        "import core\nassert core.core()\n", encoding="utf-8"
    )
    (tmp_path / "leaf.py").write_text("x = 1\n", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def budget() -> TokenBudget:
    return TokenBudget(window=1_000_000, headroom=8_000)


def _chunk(path: str, content: str, layer: str = "body") -> FileChunk:
    return FileChunk(
        path=path, content=content, tokens=max(1, len(content) // 4), hash=path, layer=layer
    )


# ---------------------------------------------------------------------------
# centrality_score
# ---------------------------------------------------------------------------

def test_centrality_counts_cross_references(repo_with_centrality: Path):
    chunks = scan_repo(repo_with_centrality, IngestConfig())
    core = next(c for c in chunks if c.path == "core.py")
    leaf = next(c for c in chunks if c.path == "leaf.py")
    core_score = centrality_score(core, chunks)
    leaf_score = centrality_score(leaf, chunks)
    assert core_score == 2  # referenced by a.py + b.py
    assert leaf_score == 0


def test_centrality_self_reference_excluded(repo_with_centrality: Path):
    chunks = scan_repo(repo_with_centrality, IngestConfig())
    core = next(c for c in chunks if c.path == "core.py")
    # core mentions `core` in its own body but must not count itself
    assert centrality_score(core, chunks) == 2


def test_centrality_short_stem_returns_zero():
    c = _chunk("x.py", "x = 1")  # stem "x" — len < 2
    assert centrality_score(c, [c]) == 0


# ---------------------------------------------------------------------------
# build_shard_plan
# ---------------------------------------------------------------------------

def test_build_shard_plan_stable_prefix_first(repo_with_centrality: Path, budget: TokenBudget):
    chunks = scan_repo(repo_with_centrality, IngestConfig())
    plan = build_shard_plan(chunks, budget)
    assert isinstance(plan, ShardPlan)
    # stable_prefix is non-empty (README + pyproject)
    assert len(plan.stable_prefix) >= 2
    # repo_body is ordered by descending centrality → core.py first
    assert plan.repo_body[0].path == "core.py"


def test_build_shard_plan_budget_fit(repo_with_centrality: Path, budget: TokenBudget):
    chunks = scan_repo(repo_with_centrality, IngestConfig())
    plan = build_shard_plan(chunks, budget)
    assert plan.total_tokens == sum(c.tokens for c in plan.ordered_chunks())
    assert plan.total_tokens <= budget.window - budget.headroom + plan.total_tokens  # sanity
    assert plan.fit_ratio > 0.0
    assert plan.files == len(plan.stable_prefix) + len(plan.repo_body)


def test_build_shard_plan_skips_overflow():
    chunks = [
        _chunk("a.py", "x" * 400, "body"),  # 100 tokens
        _chunk("b.py", "y" * 400, "body"),
    ]
    budget = TokenBudget(window=100, headroom=10)  # 90 available
    plan = build_shard_plan(chunks, budget)
    assert plan.files == 0
    assert len(plan.skipped) == 2


def test_build_shard_plan_cold_start_delta_all_hashes(repo_with_centrality: Path, budget: TokenBudget):
    chunks = scan_repo(repo_with_centrality, IngestConfig())
    plan = build_shard_plan(chunks, budget)  # no cache → cold start
    assert len(plan.delta) == plan.files  # every hash is new
    assert plan.cache_hit_rate == 0.0


def test_build_shard_plan_warm_cache_delta_shrinks(repo_with_centrality: Path, tmp_path: Path):
    chunks = scan_repo(repo_with_centrality, IngestConfig())
    budget = TokenBudget(window=1_000_000, headroom=8_000)
    with CacheStore(str(tmp_path / "cache.db")) as cache:
        for c in chunks:
            cache.put(c.hash, c.path, c.tokens, c.layer)
        plan = build_shard_plan(chunks, budget, cache=cache)
    assert len(plan.delta) == 0  # all hashes cached
    assert plan.cache_hit_rate == 1.0


# ---------------------------------------------------------------------------
# ShardPlan methods
# ---------------------------------------------------------------------------

def test_shard_plan_to_prompt_delegates_to_format(repo_with_centrality: Path, budget: TokenBudget):
    chunks = scan_repo(repo_with_centrality, IngestConfig())
    plan = build_shard_plan(chunks, budget)
    prompt = plan.to_prompt(IngestConfig())
    assert prompt.startswith("<repo_context>")
    assert prompt.endswith("</repo_context>")
    assert '<file path="core.py">' in prompt


def test_shard_plan_summary_and_to_dict(repo_with_centrality: Path, budget: TokenBudget):
    chunks = scan_repo(repo_with_centrality, IngestConfig())
    plan = build_shard_plan(chunks, budget)
    s = plan.summary()
    assert "files=" in s and "stable_prefix=" in s and "cache_hit=" in s
    d = plan.to_dict()
    assert d["files"] == plan.files
    assert d["stable_prefix"] == len(plan.stable_prefix)
    assert d["repo_body"] == len(plan.repo_body)
    assert "stable_prefix_paths" in d and "repo_body_paths" in d


def test_shard_plan_ordered_chunks_stable_first(repo_with_centrality: Path, budget: TokenBudget):
    chunks = scan_repo(repo_with_centrality, IngestConfig())
    plan = build_shard_plan(chunks, budget)
    ordered = plan.ordered_chunks()
    # First N are stable_prefix, rest are repo_body
    n_stable = len(plan.stable_prefix)
    assert [c.layer for c in ordered[:n_stable]] == ["stable_prefix"] * n_stable
    assert all(c.layer == "body" for c in ordered[n_stable:])


# ---------------------------------------------------------------------------
# ShardPlanBuilder
# ---------------------------------------------------------------------------

def test_builder_build_returns_plan(repo_with_centrality: Path, tmp_path: Path):
    cfg = IngestConfig(dry_run=True, cache_db=str(tmp_path / "cache.db"))
    with ShardPlanBuilder(cfg) as builder:
        plan = builder.build(repo_with_centrality)
    assert plan.files >= 5  # README + pyproject + core + a + b + leaf
    assert plan.total_tokens > 0


def test_builder_persist_writes_cache(repo_with_centrality: Path, tmp_path: Path):
    cfg = IngestConfig(dry_run=True, cache_db=str(tmp_path / "cache.db"))
    with ShardPlanBuilder(cfg) as builder:
        plan1 = builder.build(repo_with_centrality)
    # Second build with the same cache → delta shrinks
    with ShardPlanBuilder(cfg) as builder:
        plan2 = builder.build(repo_with_centrality)
    assert plan1.cache_hit_rate == 0.0
    assert plan2.cache_hit_rate == 1.0  # all persisted


def test_builder_no_persist_leaves_cache_cold(repo_with_centrality: Path, tmp_path: Path):
    cfg = IngestConfig(dry_run=True, cache_db=str(tmp_path / "cache.db"))
    with ShardPlanBuilder(cfg) as builder:
        builder.build(repo_with_centrality, persist=False)
        plan2 = builder.build(repo_with_centrality, persist=False)
    assert plan2.cache_hit_rate == 0.0  # nothing persisted


def test_builder_raises_on_missing_root(tmp_path: Path):
    cfg = IngestConfig(dry_run=True, cache_db=str(tmp_path / "cache.db"))
    with ShardPlanBuilder(cfg) as builder:
        with pytest.raises(FileNotFoundError):
            builder.build(tmp_path / "nope")
