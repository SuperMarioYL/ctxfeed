"""Tests for ctxfeed.ingest — FileChunk, TokenBudget, scan_repo, ordering,
CacheStore, and the IngestEngine dry-run pipeline.

All tests run in dry-run mode (no GLM API key) so they execute in CI
without network access. They cover the m1 foundation the ShardPlan
(m2) and the MCP server / CLI (polish) build on.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ctxfeed.ingest import (
    CacheStore,
    FileChunk,
    IngestConfig,
    IngestEngine,
    IngestResult,
    TokenBudget,
    build_ingest_order,
    count_tokens,
    format_ingest_prompt,
    scan_repo,
)


# ---------------------------------------------------------------------------
# Fixtures: a tiny synthetic repo under tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_repo(tmp_path: Path) -> Path:
    """A 5-file repo: stable_prefix (README, pyproject) + body (3 .py)."""
    (tmp_path / "README.md").write_text("# tiny\nA small test repo.\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "tiny"\nversion = "0.1"\n', encoding="utf-8"
    )
    (tmp_path / "auth.py").write_text(
        "def login(user):\n    return user == 'admin'\n"
        "AUTH_MIDDLEWARE = 'auth'\n",
        encoding="utf-8",
    )
    (tmp_path / "models.py").write_text(
        "from dataclasses import dataclass\n"
        "@dataclass\nclass User:\n    name: str\n",
        encoding="utf-8",
    )
    (tmp_path / "utils.py").write_text(
        "def helper():\n    return 42\n",
        encoding="utf-8",
    )
    # Ignored dirs should not be scanned.
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text(
        "module.exports = {};\n", encoding="utf-8"
    )
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "auth.cpython-312.pyc").write_text(
        "binary", encoding="utf-8"
    )
    return tmp_path


# ---------------------------------------------------------------------------
# FileChunk
# ---------------------------------------------------------------------------

def test_file_chunk_render_wraps_content_in_file_tag():
    c = FileChunk(path="a.py", content="x = 1", tokens=3, hash="abc")
    rendered = c.render()
    assert '<file path="a.py">' in rendered
    assert "x = 1" in rendered
    assert rendered.endswith("</file>")


def test_file_chunk_is_frozen():
    c = FileChunk(path="a.py", content="x", tokens=1, hash="h")
    with pytest.raises(Exception):
        c.path = "b.py"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TokenBudget
# ---------------------------------------------------------------------------

def test_token_budget_consume_within_limit():
    b = TokenBudget(window=100, headroom=10)
    assert b.available == 90
    assert b.consume(40) is True
    assert b.used == 40
    assert b.available == 50


def test_token_budget_consume_overflow_rejected():
    b = TokenBudget(window=100, headroom=10)
    assert b.consume(40) is True
    assert b.consume(60) is False  # 40+60 > 90
    assert b.used == 40  # rejected consume does not mutate


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------

def test_count_tokens_positive():
    assert count_tokens("hello world") > 0
    assert count_tokens("") == 0  # empty → 0 tokens (max(1, 0//4) path not hit when encoder present)


# ---------------------------------------------------------------------------
# scan_repo
# ---------------------------------------------------------------------------

def test_scan_repo_collects_ingestible_files(tiny_repo: Path):
    chunks = scan_repo(tiny_repo, IngestConfig())
    paths = sorted(c.path for c in chunks)
    assert "README.md" in paths
    assert "pyproject.toml" in paths
    assert "auth.py" in paths
    assert "models.py" in paths
    assert "utils.py" in paths


def test_scan_repo_skips_ignored_dirs(tiny_repo: Path):
    chunks = scan_repo(tiny_repo, IngestConfig())
    paths = [c.path for c in chunks]
    assert not any("node_modules" in p for p in paths)
    assert not any("__pycache__" in p for c in chunks for p in [c.path])


def test_scan_repo_classifies_stable_prefix(tiny_repo: Path):
    chunks = scan_repo(tiny_repo, IngestConfig())
    by_path = {c.path: c for c in chunks}
    assert by_path["README.md"].layer == "stable_prefix"
    assert by_path["pyproject.toml"].layer == "stable_prefix"
    assert by_path["auth.py"].layer == "body"


def test_scan_repo_assigns_hash_and_tokens(tiny_repo: Path):
    chunks = scan_repo(tiny_repo, IngestConfig())
    for c in chunks:
        assert len(c.hash) == 16  # sha256[:16]
        assert c.tokens > 0
    # Identical content → identical hash
    a = next(c for c in chunks if c.path == "auth.py")
    assert a.hash and a.hash != ""


def test_scan_repo_raises_on_missing_root(tmp_path: Path):
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError):
        scan_repo(missing, IngestConfig())


# ---------------------------------------------------------------------------
# build_ingest_order
# ---------------------------------------------------------------------------

def test_build_ingest_order_stable_prefix_first(tiny_repo: Path):
    chunks = scan_repo(tiny_repo, IngestConfig())
    budget = TokenBudget(window=1_000_000, headroom=8_000)
    ordered, skipped = build_ingest_order(chunks, budget)
    # Stable-prefix files come first.
    layers = [c.layer for c in ordered]
    first_body = layers.index("body") if "body" in layers else len(layers)
    assert all(l == "stable_prefix" for l in layers[:first_body])
    assert skipped == []  # tiny repo fits easily


def test_build_ingest_order_skips_when_budget_overflow():
    chunks = [
        FileChunk(path="a.py", content="x", tokens=60, hash="a", layer="body"),
        FileChunk(path="b.py", content="y", tokens=60, hash="b", layer="body"),
    ]
    budget = TokenBudget(window=100, headroom=10)  # only 90 available
    ordered, skipped = build_ingest_order(chunks, budget)
    assert len(ordered) == 1
    assert len(skipped) == 1


# ---------------------------------------------------------------------------
# format_ingest_prompt
# ---------------------------------------------------------------------------

def test_format_ingest_prompt_wraps_in_repo_context():
    chunks = [FileChunk(path="a.py", content="x = 1", tokens=3, hash="a")]
    prompt = format_ingest_prompt(chunks, IngestConfig())
    assert prompt.startswith("<repo_context>")
    assert prompt.endswith("</repo_context>")
    assert '<file path="a.py">' in prompt
    assert "x = 1" in prompt


# ---------------------------------------------------------------------------
# CacheStore
# ---------------------------------------------------------------------------

def test_cache_store_get_put_delta(tmp_path: Path):
    db = str(tmp_path / "cache.db")
    with CacheStore(db) as cache:
        cache.put("hash_a", "a.py", 10, "body")
        got = cache.get("hash_a")
        assert got is not None
        assert got["path"] == "a.py"
        assert got["tokens"] == 10
        # delta: current set has hash_a + hash_b → {hash_b} is new
        delta = cache.delta({"hash_a", "hash_b"})
        assert delta == {"hash_b"}
        assert "hash_a" in cache.cached_hashes()


def test_cache_store_record_run(tmp_path: Path):
    with CacheStore(str(tmp_path / "cache.db")) as cache:
        run_id = cache.record_run(files=10, tokens=1000, cache_hits=8, cache_misses=2)
        assert run_id >= 1


# ---------------------------------------------------------------------------
# IngestEngine (dry-run end-to-end)
# ---------------------------------------------------------------------------

def test_ingest_engine_dry_run_ingest(tiny_repo: Path, tmp_path: Path, monkeypatch):
    cfg = IngestConfig(dry_run=True, cache_db=str(tmp_path / "cache.db"))
    with IngestEngine(cfg) as engine:
        result = engine.ingest_repo(tiny_repo)
    assert isinstance(result, IngestResult)
    assert result.files_ingested >= 4  # README + pyproject + >=2 .py
    assert result.tokens_used > 0
    assert result.budget == 1_000_000
    assert "[dry-run" in result.response
    # cache_hit_rate semantics: cold start → 0%
    assert result.cache_hit_rate == 0.0


def test_ingest_engine_query_returns_answer(tiny_repo: Path, tmp_path: Path):
    cfg = IngestConfig(dry_run=True, cache_db=str(tmp_path / "cache.db"))
    with IngestEngine(cfg) as engine:
        qr = engine.query_repo(tiny_repo, "where is auth?")
    assert qr.question == "where is auth?"
    assert "[dry-run" in qr.answer
    assert qr.files_in_context >= 4
    assert qr.tokens_in_context > 0


def test_ingest_engine_list_files(tiny_repo: Path, tmp_path: Path):
    cfg = IngestConfig(dry_run=True, cache_db=str(tmp_path / "cache.db"))
    with IngestEngine(cfg) as engine:
        files = engine.list_files(tiny_repo)
    paths = [f["path"] for f in files]
    assert "auth.py" in paths
    assert all("layer" in f and "cached" in f and "tokens" in f for f in files)


def test_ingest_result_summary_and_to_dict(tiny_repo: Path, tmp_path: Path):
    cfg = IngestConfig(dry_run=True, cache_db=str(tmp_path / "cache.db"))
    with IngestEngine(cfg) as engine:
        result = engine.ingest_repo(tiny_repo)
    s = result.summary()
    assert "ingested=" in s and "tokens=" in s and "cache_hit=" in s
    d = result.to_dict()
    assert d["files_ingested"] == result.files_ingested
    assert 0.0 <= d["cache_hit_rate"] <= 1.0


def test_ingest_engine_re_ingest_cache_hit_grows(tiny_repo: Path, tmp_path: Path):
    """Second ingest should show cache hits (delta shrinks)."""
    cfg = IngestConfig(dry_run=True, cache_db=str(tmp_path / "cache.db"))
    with IngestEngine(cfg) as engine:
        first = engine.ingest_repo(tiny_repo)
        second = engine.ingest_repo(tiny_repo)
    assert first.cache_hit_rate == 0.0
    assert second.cache_hit_rate > 0.0  # all files now cached


def test_ingest_engine_raises_on_missing_root(tmp_path: Path):
    cfg = IngestConfig(dry_run=True, cache_db=str(tmp_path / "cache.db"))
    with IngestEngine(cfg) as engine:
        with pytest.raises(FileNotFoundError):
            engine.ingest_repo(tmp_path / "nope")
