"""Tests for ctxfeed.cache_plan — the v0.2 cost-path fix + DeepSeek selection.

Covers:
- fix-cost-path-cache-pollution: cost_delta / files_vs_cap / cost_and_files
  are read-only — they do NOT persist cache keys or record ingest runs, so a
  `ctxfeed cost` query on a fresh clone cannot inflate the cache_hit metric on
  the next `ctxfeed init`.
- feat-deepseek-selectable-fallback: for_repo(model="deepseek") builds a
  DeepSeekClient and recomputes the budget window to 128k.
All tests run in dry-run mode (no API key).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ctxfeed.cache_plan import CachePlan
from ctxfeed.ingest import CacheStore, IngestConfig
from ctxfeed.models.deepseek import DEEPSEEK_CONTEXT_WINDOW, DeepSeekClient
from ctxfeed.models.glm import GLMClient


@pytest.fixture()
def tiny_repo(tmp_path: Path) -> Path:
    """A 4-file repo: stable_prefix (README, pyproject) + body (2 .py)."""
    (tmp_path / "README.md").write_text("# tiny\nA small test repo.\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "tiny"\nversion = "0.1"\n', encoding="utf-8"
    )
    (tmp_path / "auth.py").write_text(
        "def login(user):\n    return user == 'admin'\n", encoding="utf-8"
    )
    (tmp_path / "utils.py").write_text("def helper():\n    return 42\n", encoding="utf-8")
    return tmp_path


def _fresh_plan(root: Path, tmp_path: Path, **kw) -> CachePlan:
    cfg = IngestConfig(dry_run=True, cache_db=str(tmp_path / "cache.db"))
    return CachePlan.for_repo(root, config=cfg, **kw)


# ---------------------------------------------------------------------------
# fix-cost-path-cache-pollution
# ---------------------------------------------------------------------------

def test_cost_delta_does_not_persist_cache(tiny_repo: Path, tmp_path: Path):
    """cost_delta() must not write cache keys or record an ingest run."""
    with _fresh_plan(tiny_repo, tmp_path) as cp:
        cp.cost_delta()  # read-only
        cache = CacheStore(cp.config.cache_db)
        assert cache.cached_hashes() == set()  # no puts
        runs = cache._connect().execute("SELECT count(*) FROM ingest_runs").fetchone()
        assert runs[0] == 0  # no record_run


def test_files_vs_cap_does_not_persist_cache(tiny_repo: Path, tmp_path: Path):
    """files_vs_cap() must not write cache keys or record an ingest run."""
    with _fresh_plan(tiny_repo, tmp_path) as cp:
        cp.files_vs_cap()
        cache = CacheStore(cp.config.cache_db)
        assert cache.cached_hashes() == set()
        runs = cache._connect().execute("SELECT count(*) FROM ingest_runs").fetchone()
        assert runs[0] == 0


def test_cost_and_files_single_pass_no_persist(tiny_repo: Path, tmp_path: Path):
    """cost_and_files() returns both numbers from one non-persisting pass."""
    with _fresh_plan(tiny_repo, tmp_path) as cp:
        delta, fv = cp.cost_and_files()
    # both numbers populated
    assert delta.glm.total_cost >= 0
    assert fv["files_accepted"] >= 3
    assert fv["chatgpt_cap"] == 40
    # nothing persisted
    cache = CacheStore(cp.config.cache_db)
    assert cache.cached_hashes() == set()
    runs = cache._connect().execute("SELECT count(*) FROM ingest_runs").fetchone()
    assert runs[0] == 0


def test_ingest_still_persists(tiny_repo: Path, tmp_path: Path):
    """The genuine ingest path still persists cache + records a run (regression
    guard: the read-only fix must not have broken the persisting path)."""
    with _fresh_plan(tiny_repo, tmp_path) as cp:
        cp.ingest()
        cache = CacheStore(cp.config.cache_db)
        assert len(cache.cached_hashes()) >= 3  # puts happened
        runs = cache._connect().execute("SELECT count(*) FROM ingest_runs").fetchone()
        assert runs[0] == 1  # one ingest run recorded


def test_cost_after_ingest_does_not_inflate_cache_hit(tiny_repo: Path, tmp_path: Path):
    """Regression for the original bug: a `cost` query must not make a later
    `init` report falsely-inflated cache hits. The cost path is read-only, so
    the cache state is owned solely by real ingests."""
    with _fresh_plan(tiny_repo, tmp_path) as cp:
        cp.cost_delta()  # read-only — must not persist
        cp.files_vs_cap()  # read-only — must not persist
        # Now a real ingest — first ingest is a cold start (0% cache hit).
        result = cp.ingest()
        assert result.plan.cache_hit_rate == 0.0  # cold start, cost path didn't pre-fill


# ---------------------------------------------------------------------------
# feat-deepseek-selectable-fallback
# ---------------------------------------------------------------------------

def test_for_repo_default_is_glm(tiny_repo: Path, tmp_path: Path):
    cp = _fresh_plan(tiny_repo, tmp_path)
    assert isinstance(cp.model, GLMClient)
    assert cp.config.window == 1_000_000  # GLM-5.2 1M


def test_for_repo_deepseek_selectable(tiny_repo: Path, tmp_path: Path):
    """for_repo(model='deepseek') builds a DeepSeekClient + 128k window."""
    cp = CachePlan.for_repo(
        tiny_repo,
        config=IngestConfig(dry_run=True, cache_db=str(tmp_path / "ds.db")),
        model="deepseek",
    )
    assert isinstance(cp.model, DeepSeekClient)
    assert cp.config.window == DEEPSEEK_CONTEXT_WINDOW  # 128_000
    assert cp.config.window == 128_000


def test_for_repo_deepseek_dry_run_query(tiny_repo: Path, tmp_path: Path):
    """A DeepSeek-selected plan still answers in dry-run (no DEEPSEEK_API_KEY)."""
    cp = CachePlan.for_repo(
        tiny_repo,
        config=IngestConfig(dry_run=True, cache_db=str(tmp_path / "ds.db")),
        model="deepseek",
    )
    with cp:
        qr = cp.query("where is auth?")
    assert "[dry-run" in qr.answer
    assert qr.files_in_context >= 3


# ---------------------------------------------------------------------------
# feat-gitignore-aware-scan
# ---------------------------------------------------------------------------

def test_scan_repo_respects_gitignore(tmp_path: Path):
    """scan_repo honors .gitignore so generated/ignored files are excluded."""
    (tmp_path / "README.md").write_text("# r\n", encoding="utf-8")
    (tmp_path / "keep.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "generated.py").write_text("y = 2\n", encoding="utf-8")
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "bundle.py").write_text("z = 3\n", encoding="utf-8")
    # .gitignore: drop generated.py and the whole dist/ dir.
    (tmp_path / ".gitignore").write_text("generated.py\ndist/\n", encoding="utf-8")

    from ctxfeed.ingest import scan_repo

    chunks = scan_repo(tmp_path, IngestConfig())
    paths = sorted(c.path for c in chunks)
    assert "keep.py" in paths
    assert "README.md" in paths
    assert "generated.py" not in paths  # gitignored
    assert not any(p.startswith("dist/") for p in paths)  # gitignored dir


def test_scan_repo_ctxfeedignore_overrides_gitignore(tmp_path: Path):
    """An explicit .ctxfeedignore is applied alongside .gitignore."""
    (tmp_path / "keep.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "local.py").write_text("y = 2\n", encoding="utf-8")
    (tmp_path / ".ctxfeedignore").write_text("local.py\n", encoding="utf-8")

    from ctxfeed.ingest import scan_repo

    chunks = scan_repo(tmp_path, IngestConfig())
    paths = sorted(c.path for c in chunks)
    assert "keep.py" in paths
    assert "local.py" not in paths  # ctxfeedignored
