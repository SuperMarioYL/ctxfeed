"""Tests for fix-cache-polluted-on-failed-model-call (v0.6).

CachePlan._build_plan used to persist cache keys + record an ingest run
BEFORE the model call that ingest()/query() make. A live call that fails
AFTER persist (a bad key surfacing as a terminal 401 ModelAPIError, or a
429/503/504 exhausting the retry budget) still warmed the cache and logged
an ingest_run, so the very next ingest/query computed delta against the
warmed cache and reported cache_hit_rate=100% with no real ingest having
succeeded. The fix gates persistence on a successful model call — mirroring
IngestEngine.ingest_repo's persist-after-call ordering (ingest.py:970-977).

These tests use a fake model whose ingest/query raises ModelAPIError on the
first call, then succeeds, and assert:
  - after the failed call the cache is empty + ingest_runs == 0 (not polluted)
  - the follow-up successful call reports a true cold-start 0% cache hit
    (not the 100% the old persist-before-call bug produced)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ctxfeed.cache_plan import CachePlan
from ctxfeed.ingest import CacheStore, IngestConfig
from ctxfeed.models import ModelAPIError, ModelConfig, ModelResponse


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


class _FakeModel:
    """Controllable model client: raises ModelAPIError on the first call(s),
    then succeeds — simulating a bad/expired key that is later fixed (or a
    transient exhaustion of the retry budget)."""

    def __init__(self, *, fail_times: int = 1, model: str = "fake"):
        self.config = ModelConfig(model=model, dry_run=True)
        self._fails_left = fail_times
        self.calls = 0

    def _maybe_fail_then_ok(self, prompt: str) -> ModelResponse:
        self.calls += 1
        if self._fails_left > 0:
            self._fails_left -= 1
            raise ModelAPIError(
                "401 Unauthorized: invalid api key",
                status_code=401,
                model=self.config.model,
                body='{"error":{"message":"invalid api key"}}',
            )
        return ModelResponse(content="ok", model=self.config.model, dry_run=True)

    def chat(self, messages):  # pragma: no cover — not exercised here
        raise NotImplementedError

    def ingest(self, prompt: str) -> ModelResponse:
        return self._maybe_fail_then_ok(prompt)

    def query(self, prompt: str, question: str) -> ModelResponse:
        return self._maybe_fail_then_ok(prompt)


def _plan_with_model(root: Path, tmp_path: Path, model) -> CachePlan:
    cfg = IngestConfig(dry_run=True, cache_db=str(tmp_path / "cache.db"))
    return CachePlan(root, config=cfg, model=model)


def _runs_count(cache: CacheStore) -> int:
    return cache._connect().execute(
        "SELECT count(*) FROM ingest_runs"
    ).fetchone()[0]


def _last_run(cache: CacheStore) -> tuple[int, int]:
    return cache._connect().execute(
        "SELECT cache_hits, cache_misses FROM ingest_runs "
        "ORDER BY run_id DESC LIMIT 1"
    ).fetchone()


# ---------------------------------------------------------------------------
# ingest() — the primary path the amendment describes (CLI init cache-hit row)
# ---------------------------------------------------------------------------

def test_ingest_failure_leaves_cache_untouched(tiny_repo: Path, tmp_path: Path):
    """A live ingest whose model call raises ModelAPIError must NOT warm the
    cache or record an ingest run; the follow-up successful call then reports
    a true cold-start 0% cache hit (not the polluted 100% the old
    persist-before-call ordering produced)."""
    fake = _FakeModel(fail_times=1)  # first ingest raises a terminal 401
    with _plan_with_model(tiny_repo, tmp_path, fake) as cp:
        with pytest.raises(ModelAPIError):
            cp.ingest()

        # The failed call must NOT have persisted (cache untouched, no run).
        cache = CacheStore(cp.config.cache_db)
        assert cache.cached_hashes() == set()  # no cache.put happened
        assert _runs_count(cache) == 0  # no record_run happened

        # Follow-up successful call: a true cold start, because the failed
        # call left the cache empty. cache_hit_rate is computed at plan time
        # against the (still-empty) cache, so 0% — NOT the 100% the old bug
        # produced (which warmed the cache on the failed call).
        result = cp.ingest()
        assert result.plan.cache_hit_rate == 0.0  # cold start, cache was honest

        # The success now persists: cache warmed + one run recorded.
        assert len(cache.cached_hashes()) >= 3
        assert _runs_count(cache) == 1


def test_ingest_success_still_persists(tiny_repo: Path, tmp_path: Path):
    """Regression guard: the persist-after-call reorder must not have dropped
    persistence on the success path. A successful ingest still warms the cache
    and records exactly one run (mirrors the v0.2 test_ingest_still_persists)."""
    fake = _FakeModel(fail_times=0)  # succeeds on the first call
    with _plan_with_model(tiny_repo, tmp_path, fake) as cp:
        cp.ingest()
        cache = CacheStore(cp.config.cache_db)
        assert len(cache.cached_hashes()) >= 3  # puts happened
        assert _runs_count(cache) == 1  # one run recorded


# ---------------------------------------------------------------------------
# query() — same persist-after-call ordering on the repo-QA path
# ---------------------------------------------------------------------------

def test_query_failure_leaves_cache_untouched(tiny_repo: Path, tmp_path: Path):
    """A live query whose model.query raises ModelAPIError must NOT warm the
    cache or record a run; the follow-up successful query records a cold-start
    run (cache_hits == 0), not the polluted full-hit run the old ordering
    produced."""
    fake = _FakeModel(fail_times=1)
    with _plan_with_model(tiny_repo, tmp_path, fake) as cp:
        with pytest.raises(ModelAPIError):
            cp.query("where is auth?")

        cache = CacheStore(cp.config.cache_db)
        assert cache.cached_hashes() == set()  # no cache.put happened
        assert _runs_count(cache) == 0  # no record_run happened

        # Follow-up success: cold start — the recorded run has cache_hits == 0
        # (all misses), the honest cold-start ledger. The old persist-before-
        # call bug recorded cache_hits == plan.files (a polluted 100% hit).
        cp.query("where is auth?")
        assert len(cache.cached_hashes()) >= 3  # success warmed the cache
        hits, misses = _last_run(cache)
        assert hits == 0  # cold start, no false hits
        assert misses >= 3  # all files were misses


def test_plan_still_persists_inline(tiny_repo: Path, tmp_path: Path):
    """plan() has no model call to gate on, so it keeps persisting inline
    (the v0.6 fix deliberately leaves the plan() contract unchanged)."""
    fake = _FakeModel(fail_times=0)
    with _plan_with_model(tiny_repo, tmp_path, fake) as cp:
        cp.plan()
        cache = CacheStore(cp.config.cache_db)
        assert len(cache.cached_hashes()) >= 3  # persisted inline
        assert _runs_count(cache) == 1
