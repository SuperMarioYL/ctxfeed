"""Regression tests for ctxfeed v0.4.0 — the three bug-fix milestones.

Covers:
- fix-count-tokens-empty-string: ``count_tokens("")`` returns 0 on BOTH the
  tiktoken-present path and the tiktoken-absent (char-fallback) path. The
  fallback used to be ``max(1, len(text)//4)`` which returns 1 for "" — masked
  wherever tiktoken is importable (all local tests) but firing on the air-gap /
  tiktoken-absent path that is ctxfeed's CN target segment.
- fix-cache-hits-chunk-vs-hash: ``record_run`` now receives chunk-level
  cache_hits / cache_misses so the ``ingest_runs`` ledger agrees with the
  user-facing ``ShardPlan.cache_hit_rate``. ``plan.files`` counts CHUNKS while
  ``plan.delta`` is a set of HASHES; ``files - len(delta)`` over-reported hits
  for duplicate-content files sharing a hash.
- fix-glm-response-choices-unguarded: a 200 with a content-filtered / malformed
  body (empty ``choices``, ``null`` content, or a body missing ``choices``)
  raises a structured ``ModelAPIError`` — not a raw KeyError / IndexError and
  not a silent None — mirroring the terminal-error branch v0.2/v0.3 established.
  Applied in BOTH ``_call_glm`` (ingest path) and ``BaseChatClient._call``
  (models path).

Non-API tests run in dry-run mode; the malformed-200 tests fake httpx + sleep
so no network or real sleeping happens. Mirrors the v0.3 test_v030.py style.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ctxfeed import ingest as ingest_module
from ctxfeed import models as models_module
from ctxfeed.cache_plan import CachePlan
from ctxfeed.ingest import CacheStore, IngestConfig, IngestEngine
from ctxfeed.models import BaseChatClient, ModelAPIError, ModelConfig
from ctxfeed.shard import ShardPlanBuilder


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def tiny_repo(tmp_path: Path) -> Path:
    """A small repo: stable_prefix (README) + body (1 .py). Must have at least
    one ingestible file so ``IngestEngine.ingest_repo`` reaches ``_call_glm``."""
    (tmp_path / "README.md").write_text("# tiny\nA small test repo.\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("def main():\n    return 0\n", encoding="utf-8")
    return tmp_path


# ===========================================================================
# Fakes for the malformed-200 httpx paths (ingest + models)
# ===========================================================================

class _FakeResp:
    """A fake httpx.Response with a queued status code + json/text payload."""

    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json payload")
        return self._payload


class _FakeClient:
    """Fake httpx.Client context manager that yields queued responses."""

    def __init__(self, responses: list[_FakeResp]):
        self._responses = list(responses)
        self.posts: list = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json=None, headers=None):
        self.posts.append((url, json, headers))
        if not self._responses:
            raise AssertionError("no queued fake response")
        return self._responses.pop(0)


def _ok_payload(content: str = "answer") -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
    }


def _patch_ingest_httpx(monkeypatch, responses: list[_FakeResp]) -> _FakeClient:
    """Patch httpx.Client + time.sleep on the ingest module (no network/sleep)."""
    fake = _FakeClient(responses)
    monkeypatch.setattr(ingest_module.httpx, "Client", lambda **kw: fake)
    monkeypatch.setattr(ingest_module.time, "sleep", lambda _s: None)
    return fake


def _patch_models_httpx(monkeypatch, responses: list[_FakeResp]) -> _FakeClient:
    """Patch httpx.Client + time.sleep on the models module (no network/sleep)."""
    fake = _FakeClient(responses)
    monkeypatch.setattr(models_module.httpx, "Client", lambda **kw: fake)
    monkeypatch.setattr(models_module.time, "sleep", lambda _s: None)
    return fake


def _live_ingest_config(tmp_path: Path) -> IngestConfig:
    """An IngestConfig with a non-empty api_key (so NOT dry-run) for live calls."""
    return IngestConfig(
        api_key="k-test",
        api_base="https://example.test/v1",
        cache_db=str(tmp_path / "cache.db"),
    )


def _live_models_client(model: str = "glm-5.2") -> BaseChatClient:
    """A BaseChatClient with a non-empty api_key (so NOT dry-run) for live calls."""
    return BaseChatClient(
        ModelConfig(api_key="k-test", api_base="https://example.test/v1", model=model)
    )


# The three malformed / content-filtered 200 bodies the unguarded
# ``data["choices"][0]["message"]["content"]`` access used to mishandle.
_BAD_200_BODIES = [
    pytest.param({"choices": []}, id="empty-choices"),
    pytest.param({"choices": [{"message": {"content": None}}]}, id="null-content"),
    pytest.param({"foo": "bar"}, id="missing-choices"),
]


# ===========================================================================
# fix-count-tokens-empty-string
# ===========================================================================

def test_count_tokens_empty_fallback_path(monkeypatch):
    """v0.4 (fix-count-tokens-empty-string): the tiktoken-ABSENT char-fallback
    must return 0 for "" (the air-gap / CN-target path). The fallback used to
    be ``max(1, len("")//4) == 1``, masked wherever tiktoken is importable (all
    local tests pass on the tiktoken path) but firing on the fallback. Force
    the fallback and assert the empty==0 contract AND the >=1 floor for any
    non-empty text so the fallback can't regress."""
    monkeypatch.setattr(ingest_module, "_TIKTOKEN_AVAILABLE", False)
    monkeypatch.setattr(ingest_module, "_encoder", None)
    assert ingest_module.count_tokens("") == 0
    # The >=1 floor still applies to non-empty input (no short-string regression).
    assert ingest_module.count_tokens("hello world") >= 1
    assert ingest_module.count_tokens("x") >= 1  # single char still floored to 1


def test_count_tokens_empty_tiktoken_path():
    """Empty string is 0 tokens on the tiktoken-present path too (acceptance #1:
    returns 0 on BOTH paths). Guard against the path being silently swapped."""
    # Only assert the contract when tiktoken is actually importable; if it is
    # absent in this env the fallback test above already covers that path.
    if ingest_module._TIKTOKEN_AVAILABLE:
        assert ingest_module.count_tokens("") == 0


# ===========================================================================
# fix-cache-hits-chunk-vs-hash
# ===========================================================================

def _assert_ledger_chunk_level(cache: CacheStore, plan) -> None:
    """Assert the latest ingest_runs row matches the plan's chunk-level
    hit/miss computation (the same one ShardPlan.cache_hit_rate uses)."""
    row = cache._connect().execute(
        "SELECT files, cache_hits, cache_misses FROM ingest_runs "
        "ORDER BY run_id DESC LIMIT 1"
    ).fetchone()
    assert row is not None, "no ingest_run recorded"
    files, hits, misses = row
    expected_hits = sum(1 for c in plan.ordered_chunks() if c.hash not in plan.delta)
    assert hits == expected_hits
    assert misses == plan.files - expected_hits
    assert hits + misses == plan.files
    # Chunk-level rate matches the plan's own cache_hit_rate (the star metric).
    if plan.files:
        assert hits / plan.files == pytest.approx(plan.cache_hit_rate)


def test_shard_builder_dup_content_ledger_chunk_level(tmp_path: Path):
    """v0.4 (fix-cache-hits-chunk-vs-hash): two files with byte-identical
    content share a hash. On a cold start every chunk's hash is in the delta,
    so the chunk-level cache_hits must be 0. The old ledger formula
    ``plan.files - len(plan.delta)`` over-reported hits because ``plan.files``
    counts CHUNKS while ``plan.delta`` counts unique HASHES (3 chunks but only
    2 unique hashes -> buggy hits == 1, correct hits == 0). ShardPlanBuilder
    path (shard.py:432-433)."""
    (tmp_path / "README.md").write_text("# r\n", encoding="utf-8")
    (tmp_path / "dup_a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "dup_b.py").write_text("x = 1\n", encoding="utf-8")  # identical -> same hash
    cfg = IngestConfig(dry_run=True, cache_db=str(tmp_path / "cache.db"))
    with ShardPlanBuilder(cfg) as builder:
        plan = builder.build(tmp_path)
    # The two dup files share a hash; the old formula would over-report here.
    assert plan.files >= 3
    assert len({c.hash for c in plan.ordered_chunks()}) < plan.files  # hashes < chunks
    cache = CacheStore(cfg.cache_db)
    _assert_ledger_chunk_level(cache, plan)
    # Specifically: cold start with duplicate content -> NO over-reported hits.
    row = cache._connect().execute(
        "SELECT cache_hits FROM ingest_runs ORDER BY run_id DESC LIMIT 1"
    ).fetchone()
    assert row[0] == 0


def test_cache_plan_dup_content_ledger_chunk_level(tmp_path: Path):
    """v0.4 (fix-cache-hits-chunk-vs-hash): the CachePlan._build_plan persisting
    path (cache_plan.py:228-229) records the ingest_run with chunk-level
    hit/miss, matching ShardPlan.cache_hit_rate. Same duplicate-content
    scenario; asserts the ledger does not over-report hits."""
    (tmp_path / "README.md").write_text("# r\n", encoding="utf-8")
    (tmp_path / "dup_a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "dup_b.py").write_text("x = 1\n", encoding="utf-8")  # identical -> same hash
    cfg = IngestConfig(dry_run=True, cache_db=str(tmp_path / "cache.db"))
    with CachePlan.for_repo(tmp_path, config=cfg) as cp:
        result = cp.ingest()
        plan = result.plan
    assert len({c.hash for c in plan.ordered_chunks()}) < plan.files  # hashes < chunks
    cache = CacheStore(cp.config.cache_db)
    _assert_ledger_chunk_level(cache, plan)
    # Cold start with duplicate content -> NO over-reported hits.
    row = cache._connect().execute(
        "SELECT cache_hits FROM ingest_runs ORDER BY run_id DESC LIMIT 1"
    ).fetchone()
    assert row[0] == 0


def test_cache_hits_ledger_unique_content_unchanged(tmp_path: Path):
    """Regression guard: the chunk-level fix must not change the ledger for a
    unique-content repo (where hashes == chunks), so existing unique-content
    assertions still hold. Warm-cache re-ingest -> 100% hits at chunk level."""
    (tmp_path / "README.md").write_text("# r\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("def main():\n    return 0\n", encoding="utf-8")
    cfg = IngestConfig(dry_run=True, cache_db=str(tmp_path / "cache.db"))
    # First build: cold start, unique content -> hits 0, misses == files.
    with ShardPlanBuilder(cfg) as builder:
        plan1 = builder.build(tmp_path)
    cache = CacheStore(cfg.cache_db)
    _assert_ledger_chunk_level(cache, plan1)
    assert plan1.cache_hit_rate == 0.0
    # Second build with the same cache -> delta shrinks -> 100% chunk-level hits.
    with ShardPlanBuilder(cfg) as builder:
        plan2 = builder.build(tmp_path)
    _assert_ledger_chunk_level(cache, plan2)
    assert plan2.cache_hit_rate == 1.0


# ===========================================================================
# fix-glm-response-choices-unguarded
# ===========================================================================

@pytest.mark.parametrize("bad_body", _BAD_200_BODIES)
def test_call_glm_malformed_200_raises_model_api_error(
    tiny_repo: Path, tmp_path: Path, monkeypatch, bad_body: dict
):
    """v0.4 (fix-glm-response-choices-unguarded): a 200 with a content-filtered
    / malformed body — empty ``choices`` (was IndexError), ``null`` content
    (was silent None that later crashes downstream slicing), or a body missing
    ``choices`` (was KeyError) — must raise a structured ``ModelAPIError``
    (status_code + model + body), not a raw exception or a silent None.
    IngestEngine / ``_call_glm`` path (ingest.py:568). 200 is not retried."""
    fake = _patch_ingest_httpx(
        monkeypatch, [_FakeResp(200, payload=bad_body, text=json.dumps(bad_body))]
    )
    with IngestEngine(_live_ingest_config(tmp_path)) as engine:
        with pytest.raises(ModelAPIError) as ei:
            engine.ingest_repo(tiny_repo)
    err = ei.value
    assert err.status_code == 200
    assert err.model == "glm-5.2"
    assert "no message content" in str(err)
    assert json.dumps(bad_body) in err.body  # body context carried
    assert len(fake.posts) == 1  # 200 is terminal, not retryable


@pytest.mark.parametrize("bad_body", _BAD_200_BODIES)
def test_base_chat_client_malformed_200_raises_model_api_error(
    monkeypatch, bad_body: dict
):
    """v0.4 (fix-glm-response-choices-unguarded): the same malformed-200 guard
    on the ``BaseChatClient._call`` path (models/__init__.py:318) — the m2/m3
    model-call path that the MCP server + CLI use. Each bad body raises a
    structured ``ModelAPIError``, not a raw exception or a silent None."""
    fake = _patch_models_httpx(
        monkeypatch, [_FakeResp(200, payload=bad_body, text=json.dumps(bad_body))]
    )
    client = _live_models_client()
    with pytest.raises(ModelAPIError) as ei:
        client.ingest("prompt")
    err = ei.value
    assert err.status_code == 200
    assert err.model == "glm-5.2"
    assert "no message content" in str(err)
    assert json.dumps(bad_body) in err.body
    assert len(fake.posts) == 1


def test_call_glm_happy_200_returns_content_unchanged(
    tiny_repo: Path, tmp_path: Path, monkeypatch
):
    """Regression guard (acceptance #3): a happy-path 200 with valid content
    returns the content unchanged on BOTH paths — the guard must not regress
    the normal ingest/query flow."""
    # Ingest path
    fake_ingest = _patch_ingest_httpx(
        monkeypatch, [_FakeResp(200, payload=_ok_payload("the-ack"))]
    )
    with IngestEngine(_live_ingest_config(tmp_path)) as engine:
        result = engine.ingest_repo(tiny_repo)
    assert result.response == "the-ack"
    assert len(fake_ingest.posts) == 1


def test_base_chat_client_happy_200_returns_content_unchanged(monkeypatch):
    """Regression guard (acceptance #3): happy-path 200 on the models path
    returns the content + the usage tokens unchanged."""
    fake = _patch_models_httpx(
        monkeypatch, [_FakeResp(200, payload=_ok_payload("the-answer"))]
    )
    client = _live_models_client()
    resp = client.query("prompt", "where is auth?")
    assert resp.content == "the-answer"
    assert resp.dry_run is False
    assert resp.prompt_tokens == 100
    assert resp.completion_tokens == 10
    assert len(fake.posts) == 1
