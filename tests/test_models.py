"""Tests for ctxfeed.models — v0.2 retry + structured ModelAPIError.

Covers fix-api-transient-errors:
- A terminal 401 raises ModelAPIError (status_code=401, model name, body) — no
  raw httpx traceback escapes.
- A transient 429 is retried with backoff and the subsequent 200 returns the
  content.
- A network-level httpx error is retried once then surfaced as ModelAPIError.
- Dry-run mode is unchanged (returns the deterministic mock, no httpx).
The httpx.Client and time.sleep are faked so no network/sleep happens.
"""

from __future__ import annotations

import pytest

from ctxfeed import models
from ctxfeed.models import (
    BaseChatClient,
    ChatMessage,
    ModelAPIError,
    ModelConfig,
    ModelResponse,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeResp:
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


def _live_client(model: str = "glm-5.2") -> BaseChatClient:
    """A client with a non-empty api_key (so NOT dry-run) for live-call tests."""
    cfg = ModelConfig(api_key="k-test", api_base="https://example.test/v1", model=model)
    return BaseChatClient(cfg)


def _patch_httpx(monkeypatch, responses: list[_FakeResp]) -> _FakeClient:
    fake = _FakeClient(responses)
    # _call does `with httpx.Client(...) as client` — patch the module-level ref.
    monkeypatch.setattr(models.httpx, "Client", lambda **kw: fake)
    # Avoid real sleeping during backoff.
    monkeypatch.setattr(models.time, "sleep", lambda _s: None)
    return fake


# ---------------------------------------------------------------------------
# fix-api-transient-errors
# ---------------------------------------------------------------------------

def test_terminal_401_raises_model_api_error(monkeypatch):
    fake = _patch_httpx(monkeypatch, [_FakeResp(401, text="unauthorized")])
    client = _live_client()
    with pytest.raises(ModelAPIError) as ei:
        client.ingest("prompt")
    err = ei.value
    assert err.status_code == 401
    assert err.model == "glm-5.2"
    assert "unauthorized" in err.body
    # Only one attempt — 401 is terminal (not retried).
    assert len(fake.posts) == 1


def test_retry_on_429_then_success(monkeypatch):
    fake = _patch_httpx(
        monkeypatch,
        [
            _FakeResp(429, text="rate limited"),
            _FakeResp(200, payload=_ok_payload("the-answer")),
        ],
    )
    client = _live_client()
    resp = client.query("prompt", "where is auth?")
    assert isinstance(resp, ModelResponse)
    assert resp.content == "the-answer"
    assert resp.dry_run is False
    # Two attempts: first 429 (retried), then 200.
    assert len(fake.posts) == 2


def test_retry_exhausted_on_repeated_429(monkeypatch):
    fake = _patch_httpx(
        monkeypatch,
        [_FakeResp(429, text="rate limited") for _ in range(3)],
    )
    client = _live_client()
    with pytest.raises(ModelAPIError) as ei:
        client.ingest("prompt")
    err = ei.value
    assert err.status_code == 429
    assert "3 attempts" in str(err)
    assert len(fake.posts) == 3  # MAX_ATTEMPTS


def test_500_is_retried(monkeypatch):
    fake = _patch_httpx(
        monkeypatch,
        [
            _FakeResp(503, text="bad gateway"),
            _FakeResp(200, payload=_ok_payload("ok")),
        ],
    )
    client = _live_client()
    resp = client.query("prompt", "q")
    assert resp.content == "ok"
    assert len(fake.posts) == 2


def test_network_error_retried_then_surfaced(monkeypatch):
    import httpx as _real_httpx

    class _BoomClient(_FakeClient):
        def post(self, url, json=None, headers=None):
            self.posts.append((url, json, headers))
            raise _real_httpx.ConnectError("boom", request=None)

    fake = _BoomClient([])
    monkeypatch.setattr(models.httpx, "Client", lambda **kw: fake)
    monkeypatch.setattr(models.time, "sleep", lambda _s: None)
    client = _live_client()
    with pytest.raises(ModelAPIError) as ei:
        client.ingest("prompt")
    assert "API request failed" in str(ei.value)
    assert len(fake.posts) == 3  # retried up to MAX_ATTEMPTS


# ---------------------------------------------------------------------------
# dry-run unchanged
# ---------------------------------------------------------------------------

def test_dry_run_unchanged_no_httpx(monkeypatch):
    """Dry-run mode (no api_key) must not touch httpx (regression guard)."""
    client = BaseChatClient(ModelConfig(api_key="", model="glm-5.2"))
    # If dry-run ever called httpx, this would explode.
    monkeypatch.setattr(
        models.httpx, "Client", lambda **kw: pytest.fail("dry-run must not use httpx")
    )
    resp = client.query("prompt", "where is auth?")
    assert resp.dry_run is True
    assert "[dry-run" in resp.content


def test_model_api_error_attributes():
    err = ModelAPIError("boom", status_code=502, model="deepseek-chat", body="upstream")
    assert err.status_code == 502
    assert err.model == "deepseek-chat"
    assert err.body == "upstream"
