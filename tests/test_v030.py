"""Regression tests for ctxfeed v0.3.0 — the three bug-fix milestones.

Covers:
- fix-ctxfeed-model-env-shadowed: ``CTXFEED_MODEL`` selects the backing model
  when ``--model`` is omitted. The v0.2 typer default was the truthy string
  ``"glm"``, which short-circuited ``_resolve_model``'s ``model or env`` chain
  so the env var was dead end-to-end (CLI -> run_stdio -> build_server).
- fix-cache-db-cwd-relative: the default ``cache_db`` (``.ctxfeed/cache.db``)
  is anchored to the repo root, not the process CWD, so two repos run from
  the same CWD no longer share one cache (false cache hits on content-equal
  files like a shared README).
- fix-ingest-engine-no-retry: ``IngestEngine._call_glm`` retries transient
  failures (429/5xx) with backoff and raises a structured ``ModelAPIError``
  on a terminal failure (e.g. 401), mirroring ``BaseChatClient._call`` so
  the m1 path matches the m2/m3 path.
Non-API tests run in dry-run mode; the retry tests fake httpx + sleep so no
network or real sleeping happens.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ctxfeed import ingest as ingest_module
from ctxfeed.cli import _resolve_model
from ctxfeed.ingest import IngestConfig, IngestEngine, _resolve_cache_db
from ctxfeed.models import ModelAPIError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_repo(tmp_path: Path) -> Path:
    """A small repo: stable_prefix (README) + body (1 .py)."""
    (tmp_path / "README.md").write_text("# tiny\nA small test repo.\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("def main():\n    return 0\n", encoding="utf-8")
    return tmp_path


def _make_repo(root: Path) -> Path:
    """Create a 2-file repo whose content is identical across callers (same hashes)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text(
        "# shared\nIdentical content across repos.\n", encoding="utf-8"
    )
    (root / "app.py").write_text("def main():\n    return 0\n", encoding="utf-8")
    return root


# ===========================================================================
# fix-ctxfeed-model-env-shadowed
# ===========================================================================

def test_resolve_model_env_wins_when_flag_omitted(monkeypatch):
    """With no --model, CTXFEED_MODEL selects the model (the v0.3 fix)."""
    monkeypatch.setenv("CTXFEED_MODEL", "deepseek")
    assert _resolve_model(None) == "deepseek"


def test_resolve_model_defaults_to_glm_without_env(monkeypatch):
    """No flag and no env -> glm (the documented default)."""
    monkeypatch.delenv("CTXFEED_MODEL", raising=False)
    assert _resolve_model(None) == "glm"


def test_resolve_model_flag_wins_over_env(monkeypatch):
    """An explicit --model flag takes precedence over CTXFEED_MODEL."""
    monkeypatch.setenv("CTXFEED_MODEL", "deepseek")
    assert _resolve_model("glm") == "glm"  # flag wins
    assert _resolve_model("deepseek") == "deepseek"


def test_resolve_model_rejects_unknown(monkeypatch):
    monkeypatch.setenv("CTXFEED_MODEL", "claude")
    with pytest.raises(SystemExit):
        _resolve_model(None)


def test_cli_init_honors_ctxfeed_model_env(tiny_repo: Path, monkeypatch):
    """End-to-end: `ctxfeed init` with CTXFEED_MODEL=deepseek and no --model
    runs the DeepSeek plan. The v0.2 truthy "glm" typer default used to shadow
    the env, silently running GLM. Asserted via the model label in the banner
    and the 128k DeepSeek window in the plan summary."""
    from typer.testing import CliRunner

    from ctxfeed.cli import get_app

    # Clean env so no stray key interferes; select deepseek via env only.
    for k in ("CTXFEED_MODEL", "DEEPSEEK_API_KEY", "ZHIPU_API_KEY", "GLM_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("CTXFEED_MODEL", "deepseek")

    runner = CliRunner()
    result = runner.invoke(get_app(), ["init", "--repo", str(tiny_repo)])
    assert result.exit_code == 0, result.stdout
    assert "DeepSeek" in result.stdout  # banner: model=DeepSeek V4 (128k ctx)
    # The DeepSeek 128k window is in the plan summary — confirms the plan,
    # not just the label, switched to DeepSeek.
    assert "128,000" in result.stdout


def test_cli_init_flag_overrides_env(tiny_repo: Path, monkeypatch):
    """--model glm with CTXFEED_MODEL=deepseek still runs GLM (flag wins)."""
    from typer.testing import CliRunner

    from ctxfeed.cli import get_app

    monkeypatch.setenv("CTXFEED_MODEL", "deepseek")
    for k in ("DEEPSEEK_API_KEY", "ZHIPU_API_KEY", "GLM_API_KEY"):
        monkeypatch.delenv(k, raising=False)

    runner = CliRunner()
    result = runner.invoke(
        get_app(), ["init", "--repo", str(tiny_repo), "--model", "glm"]
    )
    assert result.exit_code == 0, result.stdout
    assert "GLM-5.2" in result.stdout


# ===========================================================================
# fix-cache-db-cwd-relative
# ===========================================================================

def test_resolve_cache_db_relative_anchored_to_root(tmp_path: Path):
    """A relative cache_db is anchored to the repo root (absolute)."""
    root = tmp_path / "myrepo"
    root.mkdir()
    resolved = _resolve_cache_db(".ctxfeed/cache.db", root)
    assert Path(resolved).is_absolute()
    assert resolved == str(root.resolve() / ".ctxfeed" / "cache.db")


def test_resolve_cache_db_absolute_passthrough(tmp_path: Path):
    """An absolute cache_db passes through unchanged."""
    abs_path = str(tmp_path / "elsewhere" / "cache.db")
    assert _resolve_cache_db(abs_path, tmp_path) == abs_path


def test_resolve_cache_db_custom_relative_anchored(tmp_path: Path):
    """Any relative cache_db (not just the default) is anchored to root."""
    root = tmp_path / "r"
    root.mkdir()
    resolved = _resolve_cache_db("data/sub.db", root)
    assert resolved == str(root.resolve() / "data" / "sub.db")


def test_cache_plan_default_cache_db_anchored_per_repo(tmp_path: Path):
    """CachePlan.for_repo with no config anchors the default cache_db to root."""
    from ctxfeed.cache_plan import CachePlan

    repo = _make_repo(tmp_path / "repo")
    with CachePlan.for_repo(repo) as cp:
        # Default config had the relative ".ctxfeed/cache.db"; now absolute.
        assert cp.config.cache_db == str(repo.resolve() / ".ctxfeed" / "cache.db")
        assert Path(cp.config.cache_db).is_absolute()


def test_cache_plan_two_repos_same_cwd_no_cache_collision(tmp_path: Path, monkeypatch):
    """Two repos with identical content, run from the same CWD, must NOT share
    a cache. Before the fix, both opened <cwd>/.ctxfeed/cache.db, so repo B's
    delta was computed against repo A's hashes -> content-equal files (the
    shared README) falsely registered as cache hits. Now each repo gets its
    own root-anchored cache, so repo B's first ingest is a true cold start."""
    from ctxfeed.cache_plan import CachePlan

    repo_a = _make_repo(tmp_path / "repoA")
    repo_b = _make_repo(tmp_path / "repoB")
    # Shared CWD: without the fix, both would open tmp_path/.ctxfeed/cache.db.
    monkeypatch.chdir(tmp_path)

    with CachePlan.for_repo(repo_a) as cp_a:
        a_db = cp_a.config.cache_db
        cp_a.ingest()
    with CachePlan.for_repo(repo_b) as cp_b:
        b_db = cp_b.config.cache_db
        result_b = cp_b.ingest()

    # Per-repo absolute paths, distinct.
    assert a_db == str(repo_a.resolve() / ".ctxfeed" / "cache.db")
    assert b_db == str(repo_b.resolve() / ".ctxfeed" / "cache.db")
    assert a_db != b_db
    # repo B's first ingest is a cold start — no false cache hits carried over
    # from repo A despite identical file hashes and a shared process CWD.
    assert result_b.plan.cache_hit_rate == 0.0


def test_ingest_engine_default_cache_db_anchored_to_repo(tmp_path: Path):
    """IngestEngine with the default (relative) cache_db re-anchors it to the
    repo root on the first repo-rooted call, so the same CWD-collision bug is
    fixed on the lower-level m1 path too."""
    repo = _make_repo(tmp_path / "repo")
    # Default config: relative cache_db ".ctxfeed/cache.db", dry-run.
    engine = IngestEngine(IngestConfig(dry_run=True))
    # Before any repo-rooted call, cache_db is still the relative default.
    assert not Path(engine.config.cache_db).is_absolute()
    with engine:
        engine.ingest_repo(repo)
    # After ingest_repo, cache_db is anchored to the repo root (absolute).
    assert Path(engine.config.cache_db).is_absolute()
    assert engine.config.cache_db == str(repo.resolve() / ".ctxfeed" / "cache.db")


def test_ingest_engine_explicit_absolute_cache_db_unchanged(tmp_path: Path, monkeypatch):
    """An explicit absolute cache_db (the common test case) is a no-op for the
    anchoring — regression guard that the fix doesn't rewrite absolute paths."""
    repo = _make_repo(tmp_path / "repo")
    explicit = str(tmp_path / "custom" / "cache.db")
    monkeypatch.chdir(tmp_path)  # even from a shared CWD
    with IngestEngine(IngestConfig(dry_run=True, cache_db=explicit)) as engine:
        engine.ingest_repo(repo)
    assert engine.config.cache_db == explicit  # untouched


# ===========================================================================
# fix-ingest-engine-no-retry
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
    # _call_glm does `with httpx.Client(...) as client` — patch the module ref.
    monkeypatch.setattr(ingest_module.httpx, "Client", lambda **kw: fake)
    monkeypatch.setattr(ingest_module.time, "sleep", lambda _s: None)
    return fake


def _live_config(tmp_path: Path) -> IngestConfig:
    """A config with a non-empty api_key (so NOT dry-run) for live-call tests."""
    return IngestConfig(
        api_key="k-test",
        api_base="https://example.test/v1",
        cache_db=str(tmp_path / "cache.db"),
    )


def test_ingest_engine_401_raises_model_api_error(tiny_repo: Path, tmp_path: Path, monkeypatch):
    """A terminal 401 raises a structured ModelAPIError (status + model + body)
    — no raw httpx traceback escapes. 401 is terminal (not retried)."""
    fake = _patch_ingest_httpx(monkeypatch, [_FakeResp(401, text="unauthorized")])
    with IngestEngine(_live_config(tmp_path)) as engine:
        with pytest.raises(ModelAPIError) as ei:
            engine.ingest_repo(tiny_repo)
    err = ei.value
    assert err.status_code == 401
    assert err.model == "glm-5.2"  # IngestConfig.model default
    assert "unauthorized" in err.body
    assert len(fake.posts) == 1  # 401 is terminal — no retry


def test_ingest_engine_retries_on_429_then_succeeds(tiny_repo: Path, tmp_path: Path, monkeypatch):
    """A transient 429 is retried with backoff and the subsequent 200 returns
    the content."""
    fake = _patch_ingest_httpx(
        monkeypatch,
        [
            _FakeResp(429, text="rate limited"),
            _FakeResp(200, payload=_ok_payload("the-ack")),
        ],
    )
    with IngestEngine(_live_config(tmp_path)) as engine:
        result = engine.ingest_repo(tiny_repo)
    assert result.response == "the-ack"
    assert len(fake.posts) == 2  # retried once after the 429


def test_ingest_engine_retries_on_5xx_then_succeeds(tiny_repo: Path, tmp_path: Path, monkeypatch):
    """A transient 503 is retried and the subsequent 200 returns the answer
    (exercises the query_repo path)."""
    fake = _patch_ingest_httpx(
        monkeypatch,
        [
            _FakeResp(503, text="bad gateway"),
            _FakeResp(200, payload=_ok_payload("ok")),
        ],
    )
    with IngestEngine(_live_config(tmp_path)) as engine:
        qr = engine.query_repo(tiny_repo, "where is auth?")
    assert qr.answer == "ok"
    assert len(fake.posts) == 2


def test_ingest_engine_retries_exhausted_on_repeated_429(tiny_repo: Path, tmp_path: Path, monkeypatch):
    """Repeated 429s exhaust the retry budget and raise a structured
    ModelAPIError (status_code=429, '3 attempts' message)."""
    fake = _patch_ingest_httpx(
        monkeypatch,
        [_FakeResp(429, text="rate limited") for _ in range(3)],
    )
    with IngestEngine(_live_config(tmp_path)) as engine:
        with pytest.raises(ModelAPIError) as ei:
            engine.ingest_repo(tiny_repo)
    assert ei.value.status_code == 429
    assert "3 attempts" in str(ei.value)
    assert len(fake.posts) == 3  # MAX_ATTEMPTS


def test_ingest_engine_dry_run_unchanged_no_httpx(tiny_repo: Path, tmp_path: Path, monkeypatch):
    """Dry-run mode must not touch httpx (regression guard: the retry rewrite
    must not have broken the dry-run short-circuit)."""
    cfg = IngestConfig(dry_run=True, cache_db=str(tmp_path / "cache.db"))
    with IngestEngine(cfg) as engine:
        # If dry-run ever called httpx, this would fail the test.
        monkeypatch.setattr(
            ingest_module.httpx, "Client", lambda **kw: pytest.fail("dry-run must not use httpx")
        )
        result = engine.ingest_repo(tiny_repo)
    assert "[dry-run" in result.response
