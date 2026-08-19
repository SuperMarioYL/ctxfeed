# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-08-19

### Fixed

- **fix-stale-version-strings-v040**: the v0.4.0 release shipped but its version surfaces still read `0.3.0` — `__version__` (`src/ctxfeed/__init__.py`), `VERSION`, and `pyproject.toml` `version` were never bumped on the `v0.4.0` release commit (`fd659a3`), so `ctxfeed --version`, `pip show ctxfeed`, and the MCP server all misreported as `0.3.0`. Bumped all three surfaces to `0.5.0`; backfilled the missing `[0.4.0]` CHANGELOG section; and added `tests/test_version.py` asserting `__version__ == VERSION == pyproject version` to prevent a future repeat.

## [0.4.0] - 2026-08-16

### Fixed

- **fix-count-tokens-empty-string**: `count_tokens("")` returned `max(1, 0) == 1` on the tiktoken-absent char-fallback path (`max(1, len(text) // 4)`), contradicting `tests/test_ingest.py::test_count_tokens_positive` (which asserts 0). The primary tiktoken path was already correct, so the bug was masked wherever tiktoken is importable and only fired on the char-fallback path hit in air-gapped / no-net environments — precisely ctxfeed's CN air-gap target segment. Now guards `if not text: return 0` before the floor.
- **fix-cache-hits-chunk-vs-hash**: `cache_hits` / `cache_misses` in `CachePlan._build_plan` (`cache_plan.py`) and `ShardPlanBuilder.build` (`shard.py`) were computed hash-level (`plan.files - len(plan.delta)`), but `plan.files` counts CHUNKS while `plan.delta` is a set of HASHES — duplicate-content files (empty `__init__.py`, vendored copies, identical lockfiles) inflated `cache_hit`. Now computed chunk-level (`sum(1 for c in plan.ordered_chunks() if c.hash not in plan.delta)`), matching the existing correct `ShardPlan.cache_hit_rate`.
- **fix-glm-response-choices-unguarded**: `IngestEngine._call_glm` (`ingest.py`) and `BaseChatClient._call` (`models/__init__.py`) indexed `data["choices"][0]["message"]["content"]` with no shape guard on a 200, so a content-filtered or malformed response (`{"choices": []}`, `{"choices": [{"message": {"content": null}}]}`, or a missing-`choices` body) escaped as IndexError / KeyError / silent None instead of the structured `ModelAPIError`. Now validates the shape via a shared `_extract_message_content` helper and raises `ModelAPIError` on a malformed 200, mirroring the terminal-error branch.

## [0.3.0] - 2026-08-11

### Fixed

- **fix-ctxfeed-model-env-shadowed**: `CTXFEED_MODEL` now actually selects the backing model when `--model` is omitted. Every CLI command (`init` / `add` / `cost` / `mcp`) declared `model: str = typer.Option("glm", ...)`, so typer always passed the truthy string `"glm"` when `--model` was omitted, and `_resolve_model`'s `model or env` short-circuited to `"glm"` — the env var was never read end-to-end (CLI → `run_stdio` → `build_server` → `_resolve_model`). The four typer defaults are now `None` (falsy), so `_resolve_model` falls through to `CTXFEED_MODEL` (then `"glm"`). This unblocks the documented DeepSeek-via-env selection on the primary MCP surface.
- **fix-cache-db-cwd-relative**: the SQLite `cache_db` default (`.ctxfeed/cache.db`) is now anchored to the repo root, not the process CWD. Previously `CacheStore` opened it relative to the CWD, so two repos ingested from the same CWD shared one cache — content-equal files (identical README, lockfiles, vendored code) falsely registered as cache hits, inflating the user-facing `cache_hit` metric. `CachePlan`, `IngestEngine`, and `ShardPlanBuilder` now resolve a relative `cache_db` to an absolute path under the repo root; absolute paths pass through unchanged.
- **fix-ingest-engine-no-retry**: `IngestEngine._call_glm` (the m1 repo-QA path) now retries transient failures (429/503/504) with exponential backoff and raises a structured `ModelAPIError` (status_code + body excerpt + model name) on terminal failure, mirroring `BaseChatClient._call` (the m2/m3 path). Previously it did a single `httpx.Client.post` + `resp.raise_for_status()`, surfacing a raw `httpx.HTTPStatusError` on a 401 and skipping retry on 429/5xx — contradicting the v0.2 changelog's "structured error ... clear 401 message for a bad key". The lower-level API now matches the higher-level one.

### Changed

- Bumped version to 0.3.0 (`__version__`, `VERSION`, `pyproject.toml`).
- `tests/test_v030.py` covering all three fixes: env-driven model selection (incl. an end-to-end `ctxfeed init` CLI run), per-repo cache anchoring (two CWDs don't collide), and the IngestEngine retry + `ModelAPIError` contract (401 terminal, 429/5xx retried, budget exhausted).

## [0.2.0] - 2026-08-03

### Fixed

- **fix-cost-path-cache-pollution**: `ctxfeed cost` (CLI) and the MCP `cost_delta` tool no longer mutate the cache store. Previously `CachePlan._build_plan` always persisted cache keys + recorded an ingest_run, so a read-only cost query on a fresh clone falsely inflated the `cache_hit` metric on the next `ctxfeed init` and scanned the repo twice. Both numbers now come from a single non-persisting plan pass (`cost_and_files`).
- **fix-api-transient-errors**: `BaseChatClient._call` now retries transient GLM/DeepSeek API failures (429/503/504) with exponential backoff and raises a structured `ModelAPIError` (status_code + body excerpt + model name) on terminal failures, so the MCP `query_repo` tool returns a structured error instead of a raw httpx traceback (notably a clear 401 message for a bad key).

### Added

- **feat-deepseek-selectable-fallback**: DeepSeek V4 is now a reachable cost-fallback model. `ctxfeed init/cost/mcp --model deepseek` (or the `CTXFEED_MODEL` env var) selects the DeepSeek V4 client and recomputes the `TokenBudget` against its 128k window. GLM-5.2 remains the default primitive; this is explicit selection, not a unified-API veneer. Strengthens kill-criterion #3's DeepSeek hedge.
- **feat-gitignore-aware-scan**: `scan_repo` now honors the repo's `.gitignore` (and an optional `.ctxfeedignore` override) via `pathspec`, so generated/ignored files not in the hardcoded ignore tuple are excluded — wasting less of the context budget. The hardcoded ignore list stays as a baseline floor.
- New dependency `pathspec>=0.3` for gitwildmatch pattern parsing.
- `tests/test_cache_plan.py` and `tests/test_models.py` covering the cost-path no-persist invariant, the DeepSeek model selection + 128k budget, and the API retry + structured-error behavior.

### Changed

- Bumped version to 0.2.0 (`__version__`, `VERSION`, `pyproject.toml`).

## [0.1.0] - 2026-07-18

### Added

- **m1_ingest_benchmark**: Ingest 1000 files from a public medium-size repo into GLM-5.2's 1M-token window, run a repo-QA benchmark against a 200k-RAG baseline, and measure retrieval quality (kill-check).
- **m2_shard_mcp**: Cache-aware `ShardPlan` (stable_prefix + repo_body + delta) and stdio MCP server exposing `query_repo` and `list_files` tools so Claude Code / Cursor / Codex query the whole repo in one round-trip.
- **m3_ship_cli**: `uvx ctxfeed init/add/cost` CLI with rich output and a cost-delta dashboard comparing per-query token cost against Opus at equal repo size.
