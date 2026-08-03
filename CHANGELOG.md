# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
