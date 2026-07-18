# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-18

### Added

- **m1_ingest_benchmark**: Ingest 1000 files from a public medium-size repo into GLM-5.2's 1M-token window, run a repo-QA benchmark against a 200k-RAG baseline, and measure retrieval quality (kill-check).
- **m2_shard_mcp**: Cache-aware `ShardPlan` (stable_prefix + repo_body + delta) and stdio MCP server exposing `query_repo` and `list_files` tools so Claude Code / Cursor / Codex query the whole repo in one round-trip.
- **m3_ship_cli**: `uvx ctxfeed init/add/cost` CLI with rich output and a cost-delta dashboard comparing per-query token cost against Opus at equal repo size.
