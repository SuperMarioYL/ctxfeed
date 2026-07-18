<div align="right"><sub><b>English</b> | [简体中文](./README.md)</sub></div>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/hero-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="./assets/hero-light.svg">
  <img src="./assets/hero-light.svg" width="880" alt="ctxfeed — whole-repo context for GLM-5.2's 1M-token window">
</picture>

<p><sub>A local MCP project-context backend for coding agents: shard a whole repo into GLM-5.2's 1M-token window with cache-aware ingest ordering, so Claude Code / Cursor / Codex query 1000+ files in one call — below Opus per-token cost, past ChatGPT's 40-file cap.</sub></p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="license"></a>
  <a href="https://github.com/SuperMarioYL/ctxfeed/releases"><img src="https://img.shields.io/github/v/release/SuperMarioYL/ctxfeed?color=blue" alt="release"></a>
  <img src="https://img.shields.io/github/actions/workflow/status/SuperMarioYL/ctxfeed/ci.yml?branch=main&label=CI" alt="CI">
  <img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white" alt="python">
  <img src="https://img.shields.io/badge/Agent-ready-5E5CE6" alt="Agent-ready">
  <img src="https://img.shields.io/badge/Kimi_K3-wave-10A37F" alt="Kimi K3 wave">
</p>

**One-line hook**: ChatGPT caps projects at 40 files and Claude's 200k window bills tokens until you hit the limit — ctxfeed packs a whole 1000+ file repo into GLM-5.2's 1M context for your coding agent, one MCP round-trip to the answer.

<h2><img src="https://api.iconify.design/tabler:topology-star-3.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Architecture</h2>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./assets/atlas-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="./assets/atlas-light.svg">
  <img src="./assets/atlas-light.svg" width="880" alt="Architecture: repo files → ctxfeed ingest / ShardPlan builder → GLM-5.2 1M ctx; cache-key store (sqlite) → MCP server (stdio) → Claude Code / Cursor / Codex">
</picture>

The ownable primitive is the **ShardPlan** — a cache-aware file ingest ordering that separates ctxfeed from "call GLM with files." `stable_prefix` (dep manifests, README, type defs first) + `delta` (re-ingest only changed files) are the ownable part: a deterministic, prefix-cache-aligned ingest schedule the raw GLM/DeepSeek APIs do not provide. Without it, the destroyer verdict says the product degenerates to a cost-arbitrage RAG wrapper.

## Why now

GLM-5.2's 1M MIT-licensed context window is a 2026 shipped artifact — [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1uwe542/) quotes "Kimi K3 in the next few hours. DeepSeek V4 GA later in the week," multiple CN long-context models landing the same week, and MCP gave coding agents a standard ingestion seam. ctxfeed turns that supply wave into a project-context backend an Agent can consume directly — [@iSyN707](https://twitter.com/iSyN707)'s Kimi K3 wave coverage is the audience primed for it, and [affaan-m/ECC](https://github.com/affaan-m/ECC)'s 230k-star agent-harness audience is the reservoir of users hitting the 40-file / 200k-window caps ctxfeed unblocks.

<h2><img src="https://api.iconify.design/tabler:rocket.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Quickstart</h2>

```bash
uvx ctxfeed init            # scan repo, build ShardPlan, ingest (dry-run by default, no key)
uvx ctxfeed cost            # per-query token cost vs Opus
```

Go live (real GLM-5.2 calls):

```bash
export ZHIPU_API_KEY=glm-...
uvx ctxfeed init            # live GLM-5.2 ingest
```

<details><summary>sample output (dry-run)</summary>

```
╭──────────────────────────────────────────────────────────╮
│ ctxfeed init — /path/to/repo                            │
│ model=GLM-5.2 (1M ctx)  mode=dry-run                   │
╰──────────────────────────────────────────────────────────╯
┌─────────────────────┬────────────────────────────┐
│ files accepted      │ 1024  (vs ChatGPT's 40)    │
│ stable_prefix       │ 12  (3_421t)               │
│ repo_body           │ 1012  (118_733t)           │
│ tokens              │ 122,154 / 1,000,000 (12.2%)│
│ cache hit           │ 0%                         │
│ delta (new/changed) │ 1024                       │
└─────────────────────┴────────────────────────────┘
╭──────────────────────────────────────────────────────────╮
│ 1024 files accepted  (25.6x ChatGPT's cap, PAST)        │
╰──────────────────────────────────────────────────────────╯
```
</details>

<h2><img src="https://api.iconify.design/tabler:terminal-2.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Usage</h2>

```bash
# See what layer + tokens a file/dir adds to the plan (read-only, no cache mutation)
uvx ctxfeed add ./src/auth.py

# Run the stdio MCP server for Claude Code / Cursor / Codex
uvx ctxfeed mcp --repo /path/to/repo

# Register it in Claude Code:
claude mcp add ctxfeed -- uvx ctxfeed mcp --repo /path/to/repo
```

Once registered, ask Claude Code "where is the auth middleware handled?" — ctxfeed ingests the whole repo into GLM-5.2's 1M window and returns an answer citing file paths in one MCP round-trip. The server exposes three tools:

| Tool | Purpose |
|---|---|
| `query_repo(question)` | Whole repo in-context + one call to answer |
| `list_files()` | List ingestible files + layer + cache status |
| `cost_delta()` | Per-query token cost vs Opus |

Programmatic API:

```python
from ctxfeed.cache_plan import CachePlan

with CachePlan.for_repo("/path/to/repo") as cp:
    plan = cp.plan()                              # cache-aware ShardPlan
    qr = cp.query("where is the auth middleware?")  # whole-repo one round-trip
    delta = cp.cost_delta()                       # cost delta vs Opus
print(qr.answer, delta.savings_ratio_glm)
```

<h2><img src="https://api.iconify.design/tabler:photo.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Demo</h2>

![demo](assets/demo.gif)

A 10-minute path from `git clone` to first visible result: `uvx ctxfeed init` → `uvx ctxfeed cost`, surfacing the two "star-able" numbers — files accepted (1000+ vs 40) and per-query cost vs Opus.

<h2><img src="https://api.iconify.design/tabler:adjustments.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Configuration</h2>

| Env var / config | Type | Default | Meaning |
|---|---|---|---|
| `ZHIPU_API_KEY` | str | `""` | GLM-5.2 API key (empty → dry-run mode) |
| `GLM_API_KEY` | str | `""` | Alias, falls back to `ZHIPU_API_KEY` |
| `CTXFEED_REPO_ROOT` | str | — | Repo root for the MCP server (or pass `--repo`) |
| `IngestConfig.window` | int | `1_000_000` | GLM-5.2 context window |
| `IngestConfig.max_file_bytes` | int | `262144` | Skip files larger than this |
| `IngestConfig.cache_db` | str | `.ctxfeed/cache.db` | SQLite cache-key store path |

<h2><img src="https://api.iconify.design/tabler:map-2.svg?color=%230071E3&width=24" height="22" align="absmiddle" alt=""> Roadmap</h2>

- [x] **m1 ingest benchmark** — ingest 1000 files into GLM-5.2's 1M window + repo-QA vs 200k-RAG kill-check
- [x] **m2 shard + MCP** — cache-aware `ShardPlan` + stdio MCP server (`query_repo` / `list_files`)
- [x] **m3 ship CLI** — `uvx ctxfeed init/add/cost` + cost-delta dashboard
- [ ] Multi-vendor cost-fallback hedge (Kimi K3 / GLM 5.5 config stubs)
- [ ] ECC-class agent-harness integration PR (dify / ECC adopt ctxfeed as the MCP project-context backend)

### vs ChatGPT Projects / Claude Code 200k window

| Axis | ctxfeed (GLM-5.2 1M) | ChatGPT Projects | Claude Code 200k |
|---|:---:|:---:|:---:|
| File ceiling | ✓ 1000+ | ✗ 40 | partial (token-truncated) |
| Per-query cost | ✓ CN long-context pricing | — bundled | ✗ US per-token |
| Prefix-cache discount | ✓ DeepSeek V4 fallback | — | ✗ |
| Native MCP to agents | ✓ stdio | ✗ | partial |
| Enterprise compliance / data residency | ✗ (CN routing is a hard stop) | ✓ | ✓ |

Enterprise compliance buyers are out of scope — CN routing is a hard stop for that segment, not a preference (mvp_plan §6 out of scope).

## Share this

```
ctxfeed — pack a whole 1000+ file repo into GLM-5.2's 1M token window, the MCP project-context backend for coding agents. Past ChatGPT's 40-file cap, below Opus cost. https://github.com/SuperMarioYL/ctxfeed
```

## License + Contributing

MIT — see [LICENSE](./LICENSE). File an issue or PR at [github.com/SuperMarioYL/ctxfeed/issues](https://github.com/SuperMarioYL/ctxfeed/issues).

<p align="center"><sub><a href="./LICENSE">MIT</a> © 2026 SuperMarioYL</sub></p>
