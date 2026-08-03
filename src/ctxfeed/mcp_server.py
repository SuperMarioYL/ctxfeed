"""ctxfeed MCP server — stdio transport for coding agents.

This is the m2 milestone deliverable: a stdio :term:`MCP` server that
exposes two tools a coding agent (Claude Code, Cursor, Codex) calls to
query a whole repo in one round-trip, past ChatGPT Projects' 40-file
ceiling and below Opus per-token cost.

Tools
-----
- ``query_repo``  — ingest the whole repo into the model's context
  window and answer a question in one call. The repo root is resolved
  from the ``CTXFEED_REPO_ROOT`` env var (or ``--repo`` on the CLI).
  Returns the answer + how many files were in context + token count.
- ``list_files``  — list the ingestible files ctxfeed would place
  in-context, with each file's layer (stable_prefix/body) + cache
  status. Lets the agent see what's about to be served before querying.

The server is one long-lived process (stdio); the cache-key store
(SQLite) persists across calls so re-queries shrink the delta
(cache-aware). No microservices, no daemon cluster — matches the
architecture diagram in mvp_plan §4.

Usage (Claude Code)::

    claude mcp add ctxfeed -- uvx ctxfeed mcp --repo /path/to/repo

Then in Claude Code, ask "where is the auth middleware handled?" —
ctxfeed serves the in-context slice and answers in one round-trip.

Dry-run: with no GLM/DeepSeek API key, every ``query_repo`` returns a
deterministic mock, so the server runs end-to-end in CI without
credentials.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from .cache_plan import CachePlan
from .ingest import IngestConfig
from .models.deepseek import DeepSeekConfig
from .models.glm import GLMConfig

__all__ = ["build_server", "run_stdio", "CTXFEED_REPO_ROOT_ENV"]

CTXFEED_REPO_ROOT_ENV = "CTXFEED_REPO_ROOT"
CTXFEED_MODEL_ENV = "CTXFEED_MODEL"


def _resolve_repo_root(repo: Optional[str]) -> Path:
    """Resolve the repo root from arg or env, raising a clear error."""
    root = repo or os.environ.get(CTXFEED_REPO_ROOT_ENV)
    if not root:
        raise RuntimeError(
            "No repo root. Pass --repo, or set "
            f"{CTXFEED_REPO_ROOT_ENV}=/path/to/repo"
        )
    p = Path(repo if repo else root).resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"Repo root not found: {p}")
    return p


def _resolve_model(model: Optional[str]) -> str:
    """Resolve the backing model: --model arg, else CTXFEED_MODEL env, else glm."""
    resolved = (model or os.environ.get(CTXFEED_MODEL_ENV, "") or "glm").strip().lower()
    if resolved not in ("glm", "deepseek"):
        raise RuntimeError(
            f"Unknown model {resolved!r}; expected 'glm' or 'deepseek'"
        )
    return resolved


def _for_repo(root: Path, model: str) -> CachePlan:
    """Build a CachePlan for the resolved model (v0.2 feat-deepseek-selectable-fallback)."""
    if model == "deepseek":
        return CachePlan.for_repo(
            root,
            model="deepseek",
            deepseek=DeepSeekConfig(
                api_key=os.environ.get("DEEPSEEK_API_KEY", "")
            ),
        )
    return CachePlan.for_repo(
        root,
        model="glm",
        glm=GLMConfig(
            api_key=os.environ.get("ZHIPU_API_KEY", "")
            or os.environ.get("GLM_API_KEY", ""),
        ),
    )


def build_server(repo_root: Path | str | None = None, model: Optional[str] = None):
    """Build the FastMCP server with ``query_repo`` + ``list_files`` tools.

    Returns the :class:`mcp.server.fastmcp.FastMCP` instance. Callers
    that want a non-stdio transport (e.g. tests) can inspect the tools
    directly; :func:`run_stdio` runs the stdio transport.

    ``repo_root`` may be a path or None (resolved from env at call time).
    ``model`` selects the backing model (v0.2): ``"glm"`` (default, GLM-5.2 1M)
    or ``"deepseek"`` (V4 128k cost-fallback); overridable per-tool via the
    ``CTXFEED_MODEL`` env var.
    """
    from mcp.server.fastmcp import FastMCP

    server = FastMCP(
        "ctxfeed",
        instructions=(
            "ctxfeed serves a whole repo as MCP project context: "
            "call query_repo to ask a question over all files in one "
            "round-trip, or list_files to see what would be served."
        ),
    )

    def _plan(repo_arg: Optional[str], model_arg: Optional[str] = None) -> CachePlan:
        root = _resolve_repo_root(repo_arg or repo_root)
        return _for_repo(root, _resolve_model(model_arg) if model_arg else _resolve_model(model))

    @server.tool()
    def query_repo(question: str, repo: str = "", model: str = "") -> str:
        """Answer ``question`` using the whole repo's files in-context.

        Ingests the repo's ingestible files into the selected model's context
        window (cache-aware ShardPlan ordering; GLM-5.2 1M by default, DeepSeek
        V4 128k via ``model="deepseek"``) and asks the question in a single model
        round-trip. Returns the answer prefixed with a one-line context summary
        (files + tokens). Transient API errors (429/5xx) are retried with backoff;
        a terminal failure returns a structured error (v0.2).

        Args:
            question: the repo-wide question (e.g. "where is the auth
                middleware handled?").
            repo: optional repo root override (defaults to the ``--repo``
                arg or ``CTXFEED_REPO_ROOT`` env var).
            model: optional model override (``"glm"`` | ``"deepseek"``;
                defaults to the server's ``--model`` / ``CTXFEED_MODEL``).
        """
        with _plan(repo, model) as cp:
            qr = cp.query(question)
        return (
            f"[ctxfeed] {qr.files_in_context} files in context "
            f"({qr.tokens_in_context:,} tokens)\n\n{qr.answer}"
        )

    @server.tool()
    def list_files(repo: str = "", model: str = "") -> str:
        """List the ingestible files ctxfeed would serve in-context.

        One line per file: path · tokens · layer · cached?. Useful for
        an agent to see what's about to be served before querying.
        """
        with _plan(repo, model) as cp:
            files = cp.list_files()
        if not files:
            return "[ctxfeed] no ingestible files found in repo"
        lines = [f"[ctxfeed] {len(files)} ingestible files"]
        for f in files:
            cached = "cached" if f["cached"] else "new"
            lines.append(
                f"  {f['layer']:<13} {f['tokens']:>6}t  {cached:<6}  {f['path']}"
            )
        return "\n".join(lines)

    @server.tool()
    def cost_delta(repo: str = "", model: str = "") -> str:
        """Per-query token cost vs Claude Opus at equal repo size.

        The m3 "star-able" number: GLM-5.2 + DeepSeek V4 vs Opus, plus
        files-accepted vs ChatGPT's 40-file cap. Read-only (v0.2): does
        not mutate the cache store.
        """
        with _plan(repo, model) as cp:
            # v0.2 fix: single non-persisting pass for both numbers.
            d, fv = cp.cost_and_files()
        return (
            f"[ctxfeed] files accepted: {fv['files_accepted']} vs "
            f"ChatGPT's {fv['chatgpt_cap']}-file cap\n"
            f"  GLM-5.2     ${d.glm.total_cost:.4f}/query  "
            f"(saves {d.savings_ratio_glm:.0%} vs Opus)\n"
            f"  DeepSeek V4 ${d.deepseek.total_cost:.4f}/query  "
            f"(saves {d.savings_ratio_deepseek:.0%} vs Opus)\n"
            f"  Claude Opus ${d.opus.total_cost:.4f}/query  (baseline)"
        )

    return server


def run_stdio(
    repo_root: Path | str | None = None, model: Optional[str] = None
) -> None:
    """Run the ctxfeed MCP server over stdio transport.

    This is the entry point the CLI's ``ctxfeed mcp`` subcommand calls.
    Blocks the calling process; the agent client drives the protocol.
    """
    server = build_server(repo_root, model)
    server.run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover — manual launch
    run_stdio()
