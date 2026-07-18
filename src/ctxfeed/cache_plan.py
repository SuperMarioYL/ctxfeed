"""ctxfeed cache_plan — the orchestration layer over ShardPlan + models.

This module is the single integration point the higher layers (the m2
MCP server, the m3 CLI) call. It ties together:

- :class:`~ctxfeed.shard.ShardPlanBuilder` — the cache-aware ingest
  ordering primitive (stable_prefix + repo_body + delta).
- :class:`~ctxfeed.models.glm.GLMClient` (primary) / DeepSeek V4
  (cost-fallback) — the per-model API adapter.
- :class:`~ctxfeed.ingest.CacheStore` — prefix-cache reuse bookkeeping.
- :func:`~ctxfeed.cost.delta.compute_cost_delta` — the per-query token
  cost vs Opus dashboard data.

Putting this here means the MCP server and the CLI share *one* code path
for "scan a repo → plan → serve / query / report cost" — no drift
between the two consumer surfaces, and the per-model primitive stays
clean (swap GLM ↔ DeepSeek without touching ingest ordering).

Design notes:
- **Per-model**: per mvp_plan §6, the primitive is GLM-5.2 1M
  specifically. :class:`CachePlan` defaults to GLM; pass a
  :class:`~ctxfeed.models.deepseek.DeepSeekClient` to exercise the
  cost-fallback (the budget window shrinks to DeepSeek's 128k).
- **Dry-run**: with no API key, every model call returns a deterministic
  mock, so :meth:`query` and :meth:`ingest` work end-to-end in CI.
- **Cache-aware**: re-ingest after :meth:`plan` shrinks the delta
  (only changed files are "misses") — the prefix-cache reuse signal.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .cost import CostDelta, compute_cost_delta
from .cost.delta import CHATGPT_FILE_CAP
from .ingest import (
    CacheStore,
    FileChunk,
    IngestConfig,
    IngestResult,
    QueryResult,
    TokenBudget,
    format_ingest_prompt,
    scan_repo,
)
from .models import ModelClient, ModelResponse
from .models.glm import GLMClient, GLMConfig
from .shard import ShardPlan, ShardPlanBuilder, build_shard_plan

__all__ = [
    "CachePlan",
    "PlanResult",
]


@dataclass
class PlanResult:
    """A flattened view of a :class:`~ctxfeed.shard.ShardPlan` + its prompt.

    The CLI prints :meth:`summary`; the MCP server returns
    :meth:`to_dict` over the wire. Kept separate from :class:`ShardPlan`
    so the wire payload is stable even if the ShardPlan struct grows.
    """

    plan: ShardPlan
    prompt: str
    model: str
    dry_run: bool

    @property
    def files(self) -> int:
        return self.plan.files

    @property
    def tokens(self) -> int:
        return self.plan.total_tokens

    def summary(self) -> str:
        return (
            f"{self.plan.summary()}  model={self.model}  "
            f"dry_run={'yes' if self.dry_run else 'no'}"
        )

    def to_dict(self) -> dict:
        d = self.plan.to_dict()
        d["model"] = self.model
        d["dry_run"] = self.dry_run
        d["prompt_chars"] = len(self.prompt)
        return d


class CachePlan:
    """Cache-aware plan manager: scan → plan → serve / query / report cost.

    One instance per repo-root. The m2 MCP server constructs one per
    ``query_repo`` / ``list_files`` call; the m3 CLI constructs one per
    ``init`` / ``add`` / ``cost`` invocation. The shared :class:`CacheStore`
    (SQLite) persists across calls so the delta shrinks on re-ingest.

    Usage (dry-run — no API key needed, works in CI)::

        plan = CachePlan.for_repo("/path/to/repo")
        result = plan.ingest()
        print(result.summary())

        answer = plan.query("Where is the auth middleware?")
        print(answer.answer)

        delta = plan.cost_delta()
        # delta.savings_ratio_glm → the per-query savings vs Opus

    Usage (live, GLM API key from env)::

        plan = CachePlan.for_repo(
            "/path/to/1000-file-repo",
            glm=GLMConfig(api_key=os.environ["ZHIPU_API_KEY"]),
        )
    """

    def __init__(
        self,
        root: Path | str,
        *,
        config: IngestConfig | None = None,
        model: ModelClient | None = None,
    ):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Repo root not found: {self.root}")
        self.config = config or IngestConfig()
        # The model client defaults to GLM-5.2 dry-run. The IngestConfig's
        # api_key/window flow into the ShardPlan budget; the model client's
        # api_key governs the actual API call. When the caller passes a
        # model with a key, we mirror it into the config so window/budget
        # stay consistent.
        if model is None:
            glm_cfg = GLMConfig(api_key=self.config.api_key)
            model = GLMClient(glm_cfg)
        self.model = model
        self.cache = CacheStore(self.config.cache_db)
        self._builder = ShardPlanBuilder(self.config)

    @classmethod
    def for_repo(
        cls,
        root: Path | str,
        *,
        glm: Optional[GLMConfig] = None,
        config: IngestConfig | None = None,
    ) -> "CachePlan":
        """Build a :class:`CachePlan` rooted at ``root`` with a GLM client.

        Pass ``glm=GLMConfig(api_key=...)`` for live calls; omit for the
        deterministic dry-run (the default in CI / tests / first install).
        """
        cfg = config or IngestConfig()
        if glm is not None:
            # Mirror the GLM api_key into the ingest config so the
            # IngestEngine / ShardPlan budget stays consistent. The
            # window stays at the IngestConfig default (1_000_000 for
            # GLM-5.2); GLM-5.2's 1M is the primitive, DeepSeek's smaller
            # window is handled by passing a custom IngestConfig.window.
            cfg.api_key = glm.api_key
        client = GLMClient(glm or GLMConfig(api_key=cfg.api_key))
        return cls(root, config=cfg, model=client)

    # -- core operations ------------------------------------------------

    def _build_plan(self) -> tuple[ShardPlan, str]:
        """Scan + plan + render prompt. Returns (plan, prompt)."""
        chunks = scan_repo(self.root, self.config)
        budget = TokenBudget(
            window=self.config.window, headroom=self.config.headroom
        )
        plan = build_shard_plan(chunks, budget, cache=self.cache)
        # Persist cache keys so the next plan's delta shrinks.
        for chunk in plan.ordered_chunks():
            self.cache.put(
                chunk.hash, chunk.path, chunk.tokens, chunk.layer
            )
        self.cache.record_run(
            files=plan.files,
            tokens=plan.total_tokens,
            cache_hits=plan.files - len(plan.delta),
            cache_misses=len(plan.delta),
        )
        prompt = plan.to_prompt(self.config)
        return plan, prompt

    def plan(self) -> PlanResult:
        """Build the cache-aware ShardPlan + render the ingest prompt.

        This is the "scan + order + fit to window" pass — no model call.
        The CLI's ``init`` calls this to show the plan summary; the MCP
        server's ``list_files`` calls a lighter variant (:meth:`list_files`).
        """
        plan, prompt = self._build_plan()
        return PlanResult(
            plan=plan,
            prompt=prompt,
            model=self.model.config.model,
            dry_run=self.model.config.effective_dry_run(),
        )

    def ingest(self) -> PlanResult:
        """Plan + call the model's ingest (context → ack)."""
        result = self.plan()
        resp: ModelResponse = self.model.ingest(result.prompt)
        # Surface the model's ack through the result for the CLI.
        return PlanResult(
            plan=result.plan,
            prompt=result.prompt,
            model=resp.model,
            dry_run=resp.dry_run,
        )

    def query(self, question: str) -> QueryResult:
        """Repo-QA: plan + context + question → answer, one round-trip.

        This is the m1 benchmark primitive and the m2 MCP ``query_repo``
        tool. The whole repo's ingestible files are placed in-context,
        then the question is asked — one model call.
        """
        plan, prompt = self._build_plan()
        resp = self.model.query(prompt, question)
        return QueryResult(
            question=question,
            answer=resp.content,
            files_in_context=plan.files,
            tokens_in_context=plan.total_tokens,
        )

    def list_files(self) -> list[dict]:
        """List all ingestible files with layer + cache metadata.

        The m2 MCP ``list_files`` tool. Lighter than :meth:`plan` — no
        prompt rendering, just the scan + cache lookup.
        """
        chunks = scan_repo(self.root, self.config)
        return [
            {
                "path": c.path,
                "tokens": c.tokens,
                "hash": c.hash,
                "layer": c.layer,
                "cached": self.cache.get(c.hash) is not None,
            }
            for c in chunks
        ]

    def cost_delta(
        self,
        *,
        output_tokens: int = 1024,
    ) -> CostDelta:
        """Compute per-query token cost across models vs Claude Opus.

        Uses the planned repo's total tokens as the input-token count
        and the ShardPlan's stable_prefix tokens as the cached-prefix
        portion (DeepSeek V4's prefix-cache discount applies there).
        """
        plan, _prompt = self._build_plan()
        return compute_cost_delta(
            input_tokens=plan.total_tokens,
            output_tokens=output_tokens,
            cached_prefix_tokens=plan.stable_prefix_tokens,
        )

    def files_vs_cap(self) -> dict:
        """The first "star-able" number: files accepted vs ChatGPT's cap."""
        plan, _prompt = self._build_plan()
        return {
            "files_accepted": plan.files,
            "chatgpt_cap": CHATGPT_FILE_CAP,
            "ratio": round(plan.files / CHATGPT_FILE_CAP, 1) if CHATGPT_FILE_CAP else 0.0,
            "over_cap": plan.files > CHATGPT_FILE_CAP,
        }

    def close(self) -> None:
        self.cache.close()

    def __enter__(self) -> "CachePlan":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
