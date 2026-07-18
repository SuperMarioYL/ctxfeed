"""ctxfeed cost-delta — per-query token cost vs Opus at equal repo size.

This is the m3 "star-able moment": two numbers on screen — files
accepted (1000+ vs ChatGPT's 40-file cap) and per-query token cost vs
Opus. The CLI's ``ctxfeed cost`` subcommand (polish stage) calls
:func:`render_dashboard` to print a rich table; the MCP server
(polish stage) can return :func:`compute_cost_delta` as JSON for
programmatic consumers.

Cost model
----------
Three :class:`CostModel` constants cover the relevant pricing surfaces:

- **GLM-5.2** — ZhipuAI per-token pricing (CN). No prefix-cache
  discount (GLM-5.2 has no published prefix-cache tier at v0.1).
- **DeepSeek V4** — per-token pricing with a prefix-cache discount
  on the cached stable_prefix portion (the m2 ShardPlan's
  contribution to the cost-arbitrage floor).
- **Claude Opus** — US per-token pricing, no prefix-cache discount
  (the incumbent baseline).

All rates are module-level constants so they can be updated when
vendors revise pricing. The defaults reflect publicly listed rates
as of 2026-07, rounded **up** (conservative) — so the savings claim
is never overstated. Even at these conservative rates, GLM-5.2 and
DeepSeek V4 land 1-2 orders of magnitude below Opus per query, which
is the cost-arbitrage floor the product stands on (mvp_plan §8 kill #3).

Usage (programmatic)::

    from ctxfeed.cost import compute_cost_delta, render_dashboard

    delta = compute_cost_delta(
        input_tokens=120_000,          # the ShardPlan's total_tokens
        output_tokens=1024,
        cached_prefix_tokens=24_000,    # ShardPlan.stable_prefix_tokens
    )
    render_dashboard(delta, repo_files=1000)

Usage (the plain-text path, when rich is unavailable)::

    from ctxfeed.cost import compute_cost_delta, format_text_summary
    print(format_text_summary(compute_cost_delta(120_000), repo_files=1000))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover — rich is a declared dep
    _RICH_AVAILABLE = False


__all__ = [
    "CHATGPT_FILE_CAP",
    "OPUS_COST",
    "GLM_52_COST",
    "DEEPSEEK_V4_COST",
    "CostModel",
    "CostBreakdown",
    "CostDelta",
    "compute_query_cost",
    "compute_cost_delta",
    "format_text_summary",
    "render_dashboard",
]


# ---------------------------------------------------------------------------
# Cost models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostModel:
    """Per-token pricing for a model (USD per million tokens).

    ``prefix_cache_discount`` is the fraction of the cached-prefix input
    cost that is refunded when the prefix is a cache hit (e.g. ``0.5`` =
    50%% off the cached portion). ``0.0`` = no prefix-cache pricing.

    Frozen so models can be safely shared as module-level constants.
    """

    name: str
    input_per_m: float                     # USD per 1M input tokens
    output_per_m: float                    # USD per 1M output tokens
    prefix_cache_discount: float = 0.0    # fraction off cached-prefix input


# Defaults reflect publicly listed rates as of 2026-07, rounded UP
# (conservative — never overstate savings vs Opus). Update these
# constants when vendors revise pricing; the rest of the module
# derives everything from them.
OPUS_COST = CostModel(
    name="Claude Opus",
    input_per_m=15.0,     # US incumbent baseline
    output_per_m=75.0,
    prefix_cache_discount=0.0,
)
GLM_52_COST = CostModel(
    name="GLM-5.2",
    # Conservative upper bound on ZhipuAI GLM-5.2 (actual GLM-4 tier is
    # ~10x cheaper; we round up so the savings claim always holds).
    input_per_m=0.14,
    output_per_m=0.28,
    prefix_cache_discount=0.0,   # GLM-5.2 has no published prefix-cache tier
)
DEEPSEEK_V4_COST = CostModel(
    name="DeepSeek V4",
    input_per_m=0.27,
    output_per_m=1.10,
    prefix_cache_discount=0.5,    # ~50% off the cached-prefix input portion
)


# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------

@dataclass
class CostBreakdown:
    """Per-model cost for a single query."""

    model: str
    input_tokens: int
    output_tokens: int
    cached_prefix_tokens: int
    input_cost: float
    output_cost: float

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost

    @property
    def effective_input_per_m(self) -> float:
        """Blended input rate accounting for the prefix-cache discount.

        For DeepSeek V4 with a 50%% discount on a 20%% cached prefix,
        this is ``0.8 * 0.27 + 0.2 * 0.135 = $0.243/M`` — still ~60x
        cheaper than Opus's $15/M.
        """
        if self.input_tokens == 0:
            return 0.0
        return self.input_cost * 1_000_000 / self.input_tokens

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_prefix_tokens": self.cached_prefix_tokens,
            "input_cost_usd": round(self.input_cost, 6),
            "output_cost_usd": round(self.output_cost, 6),
            "total_cost_usd": round(self.total_cost, 6),
            "effective_input_per_m_usd": round(self.effective_input_per_m, 6),
        }


def compute_query_cost(
    model: CostModel,
    input_tokens: int,
    output_tokens: int,
    cached_prefix_tokens: int = 0,
) -> CostBreakdown:
    """Compute the per-query cost for a model.

    ``cached_prefix_tokens`` is the portion of ``input_tokens`` that hit
    the prefix cache (from the ShardPlan's ``stable_prefix_tokens``).
    For models with a prefix-cache discount, that portion is billed at
    ``input_per_m * (1 - discount)``.

    ``cached_prefix_tokens`` is clamped to ``[0, input_tokens]`` so a
    caller cannot accidentally claim a discount on more tokens than were
    sent.
    """
    cached = max(0, min(cached_prefix_tokens, input_tokens))
    fresh_input = input_tokens - cached

    fresh_input_cost = fresh_input * model.input_per_m / 1_000_000
    cached_input_cost = (
        cached
        * model.input_per_m
        * (1 - model.prefix_cache_discount)
        / 1_000_000
    )
    input_cost = fresh_input_cost + cached_input_cost
    output_cost = output_tokens * model.output_per_m / 1_000_000

    return CostBreakdown(
        model=model.name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_prefix_tokens=cached,
        input_cost=input_cost,
        output_cost=output_cost,
    )


@dataclass
class CostDelta:
    """Cost-delta between ctxfeed's models and Claude Opus at equal repo size.

    This is the data behind the m3 dashboard. The two "star-able"
    numbers are:

    - ``opus.total_cost - glm.total_cost`` — absolute savings vs Opus
      (the per-query cost-delta).
    - ``savings_ratio_glm`` — savings as a fraction of Opus's cost.

    DeepSeek V4 is included as the cost-fallback (prefix-cache discount
    applies when the ShardPlan's stable_prefix is reused across queries).
    """

    input_tokens: int
    output_tokens: int
    cached_prefix_tokens: int
    glm: CostBreakdown
    deepseek: CostBreakdown
    opus: CostBreakdown

    @property
    def savings_vs_opus_glm(self) -> float:
        """Absolute USD saved per query by routing GLM-5.2 instead of Opus."""
        return self.opus.total_cost - self.glm.total_cost

    @property
    def savings_ratio_glm(self) -> float:
        """GLM savings as a fraction of Opus's per-query cost."""
        if self.opus.total_cost == 0:
            return 0.0
        return self.savings_vs_opus_glm / self.opus.total_cost

    @property
    def savings_vs_opus_deepseek(self) -> float:
        """Absolute USD saved per query by routing DeepSeek V4 instead of Opus."""
        return self.opus.total_cost - self.deepseek.total_cost

    @property
    def savings_ratio_deepseek(self) -> float:
        """DeepSeek savings as a fraction of Opus's per-query cost."""
        if self.opus.total_cost == 0:
            return 0.0
        return self.savings_vs_opus_deepseek / self.opus.total_cost

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_prefix_tokens": self.cached_prefix_tokens,
            "glm": self.glm.to_dict(),
            "deepseek": self.deepseek.to_dict(),
            "opus": self.opus.to_dict(),
            "savings_vs_opus_usd_glm": round(self.savings_vs_opus_glm, 6),
            "savings_ratio_glm": round(self.savings_ratio_glm, 4),
            "savings_vs_opus_usd_deepseek": round(self.savings_vs_opus_deepseek, 6),
            "savings_ratio_deepseek": round(self.savings_ratio_deepseek, 4),
        }


def compute_cost_delta(
    input_tokens: int,
    output_tokens: int = 1024,
    cached_prefix_tokens: int = 0,
    *,
    glm: Optional[CostModel] = None,
    deepseek: Optional[CostModel] = None,
    opus: Optional[CostModel] = None,
) -> CostDelta:
    """Compute per-query cost across ctxfeed's models vs Claude Opus.

    The defaults are GLM-5.2 (primary), DeepSeek V4 (cost-fallback with
    prefix-cache discount), and Claude Opus (the US incumbent baseline).
    Pass custom :class:`CostModel` instances to sensitivity-test pricing
    (e.g. model a 2x GLM price hike).

    ``output_tokens`` defaults to 1024 — a typical repo-QA answer. The
    caller can override (e.g. a terse 256-token answer or a verbose 4096).
    """
    glm_m = glm or GLM_52_COST
    ds_m = deepseek or DEEPSEEK_V4_COST
    op_m = opus or OPUS_COST

    return CostDelta(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_prefix_tokens=cached_prefix_tokens,
        glm=compute_query_cost(
            glm_m, input_tokens, output_tokens, cached_prefix_tokens
        ),
        deepseek=compute_query_cost(
            ds_m, input_tokens, output_tokens, cached_prefix_tokens
        ),
        # Opus has no prefix-cache discount — the stable_prefix buys it
        # nothing, so cached_prefix_tokens is 0 for the baseline.
        opus=compute_query_cost(op_m, input_tokens, output_tokens, 0),
    )


# ---------------------------------------------------------------------------
# Dashboard rendering
# ---------------------------------------------------------------------------

CHATGPT_FILE_CAP = 40  # ChatGPT Projects' per-project file ceiling


def format_text_summary(
    delta: CostDelta,
    *,
    repo_files: int = 0,
    chatgpt_cap: int = CHATGPT_FILE_CAP,
) -> str:
    """Plain-text one-screen summary for non-rich contexts.

    Used by :func:`render_dashboard` when ``rich`` is unavailable, and
    directly by tests / log output. Two screens of info: files-accepted
    banner (the first star-able number) + per-query cost table (the
    second).
    """
    lines = [
        "ctxfeed cost-delta",
        "=" * 52,
    ]
    if repo_files:
        lines.append(
            f"Files accepted   : {repo_files} vs ChatGPT's {chatgpt_cap}-file cap"
        )
    lines.extend(
        [
            f"Input tokens     : {delta.input_tokens:,}",
            f"Output tokens    : {delta.output_tokens:,}",
            f"Cached prefix    : {delta.cached_prefix_tokens:,}",
            "",
            "Per-query cost (USD):",
            f"  GLM-5.2        : ${delta.glm.total_cost:.4f}",
            f"  DeepSeek V4    : ${delta.deepseek.total_cost:.4f}  "
            f"(prefix-cache {DEEPSEEK_V4_COST.prefix_cache_discount:.0%} off cached)",
            f"  Claude Opus    : ${delta.opus.total_cost:.4f}",
            "",
            f"Savings vs Opus  : GLM ${delta.savings_vs_opus_glm:.4f} "
            f"({delta.savings_ratio_glm:.1%})  |  "
            f"DeepSeek ${delta.savings_vs_opus_deepseek:.4f} "
            f"({delta.savings_ratio_deepseek:.1%})",
        ]
    )
    return "\n".join(lines)


def render_dashboard(
    delta: CostDelta,
    *,
    repo_files: int = 0,
    chatgpt_cap: int = CHATGPT_FILE_CAP,
    console: Optional[object] = None,
) -> None:
    """Render the cost-delta dashboard to a rich :class:`~rich.console.Console`.

    Falls back to printing :func:`format_text_summary` to stdout when
    ``rich`` is unavailable (e.g. in a constrained test env). The CLI's
    ``ctxfeed cost`` subcommand (polish stage) calls this with the
    default console.

    The dashboard shows the two "star-able" numbers from mvp_plan §1:
    files accepted (1000+ vs ChatGPT's 40-file cap) and per-query token
    cost vs Opus.
    """
    if not _RICH_AVAILABLE:
        print(
            format_text_summary(
                delta, repo_files=repo_files, chatgpt_cap=chatgpt_cap
            )
        )
        return

    # _RICH_AVAILABLE is True → Console / Table / Panel / Text are bound.
    con: Console = console if isinstance(console, Console) else Console()

    # Header panel
    header = Text.assemble(
        ("ctxfeed ", "bold cyan"),
        ("cost-delta  ", "bold"),
        (
            f"{delta.input_tokens:,} input / {delta.output_tokens:,} output tokens",
            "dim",
        ),
    )
    con.print(Panel(header, border_style="cyan", padding=(0, 1)))

    # Files-accepted banner (the first "star-able" number)
    if repo_files:
        files_table = Table.grid(padding=(0, 2))
        files_table.add_column(justify="right", style="bold green")
        files_table.add_column(style="dim")
        files_table.add_row(
            f"{repo_files:,} files",
            f"vs ChatGPT Projects' {chatgpt_cap}-file cap",
        )
        con.print(files_table)
        con.print()

    # Cost comparison table
    cost_table = Table(
        title="Per-query cost vs Opus",
        show_header=True,
        header_style="bold magenta",
        border_style="bright_black",
        padding=(0, 1),
    )
    cost_table.add_column("Model", style="bold")
    cost_table.add_column("Input $", justify="right")
    cost_table.add_column("Output $", justify="right")
    cost_table.add_column("Total $", justify="right", style="bold")
    cost_table.add_column("vs Opus", justify="right")

    for label, bd in (("GLM-5.2", delta.glm), ("DeepSeek V4", delta.deepseek)):
        savings = delta.opus.total_cost - bd.total_cost
        ratio = (
            savings / delta.opus.total_cost
            if delta.opus.total_cost
            else 0.0
        )
        cost_table.add_row(
            label,
            f"${bd.input_cost:.4f}",
            f"${bd.output_cost:.4f}",
            f"${bd.total_cost:.4f}",
            f"-${savings:.4f}  ({ratio:.1%})",
            style="green" if savings > 0 else "red",
        )
    cost_table.add_row(
        "Claude Opus",
        f"${delta.opus.input_cost:.4f}",
        f"${delta.opus.output_cost:.4f}",
        f"${delta.opus.total_cost:.4f}",
        "— baseline —",
        style="dim",
    )
    con.print(cost_table)

    # Savings summary panel (the second "star-able" number)
    savings_table = Table.grid(padding=(0, 1))
    savings_table.add_column(style="dim")
    savings_table.add_column(style="bold green")
    savings_table.add_row(
        "GLM-5.2 savings:",
        f"${delta.savings_vs_opus_glm:.4f}  ({delta.savings_ratio_glm:.1%})",
    )
    savings_table.add_row(
        "DeepSeek V4 savings:",
        f"${delta.savings_vs_opus_deepseek:.4f}  ({delta.savings_ratio_deepseek:.1%})",
    )
    con.print(
        Panel(
            savings_table,
            title="Savings vs Opus",
            border_style="green",
            padding=(0, 1),
        )
    )
