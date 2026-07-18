"""ctxfeed DeepSeek V4 adapter — the prefix-cache cost-fallback.

DeepSeek V4 is the cost-fallback model (mvp_plan §6 out of scope:
"DeepSeek V4 is a cost-fallback only, not a [unified-API] veneer").
Its job is to hold the cost-arbitrage floor (mvp_plan §8 kill #3):
when GLM-5.2 access retracts or pricing moves, DeepSeek V4's
prefix-cache discount keeps per-query cost below Opus.

The :class:`~ctxfeed.cost.delta.DEEPSEEK_V4_COST` model encodes the
~50%% prefix-cache discount on the cached stable_prefix portion. This
adapter is the *API* half: it wraps DeepSeek's OpenAI-compatible
``POST /chat/completions`` endpoint. DeepSeek's context caching is
applied server-side based on the prompt prefix, so the adapter does
not need to emit a separate cache-write call — the deterministic
:class:`~ctxfeed.shard.ShardPlan` ordering (stable_prefix first) is what
makes the prefix cache hit.

In dry-run mode (no API key) it returns a deterministic mock so the
ingest + query pipeline runs end-to-end in CI without credentials.

Usage (dry-run)::

    from ctxfeed.models.deepseek import DeepSeekClient, DeepSeekConfig

    client = DeepSeekClient(DeepSeekConfig())   # no key → dry-run
    resp = client.ingest(prompt)
"""

from __future__ import annotations

from dataclasses import dataclass

from . import BaseChatClient, ModelConfig

__all__ = [
    "DEEPSEEK_CONTEXT_WINDOW",
    "DEEPSEEK_CONFIG_DEFAULTS",
    "DeepSeekConfig",
    "DeepSeekClient",
]


# DeepSeek V4's context window. Smaller than GLM-5.2's 1M, so the
# ShardPlan's budget is recomputed against this when DeepSeek is the
# backing model. This is the "cost-fallback cannot hold the full 1M
# repo" tradeoff the kill-criteria (#3) names.
DEEPSEEK_CONTEXT_WINDOW = 128_000


@dataclass
class DeepSeekConfig(ModelConfig):
    """Config for the DeepSeek V4 adapter.

    Defaults target DeepSeek's public API. ``api_key`` empty → dry-run.
    Override ``model`` for newer DeepSeek releases (e.g. ``deepseek-v4``).
    """

    api_key: str = ""
    api_base: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    timeout: float = 300.0
    dry_run: bool = False


DEEPSEEK_CONFIG_DEFAULTS = DeepSeekConfig()


class DeepSeekClient(BaseChatClient):
    """DeepSeek V4 chat-completions client (the prefix-cache cost-fallback).

    Subclasses :class:`~ctxfeed.models.BaseChatClient` only to bind the
    DeepSeek config defaults; the httpx + dry-run machinery lives in the
    base. The prefix-cache discount is accounted for in
    :mod:`ctxfeed.cost.delta`, not here — the adapter just calls the API;
    the deterministic stable-prefix ordering is what makes the cache hit.
    """

    def __init__(self, config: DeepSeekConfig | None = None):
        super().__init__(config or DEEPSEEK_CONFIG_DEFAULTS)
