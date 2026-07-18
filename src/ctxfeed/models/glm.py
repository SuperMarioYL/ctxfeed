"""ctxfeed GLM-5.2 adapter — the 1M-context primary model.

GLM-5.2 (ZhipuAI / bigmodel.cn) is the per-model primitive ctxfeed is
built on: its 1,000,000-token context window is what holds a 1000+ file
repo whole, and the :class:`~ctxfeed.shard.ShardPlan`'s cache-aware
ordering is tuned to fit inside it with headroom.

Per mvp_plan §6 (out of scope), GLM-5.2 is *the* primitive model — a
unified API over many CN models is explicitly NOT v0.1; DeepSeek V4
(:mod:`ctxfeed.models.deepseek`) is a cost-fallback only.

The adapter wraps ZhipuAI's OpenAI-compatible
``POST /api/paas/v4/chat/completions`` endpoint. In dry-run mode (no
API key) it returns a deterministic mock so the ingest + query pipeline
runs end-to-end in CI without credentials.

Usage (dry-run)::

    from ctxfeed.models.glm import GLMClient, GLMConfig

    client = GLMClient(GLMConfig())   # no key → dry-run
    resp = client.query(prompt, "Where is the auth middleware?")
    print(resp.content)

Usage (live)::

    client = GLMClient(GLMConfig(api_key="your-zhipu-key"))
"""

from __future__ import annotations

from dataclasses import dataclass

from . import BaseChatClient, ModelConfig

__all__ = ["GLM_CONTEXT_WINDOW", "GLM_CONFIG_DEFAULTS", "GLMConfig", "GLMClient"]


# GLM-5.2 published context window (the m1 kill-check budget).
GLM_CONTEXT_WINDOW = 1_000_000


@dataclass
class GLMConfig(ModelConfig):
    """Config for the GLM-5.2 (ZhipuAI) adapter.

    Defaults target GLM-5.2's 1M-context API surface. ``api_key`` empty
    → dry-run (no network). Override ``api_base`` / ``model`` to point at
    a ZhipuAI-compatible gateway or a newer GLM release.
    """

    api_key: str = ""
    api_base: str = "https://open.bigmodel.cn/api/paas/v4"
    model: str = "glm-5.2"
    timeout: float = 300.0
    dry_run: bool = False


# Singleton default config (dry-run) for callers that just want "the GLM client."
GLM_CONFIG_DEFAULTS = GLMConfig()


class GLMClient(BaseChatClient):
    """GLM-5.2 chat-completions client (the 1M-context primary).

    Subclasses :class:`~ctxfeed.models.BaseChatClient` only to bind the
    GLM config defaults; the httpx + dry-run machinery lives in the base.
    """

    def __init__(self, config: GLMConfig | None = None):
        super().__init__(config or GLM_CONFIG_DEFAULTS)
