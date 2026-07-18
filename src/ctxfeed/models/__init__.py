"""ctxfeed model adapters — per-model API clients.

The plan's primitive is **per-model** (GLM-5.2 1M specifically); DeepSeek V4
is a cost-fallback only, not a unified-API veneer (mvp_plan §6 out of
scope). Each adapter in this subpackage wraps one vendor's chat
completions API behind a shared :class:`ModelClient` interface so the
higher layers (cache_plan, cli, mcp_server) can swap the backing model
without touching ingest ordering logic.

- :mod:`ctxfeed.models.glm`      — GLM-5.2 (ZhipuAI), the 1M-context primary.
- :mod:`ctxfeed.models.deepseek` — DeepSeek V4, the prefix-cache cost-fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover — httpx is a declared dep
    _HTTPX_AVAILABLE = False


__all__ = [
    "ChatMessage",
    "ModelClient",
    "ModelConfig",
    "ModelResponse",
    "BaseChatClient",
]


@dataclass
class ChatMessage:
    """A single chat-completions message."""

    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ModelResponse:
    """Result of a model call."""

    content: str          # the assistant message text
    model: str = ""       # which model answered
    prompt_tokens: int = 0
    completion_tokens: int = 0
    dry_run: bool = False

    def __str__(self) -> str:
        return self.content


@dataclass
class ModelConfig:
    """Base config for a per-model adapter.

    Subclasses (GLMConfig, DeepSeekConfig) set their own defaults via
    inheritance; the fields here are the shared surface the
    :class:`ModelClient` protocol reads.
    """

    api_key: str = ""
    api_base: str = ""
    model: str = ""
    timeout: float = 300.0
    # When True (or when api_key is empty) the client returns a
    # deterministic mock instead of calling the network — so the whole
    # ingest + query pipeline can be exercised in CI without credentials.
    dry_run: bool = False

    def effective_dry_run(self) -> bool:
        return self.dry_run or not self.api_key


class ModelClient(Protocol):
    """The per-model contract every adapter implements.

    Three entry points cover the three call shapes ctxfeed needs:

    - :meth:`chat`   — raw chat-completions (system + user → answer).
    - :meth:`ingest` — context-only ingest (prompt → ack); the
      "load the repo into the window" call.
    - :meth:`query`  — context + question → answer; the repo-QA
      primitive (m1 benchmark, m2 MCP ``query_repo``).
    """

    config: ModelConfig

    def chat(self, messages: Sequence[ChatMessage]) -> ModelResponse: ...

    def ingest(self, prompt: str) -> ModelResponse: ...

    def query(self, prompt: str, question: str) -> ModelResponse: ...


# ---------------------------------------------------------------------------
# Shared base client — httpx + deterministic dry-run mock
# ---------------------------------------------------------------------------

def _dry_run_response(
    prompt: str, question: str | None, model: str
) -> ModelResponse:
    """Deterministic mock for dry-run / testing mode.

    Counts ``<file path="...">`` markers in the prompt so the pipeline
    can be exercised end-to-end without a real API key. Token counts are
    a cheap char-based estimate (no tokenizer needed for the mock).
    """
    file_count = prompt.count('<file path="')
    token_est = len(prompt) // 4
    if question:
        content = (
            f"[dry-run:{model}] Ingested {file_count} files "
            f"(~{token_est} tokens). Query: {question[:120]}. "
            f"Mock answer: (would be answered from the {file_count} "
            f"ingested files above in a live call.)"
        )
    else:
        content = (
            f"[dry-run:{model}] Ingested {file_count} files "
            f"(~{token_est} tokens). Acknowledged."
        )
    return ModelResponse(
        content=content,
        model=model,
        prompt_tokens=token_est,
        completion_tokens=32,
        dry_run=True,
    )


class BaseChatClient:
    """Shared httpx chat-completions client for OpenAI-compatible APIs.

    GLM (ZhipuAI) and DeepSeek both expose an
    ``/chat/completions`` endpoint with Bearer auth and the standard
    ``messages`` payload, so one implementation serves both — the
    subclass just sets ``api_base`` / ``model`` / system prompts.

    In dry-run mode (no api_key or ``dry_run=True``) every call returns
    a deterministic mock from :func:`_dry_run_response`, so the full
    ingest + query pipeline runs in CI without credentials.
    """

    config: ModelConfig

    def __init__(self, config: ModelConfig):
        self.config = config

    # -- the three contract entry points -------------------------------

    def chat(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        return self._call(list(messages), extra_user=None)

    def ingest(self, prompt: str) -> ModelResponse:
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are a repo-context store. The user will provide "
                    "an entire repository's source files. Acknowledge that "
                    "you have ingested them and report the file count and "
                    "approximate token usage."
                ),
            ),
            ChatMessage(role="user", content=prompt),
        ]
        return self._call(messages, extra_user=None)

    def query(self, prompt: str, question: str) -> ModelResponse:
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are a repo-context assistant. The user has provided "
                    "an entire repository's source files as context. Answer "
                    "the user's question using ONLY the ingested files. "
                    "Cite file paths when referencing code."
                ),
            ),
            ChatMessage(role="user", content=f"{prompt}\n\nQuestion: {question}"),
        ]
        return self._call(messages, extra_user=question)

    # -- internals ------------------------------------------------------

    def _system_prompt(self) -> str:  # pragma: no cover — overridden is optional
        return ""

    def _call(
        self,
        messages: list[ChatMessage],
        *,
        extra_user: str | None,
    ) -> ModelResponse:
        cfg = self.config
        if cfg.effective_dry_run():
            return _dry_run_response(
                # reconstruct a representative prompt for the mock's file count
                messages[-1].content,
                extra_user,
                cfg.model,
            )
        if not _HTTPX_AVAILABLE:
            raise ImportError(
                "httpx is required for live model calls but is not installed"
            )
        url = f"{cfg.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": cfg.model,
            "messages": [m.to_dict() for m in messages],
            "max_tokens": 4096,
            "temperature": 0.1,
            "stream": False,
        }
        with httpx.Client(timeout=cfg.timeout) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        usage = data.get("usage", {}) or {}
        return ModelResponse(
            content=data["choices"][0]["message"]["content"],
            model=cfg.model,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            dry_run=False,
        )
