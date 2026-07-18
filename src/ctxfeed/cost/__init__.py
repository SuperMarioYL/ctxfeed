"""ctxfeed cost subpackage — per-query token cost vs Opus.

The cost-delta dashboard is the m3 "star-able moment": two numbers on
screen — files accepted (1000+ vs ChatGPT's 40-file cap) and per-query
token cost vs Opus. See :mod:`ctxfeed.cost.delta` for the API.
"""

from .delta import (
    CHATGPT_FILE_CAP,
    DEEPSEEK_V4_COST,
    GLM_52_COST,
    OPUS_COST,
    CostBreakdown,
    CostDelta,
    CostModel,
    compute_cost_delta,
    compute_query_cost,
    format_text_summary,
    render_dashboard,
)

__all__ = [
    "CHATGPT_FILE_CAP",
    "DEEPSEEK_V4_COST",
    "GLM_52_COST",
    "OPUS_COST",
    "CostBreakdown",
    "CostDelta",
    "CostModel",
    "compute_cost_delta",
    "compute_query_cost",
    "format_text_summary",
    "render_dashboard",
]
