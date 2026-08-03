"""ctxfeed — local MCP project-context backend.

Shards a whole repo into GLM-5.2's 1M-token window with cache-aware
ingest ordering, so coding agents like Claude Code and Cursor query
1000+ files in one call — below Opus per-token cost, past ChatGPT's
40-file cap.
"""

__version__ = "0.2.0"
