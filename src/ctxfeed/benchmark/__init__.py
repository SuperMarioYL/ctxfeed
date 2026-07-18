"""ctxfeed benchmark subpackage.

The m1 kill-check: ingest a repo into GLM-5.2's 1M-token window, run
repo-QA vs a 200k-RAG baseline, measure retrieval quality. If dilution
is real (1M ≈ 200k-RAG), kill before building the MCP server (mvp_plan
§8 kill #1). See :mod:`ctxfeed.benchmark.repo_qa`.
"""

from .repo_qa import RepoQABenchmark, BenchmarkResult, run_benchmark

__all__ = ["RepoQABenchmark", "BenchmarkResult", "run_benchmark"]
