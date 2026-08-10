"""ctxfeed ShardPlan — the cache-aware file ingest ordering primitive.

The ShardPlan is the ownable primitive that separates ctxfeed from
"call GLM with files." It formalizes the m1 ingest ordering into a
first-class struct with four fields (per mvp_plan §2):

    ShardPlan:
      stable_prefix: List[FileChunk]   # rarely-changing files first
      repo_body:     List[FileChunk]   # ordered by import-graph centrality
      delta:         Set[FileHash]      # only re-ingest changed shards on update
      budget:        TokenBudget        # fit to GLM-5.2's 1M window with headroom

The ``stable_prefix`` + ``delta`` fields are the ownable part: a
deterministic, prefix-cache-aligned ingest schedule the raw GLM/DeepSeek
APIs do not provide. Without this, the destroyer verdict says the product
degenerates to a cost-arbitrage RAG wrapper.

This module builds on the m1 primitives (``FileChunk``, ``TokenBudget``,
``scan_repo``, ``CacheStore``) defined in :mod:`ctxfeed.ingest`. The m2
MCP server (polish stage) consumes ``ShardPlan.to_prompt()`` to serve
repo-wide queries in one round-trip; the m3 CLI (polish stage) consumes
``ShardPlan.summary()`` for the cost-delta dashboard.

Design notes:
- **Centrality hook**: ``repo_body`` ordering uses a cheap cross-reference
  centrality proxy by default (a file referenced by many others ranks
  earlier). A real per-language import-graph parser is a future
  enhancement and is out of scope for v0.1; the hook is
  :func:`centrality_score` and can be swapped via ``centrality_fn``.
- **Cache-aware**: when a :class:`~ctxfeed.ingest.CacheStore` is supplied,
  the ``delta`` field is the set of file hashes new/changed since the last
  ingest (so re-ingest only touches the delta). Without a cache, delta =
  all hashes (cold-start).
- **Budget fit**: ``stable_prefix`` is packed first (cache-aligned), then
  ``repo_body`` by descending centrality. Files that do not fit land in
  ``skipped`` so the caller can report the drop rate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from .ingest import (
    CacheStore,
    FileChunk,
    IngestConfig,
    TokenBudget,
    _resolve_cache_db,
    format_ingest_prompt,
    scan_repo,
)

__all__ = [
    "ShardPlan",
    "ShardPlanBuilder",
    "CentralityFn",
    "build_shard_plan",
    "centrality_score",
]


# ---------------------------------------------------------------------------
# Centrality scoring (import-graph proxy)
# ---------------------------------------------------------------------------

# Identifiers of 3+ chars — skips `if`/`for` noise while catching `auth`,
# `models`, `ingest`, etc. Language-agnostic: works on any codebase.
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def _basename_stem(path: str) -> str:
    """Return the basename without extension, POSIX form.

    ``src/ctxfeed/shard.py`` -> ``shard``; ``pkg/mod.ts`` -> ``mod``.
    Used as the cross-reference key: a file ``auth.py`` is "referenced"
    by any chunk whose content mentions the identifier ``auth``.
    """
    posix = path.replace("\\", "/")
    base = posix.rsplit("/", 1)[-1]
    if "." in base:
        base = base.rsplit(".", 1)[0]
    return base


def _build_reference_index(
    chunks: Sequence[FileChunk],
) -> dict[str, set[str]]:
    """Build a stem -> set of chunk paths that reference it.

    One pass over all chunks. :func:`centrality_score` uses this when
    called from :func:`build_shard_plan`; the public
    :func:`centrality_score` falls back to a per-target scan if no index
    is supplied.
    """
    index: dict[str, set[str]] = {}
    for chunk in chunks:
        # set() dedupes: a chunk that mentions `auth` 50 times counts once.
        found = set(_TOKEN_RE.findall(chunk.content))
        for tok in found:
            index.setdefault(tok, set()).add(chunk.path)
    return index


def centrality_score(
    target: FileChunk, all_chunks: Sequence[FileChunk]
) -> int:
    """Cheap cross-reference centrality proxy.

    Counts how many *other* chunks' content references ``target``'s
    basename stem as a word-boundary token. Files imported/referenced by
    many others rank earlier in the ``repo_body`` — a language-agnostic
    stand-in for a real import-graph centrality, which would require a
    per-language parser (out of scope for v0.1).

    This is a real signal: a file like ``auth.py`` referenced by 20 other
    files gets a higher score than a leaf ``utils.py`` referenced by 1.

    This is the O(n) per-target reference implementation.
    :func:`build_shard_plan` uses an optimized index-based version
    internally (one pass to build, O(1) lookup per target).
    """
    stem = _basename_stem(target.path)
    if not stem or len(stem) < 2:
        return 0
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])" + re.escape(stem) + r"(?![A-Za-z0-9_])"
    )
    score = 0
    for chunk in all_chunks:
        if chunk.path == target.path:
            continue
        if pattern.search(chunk.content):
            score += 1
    return score


CentralityFn = Callable[[FileChunk, Sequence[FileChunk]], int]


# ---------------------------------------------------------------------------
# ShardPlan
# ---------------------------------------------------------------------------

@dataclass
class ShardPlan:
    """The cache-aware file ingest ordering primitive.

    Fields (per mvp_plan §2):
        stable_prefix: rarely-changing files first (deps, README, type
            defs, CI/lint config) — maximizes prefix-cache hit rate.
        repo_body: ordered by descending import-graph centrality
            (cross-reference proxy), tie-broken by descending token
            count, then path.
        delta: set of file hashes new/changed since the last ingest
            (empty cache = all hashes = cold-start).
        budget: the token budget the plan was fit into.

    Plus:
        skipped: files that did not fit the budget.
        total_tokens: sum of tokens across stable_prefix + repo_body.
    """

    stable_prefix: list[FileChunk] = field(default_factory=list)
    repo_body: list[FileChunk] = field(default_factory=list)
    delta: set[str] = field(default_factory=set)
    budget: Optional[TokenBudget] = None
    skipped: list[FileChunk] = field(default_factory=list)
    total_tokens: int = 0

    @property
    def files(self) -> int:
        """Total files in the plan (stable_prefix + repo_body)."""
        return len(self.stable_prefix) + len(self.repo_body)

    @property
    def stable_prefix_tokens(self) -> int:
        """Token count of the stable_prefix (the cacheable portion)."""
        return sum(c.tokens for c in self.stable_prefix)

    @property
    def repo_body_tokens(self) -> int:
        """Token count of the repo_body."""
        return sum(c.tokens for c in self.repo_body)

    @property
    def fit_ratio(self) -> float:
        """Fraction of the budget window consumed by planned files."""
        if self.budget is None or self.budget.window == 0:
            return 0.0
        return self.total_tokens / self.budget.window

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of planned chunks NOT in the delta (i.e., cached).

        Chunk-level (not hash-level) so two files sharing a hash don't
        inflate the rate. On a cold-start (no prior cache), delta = all
        hashes so every chunk's hash is in delta → 0%. On a re-ingest
        with unchanged files, delta shrinks and this climbs toward 100%
        — the prefix-cache reuse signal. Matches the m1
        :attr:`~ctxfeed.ingest.IngestResult.cache_hit_rate` semantics.
        """
        total = self.files
        if total == 0:
            return 0.0
        cached = sum(1 for c in self.ordered_chunks() if c.hash not in self.delta)
        return cached / total

    def ordered_chunks(self) -> list[FileChunk]:
        """Flat ordered list: stable_prefix first, then repo_body.

        This is the ingest order — prefix-cache friendly because the
        stable_prefix rarely changes between queries.
        """
        return list(self.stable_prefix) + list(self.repo_body)

    def to_prompt(self, config: IngestConfig | None = None) -> str:
        """Render the plan as the GLM-5.2 ingest prompt.

        Delegates to :func:`ctxfeed.ingest.format_ingest_prompt` over the
        ordered chunk list. The opening tag + stable_prefix form a
        reusable prefix that DeepSeek V4 / GLM-5.2 can cache.
        """
        return format_ingest_prompt(self.ordered_chunks(), config)

    def summary(self) -> str:
        """One-line human summary for CLI / benchmark output."""
        cached = sum(1 for c in self.ordered_chunks() if c.hash not in self.delta)
        budget_str = f"/{self.budget.window}" if self.budget else ""
        return (
            f"files={self.files} "
            f"stable_prefix={len(self.stable_prefix)} "
            f"body={len(self.repo_body)} "
            f"skipped={len(self.skipped)} "
            f"tokens={self.total_tokens}{budget_str} "
            f"({self.fit_ratio:.1%}) "
            f"cache_hit={self.cache_hit_rate:.0%} "
            f"({cached}/{self.files})"
        )

    def to_dict(self) -> dict:
        """Serialize for JSON output (benchmark results, MCP responses)."""
        return {
            "files": self.files,
            "stable_prefix": len(self.stable_prefix),
            "stable_prefix_tokens": self.stable_prefix_tokens,
            "repo_body": len(self.repo_body),
            "repo_body_tokens": self.repo_body_tokens,
            "skipped": len(self.skipped),
            "delta": len(self.delta),
            "total_tokens": self.total_tokens,
            "budget": self.budget.window if self.budget else None,
            "fit_ratio": round(self.fit_ratio, 4),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "stable_prefix_paths": [c.path for c in self.stable_prefix],
            "repo_body_paths": [c.path for c in self.repo_body],
            "skipped_paths": [c.path for c in self.skipped],
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_shard_plan(
    chunks: list[FileChunk],
    budget: TokenBudget,
    *,
    cache: CacheStore | None = None,
    centrality_fn: CentralityFn | None = None,
) -> ShardPlan:
    """Build a cache-aware :class:`ShardPlan` from scanned FileChunks.

    Steps:
        1. Split into stable_prefix / repo_body by the ``FileChunk.layer``
           field (set by :func:`ctxfeed.ingest.scan_repo`'s
           ``_classify_layer``).
        2. Order stable_prefix alphabetically by path — deterministic
           across repos, maximizing prefix-cache reuse.
        3. Order repo_body by descending centrality (default: optimized
           cross-reference index), tie-broken by descending token count,
           then path.
        4. Pack stable_prefix first (cache-aligned), then repo_body, into
           the budget. Files that do not fit land in ``skipped``.
        5. Compute delta: hashes new/changed vs the cache. Without a
           cache, delta = all hashes (cold-start).

    The ``budget`` is consumed in place (mutated) — ``budget.used`` after
    this call reflects ``plan.total_tokens``. This matches the m1
    :func:`~ctxfeed.ingest.build_ingest_order` contract.
    """
    plan = ShardPlan(budget=budget)

    # 1. Split by layer
    stable = [c for c in chunks if c.layer == "stable_prefix"]
    body = [c for c in chunks if c.layer == "body"]

    # 2. Stable prefix: alphabetical by path (deterministic across repos)
    stable.sort(key=lambda c: c.path)

    # 3. Repo body: descending centrality, tie-break by -tokens, then path
    if centrality_fn is None:
        # Optimized: one-pass index, O(1) lookup per target.
        ref_index = _build_reference_index(chunks)

        def _indexed_centrality(
            target: FileChunk, _all: Sequence[FileChunk]
        ) -> int:
            stem = _basename_stem(target.path)
            if not stem or len(stem) < 2:
                return 0
            return len(ref_index.get(stem, set()) - {target.path})

        scorer: CentralityFn = _indexed_centrality
    else:
        scorer = centrality_fn

    body_with_scores = [(c, scorer(c, chunks)) for c in body]
    body_with_scores.sort(key=lambda t: (-t[1], -t[0].tokens, t[0].path))
    body_sorted = [c for c, _ in body_with_scores]

    # 4. Pack stable_prefix first, then repo_body, into the budget
    for chunk in stable:
        if budget.consume(chunk.tokens):
            plan.stable_prefix.append(chunk)
            plan.total_tokens += chunk.tokens
        else:
            plan.skipped.append(chunk)
    for chunk in body_sorted:
        if budget.consume(chunk.tokens):
            plan.repo_body.append(chunk)
            plan.total_tokens += chunk.tokens
        else:
            plan.skipped.append(chunk)

    # 5. Delta: hashes new/changed vs the cache (cold-start = all hashes)
    current_hashes = {c.hash for c in plan.ordered_chunks()}
    if cache is not None:
        plan.delta = cache.delta(current_hashes)
    else:
        plan.delta = set(current_hashes)

    return plan


class ShardPlanBuilder:
    """Convenience builder: scan a repo and produce a ShardPlan in one call.

    Wraps :func:`ctxfeed.ingest.scan_repo` + :func:`build_shard_plan`
    with the :class:`~ctxfeed.ingest.IngestConfig`'s budget and the
    config's :class:`~ctxfeed.ingest.CacheStore`. The m2 MCP server
    (polish stage) and m3 CLI (polish stage) use this to turn a repo path
    into a ready-to-serve ShardPlan.

    Usage (dry-run, no API key needed)::

        builder = ShardPlanBuilder(IngestConfig(dry_run=True))
        plan = builder.build("/path/to/repo")
        print(plan.summary())
        prompt = plan.to_prompt()

    Usage (live, with GLM API key + cache)::

        with ShardPlanBuilder(IngestConfig(api_key="glm-...")) as b:
            plan = b.build("/path/to/1000-file-repo")
            # plan.delta shrinks on re-ingest (cache-aware)
    """

    def __init__(self, config: IngestConfig | None = None):
        self.config = config or IngestConfig()
        self.cache = CacheStore(self.config.cache_db)

    def _ensure_cache_anchored(self, root: Path) -> None:
        """Re-anchor a relative ``cache_db`` to ``root`` (v0.3 fix).

        Idempotent: a no-op when ``cache_db`` is already absolute (the common
        test case). When relative (the ``.ctxfeed/cache.db`` default), the
        cache is re-opened anchored to the repo root so two repos built from
        the same process CWD don't share one cache (false cache hits).
        Mirrors :meth:`ctxfeed.ingest.IngestEngine._ensure_cache_anchored`.
        """
        resolved = _resolve_cache_db(self.config.cache_db, root)
        if resolved != self.config.cache_db:
            self.cache.close()
            self.config.cache_db = resolved
            self.cache = CacheStore(resolved)

    def _build_budget(self) -> TokenBudget:
        return TokenBudget(
            window=self.config.window, headroom=self.config.headroom
        )

    def build(
        self,
        root: Path | str,
        *,
        centrality_fn: CentralityFn | None = None,
        persist: bool = True,
    ) -> ShardPlan:
        """Scan a repo and return a cache-aware :class:`ShardPlan`.

        When ``persist`` is True (default), the plan's ingested file
        hashes are written to the :class:`~ctxfeed.ingest.CacheStore`
        after building — mirroring m1
        :meth:`~ctxfeed.ingest.IngestEngine.ingest_repo`'s cache writes,
        so the *next* build's ``delta`` shrinks (cache-aware re-ingest).
        This matches the architecture diagram in mvp_plan §4 (ShardPlan
        builder → cache-key store). Pass ``persist=False`` for a pure
        planning pass that leaves the cache untouched.
        """
        root_path = Path(root).resolve()
        if not root_path.is_dir():
            raise FileNotFoundError(f"Repo root not found: {root_path}")
        # v0.3 fix: anchor a relative cache_db to the repo root before use.
        self._ensure_cache_anchored(root_path)
        chunks = scan_repo(root_path, self.config)
        budget = self._build_budget()
        plan = build_shard_plan(
            chunks, budget, cache=self.cache, centrality_fn=centrality_fn
        )
        if persist:
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
        return plan

    def close(self) -> None:
        self.cache.close()

    def __enter__(self) -> "ShardPlanBuilder":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
