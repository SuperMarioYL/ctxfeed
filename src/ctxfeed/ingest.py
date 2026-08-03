"""ctxfeed ingest — the m1 kill-check engine.

Scans a repo, tokenizes files, fits them into GLM-5.2's 1M-token window,
and ingests via the GLM API with cache-aware ordering. Supports the m1
repo-QA benchmark: ingest 1000+ files, query, measure retrieval quality
vs a 200k-RAG baseline.

This module is the foundation for the ShardPlan primitive (m2) and the
MCP server (m2). It defines the core data types (FileChunk, TokenBudget,
FileHash) that later milestones build on, and provides a self-contained
IngestEngine that the benchmark (polish stage) calls to run the kill-check.

Design notes:
- Self-contained: no imports from other ctxfeed modules (they arrive in
  m2/polish). All core types live here so shard.py / cache_plan.py can
  import from ingest.py without circular deps.
- GLM-5.2 specific: the window, API endpoint, and model name are tuned
  for GLM-5.2's 1M context. DeepSeek V4 is a cost-fallback only (m2+).
- Dry-run mode: when no API key is set, the engine returns mock responses
  so the ingest + query pipeline can be tested end-to-end without network.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:  # pragma: no cover — httpx is a declared dep
    _HTTPX_AVAILABLE = False

try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:  # pragma: no cover — tiktoken is a declared dep
    _TIKTOKEN_AVAILABLE = False

try:
    import pathspec

    _PATHSPEC_AVAILABLE = True
except ImportError:  # pragma: no cover — pathspec is a declared dep as of v0.2
    _PATHSPEC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FileChunk:
    """A single file's ingest unit.

    Immutable so it can be hashed and safely shared across ingest passes.
    The `layer` field is the m1 foundation for the ShardPlan's
    stable_prefix / repo_body split (m2 formalizes the ShardPlan struct).
    """

    path: str          # relative path within repo (POSIX separators)
    content: str       # decoded file content
    tokens: int        # token count for this file's content
    hash: str          # sha256[:16] of content — delta-detection key
    layer: str = "body"  # "stable_prefix" | "body"

    def header(self) -> str:
        """The XML-style header prepended to this file in the ingest prompt."""
        return f'<file path="{self.path}">'

    def render(self) -> str:
        """Render this chunk as it appears in the ingest prompt."""
        return f"\n{self.header()}\n{self.content}\n</file>"


@dataclass
class TokenBudget:
    """Context-window budget tracker.

    GLM-5.2 has a 1,000,000-token window. We reserve headroom for the
    prompt template + expected response so the ingest never overflows.
    """

    window: int            # total context window (1_000_000 for GLM-5.2)
    headroom: int           # reserved for prompt template + response
    used: int = 0           # tokens consumed so far

    @property
    def available(self) -> int:
        """Tokens still available for file content."""
        return self.window - self.headroom - self.used

    def consume(self, n: int) -> bool:
        """Try to consume ``n`` tokens. Returns False if it would overflow."""
        if n > self.available:
            return False
        self.used += n
        return True

    def __repr__(self) -> str:
        return (
            f"TokenBudget(window={self.window}, headroom={self.headroom}, "
            f"used={self.used}, available={self.available})"
        )


@dataclass
class IngestConfig:
    """Configuration for the IngestEngine.

    Defaults are tuned for GLM-5.2's 1M context window and the
    ZhipuAI (bigmodel.cn) API surface.
    """

    # GLM-5.2 API
    api_key: str = ""           # GLM / ZhipuAI API key (empty → dry-run)
    api_base: str = "https://open.bigmodel.cn/api/paas/v4"
    model: str = "glm-5.2"      # GLM-5.2 — the 1M-context primitive model
    api_timeout: float = 300.0  # seconds — large repos take a while

    # Context window
    window: int = 1_000_000     # GLM-5.2's 1M-token window
    headroom: int = 8_000       # prompt template + expected response

    # Ingest
    max_files: int = 0          # 0 = no limit
    max_file_bytes: int = 256 * 1024  # skip files > 256 KB
    extensions: tuple[str, ...] = (
        ".py", ".js", ".mjs", ".ts", ".tsx", ".jsx",
        ".java", ".kt", ".go", ".rs", ".c", ".cpp", ".h", ".hpp",
        ".cs", ".rb", ".php", ".swift", ".scala",
        ".md", ".txt", ".rst",
        ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".conf",
        ".sh", ".bash", ".zsh",
        ".sql", ".proto", ".thrift",
    )
    ignore_dirs: tuple[str, ...] = (
        ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
        "dist", "build", ".next", ".idea", ".vscode", ".mypy_cache",
        ".pytest_cache", ".tox", "target", ".eggs", "*.egg-info",
    )
    ignore_names: tuple[str, ...] = (
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
        "go.sum", ".DS_Store", "Thumbs.db",
    )

    # Cache store
    cache_db: str = ".ctxfeed/cache.db"

    # Mode
    dry_run: bool = False       # if True, never call the API

    # Prompt formatting
    context_open_tag: str = "<repo_context>"
    context_close_tag: str = "</repo_context>"

    def effective_dry_run(self) -> bool:
        """True if we should skip the real API call."""
        return self.dry_run or not self.api_key


# ---------------------------------------------------------------------------
# Tokenizer (lazy singleton with fallback)
# ---------------------------------------------------------------------------

_encoder: Optional[Any] = None


def _get_encoder() -> Any:
    """Return a cached tiktoken encoding.

    Uses cl100k_base as a close approximation for GLM-5.2's tokenizer.
    Falls back to a char-based estimator if tiktoken is unavailable.
    """
    global _encoder
    if _encoder is not None:
        return _encoder
    if _TIKTOKEN_AVAILABLE:
        _encoder = tiktoken.get_encoding("cl100k_base")
    else:  # pragma: no cover
        _encoder = None
    return _encoder


def count_tokens(text: str) -> int:
    """Count tokens in ``text``.

    Uses tiktoken's cl100k_base encoding (close to GLM-5.2's tokenizer).
    Falls back to ~4 chars/token if tiktoken is missing.
    """
    enc = _get_encoder()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Repo scanning
# ---------------------------------------------------------------------------

def _should_ignore_dir(name: str, ignore_dirs: Sequence[str]) -> bool:
    """Check if a directory should be pruned from the walk."""
    for pattern in ignore_dirs:
        if "*" in pattern:
            from fnmatch import fnmatch

            if fnmatch(name, pattern):
                return True
        elif name == pattern:
            return True
    return False


def _classify_layer(rel_path: str) -> str:
    """Classify a file into 'stable_prefix' or 'body'.

    The stable_prefix layer contains rarely-changing files (dependency
    manifests, README, type definitions, CI config, lint config). Placing
    these first in the ingest order maximizes prefix-cache hit rate on
    DeepSeek V4 / GLM-5.2, which is the ownable primitive (ShardPlan, m2).
    """
    name = os.path.basename(rel_path)
    lower = name.lower()
    # Use POSIX path for consistent prefix matching
    posix_path = rel_path.replace(os.sep, "/")

    # Dependency manifests
    if name in (
        "package.json", "pyproject.toml", "Cargo.toml", "go.mod",
        "requirements.txt", "Pipfile", "setup.py", "setup.cfg",
        "pom.xml", "build.gradle", "Gemfile", "composer.json",
    ):
        return "stable_prefix"
    # README / docs / license
    if lower.startswith("readme") or lower.startswith("license") or lower.startswith("contributing") or lower.startswith("changelog"):
        return "stable_prefix"
    # Type definitions / schema
    if name.endswith((".d.ts", ".proto", ".thrift", ".graphql", ".gql")):
        return "stable_prefix"
    # CI / CD config
    if posix_path.startswith(".github/") or posix_path.startswith(".gitlab/") or posix_path.startswith(".circleci/"):
        return "stable_prefix"
    if name in (".gitlab-ci.yml", "Jenkinsfile", "azure-pipelines.yml", "docker-compose.yml", "Dockerfile"):
        return "stable_prefix"
    # Lint / type-check config
    if name in (
        "tsconfig.json", "jsconfig.json", "eslint.config.js",
        ".eslintrc.js", ".eslintrc.json", ".eslintrc.yml",
        "mypy.ini", "ruff.toml", ".flake8", "pylintrc",
        ".prettierrc", ".prettierrc.json", "prettier.config.js",
    ):
        return "stable_prefix"
    return "body"


def _load_ignore_spec(root: Path) -> Any:
    """Build a pathspec PathSpec from ``.gitignore`` + ``.ctxfeedignore``.

    v0.2 (feat-gitignore-aware-scan): the scan used to filter only by the
    hardcoded ``ignore_dirs`` / ``ignore_names`` tuples, so generated/ignored
    files not in those lists (build output, ``*.gen.*``, local config) were
    ingested, wasting the context budget. This loads the repo's own
    ``.gitignore`` (and an optional ``.ctxfeedignore`` override) as
    gitwildmatch patterns so the scan matches the user's actual repo hygiene.
    The hardcoded tuple stays as a baseline floor (applied first, in
    :func:`scan_repo`). Returns ``None`` when pathspec is unavailable or no
    ignore files exist (degrades cleanly to the v0.1 behavior).
    """
    if not _PATHSPEC_AVAILABLE:
        return None
    patterns: list[str] = []
    for name in (".gitignore", ".ctxfeedignore"):
        p = root / name
        if p.is_file():
            try:
                patterns.extend(
                    p.read_text(encoding="utf-8", errors="replace").splitlines()
                )
            except OSError:
                continue
    if not patterns:
        return None
    return pathspec.PathSpec.from_lines("gitignore", patterns)


def scan_repo(root: Path | str, config: IngestConfig | None = None) -> list[FileChunk]:
    """Walk a repo directory and collect FileChunks for all ingestible files.

    Files are filtered by extension, size, and ignore patterns. Each file's
    content is read, tokenized, and hashed. The ``layer`` field is set by
    :func:`_classify_layer` for the stable_prefix / body split.

    v0.2: in addition to the hardcoded ``ignore_dirs`` / ``ignore_names`` tuple,
    the scan honors the repo's ``.gitignore`` (and an optional
    ``.ctxfeedignore`` override) so generated/ignored files are excluded.
    """
    config = config or IngestConfig()
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Repo root not found: {root}")

    ignore_spec = _load_ignore_spec(root)

    chunks: list[FileChunk] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored dirs in-place (modifies dirnames so os.walk skips them)
        dirnames[:] = sorted(
            d for d in dirnames if not _should_ignore_dir(d, config.ignore_dirs)
        )
        for fname in sorted(filenames):
            if fname in config.ignore_names:
                continue
            fpath = Path(dirpath) / fname
            if fpath.suffix not in config.extensions:
                continue
            try:
                stat = fpath.stat()
            except OSError:
                continue
            if stat.st_size > config.max_file_bytes:
                continue

            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
            except (OSError, UnicodeDecodeError):
                continue

            rel_path = str(fpath.relative_to(root)).replace(os.sep, "/")

            # v0.2: honor .gitignore / .ctxfeedignore (skip matched files).
            if ignore_spec is not None and ignore_spec.match_file(rel_path):
                continue

            tokens = count_tokens(content)
            file_hash = hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest()[:16]

            chunks.append(
                FileChunk(
                    path=rel_path,
                    content=content,
                    tokens=tokens,
                    hash=file_hash,
                    layer=_classify_layer(rel_path),
                )
            )

    if config.max_files and len(chunks) > config.max_files:
        chunks = chunks[: config.max_files]

    return chunks


# ---------------------------------------------------------------------------
# Ingest ordering (m1 foundation for ShardPlan)
# ---------------------------------------------------------------------------

def build_ingest_order(
    chunks: list[FileChunk], budget: TokenBudget
) -> tuple[list[FileChunk], list[FileChunk]]:
    """Order FileChunks for cache-aware ingestion within a token budget.

    Strategy (m1 foundation — m2's ShardPlan formalizes stable_prefix +
    repo_body + delta into a first-class struct):

    1. **Stable prefix first** — dependency manifests, README, type defs,
       CI/lint config. These rarely change, so placing them first maximizes
       prefix-cache hit rate on DeepSeek V4 / GLM-5.2.
    2. **Body by descending token count** — the largest source files go
       next (they are the hardest to fit, so prioritizing them ensures the
       most important code lands in-context when the budget is tight).

    Returns a tuple of ``(ordered, skipped)`` — files that did not fit the
    budget are in ``skipped`` so the caller can report the drop rate.
    """
    stable = [c for c in chunks if c.layer == "stable_prefix"]
    body = [c for c in chunks if c.layer == "body"]

    # Stable prefix: alphabetical by path (deterministic, stable across repos)
    stable.sort(key=lambda c: c.path)
    # Body: descending token count, then path for tie-breaking determinism
    body.sort(key=lambda c: (-c.tokens, c.path))

    ordered: list[FileChunk] = []
    skipped: list[FileChunk] = []

    for chunk in stable:
        if budget.consume(chunk.tokens):
            ordered.append(chunk)
        else:
            skipped.append(chunk)

    for chunk in body:
        if budget.consume(chunk.tokens):
            ordered.append(chunk)
        else:
            skipped.append(chunk)

    return ordered, skipped


# ---------------------------------------------------------------------------
# GLM-5.2 API client
# ---------------------------------------------------------------------------

def format_ingest_prompt(
    ordered: list[FileChunk], config: IngestConfig | None = None
) -> str:
    """Format ordered file chunks into a single ingest prompt for GLM-5.2.

    The prompt wraps the repo context in XML-style tags so the model can
    parse file boundaries. This format is designed to be prefix-cache
    friendly: the opening tag + first N stable files form a reusable prefix.
    """
    config = config or IngestConfig()
    parts: list[str] = [config.context_open_tag]
    for chunk in ordered:
        parts.append(chunk.render())
    parts.append(f"\n{config.context_close_tag}")
    return "".join(parts)


def _call_glm(
    config: IngestConfig,
    prompt: str,
    query: str | None = None,
) -> str:
    """Call the GLM-5.2 chat completions API.

    If ``query`` is None, this is an ingest-only call (context → ack).
    If ``query`` is provided, this is a repo-QA call (context + question → answer).

    In dry-run mode (no API key or ``dry_run=True``), returns a deterministic
    mock response so the pipeline can be tested without network access.
    """
    if config.effective_dry_run():
        return _dry_run_response(prompt, query)

    if not _HTTPX_AVAILABLE:
        raise ImportError("httpx is required for live API calls but is not installed")

    messages: list[dict[str, str]] = []
    if query is None:
        messages.append(
            {
                "role": "system",
                "content": (
                    "You are a repo-context store. The user will provide "
                    "an entire repository's source files. Acknowledge that "
                    "you have ingested them and report the file count and "
                    "approximate token usage."
                ),
            }
        )
        messages.append({"role": "user", "content": prompt})
    else:
        messages.append(
            {
                "role": "system",
                "content": (
                    "You are a repo-context assistant. The user has provided "
                    "an entire repository's source files as context. Answer "
                    "the user's question using ONLY the ingested files. "
                    "Cite file paths when referencing code."
                ),
            }
        )
        messages.append(
            {"role": "user", "content": f"{prompt}\n\nQuestion: {query}"}
        )

    url = f"{config.api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.model,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.1,
        "stream": False,
    }

    with httpx.Client(timeout=config.api_timeout) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def _dry_run_response(prompt: str, query: str | None) -> str:
    """Deterministic mock response for dry-run / testing mode.

    The mock counts files and estimates tokens from the prompt so the
    pipeline can be exercised end-to-end without a real API key.
    """
    file_count = prompt.count('<file path="')
    # Estimate prompt tokens from char count (cheap, no tiktoken needed)
    token_est = len(prompt) // 4
    if query:
        return (
            f"[dry-run] Ingested {file_count} files (~{token_est} tokens). "
            f"Query: {query[:120]}. "
            f"Mock answer: (would be answered from the {file_count} ingested "
            f"files above in a live call.)"
        )
    return f"[dry-run] Ingested {file_count} files (~{token_est} tokens). Acknowledged."


# ---------------------------------------------------------------------------
# Cache store (SQLite) — prefix-cache reuse bookkeeping
# ---------------------------------------------------------------------------

class CacheStore:
    """SQLite-backed store for cache-key bookkeeping.

    Tracks file hashes + token counts so the engine can compute deltas
    (only re-ingest changed files) and report prefix-cache hit rates.
    This is the ``cache-key store (sqlite)`` box in the architecture diagram.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            db_dir = os.path.dirname(self.db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_keys (
                    file_hash  TEXT PRIMARY KEY,
                    path       TEXT NOT NULL,
                    tokens     INTEGER NOT NULL,
                    layer      TEXT NOT NULL DEFAULT 'body',
                    ingested_at TEXT NOT NULL,
                    cache_key  TEXT
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingest_runs (
                    run_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                    files      INTEGER NOT NULL,
                    tokens     INTEGER NOT NULL,
                    cache_hits INTEGER NOT NULL,
                    cache_misses INTEGER NOT NULL,
                    started_at TEXT NOT NULL
                )
                """
            )
            self._conn.commit()
        return self._conn

    def get(self, file_hash: str) -> dict | None:
        """Return cached metadata for a file hash, or None if not seen."""
        conn = self._connect()
        row = conn.execute(
            "SELECT path, tokens, layer, ingested_at, cache_key "
            "FROM cache_keys WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()
        if row is None:
            return None
        return {
            "path": row[0],
            "tokens": row[1],
            "layer": row[2],
            "ingested_at": row[3],
            "cache_key": row[4],
        }

    def put(
        self,
        file_hash: str,
        path: str,
        tokens: int,
        layer: str = "body",
        cache_key: str | None = None,
    ) -> None:
        """Insert or update a file's cache metadata."""
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO cache_keys "
            "(file_hash, path, tokens, layer, ingested_at, cache_key) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                file_hash,
                path,
                tokens,
                layer,
                datetime.now(timezone.utc).isoformat(),
                cache_key,
            ),
        )
        conn.commit()

    def delta(self, current_hashes: set[str]) -> set[str]:
        """Return hashes present in ``current_hashes`` but NOT in the cache.

        These are new or changed files that need re-ingest. This is the
        ``delta`` field of the ShardPlan — only re-ingest what changed.
        """
        conn = self._connect()
        rows = conn.execute("SELECT file_hash FROM cache_keys").fetchall()
        cached = {r[0] for r in rows}
        return current_hashes - cached

    def cached_hashes(self) -> set[str]:
        """Return all file hashes currently in the cache."""
        conn = self._connect()
        return {r[0] for r in conn.execute("SELECT file_hash FROM cache_keys").fetchall()}

    def record_run(
        self,
        files: int,
        tokens: int,
        cache_hits: int,
        cache_misses: int,
    ) -> int:
        """Record an ingest run for historical tracking. Returns run_id."""
        conn = self._connect()
        cur = conn.execute(
            "INSERT INTO ingest_runs (files, tokens, cache_hits, cache_misses, started_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (files, tokens, cache_hits, cache_misses, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return cur.lastrowid or 0

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        self._connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class IngestResult:
    """Result of an ingest pass.

    The benchmark (polish stage) reads ``files_ingested``, ``tokens_used``,
    ``cache_hits``, and ``cache_misses`` to produce the kill-check numbers.
    """

    files_ingested: int
    files_skipped: int
    tokens_used: int
    budget: int           # the total window (e.g. 1_000_000)
    cache_hits: int
    cache_misses: int
    file_paths: list[str] = field(default_factory=list)
    response: str = ""

    @property
    def fit_ratio(self) -> float:
        """Fraction of the window consumed by ingested file content."""
        return self.tokens_used / self.budget if self.budget else 0.0

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of ingested files that were already in the cache."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0

    def summary(self) -> str:
        """One-line human summary for CLI / benchmark output."""
        return (
            f"ingested={self.files_ingested} skipped={self.files_skipped} "
            f"tokens={self.tokens_used}/{self.budget} "
            f"({self.fit_ratio:.1%}) "
            f"cache_hit={self.cache_hit_rate:.0%} "
            f"({self.cache_hits}/{self.cache_hits + self.cache_misses})"
        )

    def to_dict(self) -> dict:
        """Serialize for JSON output (benchmark results.md)."""
        return {
            "files_ingested": self.files_ingested,
            "files_skipped": self.files_skipped,
            "tokens_used": self.tokens_used,
            "budget": self.budget,
            "fit_ratio": round(self.fit_ratio, 4),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "file_paths": self.file_paths,
            "response": self.response,
        }


@dataclass
class QueryResult:
    """Result of a repo-QA query."""

    question: str
    answer: str
    files_in_context: int
    tokens_in_context: int

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "files_in_context": self.files_in_context,
            "tokens_in_context": self.tokens_in_context,
        }


# ---------------------------------------------------------------------------
# The main engine
# ---------------------------------------------------------------------------

class IngestEngine:
    """The m1 kill-check engine.

    Scans a repo, orders files cache-aware, ingests into GLM-5.2's 1M-token
    window, and supports repo-QA queries. This is the foundation for the
    ShardPlan primitive (m2) and the MCP server (m2).

    Usage (dry-run, no API key needed)::

        engine = IngestEngine(IngestConfig(dry_run=True))
        result = engine.ingest_repo("/path/to/repo")
        print(result.summary())

        answer = engine.query_repo("/path/to/repo", "Where is the auth middleware?")
        print(answer.answer)

    Usage (live, with GLM API key)::

        engine = IngestEngine(IngestConfig(api_key="your-glm-key"))
        result = engine.ingest_repo("/path/to/1000-file-repo")
        print(result.summary())
    """

    def __init__(self, config: IngestConfig | None = None):
        self.config = config or IngestConfig()
        self.cache = CacheStore(self.config.cache_db)

    def _build_budget(self) -> TokenBudget:
        return TokenBudget(
            window=self.config.window, headroom=self.config.headroom
        )

    def _scan_and_order(
        self, root: Path
    ) -> tuple[list[FileChunk], list[FileChunk], list[FileChunk]]:
        """Scan repo, classify layers, and build the ingest order.

        Returns ``(ordered, skipped, all_chunks)``.
        ``all_chunks`` is the full scan (for delta computation); ``ordered``
        fits within the budget; ``skipped`` did not fit.
        """
        all_chunks = scan_repo(root, self.config)
        budget = self._build_budget()
        ordered, skipped = build_ingest_order(all_chunks, budget)
        return ordered, skipped, all_chunks

    def ingest_repo(self, root: Path | str) -> IngestResult:
        """Scan a repo, build ingest order, and ingest into GLM-5.2.

        Steps:
        1. Scan repo for ingestible files (scan_repo).
        2. Compute delta — which files are new/changed since last ingest.
        3. Build cache-aware ingest order (stable prefix first).
        4. Format prompt and call GLM-5.2 API (or dry-run mock).
        5. Update cache store with ingested file metadata.
        6. Record the run for historical tracking.
        """
        root = Path(root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Repo root not found: {root}")

        ordered, skipped, all_chunks = self._scan_and_order(root)

        if not ordered:
            return IngestResult(
                files_ingested=0,
                files_skipped=len(skipped),
                tokens_used=0,
                budget=self.config.window,
                cache_hits=0,
                cache_misses=0,
            )

        # Delta: which ingested files are new/changed vs the cache
        current_hashes = {c.hash for c in ordered}
        changed_hashes = self.cache.delta(current_hashes)
        cache_hits = sum(1 for c in ordered if c.hash not in changed_hashes)
        cache_misses = len(ordered) - cache_hits

        # Format and call the API
        prompt = format_ingest_prompt(ordered, self.config)
        tokens_used = sum(c.tokens for c in ordered)
        response = _call_glm(self.config, prompt, query=None)

        # Update cache store
        for chunk in ordered:
            self.cache.put(
                chunk.hash, chunk.path, chunk.tokens, chunk.layer
            )
        self.cache.record_run(
            files=len(ordered),
            tokens=tokens_used,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )

        return IngestResult(
            files_ingested=len(ordered),
            files_skipped=len(skipped),
            tokens_used=tokens_used,
            budget=self.config.window,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            file_paths=[c.path for c in ordered],
            response=response,
        )

    def query_repo(self, root: Path | str, question: str) -> QueryResult:
        """Query a repo: ingest context + ask a question in one API call.

        This is the repo-QA primitive used by the m1 benchmark and the m2
        MCP ``query_repo`` tool. The entire repo's ingestible files are
        placed in-context, then the question is asked — one round-trip.
        """
        root = Path(root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Repo root not found: {root}")

        ordered, _skipped, _all = self._scan_and_order(root)

        if not ordered:
            return QueryResult(
                question=question,
                answer="[no ingestible files found in repo]",
                files_in_context=0,
                tokens_in_context=0,
            )

        prompt = format_ingest_prompt(ordered, self.config)
        tokens_used = sum(c.tokens for c in ordered)
        answer = _call_glm(self.config, prompt, query=question)

        return QueryResult(
            question=question,
            answer=answer,
            files_in_context=len(ordered),
            tokens_in_context=tokens_used,
        )

    def list_files(self, root: Path | str) -> list[dict]:
        """List all ingestible files in a repo with their metadata.

        Used by the m2 MCP ``list_files`` tool and the m1 benchmark to
        report what was scanned.
        """
        root = Path(root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Repo root not found: {root}")

        chunks = scan_repo(root, self.config)
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

    def close(self) -> None:
        self.cache.close()

    def __enter__(self) -> "IngestEngine":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
