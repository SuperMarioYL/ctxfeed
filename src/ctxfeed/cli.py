"""ctxfeed CLI — uvx ctxfeed init / add / cost / mcp.

The m3 milestone deliverable: a typer CLI with rich output. The two
"star-able" numbers from mvp_plan §1 land here:

- ``ctxfeed init``  → files accepted (1000+ vs ChatGPT's 40-file cap)
- ``ctxfeed cost``  → per-query token cost vs Opus

Both run in dry-run by default (no GLM key needed), so the 10-minute
install-to-first-result path works without credentials — the cost
dashboard uses the conservative published rates from
:mod:`ctxfeed.cost.delta`, so the savings claim holds even in dry-run.

Usage::

    uvx ctxfeed init          # scan + plan + ingest (dry-run)
    uvx ctxfeed cost          # cost-delta dashboard vs Opus
    uvx ctxfeed add ./path.py # show what a file adds to the plan
    ctxfeed mcp --repo .      # run the stdio MCP server

Set ``ZHIPU_API_KEY`` (or ``GLM_API_KEY``) for live GLM-5.2 calls.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

try:
    import typer
except ImportError:  # pragma: no cover — typer is a declared dep
    typer = None  # type: ignore[assignment]

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover — rich is a declared dep
    _RICH_AVAILABLE = False

from . import __version__
from .cache_plan import CachePlan
from .cost import (
    CHATGPT_FILE_CAP,
    compute_cost_delta,
    format_text_summary,
    render_dashboard,
)
from .cost.delta import GLM_52_COST, OPUS_COST
from .ingest import IngestConfig
from .models.glm import GLMConfig

# A single console for the whole CLI — consistent theme across subcommands.
_console: "Optional[Console]" = None


def _get_console() -> "Console":
    global _console
    if _console is None:
        if not _RICH_AVAILABLE:
            raise RuntimeError("rich is required for the CLI")
        _console = Console()
    return _console


def _glm_config_from_env() -> GLMConfig:
    """Build a GLMConfig from env (key empty → dry-run)."""
    return GLMConfig(
        api_key=os.environ.get("ZHIPU_API_KEY", "")
        or os.environ.get("GLM_API_KEY", ""),
    )


def _print_banner(text: str, *, style: str = "cyan") -> None:
    """Print a header banner; falls back to plain print without rich."""
    if _RICH_AVAILABLE:
        con = _get_console()
        con.print(Panel(Text(text, style=f"bold {style}"), border_style=style))
    else:
        print(f"=== {text} ===")


def _resolve_repo(repo: Optional[str]) -> Path:
    """Resolve the repo root from --repo or cwd."""
    root = Path(repo) if repo else Path.cwd()
    if not root.is_dir():
        _print_banner(f"Repo not found: {root}", style="red")
        raise SystemExit(2)
    return root.resolve()


# ---------------------------------------------------------------------------
# Typer app — built lazily so the module imports without typer (for tests).
# ---------------------------------------------------------------------------

def _build_app():
    if typer is None:  # pragma: no cover — typer is a declared dep
        raise ImportError("typer is required for the CLI")

    app = typer.Typer(
        name="ctxfeed",
        help=(
            "Local MCP project-context backend — shard a whole repo into "
            "GLM-5.2's 1M-token window for coding agents."
        ),
        no_args_is_help=True,
        add_completion=False,
        context_settings={"help_option_names": ["-h", "--help"]},
    )

    @app.command()
    def init(
        repo: Optional[str] = typer.Option(
            None, "--repo", "-r", help="Repo root (default: cwd)."
        ),
        quiet: bool = typer.Option(
            False, "--quiet", "-q", help="Only print the summary line."
        ),
    ) -> None:
        """Scan a repo, build the cache-aware ShardPlan, and ingest.

        The first "star-able" number lands here: files accepted (1000+
        vs ChatGPT's 40-file cap). Runs dry-run by default (set
        ``ZHIPU_API_KEY`` for a live GLM-5.2 call).
        """
        root = _resolve_repo(repo)
        glm = _glm_config_from_env()
        if not quiet:
            _print_banner(
                f"ctxfeed init — {root}\n"
                f"model=GLM-5.2 (1M ctx)  "
                f"mode={'dry-run' if glm.effective_dry_run() else 'live'}"
            )
        with CachePlan.for_repo(root, glm=glm) as cp:
            result = cp.ingest()
            fv = cp.files_vs_cap()
        if quiet:
            print(result.summary())
            return
        con = _get_console()
        # Plan summary table
        t = Table(show_header=False, border_style="bright_black", padding=(0, 1))
        t.add_column(style="dim")
        t.add_column(style="bold")
        d = result.to_dict()
        t.add_row("files accepted", f"{d['files']}  (vs ChatGPT's {fv['chatgpt_cap']})")
        t.add_row("stable_prefix", f"{d['stable_prefix']}  ({d['stable_prefix_tokens']}t)")
        t.add_row("repo_body", f"{d['repo_body']}  ({d['repo_body_tokens']}t)")
        t.add_row("skipped", str(d["skipped"]))
        t.add_row("tokens", f"{d['total_tokens']:,} / {d['budget']:,}  ({d['fit_ratio']:.1%})")
        t.add_row("cache hit", f"{d['cache_hit_rate']:.0%}")
        t.add_row("delta (new/changed)", str(d["delta"]))
        con.print(t)
        con.print()
        # Files-accepted banner
        over = "PAST" if fv["over_cap"] else "UNDER"
        cap_line = Text.assemble(
            (f"{fv['files_accepted']} ", "bold green"),
            (f"files accepted  ({fv['ratio']}x ChatGPT's cap, {over})", "dim"),
        )
        con.print(Panel(cap_line, border_style="green", padding=(0, 1)))

    @app.command()
    def add(
        path: str = typer.Argument(..., help="File or dir to add to the plan."),
        repo: Optional[str] = typer.Option(
            None, "--repo", "-r", help="Repo root (default: cwd)."
        ),
    ) -> None:
        """Show what a file or dir would add to the ShardPlan.

        Lists the ingestible files under ``path`` with their layer +
        token count. A read-only preview — does not mutate the cache.
        ``path`` is resolved against the current directory (like a
        normal shell path); ``--repo`` is the repo root for layering +
        cache bookkeeping (defaults to cwd).
        """
        root = _resolve_repo(repo)
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = (Path.cwd() / target)
        target = target.resolve()
        if not target.exists():
            _print_banner(f"Not found: {target}", style="red")
            raise SystemExit(2)
        if root not in target.parents and target != root:
            _print_banner(
                f"{target} is outside repo root {root}", style="red"
            )
            raise SystemExit(2)
        glm = _glm_config_from_env()
        with CachePlan.for_repo(root, glm=glm) as cp:
            files = cp.list_files()
        # Filter to the requested path.
        if target == root:
            shown = files
        elif target.is_dir():
            sub = str(target.relative_to(root)).replace(os.sep, "/")
            shown = [f for f in files if f["path"].startswith(sub + "/")]
        else:
            rel = str(target.relative_to(root)).replace(os.sep, "/")
            shown = [f for f in files if f["path"] == rel]
        con = _get_console()
        if not shown:
            con.print(f"[dim]No ingestible files under[/dim] {target}")
            return
        t = Table(title=f"{len(shown)} file(s) under {target}", border_style="bright_black")
        t.add_column("layer", style="dim")
        t.add_column("tokens", justify="right")
        t.add_column("cache", style="dim")
        t.add_column("path")
        for f in sorted(shown, key=lambda x: x["path"]):
            t.add_row(
                f["layer"],
                str(f["tokens"]),
                "cached" if f["cached"] else "new",
                f["path"],
            )
        con.print(t)
        total = sum(f["tokens"] for f in shown)
        con.print(f"[bold]{total:,}[/bold] tokens across {len(shown)} file(s)")

    @app.command()
    def cost(
        repo: Optional[str] = typer.Option(
            None, "--repo", "-r", help="Repo root (default: cwd)."
        ),
        output_tokens: int = typer.Option(
            1024, "--output-tokens", "-o",
            help="Expected answer length in tokens (default 1024).",
        ),
    ) -> None:
        """Per-query token cost vs Opus at equal repo size.

        The second "star-able" number: GLM-5.2 + DeepSeek V4 vs Claude
        Opus, using the conservative published rates from
        :mod:`ctxfeed.cost.delta`. Uses the repo's planned token count
        as the input-token figure.
        """
        root = _resolve_repo(repo)
        glm = _glm_config_from_env()
        with CachePlan.for_repo(root, glm=glm) as cp:
            delta = cp.cost_delta(output_tokens=output_tokens)
            fv = cp.files_vs_cap()
        _print_banner("ctxfeed cost-delta", style="cyan")
        render_dashboard(delta, repo_files=fv["files_accepted"])

    @app.command()
    def mcp(
        repo: Optional[str] = typer.Option(
            None, "--repo", "-r",
            help="Repo root (also settable via CTXFEED_REPO_ROOT env).",
        ),
    ) -> None:
        """Run the stdio MCP server (for ``claude mcp add``)."""
        from .mcp_server import run_stdio
        run_stdio(repo)

    @app.callback(invoke_without_command=True)
    def main(
        ctx: "typer.Context",
        version: bool = typer.Option(
            False, "--version", "-V", help="Print version and exit."
        ),
    ) -> None:
        """ctxfeed — local MCP project-context backend."""
        if version:
            print(f"ctxfeed {__version__}")
            raise typer.Exit()
        # No -V and no subcommand → let typer's no_args_is_help show help.
        if ctx.invoked_subcommand is None:
            print("ctxfeed — local MCP project-context backend.")
            print("Run 'ctxfeed --help' for commands.")
            raise typer.Exit()

    return app


_APP = None


def get_app():
    """Lazily build the typer app (so ``import ctxfeed.cli`` is side-effect-free)."""
    global _APP
    if _APP is None:
        _APP = _build_app()
    return _APP


def main() -> None:
    """Entry point: ``ctxfeed`` console script (pyproject [project.scripts])."""
    if typer is None:
        print(
            "ctxfeed: typer is not installed. Install with: pip install ctxfeed",
            file=sys.stderr,
        )
        raise SystemExit(1)
    get_app()()


if __name__ == "__main__":  # pragma: no cover — manual launch
    main()
