"""Version-surface hygiene guard.

Asserts the package's version surfaces agree, so a release can never again
ship with mismatched version strings — the v0.4.0 drift where ``__version__``,
``VERSION``, and ``pyproject.toml`` all read ``0.3.0`` on a ``v0.4.0`` tag,
and the v0.7.0 drift where the site provenance (``web/site.json``) still read
``v0.6.0`` on the ``v0.7.0`` tag (guarded since v0.8).
"""

import json
import re
from pathlib import Path

import ctxfeed

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match, "no project.version field in pyproject.toml"
    return match.group(1)


def _version_file() -> str:
    return (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()


def test_version_surfaces_agree() -> None:
    """``__version__``, ``VERSION``, and pyproject version must all match."""
    assert ctxfeed.__version__ == _version_file() == _pyproject_version()


def test_site_json_provenance_matches_version() -> None:
    """The site provenance must track the shipped version (v0.8 fix,
    fix-site-provenance-version-drift: the v0.7.0 tag shipped with
    ``web/site.json`` meta.source_version still reading ``v0.6.0``)."""
    site = json.loads(
        (REPO_ROOT / "web" / "site.json").read_text(encoding="utf-8")
    )
    assert site["meta"]["source_version"] == f"v{ctxfeed.__version__}"


def test_site_json_footer_and_content_version_match_version() -> None:
    """footer.tag and meta.content_version must track the shipped version
    (v0.9 fix, fix-site-footer-version-drift: the v0.8.0 release refreshed
    meta.source_version but shipped with footer.tag still reading
    "0.7.0 · Python 3.12+ · MIT" — user-visible in the published Pages
    bundle — and no guard covered footer.tag / meta.content_version)."""
    site = json.loads(
        (REPO_ROOT / "web" / "site.json").read_text(encoding="utf-8")
    )
    assert site["footer"]["tag"].startswith(f"{ctxfeed.__version__} ·")
    assert site["meta"]["content_version"] == ctxfeed.__version__


def test_demo_results_version_matches_version() -> None:
    """The demo provenance (referenced by site.json meta.demo_source) must
    track the shipped version (v0.9 fix, fix-site-footer-version-drift:
    the v0.8.0 tag shipped with docs/demo-results.json still reading
    "version": "0.7.0")."""
    demo = json.loads(
        (REPO_ROOT / "docs" / "demo-results.json").read_text(encoding="utf-8")
    )
    assert demo["version"] == ctxfeed.__version__
