"""Version-surface hygiene guard.

Asserts the package's version surfaces agree, so a release can never again
ship with mismatched version strings — the v0.4.0 drift where ``__version__``,
``VERSION``, and ``pyproject.toml`` all read ``0.3.0`` on a ``v0.4.0`` tag.
"""

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
