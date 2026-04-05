"""Shared utility helpers."""

from __future__ import annotations

import hashlib
import re


def stable_int_seed(value: str) -> int:
    """Return a process-stable integer seed for deterministic mocks."""
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 32)


def safe_slug(value: str) -> str:
    """Create a filesystem-friendly identifier."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    slug = slug.strip("._")
    return slug or "item"
