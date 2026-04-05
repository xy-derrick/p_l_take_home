"""Condensed Movies dataset clip preparation helpers."""

from __future__ import annotations

from config import DATASET_CONFIG
from data.media_utils import load_prepared_clips, prepare_clips_from_manifest


def load_condensed_movies_clips(force: bool = False, limit: int | None = None) -> list[dict]:
    """Prepare or load Condensed Movies clips."""
    prepared = [] if force else load_prepared_clips("condensed_movies")
    if prepared:
        return prepared[:limit] if limit is not None else prepared
    return prepare_clips_from_manifest(
        dataset_name="condensed_movies",
        manifest_path=DATASET_CONFIG["condensed_movies"]["manifest"],
        default_clip_type="music",
        limit=limit,
        force=force,
    )
