"""Greatest Hits dataset clip preparation helpers."""

from __future__ import annotations

from config import DATASET_CONFIG
from data.media_utils import load_prepared_clips, prepare_clips_from_manifest


def load_greatest_hits_clips(force: bool = False, limit: int | None = None) -> list[dict]:
    """Prepare or load Greatest Hits clips."""
    prepared = [] if force else load_prepared_clips("greatest_hits")
    if prepared:
        return prepared[:limit] if limit is not None else prepared
    return prepare_clips_from_manifest(
        dataset_name="greatest_hits",
        manifest_path=DATASET_CONFIG["greatest_hits"]["manifest"],
        default_clip_type="sfx",
        limit=limit,
        force=force,
    )
