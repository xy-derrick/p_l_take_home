"""AVA dataset clip preparation helpers."""

from __future__ import annotations

from config import DATASET_CONFIG
from data.media_utils import load_prepared_clips, prepare_clips_from_manifest


def load_ava_clips(force: bool = False, limit: int | None = None) -> list[dict]:
    """Prepare or load AVA ActiveSpeaker / AVA Speech clips."""
    prepared = [] if force else load_prepared_clips("ava")
    if prepared:
        return prepared[:limit] if limit is not None else prepared
    return prepare_clips_from_manifest(
        dataset_name="ava",
        manifest_path=DATASET_CONFIG["ava"]["manifest"],
        default_clip_type="speech",
        limit=limit,
        force=force,
    )
