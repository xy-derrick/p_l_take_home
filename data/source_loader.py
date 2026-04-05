"""Source clip loading across synthetic and real-video datasets."""

from __future__ import annotations

from collections import defaultdict

from config import DEFAULT_DATASETS, MAX_CLIPS_PER_DATASET, SUPPORTED_DATASETS
from data.generate_source_clips import generate_all
from data.import_ava import load_ava_clips
from data.import_condensed_movies import load_condensed_movies_clips
from data.import_greatest_hits import load_greatest_hits_clips


DATASET_IMPORTERS = {
    "ava": load_ava_clips,
    "greatest_hits": load_greatest_hits_clips,
    "condensed_movies": load_condensed_movies_clips,
}


def load_source_clips(
    datasets: list[str] | None = None,
    force_prepare: bool = False,
    limit_per_dataset: int | None = None,
    allow_synthetic_fallback: bool = True,
) -> list[dict]:
    """Load prepared real-video clips plus synthetic fallback clips."""
    requested = list(datasets or DEFAULT_DATASETS)
    limit_per_dataset = limit_per_dataset or MAX_CLIPS_PER_DATASET

    unknown = [name for name in requested if name not in SUPPORTED_DATASETS]
    if unknown:
        raise ValueError(f"Unsupported dataset(s): {', '.join(sorted(unknown))}")

    clips: list[dict] = []
    for dataset_name in requested:
        if dataset_name == "synthetic":
            clips.extend(generate_all())
            continue

        importer = DATASET_IMPORTERS[dataset_name]
        clips.extend(importer(force=force_prepare, limit=limit_per_dataset))

    if not clips and allow_synthetic_fallback:
        clips.extend(generate_all())

    return clips


def summarize_clips(clips: list[dict]) -> dict[str, int]:
    """Count clips by dataset for CLI output."""
    counts: dict[str, int] = defaultdict(int)
    for clip in clips:
        counts[clip.get("dataset", "unknown")] += 1
    return dict(sorted(counts.items()))
