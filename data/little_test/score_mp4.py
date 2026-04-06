"""Score MP4 files in this folder with the LanguageModelScorer across all seed rubrics."""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from expansion.variant_generator import TaskVariant
from scoring.language_model_scorer import LanguageModelScorer
from seeds.seed_tasks import SEED_TASKS

FOLDER = Path(__file__).parent
OUT_TSV = FOLDER / "rubric_eval.tsv"

COLUMNS = [
    "task_id", "seed_id", "dataset", "task_description", "source_clip",
    "visual_context", "audio_pillar", "tier", "corruption_type",
    "corruption_severity", "model",
    "av_sync_score", "artifact_quality_score", "speaker_consistency_score",
    "semantic_match_score", "music_coherence_score",
    "detection_correct", "ground_truth_label",
]

def make_variant(mp4_path: Path, seed_id: str) -> TaskVariant:
    seed = SEED_TASKS[seed_id]
    stem = mp4_path.stem
    param_name = list(seed.corruption_params.keys())[0]
    param_val = seed.corruption_params[param_name]
    return TaskVariant(
        task_id=f"{seed_id}_little_test_{stem}",
        seed_id=seed_id,
        source_clip=stem,
        source_audio_path=str(mp4_path),   # no separate wav; scorer falls back to video
        source_video_path=str(mp4_path),
        source_clip_name=stem,
        source_dataset="little_test",
        clip_type="mixed",
        metadata_path=None,
        segment_start_s=0.0,
        segment_end_s=None,
        visual_context_available=True,
        source_metadata={},
        corruption_type=seed.corruption_fn,
        corruption_params=seed.corruption_params,
        ground_truth={"corruption_type": seed.corruption_fn.replace("apply_", "").replace("inject_", "")},
        corrupted_audio_path=None,
        corrupted_video_path=str(mp4_path),  # use the mp4 directly
        audio_pillar=seed.pillar,
        tier=seed.tier,
        is_clean=False,
    )

def main():
    mp4_files = sorted(FOLDER.glob("*.mp4"))
    if not mp4_files:
        print("No MP4 files found in", FOLDER)
        return

    scorer = LanguageModelScorer()
    rows: list[dict] = []

    for mp4 in mp4_files:
        print(f"\nScoring: {mp4.name}")
        for seed_id, seed in SEED_TASKS.items():
            variant = make_variant(mp4, seed_id)
            print(f"  Seed {seed_id} ({seed.pillar}) ...", end=" ", flush=True)
            scores = scorer.score_variant(variant)
            param_name = list(seed.corruption_params.keys())[0]
            param_val = seed.corruption_params[param_name]
            row = {
                "task_id": variant.task_id,
                "seed_id": seed_id,
                "dataset": "little_test",
                "task_description": f"{seed_id}: {seed.corruption_fn} on little_test/{mp4.name}",
                "source_clip": mp4.stem,
                "visual_context": str(variant.has_visual_context()),
                "audio_pillar": seed.pillar,
                "tier": seed.tier,
                "corruption_type": seed.corruption_fn,
                "corruption_severity": f"{param_name}={param_val}",
                "model": "gemini_2_5_flash" if not scorer.use_mock else "mock",
                "av_sync_score": scores["av_sync_score"],
                "artifact_quality_score": scores["artifact_quality_score"],
                "speaker_consistency_score": scores["speaker_consistency_score"],
                "semantic_match_score": scores["semantic_match_score"],
                "music_coherence_score": scores["music_coherence_score"],
                "detection_correct": scores["detection_correct"],
                "ground_truth_label": seed.corruption_fn.replace("apply_", "").replace("inject_", ""),
            }
            rows.append(row)
            print(
                f"av_sync={row['av_sync_score']} artifact={row['artifact_quality_score']} "
                f"speaker={row['speaker_consistency_score']} semantic={row['semantic_match_score']} "
                f"music={row['music_coherence_score']} detected={row['detection_correct']}"
            )

    with open(OUT_TSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows -> {OUT_TSV}")

if __name__ == "__main__":
    main()
