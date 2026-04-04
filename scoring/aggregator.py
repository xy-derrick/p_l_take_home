"""Combine scores and produce per-task structured output."""

from __future__ import annotations

import csv
import os
from dataclasses import asdict

from config import OUTPUT_TSV


def aggregate_results(variants: list, signal_results: dict, gemini_results: dict,
                      output_path: str | None = None) -> list[dict]:
    """Combine signal and gemini scores into a flat table.

    Each variant gets TWO rows (one per model).
    Returns list of row dicts and writes TSV.
    """
    if output_path is None:
        output_path = OUTPUT_TSV

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    columns = [
        "task_id", "seed_id", "task_description", "source_clip", "audio_pillar",
        "tier", "corruption_type", "corruption_severity", "model",
        "av_sync_score", "artifact_quality_score", "speaker_consistency_score",
        "semantic_match_score", "music_coherence_score",
        "detection_correct", "ground_truth_label",
    ]

    rows = []
    for v in variants:
        gt = v.ground_truth or {}
        corruption_type = gt.get("corruption_type", "none")
        # Extract a severity value
        severity = _extract_severity(v)
        gt_label = "clean" if v.is_clean else corruption_type
        description = f"{v.seed_id}: {corruption_type} on {v.source_clip_name}"

        for model_name, results in [("signal_pipeline", signal_results),
                                     ("gemini_flash", gemini_results)]:
            scores = results.get(v.task_id, {})
            row = {
                "task_id": v.task_id,
                "seed_id": v.seed_id,
                "task_description": description,
                "source_clip": v.source_clip_name,
                "audio_pillar": v.audio_pillar,
                "tier": v.tier,
                "corruption_type": corruption_type,
                "corruption_severity": severity,
                "model": model_name,
                "av_sync_score": scores.get("av_sync_score", 5),
                "artifact_quality_score": scores.get("artifact_quality_score", 5),
                "speaker_consistency_score": scores.get("speaker_consistency_score", 5),
                "semantic_match_score": scores.get("semantic_match_score", 5),
                "music_coherence_score": scores.get("music_coherence_score", 5),
                "detection_correct": scores.get("detection_correct", False),
                "ground_truth_label": gt_label,
            }
            rows.append(row)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    return rows


def _extract_severity(variant) -> str:
    """Extract a human-readable severity string from variant params."""
    params = variant.corruption_params
    if not params:
        return "clean"
    for key in ["offset_ms", "shift_ms", "severity", "mood_distance", "replacement_freq"]:
        if key in params:
            return f"{key}={params[key]}"
    return str(params)
