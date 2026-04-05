"""Reconstruct benchmark outputs from a judge score JSONL log."""

from __future__ import annotations

import json
from pathlib import Path

from expansion.variant_generator import TaskVariant
from seeds.seed_tasks import SEED_TASKS
from utils import safe_slug


def derive_log_prefix(log_path: str | Path) -> str:
    """Derive an output prefix from a judge log filename."""
    stem = Path(log_path).stem
    if stem == "latest_judge_scores":
        return "latest"
    if stem.endswith("_latest_judge_scores"):
        stem = stem[: -len("_latest_judge_scores")]
    elif stem.startswith("judge_scores_"):
        stem = stem[len("judge_scores_"):]
    return safe_slug(stem)


def load_replay_bundle(log_path: str | Path) -> tuple[list[TaskVariant], dict[str, dict], dict[str, dict]]:
    """Load variants and per-model score maps from a judge log."""
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"Judge log not found: {path}")

    variants_by_task: dict[str, TaskVariant] = {}
    signal_results: dict[str, dict] = {}
    language_model_results: dict[str, dict] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc

            task_id = str(entry.get("task_id", "")).strip()
            if not task_id:
                raise ValueError(f"Missing task_id on line {line_number} of {path}")

            if task_id not in variants_by_task:
                variants_by_task[task_id] = _variant_from_entry(entry)

            model_name = str(entry.get("model", "")).strip().lower()
            scores = entry.get("scores", {}) or {}
            if model_name.startswith("signal"):
                signal_results[task_id] = scores
            else:
                language_model_results[task_id] = scores

    return list(variants_by_task.values()), signal_results, language_model_results


def _variant_from_entry(entry: dict) -> TaskVariant:
    ground_truth = dict(entry.get("ground_truth", {}) or {})
    seed_id = str(entry.get("seed_id", "")).strip()
    corruption_type = str(
        entry.get("corruption_type")
        or ground_truth.get("corruption_type")
        or ("none" if entry.get("is_clean") else "")
    ).strip()

    source_audio_path = (
        entry.get("source_audio_path")
        or ground_truth.get("original_audio_path")
        or entry.get("corrupted_audio_path")
        or ground_truth.get("corrupted_audio_path")
        or ground_truth.get("corrupted_path")
        or ""
    )
    source_video_path = (
        entry.get("source_video_path")
        or ground_truth.get("original_video_path")
        or entry.get("corrupted_video_path")
        or ground_truth.get("corrupted_video_path")
    )
    source_clip_name = str(entry.get("source_clip") or Path(str(source_audio_path)).stem or entry["task_id"])
    visual_context = bool(entry.get("visual_context", False))
    is_clean = bool(entry.get("is_clean", False) or corruption_type == "none")

    return TaskVariant(
        task_id=str(entry["task_id"]),
        seed_id=seed_id,
        source_clip=str(source_audio_path or source_clip_name),
        source_audio_path=str(source_audio_path),
        source_video_path=str(source_video_path) if source_video_path else None,
        source_clip_name=source_clip_name,
        source_dataset=str(entry.get("dataset", "synthetic")),
        clip_type=str(entry.get("clip_type", "unknown")),
        metadata_path=None,
        segment_start_s=0.0,
        segment_end_s=None,
        visual_context_available=visual_context,
        source_metadata={},
        corruption_type=corruption_type,
        corruption_params=dict(entry.get("corruption_params", {}) or {}),
        ground_truth=ground_truth,
        corrupted_audio_path=_optional_str(
            entry.get("corrupted_audio_path")
            or ground_truth.get("corrupted_audio_path")
            or ground_truth.get("corrupted_path")
        ),
        corrupted_video_path=_optional_str(
            entry.get("corrupted_video_path") or ground_truth.get("corrupted_video_path")
        ),
        audio_pillar=_audio_pillar_from_seed(seed_id),
        tier=int(entry.get("tier", _tier_from_seed(seed_id, 1))),
        difficulty_estimate=str(entry.get("difficulty_estimate", "medium")),
        is_clean=is_clean,
    )


def _audio_pillar_from_seed(seed_id: str) -> str:
    pillars = []
    for part in seed_id.split("+"):
        seed = SEED_TASKS.get(part)
        if seed is not None and seed.pillar not in pillars:
            pillars.append(seed.pillar)
    return "+".join(pillars)


def _tier_from_seed(seed_id: str, fallback: int) -> int:
    tiers = [SEED_TASKS[part].tier for part in seed_id.split("+") if part in SEED_TASKS]
    return max(tiers) if tiers else fallback


def _optional_str(value) -> str | None:
    if value in (None, ""):
        return None
    return str(value)
