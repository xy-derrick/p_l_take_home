"""Parameterized expansion from seed tasks to 500+ variants."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from config import SEVERITY_LEVELS
from seeds.seed_tasks import SEED_TASKS, SeedTask


@dataclass
class TaskVariant:
    task_id: str
    seed_id: str
    source_clip: str
    source_audio_path: str
    source_video_path: str | None
    source_clip_name: str
    source_dataset: str
    clip_type: str
    metadata_path: str | None
    segment_start_s: float
    segment_end_s: float | None
    visual_context_available: bool
    source_metadata: dict
    corruption_type: str
    corruption_params: dict
    ground_truth: dict = field(default_factory=dict)
    corrupted_audio_path: str | None = None
    corrupted_video_path: str | None = None
    audio_pillar: str = ""
    tier: int = 1
    difficulty_estimate: str = "medium"
    is_clean: bool = False

    def has_visual_context(self) -> bool:
        """Whether the variant has meaningful video context for the language model."""
        return bool(self.source_video_path and self.visual_context_available)


SEED_SEVERITY_MAP: dict[str, tuple[str, list]] = {
    "S1": ("offset_ms", SEVERITY_LEVELS["sync_drift_ms"]),
    "S2": ("replacement_freq", [120, 140, 160, 180, 200, 240, 280, 350, 450, 550]),
    "S3": ("shift_ms", SEVERITY_LEVELS["sfx_shift_ms"] + [750, 1500, 2000]),
    "S4": ("severity", [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0]),
    "S5": ("mood_distance", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
}


def _difficulty_from_severity(seed_id: str, severity_value: float) -> str:
    """Estimate difficulty based on severity level."""
    _, levels = SEED_SEVERITY_MAP[seed_id]
    if not levels:
        return "medium"
    idx = levels.index(severity_value) if severity_value in levels else 0
    fraction = idx / max(1, len(levels) - 1)
    if fraction < 0.33:
        return "hard"
    if fraction < 0.66:
        return "medium"
    return "easy"


def generate_variants(
    source_clips: list[dict],
    seeds: dict[str, SeedTask] | None = None,
) -> list[TaskVariant]:
    """Expand seed tasks into full variant families."""
    if seeds is None:
        seeds = SEED_TASKS

    variants: list[TaskVariant] = []
    clip_by_type: dict[str, list[dict]] = {}
    for clip in source_clips:
        clip_by_type.setdefault(clip["type"], []).append(clip)

    for seed_id, seed in seeds.items():
        param_name, severity_values = SEED_SEVERITY_MAP[seed_id]

        compatible_clips = []
        for clip_type in seed.compatible_clip_types:
            compatible_clips.extend(clip_by_type.get(clip_type, []))

        for clip in compatible_clips:
            for severity in severity_values:
                task_id = f"{seed_id}_{clip['dataset']}_{clip['name']}_{param_name}{severity}"
                params = dict(seed.corruption_params)
                params[param_name] = severity
                variants.append(
                    TaskVariant(
                        task_id=task_id,
                        seed_id=seed_id,
                        source_clip=clip["path"],
                        source_audio_path=clip.get("source_audio_path", clip["path"]),
                        source_video_path=clip.get("source_video_path"),
                        source_clip_name=clip["name"],
                        source_dataset=clip.get("dataset", "synthetic"),
                        clip_type=clip["type"],
                        metadata_path=clip.get("metadata_path"),
                        segment_start_s=clip.get("segment_start_s", 0.0),
                        segment_end_s=clip.get("segment_end_s"),
                        visual_context_available=bool(clip.get("visual_context_available", False)),
                        source_metadata=dict(clip.get("metadata", {})),
                        corruption_type=seed.corruption_fn,
                        corruption_params=params,
                        audio_pillar=seed.pillar,
                        tier=seed.tier,
                        difficulty_estimate=_difficulty_from_severity(seed_id, severity),
                    )
                )

            clean_id = f"{seed_id}_{clip['dataset']}_{clip['name']}_clean"
            variants.append(
                TaskVariant(
                    task_id=clean_id,
                    seed_id=seed_id,
                    source_clip=clip["path"],
                    source_audio_path=clip.get("source_audio_path", clip["path"]),
                    source_video_path=clip.get("source_video_path"),
                    source_clip_name=clip["name"],
                    source_dataset=clip.get("dataset", "synthetic"),
                    clip_type=clip["type"],
                    metadata_path=clip.get("metadata_path"),
                    segment_start_s=clip.get("segment_start_s", 0.0),
                    segment_end_s=clip.get("segment_end_s"),
                    visual_context_available=bool(clip.get("visual_context_available", False)),
                    source_metadata=dict(clip.get("metadata", {})),
                    corruption_type="none",
                    corruption_params={},
                    audio_pillar=seed.pillar,
                    tier=seed.tier,
                    difficulty_estimate="easy",
                    is_clean=True,
                )
            )

    return variants


def generate_cross_pillar_variants(base_variants: list[TaskVariant]) -> list[TaskVariant]:
    """Combine corruptions from different seeds on the same source clip."""
    rng = Random(42)

    by_clip: dict[str, list[TaskVariant]] = {}
    for variant in base_variants:
        if variant.is_clean:
            continue
        key = f"{variant.source_dataset}:{variant.source_clip_name}"
        by_clip.setdefault(key, []).append(variant)

    cross_variants: list[TaskVariant] = []
    for clip_key, clip_variants in by_clip.items():
        by_seed: dict[str, list[TaskVariant]] = {}
        for variant in clip_variants:
            by_seed.setdefault(variant.seed_id, []).append(variant)

        seed_ids = list(by_seed.keys())
        if len(seed_ids) < 2:
            continue

        for i in range(len(seed_ids)):
            for j in range(i + 1, len(seed_ids)):
                first = rng.choice(by_seed[seed_ids[i]])
                second = rng.choice(by_seed[seed_ids[j]])
                combined_id = f"CROSS_{first.task_id}_x_{second.seed_id}"
                cross_variants.append(
                    TaskVariant(
                        task_id=combined_id,
                        seed_id=f"{first.seed_id}+{second.seed_id}",
                        source_clip=first.source_clip,
                        source_audio_path=first.source_audio_path,
                        source_video_path=first.source_video_path,
                        source_clip_name=first.source_clip_name,
                        source_dataset=first.source_dataset,
                        clip_type=first.clip_type,
                        metadata_path=first.metadata_path,
                        segment_start_s=first.segment_start_s,
                        segment_end_s=first.segment_end_s,
                        visual_context_available=first.visual_context_available,
                        source_metadata=dict(first.source_metadata),
                        corruption_type=f"{first.corruption_type}+{second.corruption_type}",
                        corruption_params={**first.corruption_params, **second.corruption_params},
                        audio_pillar=f"{first.audio_pillar}+{second.audio_pillar}",
                        tier=max(first.tier, second.tier),
                        difficulty_estimate="hard",
                    )
                )

    if len(cross_variants) > 50:
        cross_variants = rng.sample(cross_variants, 50)

    return cross_variants
