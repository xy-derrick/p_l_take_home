"""Parameterized expansion from seed tasks to 500+ variants."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict

from seeds.seed_tasks import SeedTask, SEED_TASKS
from config import SEVERITY_LEVELS


@dataclass
class TaskVariant:
    task_id: str
    seed_id: str
    source_clip: str
    source_clip_name: str
    corruption_type: str
    corruption_params: dict
    ground_truth: dict = field(default_factory=dict)
    audio_pillar: str = ""
    tier: int = 1
    difficulty_estimate: str = "medium"
    is_clean: bool = False


# Map seed -> severity parameter name and values
SEED_SEVERITY_MAP: dict[str, tuple[str, list]] = {
    "S1": ("offset_ms", SEVERITY_LEVELS["sync_drift_ms"]),
    "S2": ("replacement_freq", [120, 140, 160, 180, 200, 240, 280, 350, 450, 550]),
    "S3": ("shift_ms", SEVERITY_LEVELS["sfx_shift_ms"] + [750, 1000, 1500, 2000]),
    "S4": ("severity", [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0]),
    "S5": ("mood_distance", SEVERITY_LEVELS["mood_distance"] + [0.1, 0.3, 0.5, 0.7, 0.9]),
}

# Map seed -> compatible clip types
SEED_CLIP_TYPES: dict[str, list[str]] = {
    "S1": ["speech", "mixed"],
    "S2": ["speech", "mixed"],
    "S3": ["sfx"],
    "S4": ["speech", "sfx", "music", "mixed"],
    "S5": ["music", "mixed", "speech"],
}


def _difficulty_from_severity(seed_id: str, severity_value: float) -> str:
    """Estimate difficulty based on severity level."""
    param_name, levels = SEED_SEVERITY_MAP[seed_id]
    if not levels:
        return "medium"
    idx = levels.index(severity_value) if severity_value in levels else 0
    fraction = idx / max(1, len(levels) - 1)
    if fraction < 0.33:
        return "hard"  # subtle corruption = hard to detect
    elif fraction < 0.66:
        return "medium"
    return "easy"  # severe corruption = easy to detect


def generate_variants(source_clips: list[dict],
                      seeds: dict[str, SeedTask] | None = None) -> list[TaskVariant]:
    """Expand seed tasks into full variant families.

    Returns list of TaskVariant (not yet corrupted — ground_truth is empty).
    """
    if seeds is None:
        seeds = SEED_TASKS

    variants: list[TaskVariant] = []
    clip_by_type: dict[str, list[dict]] = {}
    for clip in source_clips:
        clip_by_type.setdefault(clip["type"], []).append(clip)

    for seed_id, seed in seeds.items():
        param_name, severity_values = SEED_SEVERITY_MAP[seed_id]
        compatible_types = SEED_CLIP_TYPES[seed_id]

        compatible_clips = []
        for ct in compatible_types:
            compatible_clips.extend(clip_by_type.get(ct, []))

        for clip in compatible_clips:
            for sev in severity_values:
                task_id = f"{seed_id}_{clip['name']}_{param_name}{sev}"
                params = dict(seed.corruption_params)
                params[param_name] = sev
                variants.append(TaskVariant(
                    task_id=task_id,
                    seed_id=seed_id,
                    source_clip=clip["path"],
                    source_clip_name=clip["name"],
                    corruption_type=seed.corruption_fn,
                    corruption_params=params,
                    audio_pillar=seed.pillar,
                    tier=seed.tier,
                    difficulty_estimate=_difficulty_from_severity(seed_id, sev),
                ))

            # Add a clean control variant (~20% ratio handled by adding one clean per clip)
            clean_id = f"{seed_id}_{clip['name']}_clean"
            variants.append(TaskVariant(
                task_id=clean_id,
                seed_id=seed_id,
                source_clip=clip["path"],
                source_clip_name=clip["name"],
                corruption_type="none",
                corruption_params={},
                audio_pillar=seed.pillar,
                tier=seed.tier,
                difficulty_estimate="easy",
                is_clean=True,
            ))

    return variants


def generate_cross_pillar_variants(base_variants: list[TaskVariant]) -> list[TaskVariant]:
    """Combine corruptions from different pillars on the same clip.

    Select ~50 combinations: pick pairs from different seeds on overlapping clips.
    """
    import random
    random.seed(42)

    # Group non-clean variants by source clip
    by_clip: dict[str, list[TaskVariant]] = {}
    for v in base_variants:
        if not v.is_clean:
            by_clip.setdefault(v.source_clip_name, []).append(v)

    cross_variants: list[TaskVariant] = []
    for clip_name, clip_vars in by_clip.items():
        # Get variants from different seeds
        by_seed: dict[str, list[TaskVariant]] = {}
        for v in clip_vars:
            by_seed.setdefault(v.seed_id, []).append(v)

        seed_ids = list(by_seed.keys())
        if len(seed_ids) < 2:
            continue

        for i in range(len(seed_ids)):
            for j in range(i + 1, len(seed_ids)):
                v1 = random.choice(by_seed[seed_ids[i]])
                v2 = random.choice(by_seed[seed_ids[j]])
                combined_id = f"CROSS_{v1.task_id}_x_{v2.seed_id}"
                combined_params = {**v1.corruption_params, **v2.corruption_params}
                cross_variants.append(TaskVariant(
                    task_id=combined_id,
                    seed_id=f"{v1.seed_id}+{v2.seed_id}",
                    source_clip=v1.source_clip,
                    source_clip_name=clip_name,
                    corruption_type=f"{v1.corruption_type}+{v2.corruption_type}",
                    corruption_params=combined_params,
                    audio_pillar=f"{v1.audio_pillar}+{v2.audio_pillar}",
                    tier=max(v1.tier, v2.tier),
                    difficulty_estimate="hard",
                ))

    # Cap at ~50
    if len(cross_variants) > 50:
        cross_variants = random.sample(cross_variants, 50)

    return cross_variants
