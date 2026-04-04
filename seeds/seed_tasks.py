"""Five seed task definitions with ground truth specifications."""

from dataclasses import dataclass, field


@dataclass
class SeedTask:
    seed_id: str
    name: str
    pillar: str
    failure_mode_id: str
    tier: int
    description: str
    corruption_fn: str
    corruption_params: dict
    ground_truth_type: str  # "exact" | "measured" | "semantic"
    ground_truth_extraction: str
    question_template: str
    signal_metric: str
    pass_criteria: str
    compatible_clip_types: list[str] = field(default_factory=list)


SEED_TASKS: dict[str, SeedTask] = {
    "S1": SeedTask(
        seed_id="S1",
        name="Sync Drift Detection",
        pillar="voice",
        failure_mode_id="lip_sync_drift",
        tier=1,
        description="Detect audio-visual synchronization offset injected by shifting the audio track.",
        corruption_fn="apply_sync_shift",
        corruption_params={"offset_ms": 200},
        ground_truth_type="exact",
        ground_truth_extraction="The injected offset in milliseconds is the ground truth.",
        question_template=(
            "Watch this video clip. Is the audio synchronized with the visual events? "
            "If not, estimate by how much the audio leads or lags the visuals in milliseconds. "
            'Respond with JSON: {{"synced": bool, "offset_ms": int, "confidence": float}}'
        ),
        signal_metric="cross_correlation_lag_ms",
        pass_criteria="Model correctly identifies sync vs. no-sync AND estimates offset within 100ms.",
        compatible_clip_types=["speech", "mixed"],
    ),
    "S2": SeedTask(
        seed_id="S2",
        name="Speaker Identity Consistency",
        pillar="voice",
        failure_mode_id="speaker_identity_break",
        tier=2,
        description="Detect when the speaker identity changes at a cut point.",
        corruption_fn="apply_speaker_swap",
        corruption_params={"swap_point_s": 5.0},
        ground_truth_type="measured",
        ground_truth_extraction="Known swap point and speaker difference (frequency ratio).",
        question_template=(
            "Listen to the speakers in this clip. Does the same person speak throughout, "
            "or does the speaker identity change at some point? If it changes, at approximately "
            "what timestamp? Respond with JSON: "
            '{{"consistent": bool, "change_timestamp_s": float, "confidence": float}}'
        ),
        signal_metric="speaker_embedding_cosine_similarity",
        pass_criteria="Model correctly identifies swap vs. no-swap AND locates swap within 2 seconds.",
        compatible_clip_types=["speech", "mixed"],
    ),
    "S3": SeedTask(
        seed_id="S3",
        name="SFX Onset Misalignment",
        pillar="sfx",
        failure_mode_id="sfx_temporal_misfire",
        tier=1,
        description="Detect sound effects that are shifted in time relative to visual events.",
        corruption_fn="apply_sfx_mistime",
        corruption_params={"shift_ms": 200},
        ground_truth_type="exact",
        ground_truth_extraction="The injected shift amount and original event timestamps.",
        question_template=(
            "Watch and listen to this clip. Are the sound effects properly timed with the "
            "visual events? Identify any sounds that occur too early or too late relative to "
            "what you see. Respond with JSON: "
            '{{"aligned": bool, "misaligned_events": [{{"description": str, "offset_ms": int}}], '
            '"confidence": float}}'
        ),
        signal_metric="onset_detection_delta_ms",
        pass_criteria="Model correctly identifies aligned vs. misaligned.",
        compatible_clip_types=["sfx"],
    ),
    "S4": SeedTask(
        seed_id="S4",
        name="Audio Artifact Detection",
        pillar="voice",
        failure_mode_id="tts_artifacts",
        tier=1,
        description="Detect injected audio artifacts (clicks, dropouts, spectral anomalies).",
        corruption_fn="inject_artifacts",
        corruption_params={"artifact_type": "click", "severity": 0.5},
        ground_truth_type="exact",
        ground_truth_extraction="Exact artifact locations and types as injected.",
        question_template=(
            "Listen to this audio. Are there any artifacts, glitches, clicks, dropouts, or "
            "unnatural sounds? If so, describe them and note approximately when they occur. "
            'Respond with JSON: {{"clean": bool, "artifacts": [{{"type": str, '
            '"timestamp_s": float, "severity": str}}], "confidence": float}}'
        ),
        signal_metric="spectral_flux_anomaly",
        pass_criteria="Model correctly identifies clean vs. corrupted AND locates artifacts within 1 second.",
        compatible_clip_types=["speech", "sfx", "music", "mixed"],
    ),
    "S5": SeedTask(
        seed_id="S5",
        name="Music Mood Alignment",
        pillar="music",
        failure_mode_id="mood_scene_mismatch",
        tier=3,
        description="Detect when background music mood mismatches the visual scene.",
        corruption_fn="apply_music_mood_swap",
        corruption_params={"mood_distance": 0.8},
        ground_truth_type="measured",
        ground_truth_extraction="Mood distance between original and replacement track.",
        question_template=(
            "Watch this scene and listen to the background music. Does the music's mood "
            "(energy, emotion, tempo) match the visual scene? Rate the match from 1 "
            "(completely wrong mood) to 5 (perfect match). Respond with JSON: "
            '{{"mood_match_score": int, "scene_mood": str, "music_mood": str, "explanation": str}}'
        ),
        signal_metric="spectral_centroid_tempo_proxy",
        pass_criteria="Model score correlates with actual mood distance (r > 0.5).",
        compatible_clip_types=["music", "mixed"],
    ),
}
