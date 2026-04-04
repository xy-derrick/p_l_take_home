"""Five rubric dimensions with score definitions."""

RUBRIC_DIMENSIONS = [
    {
        "id": "av_sync",
        "name": "A/V Synchronization Accuracy",
        "tier": 1,
        "scale": "1-5",
        "levels": {
            1: "Severe offset (>500ms), clearly perceivable lag/lead",
            2: "Significant offset (200-500ms), noticeable mismatch",
            3: "Moderate offset (100-200ms), detectable on attention",
            4: "Minor offset (40-100ms), barely perceptible",
            5: "Aligned (<40ms), perceptually synchronous",
        },
        "measurement": "cross_correlation_lag_ms",
        "thresholds_ms": [500, 200, 100, 40],
        "automation": "fully_automatic",
    },
    {
        "id": "artifact_quality",
        "name": "Audio Artifact / Signal Quality",
        "tier": 1,
        "scale": "1-5",
        "levels": {
            1: "Multiple severe artifacts (clicks, dropouts, distortion)",
            2: "Noticeable artifacts that distract from content",
            3: "Minor artifacts, not distracting",
            4: "Very subtle artifacts, only detectable on close listen",
            5: "Clean, no detectable artifacts",
        },
        "measurement": "artifact_count_and_severity",
        "automation": "fully_automatic",
    },
    {
        "id": "speaker_consistency",
        "name": "Speaker Identity Consistency",
        "tier": 2,
        "scale": "1-5",
        "levels": {
            1: "Clearly different speaker (different gender/age)",
            2: "Different speaker (same gender, different voice)",
            3: "Noticeable voice quality shift",
            4: "Minor variation within natural range",
            5: "Fully consistent speaker identity",
        },
        "measurement": "speaker_embedding_cosine_similarity",
        "thresholds": [0.3, 0.5, 0.7, 0.85],
        "automation": "embedding_based",
    },
    {
        "id": "semantic_match",
        "name": "Audio-Visual Semantic Match",
        "tier": 3,
        "scale": "1-5",
        "levels": {
            1: "Audio is completely unrelated to visual content",
            2: "Audio is from wrong category for the visual scene",
            3: "Audio is plausible but not well matched",
            4: "Audio matches well with minor discrepancies",
            5: "Audio perfectly matches the visual content",
        },
        "measurement": "vlm_judgment",
        "automation": "requires_vlm",
    },
    {
        "id": "music_coherence",
        "name": "Music/Mood Coherence",
        "tier": 3,
        "scale": "1-5",
        "levels": {
            1: "Music mood directly contradicts scene emotion",
            2: "Music mood is inappropriate for the scene",
            3: "Music mood is neutral/ambiguous relative to scene",
            4: "Music mood is appropriate with minor issues",
            5: "Music mood perfectly matches and transitions naturally",
        },
        "measurement": "mood_distance + vlm_judgment",
        "automation": "hybrid",
    },
]

RUBRIC_BY_ID = {r["id"]: r for r in RUBRIC_DIMENSIONS}


def score_from_thresholds(value: float, thresholds: list[float],
                          lower_is_worse: bool = True) -> int:
    """Map a measured value to a 1-5 rubric score using threshold breakpoints.

    thresholds should be in descending order (worst to best).
    lower_is_worse=True: high values are bad (e.g., offset in ms).
    lower_is_worse=False: high values are good (e.g., similarity).
    """
    if lower_is_worse:
        # value > t[0] → 1, t[0] > value > t[1] → 2, ...
        for i, t in enumerate(thresholds):
            if value > t:
                return i + 1
        return 5
    else:
        # value < t[0] → 1, t[0] < value < t[1] → 2, ...
        for i, t in enumerate(thresholds):
            if value < t:
                return i + 1
        return 5
