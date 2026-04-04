"""Signal processing measurement pipeline.

Uses deterministic computations on synthetic audio for the prototype.
Designed with clear interfaces so real models (Whisper, pyannote, CLAP)
can be plugged in later.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

from scoring.rubric import score_from_thresholds, RUBRIC_BY_ID


class SignalScorer:
    """Scores audio using signal processing techniques (mock-precise for prototype)."""

    def score_variant(self, variant) -> dict:
        """Score a single task variant. Dispatches to the appropriate method."""
        gt = variant.ground_truth
        ctype = gt.get("corruption_type", "none") if gt else "none"

        scores = {
            "av_sync_score": 5,
            "artifact_quality_score": 5,
            "speaker_consistency_score": 5,
            "semantic_match_score": 5,
            "music_coherence_score": 5,
            "detection_correct": False,
            "raw_measurements": {},
        }

        if variant.is_clean:
            # Clean clip — signal pipeline should mostly say "fine"
            rng = np.random.default_rng(hash(variant.task_id) % 2**31)
            # ~3% false positive rate
            if rng.random() < 0.03:
                scores["detection_correct"] = False
                scores["av_sync_score"] = 4
            else:
                scores["detection_correct"] = True
            return scores

        if ctype == "sync_shift":
            return self._score_sync(variant, scores)
        elif ctype == "speaker_swap":
            return self._score_speaker(variant, scores)
        elif ctype == "artifact_inject":
            return self._score_artifacts(variant, scores)
        elif ctype == "sfx_mistime":
            return self._score_sfx_timing(variant, scores)
        elif ctype == "music_mood_swap":
            return self._score_mood(variant, scores)

        return scores

    def _score_sync(self, variant, scores: dict) -> dict:
        gt = variant.ground_truth
        true_offset = abs(gt.get("offset_ms", 0))
        # Signal pipeline: very precise, ±5ms noise
        rng = np.random.default_rng(hash(variant.task_id) % 2**31)
        measured_offset = true_offset + rng.normal(0, 5)
        measured_offset = max(0, measured_offset)

        thresholds = RUBRIC_BY_ID["av_sync"]["thresholds_ms"]
        score = score_from_thresholds(measured_offset, thresholds, lower_is_worse=True)
        scores["av_sync_score"] = score
        scores["raw_measurements"]["offset_ms"] = float(measured_offset)
        # Detection: 90-98% for signal pipeline
        detection_prob = 0.98 if true_offset > 100 else (0.90 if true_offset > 40 else 0.70)
        scores["detection_correct"] = rng.random() < detection_prob
        return scores

    def _score_speaker(self, variant, scores: dict) -> dict:
        gt = variant.ground_truth
        similarity = gt.get("similarity_score", 1.0)
        rng = np.random.default_rng(hash(variant.task_id) % 2**31)
        # Signal: embedding similarity with small noise
        measured_sim = similarity + rng.normal(0, 0.05)
        measured_sim = np.clip(measured_sim, 0, 1)

        thresholds = RUBRIC_BY_ID["speaker_consistency"]["thresholds"]
        score = score_from_thresholds(float(measured_sim), thresholds, lower_is_worse=False)
        scores["speaker_consistency_score"] = score
        scores["raw_measurements"]["similarity"] = float(measured_sim)
        # Detection: 70-85%
        detection_prob = 0.85 if similarity < 0.5 else (0.75 if similarity < 0.7 else 0.60)
        scores["detection_correct"] = rng.random() < detection_prob
        return scores

    def _score_artifacts(self, variant, scores: dict) -> dict:
        gt = variant.ground_truth
        artifacts = gt.get("artifacts", [])
        severity = gt.get("severity", 0.5)
        rng = np.random.default_rng(hash(variant.task_id) % 2**31)

        n_artifacts = len(artifacts)
        # Signal: detect most artifacts with high precision
        detected = max(0, n_artifacts - (1 if rng.random() < 0.1 else 0))

        if n_artifacts == 0:
            score = 5
        elif severity > 0.7:
            score = 1
        elif severity > 0.4:
            score = 2
        elif severity > 0.2:
            score = 3
        else:
            score = 4

        scores["artifact_quality_score"] = score
        scores["raw_measurements"]["artifacts_detected"] = detected
        scores["raw_measurements"]["artifacts_total"] = n_artifacts
        detection_prob = 0.95 if severity > 0.3 else 0.85
        scores["detection_correct"] = rng.random() < detection_prob
        return scores

    def _score_sfx_timing(self, variant, scores: dict) -> dict:
        gt = variant.ground_truth
        shift_ms = abs(gt.get("shift_ms", 0))
        rng = np.random.default_rng(hash(variant.task_id) % 2**31)
        measured_shift = shift_ms + rng.normal(0, 5)
        measured_shift = max(0, measured_shift)

        thresholds = [500, 200, 100, 50]
        score = score_from_thresholds(measured_shift, thresholds, lower_is_worse=True)
        scores["av_sync_score"] = score  # SFX timing maps to av_sync dimension
        scores["raw_measurements"]["shift_ms"] = float(measured_shift)
        detection_prob = 0.95 if shift_ms > 100 else 0.80
        scores["detection_correct"] = rng.random() < detection_prob
        return scores

    def _score_mood(self, variant, scores: dict) -> dict:
        gt = variant.ground_truth
        mood_dist = gt.get("mood_distance", 0.0)
        rng = np.random.default_rng(hash(variant.task_id) % 2**31)

        # Signal pipeline is poor at semantic mood (correlation ~0.3)
        noise = rng.normal(0, 0.3)
        measured_dist = np.clip(mood_dist + noise, 0, 1)

        # Map distance to score (high distance = low score)
        if measured_dist > 0.8:
            score = 1
        elif measured_dist > 0.6:
            score = 2
        elif measured_dist > 0.4:
            score = 3
        elif measured_dist > 0.2:
            score = 4
        else:
            score = 5

        scores["music_coherence_score"] = score
        scores["raw_measurements"]["mood_distance"] = float(measured_dist)
        # Signal pipeline: poor at semantic (30-50%)
        detection_prob = 0.50 if mood_dist > 0.6 else 0.30
        scores["detection_correct"] = rng.random() < detection_prob
        return scores
