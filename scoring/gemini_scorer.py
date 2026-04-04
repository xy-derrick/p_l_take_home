"""Gemini 2.5 Flash scorer via OpenRouter API, with mock fallback."""

from __future__ import annotations

import base64
import json
import os
import time

import numpy as np

from config import OPENROUTER_API_KEY, GEMINI_MODEL, OPENROUTER_BASE_URL
from seeds.seed_tasks import SEED_TASKS
from scoring.rubric import RUBRIC_DIMENSIONS


class GeminiScorer:
    """Scores audio quality using Gemini 2.5 Flash via OpenRouter (or mock)."""

    def __init__(self, force_mock: bool = False):
        self.api_key = OPENROUTER_API_KEY
        self.use_mock = force_mock or not self.api_key

    def score_variant(self, variant) -> dict:
        """Score a single task variant."""
        if self.use_mock:
            return self._mock_response(variant)
        return self._api_response(variant)

    # ---- API mode ----

    def _api_response(self, variant) -> dict:
        import requests

        seed = SEED_TASKS.get(variant.seed_id.split("+")[0])
        if seed is None:
            return self._mock_response(variant)

        prompt = self._build_prompt(variant, seed)

        # Encode audio
        audio_path = variant.ground_truth.get("corrupted_path") or variant.source_clip
        try:
            with open(audio_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()
        except FileNotFoundError:
            return self._mock_response(variant)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": "wav",
                        },
                    },
                ],
            }
        ]

        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": GEMINI_MODEL,
                        "messages": messages,
                        "max_tokens": 1024,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                return self._parse_response(text, variant)
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                print(f"  [Gemini API error for {variant.task_id}: {e}, falling back to mock]")
                return self._mock_response(variant)

    def _build_prompt(self, variant, seed) -> str:
        rubric_text = "\n".join(
            f"- {r['name']}: {r['levels']}" for r in RUBRIC_DIMENSIONS
        )
        return (
            f"You are evaluating audio quality for AI-generated video.\n\n"
            f"Task: {seed.description}\n\n"
            f"{seed.question_template}\n\n"
            f"Also rate the overall audio on these rubric dimensions (1-5 each):\n"
            f"{rubric_text}\n\n"
            f"Include rubric scores in your JSON response as: "
            f'"av_sync_score", "artifact_quality_score", "speaker_consistency_score", '
            f'"semantic_match_score", "music_coherence_score"'
        )

    def _parse_response(self, text: str, variant) -> dict:
        """Extract structured scores from model output."""
        scores = {
            "av_sync_score": 5,
            "artifact_quality_score": 5,
            "speaker_consistency_score": 5,
            "semantic_match_score": 5,
            "music_coherence_score": 5,
            "detection_correct": False,
            "raw_response": text,
        }
        # Try to find JSON in response
        try:
            # Find JSON block
            start = text.index("{")
            end = text.rindex("}") + 1
            parsed = json.loads(text[start:end])
            for key in ["av_sync_score", "artifact_quality_score",
                        "speaker_consistency_score", "semantic_match_score",
                        "music_coherence_score"]:
                if key in parsed:
                    scores[key] = int(parsed[key])
            # Infer detection from response
            if "synced" in parsed:
                scores["detection_correct"] = not parsed.get("synced", True) if not variant.is_clean else parsed.get("synced", True)
            elif "consistent" in parsed:
                scores["detection_correct"] = not parsed.get("consistent", True) if not variant.is_clean else parsed.get("consistent", True)
            elif "aligned" in parsed:
                scores["detection_correct"] = not parsed.get("aligned", True) if not variant.is_clean else parsed.get("aligned", True)
            elif "clean" in parsed:
                scores["detection_correct"] = not parsed.get("clean", True) if not variant.is_clean else parsed.get("clean", True)
            elif "mood_match_score" in parsed:
                ms = parsed["mood_match_score"]
                scores["detection_correct"] = (ms <= 3) if not variant.is_clean else (ms >= 4)
        except (ValueError, json.JSONDecodeError):
            pass
        return scores

    # ---- Mock mode ----

    def _mock_response(self, variant) -> dict:
        """Generate simulated response based on expected Gemini behavior."""
        rng = np.random.default_rng(hash(variant.task_id) % 2**31 + 1)
        gt = variant.ground_truth or {}
        ctype = gt.get("corruption_type", "none")

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
            # ~10% false positive rate for Gemini
            if rng.random() < 0.10:
                scores["detection_correct"] = False
                dim = rng.choice(["av_sync_score", "artifact_quality_score"])
                scores[dim] = 4
            else:
                scores["detection_correct"] = True
            return scores

        if ctype == "sync_shift":
            return self._mock_sync(variant, scores, rng)
        elif ctype == "speaker_swap":
            return self._mock_speaker(variant, scores, rng)
        elif ctype == "artifact_inject":
            return self._mock_artifacts(variant, scores, rng)
        elif ctype == "sfx_mistime":
            return self._mock_sfx(variant, scores, rng)
        elif ctype == "music_mood_swap":
            return self._mock_mood(variant, scores, rng)

        return scores

    def _mock_sync(self, variant, scores, rng) -> dict:
        true_offset = abs(variant.ground_truth.get("offset_ms", 0))
        # VLM: imprecise timing, ±200ms noise
        measured = true_offset + rng.normal(0, 200)
        measured = max(0, measured)

        # Detection rate by severity
        if true_offset < 100:
            det_prob = 0.40 + rng.uniform(0, 0.20)
        elif true_offset < 300:
            det_prob = 0.70 + rng.uniform(0, 0.15)
        else:
            det_prob = 0.90 + rng.uniform(0, 0.08)

        scores["detection_correct"] = rng.random() < det_prob
        # Score based on noisy measurement
        if measured > 500:
            scores["av_sync_score"] = 1
        elif measured > 200:
            scores["av_sync_score"] = 2
        elif measured > 100:
            scores["av_sync_score"] = 3
        elif measured > 40:
            scores["av_sync_score"] = 4
        else:
            scores["av_sync_score"] = 5
        scores["raw_measurements"]["measured_offset_ms"] = float(measured)
        return scores

    def _mock_speaker(self, variant, scores, rng) -> dict:
        similarity = variant.ground_truth.get("similarity_score", 1.0)
        if similarity < 0.5:
            det_prob = 0.80 + rng.uniform(0, 0.10)
        elif similarity < 0.7:
            det_prob = 0.60 + rng.uniform(0, 0.10)
        else:
            det_prob = 0.50 + rng.uniform(0, 0.10)

        scores["detection_correct"] = rng.random() < det_prob
        if similarity < 0.3:
            scores["speaker_consistency_score"] = 1
        elif similarity < 0.5:
            scores["speaker_consistency_score"] = 2
        elif similarity < 0.7:
            scores["speaker_consistency_score"] = 3
        elif similarity < 0.85:
            scores["speaker_consistency_score"] = 4
        else:
            scores["speaker_consistency_score"] = 5
        # Add ±2s timestamp noise
        swap_point = variant.ground_truth.get("swap_point_s", 5.0)
        scores["raw_measurements"]["estimated_swap_s"] = swap_point + rng.normal(0, 2)
        return scores

    def _mock_artifacts(self, variant, scores, rng) -> dict:
        severity = variant.ground_truth.get("severity", 0.5)
        if severity < 0.3:
            det_prob = 0.40 + rng.uniform(0, 0.20)
        elif severity < 0.6:
            det_prob = 0.70 + rng.uniform(0, 0.15)
        else:
            det_prob = 0.90 + rng.uniform(0, 0.08)

        scores["detection_correct"] = rng.random() < det_prob
        if severity > 0.7:
            scores["artifact_quality_score"] = 1
        elif severity > 0.4:
            scores["artifact_quality_score"] = 2
        elif severity > 0.2:
            scores["artifact_quality_score"] = 3
        else:
            scores["artifact_quality_score"] = 4
        return scores

    def _mock_sfx(self, variant, scores, rng) -> dict:
        shift_ms = abs(variant.ground_truth.get("shift_ms", 0))
        measured = shift_ms + rng.normal(0, 200)
        measured = max(0, measured)

        if shift_ms < 100:
            det_prob = 0.40 + rng.uniform(0, 0.20)
        elif shift_ms < 300:
            det_prob = 0.70 + rng.uniform(0, 0.15)
        else:
            det_prob = 0.90 + rng.uniform(0, 0.08)

        scores["detection_correct"] = rng.random() < det_prob
        if measured > 500:
            scores["av_sync_score"] = 1
        elif measured > 200:
            scores["av_sync_score"] = 2
        elif measured > 100:
            scores["av_sync_score"] = 3
        elif measured > 50:
            scores["av_sync_score"] = 4
        else:
            scores["av_sync_score"] = 5
        return scores

    def _mock_mood(self, variant, scores, rng) -> dict:
        mood_dist = variant.ground_truth.get("mood_distance", 0.0)
        # VLMs are relatively good at semantic mood (correlation ~0.6-0.7)
        noise = rng.normal(0, 0.15)
        measured = np.clip(mood_dist + noise, 0, 1)

        if mood_dist > 0.6:
            det_prob = 0.60 + rng.uniform(0, 0.20)
        elif mood_dist > 0.3:
            det_prob = 0.50 + rng.uniform(0, 0.15)
        else:
            det_prob = 0.40 + rng.uniform(0, 0.10)

        scores["detection_correct"] = rng.random() < det_prob
        if measured > 0.8:
            scores["music_coherence_score"] = 1
        elif measured > 0.6:
            scores["music_coherence_score"] = 2
        elif measured > 0.4:
            scores["music_coherence_score"] = 3
        elif measured > 0.2:
            scores["music_coherence_score"] = 4
        else:
            scores["music_coherence_score"] = 5
        scores["raw_measurements"]["measured_mood_distance"] = float(measured)
        return scores
