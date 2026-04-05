"""OpenRouter multimodal scorer, with mock fallback."""

from __future__ import annotations

import base64
import json
import os
import re
import time

import numpy as np

from config import MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from scoring.rubric import RUBRIC_DIMENSIONS
from seeds.seed_tasks import SEED_TASKS
from utils import stable_int_seed


class LanguageModelScorer:
    """Scores audio quality using the configured OpenRouter model, with mock fallback."""

    def __init__(self, force_mock: bool = False):
        self.api_key = OPENROUTER_API_KEY
        self.use_mock = force_mock or not self.api_key
        self._last_call_time: float = 0.0
        self._min_call_interval: float = 0.5  # 500 ms between API calls

    def score_variant(self, variant) -> dict:
        """Score a single task variant."""
        if self.use_mock:
            return self._mock_response(variant)
        return self._api_response(variant)

    def _api_response(self, variant) -> dict:
        import requests

        seed = SEED_TASKS.get(variant.seed_id.split("+")[0])
        if seed is None:
            return self._mock_response(variant)

        # Proactive rate limiting
        now = time.time()
        wait = self._min_call_interval - (now - self._last_call_time)
        if wait > 0:
            time.sleep(wait)

        prompt = self._build_prompt(variant, seed)
        messages = [{"role": "user", "content": self._build_content(variant, prompt)}]

        for attempt in range(3):
            try:
                self._last_call_time = time.time()
                response = requests.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": MODEL,
                        "messages": messages,
                        "max_tokens": 1024,
                    },
                    timeout=60,
                )
                response.raise_for_status()
                payload = response.json()
                text = payload["choices"][0]["message"]["content"]
                return self._parse_response(text, variant)
            except Exception as exc:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                print(f"  [Model API error for {variant.task_id}: {exc}, falling back to mock]")
                return self._mock_response(variant)

        # Defensive fallback — should not be reached
        return self._mock_response(variant)

    def _build_content(self, variant, prompt: str) -> list[dict]:
        content = [{"type": "text", "text": prompt}]
        video_path = variant.corrupted_video_path or variant.ground_truth.get("corrupted_video_path")

        if variant.has_visual_context() and video_path and os.path.exists(video_path):
            with open(video_path, "rb") as handle:
                encoded = base64.b64encode(handle.read()).decode("utf-8")
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{encoded}"},
                }
            )
            return content

        audio_path = (
            variant.corrupted_audio_path
            or variant.ground_truth.get("corrupted_audio_path")
            or variant.ground_truth.get("corrupted_path")
            or variant.source_audio_path
        )
        if not audio_path or not os.path.exists(audio_path):
            # No usable audio file; return text-only content
            return content
        with open(audio_path, "rb") as handle:
            audio_b64 = base64.b64encode(handle.read()).decode("utf-8")
        # OpenRouter/Gemini expects audio via audio_url with a data URI
        content.append(
            {
                "type": "audio_url",
                "audio_url": {
                    "url": f"data:audio/wav;base64,{audio_b64}",
                },
            }
        )
        return content

    def _build_prompt(self, variant, seed) -> str:
        has_visual_context = variant.has_visual_context()
        rubric_text = "\n".join(f"- {rubric['name']}: {rubric['levels']}" for rubric in RUBRIC_DIMENSIONS)
        mode = "video+audio" if has_visual_context else "audio-only fallback"
        return (
            "You are evaluating audio quality for AI-generated video.\n\n"
            f"Input mode: {mode}\n"
            f"Dataset: {variant.source_dataset}\n"
            f"Task: {seed.description}\n\n"
            f"{seed.prompt_template(has_visual_context)}\n\n"
            "Also rate the overall clip on these rubric dimensions (1-5 each):\n"
            f"{rubric_text}\n\n"
            'Include rubric scores in your JSON response as: "av_sync_score", '
            '"artifact_quality_score", "speaker_consistency_score", '
            '"semantic_match_score", "music_coherence_score"'
        )

    def _extract_json(self, text: str) -> dict | None:
        """Robustly extract the first balanced JSON object from model output."""
        # Strip markdown code fences
        clean = re.sub(r"```(?:json)?\s*", "", text)
        clean = re.sub(r"```", "", clean).strip()
        # Walk character-by-character to find the first balanced { ... }
        depth = 0
        start = None
        for i, ch in enumerate(clean):
            if ch == "{":
                if start is None:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        return json.loads(clean[start : i + 1])
                    except json.JSONDecodeError:
                        # Reset and keep searching
                        start = None
                        depth = 0
        return None

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
            "raw_measurements": {},  # normalized schema (mirrors mock output)
        }
        parsed = self._extract_json(text)
        if parsed is None:
            return scores
        for key in [
            "av_sync_score",
            "artifact_quality_score",
            "speaker_consistency_score",
            "semantic_match_score",
            "music_coherence_score",
        ]:
            if key in parsed:
                scores[key] = int(float(parsed[key]))  # tolerate float values
        if "synced" in parsed:
            scores["detection_correct"] = (not parsed.get("synced", True)) if not variant.is_clean else parsed.get("synced", True)
        elif "consistent" in parsed:
            scores["detection_correct"] = (not parsed.get("consistent", True)) if not variant.is_clean else parsed.get("consistent", True)
        elif "aligned" in parsed:
            scores["detection_correct"] = (not parsed.get("aligned", True)) if not variant.is_clean else parsed.get("aligned", True)
        elif "clean" in parsed:
            scores["detection_correct"] = (not parsed.get("clean", True)) if not variant.is_clean else parsed.get("clean", True)
        elif "mood_match_score" in parsed:
            mood_score = float(parsed["mood_match_score"])
            scores["detection_correct"] = (mood_score <= 3) if not variant.is_clean else (mood_score >= 4)
        return scores

    def _mock_response(self, variant) -> dict:
        """Generate simulated response based on expected language-model behavior."""
        rng = np.random.default_rng(stable_int_seed(f"model:{variant.task_id}"))
        gt = variant.ground_truth or {}
        corruption_type = gt.get("corruption_type", "none")

        scores = {
            "av_sync_score": 5,
            "artifact_quality_score": 5,
            "speaker_consistency_score": 5,
            "semantic_match_score": 5,
            "music_coherence_score": 5,
            "detection_correct": False,
            "raw_response": "",        # normalized schema (mirrors API output)
            "raw_measurements": {},
        }

        if variant.is_clean:
            if rng.random() < 0.10:
                scores["detection_correct"] = False
                dim = rng.choice(["av_sync_score", "artifact_quality_score"])
                scores[dim] = 4
            else:
                scores["detection_correct"] = True
            return scores

        if corruption_type == "sync_shift":
            return self._mock_sync(variant, scores, rng)
        if corruption_type == "speaker_swap":
            return self._mock_speaker(variant, scores, rng)
        if corruption_type == "artifact_inject":
            return self._mock_artifacts(variant, scores, rng)
        if corruption_type == "sfx_mistime":
            return self._mock_sfx(variant, scores, rng)
        if corruption_type == "music_mood_swap":
            return self._mock_mood(variant, scores, rng)
        return scores

    def _mock_sync(self, variant, scores, rng) -> dict:
        true_offset = abs(variant.ground_truth.get("offset_ms", 0))
        measured = max(0.0, true_offset + rng.normal(0, 200))
        # Deterministic probabilities — single RNG draw for the detection decision
        if true_offset < 100:
            det_prob = 0.50
        elif true_offset < 300:
            det_prob = 0.78
        else:
            det_prob = 0.94
        scores["detection_correct"] = rng.random() < det_prob
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
        # Deterministic probabilities — single RNG draw for the detection decision
        if similarity < 0.5:
            det_prob = 0.85
        elif similarity < 0.7:
            det_prob = 0.65
        else:
            det_prob = 0.55
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
        swap_point = variant.ground_truth.get("swap_point_s", 5.0)
        scores["raw_measurements"]["estimated_swap_s"] = float(swap_point + rng.normal(0, 2))
        return scores

    def _mock_artifacts(self, variant, scores, rng) -> dict:
        severity = variant.ground_truth.get("severity", 0.5)
        # Deterministic probabilities — single RNG draw for the detection decision
        if severity < 0.3:
            det_prob = 0.50
        elif severity < 0.6:
            det_prob = 0.78
        else:
            det_prob = 0.94
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
        measured = max(0.0, shift_ms + rng.normal(0, 200))
        # Deterministic probabilities — single RNG draw for the detection decision
        if shift_ms < 100:
            det_prob = 0.50
        elif shift_ms < 300:
            det_prob = 0.78
        else:
            det_prob = 0.94
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
        measured = np.clip(mood_dist + rng.normal(0, 0.15), 0, 1)
        # Deterministic probabilities — single RNG draw for the detection decision
        if mood_dist > 0.6:
            det_prob = 0.70
        elif mood_dist > 0.3:
            det_prob = 0.58
        else:
            det_prob = 0.45
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
