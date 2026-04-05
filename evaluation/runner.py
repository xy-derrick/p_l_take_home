"""Orchestrates the full evaluation pipeline."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from config import CORRUPTED_ROOT, LOGS_ROOT, MODEL_ID, OUTPUT_TSV, SFX_LIBRARY_ROOT, ensure_runtime_dirs
from corruption import CORRUPTION_FUNCTIONS
from corruption.speaker_swap import _estimate_dominant_freq
from data.media_utils import remux_audio_to_video
from data.source_loader import load_source_clips, summarize_clips
from expansion.variant_generator import TaskVariant, generate_cross_pillar_variants, generate_variants
from scoring.aggregator import aggregate_results
from scoring.language_model_scorer import LanguageModelScorer
from scoring.signal_scorer import SignalScorer
from seeds.seed_tasks import SEED_TASKS
from utils import safe_slug, stable_int_seed


class EvaluationRunner:
    """Orchestrates: prepare clips -> corrupt -> score -> aggregate."""

    def __init__(
        self,
        force_mock: bool = True,
        seed_filter: list[str] | None = None,
        datasets: list[str] | None = None,
        force_prepare: bool = False,
        limit_per_dataset: int | None = None,
        reuse_corrupted: bool = False,
    ):
        self.force_mock = force_mock
        self.seed_filter = seed_filter
        self.datasets = datasets
        self.force_prepare = force_prepare
        self.limit_per_dataset = limit_per_dataset
        self.reuse_corrupted = reuse_corrupted
        self.signal_scorer = SignalScorer()
        self.language_model_scorer = LanguageModelScorer(force_mock=force_mock)
        self.variants: list[TaskVariant] = []
        self.signal_results: dict[str, dict] = {}
        self.language_model_results: dict[str, dict] = {}
        self.score_log_path: Path | None = None
        self.latest_score_log_path: Path | None = None

    def run_all(self) -> list[dict]:
        """Execute the full pipeline end-to-end."""
        ensure_runtime_dirs()
        print("=" * 60)
        print("Audio Quality Verification Benchmark")
        print("=" * 60)

        print("\n[1/5] Preparing source clips...")
        source_clips = load_source_clips(
            datasets=self.datasets,
            force_prepare=self.force_prepare,
            limit_per_dataset=self.limit_per_dataset,
            allow_synthetic_fallback=True,
        )
        counts = summarize_clips(source_clips)
        print(f"  Loaded {len(source_clips)} source clips: {counts}")

        print("\n[2/5] Generating task variants...")
        seeds = SEED_TASKS
        if self.seed_filter:
            seeds = {key: value for key, value in seeds.items() if key in self.seed_filter}
        self.variants = generate_variants(source_clips, seeds)
        cross = generate_cross_pillar_variants(self.variants)
        self.variants.extend(cross)
        print(
            f"  Generated {len(self.variants)} variants "
            f"({sum(1 for variant in self.variants if variant.is_clean)} clean controls)"
        )

        print("\n[3/5] Applying corruptions...")
        CORRUPTED_ROOT.mkdir(parents=True, exist_ok=True)
        self._apply_corruptions()

        print("\n[4/5] Scoring variants...")
        self._init_score_logs()
        self._score_all()

        print("\n[5/5] Aggregating results...")
        rows = aggregate_results(self.variants, self.signal_results, self.language_model_results)
        if self.score_log_path is not None:
            print(f"Detailed judge scores saved to {self.score_log_path}")
        if self.latest_score_log_path is not None:
            print(f"Latest judge score log updated at {self.latest_score_log_path}")
        print(f"\nDone! Results written to {OUTPUT_TSV}")
        return rows

    def _apply_corruptions(self) -> None:
        for variant in tqdm(self.variants, desc="  Corrupting"):
            if variant.is_clean:
                variant.corrupted_audio_path = variant.source_audio_path
                variant.corrupted_video_path = variant.source_video_path
                variant.ground_truth = {
                    "corruption_type": "none",
                    "original_audio_path": variant.source_audio_path,
                    "corrupted_audio_path": variant.source_audio_path,
                    "original_video_path": variant.source_video_path,
                    "corrupted_video_path": variant.source_video_path,
                }
                continue

            if "+" in variant.corruption_type:
                self._apply_cross_pillar_corruption(variant)
                continue

            fn_name = variant.corruption_type
            fn = CORRUPTION_FUNCTIONS.get(fn_name)
            if fn is None:
                variant.ground_truth = {"corruption_type": "unknown"}
                continue

            output_audio = CORRUPTED_ROOT / f"{safe_slug(variant.task_id)}.wav"
            params = self._prepare_corruption_params(variant)
            if self._reuse_existing_corruption(variant, output_audio, params):
                continue

            try:
                gt = fn(variant.source_audio_path, str(output_audio), **params)
                variant.corrupted_audio_path = str(output_audio)
                gt["corrupted_audio_path"] = str(output_audio)
                gt["original_audio_path"] = variant.source_audio_path
                variant.ground_truth = gt
                self._attach_corrupted_video(variant)
            except Exception as exc:
                print(f"  Warning: corruption failed for {variant.task_id}: {exc}")
                variant.ground_truth = {"corruption_type": "error", "error": str(exc)}

    def _apply_cross_pillar_corruption(self, variant: TaskVariant) -> None:
        names = variant.corruption_type.split("+")
        audio_tmp = CORRUPTED_ROOT / f"{safe_slug(variant.task_id)}_tmp.wav"
        output_audio = CORRUPTED_ROOT / f"{safe_slug(variant.task_id)}.wav"
        params = self._prepare_corruption_params(variant)

        fn_first = CORRUPTION_FUNCTIONS.get(names[0])
        fn_second = CORRUPTION_FUNCTIONS.get(names[1])
        if fn_first is None or fn_second is None:
            variant.ground_truth = {"corruption_type": "unknown"}
            return

        if self._reuse_existing_corruption(variant, output_audio, params):
            return

        try:
            gt_first = fn_first(variant.source_audio_path, str(audio_tmp), **params)
            gt_second = fn_second(str(audio_tmp), str(output_audio), **params)
            variant.corrupted_audio_path = str(output_audio)
            variant.ground_truth = {
                **gt_first,
                **gt_second,
                "corruption_type": variant.corruption_type,
                "original_audio_path": variant.source_audio_path,
                "corrupted_audio_path": str(output_audio),
            }
            self._attach_corrupted_video(variant)
        finally:
            if audio_tmp.exists():
                audio_tmp.unlink()

    def _prepare_corruption_params(self, variant: TaskVariant) -> dict:
        params = dict(variant.corruption_params)
        if variant.corruption_type == "apply_music_mood_swap":
            self._inject_music_swap_params(variant, params)
        return params

    def _reuse_existing_corruption(self, variant: TaskVariant, output_audio: Path, params: dict) -> bool:
        if not self.reuse_corrupted or not output_audio.exists():
            return False

        variant.corrupted_audio_path = str(output_audio)
        variant.ground_truth = self._build_ground_truth_from_params(variant, params, output_audio)
        self._attach_corrupted_video(variant)
        return True

    def _inject_music_swap_params(self, variant: TaskVariant, params: dict) -> None:
        original_mood = dict(variant.source_metadata.get("mood", {}))
        if not original_mood and variant.metadata_path and os.path.exists(variant.metadata_path):
            original_mood = json.loads(Path(variant.metadata_path).read_text(encoding="utf-8")).get("mood", {})
        if original_mood:
            params["original_mood"] = original_mood

        requested_distance = float(params.get("mood_distance", 0.5))
        best_choice = self._select_replacement_track(original_mood, requested_distance, variant.task_id)
        if best_choice is not None:
            replacement_path, replacement_mood = best_choice
            params["replacement_music_path"] = str(replacement_path)
            params["replacement_mood"] = replacement_mood

    def _select_replacement_track(
        self,
        original_mood: dict,
        requested_distance: float,
        task_id: str,
    ) -> tuple[Path, dict] | None:
        if not original_mood or not SFX_LIBRARY_ROOT.exists():
            return None

        candidates = []
        for wav_path in SFX_LIBRARY_ROOT.glob("*.wav"):
            meta_path = wav_path.with_suffix(".json")
            if not meta_path.exists():
                continue
            mood = json.loads(meta_path.read_text(encoding="utf-8")).get("mood", {})
            distance = self._mood_distance(original_mood, mood)
            gap = abs(distance - requested_distance)
            tie_breaker = stable_int_seed(f"{task_id}:{wav_path.name}")
            candidates.append((gap, tie_breaker, wav_path, mood))

        if not candidates:
            return None

        _, _, best_path, best_mood = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
        return best_path, best_mood

    @staticmethod
    def _mood_distance(first: dict, second: dict) -> float:
        first_valence = first.get("valence", 0.5)
        first_energy = first.get("energy", 0.5)
        second_valence = second.get("valence", 0.5)
        second_energy = second.get("energy", 0.5)
        return math.sqrt((first_valence - second_valence) ** 2 + (first_energy - second_energy) ** 2) / math.sqrt(2)

    def _build_ground_truth_from_params(self, variant: TaskVariant, params: dict, output_audio: Path) -> dict:
        ground_truth = {
            "original_audio_path": variant.source_audio_path,
            "corrupted_audio_path": str(output_audio),
            "original_video_path": variant.source_video_path,
        }

        if "+" in variant.corruption_type:
            ground_truth["corruption_type"] = variant.corruption_type
            return ground_truth

        if variant.corruption_type == "apply_sync_shift":
            offset_ms = int(params.get("offset_ms", 0))
            ground_truth.update(
                {
                    "corruption_type": "sync_shift",
                    "offset_ms": offset_ms,
                    "direction": "lag" if offset_ms > 0 else ("lead" if offset_ms < 0 else "none"),
                }
            )
            return ground_truth

        if variant.corruption_type == "apply_speaker_swap":
            swap_point_s = float(params.get("swap_point_s", 5.0))
            replacement_freq = float(params.get("replacement_freq", 300.0))
            audio, sr = sf.read(variant.source_audio_path)
            swap_idx = min(int(swap_point_s * sr), len(audio) - 1)
            orig_freq = _estimate_dominant_freq(audio[:swap_idx], sr)
            similarity = 1.0 - abs(orig_freq - replacement_freq) / max(orig_freq, replacement_freq)
            ground_truth.update(
                {
                    "corruption_type": "speaker_swap",
                    "swap_point_s": swap_point_s,
                    "original_speaker_freq": float(orig_freq),
                    "replacement_speaker_freq": replacement_freq,
                    "similarity_score": float(max(0.0, similarity)),
                }
            )
            return ground_truth

        if variant.corruption_type == "inject_artifacts":
            severity = float(params.get("severity", 0.5))
            artifact_type = params.get("artifact_type", "click")
            timestamps = params.get("timestamps")
            if timestamps is None:
                audio, sr = sf.read(variant.source_audio_path)
                duration_s = len(audio) / sr
                rng = np.random.default_rng(int(severity * 1000))
                n_events = rng.integers(3, 6)
                timestamps = sorted(rng.uniform(0.5, duration_s - 0.5, n_events).tolist())
            artifacts = [
                {
                    "type": artifact_type,
                    "timestamp_s": float(ts),
                    "severity": severity,
                }
                for ts in timestamps
            ]
            ground_truth.update(
                {
                    "corruption_type": "artifact_inject",
                    "artifact_type": artifact_type,
                    "artifacts": artifacts,
                    "severity": severity,
                }
            )
            return ground_truth

        if variant.corruption_type == "apply_sfx_mistime":
            shift_ms = int(params.get("shift_ms", 200))
            event_timestamps = params.get("event_timestamps")
            if event_timestamps is None:
                event_timestamps = variant.source_metadata.get("event_timestamps_s", [])
            shifted_events = [
                {
                    "original_time_s": float(ts),
                    "shifted_time_s": float(ts + shift_ms / 1000.0),
                    "shift_ms": shift_ms,
                }
                for ts in event_timestamps
            ]
            ground_truth.update(
                {
                    "corruption_type": "sfx_mistime",
                    "shift_ms": shift_ms,
                    "events_shifted": len(shifted_events),
                    "shifted_events": shifted_events,
                }
            )
            return ground_truth

        if variant.corruption_type == "apply_music_mood_swap":
            original_mood = dict(params.get("original_mood", {}))
            replacement_mood = dict(params.get("replacement_mood", {}))
            ground_truth.update(
                {
                    "corruption_type": "music_mood_swap",
                    "original_mood": original_mood,
                    "replacement_mood": replacement_mood,
                    "mood_distance": self._mood_distance(original_mood, replacement_mood)
                    if original_mood and replacement_mood
                    else 0.0,
                    "requested_distance": float(params.get("mood_distance", 0.5)),
                }
            )
            if "replacement_music_path" in params:
                ground_truth["replacement_music_path"] = params["replacement_music_path"]
            return ground_truth

        ground_truth["corruption_type"] = "unknown"
        return ground_truth

    def _attach_corrupted_video(self, variant: TaskVariant) -> None:
        variant.ground_truth["original_video_path"] = variant.source_video_path
        if not variant.source_video_path or not variant.corrupted_audio_path:
            variant.corrupted_video_path = None
            variant.ground_truth["corrupted_video_path"] = None
            return

        output_video = CORRUPTED_ROOT / f"{safe_slug(variant.task_id)}.mp4"
        if self.reuse_corrupted and output_video.exists():
            variant.corrupted_video_path = str(output_video)
            variant.ground_truth["corrupted_video_path"] = str(output_video)
            return
        try:
            remux_audio_to_video(
                Path(variant.source_video_path),
                Path(variant.corrupted_audio_path),
                output_video,
            )
            variant.corrupted_video_path = str(output_video)
            variant.ground_truth["corrupted_video_path"] = str(output_video)
        except Exception as exc:
            print(f"  Warning: video remux failed for {variant.task_id}: {exc}")
            variant.corrupted_video_path = None
            variant.ground_truth["corrupted_video_path"] = None

    def _init_score_logs(self) -> None:
        LOGS_ROOT.mkdir(parents=True, exist_ok=True)
        seed_label = "all-seeds" if not self.seed_filter else "-".join(self.seed_filter)
        dataset_label = "default-datasets" if not self.datasets else "-".join(self.datasets)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"judge_scores_{safe_slug(seed_label)}_{safe_slug(dataset_label)}_{timestamp}.jsonl"
        self.score_log_path = LOGS_ROOT / filename
        self.latest_score_log_path = LOGS_ROOT / "latest_judge_scores.jsonl"

    def _score_log_entry(self, variant: TaskVariant, model_name: str, scores: dict) -> dict:
        entry = {
            "logged_at": datetime.now().isoformat(timespec="seconds"),
            "model": model_name,
            "task_id": variant.task_id,
            "seed_id": variant.seed_id,
            "dataset": variant.source_dataset,
            "source_clip": variant.source_clip_name,
            "clip_type": variant.clip_type,
            "visual_context": variant.has_visual_context(),
            "is_clean": variant.is_clean,
            "tier": variant.tier,
            "difficulty_estimate": variant.difficulty_estimate,
            "corruption_type": variant.ground_truth.get("corruption_type", variant.corruption_type),
            "corruption_params": variant.corruption_params,
            "ground_truth": variant.ground_truth,
            "source_audio_path": variant.source_audio_path,
            "source_video_path": variant.source_video_path,
            "corrupted_audio_path": variant.corrupted_audio_path,
            "corrupted_video_path": variant.corrupted_video_path,
            "scores": scores,
        }
        return self._json_safe(entry)

    def _json_safe(self, value):
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return [self._json_safe(item) for item in value.tolist()]
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _score_all(self) -> None:
        if self.score_log_path is None or self.latest_score_log_path is None:
            self._init_score_logs()

        with (
            self.score_log_path.open("w", encoding="utf-8") as score_handle,
            self.latest_score_log_path.open("w", encoding="utf-8") as latest_handle,
        ):
            for variant in tqdm(self.variants, desc="  Scoring"):
                signal_scores = self.signal_scorer.score_variant(variant)
                self.signal_results[variant.task_id] = signal_scores
                signal_entry = self._score_log_entry(variant, "signal_pipeline", signal_scores)
                payload = json.dumps(signal_entry, ensure_ascii=True)
                score_handle.write(payload + "\n")
                latest_handle.write(payload + "\n")

                language_model_scores = self.language_model_scorer.score_variant(variant)
                self.language_model_results[variant.task_id] = language_model_scores
                language_model_entry = self._score_log_entry(variant, MODEL_ID, language_model_scores)
                payload = json.dumps(language_model_entry, ensure_ascii=True)
                score_handle.write(payload + "\n")
                latest_handle.write(payload + "\n")
