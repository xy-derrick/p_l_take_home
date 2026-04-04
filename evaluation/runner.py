"""Orchestrates the full evaluation pipeline."""

from __future__ import annotations

import json
import os
import shutil

from tqdm import tqdm

from config import SOURCE_CLIPS_DIR, CORRUPTED_DIR, SFX_LIBRARY_DIR, OUTPUT_TSV
from data.generate_source_clips import generate_all as generate_source_clips
from seeds.seed_tasks import SEED_TASKS
from corruption import CORRUPTION_FUNCTIONS
from expansion.variant_generator import (
    generate_variants, generate_cross_pillar_variants, TaskVariant,
)
from scoring.signal_scorer import SignalScorer
from scoring.gemini_scorer import GeminiScorer
from scoring.aggregator import aggregate_results


class EvaluationRunner:
    """Orchestrates: generate variants -> corrupt -> score -> aggregate."""

    def __init__(self, force_mock: bool = True, seed_filter: list[str] | None = None):
        self.force_mock = force_mock
        self.seed_filter = seed_filter
        self.signal_scorer = SignalScorer()
        self.gemini_scorer = GeminiScorer(force_mock=force_mock)
        self.variants: list[TaskVariant] = []
        self.signal_results: dict[str, dict] = {}
        self.gemini_results: dict[str, dict] = {}

    def run_all(self) -> list[dict]:
        """Execute the full pipeline end-to-end."""
        print("=" * 60)
        print("Audio Quality Verification Benchmark")
        print("=" * 60)

        # Step 1: Generate source clips
        print("\n[1/5] Generating source clips...")
        source_clips = generate_source_clips()

        # Step 2: Generate variants
        print("\n[2/5] Generating task variants...")
        seeds = SEED_TASKS
        if self.seed_filter:
            seeds = {k: v for k, v in seeds.items() if k in self.seed_filter}
        self.variants = generate_variants(source_clips, seeds)
        cross = generate_cross_pillar_variants(self.variants)
        self.variants.extend(cross)
        print(f"  Generated {len(self.variants)} variants "
              f"({sum(1 for v in self.variants if v.is_clean)} clean controls)")

        # Step 3: Apply corruptions
        print("\n[3/5] Applying corruptions...")
        os.makedirs(CORRUPTED_DIR, exist_ok=True)
        self._apply_corruptions()

        # Step 4: Score with both models
        print("\n[4/5] Scoring variants...")
        self._score_all()

        # Step 5: Aggregate and write TSV
        print("\n[5/5] Aggregating results...")
        rows = aggregate_results(
            self.variants, self.signal_results, self.gemini_results
        )
        print(f"\nDone! Results written to {OUTPUT_TSV}")
        return rows

    def _apply_corruptions(self) -> None:
        for v in tqdm(self.variants, desc="  Corrupting"):
            if v.is_clean:
                # Clean variant: no corruption, just point to source
                v.ground_truth = {
                    "corruption_type": "none",
                    "original_path": v.source_clip,
                    "corrupted_path": v.source_clip,
                }
                continue

            # Handle cross-pillar variants
            if "+" in v.corruption_type:
                # Apply first corruption, then second on the result
                fns = v.corruption_type.split("+")
                output = os.path.join(CORRUPTED_DIR, f"{v.task_id}.wav")
                intermediate = os.path.join(CORRUPTED_DIR, f"{v.task_id}_tmp.wav")
                gt_combined = {}

                fn1 = CORRUPTION_FUNCTIONS.get(fns[0])
                fn2 = CORRUPTION_FUNCTIONS.get(fns[1])
                if fn1 and fn2:
                    gt1 = fn1(v.source_clip, intermediate, **v.corruption_params)
                    gt2 = fn2(intermediate, output, **v.corruption_params)
                    gt_combined = {**gt1, **gt2}
                    gt_combined["corruption_type"] = v.corruption_type
                    gt_combined["corrupted_path"] = output
                    if os.path.exists(intermediate):
                        os.remove(intermediate)
                v.ground_truth = gt_combined
                continue

            fn_name = v.corruption_type
            fn = CORRUPTION_FUNCTIONS.get(fn_name)
            if fn is None:
                v.ground_truth = {"corruption_type": "unknown"}
                continue

            output_path = os.path.join(CORRUPTED_DIR, f"{v.task_id}.wav")
            params = dict(v.corruption_params)

            # Add replacement music path for mood swap
            if fn_name == "apply_music_mood_swap":
                # Load source metadata for original mood
                json_path = v.source_clip.replace(".wav", ".json")
                if os.path.exists(json_path):
                    with open(json_path) as f:
                        meta = json.load(f)
                    params.setdefault("original_mood", meta.get("mood", {}))
                # Pick a replacement from sfx_library
                lib_files = []
                if os.path.isdir(SFX_LIBRARY_DIR):
                    lib_files = [f for f in os.listdir(SFX_LIBRARY_DIR) if f.endswith(".wav")]
                if lib_files:
                    import random
                    random.seed(hash(v.task_id))
                    rep_file = random.choice(lib_files)
                    params["replacement_music_path"] = os.path.join(SFX_LIBRARY_DIR, rep_file)
                    rep_json = os.path.join(SFX_LIBRARY_DIR, rep_file.replace(".wav", ".json"))
                    if os.path.exists(rep_json):
                        with open(rep_json) as f:
                            rep_meta = json.load(f)
                        params.setdefault("replacement_mood", rep_meta.get("mood", {}))

            try:
                gt = fn(v.source_clip, output_path, **params)
                v.ground_truth = gt
            except Exception as e:
                print(f"  Warning: corruption failed for {v.task_id}: {e}")
                v.ground_truth = {"corruption_type": "error", "error": str(e)}

    def _score_all(self) -> None:
        for v in tqdm(self.variants, desc="  Scoring"):
            self.signal_results[v.task_id] = self.signal_scorer.score_variant(v)
            self.gemini_results[v.task_id] = self.gemini_scorer.score_variant(v)
