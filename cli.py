"""CLI entry point for Audio Quality Verification Benchmark."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _parse_csv_arg(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _seed_report_prefix(seed_filter: list[str] | None) -> str:
    if not seed_filter:
        return "all_seeds"

    labels = []
    for seed in seed_filter:
        match = re.search(r"(\d+)", seed)
        if match:
            labels.append(f"seed{match.group(1)}")
        else:
            labels.append(re.sub(r"[^a-z0-9]+", "_", seed.lower()).strip("_") or "seed")
    return "_".join(labels)


def _export_report_artifacts(seed_filter: list[str] | None) -> None:
    from config import OUTPUT_ROOT, PLOTS_ROOT, REPORT_ROOT

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    prefix = _seed_report_prefix(seed_filter)

    copies = [
        (Path("logs") / "latest_judge_scores.jsonl", REPORT_ROOT / f"{prefix}_latest_judge_scores.jsonl"),
        (OUTPUT_ROOT / "comparison_report.json", REPORT_ROOT / f"{prefix}_comparison_report.json"),
        (PLOTS_ROOT / "tier_comparison.png", REPORT_ROOT / f"{prefix}_tier_comparison.png"),
    ]

    copied_any = False
    for source, destination in copies:
        source_path = source if isinstance(source, Path) and source.is_absolute() else Path(source)
        if not source_path.is_absolute():
            source_path = Path(__file__).resolve().parent / source_path
        if not source_path.exists():
            continue
        shutil.copy2(source_path, destination)
        print(f"Copied report artifact to {destination}")
        copied_any = True

    if not copied_any:
        print("No report artifacts were copied.")


def _common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--datasets",
        type=str,
        default="synthetic",
        help="Comma-separated datasets: synthetic, ava, greatest_hits, condensed_movies",
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=None,
        help="Optional cap on the number of prepared clips per dataset",
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Re-extract dataset clips even if prepared clips already exist",
    )
    parser.add_argument(
        "--reuse-corrupted",
        action="store_true",
        help="Reuse existing files under data/corrupted when present instead of regenerating them",
    )


def cmd_generate(args):
    from data.source_loader import load_source_clips, summarize_clips

    datasets = _parse_csv_arg(args.datasets)
    clips = load_source_clips(
        datasets=datasets,
        force_prepare=args.force_prepare,
        limit_per_dataset=args.limit_per_dataset,
        allow_synthetic_fallback=True,
    )
    print(f"Prepared {len(clips)} source clips: {summarize_clips(clips)}")


def cmd_prepare_datasets(args):
    from data.source_loader import load_source_clips, summarize_clips

    datasets = [dataset for dataset in (_parse_csv_arg(args.datasets) or []) if dataset != "synthetic"]
    clips = load_source_clips(
        datasets=datasets,
        force_prepare=args.force_prepare,
        limit_per_dataset=args.limit_per_dataset,
        allow_synthetic_fallback=False,
    )
    print(f"Prepared {len(clips)} real-video source clips: {summarize_clips(clips)}")
    if not clips:
        print("No prepared clips found. Add a dataset manifest under data/manifests/ and rerun.")


def cmd_extract_clips(args):
    cmd_prepare_datasets(args)


def cmd_run(args):
    from evaluation.comparator import ModelComparator
    from evaluation.runner import EvaluationRunner

    seed_filter = _parse_csv_arg(args.seeds)
    datasets = _parse_csv_arg(args.datasets)

    runner = EvaluationRunner(
        force_mock=args.mock,
        seed_filter=seed_filter,
        datasets=datasets,
        force_prepare=args.force_prepare,
        limit_per_dataset=args.limit_per_dataset,
        reuse_corrupted=args.reuse_corrupted,
    )
    runner.run_all()

    print("\nRunning model comparison analysis...")
    comparator = ModelComparator(runner.variants, runner.signal_results, runner.language_model_results)
    comparator.run_all()
    _export_report_artifacts(seed_filter)


def cmd_score(args):
    """Re-score all variants by executing the full run pipeline."""
    cmd_run(args)


def cmd_compare(args):
    from config import OUTPUT_DIR

    report_path = os.path.join(OUTPUT_DIR, "comparison_report.json")
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as handle:
            report = json.load(handle)
        print(json.dumps(report, indent=2))
    else:
        print("No comparison report found. Run 'python cli.py run' first.")


def cmd_replay_from_log(args):
    from evaluation.comparator import ModelComparator
    from evaluation.log_replay import derive_log_prefix, load_replay_bundle
    from scoring.aggregator import aggregate_results

    input_path = Path(args.input).expanduser()
    if not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()

    variants, signal_results, language_model_results = load_replay_bundle(input_path)
    prefix = derive_log_prefix(input_path)

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else input_path.parent
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tsv_path = output_dir / f"{prefix}_tasks_and_rubrics.tsv"
    report_path = output_dir / f"{prefix}_comparison_report.json"
    plot_prefix = f"{prefix}_"

    aggregate_results(variants, signal_results, language_model_results, output_path=str(tsv_path))

    comparator = ModelComparator(
        variants,
        signal_results,
        language_model_results,
        report_path=str(report_path),
        plots_dir=str(output_dir),
        plot_prefix=plot_prefix,
    )
    comparator.run_all()

    print(f"Replayed {len(variants)} variants from {input_path}")
    print(f"  TSV: {tsv_path}")
    print(f"  Comparison report: {report_path}")
    print(f"  Tier plot: {output_dir / f'{plot_prefix}tier_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Audio Quality Verification Benchmark for AI-Generated Video"
    )
    sub = parser.add_subparsers(dest="command")

    generate_parser = sub.add_parser("generate", help="Prepare source clips")
    _common_args(generate_parser)

    prepare_parser = sub.add_parser("prepare-datasets", help="Prepare real-video dataset clips")
    _common_args(prepare_parser)

    extract_parser = sub.add_parser("extract-clips", help="Extract prepared clips from dataset manifests")
    _common_args(extract_parser)

    run_parser = sub.add_parser("run", help="Full pipeline: prepare -> score -> compare")
    _common_args(run_parser)
    run_parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seed IDs (e.g., S1,S2)")
    run_parser.add_argument("--mock", action="store_true", default=True, help="Force mock mode (default: True)")
    run_parser.add_argument("--no-mock", dest="mock", action="store_false", help="Use real API calls")

    score_parser = sub.add_parser("score", help="Score all variants")
    _common_args(score_parser)
    score_parser.add_argument("--seeds", type=str, default=None)
    score_parser.add_argument("--mock", action="store_true", default=True)
    score_parser.add_argument("--no-mock", dest="mock", action="store_false")

    sub.add_parser("compare", help="Show comparison report")

    replay_parser = sub.add_parser("replay-from-log", help="Rebuild outputs from a judge score JSONL log")
    replay_parser.add_argument("--input", type=str, required=True, help="Path to a judge score JSONL file")
    replay_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for recreated outputs (default: same directory as the input log)",
    )

    args = parser.parse_args()

    if args.command is None:
        args.command = "run"
        args.seeds = None
        args.datasets = "synthetic"
        args.mock = True
        args.limit_per_dataset = None
        args.force_prepare = False
        args.reuse_corrupted = False

    commands = {
        "generate": cmd_generate,
        "prepare-datasets": cmd_prepare_datasets,
        "extract-clips": cmd_extract_clips,
        "run": cmd_run,
        "score": cmd_score,
        "compare": cmd_compare,
        "replay-from-log": cmd_replay_from_log,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
