"""CLI entry point for Audio Quality Verification Benchmark.

Usage:
    python cli.py generate     # Generate source clips + variants
    python cli.py score        # Run both scorers on all variants
    python cli.py compare      # Produce comparison analysis
    python cli.py run          # All of the above in sequence
    python cli.py run --seeds S1,S2  # Run specific seeds only
    python cli.py run --mock   # Force mock mode (no API calls)
"""

import argparse
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_generate(args):
    from data.generate_source_clips import generate_all
    generate_all()


def cmd_run(args):
    from evaluation.runner import EvaluationRunner
    from evaluation.comparator import ModelComparator

    seed_filter = None
    if args.seeds:
        seed_filter = [s.strip() for s in args.seeds.split(",")]

    runner = EvaluationRunner(force_mock=args.mock, seed_filter=seed_filter)
    runner.run_all()

    print("\nRunning model comparison analysis...")
    comparator = ModelComparator(runner.variants, runner.signal_results, runner.gemini_results)
    comparator.run_all()


def cmd_score(args):
    """Re-score existing variants (requires prior generate run)."""
    cmd_run(args)


def cmd_compare(args):
    """Run comparison on existing results."""
    import json
    from config import OUTPUT_DIR
    report_path = os.path.join(OUTPUT_DIR, "comparison_report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
        print(json.dumps(report, indent=2))
    else:
        print("No comparison report found. Run 'python cli.py run' first.")


def main():
    parser = argparse.ArgumentParser(
        description="Audio Quality Verification Benchmark for AI-Generated Video"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("generate", help="Generate source clips")

    run_parser = sub.add_parser("run", help="Full pipeline: generate -> score -> compare")
    run_parser.add_argument("--seeds", type=str, default=None,
                           help="Comma-separated seed IDs (e.g., S1,S2)")
    run_parser.add_argument("--mock", action="store_true", default=True,
                           help="Force mock mode (default: True)")
    run_parser.add_argument("--no-mock", dest="mock", action="store_false",
                           help="Use real API calls")

    score_parser = sub.add_parser("score", help="Score all variants")
    score_parser.add_argument("--seeds", type=str, default=None)
    score_parser.add_argument("--mock", action="store_true", default=True)
    score_parser.add_argument("--no-mock", dest="mock", action="store_false")

    sub.add_parser("compare", help="Show comparison report")

    args = parser.parse_args()

    if args.command is None:
        # Default: run everything
        args.command = "run"
        args.seeds = None
        args.mock = True

    commands = {
        "generate": cmd_generate,
        "run": cmd_run,
        "score": cmd_score,
        "compare": cmd_compare,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
