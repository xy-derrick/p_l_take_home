"""Generate one corrupted WAV example from a chosen corruption module."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from corruption.artifact_inject import inject_artifacts
from corruption.music_mood_swap import apply_music_mood_swap
from corruption.sfx_mistime import apply_sfx_mistime
from corruption.speaker_swap import apply_speaker_swap
from corruption.sync_shift import apply_sync_shift


def _float_list(value: str | None) -> list[float] | None:
    if not value:
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _mood_from_args(prefix: str, args: argparse.Namespace) -> dict | None:
    valence = getattr(args, f"{prefix}_valence", None)
    energy = getattr(args, f"{prefix}_energy", None)
    tempo_bpm = getattr(args, f"{prefix}_tempo_bpm", None)
    if valence is None and energy is None and tempo_bpm is None:
        return None
    return {
        "valence": 0.5 if valence is None else float(valence),
        "energy": 0.5 if energy is None else float(energy),
        "tempo_bpm": 100 if tempo_bpm is None else int(tempo_bpm),
    }


def _common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Input WAV path")
    parser.add_argument("--output", required=True, help="Output WAV path")
    parser.add_argument(
        "--metadata-out",
        default="",
        help="Optional JSON metadata output path. Defaults to <output>.json",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a single corrupted WAV example")
    sub = parser.add_subparsers(dest="command", required=True)

    sync_parser = sub.add_parser("sync_shift", help="Run apply_sync_shift")
    _common_args(sync_parser)
    sync_parser.add_argument("--offset-ms", type=int, default=200)

    speaker_parser = sub.add_parser("speaker_swap", help="Run apply_speaker_swap")
    _common_args(speaker_parser)
    speaker_parser.add_argument("--swap-point-s", type=float, default=5.0)
    speaker_parser.add_argument("--replacement-audio-path", default="")
    speaker_parser.add_argument("--replacement-freq", type=float, default=300.0)

    artifact_parser = sub.add_parser("artifact_inject", help="Run inject_artifacts")
    _common_args(artifact_parser)
    artifact_parser.add_argument("--artifact-type", default="click", choices=["click", "dropout", "spectral", "stutter"])
    artifact_parser.add_argument("--timestamps", default="")
    artifact_parser.add_argument("--severity", type=float, default=0.5)

    sfx_parser = sub.add_parser("sfx_mistime", help="Run apply_sfx_mistime")
    _common_args(sfx_parser)
    sfx_parser.add_argument("--shift-ms", type=int, default=200)
    sfx_parser.add_argument("--event-timestamps", default="")

    mood_parser = sub.add_parser("music_mood_swap", help="Run apply_music_mood_swap")
    _common_args(mood_parser)
    mood_parser.add_argument("--replacement-music-path", default="")
    mood_parser.add_argument("--mood-distance", type=float, default=0.8)
    mood_parser.add_argument("--original-valence", type=float)
    mood_parser.add_argument("--original-energy", type=float)
    mood_parser.add_argument("--original-tempo-bpm", type=int)
    mood_parser.add_argument("--replacement-valence", type=float)
    mood_parser.add_argument("--replacement-energy", type=float)
    mood_parser.add_argument("--replacement-tempo-bpm", type=int)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    metadata_path = Path(args.metadata_out) if args.metadata_out else output_path.with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    if args.command == "sync_shift":
        result = apply_sync_shift(str(input_path), str(output_path), offset_ms=args.offset_ms)
    elif args.command == "speaker_swap":
        result = apply_speaker_swap(
            str(input_path),
            str(output_path),
            swap_point_s=args.swap_point_s,
            replacement_audio_path=args.replacement_audio_path or None,
            replacement_freq=args.replacement_freq,
        )
    elif args.command == "artifact_inject":
        result = inject_artifacts(
            str(input_path),
            str(output_path),
            artifact_type=args.artifact_type,
            timestamps=_float_list(args.timestamps),
            severity=args.severity,
        )
    elif args.command == "sfx_mistime":
        result = apply_sfx_mistime(
            str(input_path),
            str(output_path),
            shift_ms=args.shift_ms,
            event_timestamps=_float_list(args.event_timestamps),
        )
    elif args.command == "music_mood_swap":
        result = apply_music_mood_swap(
            str(input_path),
            str(output_path),
            replacement_music_path=args.replacement_music_path or None,
            original_mood=_mood_from_args("original", args),
            replacement_mood=_mood_from_args("replacement", args),
            mood_distance=args.mood_distance,
        )
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    metadata_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nWrote audio to {output_path}")
    print(f"Wrote metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
