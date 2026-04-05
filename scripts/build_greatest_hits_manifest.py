"""Build a Greatest Hits manifest compatible with the benchmark CLI."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.media_utils import probe_duration_seconds  # noqa: E402


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_float_list(value: str) -> list[float]:
    values = []
    for token in re.split(r"[,\s;|]+", value.strip()):
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            continue
    return sorted(values)


def extract_timestamps_from_json(payload) -> list[float]:
    if isinstance(payload, dict):
        for key in [
            "event_timestamps_s",
            "impact_times_s",
            "hit_times_s",
            "onsets",
            "timestamps",
        ]:
            if key in payload:
                return sorted(float(item) for item in payload[key])
        if "events" in payload and isinstance(payload["events"], list):
            values = []
            for event in payload["events"]:
                if isinstance(event, dict):
                    for field in ["time", "timestamp", "timestamp_s", "onset", "onset_s"]:
                        if field in event:
                            try:
                                values.append(float(event[field]))
                            except ValueError:
                                pass
            if values:
                return sorted(values)
    return []


def load_sidecar_timestamps(video_path: Path) -> list[float]:
    stem = video_path.with_suffix("")
    for suffix in [".json", ".txt", ".csv"]:
        sidecar = stem.with_suffix(suffix)
        if not sidecar.exists():
            continue
        if suffix == ".json":
            payload = json.loads(sidecar.read_text(encoding="utf-8-sig"))
            timestamps = extract_timestamps_from_json(payload)
            if timestamps:
                return timestamps
        elif suffix == ".txt":
            timestamps = parse_float_list(sidecar.read_text(encoding="utf-8-sig"))
            if timestamps:
                return timestamps
        else:
            with sidecar.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                values = []
                for row in reader:
                    for key in ["timestamp", "timestamp_s", "time_s", "onset", "onset_s"]:
                        if key in row and row[key]:
                            try:
                                values.append(float(row[key]))
                            except ValueError:
                                pass
                if values:
                    return sorted(values)
    return []


def load_global_labels(labels_file: Path) -> dict[str, list[float]]:
    if not labels_file or not labels_file.exists():
        return {}

    suffix = labels_file.suffix.lower()
    by_name: dict[str, list[float]] = defaultdict(list)

    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            rows = [json.loads(line) for line in labels_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        else:
            payload = json.loads(labels_file.read_text(encoding="utf-8-sig"))
            rows = payload if isinstance(payload, list) else payload.get("clips", [])

        for row in rows:
            video_name = str(
                row.get("video")
                or row.get("video_id")
                or row.get("filename")
                or row.get("path")
                or ""
            ).strip()
            if not video_name:
                continue
            values = extract_timestamps_from_json(row)
            if values:
                by_name[Path(video_name).stem].extend(values)

    elif suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with labels_file.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for row in reader:
                video_name = str(
                    row.get("video")
                    or row.get("video_id")
                    or row.get("filename")
                    or row.get("path")
                    or ""
                ).strip()
                if not video_name:
                    continue
                values = []
                for key in ["event_timestamps_s", "timestamps", "onsets"]:
                    if key in row and row[key]:
                        values.extend(parse_float_list(row[key]))
                for key in ["timestamp", "timestamp_s", "time_s", "onset", "onset_s"]:
                    if key in row and row[key]:
                        try:
                            values.append(float(row[key]))
                        except ValueError:
                            pass
                if values:
                    by_name[Path(video_name).stem].extend(values)

    return {key: sorted(values) for key, values in by_name.items()}


def cluster_timestamps(
    timestamps: list[float],
    clip_duration: float,
    pre_roll: float,
    post_roll: float,
) -> list[tuple[float, float, list[float]]]:
    if not timestamps:
        return []

    clusters = []
    current = [timestamps[0]]
    current_start = max(0.0, timestamps[0] - pre_roll)
    for timestamp in timestamps[1:]:
        proposed_end = timestamp + post_roll
        if proposed_end - current_start <= clip_duration:
            current.append(timestamp)
        else:
            end_s = min(current[-1] + post_roll, current_start + clip_duration)
            clusters.append((current_start, end_s, list(current)))
            current = [timestamp]
            current_start = max(0.0, timestamp - pre_roll)

    end_s = min(current[-1] + post_roll, current_start + clip_duration)
    clusters.append((current_start, end_s, list(current)))
    return clusters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Greatest Hits JSONL manifest")
    parser.add_argument(
        "--videos-dir",
        default=str(REPO_ROOT / "data" / "raw_videos" / "greatest_hits"),
        help="Directory containing extracted Greatest Hits videos",
    )
    parser.add_argument(
        "--labels-file",
        default="",
        help="Optional global labels file (JSON/JSONL/CSV/TSV)",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data" / "manifests" / "greatest_hits_clips.jsonl"),
        help="Output JSONL manifest path",
    )
    parser.add_argument(
        "--clip-duration",
        type=float,
        default=6.0,
        help="Maximum generated clip duration",
    )
    parser.add_argument(
        "--pre-roll",
        type=float,
        default=0.75,
        help="Seconds to include before the first event in a cluster",
    )
    parser.add_argument(
        "--post-roll",
        type=float,
        default=1.25,
        help="Seconds to include after the last event in a cluster",
    )
    parser.add_argument(
        "--max-clips-per-video",
        type=int,
        default=3,
        help="Maximum manifest rows per source video",
    )
    parser.add_argument(
        "--fallback-duration",
        type=float,
        default=6.0,
        help="Clip duration when no event labels are found",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos_dir = Path(args.videos_dir)
    output_path = Path(args.output)
    if not videos_dir.exists():
        print(f"videos dir not found: {videos_dir}", file=sys.stderr)
        return 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    global_labels = load_global_labels(Path(args.labels_file)) if args.labels_file else {}

    emitted = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for video_path in sorted(path for path in videos_dir.rglob("*") if path.suffix.lower() in VIDEO_EXTS):
            stem = video_path.stem
            timestamps = global_labels.get(stem, []) or load_sidecar_timestamps(video_path)
            duration = probe_duration_seconds(video_path)

            if timestamps:
                windows = cluster_timestamps(
                    timestamps,
                    clip_duration=args.clip_duration,
                    pre_roll=args.pre_roll,
                    post_roll=args.post_roll,
                )[: args.max_clips_per_video]
            else:
                fallback_end = min(duration or args.fallback_duration, args.fallback_duration)
                windows = [(0.0, fallback_end, [])]

            for index, (start_s, end_s, local_events) in enumerate(windows):
                if end_s <= start_s:
                    continue
                record = {
                    "clip_id": f"greatest_hits_{stem}_{index:02d}",
                    "video_path": relative_or_absolute(video_path),
                    "start_s": round(start_s, 3),
                    "end_s": round(end_s, 3),
                    "clip_type": "sfx",
                    "caption": f"Greatest Hits impact clip from {stem}",
                }
                if local_events:
                    record["event_timestamps_s"] = [round(event - start_s, 3) for event in local_events]
                handle.write(json.dumps(record) + "\n")
                emitted += 1

    print(f"Wrote {emitted} Greatest Hits manifest rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
