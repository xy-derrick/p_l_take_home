"""Build an AVA manifest compatible with the benchmark CLI."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.media_utils import probe_duration_seconds  # noqa: E402


def find_local_videos(videos_dir: Path) -> dict[str, Path]:
    """Map AVA video IDs to local MP4 paths."""
    mapping = {}
    for path in videos_dir.rglob("*.mp4"):
        mapping[path.stem] = path
    return mapping


def iter_ava_speech_rows(csv_path: Path):
    """Yield normalized AVA speech rows from headered or headerless CSV."""
    lines = [line.strip() for line in csv_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines:
        return

    first_fields = next(csv.reader([lines[0]]))
    headered = "video_id" in {field.strip() for field in first_fields}

    if headered:
        reader = csv.DictReader(lines)
        for row in reader:
            yield {
                "video_id": row.get("video_id", "").strip(),
                "start_s": row.get("label_start_timestamp_seconds", "").strip(),
                "end_s": row.get("label_end_timestamp_seconds", "").strip(),
                "label": row.get("label", "").strip(),
            }
        return

    reader = csv.reader(lines)
    for row in reader:
        if len(row) < 4:
            continue
        yield {
            "video_id": row[0].strip(),
            "start_s": row[1].strip(),
            "end_s": row[2].strip(),
            "label": row[3].strip(),
        }


def load_active_speaker_times(*archives: Path) -> dict[str, list[float]]:
    """Load speaking-and-audible timestamps per video from AVA tarballs."""
    by_video: dict[str, list[float]] = defaultdict(list)
    for archive_path in archives:
        if not archive_path.exists():
            continue
        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive.getmembers():
                if not member.isfile() or not member.name.endswith(".csv"):
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                reader = csv.reader(line.decode("utf-8-sig") for line in handle)
                for row in reader:
                    if len(row) < 7:
                        continue
                    video_id = row[0].strip()
                    label = row[6].strip()
                    if label in {"SPEAKING_AUDIBLE", "SPEAKING_AND_AUDIBLE"}:
                        try:
                            by_video[video_id].append(float(row[1]))
                        except ValueError:
                            continue
    for values in by_video.values():
        values.sort()
    return by_video


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an AVA JSONL manifest")
    parser.add_argument(
        "--videos-dir",
        default=str(REPO_ROOT / "data" / "raw_videos" / "ava" / "videos"),
        help="Directory containing downloaded AVA MP4 files",
    )
    parser.add_argument(
        "--speech-labels",
        default=str(REPO_ROOT / "data" / "raw_videos" / "ava" / "annotations" / "ava_speech_labels_v1.csv"),
        help="Path to ava_speech_labels_v1.csv",
    )
    parser.add_argument(
        "--activespeaker-train",
        default=str(REPO_ROOT / "data" / "raw_videos" / "ava" / "annotations" / "ava_activespeaker_train_v1.0.tar.bz2"),
        help="Optional active speaker train tarball",
    )
    parser.add_argument(
        "--activespeaker-val",
        default=str(REPO_ROOT / "data" / "raw_videos" / "ava" / "annotations" / "ava_activespeaker_val_v1.0.tar.bz2"),
        help="Optional active speaker val tarball",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data" / "manifests" / "ava_clips.jsonl"),
        help="Output JSONL manifest path",
    )
    parser.add_argument(
        "--max-clips-per-video",
        type=int,
        default=3,
        help="Maximum number of clips emitted per local video",
    )
    parser.add_argument(
        "--clip-max-duration",
        type=float,
        default=8.0,
        help="Maximum clip duration in seconds",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=2.0,
        help="Minimum accepted segment duration",
    )
    parser.add_argument(
        "--include-noise",
        action="store_true",
        help="Include SPEECH_WITH_NOISE intervals",
    )
    parser.add_argument(
        "--require-audible-face",
        action="store_true",
        help="Require at least one SPEAKING_AND_AUDIBLE active-speaker label inside the clip window",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos_dir = Path(args.videos_dir)
    speech_labels_path = Path(args.speech_labels)
    output_path = Path(args.output)

    if not videos_dir.exists():
        print(f"videos dir not found: {videos_dir}", file=sys.stderr)
        return 1
    if not speech_labels_path.exists():
        print(f"speech labels not found: {speech_labels_path}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    local_videos = find_local_videos(videos_dir)
    active_speaker_times = load_active_speaker_times(
        Path(args.activespeaker_train),
        Path(args.activespeaker_val),
    )

    allowed_labels = {"CLEAN_SPEECH", "SPEECH_WITH_MUSIC"}
    if args.include_noise:
        allowed_labels.add("SPEECH_WITH_NOISE")

    per_video_counts: dict[str, int] = defaultdict(int)
    emitted = 0
    with output_path.open("w", encoding="utf-8") as output_handle:
        for row in iter_ava_speech_rows(speech_labels_path):
            video_id = row["video_id"]
            label = row["label"]
            if label not in allowed_labels:
                continue
            if video_id not in local_videos:
                continue
            if per_video_counts[video_id] >= args.max_clips_per_video:
                continue

            try:
                start_s = float(row["start_s"])
                end_s = float(row["end_s"])
            except (TypeError, ValueError):
                continue

            raw_duration = end_s - start_s
            if raw_duration < args.min_duration:
                continue

            video_path = local_videos[video_id]
            video_duration = probe_duration_seconds(video_path)
            clip_start = max(0.0, start_s)
            clip_end = min(video_duration or end_s, min(end_s, clip_start + args.clip_max_duration))
            if clip_end - clip_start < args.min_duration:
                continue

            local_active_times = [
                timestamp
                for timestamp in active_speaker_times.get(video_id, [])
                if clip_start <= timestamp <= clip_end
            ]
            if args.require_audible_face and not local_active_times:
                continue

            clip_type = "mixed" if label == "SPEECH_WITH_MUSIC" else "speech"
            clip_id = f"ava_{video_id}_{int(clip_start * 1000):07d}_{label.lower()}"
            record = {
                "clip_id": clip_id,
                "video_path": relative_or_absolute(video_path),
                "start_s": round(clip_start, 3),
                "end_s": round(clip_end, 3),
                "clip_type": clip_type,
                "ava_speech_label": label,
                "caption": f"AVA {label.lower()} clip from {video_id}",
            }
            if local_active_times:
                record["active_speaker_timestamps_s"] = [
                    round(timestamp - clip_start, 3) for timestamp in local_active_times
                ]

            output_handle.write(json.dumps(record) + "\n")
            per_video_counts[video_id] += 1
            emitted += 1

    print(f"Wrote {emitted} AVA manifest rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
