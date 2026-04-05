"""Media preparation helpers for real-video source datasets."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

from config import DEFAULT_CLIP_DURATION_S, PROJECT_ROOT, SAMPLE_RATE, SOURCE_CLIPS_ROOT, VIDEO_HEIGHT
from utils import safe_slug


def run_command(args: list[str]) -> None:
    """Run a subprocess and raise a readable error on failure."""
    proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise RuntimeError(stderr or "command failed")


def resolve_path(path_value: str, base_dir: Path) -> Path:
    """Resolve relative manifest paths against several sensible roots."""
    path = Path(path_value)
    if path.is_absolute():
        return path

    candidates = [
        (base_dir / path).resolve(),
        (PROJECT_ROOT / path).resolve(),
        (Path.cwd() / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_manifest_records(manifest_path: Path) -> list[dict]:
    """Load JSON, JSONL, or CSV manifest files."""
    if not manifest_path.exists():
        return []

    suffix = manifest_path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        for line in manifest_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    if suffix == ".json":
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict):
            return list(payload.get("clips", []))
        return list(payload)

    if suffix == ".csv":
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    raise ValueError(f"Unsupported manifest format: {manifest_path}")


def build_source_clip_record(meta: dict) -> dict:
    """Normalize metadata into the clip record shape used by the runner."""
    audio_path = meta.get("source_audio_path") or meta.get("path")
    return {
        "dataset": meta.get("dataset", "synthetic"),
        "name": meta["name"],
        "type": meta["type"],
        "path": audio_path,
        "source_audio_path": audio_path,
        "source_video_path": meta.get("source_video_path"),
        "metadata_path": meta.get("metadata_path"),
        "duration_s": meta.get("duration_s", 0.0),
        "segment_start_s": meta.get("segment_start_s", 0.0),
        "segment_end_s": meta.get("segment_end_s"),
        "visual_context_available": bool(meta.get("visual_context_available", False)),
        "metadata": meta,
    }


def load_prepared_clips(dataset_name: str) -> list[dict]:
    """Load already-prepared clip metadata from source_clips/<dataset>."""
    dataset_dir = SOURCE_CLIPS_ROOT / dataset_name
    if not dataset_dir.exists():
        return []

    clips = []
    for meta_path in sorted(dataset_dir.glob("*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        clips.append(build_source_clip_record(meta))
    return clips


def probe_duration_seconds(media_path: Path) -> float:
    """Return the media duration in seconds."""
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        return 0.0
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


def extract_video_clip(
    input_video: Path,
    output_video: Path,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> None:
    """Extract and lightly normalize a video segment."""
    output_video.parent.mkdir(parents=True, exist_ok=True)

    clip_args = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-ss",
        f"{start_s:.3f}",
    ]
    if end_s is not None:
        clip_args.extend(["-to", f"{end_s:.3f}"])
    clip_args.extend(
        [
            "-vf",
            f"scale=-2:{VIDEO_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-r",
            "24",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_video),
        ]
    )
    run_command(clip_args)


def extract_audio_wav(input_media: Path, output_wav: Path) -> None:
    """Extract mono PCM audio for signal processing."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_media),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(output_wav),
        ]
    )


def remux_audio_to_video(source_video: Path, replacement_audio: Path, output_video: Path) -> None:
    """Swap the audio track on a prepared MP4 clip."""
    output_video.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source_video),
            "-i",
            str(replacement_audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_video),
        ]
    )


def prepare_clips_from_manifest(
    dataset_name: str,
    manifest_path: Path,
    default_clip_type: str,
    limit: int | None = None,
    force: bool = False,
) -> list[dict]:
    """Extract prepared clips from a dataset manifest."""
    manifest_dir = manifest_path.parent
    records = load_manifest_records(manifest_path)
    clips = []

    for idx, record in enumerate(records):
        if limit is not None and len(clips) >= limit:
            break

        raw_video_path = record.get("video_path") or record.get("path")
        if not raw_video_path:
            continue

        raw_video = resolve_path(raw_video_path, manifest_dir)
        if not raw_video.exists():
            continue

        clip_id = record.get("clip_id") or f"{dataset_name}_{idx:04d}"
        stem = safe_slug(clip_id)
        dataset_dir = SOURCE_CLIPS_ROOT / dataset_name
        output_video = dataset_dir / f"{stem}.mp4"
        output_audio = dataset_dir / f"{stem}.wav"
        output_meta = dataset_dir / f"{stem}.json"

        start_s = float(record.get("start_s", 0.0) or 0.0)
        duration_s = record.get("duration_s")
        end_s = record.get("end_s")
        if end_s in (None, "", "None") and duration_s not in (None, "", "None"):
            end_s = start_s + float(duration_s)
        elif end_s in (None, "", "None"):
            end_s = start_s + DEFAULT_CLIP_DURATION_S
        end_s = float(end_s)

        if force or not (output_video.exists() and output_audio.exists() and output_meta.exists()):
            extract_video_clip(raw_video, output_video, start_s=start_s, end_s=end_s)
            extract_audio_wav(output_video, output_audio)

            clip_type = record.get("clip_type") or default_clip_type
            metadata = dict(record)
            metadata.update(
                {
                    "dataset": dataset_name,
                    "name": stem,
                    "type": clip_type,
                    "path": str(output_audio),
                    "source_audio_path": str(output_audio),
                    "source_video_path": str(output_video),
                    "metadata_path": str(output_meta),
                    "duration_s": max(0.0, end_s - start_s),
                    "segment_start_s": start_s,
                    "segment_end_s": end_s,
                    "visual_context_available": True,
                }
            )
            output_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        meta = json.loads(output_meta.read_text(encoding="utf-8"))
        clips.append(build_source_clip_record(meta))

    return clips


def copy_or_prepare_video(input_video: Path, output_video: Path) -> None:
    """Copy a clip when already prepared."""
    output_video.parent.mkdir(parents=True, exist_ok=True)
    if input_video.resolve() == output_video.resolve():
        return
    shutil.copy2(input_video, output_video)
