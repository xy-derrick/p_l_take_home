"""Download a small AVA subset and required annotations."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path


AVA_BASE = "https://s3.amazonaws.com/ava-dataset"
URLS = {
    "file_names": f"{AVA_BASE}/annotations/ava_speech_file_names_v1.txt",
    "speech_labels": f"{AVA_BASE}/annotations/ava_speech_labels_v1.csv",
    "activespeaker_train": f"{AVA_BASE}/annotations/ava_activespeaker_train_v1.0.tar.bz2",
    "activespeaker_val": f"{AVA_BASE}/annotations/ava_activespeaker_val_v1.0.tar.bz2",
}


def download(url: str, output_path: Path, skip_existing: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and output_path.exists():
        print(f"[skip] {output_path}")
        return

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, output_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    print(f"[ok]   {output_path}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Download AVA videos and annotations")
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "data" / "raw_videos" / "ava"),
        help="Directory for AVA raw assets",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of videos to download from the official file list",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start offset into the AVA file list",
    )
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help="Download annotations only, skip MP4 files",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    annotations_dir = output_dir / "annotations"
    videos_dir = output_dir / "videos"

    download(URLS["file_names"], annotations_dir / "ava_speech_file_names_v1.txt", args.skip_existing)
    download(URLS["speech_labels"], annotations_dir / "ava_speech_labels_v1.csv", args.skip_existing)
    download(URLS["activespeaker_train"], annotations_dir / "ava_activespeaker_train_v1.0.tar.bz2", args.skip_existing)
    download(URLS["activespeaker_val"], annotations_dir / "ava_activespeaker_val_v1.0.tar.bz2", args.skip_existing)

    if args.annotations_only:
        return 0

    file_list_path = annotations_dir / "ava_speech_file_names_v1.txt"
    file_names = [
        line.strip()
        for line in file_list_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    selected = file_names[args.start_index: args.start_index + args.count]

    for file_name in selected:
        download(
            f"{AVA_BASE}/trainval/{file_name}",
            videos_dir / file_name,
            args.skip_existing,
        )

    print(f"Downloaded {len(selected)} AVA videos to {videos_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
