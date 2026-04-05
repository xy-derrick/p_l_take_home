"""Download the Greatest Hits dataset archive."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


URLS = {
    "fullres": "https://web.eecs.umich.edu/~ahowens/vis/vis-data.zip",
    "lowres": "https://web.eecs.umich.edu/~ahowens/vis/vis-data-256.zip",
    "features": "https://andrewowens.com/vis/vis-sfs.zip",
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
    parser = argparse.ArgumentParser(description="Download Greatest Hits archives")
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "data" / "raw_videos" / "greatest_hits"),
        help="Directory for Greatest Hits raw assets",
    )
    parser.add_argument(
        "--variant",
        choices=sorted(URLS),
        default="lowres",
        help="Archive variant to download",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract the downloaded zip archive after download",
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
    archive_name = Path(URLS[args.variant]).name
    archive_path = output_dir / archive_name
    download(URLS[args.variant], archive_path, args.skip_existing)

    if args.extract:
        extract_dir = output_dir / args.variant
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_dir)
        print(f"[ok]   extracted to {extract_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
