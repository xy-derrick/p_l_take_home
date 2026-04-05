import os
from pathlib import Path

# Load .env file if present
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

# API Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
GEMINI_MODEL = "google/gemini-2.5-flash"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEO_DIR = DATA_DIR / "raw_videos"
MANIFEST_DIR = DATA_DIR / "manifests"
SOURCE_CLIPS_ROOT = DATA_DIR / "source_clips"
CORRUPTED_ROOT = DATA_DIR / "corrupted"
SFX_LIBRARY_ROOT = DATA_DIR / "sfx_library"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
PLOTS_ROOT = OUTPUT_ROOT / "plots"
LOGS_ROOT = PROJECT_ROOT / "logs"
REPORT_ROOT = PROJECT_ROOT / "report"

SOURCE_CLIPS_DIR = str(SOURCE_CLIPS_ROOT)
CORRUPTED_DIR = str(CORRUPTED_ROOT)
SFX_LIBRARY_DIR = str(SFX_LIBRARY_ROOT)
OUTPUT_TSV = str(OUTPUT_ROOT / "tasks_and_rubrics.tsv")
OUTPUT_DIR = str(OUTPUT_ROOT)
PLOTS_DIR = str(PLOTS_ROOT)
LOGS_DIR = str(LOGS_ROOT)
REPORT_DIR = str(REPORT_ROOT)

SUPPORTED_DATASETS = ("synthetic", "ava", "greatest_hits", "condensed_movies")
DEFAULT_DATASETS = ("synthetic",)
DATASET_CONFIG = {
    "ava": {
        "raw_dir": RAW_VIDEO_DIR / "ava",
        "manifest": MANIFEST_DIR / "ava_clips.jsonl",
    },
    "greatest_hits": {
        "raw_dir": RAW_VIDEO_DIR / "greatest_hits",
        "manifest": MANIFEST_DIR / "greatest_hits_clips.jsonl",
    },
    "condensed_movies": {
        "raw_dir": RAW_VIDEO_DIR / "condensed_movies",
        "manifest": MANIFEST_DIR / "condensed_movies_clips.jsonl",
    },
}

DEFAULT_CLIP_DURATION_S = 8.0
MAX_CLIPS_PER_DATASET = 25
VIDEO_TARGET_FPS = 24
VIDEO_HEIGHT = 720

# Expansion parameters
SEVERITY_LEVELS = {
    "sync_drift_ms": [50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000],
    "sfx_shift_ms": [50, 100, 200, 500, 1000],
    "snr_db": [20, 10, 5, 0, -3],
    "speaker_similarity_threshold": [0.9, 0.7, 0.5, 0.3, 0.1],
    "mood_distance": [0.2, 0.4, 0.6, 0.8, 1.0],
}

# Rubric thresholds
SYNC_THRESHOLDS = [500, 200, 100, 40]
SFX_TIMING_THRESHOLDS = [500, 200, 100, 50]
SPEAKER_SIM_THRESHOLDS = [0.3, 0.5, 0.7, 0.85]

SAMPLE_RATE = 22050


def ensure_runtime_dirs() -> None:
    """Create runtime directories used by the benchmark."""
    for path in [
        DATA_DIR,
        RAW_VIDEO_DIR,
        MANIFEST_DIR,
        SOURCE_CLIPS_ROOT,
        CORRUPTED_ROOT,
        SFX_LIBRARY_ROOT,
        OUTPUT_ROOT,
        PLOTS_ROOT,
        LOGS_ROOT,
        REPORT_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)
