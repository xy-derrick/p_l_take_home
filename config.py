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
GEMINI_MODEL = "google/gemini-2.5-flash-preview"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Paths
SOURCE_CLIPS_DIR = os.path.join(os.path.dirname(__file__), "data", "source_clips")
CORRUPTED_DIR = os.path.join(os.path.dirname(__file__), "data", "corrupted")
SFX_LIBRARY_DIR = os.path.join(os.path.dirname(__file__), "data", "sfx_library")
OUTPUT_TSV = os.path.join(os.path.dirname(__file__), "outputs", "tasks_and_rubrics.tsv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "plots")

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
