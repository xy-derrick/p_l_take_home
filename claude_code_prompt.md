# Claude Code Prompt: Audio Quality Verification Benchmark for AI-Generated Video

## Context

You are building evaluation infrastructure for audio quality in AI-generated video. This is a take-home assignment for Philo Labs (a company that builds reward models for video AI). The core insight: AI-generated videos increasingly look photorealistic, but their audio is conspicuously broken — dialogue doesn't match lip movements, sound effects fire at wrong moments, music ignores scene transitions. No systematic evaluation infrastructure exists for these failures. You are building that verification layer.

## Architecture Overview

The approach is to take clean real-world video clips, programmatically inject known audio corruptions at controlled severity levels, then test whether judge models can detect those corruptions. This gives us labeled ground truth for free (we know exactly what we broke and by how much). We compare two judge models:

1. **Gemini 2.5 Flash** (via OpenRouter API) — a VLM that ingests video+audio and makes semantic judgments
2. **Signal processing pipeline** — Whisper, librosa, pyannote, CLAP — modular expert tools that make precise measurements

The fundamental insight driving the design: audio-visual correspondence operates across multiple timescales (millisecond sync → second-level events → minute-level narrative mood), and no single tool covers the full hierarchy. Signal tools give precision without understanding; VLMs give understanding without precision. The benchmark explicitly tests where each approach succeeds and fails.

## Project Structure

```
audio-eval/
├── README.md
├── requirements.txt
├── setup.py
├── cli.py                          # Main CLI entry point
├── config.py                       # API keys, model endpoints, paths
├── data/
│   ├── source_clips/               # Clean source audio/video (downloaded or generated)
│   ├── corrupted/                   # Generated corrupted variants
│   └── sfx_library/                # Replacement sound effects for swap tasks
├── taxonomy/
│   └── failure_modes.py            # Structured failure mode definitions per pillar
├── seeds/
│   └── seed_tasks.py               # 5 seed task definitions with ground truth specs
├── corruption/
│   ├── __init__.py
│   ├── sync_shift.py               # A/V sync drift injection
│   ├── speaker_swap.py             # Speaker identity replacement at cut points
│   ├── artifact_inject.py          # TTS artifact / audio glitch injection
│   ├── sfx_mistime.py              # SFX onset temporal shift
│   └── music_mood_swap.py          # Replace music with mismatched-mood track
├── expansion/
│   ├── __init__.py
│   └── variant_generator.py        # Parameterized expansion from seeds to 500+ variants
├── scoring/
│   ├── __init__.py
│   ├── signal_scorer.py            # librosa/whisper/pyannote measurement pipeline
│   ├── gemini_scorer.py            # Gemini 2.5 Flash via OpenRouter
│   ├── rubric.py                   # 5 rubric dimensions with score definitions
│   └── aggregator.py               # Combine scores, produce per-task structured output
├── evaluation/
│   ├── __init__.py
│   ├── runner.py                   # Orchestrates: generate variants → corrupt → score
│   └── comparator.py               # Model comparison analysis, divergence detection
├── outputs/
│   └── tasks_and_rubrics.tsv       # Auto-generated output file
└── report/
    ├── report.tex                  # LaTeX report
    └── Makefile                    # Compile report
```

## Implementation Instructions

### Phase 1: Foundation (Do This First)

#### 1.1 Environment Setup

Create `requirements.txt`:
```
librosa>=0.10.0
soundfile
pydub
ffmpeg-python
numpy
pandas
scipy
openai          # for OpenRouter API calls
requests
tqdm
matplotlib      # for analysis plots
seaborn
```

Note: We are NOT using whisper/pyannote/CLAP in the actual implementation since they require GPU and heavy downloads. Instead, simulate their outputs for the prototype. The signal_scorer should be designed with clear interfaces so real models can be plugged in later, but use deterministic mock computations for now. This is explicitly allowed by the assignment ("Use sample audio/video, public datasets, or mock data").

#### 1.2 Source Data Strategy

Since we may not have network access or GPU, generate synthetic source data:

Create `data/generate_source_clips.py`:
- Use `librosa.tone()` and `numpy` to generate synthetic audio clips that simulate speech (modulated tones), sound effects (impulse + decay envelopes), and music (chord progressions with simple synth)
- Each clip should be 10-30 seconds, saved as WAV
- Generate 10-15 source clips covering:
  - 5 "speech" clips: modulated sine waves simulating voice with periodic envelope (simulating syllables)
  - 4 "sfx" clips: impulse events at known timestamps (simulating impacts, door closes, etc.)
  - 4 "music" clips: sustained tonal content with known key/tempo
  - 2 "mixed" clips: combination of speech + background music
- For each clip, store metadata (event timestamps, frequency content, envelope) as ground truth JSON alongside the WAV

This synthetic approach is actually BETTER for a benchmark prototype because ground truth is mathematically exact.

If network access is available, alternatively download a few clips from VGGSound or use ffmpeg to extract audio from any sample video files.

#### 1.3 Config

Create `config.py`:
```python
import os

# API Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
GEMINI_MODEL = "google/gemini-2.5-flash-preview"

# Paths
SOURCE_CLIPS_DIR = "data/source_clips"
CORRUPTED_DIR = "data/corrupted"
SFX_LIBRARY_DIR = "data/sfx_library"
OUTPUT_TSV = "outputs/tasks_and_rubrics.tsv"

# Expansion parameters
SEVERITY_LEVELS = {
    "sync_drift_ms": [50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000],
    "sfx_shift_ms": [50, 100, 200, 500, 1000],
    "snr_db": [20, 10, 5, 0, -3],
    "speaker_similarity_threshold": [0.9, 0.7, 0.5, 0.3, 0.1],
    "mood_distance": [0.2, 0.4, 0.6, 0.8, 1.0],  # 0=same mood, 1=opposite
}

# Rubric thresholds (maps measured values to 1-5 scores)
SYNC_THRESHOLDS = [500, 200, 100, 40]       # >500→1, 200-500→2, 100-200→3, 40-100→4, <40→5
SFX_TIMING_THRESHOLDS = [500, 200, 100, 50]
SPEAKER_SIM_THRESHOLDS = [0.3, 0.5, 0.7, 0.85]  # cosine similarity breakpoints
```

### Phase 2: Failure Taxonomy & Seed Tasks

#### 2.1 Failure Modes

Create `taxonomy/failure_modes.py` — a structured definition of all failure modes. Each failure mode should be a dataclass or dict containing:
- `id`: unique identifier
- `pillar`: "voice" | "sfx" | "music"
- `name`: human-readable name
- `description`: what the failure looks like
- `expected_correct`: what correct audio would look/sound like
- `timescale`: "microsecond" | "millisecond" | "second" | "multi-second" | "minute"
- `detection_tier`: 1 (signal-verifiable) | 2 (embedding-verifiable) | 3 (semantically-verifiable)
- `examples`: list of 2-3 concrete examples

Define these failure modes:

**Voice pillar:**
1. `lip_sync_drift` — audio leads or lags visuals. Tier 1, millisecond scale. Examples: (a) dialogue arrives 300ms before mouth moves, (b) laugh sound 500ms after smile, (c) progressive drift that worsens over 30s.
2. `speaker_identity_break` — voice timbre/pitch shifts mid-clip. Tier 2, multi-second scale. Examples: (a) male voice becomes female after a cut, (b) pitch shifts 2 semitones mid-sentence, (c) accent changes between shots.
3. `tts_artifacts` — robotic prosody, glitches. Tier 1, microsecond scale. Examples: (a) metallic buzzing on sibilants, (b) unnatural 500ms pause mid-word, (c) phoneme repetition stutter.

**Sound Effects pillar:**
4. `sfx_temporal_misfire` — sound effect at wrong time. Tier 1, millisecond scale. Examples: (a) glass break 1s after visual impact, (b) footstep sounds between steps not during, (c) gunshot before muzzle flash.
5. `sfx_semantic_mismatch` — wrong sound for action. Tier 3, second scale. Examples: (a) metal clang for wooden door, (b) cat meow when dog is visible, (c) indoor reverb for outdoor scene.
6. `missing_phantom_sfx` — visible action with no sound or sound with no visual cause. Tier 3, second scale. Examples: (a) car crashes silently, (b) random explosion sound with static scene, (c) footsteps with no one walking.

**Music pillar:**
7. `mood_scene_mismatch` — music emotion contradicts scene. Tier 3, minute scale. Examples: (a) upbeat pop during funeral scene, (b) horror stingers during comedy, (c) lullaby during action chase.
8. `transition_ignorance` — music continues unchanged across dramatic cuts. Tier 3, minute scale. Examples: (a) same track plays through scene change from indoor to outdoor, (b) tempo unchanged when action accelerates, (c) music continues through dialogue that should be unscored.
9. `abrupt_music_cut` — music stops mid-phrase without fade. Tier 2, second scale. Examples: (a) chord cuts off mid-sustain at scene boundary, (b) melody stops before resolving, (c) beat drops out without any transition.

#### 2.2 Seed Tasks

Create `seeds/seed_tasks.py` — define 5 seed tasks. Each seed task is a dataclass or dict:

```python
@dataclass
class SeedTask:
    seed_id: str
    name: str
    pillar: str                    # "voice" | "sfx" | "music"
    failure_mode_id: str           # references taxonomy
    tier: int                      # 1, 2, or 3
    description: str               # what the task tests
    corruption_fn: str             # name of corruption function to apply
    corruption_params: dict        # default parameters
    ground_truth_type: str         # "exact" | "measured" | "semantic"
    ground_truth_extraction: str   # how to compute ground truth
    question_template: str         # template for VLM prompt
    signal_metric: str             # what signal_scorer measures
    pass_criteria: str             # what constitutes detection
```

Define these 5 seeds:

**S1: Sync Drift Detection** (Tier 1, Voice)
- Corruption: shift audio track by N milliseconds using ffmpeg/pydub
- Ground truth: exact (we know the offset we injected)
- Signal metric: cross-correlation lag between original and shifted
- VLM question: "Watch this video clip. Is the audio synchronized with the visual events? If not, estimate by how much the audio leads or lags the visuals in milliseconds. Respond with JSON: {synced: bool, offset_ms: int, confidence: float}"
- Pass criteria: model correctly identifies sync vs. no-sync AND estimates offset within 100ms

**S2: Speaker Identity Consistency** (Tier 2, Voice)
- Corruption: replace audio segment after a cut point with a different speaker's voice (different pitch, timbre)
- Ground truth: measured (known swap point, known speaker difference)
- Signal metric: cosine similarity of speaker embeddings before/after swap point
- VLM question: "Listen to the speakers in this clip. Does the same person speak throughout, or does the speaker identity change at some point? If it changes, at approximately what timestamp? Respond with JSON: {consistent: bool, change_timestamp_s: float, confidence: float}"
- Pass criteria: model correctly identifies swap vs. no-swap AND locates swap within 2 seconds

**S3: SFX Onset Misalignment** (Tier 1, SFX)
- Corruption: detect audio onset events, shift them by N milliseconds
- Ground truth: exact (we know the shift amount and which events)
- Signal metric: onset detection delta between original and shifted
- VLM question: "Watch and listen to this clip. Are the sound effects properly timed with the visual events? Identify any sounds that occur too early or too late relative to what you see. Respond with JSON: {aligned: bool, misaligned_events: [{description: str, offset_ms: int}], confidence: float}"
- Pass criteria: model correctly identifies aligned vs. misaligned

**S4: Audio Artifact Detection** (Tier 1, cross-pillar)
- Corruption: inject clicks, dropouts, spectral anomalies at known timestamps
- Ground truth: exact (we know what we injected and where)
- Signal metric: spectral flux anomaly detection, zero-crossing rate spikes
- VLM question: "Listen to this audio. Are there any artifacts, glitches, clicks, dropouts, or unnatural sounds? If so, describe them and note approximately when they occur. Respond with JSON: {clean: bool, artifacts: [{type: str, timestamp_s: float, severity: str}], confidence: float}"
- Pass criteria: model correctly identifies clean vs. corrupted AND locates artifacts within 1 second

**S5: Music Mood Alignment** (Tier 2-3, Music)
- Corruption: replace background music with a track of different/opposite mood (use valence/energy classification)
- Ground truth: measured (mood distance between original and replacement, from a pre-computed mood label)
- Signal metric: music mood classifier output (simulated: use spectral centroid + tempo as proxy for energy/valence)
- VLM question: "Watch this scene and listen to the background music. Does the music's mood (energy, emotion, tempo) match the visual scene? Rate the match from 1 (completely wrong mood) to 5 (perfect match). Respond with JSON: {mood_match_score: int, scene_mood: str, music_mood: str, explanation: str}"
- Pass criteria: model score correlates with actual mood distance

### Phase 3: Corruption Modules

Each corruption module should:
- Take an input audio file path (WAV) + corruption parameters
- Return a corrupted audio file path + ground truth metadata dict
- Be deterministic (same params → same output)
- Log what it did

#### 3.1 `corruption/sync_shift.py`

```python
def apply_sync_shift(audio_path: str, offset_ms: int, output_path: str) -> dict:
    """
    Shift audio by offset_ms milliseconds.
    Positive = audio lags (delayed), Negative = audio leads.
    
    Implementation:
    - Load audio with pydub or soundfile
    - If positive offset: prepend silence of offset_ms duration
    - If negative offset: trim first abs(offset_ms) from audio, pad end
    - Save to output_path
    
    Returns ground truth dict:
    {
        "corruption_type": "sync_shift",
        "offset_ms": offset_ms,
        "direction": "lag" | "lead",
        "original_path": audio_path,
        "corrupted_path": output_path
    }
    """
```

#### 3.2 `corruption/speaker_swap.py`

```python
def apply_speaker_swap(audio_path: str, replacement_audio_path: str, 
                        swap_point_s: float, output_path: str) -> dict:
    """
    Replace audio after swap_point_s with audio from a different source.
    Apply short crossfade (50ms) to avoid click artifacts at splice point.
    
    For synthetic data: generate two tones at different frequencies to simulate
    different speakers. E.g., 200Hz base for speaker A, 300Hz for speaker B.
    
    Returns ground truth dict:
    {
        "corruption_type": "speaker_swap",
        "swap_point_s": swap_point_s,
        "original_speaker_freq": float,
        "replacement_speaker_freq": float,
        "similarity_score": float  # computed from frequency ratio
    }
    """
```

#### 3.3 `corruption/artifact_inject.py`

```python
def inject_artifacts(audio_path: str, artifact_type: str, 
                     timestamps: list[float], severity: float,
                     output_path: str) -> dict:
    """
    Inject audio artifacts at specified timestamps.
    
    artifact_type options:
    - "click": single-sample spike (multiply sample by severity * 10)
    - "dropout": zero out audio for duration_ms at each timestamp
    - "spectral": add narrow-band noise burst at each timestamp
    - "stutter": repeat 50ms segment at each timestamp
    
    Returns ground truth dict with exact artifact locations and types.
    """
```

#### 3.4 `corruption/sfx_mistime.py`

```python
def apply_sfx_mistime(audio_path: str, event_timestamps: list[float],
                       shift_ms: int, output_path: str) -> dict:
    """
    For audio with known event onsets (e.g., impact sounds at known times),
    shift the audio segments around each event by shift_ms.
    
    For synthetic data: the source clips have events at known timestamps
    (stored in the source clip metadata JSON). Extract the audio region
    around each event (±200ms window), shift it by shift_ms, 
    crossfade at boundaries.
    
    Returns ground truth dict with original and shifted timestamps.
    """
```

#### 3.5 `corruption/music_mood_swap.py`

```python
def apply_music_mood_swap(audio_path: str, replacement_music_path: str,
                           original_mood: dict, replacement_mood: dict,
                           output_path: str) -> dict:
    """
    Replace the music/tonal content in the audio with a different mood track.
    
    For synthetic data: 
    - "happy" music = major key, fast tempo, high spectral centroid
    - "sad" music = minor key, slow tempo, low spectral centroid
    - "tense" music = dissonant intervals, irregular rhythm
    - "calm" music = consonant, slow, low energy
    
    Generate these synthetically using numpy (sine waves in musical intervals).
    
    original_mood and replacement_mood are dicts with:
    {valence: float 0-1, energy: float 0-1, tempo_bpm: int}
    
    mood_distance = euclidean distance in (valence, energy) space
    
    Returns ground truth dict with mood labels and distance.
    """
```

### Phase 4: Variant Expansion

#### 4.1 `expansion/variant_generator.py`

```python
def generate_variants(seeds: list[SeedTask], source_clips: list[dict],
                       severity_config: dict) -> list[TaskVariant]:
    """
    Expand seed tasks into full variant families.
    
    For each seed:
      For each compatible source clip:
        For each severity level in the seed's corruption parameter range:
          Create a TaskVariant with unique task_id
    
    TaskVariant dataclass:
    {
        task_id: str,           # e.g., "S1_clip03_200ms"
        seed_id: str,           # e.g., "S1"
        source_clip: str,       # path to source
        corruption_type: str,
        corruption_params: dict,
        ground_truth: dict,     # filled after corruption is applied
        audio_pillar: str,
        tier: int,
        difficulty_estimate: str  # "easy" | "medium" | "hard" based on severity
    }
    
    Also generate "clean" variants (no corruption) as negative controls —
    approximately 20% of total variants should be uncorrupted to test
    false positive rate.
    
    Target: 500+ total variants from 5 seeds × 10-15 clips × 5-10 severity levels
    + clean controls.
    """
```

Include cross-pillar combination variants:
```python
def generate_cross_pillar_variants(base_variants: list[TaskVariant]) -> list[TaskVariant]:
    """
    Take pairs of variants from different seeds/pillars and combine their corruptions.
    E.g., sync drift + artifact injection on the same clip.
    Select ~50 combinations to add to the variant pool.
    """
```

### Phase 5: Scoring Pipeline

#### 5.1 `scoring/rubric.py`

Define the 5 scoring dimensions as a structured rubric:

```python
RUBRIC_DIMENSIONS = [
    {
        "id": "av_sync",
        "name": "A/V Synchronization Accuracy",
        "tier": 1,
        "scale": "1-5",
        "levels": {
            1: "Severe offset (>500ms), clearly perceivable lag/lead",
            2: "Significant offset (200-500ms), noticeable mismatch",
            3: "Moderate offset (100-200ms), detectable on attention",
            4: "Minor offset (40-100ms), barely perceptible",
            5: "Aligned (<40ms), perceptually synchronous"
        },
        "measurement": "cross_correlation_lag_ms",
        "thresholds_ms": [500, 200, 100, 40],
        "automation": "fully_automatic"
    },
    {
        "id": "artifact_quality",
        "name": "Audio Artifact / Signal Quality",
        "tier": 1,
        "scale": "1-5",
        "levels": {
            1: "Multiple severe artifacts (clicks, dropouts, distortion)",
            2: "Noticeable artifacts that distract from content",
            3: "Minor artifacts, not distracting",
            4: "Very subtle artifacts, only detectable on close listen",
            5: "Clean, no detectable artifacts"
        },
        "measurement": "artifact_count_and_severity",
        "automation": "fully_automatic"
    },
    {
        "id": "speaker_consistency",
        "name": "Speaker Identity Consistency",
        "tier": 2,
        "scale": "1-5",
        "levels": {
            1: "Clearly different speaker (different gender/age)",
            2: "Different speaker (same gender, different voice)",
            3: "Noticeable voice quality shift",
            4: "Minor variation within natural range",
            5: "Fully consistent speaker identity"
        },
        "measurement": "speaker_embedding_cosine_similarity",
        "thresholds": [0.3, 0.5, 0.7, 0.85],
        "automation": "embedding_based"
    },
    {
        "id": "semantic_match",
        "name": "Audio-Visual Semantic Match",
        "tier": 3,
        "scale": "1-5",
        "levels": {
            1: "Audio is completely unrelated to visual content",
            2: "Audio is from wrong category for the visual scene",
            3: "Audio is plausible but not well matched",
            4: "Audio matches well with minor discrepancies",
            5: "Audio perfectly matches the visual content"
        },
        "measurement": "vlm_judgment",
        "automation": "requires_vlm"
    },
    {
        "id": "music_coherence",
        "name": "Music/Mood Coherence",
        "tier": 3,
        "scale": "1-5",
        "levels": {
            1: "Music mood directly contradicts scene emotion",
            2: "Music mood is inappropriate for the scene",
            3: "Music mood is neutral/ambiguous relative to scene",
            4: "Music mood is appropriate with minor issues",
            5: "Music mood perfectly matches and transitions naturally"
        },
        "measurement": "mood_distance + vlm_judgment",
        "automation": "hybrid"
    }
]
```

#### 5.2 `scoring/signal_scorer.py`

```python
class SignalScorer:
    """
    Scores audio using signal processing techniques.
    
    For the prototype, this uses deterministic computations on the synthetic audio
    rather than actual ML models. The interface is designed so real models
    (Whisper, pyannote, CLAP) can be plugged in later.
    
    Methods:
    - score_sync(original_audio, corrupted_audio) -> {offset_ms, score_1_5}
      Uses cross-correlation to find lag. For synthetic audio, can compute
      directly from known parameters.
    
    - score_artifacts(audio_path, ground_truth) -> {artifact_count, locations, score_1_5}
      Detect spectral anomalies. For prototype, compare against known injection points.
    
    - score_speaker_consistency(audio_path, swap_point_s) -> {similarity, score_1_5}
      Compare spectral features before/after swap point.
      For synthetic: compare dominant frequencies in two segments.
    
    - score_sfx_timing(audio_path, expected_onsets, actual_onsets) -> {deltas, score_1_5}
      Compare expected vs actual onset times.
    
    - score_mood(audio_path, expected_mood) -> {detected_mood, distance, score_1_5}
      Use spectral centroid + tempo as mood proxy.
    
    Each method returns a dict with raw measurements AND a 1-5 rubric score
    computed by applying the thresholds from rubric.py.
    """
```

IMPORTANT: The signal scorer should work with the ground truth metadata from the corruption step. Since we injected the corruptions, we can compute "detection" scores based on how closely a hypothetical detector would match our known ground truth. For the prototype, the signal scorer essentially confirms the ground truth and produces rubric-mapped scores. This is valid because we're testing whether the *judge models* can detect what we know is there.

#### 5.3 `scoring/gemini_scorer.py`

```python
class GeminiScorer:
    """
    Scores audio quality using Gemini 2.5 Flash via OpenRouter API.
    
    For each task variant, constructs a prompt based on the seed's question_template,
    sends the audio (or audio+video if available) to Gemini, parses the structured
    JSON response, and maps it to rubric scores.
    
    Key design decisions:
    - Send audio as base64-encoded WAV in the API call
    - Use structured JSON output format to get parseable scores
    - Include the rubric level definitions in the prompt so Gemini scores
      on the same scale
    - Handle API errors gracefully (retry with backoff, log failures)
    
    If OPENROUTER_API_KEY is not set, fall back to MOCK mode:
    - Generate plausible mock responses that simulate Gemini's expected behavior
    - Mock should be worse at Tier 1 tasks (timing precision) and better at 
      Tier 3 tasks (semantic judgment) to simulate the expected model behavior
    - Add noise to mock responses to simulate imperfect detection
    
    The mock mode is important because:
    1. It lets the full pipeline run end-to-end without API access
    2. It demonstrates what the EXPECTED findings would be
    3. The assignment allows mock data explicitly
    
    Methods:
    - score_task(task_variant: TaskVariant) -> dict
      Main entry point. Builds prompt, calls API or mock, parses response.
    
    - build_prompt(task_variant: TaskVariant) -> str
      Constructs the evaluation prompt including rubric dimensions.
    
    - parse_response(response: str) -> dict
      Extracts structured scores from model output.
    
    - mock_response(task_variant: TaskVariant) -> dict
      Generates simulated response based on task parameters and expected
      model behavior patterns.
    """
```

**Mock behavior specification for Gemini:**
```
Tier 1 tasks (sync, artifacts, SFX timing):
  - Detection rate: 40-60% for subtle (severity < 0.3), 70-85% for moderate, 90%+ for severe
  - Timing precision: ±200ms noise added to true offset (VLMs are imprecise temporally)
  - False positive rate: ~10% (occasionally flags clean audio)

Tier 2 tasks (speaker consistency):
  - Detection rate: 50-70% for subtle, 80-90% for obvious
  - Timestamp precision: ±2s (can detect the change but not pinpoint it)

Tier 3 tasks (mood alignment, semantic match):
  - Detection rate: 60-80% (VLMs are relatively good at semantic judgment)
  - Accuracy on mood scoring: correlation ~0.6-0.7 with ground truth
```

**Mock behavior specification for Signal Pipeline:**
```
Tier 1 tasks:
  - Detection rate: 90-98% (signal tools are precise for what they measure)
  - Timing precision: ±5ms (very accurate)
  - False positive rate: ~3%

Tier 2 tasks:
  - Detection rate: 70-85% (embedding similarity is decent)
  - Threshold sensitivity: performance drops at subtle similarities

Tier 3 tasks:
  - Detection rate: 30-50% (signal tools can't judge semantics well)
  - Mood scoring: correlation ~0.3 with ground truth (using spectral proxy)
```

These mock profiles should make the model comparison genuinely interesting and demonstrate the expected divergence pattern.

### Phase 6: Evaluation Pipeline

#### 6.1 `evaluation/runner.py`

```python
class EvaluationRunner:
    """
    Orchestrates the full pipeline:
    
    1. Load seed tasks
    2. Load/generate source clips
    3. Generate variants (calls variant_generator)
    4. Apply corruptions to create test audio (calls corruption modules)
    5. Score each variant with both models (calls signal_scorer + gemini_scorer)
    6. Aggregate results
    7. Write tasks_and_rubrics.tsv
    
    The TSV should have columns:
    task_id, seed_id, task_description, source_clip, audio_pillar, tier,
    corruption_type, corruption_severity, model,
    av_sync_score, artifact_quality_score, speaker_consistency_score,
    semantic_match_score, music_coherence_score,
    detection_correct, ground_truth_label
    
    Each task variant gets TWO rows (one per model).
    
    CLI integration:
    - runner.run_all() executes everything
    - runner.run_seed(seed_id) runs just one seed for debugging
    - Progress bar via tqdm
    """
```

#### 6.2 `evaluation/comparator.py`

```python
class ModelComparator:
    """
    Analyzes results from both models to surface differential failure modes.
    
    Analyses to produce:
    
    1. Per-seed accuracy table:
       For each seed, what % of variants did each model correctly detect?
       Group by severity level to show the detection curve.
    
    2. Tier-level performance:
       Average accuracy per tier per model — this should show the crossover
       where signal pipeline beats Gemini on Tier 1 but loses on Tier 3.
    
    3. Divergence matrix:
       For each variant, classify into:
       - Both correct (easy task)
       - Both wrong (hard task)
       - Signal correct, Gemini wrong (signal advantage)
       - Signal wrong, Gemini correct (semantic advantage)
       Report counts and examples for each quadrant.
    
    4. Difficulty calibration report:
       For each seed, report what severity range produces 35-75% pass rate
       for each model. These are the "discriminating" variants.
    
    5. Severity-accuracy curves:
       Plot detection accuracy vs. corruption severity for each seed × model.
       Save as PNG in outputs/.
    
    Output: Print summary tables to stdout, save detailed analysis to
    outputs/comparison_report.json, generate plots to outputs/plots/.
    """
```

### Phase 7: CLI

#### `cli.py`

```python
"""
CLI entry point. Single command to reproduce everything.

Usage:
    python cli.py generate     # Generate source clips + variants
    python cli.py score        # Run both scorers on all variants
    python cli.py compare      # Produce comparison analysis
    python cli.py run          # All of the above in sequence
    python cli.py run --seeds S1,S2  # Run specific seeds only
    python cli.py run --mock   # Force mock mode (no API calls)

The default mode (no flags) should run everything end-to-end with mock
scorers and produce the TSV + comparison report. This is the single
reproducible command the assignment asks for.
"""
```

### Phase 8: Report (LaTeX)

Create `report/report.tex` — a LaTeX document, 1500-2500 words. Structure:

```latex
\documentclass[11pt]{article}
\usepackage{booktabs, graphicx, hyperref, amsmath, geometry}
\geometry{margin=1in}

\title{Audio Quality Verification in AI-Generated Video: \\
A Multi-Timescale Evaluation Framework}
\author{[Name]}
\date{April 2026}

\begin{document}
\maketitle

\begin{abstract}
% 100 words: The problem, the approach, key finding
\end{abstract}

\section{Introduction: Why Audio Verification is the Critical Bottleneck}
% ~300 words
% - Visual fidelity has raced ahead, audio remains broken
% - No systematic evaluation infrastructure exists
% - The three pillars: Voice, SFX, Music
% - Why this matters: audio is 50% of the viewing experience

\section{The Multi-Timescale Challenge}
% ~400 words — this is the first-principles section
% - Audio-visual correspondence spans 5 orders of magnitude
% - Microsecond (artifacts) → millisecond (sync) → second (events) → 
%   multi-second (identity) → minute (mood/narrative)
% - No single evaluation tool covers the hierarchy
% - Current VLMs operate well at second-to-minute scale but poorly at 
%   millisecond scale. Signal tools are the reverse.
% - The compositionality problem: real audio is layered (dialogue + SFX + music)
%   and layers interact (music masks sync failures)

\section{Signal Processing vs. Model-Based Detection: Key Tradeoffs}
% ~400 words
% - Signal processing: precise, reproducible, narrow scope
%   (can measure sync offset in ms, can't judge mood)
% - VLM-based: holistic understanding, imprecise, broad scope
%   (can judge mood appropriateness, hallucinates timing)
% - Present empirical findings from model comparison
%   (include table or figure from comparator output)
% - The crossover point: at what timescale/task tier does VLM 
%   start outperforming signal pipeline?

\section{Toward Automated Audio Task Generation}
% ~300 words
% - The corruption-based approach: inject known failures, test detection
% - Expansion pipeline: 5 seeds → 500+ variants via parameterization
% - Difficulty calibration: targeting the 35-75% discriminating band
% - Ground truth strategy: programmatic for Tier 1, embedding for Tier 2,
%   agentic pre-labeling for Tier 3
% - Path to minimal human input: human review only for Tier 3 edge cases

\section{Rubric as Reward Signal for Training}
% ~300 words
% - Per-dimension scores give structured reward signal
% - Tier 1 dimensions provide precise gradient (timing-based loss)
% - Tier 3 dimensions provide softer semantic signal (preference-based)
% - Tiered reward > single holistic score because it tells the model WHAT to fix
% - Connection to RLVR: these rubric scores can serve as verifiable rewards
%   for reinforcement learning from verifiable rewards

\section{Limitations and Future Work}
% ~200 words
% - Edge cases: intentional silence, diegetic ambiguity, stylistic choices
% - Current prototype uses synthetic audio; real video would add complexity
% - Scaling to longer sequences (full scenes, not just clips)
% - Cross-cultural audio expectations
% - Integration with visual quality benchmarks for holistic evaluation

\end{document}
```

Compile with: `pdflatex report.tex` (run twice for references)

### Phase 9: README

Create `README.md`:

```markdown
# Audio Quality Verification Benchmark for AI-Generated Video

## Overview
Evaluation infrastructure for detecting audio failures in AI-generated video.
Tests audio quality across three pillars (Voice, Sound Effects, Music) using
a corruption-based benchmark approach with two judge models.

## Quick Start
```bash
pip install -r requirements.txt
python cli.py run          # Full pipeline: generate → score → compare
```

## Architecture
- **Corruption-based evaluation**: inject known audio failures into clean clips
  at controlled severity levels
- **Two judge models compared**:
  - Signal processing pipeline (librosa-based measurements)
  - Gemini 2.5 Flash (semantic VLM judgment, via OpenRouter)
- **5 seed tasks** expanded to 500+ variants via parameterized severity scaling

## Outputs
- `outputs/tasks_and_rubrics.tsv` — full results matrix
- `outputs/comparison_report.json` — model comparison analysis
- `outputs/plots/` — severity-accuracy curves
- `report/report.pdf` — research report

## Agent Prompts Used
[Document the prompts you used with Claude Code throughout]

## API Costs
[Document actual API spend, should be well under $200]
```

## Important Notes

1. **Mock mode is first-class**: The entire pipeline MUST work end-to-end without any API keys or network access, using synthetic audio and mock model responses. API mode is a bonus.

2. **Ground truth is king**: Every corruption has exact, programmatically verifiable ground truth. This is the strongest design choice — never compromise it.

3. **The comparison is the insight**: The most important output is not the scores themselves but WHERE the two models diverge. The comparator module is where the real value lives.

4. **Code quality matters**: Clean modules, type hints, docstrings. Each module should be independently testable. The assignment weights implementation at 20% but "clean, modular, end-to-end" is the bar.

5. **The report should not just describe what you built**: It should argue WHY this approach is correct from first principles. The multi-timescale framing and the automation boundary are the intellectual contributions.

6. **Estimate ~4 hours total**: Don't over-engineer. Get the pipeline working end-to-end first, then polish.
