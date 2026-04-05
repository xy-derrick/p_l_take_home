# Audio Quality Verification Benchmark for AI-Generated Video

## Overview
This repo builds a corruption-based benchmark for audio failures in AI-generated video.
It supports two source modes:

- `synthetic`: fully local WAV generation with exact ground truth
- `real-video`: public-dataset clips prepared as MP4 + WAV pairs and sent to Gemini as video

The benchmark compares two judges:

- `SignalScorer`: deterministic mock of a modular signal-processing pipeline
- `GeminiScorer`: OpenRouter-based multimodal judge configured for `google/gemini-2.5-flash-preview`, with video input when available and mock fallback otherwise

## Supported Datasets
- `synthetic`: generated locally, no external assets required
- `ava`: intended for speech / active-speaker clips
- `greatest_hits`: intended for visually grounded impact / SFX clips (not implemented)
- `condensed_movies`: intended for scene-level music / mood clips (not implemented)

Real-video datasets are prepared from local manifest files under `data/manifests/`.

## Quick Start
```bash
pip install -r requirements.txt
python cli.py run --datasets synthetic
```

## Real-Video Workflow
1. Put raw dataset videos somewhere under `data/raw_videos/<dataset>/` or reference absolute paths.
2. Create a manifest file:
   - `data/manifests/ava_clips.jsonl`
   - `data/manifests/greatest_hits_clips.jsonl`
   - `data/manifests/condensed_movies_clips.jsonl`
3. Extract prepared clips:
```bash
python cli.py prepare-datasets --datasets ava,greatest_hits
python cli.py extract-clips --datasets ava,greatest_hits
```
4. Run the benchmark:
```bash
python cli.py run --datasets ava,greatest_hits --no-mock
```

## Cmd Download Helpers
All helper scripts below are runnable from `cmd.exe`:

```cmd
cd /d F:\philo_lab\audio-eval

scripts\download_ava_subset.cmd --count 10 --skip-existing
scripts\build_ava_manifest.cmd --max-clips-per-video 3

scripts\download_greatest_hits.cmd --variant lowres --extract --skip-existing
scripts\build_greatest_hits_manifest.cmd --max-clips-per-video 3
```

The AVA downloader writes to:

- `data\raw_videos\ava\videos`
- `data\raw_videos\ava\annotations`

The Greatest Hits downloader writes to:

- `data\raw_videos\greatest_hits`

You can then run:

```cmd
python cli.py prepare-datasets --datasets ava,greatest_hits
python cli.py run --datasets ava,greatest_hits --mock
```

## Manifest Schema
Each manifest record can be JSONL, JSON, or CSV and should contain at least:

```json
{
  "clip_id": "ava_speaker_0001",
  "video_path": "data/raw_videos/ava/example.mp4",
  "start_s": 12.0,
  "end_s": 20.0,
  "clip_type": "speech",
  "event_timestamps_s": [14.2, 16.7],
  "mood": {"valence": 0.3, "energy": 0.8, "tempo_bpm": 132},
  "caption": "A woman speaks directly to camera."
}
```

Relevant optional fields by dataset:

- `ava`: `clip_type="speech"` and any speech / speaker metadata
- `greatest_hits`: `clip_type="sfx"` plus `event_timestamps_s`
- `condensed_movies`: `clip_type="music"` or `clip_type="mixed"` plus `mood`

## CLI
```bash
python cli.py generate --datasets synthetic
python cli.py prepare-datasets --datasets ava,greatest_hits
python cli.py extract-clips --datasets condensed_movies
python cli.py run --datasets synthetic
python cli.py run --datasets ava,greatest_hits --seeds S1,S3,S4 --no-mock
python cli.py compare
```

Useful flags:

- `--datasets synthetic,ava,...`
- `--limit-per-dataset 10`
- `--force-prepare`
- `--mock` / `--no-mock`

## Corruption Examples
You can generate one example corrupted WAV directly from `cmd.exe`:

```cmd
cd /d F:\philo_lab\audio-eval

scripts\generate_corruption_example.cmd sync_shift --input data\source_clips\synthetic\speech_00.wav --output data\corrupted\examples\sync_shift.wav --offset-ms 200
scripts\generate_corruption_example.cmd speaker_swap --input data\source_clips\synthetic\speech_00.wav --output data\corrupted\examples\speaker_swap.wav --swap-point-s 2.5 --replacement-freq 440
scripts\generate_corruption_example.cmd artifact_inject --input data\source_clips\synthetic\speech_00.wav --output data\corrupted\examples\artifact_click.wav --artifact-type click --timestamps 0.5,1.5 --severity 0.7
scripts\generate_corruption_example.cmd sfx_mistime --input data\source_clips\synthetic\sfx_00.wav --output data\corrupted\examples\sfx_mistime.wav --shift-ms 200 --event-timestamps 0.6,1.4
scripts\generate_corruption_example.cmd music_mood_swap --input data\source_clips\synthetic\music_00.wav --output data\corrupted\examples\music_mood_swap.wav --original-valence 0.2 --original-energy 0.8 --original-tempo-bpm 90 --mood-distance 0.8
```

Each command also writes a sibling JSON file containing the corruption metadata.

## Gemini Input Mode
- If a variant has real video context, Gemini receives a base64 `video_url` MP4 payload.
- If no video is available, Gemini falls back to `input_audio` WAV.

## Outputs
- `outputs/tasks_and_rubrics.tsv`: per-variant results, two rows per variant
- `outputs/comparison_report.json`: comparator summary
- `outputs/plots/`: severity and tier plots
- `logs/latest_judge_scores.jsonl`: one JSON record per model score, including full Gemini responses when available

## Notes
- Mock mode is first-class and keeps the repo runnable without network access.
- Synthetic mode remains the default fallback when no dataset clips are prepared.
- Corruption modules still operate on WAV audio; for real-video clips the runner remuxes corrupted audio back onto MP4 before Gemini scoring.
