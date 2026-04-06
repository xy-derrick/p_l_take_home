# Audio Quality Verification Benchmark

This repo benchmarks audio failures in AI-generated video by:

1. preparing source clips
2. injecting controlled corruptions
3. scoring each clip with a signal-based judge and a multimodal judge
4. exporting TSV, JSON, plots, and per-judge logs

The main entry point is [cli.py](./cli.py).

## What the Repo Supports

- `synthetic` source mode: generates local WAV clips with exact ground truth
- `real-video` source mode: prepares MP4 + WAV pairs from dataset manifests
- `SignalScorer`: real signal processing for `S1` to `S4`
- `LanguageModelScorer`: OpenRouter multimodal scorer using the configured `MODEL`
- per-run report export and replay from saved judge logs

## Seeds

- `S1`: sync drift
- `S2`: speaker identity change
- `S3`: SFX mistiming
- `S4`: artifact injection
- `S5`: music mood mismatch

In practice, the strongest current benchmark path is `S1` to `S4`.

## Requirements

- Python 3.9+
- `ffmpeg` and `ffprobe` available on `PATH`
- optional: `OPENROUTER_API_KEY` for real model calls

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

The runtime reads `.env` from the repo root if present.

Example:

```env
OPENROUTER_API_KEY=your_key_here
MODEL=google/gemini-2.5-flash
MODEL_LABEL=Gemini 2.5 Flash
```

Notes:

- `MODEL` is the OpenRouter model name used in requests
- `MODEL_LABEL` is optional and controls plot/report labels
- the code still accepts legacy `GEMINI_MODEL`, but `MODEL` is preferred

## Quick Start

### 1. Synthetic smoke test

Runs end to end with no external data and no API calls:

```cmd
cd /d F:\philo_lab\audio-eval
python cli.py run --datasets synthetic --mock
```

### 2. Real-video smoke test

If you already have prepared clips:

```cmd
cd /d F:\philo_lab\audio-eval
python cli.py run --datasets ava,greatest_hits --seeds S1,S2,S3,S4 --mock
```

### 3. Real model run

```cmd
cd /d F:\philo_lab\audio-eval
python cli.py run --datasets ava,greatest_hits --seeds S1,S2,S3,S4 --no-mock
```

If you already generated corrupted files and want to rescore without regenerating them:

```cmd
python cli.py run --datasets ava,greatest_hits --seeds S1,S2,S3,S4 --no-mock --reuse-corrupted
```

## Real-Video Workflow

The real-video path has three stages:

1. download raw dataset media
2. build manifests
3. prepare clips and run the benchmark

### Supported datasets

- `ava`: speech / visible-speaker clips
- `greatest_hits`: visually grounded impact / SFX clips
- `condensed_movies`: music / mixed clips via manifest-driven preparation

### Download helpers

These helper scripts are written for `cmd.exe`:

```cmd
cd /d F:\philo_lab\audio-eval

scripts\download_ava_subset.cmd --count 10 --skip-existing
scripts\build_ava_manifest.cmd --max-clips-per-video 3

scripts\download_greatest_hits.cmd --variant lowres --extract --skip-existing
scripts\build_greatest_hits_manifest.cmd --max-clips-per-video 3
```

They populate:

- `data\raw_videos\ava`
- `data\raw_videos\greatest_hits`
- `data\manifests\ava_clips.jsonl`
- `data\manifests\greatest_hits_clips.jsonl`

### Prepare clips

This extracts short MP4 clips from raw videos, writes companion WAV files, and stores metadata in `data\source_clips\...`.

```cmd
cd /d F:\philo_lab\audio-eval
python cli.py prepare-datasets --datasets ava,greatest_hits
```

Useful flags:

- `--limit-per-dataset 10`: only prepare a subset
- `--force-prepare`: rebuild prepared clips from the manifest

Note:

- `extract-clips` is currently just an alias for `prepare-datasets`

## Common Run Commands

### Run one seed

```cmd
python cli.py run --datasets ava --seeds S1 --no-mock
```

### Run `S1` to `S4`

```cmd
python cli.py run --datasets ava,greatest_hits --seeds S1,S2,S3,S4 --no-mock
```

### Run with a smaller sample

```cmd
python cli.py run --datasets ava,greatest_hits --seeds S1,S2,S3,S4 --mock --limit-per-dataset 5
```

### Reuse existing corruptions

```cmd
python cli.py run --datasets ava,greatest_hits --seeds S1,S2,S3,S4 --no-mock --reuse-corrupted
```

## CLI Summary

```text
generate
prepare-datasets
extract-clips
run
score
compare
replay-from-log
```

Examples:

```cmd
python cli.py generate --datasets synthetic
python cli.py prepare-datasets --datasets ava,greatest_hits
python cli.py run --datasets synthetic --mock
python cli.py compare
python cli.py replay-from-log --input report\seed1_latest_judge_scores.jsonl
```

## Outputs

Normal `run` output:

- `outputs\tasks_and_rubrics.tsv`
- `outputs\comparison_report.json`
- `outputs\plots\severity_curve_*.png`
- `outputs\plots\tier_comparison.png`
- `logs\latest_judge_scores.jsonl`

After each run, the CLI also copies report artifacts into `report\` using the selected seed prefix, for example:

- `report\seed1_latest_judge_scores.jsonl`
- `report\seed1_comparison_report.json`
- `report\seed1_tier_comparison.png`

## Replay From a Saved Judge Log

You can rebuild TSV, comparison JSON, and plots directly from a saved judge log:

```cmd
cd /d F:\philo_lab\audio-eval
python cli.py replay-from-log --input report\seed1_latest_judge_scores.jsonl
```

That writes files next to the input log using the same prefix, for example:

- `report\seed1_tasks_and_rubrics.tsv`
- `report\seed1_comparison_report.json`
- `report\seed1_tier_comparison.png`
- `report\seed1_severity_curve_S1.png`

You can override the destination directory:

```cmd
python cli.py replay-from-log --input report\seed1_latest_judge_scores.jsonl --output-dir outputs
```

## Corruption Example Commands

These generate one corrupted WAV plus a sibling JSON metadata file:

```cmd
cd /d F:\philo_lab\audio-eval

scripts\generate_corruption_example.cmd sync_shift --input data\source_clips\synthetic\speech_00.wav --output data\corrupted\examples\sync_shift.wav --offset-ms 200
scripts\generate_corruption_example.cmd speaker_swap --input data\source_clips\synthetic\speech_00.wav --output data\corrupted\examples\speaker_swap.wav --swap-point-s 2.5 --replacement-freq 440
scripts\generate_corruption_example.cmd artifact_inject --input data\source_clips\synthetic\speech_00.wav --output data\corrupted\examples\artifact_click.wav --artifact-type click --timestamps 0.5,1.5 --severity 0.7
scripts\generate_corruption_example.cmd sfx_mistime --input data\source_clips\synthetic\sfx_00.wav --output data\corrupted\examples\sfx_mistime.wav --shift-ms 200 --event-timestamps 0.6,1.4
scripts\generate_corruption_example.cmd music_mood_swap --input data\source_clips\synthetic\music_00.wav --output data\corrupted\examples\music_mood_swap.wav --original-valence 0.2 --original-energy 0.8 --original-tempo-bpm 90 --mood-distance 0.8
```

## Manifest Format

Each manifest record can be JSONL, JSON, or CSV. Minimum useful shape:

```json
{
  "clip_id": "ava_example_0001",
  "video_path": "data/raw_videos/ava/example.mp4",
  "start_s": 12.0,
  "end_s": 20.0,
  "clip_type": "speech",
  "caption": "A speaker talks to camera."
}
```

Optional fields used by some seeds:

- `event_timestamps_s`: needed for `S3`
- `mood`: used for `S5`
- dataset-specific metadata such as active-speaker timestamps

## Tests

Run the corruption unit tests:

```cmd
cd /d F:\philo_lab\audio-eval
python -m unittest discover -s tests -v
```

## Agent Prompts Used

The full prompt given to Claude Code (the coding agent used to build this repo) is in
[claude_code_prompt.md](./claude_code_prompt.md).

It covers the intended architecture, all module specifications, corruption function
signatures, scorer mock-behavior profiles, and the report structure. The implementation
tracks the spec closely; deviations (real signal processing replacing mocked signal
scorers, SHA-256-based seeding replacing `random.seed(hash(...))`, keyword-only
corruption function args, etc.) are documented in git history.

## How This Was Built: Dual-Agent Workflow

This project was completed entirely through AI coding agents — no code was written manually.
Two agents played complementary roles throughout:

**Claude Code** (Anthropic) handled the full implementation lifecycle. Given
[claude_code_prompt.md](./claude_code_prompt.md) as the initial spec, it generated the
entire repository structure, all Python modules, the CLI, the LaTeX report, and the
test suite. It also iterated on the implementation in response to review findings —
replacing mocked signal scorers with real librosa-based signal processing, fixing
seeding to use SHA-256 instead of `hash()`, upgrading the JSON parser in the VLM scorer
to handle markdown code fences and float rubric values, and rewriting the report to
incorporate actual experiment results with correct numbers.

**Codex** (OpenAI) ran the benchmark experiments. It executed the pipeline end-to-end
with `--no-mock` against real AVA and Greatest Hits dataset clips, making actual
OpenRouter API calls to Gemini 2.5 Flash, and saved the per-seed JSONL judge logs,
comparison JSONs, and TSVs now committed under `report/`. It also performed a
post-experiment audit of the codebase and report, identifying three concrete issues
(duplicate severity level in S3, wrong VLM modality description in the report, stale
divergence count) that Claude Code then patched.

**The human was the communication bridge between agents.** There was no direct
agent-to-agent channel. The coordination loop was:

1. Claude Code produces code → committed to git
2. Codex reads the git repo, runs the pipeline, saves result files → committed to git
3. Codex reads result files + source code → produces a structured findings report
4. **The user copies Codex's findings and pastes them into the Claude Code session**
5. Claude Code reviews the findings and patches code and report
6. Changes committed → Codex can read them on the next pass

The critical step is 4: the user copy-pasted Codex's audit output directly
into Claude Code's context and asked it to review and act on the findings. This is the
simplest possible inter-agent communication pattern — no orchestration framework, no
shared memory system, just a human relaying outputs between two independent chat
sessions. The spec file (`claude_code_prompt.md`) served as the shared ground truth
both agents referenced independently, and the git repository was the shared workspace.

## Ad-hoc Scoring: `data/little_test/`

The `data/little_test/` folder contains a single real-world clip used to validate the
model scorer outside the benchmark pipeline:

**Source:** 华强买西瓜 ("Huaqiang Buys a Watermelon"), a popular skit from Bilibili.  
**File:** `37269210935-1-192.mp4`  
**Script:** `score_mp4.py` — sends the original MP4 directly to Gemini 2.5 Flash and
evaluates it against each seed task's rubric prompt. No corruption is applied; the
original video is scored as-is.

**Results (`rubric_eval.tsv`):**

| Seed | Task | av_sync | artifact | speaker | semantic | music | detected |
|------|------|---------|----------|---------|----------|-------|----------|
| S1 | Sync Drift | 5 | 5 | 5 | 5 | 5 | False |
| S2 | Speaker Swap | 5 | 5 | **1** | 5 | 5 | **True** |
| S3 | SFX Mistime | **3** | 5 | 5 | 5 | 5 | **True** |
| S4 | Artifact Inject | 5 | 5 | 5 | 5 | 5 | False |
| S5 | Music Mood | 5 | 5 | 5 | 5 | 4 | True |

Notable findings: Gemini flagged `speaker_consistency=1` under S2 (the skit has multiple
characters with distinct voices, triggering the speaker-change detector) and `av_sync=3`
under S3 (SFX timing in the original is judged as only moderately aligned). S5 fell back
to mock due to a transient 502 from OpenRouter.

## Notes

- synthetic mode is the default fallback if no real clips are available
- corruption functions operate on WAV audio
- for real-video clips, the runner remuxes corrupted audio back onto MP4 before multimodal scoring
- `SignalScorer` is implemented as a real signal baseline for `S1` to `S4`
