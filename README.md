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
- **9 failure modes** across 3 pillars (voice, SFX, music) and 3 detection tiers

## Pipeline Stages
1. **Generate** — synthetic source clips with mathematically exact ground truth
2. **Expand** — 5 seeds × 15 clips × 5-10 severity levels → 500+ variants
3. **Corrupt** — apply controlled audio failures (sync shift, speaker swap, artifacts, SFX mistime, mood swap)
4. **Score** — evaluate with both signal pipeline and Gemini (mock or real API)
5. **Compare** — divergence analysis, tier-level performance, severity curves

## CLI Commands
```bash
python cli.py run                # Full pipeline (default: mock mode)
python cli.py run --seeds S1,S2  # Run specific seeds only
python cli.py run --no-mock      # Use real Gemini API (requires OPENROUTER_API_KEY)
python cli.py generate           # Generate source clips only
python cli.py compare            # View comparison report
```

## Outputs
- `outputs/tasks_and_rubrics.tsv` — full results matrix (2 rows per variant, one per model)
- `outputs/comparison_report.json` — model comparison analysis
- `outputs/plots/` — severity-accuracy curves per seed, tier comparison chart
- `report/report.pdf` — research report (compile with `make -C report/`)

## Key Design Decisions
- **Mock mode is first-class**: entire pipeline runs without API keys or network
- **Ground truth is exact**: every corruption has programmatically verifiable truth
- **The comparison is the insight**: the value is WHERE the two models diverge
- **Tiered rubric**: 5 dimensions × 5-point scale = structured reward signal

## Environment
```bash
# Optional: set for real Gemini API calls
export OPENROUTER_API_KEY="your-key-here"
```

## Project Structure
```
audio-eval/
├── cli.py                      # Main entry point
├── config.py                   # Configuration and thresholds
├── data/
│   ├── generate_source_clips.py  # Synthetic audio generation
│   ├── source_clips/             # Generated clean clips
│   ├── corrupted/                # Corrupted variants
│   └── sfx_library/             # Replacement tracks
├── taxonomy/
│   └── failure_modes.py        # 9 failure mode definitions
├── seeds/
│   └── seed_tasks.py           # 5 seed task specifications
├── corruption/
│   ├── sync_shift.py           # A/V sync drift
│   ├── speaker_swap.py         # Speaker identity change
│   ├── artifact_inject.py      # Click/dropout/spectral artifacts
│   ├── sfx_mistime.py          # SFX temporal shift
│   └── music_mood_swap.py      # Mood-mismatched music
├── expansion/
│   └── variant_generator.py    # Seed → 500+ variant expansion
├── scoring/
│   ├── rubric.py               # 5-dimension rubric definitions
│   ├── signal_scorer.py        # Signal processing pipeline
│   ├── gemini_scorer.py        # Gemini VLM scorer (+ mock)
│   └── aggregator.py           # Combine and write TSV
├── evaluation/
│   ├── runner.py               # Pipeline orchestration
│   └── comparator.py           # Model comparison analysis
├── outputs/                    # Generated results
└── report/
    ├── report.tex              # LaTeX research report
    └── Makefile
```

## Agent Prompts Used
This project was built using Claude Code with a single comprehensive prompt
covering architecture, implementation details, mock behavior specifications,
and report structure. The prompt defined the corruption-based evaluation
approach, two-model comparison strategy, and tiered failure taxonomy.

## API Costs
- Mock mode: $0 (no API calls)
- Full Gemini mode: estimated $5-15 for 500+ variants at ~$0.01-0.03 per call
