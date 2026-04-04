"""Structured failure mode definitions for audio quality evaluation."""

from dataclasses import dataclass, field


@dataclass
class FailureMode:
    id: str
    pillar: str  # "voice" | "sfx" | "music"
    name: str
    description: str
    expected_correct: str
    timescale: str  # "microsecond" | "millisecond" | "second" | "multi-second" | "minute"
    detection_tier: int  # 1 (signal) | 2 (embedding) | 3 (semantic)
    examples: list[str] = field(default_factory=list)


FAILURE_MODES: dict[str, FailureMode] = {
    # --- Voice pillar ---
    "lip_sync_drift": FailureMode(
        id="lip_sync_drift",
        pillar="voice",
        name="Lip Sync Drift",
        description="Audio leads or lags the visual speech movements.",
        expected_correct="Audio is synchronized with lip movements within perceptual threshold (~40ms).",
        timescale="millisecond",
        detection_tier=1,
        examples=[
            "Dialogue arrives 300ms before mouth moves",
            "Laugh sound 500ms after smile",
            "Progressive drift that worsens over 30s",
        ],
    ),
    "speaker_identity_break": FailureMode(
        id="speaker_identity_break",
        pillar="voice",
        name="Speaker Identity Break",
        description="Voice timbre/pitch shifts mid-clip indicating a different speaker.",
        expected_correct="Consistent speaker identity throughout contiguous speech segments.",
        timescale="multi-second",
        detection_tier=2,
        examples=[
            "Male voice becomes female after a cut",
            "Pitch shifts 2 semitones mid-sentence",
            "Accent changes between shots",
        ],
    ),
    "tts_artifacts": FailureMode(
        id="tts_artifacts",
        pillar="voice",
        name="TTS Artifacts",
        description="Robotic prosody, glitches, or unnatural audio artifacts from speech synthesis.",
        expected_correct="Natural, clean speech without digital artifacts.",
        timescale="microsecond",
        detection_tier=1,
        examples=[
            "Metallic buzzing on sibilants",
            "Unnatural 500ms pause mid-word",
            "Phoneme repetition stutter",
        ],
    ),
    # --- SFX pillar ---
    "sfx_temporal_misfire": FailureMode(
        id="sfx_temporal_misfire",
        pillar="sfx",
        name="SFX Temporal Misfire",
        description="Sound effect fires at the wrong time relative to the visual action.",
        expected_correct="Sound effects aligned with visual events within ~50ms.",
        timescale="millisecond",
        detection_tier=1,
        examples=[
            "Glass break 1s after visual impact",
            "Footstep sounds between steps not during",
            "Gunshot before muzzle flash",
        ],
    ),
    "sfx_semantic_mismatch": FailureMode(
        id="sfx_semantic_mismatch",
        pillar="sfx",
        name="SFX Semantic Mismatch",
        description="Sound effect is the wrong type for the visual action.",
        expected_correct="Sound effects semantically match the visible source.",
        timescale="second",
        detection_tier=3,
        examples=[
            "Metal clang for wooden door",
            "Cat meow when dog is visible",
            "Indoor reverb for outdoor scene",
        ],
    ),
    "missing_phantom_sfx": FailureMode(
        id="missing_phantom_sfx",
        pillar="sfx",
        name="Missing / Phantom SFX",
        description="Visible action has no sound, or sound occurs with no visual cause.",
        expected_correct="All visible sound-producing actions have corresponding audio and vice versa.",
        timescale="second",
        detection_tier=3,
        examples=[
            "Car crashes silently",
            "Random explosion sound with static scene",
            "Footsteps with no one walking",
        ],
    ),
    # --- Music pillar ---
    "mood_scene_mismatch": FailureMode(
        id="mood_scene_mismatch",
        pillar="music",
        name="Mood-Scene Mismatch",
        description="Background music emotion contradicts the visual scene mood.",
        expected_correct="Music mood reinforces or complements the visual narrative emotion.",
        timescale="minute",
        detection_tier=3,
        examples=[
            "Upbeat pop during funeral scene",
            "Horror stingers during comedy",
            "Lullaby during action chase",
        ],
    ),
    "transition_ignorance": FailureMode(
        id="transition_ignorance",
        pillar="music",
        name="Transition Ignorance",
        description="Music continues unchanged across dramatic visual scene changes.",
        expected_correct="Music adapts to scene transitions with appropriate changes in mood/tempo/energy.",
        timescale="minute",
        detection_tier=3,
        examples=[
            "Same track plays through scene change from indoor to outdoor",
            "Tempo unchanged when action accelerates",
            "Music continues through dialogue that should be unscored",
        ],
    ),
    "abrupt_music_cut": FailureMode(
        id="abrupt_music_cut",
        pillar="music",
        name="Abrupt Music Cut",
        description="Music stops mid-phrase without fade or transition.",
        expected_correct="Music fades, resolves, or transitions naturally at edit points.",
        timescale="second",
        detection_tier=2,
        examples=[
            "Chord cuts off mid-sustain at scene boundary",
            "Melody stops before resolving",
            "Beat drops out without any transition",
        ],
    ),
}
