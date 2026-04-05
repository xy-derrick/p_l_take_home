"""Generate synthetic source audio clips with known ground truth metadata."""

from __future__ import annotations

import json
import os

import numpy as np
import soundfile as sf

from config import SAMPLE_RATE, SFX_LIBRARY_ROOT, SOURCE_CLIPS_ROOT, ensure_runtime_dirs

OUTPUT_DIR = os.path.join(SOURCE_CLIPS_ROOT, "synthetic")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _clip_record(name: str, wav_path: str, duration_s: float, clip_type: str) -> dict:
    meta_path = wav_path.replace(".wav", ".json")
    metadata = json.loads(open(meta_path, encoding="utf-8").read())
    return {
        "dataset": "synthetic",
        "path": wav_path,
        "name": name,
        "type": clip_type,
        "source_audio_path": wav_path,
        "source_video_path": None,
        "metadata_path": meta_path,
        "duration_s": duration_s,
        "segment_start_s": 0.0,
        "segment_end_s": duration_s,
        "visual_context_available": False,
        "metadata": metadata,
    }


def _save_clip(name: str, audio: np.ndarray, metadata: dict) -> str:
    _ensure_dir(OUTPUT_DIR)
    wav_path = os.path.join(OUTPUT_DIR, f"{name}.wav")
    json_path = os.path.join(OUTPUT_DIR, f"{name}.json")
    payload = dict(metadata)
    payload.update(
        {
            "dataset": "synthetic",
            "name": name,
            "path": wav_path,
            "source_audio_path": wav_path,
            "source_video_path": None,
            "metadata_path": json_path,
            "visual_context_available": False,
        }
    )
    sf.write(wav_path, audio, SAMPLE_RATE)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return wav_path


def _amplitude_envelope(t: np.ndarray, syllable_rate: float = 4.0) -> np.ndarray:
    """Simulate speech-like amplitude modulation."""
    return 0.5 * (1.0 + np.sin(2 * np.pi * syllable_rate * t))


def generate_speech_clip(
    name: str,
    duration_s: float,
    base_freq: float,
    syllable_rate: float = 4.0,
) -> str:
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    signal = np.sin(2 * np.pi * base_freq * t)
    signal += 0.5 * np.sin(2 * np.pi * base_freq * 2 * t)
    signal += 0.25 * np.sin(2 * np.pi * base_freq * 3 * t)
    envelope = _amplitude_envelope(t, syllable_rate)
    audio = (signal * envelope * 0.3).astype(np.float32)
    metadata = {
        "type": "speech",
        "clip_type": "speech",
        "duration_s": duration_s,
        "base_freq_hz": base_freq,
        "syllable_rate_hz": syllable_rate,
        "sample_rate": SAMPLE_RATE,
    }
    return _save_clip(name, audio, metadata)


def generate_sfx_clip(name: str, duration_s: float, event_times: list[float]) -> str:
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    audio = np.zeros_like(t, dtype=np.float32)
    for et in event_times:
        idx = int(et * SAMPLE_RATE)
        decay_len = int(0.15 * SAMPLE_RATE)
        end_idx = min(idx + decay_len, len(audio))
        n = end_idx - idx
        if n > 0:
            decay = np.exp(-np.linspace(0, 5, n))
            noise_burst = np.random.default_rng(42).normal(0, 0.4, n)
            audio[idx:end_idx] += (noise_burst * decay).astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    metadata = {
        "type": "sfx",
        "clip_type": "sfx",
        "duration_s": duration_s,
        "event_timestamps_s": event_times,
        "sample_rate": SAMPLE_RATE,
    }
    return _save_clip(name, audio, metadata)


def generate_music_clip(name: str, duration_s: float, key_freq: float, tempo_bpm: int, mood: dict) -> str:
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    beat_period = 60.0 / tempo_bpm
    third_ratio = 5 / 4 if mood.get("valence", 0.5) > 0.5 else 6 / 5
    signal = np.sin(2 * np.pi * key_freq * t)
    signal += 0.7 * np.sin(2 * np.pi * key_freq * third_ratio * t)
    signal += 0.6 * np.sin(2 * np.pi * key_freq * 1.5 * t)
    beat_env = 0.5 * (1.0 + np.sin(2 * np.pi * (1.0 / beat_period) * t))
    audio = (signal * beat_env * 0.2).astype(np.float32)
    metadata = {
        "type": "music",
        "clip_type": "music",
        "duration_s": duration_s,
        "key_freq_hz": key_freq,
        "tempo_bpm": tempo_bpm,
        "mood": mood,
        "sample_rate": SAMPLE_RATE,
    }
    return _save_clip(name, audio, metadata)


def generate_mixed_clip(
    name: str,
    duration_s: float,
    speech_freq: float,
    music_freq: float,
    tempo_bpm: int,
    mood: dict | None = None,
) -> str:
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    speech = np.sin(2 * np.pi * speech_freq * t)
    speech += 0.5 * np.sin(2 * np.pi * speech_freq * 2 * t)
    speech *= _amplitude_envelope(t, 3.5)
    music = np.sin(2 * np.pi * music_freq * t)
    music += 0.6 * np.sin(2 * np.pi * music_freq * 1.5 * t)
    beat_env = 0.5 * (1.0 + np.sin(2 * np.pi * (tempo_bpm / 60.0) * t))
    music *= beat_env
    audio = (speech * 0.25 + music * 0.1).astype(np.float32)

    if mood is None:
        mood = {
            "valence": 0.7 if music_freq >= 220 else 0.4,
            "energy": 0.65 if tempo_bpm >= 90 else 0.35,
            "tempo_bpm": tempo_bpm,
        }

    metadata = {
        "type": "mixed",
        "clip_type": "mixed",
        "duration_s": duration_s,
        "speech_freq_hz": speech_freq,
        "music_freq_hz": music_freq,
        "tempo_bpm": tempo_bpm,
        "mood": mood,
        "sample_rate": SAMPLE_RATE,
    }
    return _save_clip(name, audio, metadata)


def generate_all() -> list[dict]:
    """Generate all source clips. Returns normalized clip records."""
    ensure_runtime_dirs()
    clips = []

    speech_specs = [(150, 15), (200, 20), (250, 12), (180, 25), (220, 18)]
    for i, (freq, dur) in enumerate(speech_specs):
        name = f"speech_{i:02d}"
        path = generate_speech_clip(name, dur, freq)
        clips.append(_clip_record(name, path, dur, "speech"))

    sfx_events = [
        [1.0, 3.5, 6.0, 9.0],
        [2.0, 5.0, 8.0],
        [0.5, 2.5, 4.5, 7.0, 10.0],
        [1.5, 4.0, 7.5],
    ]
    for i, events in enumerate(sfx_events):
        name = f"sfx_{i:02d}"
        duration = max(events) + 3.0
        path = generate_sfx_clip(name, duration, events)
        clips.append(_clip_record(name, path, duration, "sfx"))

    music_specs = [
        (262, 120, {"valence": 0.8, "energy": 0.7, "tempo_bpm": 120}),
        (220, 72, {"valence": 0.2, "energy": 0.3, "tempo_bpm": 72}),
        (233, 140, {"valence": 0.3, "energy": 0.9, "tempo_bpm": 140}),
        (196, 60, {"valence": 0.7, "energy": 0.2, "tempo_bpm": 60}),
    ]
    for i, (freq, tempo, mood) in enumerate(music_specs):
        name = f"music_{i:02d}"
        path = generate_music_clip(name, 20, freq, tempo, mood)
        clips.append(_clip_record(name, path, 20.0, "music"))

    mixed_specs = [
        (180, 262, 100, {"valence": 0.7, "energy": 0.6, "tempo_bpm": 100}),
        (220, 196, 80, {"valence": 0.4, "energy": 0.3, "tempo_bpm": 80}),
    ]
    for i, (speech_freq, music_freq, tempo, mood) in enumerate(mixed_specs):
        name = f"mixed_{i:02d}"
        path = generate_mixed_clip(name, 20, speech_freq, music_freq, tempo, mood=mood)
        clips.append(_clip_record(name, path, 20.0, "mixed"))

    sfx_lib_dir = str(SFX_LIBRARY_ROOT)
    _ensure_dir(sfx_lib_dir)
    replacement_moods = [
        ("happy_replacement", 330, 130, {"valence": 0.9, "energy": 0.8, "tempo_bpm": 130}),
        ("sad_replacement", 196, 60, {"valence": 0.1, "energy": 0.2, "tempo_bpm": 60}),
        ("tense_replacement", 277, 150, {"valence": 0.2, "energy": 0.95, "tempo_bpm": 150}),
        ("calm_replacement", 220, 55, {"valence": 0.8, "energy": 0.15, "tempo_bpm": 55}),
    ]
    for name, freq, tempo, mood in replacement_moods:
        t = np.linspace(0, 20, int(SAMPLE_RATE * 20), endpoint=False)
        third_ratio = 5 / 4 if mood["valence"] > 0.5 else 6 / 5
        signal = np.sin(2 * np.pi * freq * t)
        signal += 0.7 * np.sin(2 * np.pi * freq * third_ratio * t)
        signal += 0.6 * np.sin(2 * np.pi * freq * 1.5 * t)
        beat_env = 0.5 * (1.0 + np.sin(2 * np.pi * (tempo / 60.0) * t))
        audio = (signal * beat_env * 0.2).astype(np.float32)
        wav_path = os.path.join(sfx_lib_dir, f"{name}.wav")
        json_path = os.path.join(sfx_lib_dir, f"{name}.json")
        sf.write(wav_path, audio, SAMPLE_RATE)
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "type": "music",
                    "mood": mood,
                    "tempo_bpm": tempo,
                    "key_freq_hz": freq,
                    "duration_s": 20,
                },
                handle,
                indent=2,
            )

    print(f"Generated {len(clips)} synthetic source clips in {OUTPUT_DIR}")
    return clips


if __name__ == "__main__":
    generate_all()
