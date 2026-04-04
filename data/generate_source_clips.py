"""Generate synthetic source audio clips with known ground truth metadata."""

import json
import os
import numpy as np
import soundfile as sf

SAMPLE_RATE = 22050
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "source_clips")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_clip(name: str, audio: np.ndarray, metadata: dict) -> str:
    _ensure_dir(OUTPUT_DIR)
    wav_path = os.path.join(OUTPUT_DIR, f"{name}.wav")
    json_path = os.path.join(OUTPUT_DIR, f"{name}.json")
    sf.write(wav_path, audio, SAMPLE_RATE)
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return wav_path


def _amplitude_envelope(t: np.ndarray, syllable_rate: float = 4.0) -> np.ndarray:
    """Simulate speech-like amplitude modulation (syllable envelope)."""
    return 0.5 * (1.0 + np.sin(2 * np.pi * syllable_rate * t))


def generate_speech_clip(name: str, duration_s: float, base_freq: float,
                         syllable_rate: float = 4.0) -> str:
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    # Fundamental + harmonics to simulate voice
    signal = np.sin(2 * np.pi * base_freq * t)
    signal += 0.5 * np.sin(2 * np.pi * base_freq * 2 * t)
    signal += 0.25 * np.sin(2 * np.pi * base_freq * 3 * t)
    envelope = _amplitude_envelope(t, syllable_rate)
    audio = (signal * envelope * 0.3).astype(np.float32)
    metadata = {
        "type": "speech",
        "duration_s": duration_s,
        "base_freq_hz": base_freq,
        "syllable_rate_hz": syllable_rate,
        "sample_rate": SAMPLE_RATE,
    }
    return _save_clip(name, audio, metadata)


def generate_sfx_clip(name: str, duration_s: float,
                      event_times: list[float]) -> str:
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
        "duration_s": duration_s,
        "event_timestamps_s": event_times,
        "sample_rate": SAMPLE_RATE,
    }
    return _save_clip(name, audio, metadata)


def generate_music_clip(name: str, duration_s: float, key_freq: float,
                        tempo_bpm: int, mood: dict) -> str:
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    beat_period = 60.0 / tempo_bpm
    # Build chord: root + third + fifth
    third_ratio = 5 / 4 if mood.get("valence", 0.5) > 0.5 else 6 / 5  # major vs minor
    signal = np.sin(2 * np.pi * key_freq * t)
    signal += 0.7 * np.sin(2 * np.pi * key_freq * third_ratio * t)
    signal += 0.6 * np.sin(2 * np.pi * key_freq * 1.5 * t)  # fifth
    # Rhythmic envelope
    beat_env = 0.5 * (1.0 + np.sin(2 * np.pi * (1.0 / beat_period) * t))
    audio = (signal * beat_env * 0.2).astype(np.float32)
    metadata = {
        "type": "music",
        "duration_s": duration_s,
        "key_freq_hz": key_freq,
        "tempo_bpm": tempo_bpm,
        "mood": mood,
        "sample_rate": SAMPLE_RATE,
    }
    return _save_clip(name, audio, metadata)


def generate_mixed_clip(name: str, duration_s: float, speech_freq: float,
                        music_freq: float, tempo_bpm: int) -> str:
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    # Speech component
    speech = np.sin(2 * np.pi * speech_freq * t)
    speech += 0.5 * np.sin(2 * np.pi * speech_freq * 2 * t)
    speech *= _amplitude_envelope(t, 3.5)
    # Music component (quieter)
    music = np.sin(2 * np.pi * music_freq * t)
    music += 0.6 * np.sin(2 * np.pi * music_freq * 1.5 * t)
    beat_env = 0.5 * (1.0 + np.sin(2 * np.pi * (tempo_bpm / 60.0) * t))
    music *= beat_env
    audio = (speech * 0.25 + music * 0.1).astype(np.float32)
    metadata = {
        "type": "mixed",
        "duration_s": duration_s,
        "speech_freq_hz": speech_freq,
        "music_freq_hz": music_freq,
        "tempo_bpm": tempo_bpm,
        "sample_rate": SAMPLE_RATE,
    }
    return _save_clip(name, audio, metadata)


def generate_all() -> list[dict]:
    """Generate all source clips. Returns list of metadata dicts with paths."""
    clips = []

    # 5 speech clips
    for i, (freq, dur) in enumerate([
        (150, 15), (200, 20), (250, 12), (180, 25), (220, 18)
    ]):
        path = generate_speech_clip(f"speech_{i:02d}", dur, freq)
        clips.append({"path": path, "name": f"speech_{i:02d}", "type": "speech"})

    # 4 SFX clips
    sfx_events = [
        [1.0, 3.5, 6.0, 9.0],
        [2.0, 5.0, 8.0],
        [0.5, 2.5, 4.5, 7.0, 10.0],
        [1.5, 4.0, 7.5],
    ]
    for i, events in enumerate(sfx_events):
        dur = max(events) + 3.0
        path = generate_sfx_clip(f"sfx_{i:02d}", dur, events)
        clips.append({"path": path, "name": f"sfx_{i:02d}", "type": "sfx"})

    # 4 music clips
    music_specs = [
        (262, 120, {"valence": 0.8, "energy": 0.7}),   # happy
        (220, 72,  {"valence": 0.2, "energy": 0.3}),    # sad
        (233, 140, {"valence": 0.3, "energy": 0.9}),    # tense
        (196, 60,  {"valence": 0.7, "energy": 0.2}),    # calm
    ]
    for i, (freq, tempo, mood) in enumerate(music_specs):
        path = generate_music_clip(f"music_{i:02d}", 20, freq, tempo, mood)
        clips.append({"path": path, "name": f"music_{i:02d}", "type": "music"})

    # 2 mixed clips
    for i, (sf_, mf, tempo) in enumerate([(180, 262, 100), (220, 196, 80)]):
        path = generate_mixed_clip(f"mixed_{i:02d}", 20, sf_, mf, tempo)
        clips.append({"path": path, "name": f"mixed_{i:02d}", "type": "mixed"})

    # Generate SFX library for mood swap replacements
    sfx_lib_dir = os.path.join(os.path.dirname(__file__), "sfx_library")
    _ensure_dir(sfx_lib_dir)
    replacement_moods = [
        ("happy_replacement", 330, 130, {"valence": 0.9, "energy": 0.8}),
        ("sad_replacement", 196, 60, {"valence": 0.1, "energy": 0.2}),
        ("tense_replacement", 277, 150, {"valence": 0.2, "energy": 0.95}),
        ("calm_replacement", 220, 55, {"valence": 0.8, "energy": 0.15}),
    ]
    for name, freq, tempo, mood in replacement_moods:
        t = np.linspace(0, 20, int(SAMPLE_RATE * 20), endpoint=False)
        third_ratio = 5 / 4 if mood["valence"] > 0.5 else 6 / 5
        sig = np.sin(2 * np.pi * freq * t)
        sig += 0.7 * np.sin(2 * np.pi * freq * third_ratio * t)
        sig += 0.6 * np.sin(2 * np.pi * freq * 1.5 * t)
        beat_env = 0.5 * (1.0 + np.sin(2 * np.pi * (tempo / 60.0) * t))
        audio = (sig * beat_env * 0.2).astype(np.float32)
        wav_path = os.path.join(sfx_lib_dir, f"{name}.wav")
        json_path = os.path.join(sfx_lib_dir, f"{name}.json")
        sf.write(wav_path, audio, SAMPLE_RATE)
        with open(json_path, "w") as f:
            json.dump({"type": "music", "mood": mood, "tempo_bpm": tempo,
                       "key_freq_hz": freq, "duration_s": 20}, f, indent=2)

    print(f"Generated {len(clips)} source clips in {OUTPUT_DIR}")
    return clips


if __name__ == "__main__":
    generate_all()
