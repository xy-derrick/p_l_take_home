"""Replace music with mismatched-mood track."""

import math
import numpy as np
import soundfile as sf


def apply_music_mood_swap(audio_path: str, output_path: str,
                          replacement_music_path: str | None = None,
                          original_mood: dict | None = None,
                          replacement_mood: dict | None = None,
                          mood_distance: float = 0.8,
                          **kwargs) -> dict:
    """Replace the music content with a different mood track.

    If replacement_music_path is provided, use it. Otherwise generate
    a synthetic replacement based on mood_distance from the original.
    """
    audio, sr = sf.read(audio_path)

    if original_mood is None:
        original_mood = {"valence": 0.5, "energy": 0.5, "tempo_bpm": 100}

    if replacement_music_path:
        replacement, _ = sf.read(replacement_music_path)
        if len(replacement) < len(audio):
            replacement = np.tile(replacement, (len(audio) // len(replacement)) + 1)
        replacement = replacement[:len(audio)]
    else:
        # Generate synthetic replacement with shifted mood
        if replacement_mood is None:
            # Invert mood based on distance
            replacement_mood = {
                "valence": max(0, min(1, 1.0 - original_mood.get("valence", 0.5))),
                "energy": max(0, min(1, 1.0 - original_mood.get("energy", 0.5))),
                "tempo_bpm": int(original_mood.get("tempo_bpm", 100) * (1.5 if mood_distance > 0.5 else 0.7)),
            }
        t = np.linspace(0, len(audio) / sr, len(audio), endpoint=False)
        freq = 220 * (1 + replacement_mood["valence"])
        third_ratio = 5 / 4 if replacement_mood["valence"] > 0.5 else 6 / 5
        sig = np.sin(2 * np.pi * freq * t)
        sig += 0.7 * np.sin(2 * np.pi * freq * third_ratio * t)
        sig += 0.6 * np.sin(2 * np.pi * freq * 1.5 * t)
        tempo = replacement_mood["tempo_bpm"]
        beat_env = 0.5 * (1.0 + np.sin(2 * np.pi * (tempo / 60.0) * t))
        replacement = (sig * beat_env * 0.2).astype(np.float32)

    if replacement_mood is None:
        replacement_mood = {"valence": 1.0 - original_mood.get("valence", 0.5),
                            "energy": 1.0 - original_mood.get("energy", 0.5),
                            "tempo_bpm": 120}

    # Compute actual mood distance
    ov = original_mood.get("valence", 0.5)
    oe = original_mood.get("energy", 0.5)
    rv = replacement_mood.get("valence", 0.5)
    re = replacement_mood.get("energy", 0.5)
    actual_distance = math.sqrt((ov - rv) ** 2 + (oe - re) ** 2) / math.sqrt(2)

    sf.write(output_path, replacement[:len(audio)], sr)

    return {
        "corruption_type": "music_mood_swap",
        "original_mood": original_mood,
        "replacement_mood": replacement_mood,
        "mood_distance": float(actual_distance),
        "requested_distance": float(mood_distance),
        "original_path": audio_path,
        "corrupted_path": output_path,
    }
