"""SFX onset temporal shift — move sound events by N milliseconds."""

import numpy as np
import soundfile as sf


def apply_sfx_mistime(audio_path: str, output_path: str, *,
                      shift_ms: int = 200,
                      event_timestamps: list[float] | None = None,
                      **kwargs) -> dict:
    """Shift audio segments around each event by shift_ms.

    event_timestamps: known onset times in seconds.
    If None, attempts to read from source clip metadata.
    """
    audio, sr = sf.read(audio_path)

    if event_timestamps is None:
        # Try loading from companion JSON
        import json, os
        json_path = audio_path.replace(".wav", ".json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                meta = json.load(f)
            event_timestamps = meta.get("event_timestamps_s", [])
        else:
            event_timestamps = []

    if not event_timestamps:
        # No events to shift — just copy
        sf.write(output_path, audio, sr)
        return {
            "corruption_type": "sfx_mistime",
            "shift_ms": shift_ms,
            "events_shifted": 0,
            "original_path": audio_path,
            "corrupted_path": output_path,
        }

    result = audio.copy()
    shift_samples = int(shift_ms * sr / 1000)
    window_ms = 200  # ±200ms window around each event
    window_samples = int(window_ms * sr / 1000)
    shifted_events = []

    for et in event_timestamps:
        center = int(et * sr)
        src_start = max(0, center - window_samples)
        src_end = min(len(audio), center + window_samples)
        segment = audio[src_start:src_end].copy()

        # Zero out original location
        result[src_start:src_end] = 0.0

        # Place at shifted location
        dst_start = max(0, src_start + shift_samples)
        dst_end = min(len(result), dst_start + len(segment))
        n = dst_end - dst_start
        if n > 0:
            # Crossfade boundaries
            xfade = min(int(0.01 * sr), n)
            fade_in = np.linspace(0, 1, xfade).astype(np.float32)
            fade_out = np.linspace(1, 0, xfade).astype(np.float32)
            seg = segment[:n].copy()
            seg[:xfade] *= fade_in
            seg[-xfade:] *= fade_out
            result[dst_start:dst_end] += seg

        shifted_events.append({
            "original_time_s": float(et),
            "shifted_time_s": float(et + shift_ms / 1000.0),
            "shift_ms": shift_ms,
        })

    result = np.clip(result, -1.0, 1.0).astype(np.float32)
    sf.write(output_path, result, sr)

    return {
        "corruption_type": "sfx_mistime",
        "shift_ms": shift_ms,
        "events_shifted": len(shifted_events),
        "shifted_events": shifted_events,
        "original_path": audio_path,
        "corrupted_path": output_path,
    }
