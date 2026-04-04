"""Audio artifact injection — clicks, dropouts, spectral anomalies, stutters."""

import numpy as np
import soundfile as sf


def inject_artifacts(audio_path: str, output_path: str,
                     artifact_type: str = "click",
                     timestamps: list[float] | None = None,
                     severity: float = 0.5,
                     **kwargs) -> dict:
    """Inject audio artifacts at specified timestamps.

    artifact_type: "click" | "dropout" | "spectral" | "stutter"
    severity: 0.0 to 1.0, controls intensity.
    """
    audio, sr = sf.read(audio_path)
    duration_s = len(audio) / sr

    if timestamps is None:
        # Auto-generate 3-5 timestamps spread across the clip
        rng = np.random.default_rng(int(severity * 1000))
        n_events = rng.integers(3, 6)
        timestamps = sorted(rng.uniform(0.5, duration_s - 0.5, n_events).tolist())

    result = audio.copy()
    injected = []

    for ts in timestamps:
        idx = int(ts * sr)
        if idx >= len(result):
            continue

        if artifact_type == "click":
            # Single-sample spike
            spike_len = max(1, int(0.001 * sr))
            end = min(idx + spike_len, len(result))
            result[idx:end] *= (1.0 + severity * 10)
            result[idx:end] = np.clip(result[idx:end], -1.0, 1.0)

        elif artifact_type == "dropout":
            # Zero out audio
            dropout_ms = int(50 + severity * 200)
            dropout_samples = int(dropout_ms * sr / 1000)
            end = min(idx + dropout_samples, len(result))
            result[idx:end] = 0.0

        elif artifact_type == "spectral":
            # Narrow-band noise burst
            burst_len = int(0.05 * sr)
            end = min(idx + burst_len, len(result))
            n = end - idx
            t = np.linspace(0, n / sr, n, endpoint=False)
            noise = severity * 0.5 * np.sin(2 * np.pi * 4000 * t)
            result[idx:end] += noise.astype(np.float32)

        elif artifact_type == "stutter":
            # Repeat 50ms segment
            seg_len = int(0.05 * sr)
            end = min(idx + seg_len, len(result))
            segment = result[idx:end].copy()
            repeat_start = end
            repeat_end = min(repeat_start + len(segment), len(result))
            n = repeat_end - repeat_start
            if n > 0:
                result[repeat_start:repeat_end] = segment[:n]

        injected.append({
            "type": artifact_type,
            "timestamp_s": float(ts),
            "severity": float(severity),
        })

    result = np.clip(result, -1.0, 1.0).astype(np.float32)
    sf.write(output_path, result, sr)

    return {
        "corruption_type": "artifact_inject",
        "artifact_type": artifact_type,
        "artifacts": injected,
        "severity": float(severity),
        "original_path": audio_path,
        "corrupted_path": output_path,
    }
