"""Speaker identity replacement at cut points."""

import numpy as np
import soundfile as sf


def apply_speaker_swap(audio_path: str, output_path: str,
                       swap_point_s: float = 5.0,
                       replacement_audio_path: str | None = None,
                       replacement_freq: float = 300.0,
                       **kwargs) -> dict:
    """Replace audio after swap_point_s with a different speaker.

    If replacement_audio_path is given, splice it in.
    Otherwise, generate a synthetic replacement at replacement_freq.
    """
    audio, sr = sf.read(audio_path)
    swap_idx = int(swap_point_s * sr)
    swap_idx = min(swap_idx, len(audio) - 1)

    if replacement_audio_path:
        replacement, _ = sf.read(replacement_audio_path)
        needed = len(audio) - swap_idx
        if len(replacement) < needed:
            replacement = np.tile(replacement, (needed // len(replacement)) + 1)
        replacement = replacement[:needed]
    else:
        # Generate synthetic replacement tone
        t = np.linspace(0, (len(audio) - swap_idx) / sr,
                        len(audio) - swap_idx, endpoint=False)
        replacement = np.sin(2 * np.pi * replacement_freq * t)
        replacement += 0.5 * np.sin(2 * np.pi * replacement_freq * 2 * t)
        replacement *= 0.3
        replacement = replacement.astype(np.float32)

    # Crossfade (50ms)
    xfade_len = min(int(0.05 * sr), swap_idx, len(replacement))
    fade_out = np.linspace(1, 0, xfade_len).astype(np.float32)
    fade_in = np.linspace(0, 1, xfade_len).astype(np.float32)

    result = audio.copy()
    result[swap_idx:] = replacement[:len(audio) - swap_idx]
    # Apply crossfade
    result[swap_idx:swap_idx + xfade_len] = (
        audio[swap_idx:swap_idx + xfade_len] * fade_out +
        replacement[:xfade_len] * fade_in
    )

    sf.write(output_path, result, sr)

    # Estimate original speaker freq from first segment
    orig_segment = audio[:swap_idx]
    orig_freq = _estimate_dominant_freq(orig_segment, sr)
    similarity = 1.0 - abs(orig_freq - replacement_freq) / max(orig_freq, replacement_freq)

    return {
        "corruption_type": "speaker_swap",
        "swap_point_s": swap_point_s,
        "original_speaker_freq": float(orig_freq),
        "replacement_speaker_freq": float(replacement_freq),
        "similarity_score": float(max(0, similarity)),
        "original_path": audio_path,
        "corrupted_path": output_path,
    }


def _estimate_dominant_freq(audio: np.ndarray, sr: int) -> float:
    """Estimate dominant frequency via FFT."""
    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    # Ignore DC and very low frequencies
    mask = freqs > 50
    if not np.any(mask):
        return 200.0
    return float(freqs[mask][np.argmax(fft[mask])])
