"""A/V sync drift injection — shift audio by N milliseconds."""

import numpy as np
import soundfile as sf


def apply_sync_shift(audio_path: str, output_path: str, *,
                     offset_ms: int = 0, **kwargs) -> dict:
    """Shift audio by offset_ms milliseconds.

    Positive = audio lags (delayed), Negative = audio leads.
    """
    audio, sr = sf.read(audio_path)
    offset_samples = int(abs(offset_ms) * sr / 1000)

    if offset_ms > 0:
        # Lag: prepend silence
        silence = np.zeros(offset_samples, dtype=audio.dtype)
        shifted = np.concatenate([silence, audio])[:len(audio)]
    elif offset_ms < 0:
        # Lead: trim beginning, pad end
        shifted = np.concatenate([audio[offset_samples:],
                                  np.zeros(offset_samples, dtype=audio.dtype)])
    else:
        shifted = audio.copy()

    sf.write(output_path, shifted, sr)
    return {
        "corruption_type": "sync_shift",
        "offset_ms": offset_ms,
        "direction": "lag" if offset_ms > 0 else ("lead" if offset_ms < 0 else "none"),
        "original_path": audio_path,
        "corrupted_path": output_path,
    }
