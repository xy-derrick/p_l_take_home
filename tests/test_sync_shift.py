import numpy as np

from corruption.sync_shift import apply_sync_shift
from corruption_test_utils import CorruptionTestCase, read_wav, write_wav


class SyncShiftTests(CorruptionTestCase):
    def test_apply_sync_shift_shifts_audio_and_reports_direction(self) -> None:
        sr = 1000
        original = np.linspace(-0.9, 0.9, sr, dtype=np.float32)
        input_path = self.tmp / "sync_input.wav"
        write_wav(input_path, original, sr)

        lag_path = self.tmp / "sync_lag.wav"
        lag_meta = apply_sync_shift(str(input_path), str(lag_path), offset_ms=100)
        lagged, out_sr = read_wav(lag_path)
        self.assertEqual(out_sr, sr)
        self.assertEqual(lag_meta["direction"], "lag")
        np.testing.assert_allclose(lagged[:100], 0.0, atol=5e-5)
        np.testing.assert_allclose(lagged[100:], original[:-100], atol=5e-5)

        lead_path = self.tmp / "sync_lead.wav"
        lead_meta = apply_sync_shift(str(input_path), str(lead_path), offset_ms=-100)
        led, out_sr = read_wav(lead_path)
        self.assertEqual(out_sr, sr)
        self.assertEqual(lead_meta["direction"], "lead")
        np.testing.assert_allclose(led[:-100], original[100:], atol=5e-5)
        np.testing.assert_allclose(led[-100:], 0.0, atol=5e-5)
