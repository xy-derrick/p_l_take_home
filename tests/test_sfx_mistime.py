import numpy as np

from corruption.sfx_mistime import apply_sfx_mistime
from corruption_test_utils import CorruptionTestCase, read_wav, write_wav


class SfxMistimeTests(CorruptionTestCase):
    def test_apply_sfx_mistime_moves_event_and_reports_shift(self) -> None:
        sr = 1000
        original = np.zeros(sr, dtype=np.float32)
        original[500] = 1.0
        input_path = self.tmp / "sfx_input.wav"
        output_path = self.tmp / "sfx_output.wav"
        write_wav(input_path, original, sr)

        meta = apply_sfx_mistime(
            str(input_path),
            str(output_path),
            shift_ms=100,
            event_timestamps=[0.5],
        )
        shifted, out_sr = read_wav(output_path)
        self.assertEqual(out_sr, sr)
        self.assertEqual(meta["corruption_type"], "sfx_mistime")
        self.assertEqual(meta["events_shifted"], 1)
        self.assertEqual(meta["shift_ms"], 100)
        self.assertAlmostEqual(meta["shifted_events"][0]["original_time_s"], 0.5)
        self.assertAlmostEqual(meta["shifted_events"][0]["shifted_time_s"], 0.6)
        self.assertEqual(int(np.argmax(np.abs(shifted))), 600)
