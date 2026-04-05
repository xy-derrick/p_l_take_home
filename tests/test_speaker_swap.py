import numpy as np

from corruption.speaker_swap import apply_speaker_swap
from corruption_test_utils import CorruptionTestCase, read_wav, sine, write_wav


class SpeakerSwapTests(CorruptionTestCase):
    def test_apply_speaker_swap_splices_tail_and_reports_similarity(self) -> None:
        sr = 8000
        original = sine(220.0, sr, duration_s=1.0)
        input_path = self.tmp / "speaker_input.wav"
        output_path = self.tmp / "speaker_output.wav"
        write_wav(input_path, original, sr)

        meta = apply_speaker_swap(
            str(input_path),
            str(output_path),
            swap_point_s=0.5,
            replacement_freq=440.0,
        )
        swapped, out_sr = read_wav(output_path)
        self.assertEqual(out_sr, sr)
        self.assertEqual(meta["corruption_type"], "speaker_swap")
        self.assertAlmostEqual(meta["swap_point_s"], 0.5)
        self.assertAlmostEqual(meta["replacement_speaker_freq"], 440.0)
        self.assertGreater(meta["original_speaker_freq"], 200.0)
        self.assertLess(meta["original_speaker_freq"], 240.0)
        self.assertGreaterEqual(meta["similarity_score"], 0.0)
        self.assertLess(meta["similarity_score"], 1.0)

        np.testing.assert_allclose(swapped[:3800], original[:3800], atol=1e-4)
        self.assertGreater(np.mean(np.abs(swapped[5200:] - original[5200:])), 0.05)
