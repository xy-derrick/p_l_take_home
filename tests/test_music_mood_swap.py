import numpy as np

from corruption.music_mood_swap import apply_music_mood_swap
from corruption_test_utils import CorruptionTestCase, read_wav, sine, write_wav


class MusicMoodSwapTests(CorruptionTestCase):
    def test_apply_music_mood_swap_generates_replacement_and_reports_distance(self) -> None:
        sr = 4000
        original = sine(220.0, sr, duration_s=1.0)
        input_path = self.tmp / "music_input.wav"
        output_path = self.tmp / "music_output.wav"
        write_wav(input_path, original, sr)

        original_mood = {"valence": 0.2, "energy": 0.8, "tempo_bpm": 90}
        meta = apply_music_mood_swap(
            str(input_path),
            str(output_path),
            original_mood=original_mood,
            mood_distance=0.8,
        )
        swapped, out_sr = read_wav(output_path)
        self.assertEqual(out_sr, sr)
        self.assertEqual(meta["corruption_type"], "music_mood_swap")
        self.assertEqual(meta["original_mood"], original_mood)
        self.assertAlmostEqual(meta["requested_distance"], 0.8)
        self.assertAlmostEqual(meta["replacement_mood"]["valence"], 0.8)
        self.assertAlmostEqual(meta["replacement_mood"]["energy"], 0.2)
        self.assertAlmostEqual(meta["mood_distance"], 0.6, places=6)
        self.assertEqual(len(swapped), len(original))
        self.assertGreater(np.mean(np.abs(swapped - original)), 0.05)
