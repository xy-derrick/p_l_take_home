import numpy as np

from corruption.artifact_inject import inject_artifacts
from corruption_test_utils import CorruptionTestCase, read_wav, write_wav


class ArtifactInjectTests(CorruptionTestCase):
    def test_inject_artifacts_handles_each_supported_artifact_type(self) -> None:
        ts = 0.2
        cases = [
            ("click", 1000, np.full(1000, 0.1, dtype=np.float32)),
            ("dropout", 1000, np.full(1000, 0.1, dtype=np.float32)),
            ("spectral", 16000, np.full(16000, 0.1, dtype=np.float32)),
            ("stutter", 1000, np.linspace(-0.5, 0.5, 1000, dtype=np.float32)),
        ]

        for artifact_type, sr, original in cases:
            with self.subTest(artifact_type=artifact_type):
                idx = int(ts * sr)
                input_path = self.tmp / f"{artifact_type}_input.wav"
                output_path = self.tmp / f"{artifact_type}_output.wav"
                write_wav(input_path, original, sr)

                meta = inject_artifacts(
                    str(input_path),
                    str(output_path),
                    artifact_type=artifact_type,
                    timestamps=[ts],
                    severity=0.5,
                )
                corrupted, out_sr = read_wav(output_path)
                self.assertEqual(out_sr, sr)
                self.assertEqual(meta["corruption_type"], "artifact_inject")
                self.assertEqual(meta["artifact_type"], artifact_type)
                self.assertEqual(len(meta["artifacts"]), 1)
                self.assertAlmostEqual(meta["artifacts"][0]["timestamp_s"], ts)

                if artifact_type == "click":
                    self.assertGreater(abs(corrupted[idx]), abs(original[idx]))
                elif artifact_type == "dropout":
                    dropout_samples = int((50 + 0.5 * 200) * sr / 1000)
                    np.testing.assert_allclose(corrupted[idx:idx + dropout_samples], 0.0, atol=5e-5)
                elif artifact_type == "spectral":
                    self.assertGreater(
                        np.max(np.abs(corrupted[idx:idx + int(0.05 * sr)] - original[idx:idx + int(0.05 * sr)])),
                        0.05,
                    )
                elif artifact_type == "stutter":
                    np.testing.assert_allclose(corrupted[idx + 50:idx + 100], original[idx:idx + 50], atol=5e-5)
