import shutil
import sys
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def write_wav(path: Path, data: np.ndarray, sr: int) -> None:
    sf.write(path, data.astype(np.float32), sr, subtype="FLOAT")


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path)
    return data.astype(np.float32), sr


def sine(freq: float, sr: int, duration_s: float, amplitude: float = 0.3) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class CorruptionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        safe_name = self.id().replace(".", "_")
        self.tmp = REPO_ROOT / "tests_tmp" / safe_name
        if self.tmp.exists():
            shutil.rmtree(self.tmp)
        self.tmp.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.tmp.exists():
            shutil.rmtree(self.tmp)
