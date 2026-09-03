import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from helprop_surrogate.fno.train import main as train_fno_main
from helprop_surrogate.tests.test_fno import make_matrix_dataset


class NoCheckpointsTest(unittest.TestCase):
    def test_outputs_are_written_without_checkpoint_directory(self):
        dataset = make_matrix_dataset(n_samples=9)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "data" / "matrices.npz"
            dataset.save_npz(dataset_path)
            np.savetxt(dataset_path.parent / "train_indices.txt", np.arange(0, 6), fmt="%d")
            np.savetxt(dataset_path.parent / "val_indices.txt", np.arange(6, 8), fmt="%d")
            np.savetxt(dataset_path.parent / "test_indices.txt", np.arange(8, 9), fmt="%d")
            lis_energy = np.geomspace(0.1, 10.0, 20)
            lis_path = root / "lis.txt"
            np.savetxt(lis_path, np.column_stack([lis_energy, lis_energy ** -2.7]))
            outdir = root / "run"

            self.assertEqual(
                train_fno_main(
                    [
                        "--dataset", str(dataset_path),
                        "--outdir", str(outdir),
                        "--lis", str(lis_path),
                        "--epochs", "1",
                        "--batch-size", "3",
                        "--width", "8",
                        "--layers", "1",
                        "--modes", "2",
                        "--projection-size", "8",
                        "--dropout", "0",
                        "--device", "cpu",
                        "--checkpoint-every", "1",
                        "--no-checkpoints",
                        "--reserve-checkpoint", str(root / "reserve.pt"),
                    ]
                ),
                0,
            )
            self.assertFalse((outdir / "checkpoints").exists())
            self.assertTrue((root / "reserve.pt").exists())
            self.assertTrue((outdir / "kernel_fno.pkl").exists())
            self.assertTrue((outdir / "logs" / "loss_history.csv").exists())
            self.assertTrue((outdir / "validation" / "val_metrics.csv").exists())
            self.assertTrue((outdir / "testing" / "test_metrics.json").exists())
            self.assertFalse(json.loads((outdir / "config.json").read_text())["checkpoints"])


if __name__ == "__main__":
    unittest.main()
