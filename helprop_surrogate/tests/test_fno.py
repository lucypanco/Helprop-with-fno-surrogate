import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from helprop_surrogate.fno.model import TorchFNOTransferMatrixSurrogate
from helprop_surrogate.matrix_data import MatrixDataset
from helprop_surrogate.model import HelPropKernelModel, load_model
from helprop_surrogate.fno.train import main as train_fno_main


def make_matrix_dataset(n_samples=9):
    theta = np.column_stack(
        [
            np.linspace(1.0, 9.0, n_samples),
            np.linspace(-0.4, 0.4, n_samples),
        ]
    )
    etoa = np.geomspace(0.2, 2.0, 4)
    elis = np.geomspace(0.2, 4.0, 5)
    matrices = []
    for d0, m in theta:
        rows = []
        for energy in etoa:
            center = np.log(energy * (1.15 + 0.02 * np.log10(d0) - 0.01 * m))
            widths = 0.45 + 0.02 * abs(m)
            row = np.exp(-0.5 * ((np.log(elis) - center) / widths) ** 2)
            rows.append(row / row.sum())
        matrices.append(rows)
    return MatrixDataset(
        theta=theta,
        matrices=np.asarray(matrices),
        param_names=("D0", "m"),
        etoa_grid=etoa,
        elis_grid=elis,
    )


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class FNOTest(unittest.TestCase):
    def test_fno_matrix_is_probabilistic_and_saved_model_loads(self):
        dataset = make_matrix_dataset(n_samples=6)
        model = TorchFNOTransferMatrixSurrogate(
            param_names=("D0", "m"),
            width=8,
            modes_etoa=2,
            modes_elis=2,
            n_layers=1,
            projection_size=8,
            dropout=0.0,
            epochs=1,
            batch_size=3,
            learning_rate=0.002,
            device="cpu",
            random_state=5,
        ).fit(
            dataset.theta[:4],
            dataset.matrices[:4],
            dataset.etoa_grid,
            dataset.elis_grid,
            val_theta=dataset.theta[4:],
            val_matrices=dataset.matrices[4:],
        )

        self.assertEqual(model.optimizer_name_, "AdamW")
        matrix = model.matrix(dataset.etoa_grid, dataset.elis_grid, {"D0": 4.0, "m": 0.0})
        self.assertEqual(matrix.shape, (4, 5))
        self.assertTrue(np.all(matrix >= 0.0))
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6)

        wrapped = HelPropKernelModel(
            kernel=model,
            learned=("D0", "m"),
            etoa_grid=tuple(dataset.etoa_grid),
            elis_grid=tuple(dataset.elis_grid),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "kernel_fno.pkl"
            wrapped.save(path)
            loaded = load_model(path)
            loaded_matrix = loaded.matrix({"D0": 4.0, "m": 0.0})

        np.testing.assert_allclose(loaded_matrix.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6)

    def test_train_fno_writes_loss_validation_and_test_outputs(self):
        dataset = make_matrix_dataset(n_samples=9)
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "fno_runs" / "run_0001" / "data" / "matrices.npz"
            dataset.save_npz(dataset_path)
            outdir = dataset_path.parent.parent

            code = train_fno_main(
                [
                    "--dataset",
                    str(dataset_path),
                    "--outdir",
                    str(outdir),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "3",
                    "--width",
                    "8",
                    "--layers",
                    "1",
                    "--modes",
                    "2",
                    "--projection-size",
                    "8",
                    "--dropout",
                    "0",
                    "--device",
                    "cpu",
                    "--checkpoint-every",
                    "1",
                    "--seed",
                    "11",
                ]
            )

            self.assertEqual(code, 0)
            self.assertTrue((outdir / "kernel_fno.pkl").exists())
            self.assertTrue((outdir / "logs" / "loss_history.csv").exists())
            self.assertTrue((outdir / "validation" / "val_predictions.npz").exists())
            self.assertTrue((outdir / "validation" / "val_residuals.npz").exists())
            self.assertTrue((outdir / "testing" / "test_metrics.json").exists())
            self.assertFalse((outdir / "testing" / "test_predictions.npz").exists())
            self.assertTrue((outdir / "checkpoints" / "best.pt").exists())

            config = json.loads((outdir / "config.json").read_text())
            self.assertEqual(config["optimizer"], "AdamW")
            checkpoint = torch.load(
                outdir / "checkpoints" / "best.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(checkpoint["optimizer"], "AdamW")


if __name__ == "__main__":
    unittest.main()
