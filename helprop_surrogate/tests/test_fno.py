import json
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

from helprop_surrogate.fno.model import TorchFNOTransferMatrixSurrogate, _piecewise_bin_error_loss
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
    def test_piecewise_bin_loss_penalizes_one_bad_bin(self):
        bad_one_bin = torch.tensor([[0.020, 0.0, 0.0, 0.0]], requires_grad=True)
        spread_below_margin = torch.full((1, 4), 0.005, requires_grad=True)

        bad_loss = _piecewise_bin_error_loss(
            bad_one_bin,
            threshold=0.01,
            temperature=0.0001,
            huber_delta=0.0001,
            torch=torch,
        )
        spread_loss = _piecewise_bin_error_loss(
            spread_below_margin,
            threshold=0.01,
            temperature=0.0001,
            huber_delta=0.0001,
            torch=torch,
        )

        self.assertGreater(float(bad_loss.detach()), 100.0 * float(spread_loss.detach()))
        huge_loss = _piecewise_bin_error_loss(
            torch.tensor([[1.0e7]]),
            threshold=0.009,
            temperature=0.0001,
            huber_delta=0.001,
            torch=torch,
        )
        self.assertLess(float(huge_loss.detach()), 1.1e7)
        bad_loss.sum().backward()
        spread_loss.sum().backward()
        self.assertTrue(torch.isfinite(bad_one_bin.grad).all())
        self.assertTrue(torch.isfinite(spread_below_margin.grad).all())

    def test_fno_matrix_is_probabilistic_and_saved_model_loads(self):
        dataset = make_matrix_dataset(n_samples=6)
        model = TorchFNOTransferMatrixSurrogate(
            param_names=("D0", "m"),
            width=8,
            modes_etoa=2,
            modes_elis=2,
            n_layers=1,
            projection_size=8,
            boundary_padding=1,
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
            spectrum_lis_flux=dataset.elis_grid ** -2.7,
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
            np.savetxt(dataset_path.parent / "train_indices.txt", np.arange(0, 6), fmt="%d")
            np.savetxt(dataset_path.parent / "val_indices.txt", np.arange(6, 8), fmt="%d")
            np.savetxt(dataset_path.parent / "test_indices.txt", np.arange(8, 9), fmt="%d")
            outdir = dataset_path.parent.parent
            lis_energy = np.geomspace(0.1, 10.0, 20)
            lis_path = Path(tmpdir) / "lis.txt"
            np.savetxt(lis_path, np.column_stack([lis_energy, lis_energy ** -2.7]))

            code = train_fno_main(
                [
                    "--dataset",
                    str(dataset_path),
                    "--lis",
                    str(lis_path),
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
                    "--boundary-padding",
                    "1",
                    "--dropout",
                    "0",
                    "--device",
                    "cpu",
                    "--checkpoint-every",
                    "1",
                    "--lr-scheduler-patience",
                    "1",
                    "--early-stopping-min-epochs",
                    "1",
                    "--early-stopping-patience",
                    "2",
                    "--matrix-cross-entropy-weight",
                    "0",
                    "--matrix-probability-loss-weight",
                    "2.0",
                    "--spectrum-loss-weight",
                    "2.5",
                    "--spectrum-max-error-percent",
                    "1",
                    "--spectrum-max-error-temperature-percent",
                    "0.01",
                    "--spectrum-huber-delta-percent",
                    "0.1",
                    "--spectrum-top-k",
                    "2",
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
            self.assertTrue(config["checkpoints"])
            self.assertEqual(config["lr_scheduler"], "plateau")
            self.assertEqual(config["lr_scheduler_factor"], 0.3)
            self.assertEqual(config["lr_scheduler_patience"], 1)
            self.assertEqual(config["lr_scheduler_cooldown"], 5)
            self.assertEqual(config["lr_scheduler_min_lr"], 1.0e-6)
            self.assertTrue(config["early_stopping"])
            self.assertEqual(config["early_stopping_patience"], 2)
            self.assertEqual(config["early_stopping_min_delta"], 0.1)
            self.assertEqual(config["early_stopping_min_epochs"], 1)
            self.assertEqual(config["lis"], str(lis_path))
            self.assertEqual(config["matrix_cross_entropy_weight"], 0.0)
            self.assertEqual(config["matrix_probability_loss_weight"], 2.0)
            self.assertEqual(config["boundary_padding"], 1)
            self.assertEqual(config["spectrum_loss_weight"], 2.5)
            self.assertEqual(config["spectrum_max_error_percent"], 1.0)
            self.assertEqual(config["spectrum_max_error_temperature_percent"], 0.01)
            self.assertEqual(config["spectrum_huber_delta_percent"], 0.1)
            self.assertEqual(config["spectrum_top_k"], 2)
            checkpoint = torch.load(
                outdir / "checkpoints" / "best.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(checkpoint["optimizer"], "AdamW")
            self.assertEqual(checkpoint["lr_scheduler"], "plateau")
            self.assertEqual(checkpoint["lr_scheduler_factor"], 0.3)
            self.assertEqual(checkpoint["lr_scheduler_patience"], 1)
            self.assertEqual(checkpoint["lr_scheduler_cooldown"], 5)
            self.assertEqual(checkpoint["lr_scheduler_min_lr"], 1.0e-6)
            self.assertTrue(checkpoint["early_stopping"])
            self.assertEqual(checkpoint["early_stopping_patience"], 2)
            self.assertEqual(checkpoint["early_stopping_min_delta"], 0.1)
            self.assertEqual(checkpoint["early_stopping_min_epochs"], 1)
            self.assertEqual(checkpoint["matrix_cross_entropy_weight"], 0.0)
            self.assertEqual(checkpoint["matrix_probability_loss_weight"], 2.0)
            self.assertEqual(checkpoint["boundary_padding"], 1)
            self.assertEqual(checkpoint["spectrum_loss_weight"], 2.5)
            self.assertEqual(checkpoint["spectrum_max_error_percent"], 1.0)
            self.assertEqual(checkpoint["spectrum_max_error_temperature_percent"], 0.01)
            self.assertEqual(checkpoint["spectrum_huber_delta_percent"], 0.1)
            self.assertEqual(checkpoint["spectrum_top_k"], 2)


if __name__ == "__main__":
    unittest.main()
