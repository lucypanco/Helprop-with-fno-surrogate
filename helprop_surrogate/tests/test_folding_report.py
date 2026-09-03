import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from helprop_surrogate.folding_report import folding_report, folded_relative_error, main
from helprop_surrogate.matrix_data import MatrixDataset


class FakeModel:
    learned = ("D0",)
    fixed = {"A": 1}
    kernel = object()

    def __init__(self, matrices):
        self._matrices = matrices

    def matrix(self, options, etoa_grid, elis_grid):
        index = int(round(options["D0"]))
        return self._matrices[index]


class FoldingReportTest(unittest.TestCase):
    def test_report_can_restrict_error_to_an_etoa_window(self):
        etoa = np.array([0.2, 1.0, 10.0, 100.0, 200.0])
        elis = np.array([0.2, 1.0, 10.0, 100.0, 200.0])
        true = np.tile(np.eye(5)[None, :, :], (2, 1, 1))
        predicted = true.copy()
        predicted[1, 0, 0] = 0.5
        predicted[1, 0, 1] = 0.5
        dataset = MatrixDataset(
            theta=np.array([[0.0], [1.0]]),
            matrices=true,
            param_names=("D0",),
            etoa_grid=etoa,
            elis_grid=elis,
        )

        full = folding_report(FakeModel(predicted), dataset, elis ** -2.7)
        restricted = folding_report(
            FakeModel(predicted),
            dataset,
            elis ** -2.7,
            etoa_min=1.0,
            etoa_max=100.0,
        )

        self.assertGreater(full["max_relative_error"], restricted["max_relative_error"])
        self.assertEqual(restricted["etoa_range"]["n_grid_points"], 3)
        self.assertEqual(restricted["matrices"][1]["max_error_etoa"], 1.0)

    def test_folded_relative_error_and_per_matrix_threshold(self):
        etoa = np.array([0.2, 0.5])
        elis = np.array([0.2, 1.0, 3.0])
        true = np.array(
            [
                [[0.2, 0.5, 0.3], [0.1, 0.2, 0.7]],
                [[0.3, 0.4, 0.3], [0.2, 0.5, 0.3]],
            ]
        )
        predicted = true.copy()
        predicted[1, 0, 0] += 0.001
        predicted[1, 0] /= predicted[1, 0].sum()
        dataset = MatrixDataset(
            theta=np.array([[0.0], [1.0]]),
            matrices=true,
            param_names=("D0",),
            etoa_grid=etoa,
            elis_grid=elis,
        )
        lis = elis ** -2.7
        errors = folded_relative_error(true[1], predicted[1], etoa, elis, lis)
        self.assertGreater(float(np.max(errors)), 0.0)

        report = folding_report(FakeModel(predicted), dataset, lis, threshold=1.0e-12)
        self.assertFalse(report["passed"])
        self.assertEqual(report["n_matrices"], 2)
        self.assertEqual(report["n_failed"], 1)
        self.assertTrue(report["matrices"][0]["passed"])
        self.assertFalse(report["matrices"][1]["passed"])
        self.assertEqual(
            report["matrices"][1]["max_relative_error_point"]["etoa"],
            report["matrices"][1]["max_error_etoa"],
        )
        self.assertEqual(
            report["max_relative_error_point"]["etoa"],
            report["matrices"][1]["max_error_etoa"],
        )

        boundary_report = folding_report(
            FakeModel(predicted), dataset, lis, threshold=float(np.max(errors))
        )
        self.assertFalse(boundary_report["matrices"][1]["passed"])

        limited = folding_report(FakeModel(predicted), dataset, lis, max_matrices=1)
        self.assertEqual(limited["n_matrices"], 1)
        self.assertEqual(limited["selection"]["stop_index_exclusive"], 1)
        self.assertEqual([row["index"] for row in limited["matrices"]], [0])

    def test_report_uses_per_matrix_workers_and_progress(self):
        etoa = np.array([0.2, 0.5])
        elis = np.array([0.2, 1.0, 3.0])
        matrices = np.array(
            [
                [[0.2, 0.5, 0.3], [0.1, 0.2, 0.7]],
                [[0.3, 0.4, 0.3], [0.2, 0.5, 0.3]],
            ]
        )
        dataset = MatrixDataset(
            theta=np.array([[0.0], [1.0]]),
            matrices=matrices,
            param_names=("D0",),
            etoa_grid=etoa,
            elis_grid=elis,
        )
        model = FakeModel(matrices)
        progress = []

        report = folding_report(
            model,
            dataset,
            elis ** -2.7,
            workers=2,
            progress_every=1,
            progress=lambda processed, total, elapsed: progress.append(processed),
        )

        self.assertTrue(report["passed"])
        self.assertEqual(progress, [1, 2])

    def test_cli_writes_standalone_json_report(self):
        etoa = np.array([0.2, 0.5])
        elis = np.array([0.2, 1.0, 3.0])
        matrices = np.array(
            [
                [[0.2, 0.5, 0.3], [0.1, 0.2, 0.7]],
                [[0.3, 0.4, 0.3], [0.2, 0.5, 0.3]],
            ]
        )
        dataset = MatrixDataset(
            theta=np.array([[0.0], [1.0]]),
            matrices=matrices,
            param_names=("D0",),
            etoa_grid=etoa,
            elis_grid=elis,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "matrices.npz"
            lis_path = root / "lis.txt"
            report_path = root / "reports" / "folding.json"
            dataset.save_npz(dataset_path)
            np.savetxt(lis_path, np.column_stack([np.array([0.1, 5.0]), np.array([1.0, 0.01])]))

            with patch("helprop_surrogate.folding_report.load_model", return_value=FakeModel(matrices)):
                code = main(
                    [
                        "--model", str(root / "kernel.pkl"),
                        "--dataset", str(dataset_path),
                        "--lis", str(lis_path),
                        "--report-out", str(report_path),
                    ]
                )

            self.assertEqual(code, 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertEqual(report["n_matrices"], 2)


if __name__ == "__main__":
    unittest.main()
