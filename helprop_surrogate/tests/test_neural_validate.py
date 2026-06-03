import unittest

import numpy as np

from helprop_surrogate.data import TransitionDataset
from helprop_surrogate.neural import NeuralConditionalKernelSurrogate
from helprop_surrogate.validate import (
    append_dataset,
    batch_quality_report,
    empirical_matrix,
    matrix_validation_report,
    row_kl,
    should_accept,
)


def make_dataset(n_per=12):
    etoa_grid = np.geomspace(0.2, 3.0, 5)
    etoa = np.repeat(etoa_grid, n_per)
    d0 = np.linspace(2.0, 8.0, etoa.size)
    m = np.linspace(-0.5, 0.5, etoa.size)
    elis = etoa * np.exp(0.18 + 0.02 * np.log10(d0) - 0.01 * m)
    return TransitionDataset(etoa=etoa, elis=elis, params={"D0": d0, "m": m})


class NeuralAndValidationTest(unittest.TestCase):
    def test_neural_surrogate_matrix_is_probabilistic(self):
        dataset = make_dataset()
        model = NeuralConditionalKernelSurrogate(
            param_names=("D0", "m"),
            hidden_size=8,
            n_components=2,
            epochs=8,
            batch_size=16,
            learning_rate=0.02,
            random_state=4,
        ).fit(dataset.etoa, dataset.elis, dataset.select_params(("D0", "m")))

        etoa_grid = np.geomspace(0.2, 3.0, 5)
        elis_grid = np.geomspace(0.2, 4.0, 7)
        matrix = model.matrix(etoa_grid, elis_grid, {"D0": 5.0, "m": 0.0})

        self.assertEqual(matrix.shape, (5, 7))
        self.assertTrue(np.all(matrix >= 0.0))
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0)

    def test_validation_metrics_and_adoption(self):
        dataset = make_dataset()
        etoa_grid = np.geomspace(0.2, 3.0, 5)
        elis_grid = np.geomspace(0.2, 4.0, 7)
        matrix = empirical_matrix(dataset, etoa_grid, elis_grid)
        report = batch_quality_report(dataset, etoa_grid, elis_grid, min_samples_per_row=2)

        self.assertTrue(report["accepted_basic"])
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
        np.testing.assert_allclose(row_kl(matrix, matrix), 0.0, atol=1e-12)

        summary = matrix_validation_report(matrix, matrix)
        self.assertEqual(summary["mean_row_kl"], 0.0)
        self.assertTrue(should_accept({**report, **summary}, max_mean_kl=0.01, max_flux_relerr=None))

        combined = append_dataset(dataset, dataset)
        self.assertEqual(combined.etoa.size, 2 * dataset.etoa.size)


if __name__ == "__main__":
    unittest.main()
