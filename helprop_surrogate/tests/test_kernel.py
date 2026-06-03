import os
import tempfile
import unittest

import numpy as np

from helprop_surrogate import ConditionalKernelSurrogate, TransitionDataset
from helprop_surrogate.data import load_npz_transitions
from helprop_surrogate.kernel import fold_lis


def make_training_data(n=160):
    rng = np.random.default_rng(123)
    etoa_base = np.geomspace(0.2, 5.0, 8)
    etoa = np.repeat(etoa_base, n // etoa_base.size)
    d0 = np.linspace(2.0, 9.0, etoa.size)
    m = np.sin(np.linspace(0.0, 2.0 * np.pi, etoa.size))
    shift = 0.18 + 0.02 * np.log10(d0) - 0.01 * m
    elis = etoa * np.exp(shift + 0.04 * rng.standard_normal(etoa.size))
    return etoa, elis, {"D0": d0, "m": m}


class ConditionalKernelSurrogateTest(unittest.TestCase):
    def test_matrix_is_nonnegative_and_row_normalized(self):
        etoa, elis, params = make_training_data()
        model = ConditionalKernelSurrogate(condition_bandwidth=0.8).fit(
            etoa, elis, params
        )

        etoa_grid = np.geomspace(0.2, 5.0, 7)
        elis_grid = np.geomspace(0.15, 8.0, 12)
        matrix = model.matrix(etoa_grid, elis_grid, {"D0": 5.0, "m": 0.0})

        self.assertEqual(matrix.shape, (7, 12))
        self.assertTrue(np.all(np.isfinite(matrix)))
        self.assertTrue(np.all(matrix >= 0.0))
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0, rtol=1e-12, atol=1e-12)

    def test_predict_spectrum_matches_explicit_fold(self):
        etoa, elis, params = make_training_data()
        model = ConditionalKernelSurrogate(condition_bandwidth=0.8).fit(
            etoa, elis, params
        )

        etoa_grid = np.geomspace(0.2, 5.0, 6)
        elis_grid = np.geomspace(0.15, 8.0, 10)
        lis_flux = elis_grid ** -2.7
        theta = [4.0, 0.2]

        matrix = model.matrix(etoa_grid, elis_grid, theta)
        expected = fold_lis(matrix, etoa_grid, elis_grid, lis_flux)
        actual = model.predict_spectrum(etoa_grid, elis_grid, lis_flux, theta)
        np.testing.assert_allclose(actual, expected)

    def test_transition_dataset_npz_roundtrip(self):
        etoa, elis, params = make_training_data(n=80)
        dataset = TransitionDataset(etoa=etoa, elis=elis, params=params)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "transitions.npz")
            dataset.save_npz(path)
            loaded = load_npz_transitions(path)

        np.testing.assert_allclose(loaded.etoa, dataset.etoa)
        np.testing.assert_allclose(loaded.elis, dataset.elis)
        np.testing.assert_allclose(loaded.params["D0"], dataset.params["D0"])
        np.testing.assert_allclose(loaded.params["m"], dataset.params["m"])


if __name__ == "__main__":
    unittest.main()
