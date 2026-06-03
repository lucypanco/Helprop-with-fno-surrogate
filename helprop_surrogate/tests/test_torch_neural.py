import unittest

import numpy as np

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from helprop_surrogate.data import TransitionDataset
from helprop_surrogate.torch_neural import TorchNeuralConditionalKernelSurrogate


def make_dataset(n_per=8):
    etoa_grid = np.geomspace(0.2, 2.0, 4)
    etoa = np.repeat(etoa_grid, n_per)
    d0 = np.linspace(2.0, 8.0, etoa.size)
    m = np.linspace(-0.5, 0.5, etoa.size)
    elis = etoa * np.exp(0.18 + 0.02 * np.log10(d0) - 0.01 * m)
    return TransitionDataset(etoa=etoa, elis=elis, params={"D0": d0, "m": m})


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class TorchNeuralTest(unittest.TestCase):
    def test_torch_neural_matrix_is_probabilistic(self):
        dataset = make_dataset()
        model = TorchNeuralConditionalKernelSurrogate(
            param_names=("D0", "m"),
            hidden_sizes=(8,),
            n_components=2,
            epochs=2,
            batch_size=16,
            learning_rate=0.01,
            device="cpu",
            random_state=3,
        ).fit(dataset.etoa, dataset.elis, dataset.select_params(("D0", "m")))

        self.assertEqual(model.optimizer_name_, "AdamW")
        matrix = model.matrix(
            np.geomspace(0.2, 2.0, 4),
            np.geomspace(0.2, 3.0, 6),
            {"D0": 5.0, "m": 0.0},
        )

        self.assertEqual(matrix.shape, (4, 6))
        self.assertTrue(np.all(matrix >= 0.0))
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()
