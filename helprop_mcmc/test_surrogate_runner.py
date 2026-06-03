import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

from helprop_surrogate.kernel import ConditionalKernelSurrogate
from helprop_surrogate.model import HelPropKernelModel
from surrogate_runner import SurrogateRunner


class SurrogateRunnerTest(unittest.TestCase):
    def test_runner_returns_log_interp_from_saved_model(self):
        etoa = np.repeat(np.geomspace(0.2, 2.0, 4), 6)
        d0 = np.linspace(2.0, 8.0, etoa.size)
        m = np.linspace(-0.2, 0.2, etoa.size)
        elis = etoa * np.exp(0.2 + 0.01 * np.log10(d0))
        kernel = ConditionalKernelSurrogate(param_names=("D0", "m")).fit(
            etoa,
            elis,
            {"D0": d0, "m": m},
        )
        model = HelPropKernelModel(
            kernel=kernel,
            learned=("D0", "m"),
            fixed={"A": 1.0},
            etoa_grid=tuple(np.geomspace(0.2, 2.0, 4)),
            elis_grid=tuple(np.geomspace(0.2, 3.0, 6)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.pkl")
            lis_path = os.path.join(tmpdir, "lis.dat")
            model.save(model_path)
            energy = np.geomspace(0.1, 5.0, 20)
            np.savetxt(lis_path, np.column_stack([energy, energy ** -2.7]))

            runner = SurrogateRunner(model_path, lis_path)
            interp = runner.run(5.0, 0.0)
            values = interp(np.array([0.25, 1.0]))

        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(values > 0.0))


if __name__ == "__main__":
    unittest.main()
