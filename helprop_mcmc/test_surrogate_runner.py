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
from surrogate_runner import CompositeSurrogateRunner, SurrogateRunner


class ConstantSpectrumKernel:
    def __init__(self, param_names=("D0", "m"), scale=1.0):
        self.param_names = tuple(param_names)
        self.scale = float(scale)

    def predict_spectrum(self, etoa_grid, elis_grid, lis_flux, theta, A=1):
        return np.full(np.asarray(etoa_grid, dtype=float).shape, self.scale, dtype=float)


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

    def test_composite_runner_splices_two_models_with_log_blend(self):
        low_model = HelPropKernelModel(
            kernel=ConstantSpectrumKernel(scale=10.0),
            learned=("D0", "m"),
            fixed={"A": 1.0},
            etoa_grid=tuple(np.geomspace(1.0e-3, 2.0, 40)),
            elis_grid=tuple(np.geomspace(1.0e-3, 3.0, 45)),
        )
        high_model = HelPropKernelModel(
            kernel=ConstantSpectrumKernel(scale=1000.0),
            learned=("D0", "m"),
            fixed={"A": 1.0},
            etoa_grid=tuple(np.geomspace(0.5, 1.0e6, 40)),
            elis_grid=tuple(np.geomspace(0.5, 1.0e7, 45)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            low_path = os.path.join(tmpdir, "low.pkl")
            high_path = os.path.join(tmpdir, "high.pkl")
            lis_path = os.path.join(tmpdir, "lis.dat")
            low_model.save(low_path)
            high_model.save(high_path)
            energy = np.geomspace(1.0e-4, 1.0e8, 200)
            np.savetxt(lis_path, np.column_stack([energy, energy ** -2.7]))

            runner = CompositeSurrogateRunner(
                low_path,
                high_path,
                lis_path,
                split_energy=1.0,
                blend_dex=0.2,
            )
            interp = runner.run(5.0, 0.0)
            values = interp(np.asarray([0.2, 1.0, 10.0]))

        self.assertAlmostEqual(values[0], 10.0)
        self.assertAlmostEqual(values[1], 100.0)
        self.assertAlmostEqual(values[2], 1000.0)

    def test_composite_runner_cache_stats(self):
        low_model = HelPropKernelModel(
            kernel=ConstantSpectrumKernel(scale=1.0),
            learned=("D0", "m"),
            etoa_grid=tuple(np.geomspace(0.1, 2.0, 8)),
            elis_grid=tuple(np.geomspace(0.1, 3.0, 9)),
        )
        high_model = HelPropKernelModel(
            kernel=ConstantSpectrumKernel(scale=2.0),
            learned=("D0", "m"),
            etoa_grid=tuple(np.geomspace(0.5, 10.0, 8)),
            elis_grid=tuple(np.geomspace(0.5, 20.0, 9)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            low_path = os.path.join(tmpdir, "low.pkl")
            high_path = os.path.join(tmpdir, "high.pkl")
            lis_path = os.path.join(tmpdir, "lis.dat")
            low_model.save(low_path)
            high_model.save(high_path)
            energy = np.geomspace(0.01, 30.0, 80)
            np.savetxt(lis_path, np.column_stack([energy, energy ** -2.7]))

            runner = CompositeSurrogateRunner(low_path, high_path, lis_path)
            first = runner.run(5.0, 0.0)
            second = runner.run(5.0, 0.0)
            stats = runner.stats()

        self.assertIs(first, second)
        self.assertEqual(stats["total_calls"], 1)
        self.assertEqual(stats["cache_hits"], 1)
        self.assertEqual(stats["cache_size"], 1)

    def test_composite_runner_accepts_full_theta_mapping(self):
        names = ("D0", "m", "indexA", "indexB", "angle", "hcs-osc-amp", "hcs-osc-phase")
        low_model = HelPropKernelModel(
            kernel=ConstantSpectrumKernel(param_names=names, scale=1.0),
            learned=names,
            etoa_grid=tuple(np.geomspace(0.1, 2.0, 8)),
            elis_grid=tuple(np.geomspace(0.1, 3.0, 9)),
        )
        high_model = HelPropKernelModel(
            kernel=ConstantSpectrumKernel(param_names=names, scale=2.0),
            learned=names,
            etoa_grid=tuple(np.geomspace(0.5, 10.0, 8)),
            elis_grid=tuple(np.geomspace(0.5, 20.0, 9)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            low_path = os.path.join(tmpdir, "low.pkl")
            high_path = os.path.join(tmpdir, "high.pkl")
            lis_path = os.path.join(tmpdir, "lis.dat")
            low_model.save(low_path)
            high_model.save(high_path)
            energy = np.geomspace(0.01, 30.0, 80)
            np.savetxt(lis_path, np.column_stack([energy, energy ** -2.7]))

            runner = CompositeSurrogateRunner(low_path, high_path, lis_path)
            interp = runner.run({
                "D0": 5.0,
                "m": 0.0,
                "indexA": 1.0,
                "indexB": 1.0,
                "angle": 15.0,
                "hcs-osc-amp": 0.0,
                "hcs-osc-phase": 180.0,
            })

        values = interp(np.asarray([0.2, 2.0]))
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(values > 0.0))

    def test_composite_runner_rejects_missing_blend_coverage(self):
        low_model = HelPropKernelModel(
            kernel=ConstantSpectrumKernel(scale=1.0),
            learned=("D0", "m"),
            etoa_grid=tuple(np.geomspace(1.0e-3, 0.7, 8)),
            elis_grid=tuple(np.geomspace(1.0e-3, 1.0, 9)),
        )
        high_model = HelPropKernelModel(
            kernel=ConstantSpectrumKernel(scale=2.0),
            learned=("D0", "m"),
            etoa_grid=tuple(np.geomspace(0.5, 10.0, 8)),
            elis_grid=tuple(np.geomspace(0.5, 20.0, 9)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            low_path = os.path.join(tmpdir, "low.pkl")
            high_path = os.path.join(tmpdir, "high.pkl")
            lis_path = os.path.join(tmpdir, "lis.dat")
            low_model.save(low_path)
            high_model.save(high_path)
            energy = np.geomspace(0.01, 30.0, 80)
            np.savetxt(lis_path, np.column_stack([energy, energy ** -2.7]))

            with self.assertRaisesRegex(ValueError, "blend window"):
                CompositeSurrogateRunner(low_path, high_path, lis_path)


if __name__ == "__main__":
    unittest.main()
