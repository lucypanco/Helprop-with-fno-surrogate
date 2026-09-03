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

from formula_lis import shen_lis_flux, write_shen_lis
from mcmc_analysis import DEFAULT_PARAM_RANGES, is_formula_lis, make_log_likelihood
from surrogate_runner import _LISSource


class _PowerLawRunner:
    def run(self, theta):
        return lambda energy: 2.0 * np.asarray(energy, dtype=float) ** -2.0


class FormulaLISTest(unittest.TestCase):
    def test_m_parameter_is_not_an_mcmc_parameter(self):
        self.assertNotIn("m", DEFAULT_PARAM_RANGES)

    def test_lis_argument_selects_formula_mode(self):
        self.assertTrue(is_formula_lis("formula"))
        self.assertTrue(is_formula_lis("Shen2025"))
        self.assertFalse(is_formula_lis("my_lis.dat"))

    def test_surrogate_formula_source_responds_to_shape_parameters(self):
        source = _LISSource(
            "formula", formula_lis=True, lis_min=1.0e-3,
            lis_max=100.0, lis_bins=32,
        )
        base = {"lis_a1": 0.0, "lis_a2": 0.5, "lis_a3": 2.0,
                "lis_a4": -2.0, "lis_a5": 5.0, "lis_a6": 2.0,
                "lis_a7": -1.0}
        changed = dict(base, lis_a1=1.0)
        self.assertNotEqual(source.cache_key(base), source.cache_key(changed))
        first = source.interpolate(np.asarray([0.1, 1.0, 10.0]), base)
        second = source.interpolate(np.asarray([0.1, 1.0, 10.0]), changed)
        self.assertFalse(np.allclose(first, second))
        scaled = source.interpolate(
            np.asarray([0.1, 1.0, 10.0]), dict(base, lis_a0=3.0)
        )
        np.testing.assert_allclose(scaled, 3.0 * first)

    def test_shen_equation_uses_two_smooth_breaks(self):
        params = {
            "lis_a0": 3.0,
            "lis_a1": 1.0,
            "lis_a2": 2.0,
            "lis_a3": 1.0,
            "lis_a4": -1.0,
            "lis_a5": 4.0,
            "lis_a6": 1.0,
            "lis_a7": -2.0,
        }
        energy = np.asarray([1.0, 2.0, 4.0])
        expected = (
            3.0 * energy
            * (1.0 + energy / 2.0) ** -1.0
            * (1.0 + energy / 4.0) ** -2.0
        )
        np.testing.assert_allclose(shen_lis_flux(energy, params), expected)

    def test_write_shen_lis_is_helprop_compatible(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "lis.dat")
            energy = np.geomspace(1.0e-3, 100.0, 16)
            params = {
                "lis_a0": 1.0,
                "lis_a1": 0.0,
                "lis_a2": 0.5,
                "lis_a3": 2.0,
                "lis_a4": -2.0,
                "lis_a5": 5.0,
                "lis_a6": 2.0,
                "lis_a7": -1.0,
            }
            write_shen_lis(path, energy, params)
            loaded = np.loadtxt(path)
        self.assertEqual(loaded.shape, (16, 2))
        self.assertTrue(np.all(np.diff(loaded[:, 0]) > 0.0))
        self.assertTrue(np.all(loaded[:, 1] > 0.0))

    def test_profiled_normalization_removes_pure_amplitude(self):
        energies = np.asarray([0.5, 1.0, 2.0])
        model_shape = 2.0 * energies ** -2.0
        observed = 7.0 * model_shape
        likelihood = make_log_likelihood(
            _PowerLawRunner(), energies, observed, 0.05 * observed,
            ("D0", "m"), profile_lis_norm=True,
        )
        self.assertAlmostEqual(likelihood(np.asarray([1.0, 0.0])), 0.0)

    def test_fixed_shen_parameters_are_used_by_lis_source(self):
        source = _LISSource(
            "formula", formula_lis=True, lis_min=1.0e-3,
            lis_max=100.0, lis_bins=32,
            lis_params={"lis_a6": 1.53, "lis_a7": -1.2},
        )
        self.assertAlmostEqual(source.fixed_params["lis_a6"], 1.53)
        self.assertAlmostEqual(source.fixed_params["lis_a7"], -1.2)
        energy = np.asarray([0.1, 1.0, 10.0])
        fixed = source.interpolate(energy, {})
        sampled = source.interpolate(energy, {"lis_a6": 2.0})
        self.assertFalse(np.allclose(fixed, sampled))

if __name__ == "__main__":
    unittest.main()
