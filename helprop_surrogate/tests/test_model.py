import os
import tempfile
import unittest

import numpy as np

from helprop_surrogate.kernel import ConditionalKernelSurrogate
from helprop_surrogate.model import (
    HelPropKernelModel,
    load_model,
    parse_key_value_options,
    parse_range_options,
)
from helprop_surrogate.predict_kernel import _runtime_options_from_args
from helprop_surrogate.predict_kernel import main as predict_kernel_main


def make_kernel(param_names=("D0", "m")):
    etoa = np.repeat(np.geomspace(0.2, 3.0, 6), 8)
    d0 = np.linspace(1.0, 10.0, etoa.size)
    m = np.linspace(-1.0, 1.0, etoa.size)
    params = {"D0": d0, "m": m}
    elis = etoa * np.exp(0.2 + 0.01 * np.log10(d0) - 0.01 * m)
    return ConditionalKernelSurrogate(param_names=param_names).fit(
        etoa,
        elis,
        {name: params[name] for name in param_names},
    )


class HelPropKernelModelTest(unittest.TestCase):
    def test_fixed_parameters_are_rejected_at_runtime(self):
        model = HelPropKernelModel(
            kernel=make_kernel(),
            learned=("D0", "m"),
            fixed={"A": 1.0, "Z": 1.0, "R0": 1.0},
            ranges={"D0": (0.1, 50.0), "m": (-2.0, 2.0)},
            etoa_grid=(0.2, 1.0, 3.0),
            elis_grid=(0.2, 1.0, 3.0, 5.0),
        )

        with self.assertRaisesRegex(ValueError, "fixed parameters"):
            model.theta_from_options({"D0": 5.0, "m": 0.0, "A": 1.0})

    def test_learned_parameters_are_required_and_unknowns_rejected(self):
        model = HelPropKernelModel(kernel=make_kernel(), learned=("D0", "m"))

        with self.assertRaisesRegex(ValueError, "missing learned"):
            model.theta_from_options({"D0": 5.0})

        with self.assertRaisesRegex(ValueError, "unknown surrogate"):
            model.theta_from_options({"D0": 5.0, "m": 0.0, "angle": 15.0})

    def test_matrix_uses_stored_grids_and_learned_options(self):
        model = HelPropKernelModel(
            kernel=make_kernel(),
            learned=("D0", "m"),
            fixed={"A": 1.0},
            etoa_grid=(0.2, 1.0, 3.0),
            elis_grid=(0.2, 1.0, 3.0, 5.0),
        )

        matrix = model.matrix({"D0": 5.0, "m": 0.0})
        self.assertEqual(matrix.shape, (3, 4))
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0)

    def test_model_roundtrip(self):
        model = HelPropKernelModel(kernel=make_kernel(), learned=("D0", "m"))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "kernel.pkl")
            model.save(path)
            loaded = load_model(path)

        self.assertEqual(loaded.learned, ("D0", "m"))
        self.assertEqual(tuple(loaded.kernel.param_names), ("D0", "m"))

    def test_runtime_accepts_helprop_style_learned_flags_only(self):
        model = HelPropKernelModel(
            kernel=make_kernel(),
            learned=("D0", "m"),
            fixed={"A": 1.0},
        )

        options = _runtime_options_from_args(model, [], ["--D0", "5", "--m=0.1"])
        self.assertEqual(options, {"D0": 5.0, "m": 0.1})

        with self.assertRaisesRegex(ValueError, "fixed parameters"):
            _runtime_options_from_args(model, [], ["--D0", "5", "--m", "0", "--A", "1"])

    def test_predict_parser_does_not_abbreviate_learned_m_as_matrix_out(self):
        from helprop_surrogate.predict_kernel import build_parser

        parser = build_parser()
        args, unknown = parser.parse_known_args(
            ["kernel.pkl", "--D0", "5", "--m", "0", "--spectrum-out", "spectrum.txt"]
        )

        self.assertEqual(args.model.name, "kernel.pkl")
        self.assertEqual(args.spectrum_out.name, "spectrum.txt")
        self.assertIsNone(args.matrix_out)
        self.assertEqual(unknown, ["--D0", "5", "--m", "0"])

    def test_predict_spectrum_can_write_requested_output_etoa_grid(self):
        model = HelPropKernelModel(
            kernel=make_kernel(),
            learned=("D0", "m"),
            fixed={"A": 1.0},
            etoa_grid=tuple(np.geomspace(0.2, 5.0, 7)),
            elis_grid=tuple(np.geomspace(0.15, 8.0, 12)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "kernel.pkl")
            lis_path = os.path.join(tmpdir, "lis.txt")
            default_spectrum_path = os.path.join(tmpdir, "default_spectrum.txt")
            requested_spectrum_path = os.path.join(tmpdir, "requested_spectrum.txt")
            model.save(model_path)
            lis_energy = np.geomspace(0.1, 10.0, 41)
            np.savetxt(lis_path, np.column_stack([lis_energy, lis_energy ** -2.7]))

            predict_kernel_main(
                [
                    model_path,
                    "--D0",
                    "5",
                    "--m",
                    "0",
                    "--lis",
                    lis_path,
                    "--spectrum-out",
                    default_spectrum_path,
                ]
            )
            predict_kernel_main(
                [
                    model_path,
                    "--D0",
                    "5",
                    "--m",
                    "0",
                    "--lis",
                    lis_path,
                    "--spectrum-out",
                    requested_spectrum_path,
                    "--spectrum-etoa",
                    "0.25,4.0,11",
                ]
            )

            default_output = np.loadtxt(default_spectrum_path)
            requested_output = np.loadtxt(requested_spectrum_path)

        self.assertEqual(default_output.shape, (7, 2))
        self.assertEqual(requested_output.shape, (11, 2))
        np.testing.assert_allclose(requested_output[:, 0], np.geomspace(0.25, 4.0, 11))
        self.assertTrue(np.all(requested_output[:, 1] > 0.0))

    def test_predict_spectrum_rejects_output_etoa_outside_trained_range(self):
        model = HelPropKernelModel(
            kernel=make_kernel(),
            learned=("D0", "m"),
            fixed={"A": 1.0},
            etoa_grid=tuple(np.geomspace(0.2, 5.0, 7)),
            elis_grid=tuple(np.geomspace(0.15, 8.0, 12)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "kernel.pkl")
            lis_path = os.path.join(tmpdir, "lis.txt")
            spectrum_path = os.path.join(tmpdir, "spectrum.txt")
            model.save(model_path)
            lis_energy = np.geomspace(0.1, 10.0, 41)
            np.savetxt(lis_path, np.column_stack([lis_energy, lis_energy ** -2.7]))

            with self.assertRaisesRegex(ValueError, "trained ETOA range"):
                predict_kernel_main(
                    [
                        model_path,
                        "--D0",
                        "5",
                        "--m",
                        "0",
                        "--lis",
                        lis_path,
                        "--spectrum-out",
                        spectrum_path,
                        "--spectrum-etoa",
                        "0.1,6.0,11",
                    ]
                )

    def test_parse_helpers_preserve_names(self):
        fixed = parse_key_value_options(["hcs-osc-amp=0", "Z=1"])
        ranges = parse_range_options(["hcs-osc-phase:0:360"])

        self.assertEqual(fixed, {"hcs-osc-amp": 0.0, "Z": 1.0})
        self.assertEqual(ranges, {"hcs-osc-phase": (0.0, 360.0)})


if __name__ == "__main__":
    unittest.main()
