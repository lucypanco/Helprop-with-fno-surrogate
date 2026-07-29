import os
import tempfile
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in os.sys.path:
    os.sys.path.insert(0, ROOT)

from helprop_mcmc.parameter_effect_scan import (
    SpectrumScanRun,
    build_hcs_osc_values,
    build_helprop_spectrum_command,
    collect_flux_at_energy,
    default_scan_values,
    main,
    make_scan_runs,
)


class ParameterEffectScanTest(unittest.TestCase):
    def test_default_scan_is_hcs_osc_only_with_two_seeds_and_bins(self):
        hcs_osc_values, seed_values, bin_values = default_scan_values()

        self.assertEqual(hcs_osc_values[0], -50.0)
        self.assertEqual(hcs_osc_values[-1], 50.0)
        self.assertEqual(len(hcs_osc_values), 51)
        self.assertIn(0.0, hcs_osc_values)
        self.assertEqual(len(seed_values), 2)
        self.assertEqual(bin_values, (150, 300))
        self.assertEqual(len(hcs_osc_values) * len(seed_values) * len(bin_values), 204)

    def test_build_hcs_osc_values_is_inclusive_and_forces_zero(self):
        self.assertEqual(build_hcs_osc_values(-2, 2, 1), (-2.0, -1.0, 0.0, 1.0, 2.0))
        self.assertEqual(build_hcs_osc_values(1, 3, 1), (0.0, 1.0, 2.0, 3.0))

    def test_command_contains_seed_etoa_bins_hcs_osc_and_fixed_m(self):
        command = build_helprop_spectrum_command(
            "HelProp",
            "lis.txt",
            "out.txt",
            ["--iotype=TXT", "--nthread=2", "--number=10"],
            seed=123,
            etoa_bins=500,
            etoa_min=0.1,
            etoa_max=100000.0,
            hcs_osc_amp=-50.0,
        )

        self.assertEqual(command[0], "HelProp")
        self.assertIn("--seed=123", command)
        self.assertIn("--number=10", command)
        self.assertIn("--etoa=0.1,100000,500", command)
        self.assertIn("--m=0", command)
        self.assertIn("--hcs-osc-amp=-50", command)
        self.assertIn("--hcs-osc-phase=0", command)
        self.assertEqual(command[-2:], ["lis.txt", "out.txt"])

    def test_make_scan_runs_uses_seed_bin_hcs_osc_output_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runs = make_scan_runs(
                helprop="HelProp",
                lis_input="lis.txt",
                outdir=tmpdir,
                fixed_options=[],
                hcs_osc_values=(-1.0, 0.0, 1.0),
                seed_values=(11, 22),
                bin_values=(100,),
            )

            self.assertEqual(len(runs), 6)
            self.assertTrue(
                str(runs[0].output).endswith(os.path.join("spectra", "seed_11", "bins_100", "hcs_osc_-1.txt"))
            )
            self.assertTrue(
                str(runs[-1].output).endswith(os.path.join("spectra", "seed_22", "bins_100", "hcs_osc_1.txt"))
            )

    def test_collect_flux_at_energy_uses_hcs_osc_zero_baseline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = os.path.join(tmpdir, "hcs_osc_0.txt")
            shifted_path = os.path.join(tmpdir, "hcs_osc_2.txt")
            np.savetxt(
                baseline_path,
                np.column_stack([[1.0, np.sqrt(10.0), 10.0], [10.0, 10.0 * np.sqrt(10.0), 100.0]]),
            )
            np.savetxt(
                shifted_path,
                np.column_stack([[1.0, np.sqrt(10.0), 10.0], [20.0, 20.0 * np.sqrt(10.0), 200.0]]),
            )
            runs = [
                SpectrumScanRun(0, 123, 200, 0.0, baseline_path, []),
                SpectrumScanRun(1, 123, 200, 2.0, shifted_path, []),
            ]

            records = collect_flux_at_energy(runs, 10.0)

        self.assertEqual(len(records), 2)
        self.assertAlmostEqual(records[0].flux, 100.0)
        self.assertAlmostEqual(records[0].percent, 0.0)
        self.assertAlmostEqual(records[1].percent, 100.0)

    def test_dry_run_writes_manifest_without_executing_helprop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = main(
                [
                    "missing_lis.txt",
                    "--helprop",
                    "missing_helprop",
                    "--outdir",
                    tmpdir,
                    "--hcs-osc-min=0",
                    "--hcs-osc-max=0",
                    "--seed-values=1",
                    "--bin-values=2",
                    "--number=3",
                    "--dry-run",
                ]
            )
            manifest = os.path.join(tmpdir, "manifest.csv")

            self.assertEqual(result, 0)
            self.assertTrue(os.path.exists(manifest))
            with open(manifest, encoding="utf-8") as stream:
                content = stream.read()
            self.assertIn("missing_lis.txt", content)
            self.assertIn("--seed=1", content)
            self.assertIn("--number=3", content)
            self.assertIn("--etoa=0.1,100000,2", content)
            self.assertIn("--m=0", content)
            self.assertIn("--hcs-osc-amp=0", content)
            self.assertIn("--hcs-omega=1", content)


if __name__ == "__main__":
    unittest.main()
