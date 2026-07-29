import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from helprop_surrogate.data import TransitionDataset
from helprop_surrogate.generate_samples import (
    build_helprop_command,
    consolidate_bson_outputs,
    fixed_to_helprop_options,
    latin_hypercube,
    make_sample_runs,
    write_manifest,
)


class GenerateSamplesTest(unittest.TestCase):
    def test_latin_hypercube_is_reproducible_and_in_range(self):
        ranges = {"D0": (0.1, 50.0), "m": (-2.0, 2.0)}
        first = latin_hypercube(ranges, n_runs=12, seed=7)
        second = latin_hypercube(ranges, n_runs=12, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        for row in first:
            self.assertGreaterEqual(row["D0"], 0.1)
            self.assertLessEqual(row["D0"], 50.0)
            self.assertGreaterEqual(row["m"], -2.0)
            self.assertLessEqual(row["m"], 2.0)

    def test_build_helprop_command_uses_bson_sample_matrix_mode(self):
        cmd = build_helprop_command(
            helprop="./HelProp",
            output="run.bson",
            params={"D0": 5.0, "m": 0.25},
            etoa="0.1,10,8",
            elis="0.1,20,9",
            number=50,
            nthread=2,
            seed=123,
            fixed_options=["--A=1", "--Z=1"],
        )

        self.assertEqual(cmd[0], "./HelProp")
        self.assertIn("--iotype=BSON", cmd)
        self.assertIn("--sample", cmd)
        self.assertIn("--etoa=0.1,10,8", cmd)
        self.assertIn("--elis=0.1,20,9", cmd)
        self.assertIn("--number=50", cmd)
        self.assertIn("--nthread=2", cmd)
        self.assertIn("--seed=123", cmd)
        self.assertIn("--D0=5", cmd)
        self.assertIn("--m=0.25", cmd)
        self.assertIn("--A=1", cmd)
        self.assertIn("--Z=1", cmd)
        self.assertEqual(cmd[-1], "run.bson")

    def test_fixed_to_helprop_options_preserves_names(self):
        options = fixed_to_helprop_options({"A": 1.0, "hcs-osc-amp": 0.0, "hcs-omega": 1.0})

        self.assertEqual(options, ["--A=1", "--hcs-osc-amp=0", "--hcs-omega=1"])

    def test_make_sample_runs_assigns_stable_outputs_and_seeds(self):
        design = [{"D0": 1.0, "m": -0.1}, {"D0": 2.0, "m": 0.2}]
        runs = make_sample_runs(
            helprop="HelProp.exe",
            outdir="out",
            design=design,
            etoa="1,2,3",
            elis="1,3,4",
            number=10,
            nthread=1,
            seed=5,
        )

        self.assertEqual([run.index for run in runs], [0, 1])
        self.assertEqual(runs[0].output, Path("out") / "samples_0000.bson")
        self.assertEqual(runs[1].output, Path("out") / "samples_0001.bson")
        self.assertIn("--seed=5", runs[0].command)
        self.assertIn("--seed=1000008", runs[1].command)

    def test_write_manifest_records_commands(self):
        design = [{"D0": 1.0, "m": 0.0}]
        runs = make_sample_runs(
            helprop="HelProp",
            outdir="out",
            design=design,
            etoa="1,2,3",
            elis="1,2,3",
            number=10,
            nthread=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.csv"
            write_manifest(manifest, runs)
            text = manifest.read_text()

        self.assertIn("index,output,D0,m,command", text)
        self.assertIn("samples_0000.bson", text)
        self.assertIn("--iotype=BSON", text)

    def test_consolidate_bson_outputs_concatenates_loaded_datasets(self):
        first = TransitionDataset(
            etoa=np.array([1.0, 2.0]),
            elis=np.array([1.2, 2.4]),
            params={"D0": np.array([3.0, 3.0]), "m": np.array([0.1, 0.1])},
        )
        second = TransitionDataset(
            etoa=np.array([3.0]),
            elis=np.array([3.3]),
            params={"D0": np.array([4.0]), "m": np.array([-0.2])},
        )

        with mock.patch(
            "helprop_surrogate.generate_samples.load_bson_transitions",
            side_effect=[first, second],
        ):
            dataset = consolidate_bson_outputs(
                ["first.bson", "second.bson"],
                param_names=("D0", "m"),
            )

        np.testing.assert_allclose(dataset.etoa, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(dataset.elis, [1.2, 2.4, 3.3])
        np.testing.assert_allclose(dataset.params["D0"], [3.0, 3.0, 4.0])
        np.testing.assert_allclose(dataset.params["m"], [0.1, 0.1, -0.2])


if __name__ == "__main__":
    unittest.main()
