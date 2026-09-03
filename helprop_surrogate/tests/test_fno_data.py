import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from helprop_surrogate.matrix_data import (
    MatrixDataset,
    build_helprop_matrix_command,
    load_npz_matrices,
    make_matrix_runs,
    mixed_parameter_design,
    next_serial_run_dir,
    parse_choice_options,
    run_helprop_matrices,
    split_indices,
    write_split_files,
    main,
)


def make_matrix_dataset(n_samples=8):
    theta = np.column_stack(
        [
            np.linspace(1.0, 8.0, n_samples),
            np.linspace(-0.5, 0.5, n_samples),
        ]
    )
    etoa = np.geomspace(0.2, 2.0, 4)
    elis = np.geomspace(0.2, 4.0, 5)
    matrices = []
    for d0, m in theta:
        rows = []
        for energy in etoa:
            center = np.log(energy * (1.2 + 0.01 * d0 - 0.02 * m))
            weights = np.exp(-0.5 * ((np.log(elis) - center) / 0.45) ** 2)
            rows.append(weights / weights.sum())
        matrices.append(rows)
    return MatrixDataset(
        theta=theta,
        matrices=np.asarray(matrices),
        param_names=("D0", "m"),
        etoa_grid=etoa,
        elis_grid=elis,
    )


class FNODataTest(unittest.TestCase):
    def test_matrix_dataset_npz_roundtrip_and_row_normalization(self):
        dataset = make_matrix_dataset()
        scaled = MatrixDataset(
            theta=dataset.theta,
            matrices=dataset.matrices * 3.0,
            param_names=dataset.param_names,
            etoa_grid=dataset.etoa_grid,
            elis_grid=dataset.elis_grid,
        )

        np.testing.assert_allclose(scaled.matrices.sum(axis=2), 1.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "matrices.npz"
            scaled.save_npz(path)
            loaded = load_npz_matrices(path)

        np.testing.assert_allclose(loaded.theta, scaled.theta)
        np.testing.assert_allclose(loaded.matrices, scaled.matrices)
        self.assertEqual(loaded.param_names, ("D0", "m"))

    def test_split_indices_are_disjoint_and_written(self):
        train, val, test = split_indices(20, train_fraction=0.7, val_fraction=0.15, seed=4)
        all_indices = np.concatenate([train, val, test])

        self.assertEqual(train.size, 14)
        self.assertEqual(val.size, 3)
        self.assertEqual(test.size, 3)
        self.assertEqual(np.unique(all_indices).size, 20)

        with tempfile.TemporaryDirectory() as tmpdir:
            write_split_files(tmpdir, train, val, test)
            self.assertTrue((Path(tmpdir) / "train_indices.txt").exists())
            self.assertTrue((Path(tmpdir) / "val_indices.txt").exists())
            self.assertTrue((Path(tmpdir) / "test_indices.txt").exists())

    def test_zero_validation_fraction_disables_validation_split(self):
        train, val, test = split_indices(20, train_fraction=0.9, val_fraction=0.0, seed=4)

        self.assertEqual(train.size, 18)
        self.assertEqual(val.size, 0)
        self.assertEqual(test.size, 2)
        self.assertEqual(np.unique(np.concatenate([train, test])).size, 20)

    def test_matrix_command_uses_bson_matrix_mode(self):
        cmd = build_helprop_matrix_command(
            helprop="./HelProp",
            output="matrix.bson",
            params={"D0": 5.0, "m": 0.0, "A": 1.6},
            etoa="0.1,10,8",
            elis="0.1,20,9",
            number=50,
            nthread=2,
            seed=123,
            fixed_options=["--Z=1"],
            sample=False,
            integer_params=["A"],
        )

        self.assertIn("--iotype=BSON", cmd)
        self.assertNotIn("--sample", cmd)
        self.assertIn("--D0=5", cmd)
        self.assertIn("--m=0", cmd)
        self.assertIn("--A=2", cmd)
        self.assertIn("--Z=1", cmd)
        self.assertEqual(cmd[-1], "matrix.bson")

    def test_make_matrix_runs_and_serial_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = next_serial_run_dir(Path(tmpdir) / "fno_runs")
            second = next_serial_run_dir(Path(tmpdir) / "fno_runs")
            self.assertEqual(first.name, "run_0001")
            self.assertEqual(second.name, "run_0002")

            runs = make_matrix_runs(
                helprop="HelProp",
                outdir=first / "data",
                design=[{"D0": 1.0, "m": 0.0}],
                etoa="1,2,3",
                elis="1,2,3",
                number=10,
                nthread=1,
                seed=7,
            )
            self.assertEqual(runs[0].output, first / "data" / "matrix_0000.bson")
            self.assertIn("--seed=7", runs[0].command)

    def test_mixed_design_supports_categorical_choices(self):
        choices = parse_choice_options(["polarity:-1,1", "A:1,2"])
        design = mixed_parameter_design(
            learned=("D0", "polarity", "A"),
            ranges={"D0": (1.0, 10.0)},
            choices=choices,
            n_runs=8,
            seed=9,
        )

        self.assertEqual(len(design), 8)
        self.assertTrue({row["polarity"] for row in design}.issubset({-1.0, 1.0}))
        self.assertTrue({row["A"] for row in design}.issubset({1.0, 2.0}))
        for row in design:
            self.assertGreaterEqual(row["D0"], 1.0)
            self.assertLessEqual(row["D0"], 10.0)

    def test_parallel_runner_rejects_invalid_jobs_before_execution(self):
        with self.assertRaisesRegex(ValueError, "jobs"):
            run_helprop_matrices([], jobs=0, dry_run=True)

    def test_continue_uses_saved_config_and_skips_completed_matrix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "fno_runs" / "run_0001"
            data_dir = run_dir / "data"
            data_dir.mkdir(parents=True)
            design = [{"D0": 1.0, "m": 0.0}, {"D0": 2.0, "m": 0.1}]
            runs = make_matrix_runs(
                helprop="HelProp",
                outdir=data_dir,
                design=design,
                etoa="1,2,3",
                elis="1,2,3",
                number=10,
                nthread=1,
            )
            runs[0].output.write_bytes(b"completed")
            config = {
                "helprop": "HelProp",
                "run_dir": str(run_dir),
                "learned": ["D0", "m"],
                "fixed": {},
                "ranges": {"D0": [1.0, 2.0], "m": [0.0, 0.1]},
                "choices": {},
                "etoa": "1,2,3",
                "elis": "1,2,3",
                "number": 10,
                "nthread": 1,
                "jobs": 1,
                "n_runs": 2,
                "seed": 123,
                "train_fraction": 0.7,
                "val_fraction": 0.15,
                "sample": False,
                "integer_params": [],
                "timeout": None,
            }
            (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
            old_cwd = Path.cwd()
            os.chdir(root)
            try:
                dataset = make_matrix_dataset(2)
                with mock.patch(
                    "helprop_surrogate.matrix_data.run_helprop_matrices"
                ) as runner, mock.patch(
                    "helprop_surrogate.matrix_data.consolidate_bson_matrices",
                    return_value=dataset,
                ):
                    result = main(["--continue", str(run_dir)])
            finally:
                os.chdir(old_cwd)

        self.assertEqual(result, 0)
        pending = runner.call_args.args[0]
        self.assertEqual([run.index for run in pending], [1])
        self.assertEqual(runner.call_args.kwargs["jobs"], 1)
        self.assertFalse((run_dir / "continue.json").exists())


if __name__ == "__main__":
    unittest.main()
