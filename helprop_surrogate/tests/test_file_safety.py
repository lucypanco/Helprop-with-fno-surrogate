import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from helprop_surrogate.data import TransitionDataset
from helprop_surrogate.file_safety import (
    atomic_promote,
    atomic_replace_text,
    ensure_unique_outputs,
    prepare_output_path,
    temp_output_path,
)
from helprop_surrogate.generate_samples import make_sample_runs, preflight_outputs, write_manifest
from helprop_surrogate.kernel import ConditionalKernelSurrogate
from helprop_surrogate.model import HelPropKernelModel


class FileSafetyTest(unittest.TestCase):
    def test_prepare_output_path_allows_existing_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.txt"
            path.write_text("old")

            self.assertEqual(prepare_output_path(path), path)

    def test_duplicate_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "same.txt"
            with self.assertRaisesRegex(ValueError, "duplicate output path"):
                ensure_unique_outputs([path, path])

    def test_atomic_replace_text_replaces_existing_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            atomic_replace_text(path, "new")
            self.assertEqual(path.read_text(), "new")

            atomic_replace_text(path, "again")
            self.assertEqual(path.read_text(), "again")

    def test_temp_output_promotes_to_final(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            final = Path(tmpdir) / "sample.bson"
            final.write_text("old")
            temp = temp_output_path(final)
            temp.write_text("new")
            atomic_promote(temp, final)

            self.assertEqual(final.read_text(), "new")
            self.assertFalse(temp.exists())

    def test_generate_preflight_allows_existing_bson_but_rejects_duplicate_final_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runs = make_sample_runs(
                helprop="HelProp",
                outdir=tmpdir,
                design=[{"D0": 1.0, "m": 0.0}],
                etoa="1,2,3",
                elis="1,2,3",
                number=10,
                nthread=1,
            )
            runs[0].output.write_text("old")
            preflight_outputs(runs, Path(tmpdir) / "manifest.csv")

            with self.assertRaisesRegex(ValueError, "duplicate output path"):
                preflight_outputs([runs[0], runs[0]], Path(tmpdir) / "manifest.csv")

    def test_manifest_model_and_dataset_replace_existing_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runs = make_sample_runs(
                helprop="HelProp",
                outdir=tmpdir,
                design=[{"D0": 1.0, "m": 0.0}],
                etoa="1,2,3",
                elis="1,2,3",
                number=10,
                nthread=1,
            )
            manifest = Path(tmpdir) / "manifest.csv"
            write_manifest(manifest, runs)
            write_manifest(manifest, runs)
            self.assertIn("samples_0000.bson", manifest.read_text())

            etoa = np.repeat([1.0, 2.0], 4)
            kernel = ConditionalKernelSurrogate().fit(
                etoa,
                etoa * 1.2,
                {"D0": np.ones_like(etoa), "m": np.zeros_like(etoa)},
            )
            model = HelPropKernelModel(kernel=kernel, learned=("D0", "m"))
            model_path = Path(tmpdir) / "model.pkl"
            model.save(model_path)
            model.save(model_path)

            dataset = TransitionDataset(
                etoa=np.array([1.0]),
                elis=np.array([1.2]),
                params={"D0": np.array([3.0]), "m": np.array([0.0])},
            )
            data_path = os.path.join(tmpdir, "data.npz")
            dataset.save_npz(data_path)
            dataset.save_npz(data_path)


if __name__ == "__main__":
    unittest.main()
