"""Train, validate, and test a HelProp FNO transfer-matrix surrogate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from ..file_safety import atomic_replace_text, prepare_output_path, temp_output_path, atomic_promote
from .model import TorchFNOTransferMatrixSurrogate, matrix_rmse
from ..matrix_data import (
    load_npz_matrices,
    next_serial_run_dir,
    read_index_file,
    split_indices,
    write_split_files,
)
from ..model import HelPropKernelModel, parse_key_value_options, parse_range_options
from ..validate import row_kl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a HelProp FNO matrix surrogate.")
    parser.add_argument("--dataset", type=Path, required=True, help="Matrix dataset .npz")
    parser.add_argument("--outdir", type=Path, default=None, help="Run directory; allocated under --runs-root if omitted")
    parser.add_argument("--runs-root", type=Path, default=Path("fno_runs"))
    parser.add_argument("--model-out", type=Path, default=None, help="Saved HelPropKernelModel path")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--modes", type=int, default=10, help="Use same Fourier modes in both dimensions")
    parser.add_argument("--modes-etoa", type=int, default=None)
    parser.add_argument("--modes-elis", type=int, default=None)
    parser.add_argument("--projection-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--train-indices", type=Path, default=None)
    parser.add_argument("--val-indices", type=Path, default=None)
    parser.add_argument("--test-indices", type=Path, default=None)
    parser.add_argument("--fixed", action="append", default=[], help="Fixed HelProp parameter name=value")
    parser.add_argument("--range", dest="ranges", action="append", default=[], help="Training range name:min:max")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--verbose-train", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.outdir or _infer_run_dir(args.dataset) or next_serial_run_dir(args.runs_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    checkpoints_dir = run_dir / "checkpoints"
    validation_dir = run_dir / "validation"
    testing_dir = run_dir / "testing"
    data_dir = run_dir / "data"
    for directory in (logs_dir, checkpoints_dir, validation_dir, testing_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)

    dataset = load_npz_matrices(args.dataset)
    train_idx, val_idx, test_idx = _resolve_splits(
        args,
        dataset.theta.shape[0],
        data_dir,
        args.dataset.parent,
    )

    modes_etoa = args.modes_etoa if args.modes_etoa is not None else args.modes
    modes_elis = args.modes_elis if args.modes_elis is not None else args.modes
    kernel = TorchFNOTransferMatrixSurrogate(
        param_names=dataset.param_names,
        width=args.width,
        modes_etoa=modes_etoa,
        modes_elis=modes_elis,
        n_layers=args.layers,
        projection_size=args.projection_size,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        random_state=args.seed,
        device=args.device,
        verbose=args.verbose_train,
    ).fit(
        dataset.theta[train_idx],
        dataset.matrices[train_idx],
        dataset.etoa_grid,
        dataset.elis_grid,
        val_theta=dataset.theta[val_idx] if val_idx.size else None,
        val_matrices=dataset.matrices[val_idx] if val_idx.size else None,
        checkpoint_dir=checkpoints_dir,
        checkpoint_every=args.checkpoint_every,
    )

    fixed = parse_key_value_options(args.fixed)
    ranges = parse_range_options(args.ranges)
    unknown_ranges = set(ranges).difference(dataset.param_names)
    if unknown_ranges:
        names = ", ".join(sorted(unknown_ranges))
        raise ValueError(f"ranges supplied for non-learned parameters: {names}")

    model = HelPropKernelModel(
        kernel=kernel,
        learned=dataset.param_names,
        fixed=fixed,
        ranges=ranges,
        etoa_grid=tuple(float(value) for value in dataset.etoa_grid),
        elis_grid=tuple(float(value) for value in dataset.elis_grid),
    )
    model_out = args.model_out or (run_dir / "kernel_fno.pkl")
    model.save(model_out)

    _write_loss_history(logs_dir / "loss_history.csv", kernel.history_)
    _write_config(
        run_dir / "config.json",
        args,
        dataset,
        model_out,
        train_idx,
        val_idx,
        test_idx,
        kernel,
    )

    if val_idx.size:
        _write_prediction_outputs(
            directory=validation_dir,
            prefix="val",
            theta=dataset.theta[val_idx],
            true_matrices=dataset.matrices[val_idx],
            pred_matrices=kernel.predict_matrices(dataset.theta[val_idx]),
            param_names=dataset.param_names,
        )

    if test_idx.size:
        test_pred = kernel.predict_matrices(dataset.theta[test_idx])
        test_metrics = {
            "rmse": matrix_rmse(dataset.matrices[test_idx], test_pred),
            "n_samples": int(test_idx.size),
        }
        atomic_replace_text(
            testing_dir / "test_metrics.json",
            json.dumps(test_metrics, indent=2, sort_keys=True),
        )
        print(f"test_rmse: {test_metrics['rmse']:.8g}")

    print(f"run_dir: {run_dir}")
    print(f"model: {model_out}")
    print(f"best_epoch: {kernel.best_epoch_}")
    return 0


def _infer_run_dir(dataset_path: Path) -> Path | None:
    """Use ``fno_runs/run_xxxx`` when dataset lives in that run's data dir."""
    parent = dataset_path.resolve().parent
    if parent.name == "data" and parent.parent.name.startswith("run_"):
        return parent.parent
    return None


def _resolve_splits(args, n_samples: int, data_dir: Path, dataset_dir: Path):
    if args.train_indices is not None or args.val_indices is not None or args.test_indices is not None:
        if args.train_indices is None or args.val_indices is None or args.test_indices is None:
            raise ValueError("train, validation, and test index files must be supplied together")
        train_idx = read_index_file(args.train_indices)
        val_idx = read_index_file(args.val_indices)
        test_idx = read_index_file(args.test_indices)
    elif _has_split_files(dataset_dir):
        train_idx = read_index_file(dataset_dir / "train_indices.txt")
        val_idx = read_index_file(dataset_dir / "val_indices.txt")
        test_idx = read_index_file(dataset_dir / "test_indices.txt")
    elif _has_split_files(data_dir):
        train_idx = read_index_file(data_dir / "train_indices.txt")
        val_idx = read_index_file(data_dir / "val_indices.txt")
        test_idx = read_index_file(data_dir / "test_indices.txt")
    else:
        train_idx, val_idx, test_idx = split_indices(
            n_samples,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
    _check_split_indices(n_samples, train_idx, val_idx, test_idx)
    write_split_files(data_dir, train_idx, val_idx, test_idx)
    return train_idx, val_idx, test_idx


def _has_split_files(directory: Path) -> bool:
    return (
        (directory / "train_indices.txt").exists()
        and (directory / "val_indices.txt").exists()
        and (directory / "test_indices.txt").exists()
    )


def _check_split_indices(n_samples: int, *groups: np.ndarray) -> None:
    seen = set()
    for group in groups:
        for index in np.asarray(group, dtype=int):
            if index < 0 or index >= n_samples:
                raise ValueError(f"split index {index} is out of range")
            if int(index) in seen:
                raise ValueError(f"split index {index} appears in more than one split")
            seen.add(int(index))


def _write_loss_history(path: Path, rows: Sequence[dict[str, float]]) -> None:
    path = prepare_output_path(path)
    tmp = temp_output_path(path)
    try:
        with tmp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "train_rmse",
                    "val_rmse",
                    "learning_rate",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        atomic_promote(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _write_prediction_outputs(
    directory: Path,
    prefix: str,
    theta: np.ndarray,
    true_matrices: np.ndarray,
    pred_matrices: np.ndarray,
    param_names: Sequence[str],
) -> None:
    residuals = pred_matrices - true_matrices
    relative = residuals / np.maximum(np.abs(true_matrices), 1.0e-12)
    predictions_path = directory / f"{prefix}_predictions.npz"
    residuals_path = directory / f"{prefix}_residuals.npz"
    _atomic_save_npz(
        predictions_path,
        theta=theta,
        param_names=np.asarray(param_names),
        M_true=true_matrices,
        M_pred=pred_matrices,
    )
    _atomic_save_npz(
        residuals_path,
        theta=theta,
        param_names=np.asarray(param_names),
        residual=residuals,
        relative_residual=relative,
        row_rmse=np.sqrt(np.mean(residuals * residuals, axis=2)),
        matrix_rmse=np.sqrt(np.mean(residuals * residuals, axis=(1, 2))),
    )
    _write_metrics_csv(directory / f"{prefix}_metrics.csv", theta, true_matrices, pred_matrices, param_names)


def _write_metrics_csv(
    path: Path,
    theta: np.ndarray,
    true_matrices: np.ndarray,
    pred_matrices: np.ndarray,
    param_names: Sequence[str],
) -> None:
    path = prepare_output_path(path)
    tmp = temp_output_path(path)
    kl = np.asarray([row_kl(true, pred) for true, pred in zip(true_matrices, pred_matrices)])
    sample_rmse = np.sqrt(np.mean((pred_matrices - true_matrices) ** 2, axis=(1, 2)))
    row_sum_error = np.max(np.abs(pred_matrices.sum(axis=2) - 1.0), axis=1)
    try:
        with tmp.open("w", newline="", encoding="utf-8") as stream:
            fieldnames = [*param_names, "matrix_rmse", "mean_row_kl", "max_row_kl", "max_row_sum_error"]
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for irow in range(theta.shape[0]):
                finite_kl = kl[irow][np.isfinite(kl[irow])]
                row = {name: float(theta[irow, index]) for index, name in enumerate(param_names)}
                row.update(
                    {
                        "matrix_rmse": float(sample_rmse[irow]),
                        "mean_row_kl": float(np.mean(finite_kl)) if finite_kl.size else float("inf"),
                        "max_row_kl": float(np.max(finite_kl)) if finite_kl.size else float("inf"),
                        "max_row_sum_error": float(row_sum_error[irow]),
                    }
                )
                writer.writerow(row)
        atomic_promote(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _write_config(
    path: Path,
    args,
    dataset,
    model_out: Path,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    kernel: TorchFNOTransferMatrixSurrogate,
) -> None:
    config = {
        "dataset": str(args.dataset),
        "model_out": str(model_out),
        "param_names": list(dataset.param_names),
        "n_samples": int(dataset.theta.shape[0]),
        "etoa_grid": [float(value) for value in dataset.etoa_grid],
        "elis_grid": [float(value) for value in dataset.elis_grid],
        "train_size": int(train_idx.size),
        "val_size": int(val_idx.size),
        "test_size": int(test_idx.size),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "width": int(args.width),
        "layers": int(args.layers),
        "modes_etoa": int(kernel.modes_etoa),
        "modes_elis": int(kernel.modes_elis),
        "projection_size": int(args.projection_size),
        "dropout": float(args.dropout),
        "optimizer": getattr(kernel, "optimizer_name_", "AdamW"),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "best_epoch": int(kernel.best_epoch_),
        "best_loss": float(kernel.best_loss_),
    }
    atomic_replace_text(path, json.dumps(config, indent=2, sort_keys=True))


def _atomic_save_npz(path: Path, **payload) -> None:
    path = prepare_output_path(path)
    tmp = temp_output_path(path)
    try:
        with tmp.open("wb") as stream:
            np.savez(stream, **payload)
        atomic_promote(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
