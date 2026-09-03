"""Train, validate, and test a HelProp FNO transfer-matrix surrogate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from ..file_safety import atomic_replace_text, prepare_output_path, temp_output_path, atomic_promote
from .model import (
    DEFAULT_MATRIX_CROSS_ENTROPY_WEIGHT,
    DEFAULT_MATRIX_PROBABILITY_LOSS_WEIGHT,
    DEFAULT_SPECTRUM_LOSS_WEIGHT,
    DEFAULT_SPECTRUM_MAX_ERROR_PERCENT,
    DEFAULT_SPECTRUM_MAX_ERROR_TEMPERATURE_PERCENT,
    DEFAULT_SPECTRUM_HUBER_DELTA_PERCENT,
    DEFAULT_SPECTRUM_TOP_K,
    DEFAULT_BOUNDARY_PADDING,
    DEFAULT_LR_SCHEDULER,
    DEFAULT_LR_SCHEDULER_FACTOR,
    DEFAULT_LR_SCHEDULER_PATIENCE,
    DEFAULT_LR_SCHEDULER_COOLDOWN,
    DEFAULT_LR_SCHEDULER_MIN_LR,
    DEFAULT_EARLY_STOPPING_MIN_DELTA,
    DEFAULT_EARLY_STOPPING_MIN_EPOCHS,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    TorchFNOTransferMatrixSurrogate,
    matrix_rmse,
)
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
    parser.add_argument("--outdir", type=Path, default=None, help="Run directory; allocated under --run-root if omitted")
    parser.add_argument("--run-root", type=Path, default=Path("fno_runs"))
    parser.add_argument("--model-out", type=Path, default=None, help="Saved HelPropKernelModel path")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--modes", type=int, default=10, help="Use same Fourier modes in both dimensions")
    parser.add_argument("--modes-etoa", type=int, default=None)
    parser.add_argument("--modes-elis", type=int, default=None)
    parser.add_argument("--projection-size", type=int, default=64)
    parser.add_argument(
        "--boundary-padding",
        type=int,
        default=DEFAULT_BOUNDARY_PADDING,
        help="Reflect-padding cells on each ETOA and ELIS boundary before Fourier layers",
    )
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--lr-scheduler", choices=("plateau", "none"), default=DEFAULT_LR_SCHEDULER)
    parser.add_argument("--lr-scheduler-factor", type=float, default=DEFAULT_LR_SCHEDULER_FACTOR)
    parser.add_argument("--lr-scheduler-patience", type=int, default=DEFAULT_LR_SCHEDULER_PATIENCE)
    parser.add_argument("--lr-scheduler-cooldown", type=int, default=DEFAULT_LR_SCHEDULER_COOLDOWN)
    parser.add_argument("--lr-scheduler-min-lr", type=float, default=DEFAULT_LR_SCHEDULER_MIN_LR)
    early_group = parser.add_mutually_exclusive_group()
    early_group.add_argument("--early-stopping", dest="early_stopping", action="store_true", default=True)
    early_group.add_argument("--no-early-stopping", dest="early_stopping", action="store_false")
    parser.add_argument("--early-stopping-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=DEFAULT_EARLY_STOPPING_MIN_DELTA,
        help="Minimum improvement; 0.1 means 0.1%% for spectrum-error monitoring",
    )
    parser.add_argument("--early-stopping-min-epochs", type=int, default=DEFAULT_EARLY_STOPPING_MIN_EPOCHS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--train-indices",
        type=str,
        default=None,
        help="Index-file path or train fraction; overrides dataset split files (0 disables)",
    )
    parser.add_argument(
        "--val-indices",
        type=str,
        default=None,
        help="Index-file path or validation fraction; overrides dataset split files (0 disables)",
    )
    parser.add_argument(
        "--test-indices",
        type=str,
        default=None,
        help="Index-file path or test fraction; overrides dataset split files (0 disables)",
    )
    parser.add_argument("--fixed", action="append", default=[], help="Fixed HelProp parameter name=value")
    parser.add_argument("--range", dest="ranges", action="append", default=[], help="Training range name:min:max")
    parser.add_argument(
        "--matrix-cross-entropy-weight",
        type=float,
        default=DEFAULT_MATRIX_CROSS_ENTROPY_WEIGHT,
        help="Weight of matrix cross-entropy; set 0 for the probability-MSE ablation",
    )
    parser.add_argument(
        "--matrix-probability-loss-weight",
        type=float,
        default=DEFAULT_MATRIX_PROBABILITY_LOSS_WEIGHT,
        help="Weight of plain per-bin probability MSE added to matrix cross-entropy",
    )
    parser.add_argument(
        "--lis",
        type=Path,
        default=None,
        help="LIS spectrum file used by folded-spectrum loss and spectrum metrics",
    )
    parser.add_argument(
        "--spectrum-loss-weight",
        type=float,
        default=DEFAULT_SPECTRUM_LOSS_WEIGHT,
        help="Weight multiplying folded-spectrum loss; set 0 for pure matrix cross entropy",
    )
    parser.add_argument(
        "--spectrum-max-error-percent",
        type=float,
        default=DEFAULT_SPECTRUM_MAX_ERROR_PERCENT,
        help="Exact validation/test maximum percentage-error threshold",
    )
    parser.add_argument(
        "--spectrum-max-error-temperature-percent",
        type=float,
        default=DEFAULT_SPECTRUM_MAX_ERROR_TEMPERATURE_PERCENT,
        help="Smooth transition width around the per-bin error threshold",
    )
    parser.add_argument(
        "--spectrum-huber-delta-percent",
        type=float,
        default=DEFAULT_SPECTRUM_HUBER_DELTA_PERCENT,
        help="Independent Huber transition scale for excess relative error",
    )
    parser.add_argument(
        "--spectrum-top-k",
        type=int,
        default=DEFAULT_SPECTRUM_TOP_K,
        help="Number of worst spectra retained in each batch spectrum loss",
    )
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Do not create checkpoints/ or save epoch/best PyTorch checkpoints",
    )
    parser.add_argument(
        "--reserve-checkpoint",
        type=Path,
        default=None,
        help="Single rotating checkpoint path for interruption recovery",
    )
    parser.add_argument(
        "--resume",
        dest="resume_checkpoint",
        type=Path,
        default=None,
        help="Resume model weights from a reserve checkpoint",
    )
    parser.add_argument("--verbose-train", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.outdir or _infer_run_dir(args.dataset) or next_serial_run_dir(args.run_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    checkpoints_dir = run_dir / "checkpoints"
    validation_dir = run_dir / "validation"
    testing_dir = run_dir / "testing"
    data_dir = run_dir / "data"
    directories = (logs_dir, validation_dir, testing_dir, data_dir)
    if not args.no_checkpoints:
        directories += (checkpoints_dir,)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    dataset = load_npz_matrices(args.dataset)
    fixed = parse_key_value_options(args.fixed)
    _check_generation_fixed_parameters(args.dataset, fixed)
    repeated_fixed = set(dataset.param_names).intersection(fixed)
    if repeated_fixed:
        names = ", ".join(sorted(repeated_fixed))
        raise ValueError(f"parameters cannot be both learned and fixed: {names}")
    if args.lis is None and args.spectrum_loss_weight > 0.0:
        raise ValueError("--lis is required when --spectrum-loss-weight > 0")
    loss_lis_flux = _read_lis_on_grid(args.lis, dataset.elis_grid) if args.lis is not None else None

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
        boundary_padding=args.boundary_padding,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        random_state=args.seed,
        device=args.device,
        verbose=args.verbose_train,
        matrix_cross_entropy_weight=args.matrix_cross_entropy_weight,
        matrix_probability_loss_weight=args.matrix_probability_loss_weight,
        spectrum_loss_weight=args.spectrum_loss_weight,
        spectrum_max_error_percent=args.spectrum_max_error_percent,
        spectrum_max_error_temperature_percent=args.spectrum_max_error_temperature_percent,
        spectrum_huber_delta_percent=args.spectrum_huber_delta_percent,
        spectrum_top_k=args.spectrum_top_k,
        lr_scheduler=args.lr_scheduler,
        lr_scheduler_factor=args.lr_scheduler_factor,
        lr_scheduler_patience=args.lr_scheduler_patience,
        lr_scheduler_cooldown=args.lr_scheduler_cooldown,
        lr_scheduler_min_lr=args.lr_scheduler_min_lr,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_min_epochs=args.early_stopping_min_epochs,
        spectrum_A=int(round(fixed.get("A", 1))),
    ).fit(
        dataset.theta[train_idx],
        dataset.matrices[train_idx],
        dataset.etoa_grid,
        dataset.elis_grid,
        val_theta=dataset.theta[val_idx] if val_idx.size else None,
        val_matrices=dataset.matrices[val_idx] if val_idx.size else None,
        spectrum_lis_flux=loss_lis_flux,
        checkpoint_dir=None if args.no_checkpoints else checkpoints_dir,
        checkpoint_every=args.checkpoint_every,
        reserve_checkpoint=args.reserve_checkpoint,
        resume_checkpoint=args.resume_checkpoint,
    )

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
        ranges,
    )

    if val_idx.size:
        _write_prediction_outputs(
            directory=validation_dir,
            prefix="val",
            theta=dataset.theta[val_idx],
            true_matrices=dataset.matrices[val_idx],
            pred_matrices=kernel.predict_matrices(dataset.theta[val_idx]),
            param_names=dataset.param_names,
            etoa_grid=dataset.etoa_grid,
            elis_grid=dataset.elis_grid,
            lis_flux=loss_lis_flux,
            A=int(round(fixed.get("A", 1))),
        )

    if test_idx.size:
        test_pred = kernel.predict_matrices(dataset.theta[test_idx])
        test_metrics = {
            "rmse": matrix_rmse(dataset.matrices[test_idx], test_pred),
            "n_samples": int(test_idx.size),
        }
        if loss_lis_flux is not None:
            test_metrics.update(
                _spectrum_metrics(
                    dataset.matrices[test_idx],
                    test_pred,
                    dataset.etoa_grid,
                    dataset.elis_grid,
                    loss_lis_flux,
                    A=int(round(fixed.get("A", 1))),
                )
            )
        atomic_replace_text(
            testing_dir / "test_metrics.json",
            json.dumps(test_metrics, indent=2, sort_keys=True),
        )
        print(f"test_rmse: {test_metrics['rmse']:.8g}")

    print(f"run_dir: {run_dir}")
    print(f"model: {model_out}")
    print(f"best_epoch: {kernel.best_epoch_}")
    return 0


def _check_generation_fixed_parameters(dataset_path: Path, fixed: dict[str, float]) -> None:
    """Reject a training run whose fixed physics metadata disagrees with generation."""
    path = dataset_path.resolve()
    if path.parent.name != "data":
        return
    generation_config = path.parent.parent / "config.json"
    if not generation_config.exists():
        return
    try:
        payload = json.loads(generation_config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    recorded = payload.get("fixed")
    if not isinstance(recorded, dict):
        return
    mismatches = []
    for name, raw_value in recorded.items():
        if name not in fixed:
            mismatches.append(f"{name}: generated={float(raw_value):g}, training=<missing>")
            continue
        value = fixed[name]
        if float(raw_value) != float(value):
            mismatches.append(
                f"{name}: generated={float(raw_value):g}, training={float(value):g}"
            )
    if mismatches:
        raise ValueError(
            "training fixed parameters disagree with the matrix-generation config "
            f"({generation_config}): " + ", ".join(mismatches)
        )


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
        values = [args.train_indices, args.val_indices, args.test_indices]
        fractions = [_as_fraction(value) for value in values]
        if all(value is not None for value in fractions):
            if not np.isclose(sum(fractions), 1.0, rtol=0.0, atol=1.0e-8):
                raise ValueError("train, validation, and test fractions must sum to 1")
            train_idx, val_idx, test_idx = split_indices(
                n_samples,
                train_fraction=fractions[0],
                val_fraction=fractions[1],
                seed=args.seed,
            )
        elif any(value is not None for value in fractions):
            raise ValueError("all three --*-indices values must be paths or all three must be fractions")
        else:
            train_idx = read_index_file(Path(args.train_indices))
            val_idx = read_index_file(Path(args.val_indices))
            test_idx = read_index_file(Path(args.test_indices))
    elif _has_split_files(dataset_dir):
        train_idx = read_index_file(dataset_dir / "train_indices.txt")
        val_idx = read_index_file(dataset_dir / "val_indices.txt")
        test_idx = read_index_file(dataset_dir / "test_indices.txt")
    elif _has_split_files(data_dir):
        train_idx = read_index_file(data_dir / "train_indices.txt")
        val_idx = read_index_file(data_dir / "val_indices.txt")
        test_idx = read_index_file(data_dir / "test_indices.txt")
    else:
        raise ValueError(
            "training requires train_indices.txt, val_indices.txt, and "
            "test_indices.txt, or all three explicit --*-indices paths"
        )
    _check_split_indices(n_samples, train_idx, val_idx, test_idx)
    write_split_files(data_dir, train_idx, val_idx, test_idx)
    return train_idx, val_idx, test_idx


def _as_fraction(value: str) -> float | None:
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"split fraction must be between 0 and 1, got {value!r}")
    return fraction


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


def _read_lis_on_grid(path: Path, elis_grid: np.ndarray) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("--lis file must contain at least two columns: E flux")
    energy = np.asarray(data[:, 0], dtype=float)
    flux = np.asarray(data[:, 1], dtype=float)
    if np.any(~np.isfinite(energy)) or np.any(~np.isfinite(flux)):
        raise ValueError("--lis contains non-finite values")
    if np.any(energy <= 0.0) or np.any(flux <= 0.0):
        raise ValueError("--lis energies and fluxes must be positive")
    if np.any(np.diff(energy) <= 0.0):
        raise ValueError("--lis energies must be strictly increasing")
    if elis_grid[0] < energy[0] or elis_grid[-1] > energy[-1]:
        raise ValueError(
            "--lis must cover the full ELIS training grid "
            f"[{elis_grid[0]}, {elis_grid[-1]}], got [{energy[0]}, {energy[-1]}]"
        )
    return np.exp(np.interp(np.log(elis_grid), np.log(energy), np.log(flux)))


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
                    "train_matrix_loss",
                    "val_matrix_loss",
                    "train_matrix_cross_entropy_loss",
                    "val_matrix_cross_entropy_loss",
                    "train_matrix_probability_loss",
                    "val_matrix_probability_loss",
                    "train_spectrum_loss",
                    "val_spectrum_loss",
                    "train_max_spectrum_percentage_error",
                    "val_max_spectrum_percentage_error",
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
    etoa_grid: np.ndarray | None = None,
    elis_grid: np.ndarray | None = None,
    lis_flux: np.ndarray | None = None,
    A: int = 1,
) -> None:
    residuals = pred_matrices - true_matrices
    relative = residuals / np.maximum(np.abs(true_matrices), 1.0e-12)
    predictions_path = directory / f"{prefix}_predictions.npz"
    residuals_path = directory / f"{prefix}_residuals.npz"
    spectrum_payload = {}
    if lis_flux is not None and etoa_grid is not None and elis_grid is not None:
        spectrum_true = _fold_lis_batch(true_matrices, etoa_grid, elis_grid, lis_flux, A=A)
        spectrum_pred = _fold_lis_batch(pred_matrices, etoa_grid, elis_grid, lis_flux, A=A)
        spectrum_payload = {
            "spectrum_true": spectrum_true,
            "spectrum_pred": spectrum_pred,
            "spectrum_etoa": etoa_grid,
            "spectrum_log_residual": np.log(np.maximum(spectrum_pred, 1.0e-300))
            - np.log(np.maximum(spectrum_true, 1.0e-300)),
        }
    _atomic_save_npz(
        predictions_path,
        theta=theta,
        param_names=np.asarray(param_names),
        M_true=true_matrices,
        M_pred=pred_matrices,
        **spectrum_payload,
    )
    _atomic_save_npz(
        residuals_path,
        theta=theta,
        param_names=np.asarray(param_names),
        residual=residuals,
        relative_residual=relative,
        row_rmse=np.sqrt(np.mean(residuals * residuals, axis=2)),
        matrix_rmse=np.sqrt(np.mean(residuals * residuals, axis=(1, 2))),
        **spectrum_payload,
    )
    _write_metrics_csv(
        directory / f"{prefix}_metrics.csv",
        theta,
        true_matrices,
        pred_matrices,
        param_names,
        etoa_grid=etoa_grid,
        elis_grid=elis_grid,
        lis_flux=lis_flux,
        A=A,
    )


def _write_metrics_csv(
    path: Path,
    theta: np.ndarray,
    true_matrices: np.ndarray,
    pred_matrices: np.ndarray,
    param_names: Sequence[str],
    etoa_grid: np.ndarray | None = None,
    elis_grid: np.ndarray | None = None,
    lis_flux: np.ndarray | None = None,
    A: int = 1,
) -> None:
    path = prepare_output_path(path)
    tmp = temp_output_path(path)
    kl = np.asarray([row_kl(true, pred) for true, pred in zip(true_matrices, pred_matrices)])
    sample_rmse = np.sqrt(np.mean((pred_matrices - true_matrices) ** 2, axis=(1, 2)))
    row_sum_error = np.max(np.abs(pred_matrices.sum(axis=2) - 1.0), axis=1)
    try:
        with tmp.open("w", newline="", encoding="utf-8") as stream:
            fieldnames = [*param_names, "matrix_rmse", "mean_row_kl", "max_row_kl", "max_row_sum_error"]
            spectrum_rows = None
            if lis_flux is not None and etoa_grid is not None and elis_grid is not None:
                spectrum_rows = _spectrum_metric_rows(true_matrices, pred_matrices, etoa_grid, elis_grid, lis_flux, A=A)
                fieldnames.extend(
                    [
                        "spectrum_log_rmse",
                        "max_spectrum_log_error",
                        "high_energy_log_error",
                        "spectrum_percentage_rmse",
                        "max_spectrum_percentage_error",
                        "high_energy_percentage_error",
                        "spectrum_slope_rmse",
                        "high_energy_slope_error",
                    ]
                )
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
                if spectrum_rows is not None:
                    row.update(spectrum_rows[irow])
                writer.writerow(row)
        atomic_promote(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _fold_lis_batch(
    matrices: np.ndarray,
    etoa_grid: np.ndarray,
    elis_grid: np.ndarray,
    lis_flux: np.ndarray,
    A: int = 1,
) -> np.ndarray:
    from ..kernel import ekin_to_momentum

    matrix_batch = np.asarray(matrices, dtype=float)
    p_lis = ekin_to_momentum(elis_grid, A=A)
    p_toa = ekin_to_momentum(etoa_grid, A=A)
    weighted_lis = lis_flux / np.maximum(p_lis * p_lis, 1.0e-300)
    return (matrix_batch @ weighted_lis) * (p_toa * p_toa)[None, :]


def _spectrum_metric_rows(
    true_matrices: np.ndarray,
    pred_matrices: np.ndarray,
    etoa_grid: np.ndarray,
    elis_grid: np.ndarray,
    lis_flux: np.ndarray,
    A: int = 1,
) -> list[dict[str, float]]:
    true_spectrum = _fold_lis_batch(true_matrices, etoa_grid, elis_grid, lis_flux, A=A)
    pred_spectrum = _fold_lis_batch(pred_matrices, etoa_grid, elis_grid, lis_flux, A=A)
    log_delta = (
        np.log(np.maximum(pred_spectrum, 1.0e-300))
        - np.log(np.maximum(true_spectrum, 1.0e-300))
    )
    percentage_delta = (pred_spectrum - true_spectrum) / np.maximum(
        np.abs(true_spectrum),
        1.0e-300,
    )
    log_etoa = np.log(etoa_grid)
    slope_delta = np.diff(np.log(np.maximum(pred_spectrum, 1.0e-300)), axis=1) / np.diff(log_etoa)[None, :]
    slope_delta -= np.diff(np.log(np.maximum(true_spectrum, 1.0e-300)), axis=1) / np.diff(log_etoa)[None, :]
    rows = []
    for irow, values in enumerate(log_delta):
        rows.append(
            {
                "spectrum_log_rmse": float(np.sqrt(np.mean(values * values))),
                "max_spectrum_log_error": float(np.max(np.abs(values))),
                "high_energy_log_error": float(values[-1]),
                "spectrum_percentage_rmse": float(100.0 * np.sqrt(np.mean(percentage_delta[irow] ** 2))),
                "max_spectrum_percentage_error": float(100.0 * np.max(np.abs(percentage_delta[irow]))),
                "high_energy_percentage_error": float(100.0 * percentage_delta[irow, -1]),
                "spectrum_slope_rmse": float(np.sqrt(np.mean(slope_delta[irow] * slope_delta[irow]))),
                "high_energy_slope_error": float(slope_delta[irow, -1]),
            }
        )
    return rows


def _spectrum_metrics(
    true_matrices: np.ndarray,
    pred_matrices: np.ndarray,
    etoa_grid: np.ndarray,
    elis_grid: np.ndarray,
    lis_flux: np.ndarray,
    A: int = 1,
) -> dict[str, float]:
    rows = _spectrum_metric_rows(true_matrices, pred_matrices, etoa_grid, elis_grid, lis_flux, A=A)
    return {
        "spectrum_log_rmse": float(np.mean([row["spectrum_log_rmse"] for row in rows])),
        "max_spectrum_log_error": float(max(row["max_spectrum_log_error"] for row in rows)),
        "mean_high_energy_log_error": float(np.mean([row["high_energy_log_error"] for row in rows])),
        "spectrum_percentage_rmse": float(np.mean([row["spectrum_percentage_rmse"] for row in rows])),
        "max_spectrum_percentage_error": float(max(row["max_spectrum_percentage_error"] for row in rows)),
        "mean_high_energy_percentage_error": float(np.mean([row["high_energy_percentage_error"] for row in rows])),
        "spectrum_slope_rmse": float(np.mean([row["spectrum_slope_rmse"] for row in rows])),
        "mean_high_energy_slope_error": float(np.mean([row["high_energy_slope_error"] for row in rows])),
    }


def _write_config(
    path: Path,
    args,
    dataset,
    model_out: Path,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    kernel: TorchFNOTransferMatrixSurrogate,
    ranges: dict[str, tuple[float, float]],
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
        "boundary_padding": int(kernel.boundary_padding),
        "dropout": float(args.dropout),
        "optimizer": getattr(kernel, "optimizer_name_", "AdamW"),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "lr_scheduler": kernel.lr_scheduler,
        "lr_scheduler_factor": float(kernel.lr_scheduler_factor),
        "lr_scheduler_patience": int(kernel.lr_scheduler_patience),
        "lr_scheduler_cooldown": int(kernel.lr_scheduler_cooldown),
        "lr_scheduler_min_lr": float(kernel.lr_scheduler_min_lr),
        "early_stopping": bool(kernel.early_stopping),
        "early_stopping_patience": int(kernel.early_stopping_patience),
        "early_stopping_min_delta": float(kernel.early_stopping_min_delta),
        "early_stopping_min_epochs": int(kernel.early_stopping_min_epochs),
        "stopped_epoch": int(getattr(kernel, "stopped_epoch_", kernel.epochs)),
        "checkpoints": not bool(args.no_checkpoints),
        "reserve_checkpoint": str(args.reserve_checkpoint) if args.reserve_checkpoint is not None else None,
        "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint is not None else None,
        "seed": int(args.seed),
        "best_epoch": int(kernel.best_epoch_),
        "best_loss": float(kernel.best_loss_),
        "lis": str(args.lis) if args.lis is not None else None,
        "fixed": {name: float(value) for name, value in parse_key_value_options(args.fixed).items()},
        "ranges": {
            name: [float(bounds[0]), float(bounds[1])]
            for name, bounds in ranges.items()
        },
        "spectrum_A": int(round(parse_key_value_options(args.fixed).get("A", 1))),
        "matrix_probability_loss_weight": float(kernel.matrix_probability_loss_weight),
        "matrix_cross_entropy_weight": float(kernel.matrix_cross_entropy_weight),
        "spectrum_loss_weight": float(kernel.spectrum_loss_weight),
        "spectrum_max_error_percent": float(kernel.spectrum_max_error_percent),
        "spectrum_max_error_temperature_percent": float(kernel.spectrum_max_error_temperature_percent),
        "spectrum_huber_delta_percent": float(kernel.spectrum_huber_delta_percent),
        "spectrum_top_k": int(kernel.spectrum_top_k),
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
