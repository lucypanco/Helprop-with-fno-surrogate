"""Check folded-spectrum accuracy for every matrix in a surrogate dataset.

This module deliberately runs after training.  It loads the saved surrogate,
folds each reference matrix and its predicted matrix with the same LIS, and
writes a standalone JSON report.  It does not modify the model or any kernel
output.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .file_safety import atomic_replace_text
from .kernel import ekin_to_momentum, fold_lis
from .matrix_data import MatrixDataset, load_bson_matrices, load_npz_matrices
from .model import HelPropKernelModel, load_model


DEFAULT_MAX_RELATIVE_ERROR = 0.01
DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 1))
DEFAULT_PROGRESS_EVERY = 1000
ProgressCallback = Callable[[int, int, float], None]


def folded_relative_error(
    true_matrix: np.ndarray,
    predicted_matrix: np.ndarray,
    etoa_grid: Sequence[float],
    elis_grid: Sequence[float],
    lis_flux: Sequence[float],
    *,
    A: int = 1,
) -> np.ndarray:
    """Return absolute relative folded-spectrum error at each TOA energy."""
    true_spectrum = fold_lis(true_matrix, etoa_grid, elis_grid, lis_flux, A=A)
    predicted_spectrum = fold_lis(predicted_matrix, etoa_grid, elis_grid, lis_flux, A=A)
    return np.abs(predicted_spectrum - true_spectrum) / np.maximum(np.abs(true_spectrum), 1.0e-300)


def folding_report(
    model: HelPropKernelModel,
    dataset: MatrixDataset,
    lis_flux: Sequence[float],
    *,
    threshold: float = DEFAULT_MAX_RELATIVE_ERROR,
    workers: int = DEFAULT_WORKERS,
    progress_every: int = DEFAULT_PROGRESS_EVERY,
    max_matrices: int | None = None,
    etoa_min: float | None = None,
    etoa_max: float | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Evaluate every matrix and return a JSON-serializable report.

    ``threshold`` is a fractional error, so the default ``0.01`` means 1%.
    A matrix passes only when its maximum error is strictly below the limit.
    """
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be finite and positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    if max_matrices is not None and max_matrices <= 0:
        raise ValueError("max_matrices must be positive")
    _validate_energy_range(etoa_min, etoa_max)
    if set(dataset.param_names) != set(model.learned):
        raise ValueError(
            "dataset parameters must match model learned parameters: "
            f"dataset={dataset.param_names}, model={model.learned}"
        )

    etoa_grid = np.asarray(dataset.etoa_grid, dtype=float)
    elis_grid = np.asarray(dataset.elis_grid, dtype=float)
    lis_array = np.asarray(lis_flux, dtype=float)
    if lis_array.shape != elis_grid.shape:
        raise ValueError("lis_flux must match the dataset ELIS grid")
    selected_etoa = _select_energy_window(etoa_grid, etoa_min, etoa_max)
    if selected_etoa.size == 0:
        raise ValueError(
            "the requested ETOA range contains no dataset grid points: "
            f"[{etoa_min}, {etoa_max}]"
        )
    etoa_mask = np.isin(etoa_grid, selected_etoa)

    mass_number = int(round(model.fixed.get("A", 1)))
    weighted_lis, toa_factor = _folding_weights(elis_grid, etoa_grid, lis_array, mass_number)
    available_matrices = dataset.matrices.shape[0]
    n_matrices = (
        available_matrices
        if max_matrices is None
        else min(max_matrices, available_matrices)
    )
    model_columns = [dataset.param_names.index(name) for name in model.learned]
    matrix_rows: list[dict[str, object] | None] = [None] * n_matrices
    started = time.perf_counter()
    if workers > 1 and n_matrices and callable(getattr(model.kernel, "predict_matrices", None)):
        _warm_model_for_threads(model, dataset, model_columns)
    worker = partial(
        _evaluate_one,
        model=model,
        dataset=dataset,
        model_columns=model_columns,
        weighted_lis=weighted_lis,
        toa_factor=toa_factor,
        etoa_grid=etoa_grid,
        etoa_mask=etoa_mask,
        threshold=threshold,
    )
    if workers == 1:
        results = map(worker, range(n_matrices))
        _collect_results(results, matrix_rows, n_matrices, progress_every, progress, started)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fold") as executor:
            _collect_results(
                executor.map(worker, range(n_matrices)),
                matrix_rows,
                n_matrices,
                progress_every,
                progress,
                started,
            )

    completed_rows = [row for row in matrix_rows if row is not None]
    if len(completed_rows) != n_matrices:
        raise RuntimeError("folding report did not produce one result per matrix")
    errors = np.asarray([row["max_relative_error"] for row in completed_rows], dtype=float)
    failed = [row for row in completed_rows if not bool(row["passed"])]
    global_max_index = int(np.argmax(errors)) if errors.size else None
    if global_max_index is not None:
        global_max_point = dict(completed_rows[global_max_index]["max_relative_error_point"])
        global_max_point["matrix_index"] = int(completed_rows[global_max_index]["index"])
    else:
        global_max_point = None
    return {
        "threshold": float(threshold),
        "threshold_percent": 100.0 * float(threshold),
        "strict_comparison": "max_relative_error < threshold",
        "mass_number": mass_number,
        "etoa_range": {
            "min": float(selected_etoa[0]),
            "max": float(selected_etoa[-1]),
            "requested_min": None if etoa_min is None else float(etoa_min),
            "requested_max": None if etoa_max is None else float(etoa_max),
            "n_grid_points": int(selected_etoa.size),
        },
        "selection": {
            "start_index": 0,
            "stop_index_exclusive": int(n_matrices),
            "available_matrices": int(available_matrices),
        },
        "n_matrices": int(n_matrices),
        "n_passed": int(n_matrices - len(failed)),
        "n_failed": int(len(failed)),
        "max_relative_error": float(np.max(errors)) if errors.size else 0.0,
        "max_relative_error_percent": float(100.0 * np.max(errors)) if errors.size else 0.0,
        "max_relative_error_point": global_max_point,
        "passed": not failed,
        "matrices": completed_rows,
    }


def _collect_results(results, matrix_rows, total, progress_every, progress, started) -> None:
    for processed, row in enumerate(results, 1):
        matrix_rows[row["index"]] = row
        if progress is not None and (processed % progress_every == 0 or processed == total):
            elapsed = time.perf_counter() - started
            progress(processed, total, elapsed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write an independent folded-spectrum max-error report for every matrix."
    )
    parser.add_argument("--model", type=Path, required=True, help="Saved surrogate model (.pkl)")
    parser.add_argument("--dataset", type=Path, required=True, help="Matrix dataset (.npz or BSON)")
    parser.add_argument("--format", choices=("auto", "npz", "bson"), default="auto")
    parser.add_argument(
        "--learn",
        action="append",
        default=[],
        help="Learned parameter name, required for BSON datasets; repeat for each parameter",
    )
    parser.add_argument("--lis", type=Path, required=True, help="LIS spectrum file with E and flux columns")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_MAX_RELATIVE_ERROR,
        help="Maximum allowed fractional folded-spectrum error (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Concurrent per-matrix workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Print progress after approximately this many matrices (default: 1000)",
    )
    parser.add_argument(
        "--max-matrices",
        type=int,
        default=None,
        help="Test only the first N matrices; the input dataset is not modified",
    )
    parser.add_argument(
        "--etoa-range",
        type=_parse_energy_range,
        default=None,
        metavar="MIN,MAX",
        help=(
            "Restrict error evaluation to the inclusive ETOA range in GeV; "
            "only existing dataset grid points are evaluated"
        ),
    )
    parser.add_argument("--report-out", type=Path, required=True, help="Standalone JSON report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model = load_model(args.model)
    dataset = _load_dataset(args.dataset, args.format, model, args.learn)
    lis_flux = _read_lis_on_grid(args.lis, dataset.elis_grid)
    selected_count = (
        dataset.matrices.shape[0]
        if args.max_matrices is None
        else min(args.max_matrices, dataset.matrices.shape[0])
    )
    print(
        f"Processing each of the first {selected_count}/{dataset.matrices.shape[0]} matrices individually "
        f"with {args.workers} worker(s)...",
        file=sys.stderr,
        flush=True,
    )
    report = folding_report(
        model,
        dataset,
        lis_flux,
        threshold=args.threshold,
        workers=args.workers,
        progress_every=args.progress_every,
        max_matrices=args.max_matrices,
        etoa_min=None if args.etoa_range is None else args.etoa_range[0],
        etoa_max=None if args.etoa_range is None else args.etoa_range[1],
        progress=_print_progress,
    )
    report["model"] = str(args.model)
    report["dataset"] = str(args.dataset)
    report["lis"] = str(args.lis)
    atomic_replace_text(args.report_out, json.dumps(report, indent=2, sort_keys=True))
    print(
        f"Report written to {args.report_out}: "
        f"{report['n_passed']}/{report['n_matrices']} passed; "
        f"maximum error {report['max_relative_error_percent']:.6g}%",
        file=sys.stderr,
        flush=True,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "n_matrices": report["n_matrices"],
                "n_failed": report["n_failed"],
                "max_relative_error_percent": report["max_relative_error_percent"],
            },
            sort_keys=True,
        )
    )
    return 0 if bool(report["passed"]) else 2


def _evaluate_one(
    index: int,
    *,
    model: HelPropKernelModel,
    dataset: MatrixDataset,
    model_columns: Sequence[int],
    weighted_lis: np.ndarray,
    toa_factor: np.ndarray,
    etoa_grid: np.ndarray,
    etoa_mask: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    """Predict and fold exactly one matrix, without averaging samples."""
    theta = dataset.theta[index, model_columns]
    options = {name: float(value) for name, value in zip(model.learned, theta)}
    predicted_matrix = model.matrix(options, dataset.etoa_grid, dataset.elis_grid)
    if predicted_matrix.shape != dataset.matrices[index].shape:
        raise ValueError(
            f"predicted matrix {index} has shape {predicted_matrix.shape}; "
            f"expected {dataset.matrices[index].shape}"
        )
    true_spectrum = dataset.matrices[index] @ weighted_lis * toa_factor
    predicted_spectrum = predicted_matrix @ weighted_lis * toa_factor
    errors = np.abs(predicted_spectrum - true_spectrum) / np.maximum(
        np.abs(true_spectrum), 1.0e-300
    )
    selected_errors = errors[etoa_mask]
    selected_etoa = etoa_grid[etoa_mask]
    max_index = int(np.argmax(selected_errors))
    max_error = float(selected_errors[max_index])
    point_eto = float(selected_etoa[max_index])
    point_true = float(true_spectrum[etoa_mask][max_index])
    point_predicted = float(predicted_spectrum[etoa_mask][max_index])
    point_absolute = abs(point_predicted - point_true)
    return {
        "index": index,
        "source": dataset.sources[index] if dataset.sources else None,
        "parameters": {
            name: float(dataset.theta[index, column])
            for column, name in enumerate(dataset.param_names)
        },
        "max_relative_error": max_error,
        "max_relative_error_percent": 100.0 * max_error,
        "max_error_etoa": point_eto,
        "max_relative_error_point": {
            "etoa": point_eto,
            "true_spectrum": point_true,
            "predicted_spectrum": point_predicted,
            "absolute_error": point_absolute,
            "relative_error": max_error,
            "relative_error_percent": 100.0 * max_error,
        },
        "passed": bool(max_error < threshold),
    }


def _warm_model_for_threads(
    model: HelPropKernelModel,
    dataset: MatrixDataset,
    model_columns: Sequence[int],
) -> None:
    """Build lazy inference state before concurrent FNO calls begin."""
    if hasattr(model.kernel, "model_"):
        return
    theta = dataset.theta[0, model_columns]
    options = {name: float(value) for name, value in zip(model.learned, theta)}
    model.matrix(options, dataset.etoa_grid, dataset.elis_grid)


def _folding_weights(
    elis_grid: np.ndarray,
    etoa_grid: np.ndarray,
    lis_flux: np.ndarray,
    A: int,
) -> tuple[np.ndarray, np.ndarray]:
    if np.any(~np.isfinite(lis_flux)) or np.any(lis_flux <= 0.0):
        raise ValueError("lis_flux must contain finite positive values")
    p_lis = ekin_to_momentum(elis_grid, A=A)
    p_toa = ekin_to_momentum(etoa_grid, A=A)
    return lis_flux / (p_lis * p_lis), p_toa * p_toa


def _parse_energy_range(value: str) -> tuple[float, float]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("energy range must use MIN,MAX")
    try:
        lower, upper = (float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("energy range must use numeric MIN,MAX") from exc
    _validate_energy_range(lower, upper)
    return lower, upper


def _validate_energy_range(lower: float | None, upper: float | None) -> None:
    if lower is not None and (not np.isfinite(lower) or lower <= 0.0):
        raise ValueError("etoa_min must be finite and positive")
    if upper is not None and (not np.isfinite(upper) or upper <= 0.0):
        raise ValueError("etoa_max must be finite and positive")
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("etoa_min must not exceed etoa_max")


def _select_energy_window(
    etoa_grid: np.ndarray,
    lower: float | None,
    upper: float | None,
) -> np.ndarray:
    mask = np.ones(etoa_grid.shape, dtype=bool)
    if lower is not None:
        mask &= etoa_grid >= lower
    if upper is not None:
        mask &= etoa_grid <= upper
    return etoa_grid[mask]


def _print_progress(processed: int, total: int, elapsed: float) -> None:
    rate = processed / elapsed if elapsed > 0.0 else 0.0
    print(
        f"Processed {processed}/{total} matrices "
        f"({100.0 * processed / total:.1f}%, {rate:.1f} matrices/s)",
        file=sys.stderr,
        flush=True,
    )


def _load_dataset(
    path: Path,
    fmt: str,
    model: HelPropKernelModel,
    learned: Sequence[str],
) -> MatrixDataset:
    if fmt == "auto":
        fmt = "bson" if path.suffix.lower() in {".bson", ".bin"} else "npz"
    if fmt == "npz":
        return load_npz_matrices(path)
    if fmt == "bson":
        names = tuple(learned) or tuple(model.learned)
        return load_bson_matrices(path, names)
    raise ValueError(f"unknown dataset format {fmt}")


def _read_lis_on_grid(path: Path, elis_grid: np.ndarray) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("LIS file must contain at least two columns: E flux")
    energy = np.asarray(data[:, 0], dtype=float)
    flux = np.asarray(data[:, 1], dtype=float)
    if np.any(~np.isfinite(energy)) or np.any(~np.isfinite(flux)):
        raise ValueError("LIS contains non-finite values")
    if np.any(energy <= 0.0) or np.any(flux <= 0.0):
        raise ValueError("LIS energies and fluxes must be positive")
    if np.any(np.diff(energy) <= 0.0):
        raise ValueError("LIS energies must be strictly increasing")
    if elis_grid[0] < energy[0] or elis_grid[-1] > energy[-1]:
        raise ValueError(
            "LIS must cover the full dataset ELIS grid "
            f"[{elis_grid[0]}, {elis_grid[-1]}]"
        )
    return np.exp(np.interp(np.log(elis_grid), np.log(energy), np.log(flux)))


if __name__ == "__main__":
    raise SystemExit(main())
