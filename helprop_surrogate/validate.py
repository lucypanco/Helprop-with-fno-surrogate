"""Validation and batch adoption checks for HelProp surrogate datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .data import TransitionDataset, load_bson_transitions, load_npz_transitions
from .file_safety import atomic_replace_text
from .kernel import energy_bin_edges, fold_lis
from .model import load_model, parse_key_value_options


def empirical_matrix(
    dataset: TransitionDataset,
    etoa_grid: Sequence[float],
    elis_grid: Sequence[float],
) -> np.ndarray:
    """Bin particle transitions into a row-normalized empirical matrix."""
    etoa_grid = np.asarray(etoa_grid, dtype=float)
    elis_grid = np.asarray(elis_grid, dtype=float)
    etoa_edges = energy_bin_edges(etoa_grid)
    elis_edges = energy_bin_edges(elis_grid)
    matrix = np.zeros((etoa_grid.size, elis_grid.size), dtype=float)
    for irow in range(etoa_grid.size):
        mask = (dataset.etoa >= etoa_edges[irow]) & (dataset.etoa < etoa_edges[irow + 1])
        if not np.any(mask):
            continue
        counts, _ = np.histogram(dataset.elis[mask], bins=elis_edges)
        total = counts.sum()
        if total > 0:
            matrix[irow] = counts / total
    return matrix


def row_kl(true_matrix: np.ndarray, pred_matrix: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    """Compute KL(M_true || M_pred) per non-empty row."""
    true = np.asarray(true_matrix, dtype=float)
    pred = np.asarray(pred_matrix, dtype=float)
    if true.shape != pred.shape:
        raise ValueError("matrix shapes do not match")
    true = np.maximum(true, 0.0)
    pred = np.maximum(pred, 0.0)
    true_sum = true.sum(axis=1)
    pred_sum = pred.sum(axis=1)
    valid = (true_sum > 0.0) & (pred_sum > 0.0)
    result = np.full(true.shape[0], np.nan)
    true_norm = true[valid] / true_sum[valid, None]
    pred_norm = pred[valid] / pred_sum[valid, None]
    result[valid] = np.sum(true_norm * np.log((true_norm + eps) / (pred_norm + eps)), axis=1)
    return result


def matrix_validation_report(
    true_matrix: np.ndarray,
    pred_matrix: np.ndarray,
    lis_flux: Sequence[float] | None = None,
    etoa_grid: Sequence[float] | None = None,
    elis_grid: Sequence[float] | None = None,
    A: int = 1,
) -> dict[str, float]:
    """Summarize matrix and optional folded-spectrum agreement."""
    kl = row_kl(true_matrix, pred_matrix)
    finite = kl[np.isfinite(kl)]
    report = {
        "valid_rows": float(finite.size),
        "mean_row_kl": float(np.mean(finite)) if finite.size else float("inf"),
        "max_row_kl": float(np.max(finite)) if finite.size else float("inf"),
    }
    if lis_flux is not None:
        if etoa_grid is None or elis_grid is None:
            raise ValueError("etoa_grid and elis_grid are required with lis_flux")
        true_flux = fold_lis(true_matrix, etoa_grid, elis_grid, lis_flux, A=A)
        pred_flux = fold_lis(pred_matrix, etoa_grid, elis_grid, lis_flux, A=A)
        rel = np.abs(pred_flux - true_flux) / np.maximum(np.abs(true_flux), 1.0e-300)
        report["mean_flux_relerr"] = float(np.mean(rel))
        report["max_flux_relerr"] = float(np.max(rel))
    return report


def batch_quality_report(
    dataset: TransitionDataset,
    etoa_grid: Sequence[float],
    elis_grid: Sequence[float],
    min_samples_per_row: int = 1,
) -> dict[str, float | bool]:
    """Check basic batch quality before adopting it."""
    etoa_grid = np.asarray(etoa_grid, dtype=float)
    etoa_edges = energy_bin_edges(etoa_grid)
    counts = np.zeros(etoa_grid.size, dtype=int)
    for irow in range(etoa_grid.size):
        counts[irow] = int(np.sum((dataset.etoa >= etoa_edges[irow]) & (dataset.etoa < etoa_edges[irow + 1])))
    matrix = empirical_matrix(dataset, etoa_grid, elis_grid)
    row_sums = matrix.sum(axis=1)
    finite = bool(np.all(np.isfinite(dataset.etoa)) and np.all(np.isfinite(dataset.elis)))
    enough = bool(np.all(counts >= min_samples_per_row))
    normalized = bool(np.allclose(row_sums[counts > 0], 1.0))
    return {
        "finite": finite,
        "min_samples_per_row": float(counts.min()) if counts.size else 0.0,
        "rows_normalized": normalized,
        "accepted_basic": finite and enough and normalized,
    }


def should_accept(report: Mapping[str, float | bool], max_mean_kl: float, max_flux_relerr: float | None) -> bool:
    """Apply validation thresholds to a report."""
    if "accepted_basic" in report and not bool(report["accepted_basic"]):
        return False
    if float(report.get("mean_row_kl", 0.0)) > max_mean_kl:
        return False
    if max_flux_relerr is not None and float(report.get("max_flux_relerr", 0.0)) > max_flux_relerr:
        return False
    return True


def append_dataset(base: TransitionDataset, batch: TransitionDataset) -> TransitionDataset:
    names = tuple(base.params.keys())
    if set(names) != set(batch.params.keys()):
        raise ValueError("base and batch parameter names do not match")
    return TransitionDataset(
        etoa=np.concatenate([base.etoa, batch.etoa]),
        elis=np.concatenate([base.elis, batch.elis]),
        params={name: np.concatenate([base.params[name], batch.params[name]]) for name in names},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate or adopt a HelProp surrogate batch.")
    parser.add_argument("--model", type=Path, help="Saved surrogate model for KL validation")
    parser.add_argument("--batch", type=Path, required=True, help="Batch .npz or BSON file")
    parser.add_argument("--format", choices=["auto", "npz", "bson"], default="auto")
    parser.add_argument("--etoa", required=True, help="TOA grid file or min,max,n")
    parser.add_argument("--elis", required=True, help="LIS grid file or min,max,n")
    parser.add_argument("--param", action="append", default=[], help="Learned parameter as name=value")
    parser.add_argument("--base", type=Path, help="Existing dataset for adoption")
    parser.add_argument("--adopt-out", type=Path, help="Write combined dataset if accepted")
    parser.add_argument("--max-mean-kl", type=float, default=0.1)
    parser.add_argument("--max-flux-relerr", type=float, default=None)
    parser.add_argument("--min-samples-per-row", type=int, default=1)
    parser.add_argument("--report-out", type=Path, help="Write JSON report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    etoa = _read_grid(args.etoa)
    elis = _read_grid(args.elis)
    batch = _load_dataset(args.batch, args.format, ())
    report = batch_quality_report(batch, etoa, elis, args.min_samples_per_row)

    if args.model is not None:
        model = load_model(args.model)
        options = parse_key_value_options(args.param)
        true_matrix = empirical_matrix(batch, etoa, elis)
        pred_matrix = model.matrix(options, etoa, elis)
        report.update(matrix_validation_report(true_matrix, pred_matrix))

    accepted = should_accept(report, args.max_mean_kl, args.max_flux_relerr)
    report["accepted"] = bool(accepted)

    if accepted and args.base is not None and args.adopt_out is not None:
        base = load_npz_transitions(args.base)
        combined = append_dataset(base, batch)
        combined.save_npz(args.adopt_out)
        report["adopted_dataset"] = str(args.adopt_out)

    if args.report_out is not None:
        atomic_replace_text(
            args.report_out,
            json.dumps(report, indent=2, sort_keys=True),
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if accepted else 2


def _load_dataset(path: Path, fmt: str, names: Sequence[str]) -> TransitionDataset:
    if fmt == "auto":
        fmt = "bson" if path.suffix.lower() in {".bson", ".bin"} else "npz"
    if fmt == "npz":
        return load_npz_transitions(path)
    if fmt == "bson":
        return load_bson_transitions(path, param_names=names)
    raise ValueError(f"unknown format {fmt}")


def _read_grid(spec: str) -> np.ndarray:
    path = Path(spec)
    if path.exists():
        data = np.loadtxt(path)
        return np.asarray(data if data.ndim == 1 else data[:, 0], dtype=float)
    parts = [float(item) for item in spec.split(",")]
    if len(parts) != 3:
        raise ValueError("grid must be file path or min,max,n")
    return np.geomspace(parts[0], parts[1], int(parts[2]))


if __name__ == "__main__":
    raise SystemExit(main())
