"""Shared HelProp transfer-matrix dataset generation for surrogate backends."""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .file_safety import (
    atomic_promote,
    atomic_replace_text,
    ensure_unique_outputs,
    prepare_output_path,
    temp_output_path,
)
from .generate_samples import fixed_to_helprop_options, latin_hypercube
from .model import parse_key_value_options


DEFAULT_MATRIX_PARAM_RANGES = {
    "D0": (0.1, 50.0),
    "m": (-2.0, 2.0),
}


@dataclass(frozen=True)
class MatrixDataset:
    """Full HelProp transfer matrices indexed by parameter point."""

    theta: np.ndarray
    matrices: np.ndarray
    param_names: tuple[str, ...]
    etoa_grid: np.ndarray
    elis_grid: np.ndarray
    sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        theta = np.asarray(self.theta, dtype=float)
        matrices = np.asarray(self.matrices, dtype=float)
        etoa_grid = np.asarray(self.etoa_grid, dtype=float)
        elis_grid = np.asarray(self.elis_grid, dtype=float)
        param_names = tuple(self.param_names)

        if theta.ndim != 2:
            raise ValueError("theta must have shape (n_samples, n_params)")
        if matrices.ndim != 3:
            raise ValueError("matrices must have shape (n_samples, n_etoa, n_elis)")
        if theta.shape[0] != matrices.shape[0]:
            raise ValueError("theta and matrices must have the same sample count")
        if theta.shape[1] != len(param_names):
            raise ValueError("theta column count must match param_names")
        if matrices.shape[1:] != (etoa_grid.size, elis_grid.size):
            raise ValueError("matrix shape must match etoa_grid and elis_grid")
        if etoa_grid.ndim != 1 or elis_grid.ndim != 1:
            raise ValueError("energy grids must be one-dimensional")
        if etoa_grid.size == 0 or elis_grid.size == 0:
            raise ValueError("energy grids must not be empty")
        if np.any(~np.isfinite(theta)):
            raise ValueError("theta contains non-finite values")
        if np.any(~np.isfinite(matrices)) or np.any(matrices < 0.0):
            raise ValueError("matrices must be finite and non-negative")
        if np.any(~np.isfinite(etoa_grid)) or np.any(etoa_grid <= 0.0):
            raise ValueError("etoa_grid must contain finite positive values")
        if np.any(~np.isfinite(elis_grid)) or np.any(elis_grid <= 0.0):
            raise ValueError("elis_grid must contain finite positive values")
        if np.any(np.diff(etoa_grid) <= 0.0) or np.any(np.diff(elis_grid) <= 0.0):
            raise ValueError("energy grids must be strictly increasing")

        row_sums = matrices.sum(axis=2, keepdims=True)
        if np.any(row_sums <= 0.0) or np.any(~np.isfinite(row_sums)):
            raise ValueError("every matrix row must have positive finite probability")
        matrices = matrices / row_sums

        sources = tuple(str(item) for item in self.sources)
        if sources and len(sources) != theta.shape[0]:
            raise ValueError("sources must be empty or match sample count")

        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "matrices", matrices)
        object.__setattr__(self, "param_names", param_names)
        object.__setattr__(self, "etoa_grid", etoa_grid)
        object.__setattr__(self, "elis_grid", elis_grid)
        object.__setattr__(self, "sources", sources)

    def save_npz(self, path: str | Path) -> None:
        """Write this matrix dataset atomically."""
        path = prepare_output_path(path)
        tmp = temp_output_path(path)
        payload = {
            "theta": self.theta,
            "matrices": self.matrices,
            "param_names": np.asarray(self.param_names),
            "etoa_grid": self.etoa_grid,
            "elis_grid": self.elis_grid,
            "sources": np.asarray(self.sources),
        }
        try:
            with tmp.open("wb") as stream:
                np.savez(stream, **payload)
            atomic_promote(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()


@dataclass(frozen=True)
class HelPropMatrixRun:
    """Description of one HelProp matrix-generation run."""

    index: int
    params: dict[str, float]
    output: Path
    command: list[str]


def load_npz_matrices(path: str | Path) -> MatrixDataset:
    """Load ``MatrixDataset.save_npz`` output."""
    with np.load(path, allow_pickle=False) as data:
        sources = tuple(str(item) for item in data["sources"]) if "sources" in data else ()
        return MatrixDataset(
            theta=np.asarray(data["theta"], dtype=float),
            matrices=np.asarray(data["matrices"], dtype=float),
            param_names=tuple(str(item) for item in data["param_names"]),
            etoa_grid=np.asarray(data["etoa_grid"], dtype=float),
            elis_grid=np.asarray(data["elis_grid"], dtype=float),
            sources=sources,
        )


def load_bson_matrices(
    path: str | Path,
    param_names: Sequence[str],
) -> MatrixDataset:
    """Load one or more HelProp BSON matrix documents."""
    docs = _read_bson_documents(path)
    if not docs:
        raise ValueError(f"no BSON documents found in {path}")

    theta_rows = []
    matrices = []
    etoa_grid = None
    elis_grid = None
    for index, doc in enumerate(docs):
        params = doc.get("params")
        if not isinstance(params, Mapping):
            raise ValueError(f"BSON document {index} does not contain params")
        try:
            theta_rows.append([float(params[name]) for name in param_names])
        except KeyError as exc:
            raise KeyError(f"BSON document {index} missing parameter '{exc.args[0]}'") from exc

        matrix = np.asarray(doc.get("M"), dtype=float)
        etoa = np.asarray(doc.get("ETOA"), dtype=float)
        elis = np.asarray(doc.get("ELIS"), dtype=float)
        if matrix.ndim != 2:
            raise ValueError(f"BSON document {index} does not contain a 2D matrix")
        if etoa_grid is None:
            etoa_grid = etoa
            elis_grid = elis
        elif not np.allclose(etoa_grid, etoa) or not np.allclose(elis_grid, elis):
            raise ValueError("all BSON matrix documents must use the same grids")
        matrices.append(matrix)

    return MatrixDataset(
        theta=np.asarray(theta_rows, dtype=float),
        matrices=np.asarray(matrices, dtype=float),
        param_names=tuple(param_names),
        etoa_grid=np.asarray(etoa_grid, dtype=float),
        elis_grid=np.asarray(elis_grid, dtype=float),
        sources=tuple(str(path) for _ in docs),
    )


def consolidate_bson_matrices(
    paths: Sequence[str | Path],
    param_names: Sequence[str],
) -> MatrixDataset:
    """Load multiple BSON matrix files into one matrix dataset."""
    datasets = [load_bson_matrices(path, param_names) for path in paths]
    if not datasets:
        raise ValueError("no BSON matrix paths were provided")
    etoa_grid = datasets[0].etoa_grid
    elis_grid = datasets[0].elis_grid
    for dataset in datasets[1:]:
        if not np.allclose(dataset.etoa_grid, etoa_grid) or not np.allclose(dataset.elis_grid, elis_grid):
            raise ValueError("all matrix datasets must use the same grids")
    return MatrixDataset(
        theta=np.concatenate([dataset.theta for dataset in datasets], axis=0),
        matrices=np.concatenate([dataset.matrices for dataset in datasets], axis=0),
        param_names=tuple(param_names),
        etoa_grid=etoa_grid,
        elis_grid=elis_grid,
        sources=tuple(source for dataset in datasets for source in dataset.sources),
    )


def parse_choice_options(items: Sequence[str]) -> dict[str, tuple[float, ...]]:
    """Parse categorical learned-parameter choices as ``name:v1,v2,...``."""
    choices = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"choice must be name:v1,v2,..., got {item!r}")
        name, raw_values = item.split(":", 1)
        if not name:
            raise ValueError(f"empty choice name in {item!r}")
        values = tuple(float(value) for value in raw_values.split(",") if value)
        if not values:
            raise ValueError(f"choice list for {name} must not be empty")
        choices[name] = values
    return choices


def mixed_parameter_design(
    learned: Sequence[str],
    ranges: Mapping[str, tuple[float, float]],
    choices: Mapping[str, Sequence[float]],
    n_runs: int,
    seed: int,
) -> list[dict[str, float]]:
    """Build a deterministic mixed continuous/categorical parameter design."""
    learned = tuple(learned)
    continuous_names = [name for name in learned if name in ranges]
    if continuous_names:
        continuous_design = latin_hypercube(
            {name: ranges[name] for name in continuous_names},
            n_runs,
            seed=seed,
        )
    else:
        continuous_design = [{} for _ in range(n_runs)]

    rng = np.random.default_rng(seed)
    categorical_columns = {}
    for name in learned:
        if name not in choices:
            continue
        values = np.asarray(choices[name], dtype=float)
        repeats = int(np.ceil(n_runs / values.size))
        column = np.tile(values, repeats)[:n_runs].copy()
        rng.shuffle(column)
        categorical_columns[name] = column

    design = []
    for index in range(n_runs):
        row = {}
        for name in learned:
            if name in ranges:
                row[name] = float(continuous_design[index][name])
            elif name in categorical_columns:
                row[name] = float(categorical_columns[name][index])
            else:
                raise KeyError(f"no range or choice supplied for learned parameter {name}")
        design.append(row)
    return design


def split_indices(
    n_samples: int,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/validation/test indices split by complete matrix sample."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be less than 1")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    n_train = int(np.floor(n_samples * train_fraction))
    n_val = int(np.floor(n_samples * val_fraction))
    if n_samples >= 3:
        n_train = max(1, min(n_train, n_samples - 2))
        n_val = max(1, min(n_val, n_samples - n_train - 1))
    n_test = n_samples - n_train - n_val
    if n_test < 0:
        raise ValueError("invalid split produced negative test size")
    return (
        np.sort(indices[:n_train]),
        np.sort(indices[n_train : n_train + n_val]),
        np.sort(indices[n_train + n_val :]),
    )


def write_split_files(
    directory: str | Path,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    test_indices: Sequence[int],
) -> None:
    """Write train/validation/test split index files."""
    directory = prepare_output_path(Path(directory) / "placeholder").parent
    for name, values in {
        "train_indices.txt": train_indices,
        "val_indices.txt": val_indices,
        "test_indices.txt": test_indices,
    }.items():
        path = directory / name
        tmp = temp_output_path(path)
        try:
            np.savetxt(tmp, np.asarray(values, dtype=int), fmt="%d")
            atomic_promote(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()


def read_index_file(path: str | Path) -> np.ndarray:
    """Load an index file written by ``write_split_files``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.loadtxt(path, dtype=int)
    if data.size == 0:
        return np.asarray([], dtype=int)
    return np.atleast_1d(data).astype(int)


def next_serial_run_dir(root: str | Path = "fno_runs", prefix: str = "run_", width: int = 4) -> Path:
    """Create and return the next root-level serial FNO run directory."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for number in range(1, 100000):
        path = root / f"{prefix}{number:0{width}d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f"could not allocate a serial run directory under {root}")


def build_helprop_matrix_command(
    helprop: str | Path,
    output: str | Path,
    params: Mapping[str, float],
    etoa: str,
    elis: str,
    number: int,
    nthread: int,
    seed: int | None = None,
    fixed_options: Sequence[str] = (),
    sample: bool = False,
    integer_params: Sequence[str] = (),
) -> list[str]:
    """Build a HelProp command that writes a BSON transfer matrix."""
    if number <= 0:
        raise ValueError("number must be positive")
    if nthread <= 0:
        raise ValueError("nthread must be positive")

    command = [
        str(helprop),
        "--iotype=BSON",
        f"--etoa={etoa}",
        f"--elis={elis}",
        f"--number={number}",
        f"--nthread={nthread}",
    ]
    if sample:
        command.append("--sample")
    if seed is not None:
        command.append(f"--seed={seed}")
    command.extend(str(option) for option in fixed_options)
    integer_names = set(integer_params)
    for name, value in params.items():
        if name in integer_names:
            command.append(f"--{name}={int(round(float(value)))}")
        else:
            command.append(f"--{name}={value:.17g}")
    command.append(str(output))
    return command


def make_matrix_runs(
    helprop: str | Path,
    outdir: str | Path,
    design: Sequence[Mapping[str, float]],
    etoa: str,
    elis: str,
    number: int,
    nthread: int,
    seed: int | None = None,
    fixed_options: Sequence[str] = (),
    sample: bool = False,
    integer_params: Sequence[str] = (),
) -> list[HelPropMatrixRun]:
    """Create matrix-generation run descriptors for a parameter design."""
    outdir = Path(outdir)
    width = max(4, len(str(max(len(design) - 1, 0))))
    runs = []
    integer_names = set(integer_params)
    for index, params in enumerate(design):
        normalized_params = {
            name: float(int(round(float(value)))) if name in integer_names else float(value)
            for name, value in params.items()
        }
        run_seed = None if seed is None else seed + index * 1000003
        output = outdir / f"matrix_{index:0{width}d}.bson"
        command = build_helprop_matrix_command(
            helprop=helprop,
            output=output,
            params=normalized_params,
            etoa=etoa,
            elis=elis,
            number=number,
            nthread=nthread,
            seed=run_seed,
            fixed_options=fixed_options,
            sample=sample,
            integer_params=integer_params,
        )
        runs.append(
            HelPropMatrixRun(
                index=index,
                params=normalized_params,
                output=output,
                command=command,
            )
        )
    return runs


def write_matrix_manifest(path: str | Path, runs: Sequence[HelPropMatrixRun]) -> None:
    """Write a CSV manifest for FNO matrix-generation runs."""
    stream = io.StringIO()
    param_names = list(runs[0].params.keys()) if runs else []
    writer = csv.writer(stream)
    writer.writerow(["index", "output", *param_names, "command"])
    for run in runs:
        writer.writerow(
            [
                run.index,
                run.output,
                *[run.params[name] for name in param_names],
                " ".join(run.command),
            ]
        )
    atomic_replace_text(path, stream.getvalue())


def run_helprop_matrices(
    runs: Sequence[HelPropMatrixRun],
    timeout: float | None = None,
    dry_run: bool = False,
    jobs: int = 1,
) -> None:
    """Execute HelProp matrix commands."""
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    for run in runs:
        run.output.parent.mkdir(parents=True, exist_ok=True)
        if dry_run:
            print(" ".join(run.command))
    if dry_run:
        return

    total = len(runs)
    if jobs == 1:
        for ordinal, run in enumerate(runs, start=1):
            print(f"[{ordinal}/{total}] {run.output}")
            _run_one_helprop_matrix(run, timeout)
        return

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(_run_one_helprop_matrix, run, timeout): run
            for run in runs
        }
        completed = 0
        for future in as_completed(futures):
            run = futures[future]
            future.result()
            completed += 1
            print(f"[{completed}/{total}] {run.output}")


def _run_one_helprop_matrix(run: HelPropMatrixRun, timeout: float | None = None) -> None:
    """Run one HelProp matrix command and atomically promote its output."""
    temporary_output = temp_output_path(run.output)
    command = list(run.command)
    command[-1] = str(temporary_output)
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"HelProp failed for matrix run {run.index} with exit code "
                f"{result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        atomic_promote(temporary_output, run.output)
    finally:
        if temporary_output.exists():
            temporary_output.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a numbered HelProp transfer-matrix dataset.")
    parser.add_argument("--helprop", default="./HelProp", help="Path to HelProp executable")
    parser.add_argument("--runs-root", type=Path, default=Path("fno_runs"))
    parser.add_argument("--run-dir", type=Path, default=None, help="Use this run directory instead of allocating one")
    parser.add_argument("--n-runs", type=int, default=200, help="Number of parameter points")
    parser.add_argument("--learn", nargs="+", default=["D0", "m"], help="Learned parameter names")
    parser.add_argument("--range", dest="ranges", action="append", default=[], help="Parameter range name:min:max")
    parser.add_argument(
        "--choice",
        action="append",
        default=[],
        help="Categorical learned parameter choices as name:v1,v2,...",
    )
    parser.add_argument("--fixed", action="append", default=[], help="Fixed HelProp parameter name=value")
    parser.add_argument("--etoa", default="0.1,100,30", help="HelProp --etoa min,max,n")
    parser.add_argument("--elis", default="0.1,100,30", help="HelProp --elis min,max,n")
    parser.add_argument("--number", type=int, default=200, help="Particles per ETOA bin")
    parser.add_argument("--nthread", type=int, default=1)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of HelProp parameter-point runs to execute concurrently",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--sample", action="store_true", help="Also store particle samples in BSON")
    parser.add_argument(
        "--integer-param",
        action="append",
        default=[],
        help="Learned parameter rounded before calling HelProp, e.g. A or Z",
    )
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Write manifests/config and print commands only")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir or next_serial_run_dir(args.runs_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    ranges = _parse_ranges(args.ranges)
    choices = parse_choice_options(args.choice)
    learned = tuple(args.learn)
    if not args.ranges and not args.choice and learned == ("D0", "m"):
        ranges = {name: DEFAULT_MATRIX_PARAM_RANGES[name] for name in learned}
    repeated_design = set(ranges).intersection(choices)
    if repeated_design:
        names = ", ".join(sorted(repeated_design))
        raise ValueError(f"parameters cannot have both --range and --choice: {names}")
    missing_design = set(learned).difference(ranges).difference(choices)
    if missing_design:
        names = ", ".join(sorted(missing_design))
        raise ValueError(f"missing --range or --choice for learned parameters: {names}")
    extra_design = set(ranges).union(choices).difference(learned)
    if (args.ranges or args.choice) and extra_design:
        names = ", ".join(sorted(extra_design))
        raise ValueError(f"design supplied for non-learned parameters: {names}")

    fixed = parse_key_value_options(args.fixed)
    repeated = set(learned).intersection(fixed)
    if repeated:
        names = ", ".join(sorted(repeated))
        raise ValueError(f"parameters cannot be both learned and fixed: {names}")
    unknown_integer = set(args.integer_param).difference(learned)
    if unknown_integer:
        names = ", ".join(sorted(unknown_integer))
        raise ValueError(f"integer parameters must be learned parameters: {names}")

    design = mixed_parameter_design(learned, ranges, choices, args.n_runs, seed=args.seed)
    fixed_options = fixed_to_helprop_options(fixed)
    runs = make_matrix_runs(
        helprop=args.helprop,
        outdir=data_dir,
        design=design,
        etoa=args.etoa,
        elis=args.elis,
        number=args.number,
        nthread=args.nthread,
        seed=args.seed,
        fixed_options=fixed_options,
        sample=args.sample,
        integer_params=args.integer_param,
    )

    manifest = data_dir / "manifest.csv"
    dataset_out = data_dir / "matrices.npz"
    config = {
        "learned": list(learned),
        "fixed": fixed,
        "ranges": {name: list(value) for name, value in ranges.items()},
        "choices": {name: list(value) for name, value in choices.items()},
        "etoa": args.etoa,
        "elis": args.elis,
        "number": args.number,
        "nthread": args.nthread,
        "jobs": args.jobs,
        "n_runs": args.n_runs,
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "sample": bool(args.sample),
        "integer_params": list(args.integer_param),
    }
    ensure_unique_outputs([manifest, dataset_out, run_dir / "config.json", *[run.output for run in runs]])
    atomic_replace_text(run_dir / "config.json", json.dumps(config, indent=2, sort_keys=True))
    write_matrix_manifest(manifest, runs)
    print(f"run_dir: {run_dir}")
    print(f"manifest: {manifest}")

    run_helprop_matrices(runs, timeout=args.timeout, dry_run=args.dry_run, jobs=args.jobs)
    if args.dry_run:
        return 0

    dataset = consolidate_bson_matrices([run.output for run in runs], learned)
    dataset.save_npz(dataset_out)
    train_idx, val_idx, test_idx = split_indices(
        dataset.theta.shape[0],
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    write_split_files(data_dir, train_idx, val_idx, test_idx)
    print(f"dataset: {dataset_out}")
    print(f"split: train={train_idx.size} val={val_idx.size} test={test_idx.size}")
    return 0


def _parse_ranges(specs: Sequence[str]) -> dict[str, tuple[float, float]]:
    ranges = {}
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"range must be name:min:max, got {spec!r}")
        name, low, high = parts
        low_value = float(low)
        high_value = float(high)
        if high_value <= low_value:
            raise ValueError(f"range upper bound must exceed lower bound for {name}")
        ranges[name] = (low_value, high_value)
    return ranges


def _read_bson_documents(path: str | Path) -> list[Mapping[str, object]]:
    try:
        from bson import BSON
    except ImportError as exc:
        raise ImportError("load_bson_matrices requires pymongo's bson module") from exc

    filename = Path(path)
    docs = []
    with filename.open("rb") as stream:
        while True:
            header = stream.read(4)
            if not header:
                break
            if len(header) != 4:
                raise ValueError(f"truncated BSON size header in {filename}")
            size = int.from_bytes(header, byteorder="little", signed=True)
            if size < 5:
                raise ValueError(f"invalid BSON document size {size} in {filename}")
            body = stream.read(size - 4)
            if len(body) != size - 4:
                raise ValueError(f"truncated BSON document in {filename}")
            docs.append(BSON(header + body).decode())
    return docs


if __name__ == "__main__":
    raise SystemExit(main())
