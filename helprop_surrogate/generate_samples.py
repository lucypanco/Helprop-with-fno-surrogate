"""Generate HelProp particle-transition datasets for surrogate training."""

from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .data import TransitionDataset, load_bson_transitions
from .file_safety import (
    atomic_promote,
    atomic_replace_text,
    ensure_unique_outputs,
    temp_output_path,
)
from .model import parse_key_value_options


DEFAULT_PARAM_RANGES = {
    "D0": (0.1, 50.0),
    "m": (-2.0, 2.0),
}


@dataclass(frozen=True)
class HelPropSampleRun:
    """Description of one HelProp BSON sample run."""

    index: int
    params: dict[str, float]
    output: Path
    command: list[str]


def latin_hypercube(
    ranges: Mapping[str, tuple[float, float]],
    n_runs: int,
    seed: int | None = None,
) -> list[dict[str, float]]:
    """Create a deterministic Latin-hypercube design over parameter ranges."""
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    names = list(ranges.keys())
    if not names:
        raise ValueError("ranges must not be empty")

    rng = np.random.default_rng(seed)
    design = np.empty((n_runs, len(names)), dtype=float)
    for col, name in enumerate(names):
        low, high = ranges[name]
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError(f"invalid range for {name}: {low}, {high}")
        strata = (np.arange(n_runs) + rng.random(n_runs)) / n_runs
        rng.shuffle(strata)
        design[:, col] = low + strata * (high - low)

    return [
        {name: float(design[row, col]) for col, name in enumerate(names)}
        for row in range(n_runs)
    ]


def parse_ranges(specs: Sequence[str]) -> dict[str, tuple[float, float]]:
    """Parse ``name:min:max`` strings into ordered numeric ranges."""
    ranges = {}
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"range must be name:min:max, got {spec!r}")
        name, low, high = parts
        ranges[name] = (float(low), float(high))
    if not ranges:
        ranges.update(DEFAULT_PARAM_RANGES)
    return ranges


def fixed_to_helprop_options(fixed: Mapping[str, float]) -> list[str]:
    """Convert fixed ``name=value`` metadata to HelProp ``--name=value`` flags."""
    return [f"--{name}={value:.17g}" for name, value in fixed.items()]


def build_helprop_command(
    helprop: str | Path,
    output: str | Path,
    params: Mapping[str, float],
    etoa: str,
    elis: str,
    number: int,
    nthread: int,
    seed: int | None = None,
    fixed_options: Sequence[str] = (),
) -> list[str]:
    """Build a HelProp command that writes BSON matrix data with samples."""
    if number <= 0:
        raise ValueError("number must be positive")
    if nthread <= 0:
        raise ValueError("nthread must be positive")

    cmd = [
        str(helprop),
        "--iotype=BSON",
        "--sample",
        f"--etoa={etoa}",
        f"--elis={elis}",
        f"--number={number}",
        f"--nthread={nthread}",
    ]
    if seed is not None:
        cmd.append(f"--seed={seed}")
    cmd.extend(str(opt) for opt in fixed_options)
    for name, value in params.items():
        cmd.append(f"--{name}={value:.17g}")
    cmd.append(str(output))
    return cmd


def make_sample_runs(
    helprop: str | Path,
    outdir: str | Path,
    design: Sequence[Mapping[str, float]],
    etoa: str,
    elis: str,
    number: int,
    nthread: int,
    seed: int | None = None,
    fixed_options: Sequence[str] = (),
) -> list[HelPropSampleRun]:
    """Create run descriptors for a parameter design."""
    outdir = Path(outdir)
    runs = []
    width = max(4, len(str(max(len(design) - 1, 0))))
    for index, params in enumerate(design):
        run_seed = None if seed is None else seed + index * 1000003
        output = outdir / f"samples_{index:0{width}d}.bson"
        command = build_helprop_command(
            helprop=helprop,
            output=output,
            params=params,
            etoa=etoa,
            elis=elis,
            number=number,
            nthread=nthread,
            seed=run_seed,
            fixed_options=fixed_options,
        )
        runs.append(
            HelPropSampleRun(
                index=index,
                params={name: float(value) for name, value in params.items()},
                output=output,
                command=command,
            )
        )
    return runs


def write_manifest(path: str | Path, runs: Sequence[HelPropSampleRun]) -> None:
    """Write a CSV manifest of the generated HelProp runs."""
    import io

    path = Path(path)
    param_names = list(runs[0].params.keys()) if runs else []
    stream = io.StringIO()
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


def preflight_outputs(
    runs: Sequence[HelPropSampleRun],
    manifest: str | Path,
    dataset_out: str | Path | None = None,
) -> None:
    """Check duplicate final output paths before starting expensive HelProp runs."""
    paths = [run.output for run in runs]
    paths.append(Path(manifest))
    if dataset_out is not None:
        paths.append(Path(dataset_out))
    ensure_unique_outputs(paths)


def run_helprop_samples(
    runs: Sequence[HelPropSampleRun],
    timeout: float | None = None,
    dry_run: bool = False,
) -> None:
    """Execute HelProp sample commands."""
    for run in runs:
        run.output.parent.mkdir(parents=True, exist_ok=True)
        if dry_run:
            print(" ".join(run.command))
            continue
        print(f"[{run.index + 1}/{len(runs)}] {run.output}")
        final_output = run.output
        temporary_output = temp_output_path(final_output)
        command = list(run.command)
        command[-1] = str(temporary_output)
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"HelProp failed for run {run.index} with exit code "
                f"{result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        atomic_promote(temporary_output, final_output)


def consolidate_bson_outputs(
    paths: Sequence[str | Path],
    param_names: Sequence[str],
) -> TransitionDataset:
    """Load multiple BSON outputs and concatenate them into one dataset."""
    datasets = [load_bson_transitions(path, param_names=param_names) for path in paths]
    if not datasets:
        raise ValueError("no BSON paths were provided")

    params = {
        name: np.concatenate([dataset.params[name] for dataset in datasets])
        for name in param_names
    }
    return TransitionDataset(
        etoa=np.concatenate([dataset.etoa for dataset in datasets]),
        elis=np.concatenate([dataset.elis for dataset in datasets]),
        params=params,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run HelProp BSON sample jobs and build a transition dataset."
    )
    parser.add_argument("--helprop", default="./HelProp", help="Path to HelProp executable")
    parser.add_argument("--outdir", type=Path, default=Path("surrogate_samples"))
    parser.add_argument("--dataset-out", type=Path, default=Path("surrogate_samples/transitions.npz"))
    parser.add_argument("--manifest", type=Path, default=None, help="CSV manifest path")
    parser.add_argument("--n-runs", type=int, default=16, help="Number of parameter design points")
    parser.add_argument(
        "--learn",
        nargs="+",
        default=["D0", "m"],
        help="Learned HelProp parameter names; each must have a --range",
    )
    parser.add_argument(
        "--range",
        dest="ranges",
        action="append",
        default=[],
        help="Parameter range as name:min:max; repeat for multiple parameters",
    )
    parser.add_argument("--etoa", default="0.1,100,30", help="HelProp --etoa min,max,n")
    parser.add_argument("--elis", default="0.1,100,30", help="HelProp --elis min,max,n")
    parser.add_argument("--number", type=int, default=200, help="Particles per ETOA bin")
    parser.add_argument("--nthread", type=int, default=1, help="HelProp thread count")
    parser.add_argument("--seed", type=int, default=12345, help="Base seed for design and runs")
    parser.add_argument(
        "--fixed",
        action="append",
        default=[],
        help="Fixed HelProp parameter as name=value; repeat as needed",
    )
    parser.add_argument(
        "--fixed-option",
        action="append",
        default=[],
        help="Raw fixed HelProp option, e.g. --A=1; compatibility escape hatch",
    )
    parser.add_argument("--timeout", type=float, default=None, help="Timeout per HelProp run in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running HelProp")
    parser.add_argument(
        "--no-consolidate",
        action="store_true",
        help="Do not load BSON outputs into dataset-out after running",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ranges = parse_ranges(args.ranges)
    learned = tuple(args.learn)
    if args.ranges:
        missing_ranges = set(learned).difference(ranges)
        if missing_ranges:
            names = ", ".join(sorted(missing_ranges))
            raise ValueError(f"missing --range for learned parameters: {names}")
        extra_ranges = set(ranges).difference(learned)
        if extra_ranges:
            names = ", ".join(sorted(extra_ranges))
            raise ValueError(f"ranges supplied for non-learned parameters: {names}")
    else:
        learned = tuple(ranges.keys())

    fixed = parse_key_value_options(args.fixed)
    repeated = set(learned).intersection(fixed)
    if repeated:
        names = ", ".join(sorted(repeated))
        raise ValueError(f"parameters cannot be both learned and fixed: {names}")

    design = latin_hypercube(ranges, n_runs=args.n_runs, seed=args.seed)
    fixed_options = fixed_to_helprop_options(fixed) + list(args.fixed_option)
    runs = make_sample_runs(
        helprop=args.helprop,
        outdir=args.outdir,
        design=design,
        etoa=args.etoa,
        elis=args.elis,
        number=args.number,
        nthread=args.nthread,
        seed=args.seed,
        fixed_options=fixed_options,
    )

    manifest = args.manifest or (args.outdir / "manifest.csv")
    dataset_out = None if args.dry_run or args.no_consolidate else args.dataset_out
    preflight_outputs(runs, manifest, dataset_out=dataset_out)
    write_manifest(manifest, runs)
    print(f"manifest: {manifest}")

    run_helprop_samples(runs, timeout=args.timeout, dry_run=args.dry_run)
    if args.dry_run or args.no_consolidate:
        return 0

    dataset = consolidate_bson_outputs(
        [run.output for run in runs],
        param_names=learned,
    )
    dataset.save_npz(args.dataset_out)
    print(f"samples: {dataset.etoa.size}")
    print(f"dataset: {args.dataset_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
