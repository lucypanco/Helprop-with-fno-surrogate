"""Rebuild a HelProp matrix NPZ dataset from existing BSON matrix files."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helprop_surrogate.matrix_data import (
    consolidate_bson_matrices,
    split_indices,
    write_split_files,
)


def expand_paths(inputs: Sequence[str]) -> list[Path]:
    """Expand files, directories, and glob patterns into BSON file paths."""
    paths: list[Path] = []
    for item in inputs:
        matches = [Path(match) for match in glob.glob(item)]
        candidates = matches if matches else [Path(item)]
        for candidate in candidates:
            if candidate.is_dir():
                paths.extend(sorted(candidate.glob("*.bson")))
            else:
                paths.append(candidate)
    return sorted(dict.fromkeys(paths))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild matrices.npz and split index files from existing BSON matrices."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="BSON files, directories, or glob patterns such as 'surrogate_runs/run_0001/data/*.bson'",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output NPZ path. Defaults to matrices.npz beside the first input file.",
    )
    parser.add_argument("--learn", nargs="+", default=["D0", "m"], help="Learned parameter names")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args(argv)

    paths = expand_paths(args.paths)
    if not paths:
        print("no BSON files matched")
        return 2

    missing = [path for path in paths if not path.exists()]
    if missing:
        for path in missing:
            print(f"missing BSON file: {path}")
        return 2

    output = args.out or (paths[0].parent / "matrices.npz")
    dataset = consolidate_bson_matrices(paths, args.learn)
    dataset.save_npz(output)

    train_idx, val_idx, test_idx = split_indices(
        dataset.theta.shape[0],
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    write_split_files(output.parent, train_idx, val_idx, test_idx)

    print(f"dataset: {output}")
    print(f"samples: {dataset.theta.shape[0]}")
    print(f"split: train={train_idx.size} val={val_idx.size} test={test_idx.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
