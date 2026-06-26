"""Scan HelProp BSON transfer-matrix files for invalid matrix values."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def expand_paths(inputs: Sequence[str]) -> list[Path]:
    """Expand files, directories, and glob patterns into BSON file paths."""
    paths: list[Path] = []
    for item in inputs:
        matches = [Path(match) for match in glob.glob(item)]
        candidates = matches if matches else [Path(item)]
        for candidate in candidates:
            if candidate.is_dir():
                paths.extend(sorted(candidate.rglob("*.bson")))
            else:
                paths.append(candidate)
    return sorted(dict.fromkeys(paths))


def read_bson_documents(path: Path) -> list[Mapping[str, object]]:
    """Read concatenated BSON documents from one HelProp output file."""
    try:
        from bson import BSON
    except ImportError as exc:
        raise ImportError("this checker requires pymongo's bson module") from exc

    docs = []
    with path.open("rb") as stream:
        while True:
            header = stream.read(4)
            if not header:
                break
            if len(header) != 4:
                raise ValueError("truncated BSON size header")
            size = int.from_bytes(header, byteorder="little", signed=True)
            if size < 5:
                raise ValueError(f"invalid BSON document size {size}")
            body = stream.read(size - 4)
            if len(body) != size - 4:
                raise ValueError("truncated BSON document")
            docs.append(BSON(header + body).decode())
    return docs


def check_document(path: Path, index: int, doc: Mapping[str, object]) -> list[str]:
    """Return validation messages for one BSON matrix document."""
    messages = []
    matrix = np.asarray(doc.get("M"), dtype=float)
    etoa = np.asarray(doc.get("ETOA"), dtype=float)
    elis = np.asarray(doc.get("ELIS"), dtype=float)

    if matrix.ndim != 2:
        return [f"doc {index}: M is not 2D, shape={matrix.shape}"]
    if matrix.shape != (etoa.size, elis.size):
        messages.append(
            f"doc {index}: M shape {matrix.shape} does not match "
            f"ETOA x ELIS {(etoa.size, elis.size)}"
        )

    nonfinite = ~np.isfinite(matrix)
    if np.any(nonfinite):
        rows, cols = np.where(nonfinite)
        messages.append(
            f"doc {index}: {rows.size} non-finite M values; "
            f"first at row={int(rows[0])}, col={int(cols[0])}, value={matrix[rows[0], cols[0]]}"
        )

    negative = matrix < 0.0
    if np.any(negative):
        rows, cols = np.where(negative)
        messages.append(
            f"doc {index}: {rows.size} negative M values; "
            f"first at row={int(rows[0])}, col={int(cols[0])}, value={matrix[rows[0], cols[0]]}"
        )

    row_sums = matrix.sum(axis=1)
    bad_rows = np.where((row_sums <= 0.0) | ~np.isfinite(row_sums))[0]
    if bad_rows.size:
        first = int(bad_rows[0])
        messages.append(
            f"doc {index}: {bad_rows.size} rows have non-positive or non-finite sums; "
            f"first row={first}, row_sum={row_sums[first]}"
        )

    if np.any(~np.isfinite(etoa)) or np.any(etoa <= 0.0):
        messages.append(f"doc {index}: ETOA contains non-finite or non-positive values")
    if np.any(~np.isfinite(elis)) or np.any(elis <= 0.0):
        messages.append(f"doc {index}: ELIS contains non-finite or non-positive values")
    if etoa.ndim != 1 or elis.ndim != 1:
        messages.append(f"doc {index}: ETOA/ELIS must be one-dimensional")
    elif np.any(np.diff(etoa) <= 0.0) or np.any(np.diff(elis) <= 0.0):
        messages.append(f"doc {index}: ETOA/ELIS grids are not strictly increasing")

    params = doc.get("params", {})
    if messages and params:
        messages.append(f"doc {index}: params={params}")
    return messages


def check_file(path: Path) -> list[str]:
    """Return validation messages for one BSON file."""
    try:
        docs = read_bson_documents(path)
    except Exception as exc:
        return [f"could not read BSON: {exc}"]

    if not docs:
        return ["no BSON documents found"]

    messages = []
    for index, doc in enumerate(docs):
        doc_messages = check_document(path, index, doc)
        if doc_messages:
            messages.extend(doc_messages)
    return messages


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Find invalid HelProp BSON transfer-matrix files."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="BSON files, directories, or glob patterns such as 'surrogate_runs/run_0001/data/*.bson'",
    )
    parser.add_argument("--stop-first", action="store_true", help="Stop after the first invalid file")
    args = parser.parse_args(argv)

    paths = expand_paths(args.paths)
    if not paths:
        print("no BSON files matched")
        return 2

    invalid = 0
    for ordinal, path in enumerate(paths, start=1):
        messages = check_file(path)
        if messages:
            invalid += 1
            print(f"INVALID [{ordinal}/{len(paths)}] {path}")
            for message in messages:
                print(f"  - {message}")
            if args.stop_first:
                break

    checked = ordinal if paths else 0
    print(f"checked={checked} invalid={invalid}")
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
