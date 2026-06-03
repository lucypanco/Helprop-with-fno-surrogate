"""File-output safety helpers for surrogate scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def prepare_output_path(path: str | Path) -> Path:
    """Create the output directory and return a normalized path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_unique_outputs(paths: Iterable[str | Path]) -> None:
    """Reject duplicate final output paths in a single command."""
    seen = set()
    for item in paths:
        path = Path(item)
        resolved = path.resolve()
        if resolved in seen:
            raise ValueError(f"duplicate output path in one run: {path}")
        seen.add(resolved)
        prepare_output_path(path)


def atomic_replace_bytes(path: str | Path, payload: bytes) -> None:
    path = prepare_output_path(path)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_bytes(payload)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_replace_text(path: str | Path, text: str) -> None:
    atomic_replace_bytes(path, text.encode("utf-8"))


def temp_output_path(path: str | Path) -> Path:
    """Return a sibling temp path for an external writer."""
    path = prepare_output_path(path)
    return path.with_name(f"{path.name}.tmp.{os.getpid()}")


def atomic_promote(temp_path: str | Path, final_path: str | Path) -> None:
    """Atomically promote a completed temp output to its final path."""
    temp = Path(temp_path)
    final = prepare_output_path(final_path)
    os.replace(temp, final)
