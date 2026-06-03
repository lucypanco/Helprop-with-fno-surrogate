"""Data loading helpers for HelProp surrogate training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .file_safety import prepare_output_path


DEFAULT_PARAM_NAMES = ("D0", "m")


@dataclass(frozen=True)
class TransitionDataset:
    """Particle-level transition samples used by the kernel surrogate."""

    etoa: np.ndarray
    elis: np.ndarray
    params: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        etoa = np.asarray(self.etoa, dtype=float)
        elis = np.asarray(self.elis, dtype=float)
        if etoa.ndim != 1 or elis.ndim != 1:
            raise ValueError("etoa and elis must be one-dimensional")
        if etoa.shape != elis.shape:
            raise ValueError("etoa and elis must have the same shape")
        if etoa.size == 0:
            raise ValueError("transition dataset is empty")
        if np.any(~np.isfinite(etoa)) or np.any(etoa <= 0.0):
            raise ValueError("etoa must contain finite positive values")
        if np.any(~np.isfinite(elis)) or np.any(elis <= 0.0):
            raise ValueError("elis must contain finite positive values")

        normalized = {}
        for name, values in self.params.items():
            array = np.asarray(values, dtype=float)
            if array.ndim == 0:
                array = np.full(etoa.size, float(array))
            if array.shape != etoa.shape:
                raise ValueError(f"parameter '{name}' must be scalar or length {etoa.size}")
            if np.any(~np.isfinite(array)):
                raise ValueError(f"parameter '{name}' contains non-finite values")
            normalized[name] = array

        object.__setattr__(self, "etoa", etoa)
        object.__setattr__(self, "elis", elis)
        object.__setattr__(self, "params", normalized)

    def select_params(self, param_names: Sequence[str] = DEFAULT_PARAM_NAMES) -> dict[str, np.ndarray]:
        """Return a parameter mapping containing only ``param_names``."""
        selected = {}
        for name in param_names:
            if name not in self.params:
                raise KeyError(f"dataset is missing parameter '{name}'")
            selected[name] = self.params[name]
        return selected

    def save_npz(self, path: str | Path) -> None:
        """Write the dataset to a compact NumPy archive."""
        path = prepare_output_path(path)
        payload = {
            "etoa": self.etoa,
            "elis": self.elis,
            "param_names": np.asarray(list(self.params.keys())),
        }
        for name, values in self.params.items():
            payload[f"param_{name}"] = values
        np.savez(path, **payload)


def load_npz_transitions(path: str | Path) -> TransitionDataset:
    """Load particle transitions from ``TransitionDataset.save_npz`` output."""
    with np.load(path, allow_pickle=False) as data:
        param_names = [str(name) for name in data["param_names"]]
        params = {name: np.asarray(data[f"param_{name}"], dtype=float) for name in param_names}
        return TransitionDataset(
            etoa=np.asarray(data["etoa"], dtype=float),
            elis=np.asarray(data["elis"], dtype=float),
            params=params,
        )


def load_bson_transitions(
    path: str | Path,
    param_names: Sequence[str] = DEFAULT_PARAM_NAMES,
) -> TransitionDataset:
    """Load HelProp BSON ``--sample`` transition records.

    HelProp writes one BSON document per run when ``--iotype=BSON --sample`` is
    used.  Each document includes scalar ``params`` plus particle arrays
    ``etoa`` and ``elis``.  This loader concatenates all documents in the file.

    The optional dependency is PyMongo's ``bson`` module.  If it is not
    installed, convert data to ``.npz`` with an environment that has PyMongo or
    install it in this environment.
    """
    try:
        from bson import BSON
    except ImportError as exc:
        raise ImportError(
            "load_bson_transitions requires PyMongo's bson module; "
            "install pymongo or use load_npz_transitions"
        ) from exc

    filename = Path(path)
    chunks = []
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
            chunks.append(BSON(header + body).decode())

    if not chunks:
        raise ValueError(f"no BSON documents found in {filename}")
    return _dataset_from_bson_docs(chunks, param_names)


def _dataset_from_bson_docs(
    docs: Sequence[Mapping[str, object]],
    param_names: Sequence[str],
) -> TransitionDataset:
    etoa_parts = []
    elis_parts = []
    params = {name: [] for name in param_names}

    for index, doc in enumerate(docs):
        if "etoa" not in doc or "elis" not in doc:
            raise ValueError(f"BSON document {index} does not contain sample arrays")
        etoa = np.asarray(doc["etoa"], dtype=float)
        elis = np.asarray(doc["elis"], dtype=float)
        if etoa.shape != elis.shape:
            raise ValueError(f"BSON document {index} has mismatched etoa/elis arrays")

        raw_params = doc.get("params")
        if not isinstance(raw_params, Mapping):
            raise ValueError(f"BSON document {index} does not contain params")

        etoa_parts.append(etoa)
        elis_parts.append(elis)
        for name in param_names:
            if name not in raw_params:
                raise KeyError(f"BSON document {index} missing parameter '{name}'")
            params[name].append(np.full(etoa.size, float(raw_params[name])))

    return TransitionDataset(
        etoa=np.concatenate(etoa_parts),
        elis=np.concatenate(elis_parts),
        params={name: np.concatenate(parts) for name, parts in params.items()},
    )
