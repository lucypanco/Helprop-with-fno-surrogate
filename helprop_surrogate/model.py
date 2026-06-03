"""Saved surrogate model wrapper with HelProp-compatible parameter metadata."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .file_safety import prepare_output_path
from .kernel import ConditionalKernelSurrogate
from .neural import NeuralConditionalKernelSurrogate
from .torch_neural import TorchNeuralConditionalKernelSurrogate
from .fno.model import TorchFNOTransferMatrixSurrogate


def parse_key_value_options(items: Sequence[str]) -> dict[str, float]:
    """Parse ``name=value`` options while preserving HelProp parameter names."""
    parsed: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected name=value, got {item!r}")
        name, value = item.split("=", 1)
        if not name:
            raise ValueError(f"empty option name in {item!r}")
        parsed[name] = float(value)
    return parsed


def parse_range_options(items: Sequence[str]) -> dict[str, tuple[float, float]]:
    """Parse ``name:min:max`` range options."""
    parsed: dict[str, tuple[float, float]] = {}
    for item in items:
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"expected name:min:max, got {item!r}")
        name, low, high = parts
        low_value = float(low)
        high_value = float(high)
        if high_value <= low_value:
            raise ValueError(f"range upper bound must exceed lower bound for {name}")
        parsed[name] = (low_value, high_value)
    return parsed


@dataclass
class HelPropKernelModel:
    """Trained kernel plus metadata that hides fixed parameters at runtime."""

    kernel: (
        ConditionalKernelSurrogate
        | NeuralConditionalKernelSurrogate
        | TorchNeuralConditionalKernelSurrogate
        | TorchFNOTransferMatrixSurrogate
    )
    learned: tuple[str, ...]
    fixed: dict[str, float] = field(default_factory=dict)
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    etoa_grid: tuple[float, ...] | None = None
    elis_grid: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        self.learned = tuple(self.learned)
        if not self.learned:
            raise ValueError("learned parameter list must not be empty")
        repeated = set(self.learned).intersection(self.fixed)
        if repeated:
            names = ", ".join(sorted(repeated))
            raise ValueError(f"parameters cannot be both learned and fixed: {names}")
        if tuple(self.kernel.param_names) != self.learned:
            raise ValueError("kernel param_names must match learned parameters")

    def theta_from_options(
        self,
        options: Mapping[str, float],
        *,
        strict_ranges: bool = False,
    ) -> dict[str, float]:
        """Validate runtime options and return learned theta mapping.

        Runtime options must contain exactly the learned parameters.  Fixed
        parameters are stored in the model and intentionally rejected here.
        """
        option_names = set(options)
        learned_names = set(self.learned)
        fixed_names = set(self.fixed)

        fixed_supplied = option_names.intersection(fixed_names)
        if fixed_supplied:
            names = ", ".join(sorted(fixed_supplied))
            raise ValueError(f"fixed parameters are baked into the model: {names}")

        missing = learned_names.difference(option_names)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"missing learned parameters: {names}")

        unknown = option_names.difference(learned_names)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown surrogate parameters: {names}")

        theta = {name: float(options[name]) for name in self.learned}
        for name, value in theta.items():
            if not np.isfinite(value):
                raise ValueError(f"parameter {name} is not finite")
            if name in self.ranges:
                low, high = self.ranges[name]
                outside = value < low or value > high
                if outside and strict_ranges:
                    raise ValueError(
                        f"parameter {name}={value} is outside training range "
                        f"[{low}, {high}]"
                    )
        return theta

    def matrix(
        self,
        options: Mapping[str, float],
        etoa_grid: Sequence[float] | None = None,
        elis_grid: Sequence[float] | None = None,
        *,
        strict_ranges: bool = False,
    ) -> np.ndarray:
        theta = self.theta_from_options(options, strict_ranges=strict_ranges)
        etoa = _resolve_grid("etoa", etoa_grid, self.etoa_grid)
        elis = _resolve_grid("elis", elis_grid, self.elis_grid)
        return self.kernel.matrix(etoa, elis, theta)

    def predict_spectrum(
        self,
        options: Mapping[str, float],
        lis_flux: Sequence[float],
        etoa_grid: Sequence[float] | None = None,
        elis_grid: Sequence[float] | None = None,
        *,
        strict_ranges: bool = False,
    ) -> np.ndarray:
        theta = self.theta_from_options(options, strict_ranges=strict_ranges)
        etoa = _resolve_grid("etoa", etoa_grid, self.etoa_grid)
        elis = _resolve_grid("elis", elis_grid, self.elis_grid)
        A = int(self.fixed.get("A", 1))
        return self.kernel.predict_spectrum(etoa, elis, lis_flux, theta, A=A)

    def save(self, path: str | Path) -> None:
        path = prepare_output_path(path)
        with path.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: str | Path) -> HelPropKernelModel:
    with Path(path).open("rb") as stream:
        model = pickle.load(stream)
    if isinstance(model, HelPropKernelModel):
        return model
    if isinstance(
        model,
        (
            ConditionalKernelSurrogate,
            NeuralConditionalKernelSurrogate,
            TorchNeuralConditionalKernelSurrogate,
            TorchFNOTransferMatrixSurrogate,
        ),
    ):
        return HelPropKernelModel(kernel=model, learned=tuple(model.param_names))
    raise TypeError(f"unsupported surrogate model type: {type(model).__name__}")


def _resolve_grid(
    name: str,
    supplied: Sequence[float] | None,
    stored: tuple[float, ...] | None,
) -> np.ndarray:
    if supplied is not None:
        return np.asarray(supplied, dtype=float)
    if stored is not None:
        return np.asarray(stored, dtype=float)
    raise ValueError(f"{name}_grid must be supplied; this model has no stored grid")
