"""Conditional kernel-density surrogate for HelProp transition samples.

This module models the reusable HelProp object:

    p(log(E_LIS / E_TOA) | log(E_TOA), theta)

from particle-level samples.  It is intentionally dependency-light so the
surrogate can be trained and exercised before a neural spline-flow stack is
introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


DEFAULT_PARAM_NAMES = ("D0", "m")


def _as_1d_positive(name: str, values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.any(~np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"{name} must contain finite positive values")
    return array


def _normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _standardize(
    values: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return (values - center) / scale


def energy_bin_edges(centers: Sequence[float]) -> np.ndarray:
    """Return logarithmic bin edges for strictly positive energy centers."""
    centers = _as_1d_positive("centers", centers)
    if centers.size == 1:
        factor = np.sqrt(10.0)
        return np.array([centers[0] / factor, centers[0] * factor])

    log_centers = np.log(centers)
    if np.any(np.diff(log_centers) <= 0):
        raise ValueError("centers must be strictly increasing")

    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = np.sqrt(centers[:-1] * centers[1:])
    edges[0] = centers[0] * np.sqrt(centers[0] / centers[1])
    edges[-1] = centers[-1] * np.sqrt(centers[-1] / centers[-2])
    return edges


def _resolve_param_matrix(
    params: Mapping[str, float | Sequence[float]] | np.ndarray,
    n_samples: int,
    param_names: Sequence[str],
) -> np.ndarray:
    if isinstance(params, Mapping):
        columns = []
        for name in param_names:
            if name not in params:
                raise KeyError(f"missing parameter '{name}'")
            value = np.asarray(params[name], dtype=float)
            if value.ndim == 0:
                value = np.full(n_samples, float(value))
            if value.shape != (n_samples,):
                raise ValueError(
                    f"parameter '{name}' must be scalar or length {n_samples}"
                )
            columns.append(value)
        matrix = np.column_stack(columns)
    else:
        matrix = np.asarray(params, dtype=float)
        if matrix.ndim == 1:
            if matrix.size != len(param_names):
                raise ValueError("one-dimensional params must match param_names")
            matrix = np.tile(matrix, (n_samples, 1))
        if matrix.shape != (n_samples, len(param_names)):
            raise ValueError(
                "params array must have shape "
                f"({n_samples}, {len(param_names)})"
            )

    if np.any(~np.isfinite(matrix)):
        raise ValueError("params contain non-finite values")
    return matrix


def _theta_vector(
    theta: Mapping[str, float] | Sequence[float],
    param_names: Sequence[str],
) -> np.ndarray:
    if isinstance(theta, Mapping):
        try:
            values = [float(theta[name]) for name in param_names]
        except KeyError as exc:
            raise KeyError(f"missing theta parameter '{exc.args[0]}'") from exc
        return np.asarray(values, dtype=float)

    values = np.asarray(theta, dtype=float)
    if values.shape != (len(param_names),):
        raise ValueError(f"theta must have shape ({len(param_names)},)")
    return values


@dataclass
class ConditionalKernelSurrogate:
    """Nadaraya-Watson conditional KDE for HelProp transition kernels.

    The model smooths over standardized conditioning variables
    ``[log(ETOA), transformed theta...]`` and returns a conditional density in
    ``y = log(ELIS / ETOA)``.  Matrix rows are produced by evaluating this
    density at LIS-bin log-ratio centers and normalizing over the requested
    grid, which enforces non-negative rows that sum to one.
    """

    param_names: tuple[str, ...] = DEFAULT_PARAM_NAMES
    condition_bandwidth: float | Sequence[float] = 0.6
    target_bandwidth: float | None = None
    min_effective_samples: float = 1.0e-12

    def __post_init__(self) -> None:
        self.param_names = tuple(self.param_names)
        if not self.param_names:
            raise ValueError("param_names must not be empty")
        self._is_fit = False

    def fit(
        self,
        etoa: Sequence[float],
        elis: Sequence[float],
        params: Mapping[str, float | Sequence[float]] | np.ndarray,
    ) -> "ConditionalKernelSurrogate":
        """Fit the surrogate from particle transitions.

        Parameters
        ----------
        etoa, elis
            Positive TOA and LIS kinetic energies for each simulated particle.
        params
            Either a mapping from parameter name to scalar/vector, or an array
            with columns ordered as ``param_names``.
        """
        etoa_array = _as_1d_positive("etoa", etoa)
        elis_array = _as_1d_positive("elis", elis)
        if etoa_array.shape != elis_array.shape:
            raise ValueError("etoa and elis must have the same shape")

        n_samples = etoa_array.size
        param_matrix = _resolve_param_matrix(params, n_samples, self.param_names)
        condition = self._build_condition_matrix(etoa_array, param_matrix)

        self.etoa_train_ = etoa_array
        self.elis_train_ = elis_array
        self.params_train_ = param_matrix
        self.condition_center_ = condition.mean(axis=0)
        scale = condition.std(axis=0, ddof=0)
        self.condition_scale_ = np.where(scale > 0.0, scale, 1.0)
        self.condition_train_ = _standardize(
            condition, self.condition_center_, self.condition_scale_
        )
        self.target_train_ = np.log(elis_array / etoa_array)

        target_scale = self.target_train_.std(ddof=1) if n_samples > 1 else 0.0
        if self.target_bandwidth is None:
            if target_scale <= 0.0:
                self.target_bandwidth_ = 0.1
            else:
                self.target_bandwidth_ = 1.06 * target_scale * n_samples ** (-1.0 / 5.0)
                self.target_bandwidth_ = max(float(self.target_bandwidth_), 1.0e-3)
        else:
            self.target_bandwidth_ = float(self.target_bandwidth)
            if self.target_bandwidth_ <= 0.0:
                raise ValueError("target_bandwidth must be positive")

        bandwidth = np.asarray(self.condition_bandwidth, dtype=float)
        if bandwidth.ndim == 0:
            bandwidth = np.full(self.condition_train_.shape[1], float(bandwidth))
        if bandwidth.shape != (self.condition_train_.shape[1],):
            raise ValueError(
                "condition_bandwidth must be scalar or match condition dimension"
            )
        if np.any(~np.isfinite(bandwidth)) or np.any(bandwidth <= 0.0):
            raise ValueError("condition_bandwidth must be finite and positive")
        self.condition_bandwidth_ = bandwidth

        self._is_fit = True
        return self

    def density(
        self,
        etoa: float,
        theta: Mapping[str, float] | Sequence[float],
        log_ratio: Sequence[float],
    ) -> np.ndarray:
        """Evaluate ``p(log(ELIS / ETOA) | log(ETOA), theta)``."""
        self._require_fit()
        log_ratio_array = np.asarray(log_ratio, dtype=float)
        if np.any(~np.isfinite(log_ratio_array)):
            raise ValueError("log_ratio contains non-finite values")

        weights = self._condition_weights(etoa, theta)
        if weights.sum() <= self.min_effective_samples:
            return np.zeros_like(log_ratio_array, dtype=float)

        z = (log_ratio_array[..., None] - self.target_train_) / self.target_bandwidth_
        density = np.sum(weights * _normal_pdf(z), axis=-1)
        density /= weights.sum() * self.target_bandwidth_
        return density

    def matrix(
        self,
        etoa_grid: Sequence[float],
        elis_grid: Sequence[float],
        theta: Mapping[str, float] | Sequence[float],
    ) -> np.ndarray:
        """Construct a row-normalized transfer matrix on an energy grid."""
        self._require_fit()
        etoa_array = _as_1d_positive("etoa_grid", etoa_grid)
        elis_array = _as_1d_positive("elis_grid", elis_grid)
        if np.any(np.diff(etoa_array) <= 0.0):
            raise ValueError("etoa_grid must be strictly increasing")
        if np.any(np.diff(elis_array) <= 0.0):
            raise ValueError("elis_grid must be strictly increasing")

        edges = energy_bin_edges(elis_array)
        log_widths = np.diff(np.log(edges))
        matrix = np.empty((etoa_array.size, elis_array.size), dtype=float)

        for irow, etoa in enumerate(etoa_array):
            log_ratio_centers = np.log(elis_array / etoa)
            row = self.density(etoa, theta, log_ratio_centers) * log_widths
            row = np.maximum(row, 0.0)
            row_sum = row.sum()
            if row_sum <= 0.0 or not np.isfinite(row_sum):
                row = self._nearest_empirical_row(etoa, theta, elis_array)
                row_sum = row.sum()
            matrix[irow] = row / row_sum

        return matrix

    def predict_spectrum(
        self,
        etoa_grid: Sequence[float],
        elis_grid: Sequence[float],
        lis_flux: Sequence[float],
        theta: Mapping[str, float] | Sequence[float],
        A: int = 1,
    ) -> np.ndarray:
        """Predict the modulated TOA spectrum by folding the kernel with LIS."""
        matrix = self.matrix(etoa_grid, elis_grid, theta)
        return fold_lis(matrix, etoa_grid, elis_grid, lis_flux, A=A)

    def _build_condition_matrix(
        self,
        etoa: np.ndarray,
        params: np.ndarray,
    ) -> np.ndarray:
        columns = [np.log(etoa)]
        for index, name in enumerate(self.param_names):
            column = params[:, index]
            if name.lower() in {"d0", "diffusion", "diffusion_coefficient"}:
                if np.any(column <= 0.0):
                    raise ValueError(f"parameter '{name}' must be positive")
                column = np.log10(column)
            columns.append(column)
        return np.column_stack(columns)

    def _condition_weights(
        self,
        etoa: float,
        theta: Mapping[str, float] | Sequence[float],
    ) -> np.ndarray:
        if not np.isfinite(etoa) or etoa <= 0.0:
            raise ValueError("etoa must be finite and positive")
        theta_array = _theta_vector(theta, self.param_names)
        condition = self._build_condition_matrix(
            np.asarray([etoa], dtype=float),
            theta_array.reshape(1, -1),
        )
        standardized = _standardize(
            condition, self.condition_center_, self.condition_scale_
        )[0]
        delta = (self.condition_train_ - standardized) / self.condition_bandwidth_
        log_weights = -0.5 * np.sum(delta * delta, axis=1)
        log_weights -= np.max(log_weights)
        return np.exp(log_weights)

    def _nearest_empirical_row(
        self,
        etoa: float,
        theta: Mapping[str, float] | Sequence[float],
        elis_grid: np.ndarray,
    ) -> np.ndarray:
        weights = self._condition_weights(etoa, theta)
        index = int(np.argmax(weights))
        edges = energy_bin_edges(elis_grid)
        row, _ = np.histogram([self.elis_train_[index]], bins=edges)
        row = row.astype(float)
        if row.sum() <= 0.0:
            nearest = int(np.argmin(np.abs(np.log(elis_grid / self.elis_train_[index]))))
            row[nearest] = 1.0
        return row

    def _require_fit(self) -> None:
        if not self._is_fit:
            raise RuntimeError("ConditionalKernelSurrogate.fit must be called first")


def ekin_to_momentum(ekin: Sequence[float], A: int = 1) -> np.ndarray:
    """Convert kinetic energy per nucleon to particle momentum.

    This matches the helper in ``src/HelProp.cc`` for proton mass units.
    """
    ekin_array = _as_1d_positive("ekin", ekin)
    if A <= 0:
        raise ValueError("A must be positive")
    m_proton = 0.938272
    return np.sqrt(ekin_array * (ekin_array + 2.0 * m_proton)) * A


def fold_lis(
    matrix: np.ndarray,
    etoa_grid: Sequence[float],
    elis_grid: Sequence[float],
    lis_flux: Sequence[float],
    A: int = 1,
) -> np.ndarray:
    """Fold a transfer matrix with LIS flux using the HelProp equation."""
    matrix = np.asarray(matrix, dtype=float)
    etoa_array = _as_1d_positive("etoa_grid", etoa_grid)
    elis_array = _as_1d_positive("elis_grid", elis_grid)
    lis_array = _as_1d_positive("lis_flux", lis_flux)

    if matrix.shape != (etoa_array.size, elis_array.size):
        raise ValueError("matrix shape must be (len(etoa_grid), len(elis_grid))")
    if lis_array.shape != elis_array.shape:
        raise ValueError("lis_flux must match elis_grid")
    if np.any(matrix < 0.0) or np.any(~np.isfinite(matrix)):
        raise ValueError("matrix must be finite and non-negative")

    p_lis = ekin_to_momentum(elis_array, A=A)
    p_toa = ekin_to_momentum(etoa_array, A=A)
    weighted_lis = lis_array / (p_lis * p_lis)
    return (matrix @ weighted_lis) * (p_toa * p_toa)
