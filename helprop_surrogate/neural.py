"""Lightweight neural conditional density surrogate for HelProp kernels.

This is a dependency-free mixture-density network.  It is intentionally small:
one hidden layer predicts Gaussian-mixture parameters for
``log(ELIS / ETOA)`` conditioned on ``log(ETOA)`` and learned HelProp
parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .kernel import DEFAULT_PARAM_NAMES, energy_bin_edges, fold_lis


def _as_1d_positive(name: str, values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain finite positive values")
    return array


def _logsumexp(values: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    max_value = np.max(values, axis=axis, keepdims=True)
    result = max_value + np.log(np.sum(np.exp(values - max_value), axis=axis, keepdims=True))
    if keepdims:
        return result
    return np.squeeze(result, axis=axis)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def _normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


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
            values = np.asarray(params[name], dtype=float)
            if values.ndim == 0:
                values = np.full(n_samples, float(values))
            if values.shape != (n_samples,):
                raise ValueError(f"parameter '{name}' must be scalar or length {n_samples}")
            columns.append(values)
        matrix = np.column_stack(columns)
    else:
        matrix = np.asarray(params, dtype=float)
        if matrix.ndim == 1:
            if matrix.size != len(param_names):
                raise ValueError("one-dimensional params must match param_names")
            matrix = np.tile(matrix, (n_samples, 1))
        if matrix.shape != (n_samples, len(param_names)):
            raise ValueError("params array shape does not match samples and param_names")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("params contain non-finite values")
    return matrix


def _theta_vector(theta: Mapping[str, float] | Sequence[float], param_names: Sequence[str]) -> np.ndarray:
    if isinstance(theta, Mapping):
        return np.asarray([float(theta[name]) for name in param_names], dtype=float)
    values = np.asarray(theta, dtype=float)
    if values.shape != (len(param_names),):
        raise ValueError(f"theta must have shape ({len(param_names)},)")
    return values


@dataclass
class NeuralConditionalKernelSurrogate:
    """Neural mixture-density surrogate with the same API as the KDE model."""

    param_names: tuple[str, ...] = DEFAULT_PARAM_NAMES
    n_components: int = 4
    hidden_size: int = 24
    learning_rate: float = 1.0e-2
    epochs: int = 300
    batch_size: int = 256
    random_state: int = 123
    min_scale: float = 5.0e-3
    verbose: bool = False

    def __post_init__(self) -> None:
        self.param_names = tuple(self.param_names)
        if not self.param_names:
            raise ValueError("param_names must not be empty")
        if self.n_components <= 0 or self.hidden_size <= 0:
            raise ValueError("n_components and hidden_size must be positive")
        self._is_fit = False

    def fit(
        self,
        etoa: Sequence[float],
        elis: Sequence[float],
        params: Mapping[str, float | Sequence[float]] | np.ndarray,
    ) -> "NeuralConditionalKernelSurrogate":
        etoa_array = _as_1d_positive("etoa", etoa)
        elis_array = _as_1d_positive("elis", elis)
        if etoa_array.shape != elis_array.shape:
            raise ValueError("etoa and elis must have the same shape")

        param_matrix = _resolve_param_matrix(params, etoa_array.size, self.param_names)
        condition = self._build_condition_matrix(etoa_array, param_matrix)
        self.condition_center_ = condition.mean(axis=0)
        condition_scale = condition.std(axis=0)
        self.condition_scale_ = np.where(condition_scale > 0.0, condition_scale, 1.0)
        x = (condition - self.condition_center_) / self.condition_scale_

        target = np.log(elis_array / etoa_array)
        self.target_center_ = float(target.mean())
        target_scale = float(target.std())
        self.target_scale_ = target_scale if target_scale > 0.0 else 1.0
        z = ((target - self.target_center_) / self.target_scale_).reshape(-1, 1)

        self._initialize_weights(x.shape[1])
        self._train(x, z)
        self._is_fit = True
        return self

    def density(
        self,
        etoa: float,
        theta: Mapping[str, float] | Sequence[float],
        log_ratio: Sequence[float],
    ) -> np.ndarray:
        self._require_fit()
        log_ratio_array = np.asarray(log_ratio, dtype=float)
        if np.any(~np.isfinite(log_ratio_array)):
            raise ValueError("log_ratio contains non-finite values")

        x = self._condition_vector(etoa, theta)
        weights, means, scales = self._mixture(x.reshape(1, -1))
        z = (log_ratio_array.reshape(-1, 1) - self.target_center_) / self.target_scale_
        standardized_density = np.sum(
            weights[0] * _normal_pdf((z - means[0]) / scales[0]) / scales[0],
            axis=1,
        )
        return standardized_density / self.target_scale_

    def matrix(
        self,
        etoa_grid: Sequence[float],
        elis_grid: Sequence[float],
        theta: Mapping[str, float] | Sequence[float],
    ) -> np.ndarray:
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
            log_ratio = np.log(elis_array / etoa)
            row = np.maximum(self.density(etoa, theta, log_ratio) * log_widths, 0.0)
            total = row.sum()
            if total <= 0.0 or not np.isfinite(total):
                row[:] = 1.0 / row.size
            else:
                row /= total
            matrix[irow] = row
        return matrix

    def predict_spectrum(
        self,
        etoa_grid: Sequence[float],
        elis_grid: Sequence[float],
        lis_flux: Sequence[float],
        theta: Mapping[str, float] | Sequence[float],
        A: int = 1,
    ) -> np.ndarray:
        matrix = self.matrix(etoa_grid, elis_grid, theta)
        return fold_lis(matrix, etoa_grid, elis_grid, lis_flux, A=A)

    def _initialize_weights(self, input_size: int) -> None:
        rng = np.random.default_rng(self.random_state)
        scale1 = np.sqrt(2.0 / max(input_size + self.hidden_size, 1))
        scale2 = np.sqrt(2.0 / max(self.hidden_size + 3 * self.n_components, 1))
        self.W1_ = rng.normal(0.0, scale1, size=(input_size, self.hidden_size))
        self.b1_ = np.zeros(self.hidden_size)
        self.W2_ = rng.normal(0.0, scale2, size=(self.hidden_size, 3 * self.n_components))
        self.b2_ = np.zeros(3 * self.n_components)

    def _train(self, x: np.ndarray, z: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        m = {name: np.zeros_like(value) for name, value in self._params().items()}
        v = {name: np.zeros_like(value) for name, value in self._params().items()}
        beta1, beta2, eps = 0.9, 0.999, 1.0e-8
        step = 0

        for epoch in range(self.epochs):
            order = rng.permutation(x.shape[0])
            for start in range(0, x.shape[0], self.batch_size):
                batch = order[start:start + self.batch_size]
                loss, grads = self._loss_and_grads(x[batch], z[batch])
                step += 1
                params = self._params()
                for name, param in params.items():
                    m[name] = beta1 * m[name] + (1.0 - beta1) * grads[name]
                    v[name] = beta2 * v[name] + (1.0 - beta2) * grads[name] * grads[name]
                    m_hat = m[name] / (1.0 - beta1 ** step)
                    v_hat = v[name] / (1.0 - beta2 ** step)
                    param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
            if self.verbose and (epoch + 1) % max(self.epochs // 10, 1) == 0:
                full_loss, _ = self._loss_and_grads(x, z)
                print(f"epoch {epoch + 1}/{self.epochs} nll={full_loss:.6g}")

    def _loss_and_grads(self, x: np.ndarray, z: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        n = x.shape[0]
        hidden_pre = x @ self.W1_ + self.b1_
        hidden = np.tanh(hidden_pre)
        output = hidden @ self.W2_ + self.b2_
        logits, means, raw_scales = np.split(output, 3, axis=1)
        log_scales = np.clip(raw_scales, np.log(self.min_scale), 4.0)
        scales = np.exp(log_scales)
        weights = _softmax(logits)

        normal_logp = -0.5 * ((z - means) / scales) ** 2 - log_scales - 0.5 * np.log(2.0 * np.pi)
        log_component = np.log(weights + 1.0e-300) + normal_logp
        log_prob = _logsumexp(log_component, axis=1)
        loss = float(-np.mean(log_prob))

        resp = np.exp(log_component - log_prob[:, None])
        d_logits = (weights - resp) / n
        d_means = resp * (means - z) / (scales * scales) / n
        d_log_scales = resp * (1.0 - ((z - means) / scales) ** 2) / n
        active_scale = (raw_scales >= np.log(self.min_scale)) & (raw_scales <= 4.0)
        d_raw_scales = d_log_scales * active_scale
        d_output = np.concatenate([d_logits, d_means, d_raw_scales], axis=1)

        dW2 = hidden.T @ d_output
        db2 = d_output.sum(axis=0)
        d_hidden = d_output @ self.W2_.T
        d_hidden_pre = d_hidden * (1.0 - hidden * hidden)
        dW1 = x.T @ d_hidden_pre
        db1 = d_hidden_pre.sum(axis=0)
        return loss, {"W1_": dW1, "b1_": db1, "W2_": dW2, "b2_": db2}

    def _mixture(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hidden = np.tanh(x @ self.W1_ + self.b1_)
        output = hidden @ self.W2_ + self.b2_
        logits, means, raw_scales = np.split(output, 3, axis=1)
        return _softmax(logits), means, np.exp(np.clip(raw_scales, np.log(self.min_scale), 4.0))

    def _params(self) -> dict[str, np.ndarray]:
        return {"W1_": self.W1_, "b1_": self.b1_, "W2_": self.W2_, "b2_": self.b2_}

    def _build_condition_matrix(self, etoa: np.ndarray, params: np.ndarray) -> np.ndarray:
        columns = [np.log(etoa)]
        for index, name in enumerate(self.param_names):
            column = params[:, index]
            if name.lower() in {"d0", "diffusion", "diffusion_coefficient"}:
                if np.any(column <= 0.0):
                    raise ValueError(f"parameter '{name}' must be positive")
                column = np.log10(column)
            columns.append(column)
        return np.column_stack(columns)

    def _condition_vector(
        self,
        etoa: float,
        theta: Mapping[str, float] | Sequence[float],
    ) -> np.ndarray:
        if not np.isfinite(etoa) or etoa <= 0.0:
            raise ValueError("etoa must be finite and positive")
        theta_values = _theta_vector(theta, self.param_names)
        condition = self._build_condition_matrix(
            np.asarray([etoa], dtype=float),
            theta_values.reshape(1, -1),
        )
        return ((condition - self.condition_center_) / self.condition_scale_)[0]

    def _require_fit(self) -> None:
        if not self._is_fit:
            raise RuntimeError("NeuralConditionalKernelSurrogate.fit must be called first")
