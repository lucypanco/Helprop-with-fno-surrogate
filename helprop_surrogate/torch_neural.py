"""PyTorch/GPU conditional neural density surrogate for HelProp kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .kernel import DEFAULT_PARAM_NAMES, energy_bin_edges, fold_lis
from .neural import _as_1d_positive, _resolve_param_matrix, _theta_vector


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError(
            "TorchNeuralConditionalKernelSurrogate requires PyTorch. "
            "Install torch on the training server."
        ) from exc
    return torch, nn, F


@dataclass
class TorchNeuralConditionalKernelSurrogate:
    """GPU-capable PyTorch mixture-density kernel surrogate."""

    param_names: tuple[str, ...] = DEFAULT_PARAM_NAMES
    n_components: int = 6
    hidden_sizes: tuple[int, ...] = (128, 128)
    learning_rate: float = 1.0e-3
    epochs: int = 1000
    batch_size: int = 4096
    random_state: int = 123
    min_scale: float = 2.0e-3
    weight_decay: float = 1.0e-5
    device: str = "auto"
    verbose: bool = False

    def __post_init__(self) -> None:
        self.param_names = tuple(self.param_names)
        self.hidden_sizes = tuple(self.hidden_sizes)
        if not self.param_names:
            raise ValueError("param_names must not be empty")
        if self.n_components <= 0 or any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("n_components and hidden sizes must be positive")
        self._is_fit = False

    def fit(
        self,
        etoa: Sequence[float],
        elis: Sequence[float],
        params: Mapping[str, float | Sequence[float]] | np.ndarray,
    ) -> "TorchNeuralConditionalKernelSurrogate":
        torch, nn, F = _require_torch()
        torch.manual_seed(self.random_state)

        etoa_array = _as_1d_positive("etoa", etoa)
        elis_array = _as_1d_positive("elis", elis)
        if etoa_array.shape != elis_array.shape:
            raise ValueError("etoa and elis must have the same shape")
        param_matrix = _resolve_param_matrix(params, etoa_array.size, self.param_names)
        condition = self._build_condition_matrix(etoa_array, param_matrix)

        self.condition_center_ = condition.mean(axis=0)
        scale = condition.std(axis=0)
        self.condition_scale_ = np.where(scale > 0.0, scale, 1.0)
        x = ((condition - self.condition_center_) / self.condition_scale_).astype(np.float32)

        target = np.log(elis_array / etoa_array)
        self.target_center_ = float(target.mean())
        target_scale = float(target.std())
        self.target_scale_ = target_scale if target_scale > 0.0 else 1.0
        y = ((target - self.target_center_) / self.target_scale_).astype(np.float32).reshape(-1, 1)

        self.device_ = self._resolve_device(torch)
        self.model_ = _TorchMDN(
            input_size=x.shape[1],
            hidden_sizes=self.hidden_sizes,
            n_components=self.n_components,
            min_scale=self.min_scale,
            nn=nn,
            F=F,
        ).to(self.device_)

        x_tensor = torch.as_tensor(x, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
        self.optimizer_name_ = "AdamW"
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.model_.train()
        for epoch in range(self.epochs):
            losses = []
            for xb, yb in loader:
                xb = xb.to(self.device_)
                yb = yb.to(self.device_)
                optimizer.zero_grad(set_to_none=True)
                loss = self.model_.negative_log_likelihood(xb, yb)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            if self.verbose and (epoch + 1) % max(self.epochs // 10, 1) == 0:
                print(f"epoch {epoch + 1}/{self.epochs} nll={np.mean(losses):.6g}")

        self.model_.eval()
        self.model_state_ = {key: value.detach().cpu() for key, value in self.model_.state_dict().items()}
        self._is_fit = True
        return self

    def density(
        self,
        etoa: float,
        theta: Mapping[str, float] | Sequence[float],
        log_ratio: Sequence[float],
    ) -> np.ndarray:
        self._require_fit()
        torch, _, _ = _require_torch()
        model = self._model_for_inference(torch)
        x = self._condition_vector(etoa, theta).astype(np.float32)
        values = np.asarray(log_ratio, dtype=float)
        if np.any(~np.isfinite(values)):
            raise ValueError("log_ratio contains non-finite values")
        z = ((values - self.target_center_) / self.target_scale_).astype(np.float32).reshape(-1, 1)
        with torch.no_grad():
            x_tensor = torch.as_tensor(np.repeat(x.reshape(1, -1), z.shape[0], axis=0), device=self.device_)
            z_tensor = torch.as_tensor(z, device=self.device_)
            logp = model.log_prob(x_tensor, z_tensor).detach().cpu().numpy()
        return np.exp(logp) / self.target_scale_

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
        widths = np.diff(np.log(energy_bin_edges(elis_array)))
        matrix = np.empty((etoa_array.size, elis_array.size), dtype=float)
        for irow, etoa in enumerate(etoa_array):
            log_ratio = np.log(elis_array / etoa)
            row = np.maximum(self.density(etoa, theta, log_ratio) * widths, 0.0)
            total = row.sum()
            matrix[irow] = row / total if total > 0.0 and np.isfinite(total) else 1.0 / row.size
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

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("model_", None)
        return state

    def _model_for_inference(self, torch):
        if not hasattr(self, "model_"):
            _, nn, F = _require_torch()
            input_size = len(self.condition_center_)
            self.model_ = _TorchMDN(
                input_size=input_size,
                hidden_sizes=self.hidden_sizes,
                n_components=self.n_components,
                min_scale=self.min_scale,
                nn=nn,
                F=F,
            ).to(self.device_)
            self.model_.load_state_dict(self.model_state_)
            self.model_.eval()
        return self.model_

    def _resolve_device(self, torch):
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

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

    def _condition_vector(self, etoa: float, theta: Mapping[str, float] | Sequence[float]) -> np.ndarray:
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
            raise RuntimeError("TorchNeuralConditionalKernelSurrogate.fit must be called first")


class _TorchMDN:
    def __new__(cls, input_size, hidden_sizes, n_components, min_scale, nn, F):
        import torch

        class MDN(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                prev = input_size
                for size in hidden_sizes:
                    layers.append(nn.Linear(prev, size))
                    layers.append(nn.SiLU())
                    prev = size
                self.net = nn.Sequential(*layers)
                self.out = nn.Linear(prev, 3 * n_components)
                self.n_components = n_components
                self.min_scale = min_scale

            def forward(self, x):
                raw = self.out(self.net(x))
                logits, means, raw_scales = raw.chunk(3, dim=-1)
                scales = F.softplus(raw_scales) + self.min_scale
                return logits, means, scales

            def log_prob(self, x, y):
                logits, means, scales = self.forward(x)
                y = y.expand_as(means)
                log_weights = F.log_softmax(logits, dim=-1)
                log_normal = -0.5 * ((y - means) / scales) ** 2 - torch.log(scales) - 0.5 * np.log(2.0 * np.pi)
                return torch.logsumexp(log_weights + log_normal, dim=-1)

            def negative_log_likelihood(self, x, y):
                return -self.log_prob(x, y).mean()

        return MDN()
