"""PyTorch Fourier Neural Operator surrogate for HelProp transfer matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..kernel import DEFAULT_PARAM_NAMES, ekin_to_momentum, fold_lis
from ..neural import _as_1d_positive, _theta_vector
from .convergence import (
    DEFAULT_EARLY_STOPPING_MIN_DELTA,
    DEFAULT_EARLY_STOPPING_MIN_EPOCHS,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    ConvergenceMonitor,
)

DEFAULT_MATRIX_PROBABILITY_LOSS_WEIGHT = 1.0
DEFAULT_MATRIX_CROSS_ENTROPY_WEIGHT = 1.0
DEFAULT_SPECTRUM_LOSS_WEIGHT = 1.0
DEFAULT_SPECTRUM_MAX_ERROR_PERCENT = 1.0
DEFAULT_SPECTRUM_MAX_ERROR_TEMPERATURE_PERCENT = 0.01
DEFAULT_SPECTRUM_HUBER_DELTA_PERCENT = 0.1
DEFAULT_SPECTRUM_TOP_K = 8
DEFAULT_BOUNDARY_PADDING = 8
DEFAULT_LR_SCHEDULER = "plateau"
DEFAULT_LR_SCHEDULER_FACTOR = 0.3
DEFAULT_LR_SCHEDULER_PATIENCE = 20
DEFAULT_LR_SCHEDULER_COOLDOWN = 5
DEFAULT_LR_SCHEDULER_MIN_LR = 1.0e-6


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError(
            "TorchFNOTransferMatrixSurrogate requires PyTorch. "
            "Install torch on the training server."
        ) from exc
    return torch, nn, F


def _piecewise_bin_error_loss(relative_error, threshold, temperature, huber_delta, torch):
    """Huber-penalize per-bin relative error above a threshold without bin dilution."""
    violation = temperature * torch.nn.functional.softplus(
        (relative_error - threshold) / temperature
    )
    absolute_violation = violation.abs()
    quadratic = 0.5 * violation * violation / huber_delta
    linear = absolute_violation - 0.5 * huber_delta
    huber_violation = torch.where(
        absolute_violation <= huber_delta,
        quadratic,
        linear,
    )
    return relative_error.shape[-1] * huber_violation.mean(dim=-1)


@dataclass
class TorchFNOTransferMatrixSurrogate:
    """Fixed-grid 2D FNO mapping HelProp parameters to transfer matrices."""

    param_names: tuple[str, ...] = DEFAULT_PARAM_NAMES
    width: int = 32
    modes_etoa: int = 10
    modes_elis: int = 10
    n_layers: int = 4
    projection_size: int = 64
    boundary_padding: int = DEFAULT_BOUNDARY_PADDING
    dropout: float = 0.05
    learning_rate: float = 1.0e-3
    epochs: int = 300
    batch_size: int = 16
    weight_decay: float = 1.0e-4
    random_state: int = 123
    device: str = "auto"
    verbose: bool = False
    matrix_cross_entropy_weight: float = DEFAULT_MATRIX_CROSS_ENTROPY_WEIGHT
    matrix_probability_loss_weight: float = DEFAULT_MATRIX_PROBABILITY_LOSS_WEIGHT
    spectrum_loss_weight: float = DEFAULT_SPECTRUM_LOSS_WEIGHT
    spectrum_max_error_percent: float = DEFAULT_SPECTRUM_MAX_ERROR_PERCENT
    spectrum_max_error_temperature_percent: float = DEFAULT_SPECTRUM_MAX_ERROR_TEMPERATURE_PERCENT
    spectrum_huber_delta_percent: float = DEFAULT_SPECTRUM_HUBER_DELTA_PERCENT
    spectrum_top_k: int = DEFAULT_SPECTRUM_TOP_K
    lr_scheduler: str = DEFAULT_LR_SCHEDULER
    lr_scheduler_factor: float = DEFAULT_LR_SCHEDULER_FACTOR
    lr_scheduler_patience: int = DEFAULT_LR_SCHEDULER_PATIENCE
    lr_scheduler_cooldown: int = DEFAULT_LR_SCHEDULER_COOLDOWN
    lr_scheduler_min_lr: float = DEFAULT_LR_SCHEDULER_MIN_LR
    early_stopping: bool = True
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA
    early_stopping_min_epochs: int = DEFAULT_EARLY_STOPPING_MIN_EPOCHS
    spectrum_A: int = 1

    def __post_init__(self) -> None:
        self.param_names = tuple(self.param_names)
        if not self.param_names:
            raise ValueError("param_names must not be empty")
        if self.width <= 0 or self.n_layers <= 0 or self.projection_size <= 0:
            raise ValueError("width, n_layers, and projection_size must be positive")
        if self.modes_etoa <= 0 or self.modes_elis <= 0:
            raise ValueError("Fourier modes must be positive")
        if self.boundary_padding < 0:
            raise ValueError("boundary_padding must be non-negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.matrix_probability_loss_weight < 0.0:
            raise ValueError("matrix_probability_loss_weight must be non-negative")
        if self.matrix_cross_entropy_weight < 0.0:
            raise ValueError("matrix_cross_entropy_weight must be non-negative")
        if self.spectrum_loss_weight < 0.0:
            raise ValueError("spectrum_loss_weight must be non-negative")
        if self.spectrum_max_error_percent <= 0.0:
            raise ValueError("spectrum_max_error_percent must be positive")
        if self.spectrum_max_error_temperature_percent <= 0.0:
            raise ValueError("spectrum_max_error_temperature_percent must be positive")
        if self.spectrum_huber_delta_percent <= 0.0:
            raise ValueError("spectrum_huber_delta_percent must be positive")
        if self.spectrum_top_k <= 0:
            raise ValueError("spectrum_top_k must be positive")
        if self.lr_scheduler not in {"plateau", "none"}:
            raise ValueError("lr_scheduler must be 'plateau' or 'none'")
        if not 0.0 < self.lr_scheduler_factor < 1.0:
            raise ValueError("lr_scheduler_factor must be in (0, 1)")
        if self.lr_scheduler_patience < 0 or self.lr_scheduler_cooldown < 0:
            raise ValueError("lr_scheduler_patience and lr_scheduler_cooldown must be non-negative")
        if self.lr_scheduler_min_lr < 0.0:
            raise ValueError("lr_scheduler_min_lr must be non-negative")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        if self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        if self.early_stopping_min_epochs < 1:
            raise ValueError("early_stopping_min_epochs must be positive")
        self._is_fit = False

    def fit(
        self,
        theta: Sequence[Sequence[float]],
        matrices: Sequence[Sequence[Sequence[float]]],
        etoa_grid: Sequence[float],
        elis_grid: Sequence[float],
        *,
        val_theta: Sequence[Sequence[float]] | None = None,
        val_matrices: Sequence[Sequence[Sequence[float]]] | None = None,
        spectrum_lis_flux: Sequence[float] | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_every: int = 0,
        reserve_checkpoint: str | Path | None = None,
        resume_checkpoint: str | Path | None = None,
    ) -> "TorchFNOTransferMatrixSurrogate":
        """Fit the FNO and keep the best validation-loss state."""
        torch, nn, F = _require_torch()
        torch.manual_seed(self.random_state)

        theta_array = self._prepare_theta(theta, fit=True)
        target = self._prepare_matrices(matrices)
        self.etoa_grid_ = _as_1d_positive("etoa_grid", etoa_grid)
        self.elis_grid_ = _as_1d_positive("elis_grid", elis_grid)
        if target.shape[1:] != (self.etoa_grid_.size, self.elis_grid_.size):
            raise ValueError("matrices shape must match energy grids")
        self._configure_loss_terms(spectrum_lis_flux)

        val_theta_array = None
        val_target = None
        if val_theta is not None or val_matrices is not None:
            if val_theta is None or val_matrices is None:
                raise ValueError("val_theta and val_matrices must be supplied together")
            val_theta_array = self._prepare_theta(val_theta, fit=False)
            val_target = self._prepare_matrices(val_matrices)
            if val_target.shape[1:] != target.shape[1:]:
                raise ValueError("validation matrices must match training matrix shape")

        self.device_ = self._resolve_device(torch)
        self.model_ = _make_fno_model(
            input_channels=len(self.param_names) + 2,
            width=self.width,
            modes_etoa=self.modes_etoa,
            modes_elis=self.modes_elis,
            n_layers=self.n_layers,
            projection_size=self.projection_size,
            boundary_padding=self.boundary_padding,
            dropout=self.dropout,
            torch=torch,
            nn=nn,
            F=F,
        ).to(self.device_)

        x_tensor = torch.as_tensor(theta_array, dtype=torch.float32)
        y_tensor = torch.as_tensor(target, dtype=torch.float32)
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
        scheduler = None
        if self.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
                threshold=0.0,
                threshold_mode="abs",
                cooldown=self.lr_scheduler_cooldown,
                min_lr=self.lr_scheduler_min_lr,
            )
        if resume_checkpoint is not None:
            resume_state = torch.load(
                Path(resume_checkpoint),
                map_location=self.device_,
                weights_only=False,
            )
            self.model_.load_state_dict(resume_state["model_state"])
            start_epoch = int(resume_state.get("epoch", 0)) + 1
        else:
            start_epoch = 1
        monitor = (
            ConvergenceMonitor(
                patience=self.early_stopping_patience,
                min_delta=self.early_stopping_min_delta,
                min_epochs=self.early_stopping_min_epochs,
            )
            if self.early_stopping
            else None
        )

        checkpoint_dir_path = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if checkpoint_dir_path is not None:
            checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
        reserve_checkpoint_path = Path(reserve_checkpoint) if reserve_checkpoint is not None else None
        if reserve_checkpoint_path is not None:
            reserve_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        best_loss = float("inf")
        best_selection_key = None
        best_epoch = 0
        best_state = None
        self.history_ = []
        for epoch in range(start_epoch, self.epochs + 1):
            self.model_.train()
            for theta_batch, target_batch in loader:
                theta_batch = theta_batch.to(self.device_)
                target_batch = target_batch.to(self.device_)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(self._input_grid(theta_batch, torch))
                loss, _ = self._loss_components(logits, target_batch, torch, F)
                loss.backward()
                optimizer.step()

            train_metrics = self._evaluate_arrays(theta_array, target, torch, F)
            train_loss = train_metrics["loss"]
            train_rmse = train_metrics["rmse"]
            if val_theta_array is not None and val_target is not None:
                val_metrics = self._evaluate_arrays(val_theta_array, val_target, torch, F)
                val_loss = val_metrics["loss"]
                val_rmse = val_metrics["rmse"]
                selection_key = (val_loss,)
            else:
                val_metrics = {
                    "matrix_loss": float("nan"),
                    "matrix_cross_entropy_loss": float("nan"),
                    "matrix_probability_loss": float("nan"),
                    "spectrum_loss": float("nan"),
                    "max_spectrum_percentage_error": float("nan"),
                }
                val_loss = float("nan")
                val_rmse = float("nan")
                selection_key = (train_loss,)

            if val_theta_array is not None and val_target is not None and self.spectrum_loss_weight > 0.0:
                selection_key = (
                    val_metrics["max_spectrum_percentage_error"],
                    val_loss,
                )

            if scheduler is not None:
                scheduler_metric = (
                    val_metrics["max_spectrum_percentage_error"]
                    if val_theta_array is not None and val_target is not None and self.spectrum_loss_weight > 0.0
                    else (val_loss if val_theta_array is not None and val_target is not None else train_loss)
                )
                scheduler.step(scheduler_metric)
            else:
                scheduler_metric = (
                    val_metrics["max_spectrum_percentage_error"]
                    if val_theta_array is not None and val_target is not None and self.spectrum_loss_weight > 0.0
                    else (val_loss if val_theta_array is not None and val_target is not None else train_loss)
                )

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_matrix_loss": train_metrics["matrix_loss"],
                "val_matrix_loss": val_metrics["matrix_loss"],
                "train_matrix_cross_entropy_loss": train_metrics["matrix_cross_entropy_loss"],
                "val_matrix_cross_entropy_loss": val_metrics["matrix_cross_entropy_loss"],
                "train_matrix_probability_loss": train_metrics["matrix_probability_loss"],
                "val_matrix_probability_loss": val_metrics["matrix_probability_loss"],
                "train_spectrum_loss": train_metrics["spectrum_loss"],
                "val_spectrum_loss": val_metrics["spectrum_loss"],
                "train_max_spectrum_percentage_error": train_metrics["max_spectrum_percentage_error"],
                "val_max_spectrum_percentage_error": val_metrics["max_spectrum_percentage_error"],
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            self.history_.append(row)
            if self.verbose and (epoch == 1 or epoch % max(self.epochs // 10, 1) == 0):
                print(
                    f"epoch {epoch}/{self.epochs} "
                    f"train_loss={train_loss:.6g} val_loss={val_loss:.6g} "
                    f"train_spec={train_metrics['spectrum_loss']:.6g} "
                    f"val_spec={val_metrics['spectrum_loss']:.6g} "
                    f"train_max_pct={train_metrics['max_spectrum_percentage_error']:.6g} "
                    f"val_max_pct={val_metrics['max_spectrum_percentage_error']:.6g} "
                    f"train_rmse={train_rmse:.6g} val_rmse={val_rmse:.6g}"
                )

            if best_selection_key is None or selection_key < best_selection_key:
                best_selection_key = selection_key
                best_loss = val_loss if val_theta_array is not None else train_loss
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model_.state_dict().items()
                }

            if checkpoint_dir_path is not None and checkpoint_every > 0 and epoch % checkpoint_every == 0:
                self._save_torch_checkpoint(checkpoint_dir_path / f"epoch_{epoch:04d}.pt", epoch, torch)
            if reserve_checkpoint_path is not None and checkpoint_every > 0 and epoch % checkpoint_every == 0:
                self._save_torch_checkpoint(reserve_checkpoint_path, epoch, torch)

            if monitor is not None and monitor.update(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss if val_theta_array is not None and val_target is not None else train_loss,
                monitor_value=scheduler_metric,
            ):
                break

        if best_state is None and resume_checkpoint is not None:
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in self.model_.state_dict().items()
            }
            best_epoch = int(start_epoch - 1)
            best_loss = float("nan")
        if best_state is None:
            raise RuntimeError("FNO training did not produce a model state")
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.model_state_ = best_state
        self.best_epoch_ = int(best_epoch)
        self.best_loss_ = float(best_loss)
        self.stopped_epoch_ = int(monitor.stopped_epoch) if monitor is not None and monitor.stopped_epoch is not None else int(self.epochs)
        self.early_stopping_monitor_ = monitor.history if monitor is not None else []
        self._is_fit = True

        if checkpoint_dir_path is not None:
            self.model_state_ = {
                key: value.detach().cpu().clone()
                for key, value in self.model_.state_dict().items()
            }
            self._save_torch_checkpoint(checkpoint_dir_path / "best.pt", self.best_epoch_, torch)
        if reserve_checkpoint_path is not None and self.epochs < start_epoch:
            self._save_torch_checkpoint(reserve_checkpoint_path, start_epoch - 1, torch)
        return self

    def matrix(
        self,
        etoa_grid: Sequence[float],
        elis_grid: Sequence[float],
        theta: Mapping[str, float] | Sequence[float],
    ) -> np.ndarray:
        """Predict a row-normalized transfer matrix on the trained fixed grid."""
        self._require_fit()
        etoa = _as_1d_positive("etoa_grid", etoa_grid)
        elis = _as_1d_positive("elis_grid", elis_grid)
        if etoa.shape != self.etoa_grid_.shape or not np.allclose(etoa, self.etoa_grid_):
            raise ValueError("FNO surrogate requires the trained ETOA grid")
        if elis.shape != self.elis_grid_.shape or not np.allclose(elis, self.elis_grid_):
            raise ValueError("FNO surrogate requires the trained ELIS grid")
        theta_array = self._theta_matrix(theta)
        return self.predict_matrices(theta_array)[0]

    def predict_matrices(
        self,
        theta: Sequence[Sequence[float]] | np.ndarray,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Predict matrices for a batch of parameter vectors."""
        self._require_fit()
        torch, _, F = _require_torch()
        model = self._model_for_inference(torch)
        theta_array = self._prepare_theta(theta, fit=False)
        step = self.batch_size if batch_size is None else int(batch_size)
        if step <= 0:
            raise ValueError("batch_size must be positive")
        probs = []
        with torch.no_grad():
            for start in range(0, theta_array.shape[0], step):
                stop = start + step
                theta_tensor = torch.as_tensor(theta_array[start:stop], dtype=torch.float32, device=self.device_)
                logits = model(self._input_grid(theta_tensor, torch))
                probs.append(F.softmax(logits, dim=-1).detach().cpu().numpy())
        return np.concatenate(probs, axis=0)

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

    def _prepare_theta(self, theta: Sequence[Sequence[float]] | np.ndarray, *, fit: bool) -> np.ndarray:
        array = np.asarray(theta, dtype=float)
        if array.ndim == 1:
            if array.size != len(self.param_names):
                raise ValueError("theta vector length must match param_names")
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != len(self.param_names):
            raise ValueError("theta must have shape (n_samples, n_params)")
        if np.any(~np.isfinite(array)):
            raise ValueError("theta contains non-finite values")

        transformed = array.copy()
        for index, name in enumerate(self.param_names):
            if name.lower() in {"d0", "diffusion", "diffusion_coefficient"}:
                if np.any(transformed[:, index] <= 0.0):
                    raise ValueError(f"parameter '{name}' must be positive")
                transformed[:, index] = np.log10(transformed[:, index])

        if fit:
            self.theta_center_ = transformed.mean(axis=0)
            scale = transformed.std(axis=0)
            self.theta_scale_ = np.where(scale > 0.0, scale, 1.0)
        return ((transformed - self.theta_center_) / self.theta_scale_).astype(np.float32)

    def _theta_matrix(self, theta: Mapping[str, float] | Sequence[float]) -> np.ndarray:
        return _theta_vector(theta, self.param_names).reshape(1, -1)

    def _prepare_matrices(self, matrices: Sequence[Sequence[Sequence[float]]]) -> np.ndarray:
        array = np.asarray(matrices, dtype=float)
        if array.ndim != 3:
            raise ValueError("matrices must have shape (n_samples, n_etoa, n_elis)")
        if np.any(~np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError("matrices must be finite and non-negative")
        row_sums = array.sum(axis=2, keepdims=True)
        if np.any(row_sums <= 0.0) or np.any(~np.isfinite(row_sums)):
            raise ValueError("every matrix row must have positive finite probability")
        return (array / row_sums).astype(np.float32)

    def _coordinate_channels(self, n_batch: int, torch):
        log_etoa = np.log(self.etoa_grid_)
        log_elis = np.log(self.elis_grid_)
        etoa_scale = log_etoa.std() if log_etoa.std() > 0.0 else 1.0
        elis_scale = log_elis.std() if log_elis.std() > 0.0 else 1.0
        etoa_coord = ((log_etoa - log_etoa.mean()) / etoa_scale).astype(np.float32)
        elis_coord = ((log_elis - log_elis.mean()) / elis_scale).astype(np.float32)
        etoa_channel = np.repeat(etoa_coord[:, None], log_elis.size, axis=1)
        elis_channel = np.repeat(elis_coord[None, :], log_etoa.size, axis=0)
        coords = np.stack([etoa_channel, elis_channel], axis=0)
        coords = np.repeat(coords[None, ...], n_batch, axis=0)
        return torch.as_tensor(coords, dtype=torch.float32, device=self.device_)

    def _input_grid(self, theta_tensor, torch):
        n_batch = int(theta_tensor.shape[0])
        coords = self._coordinate_channels(n_batch, torch)
        param_channels = theta_tensor[:, :, None, None].expand(
            n_batch,
            theta_tensor.shape[1],
            self.etoa_grid_.size,
            self.elis_grid_.size,
        )
        return torch.cat([coords, param_channels], dim=1)

    def _configure_loss_terms(self, spectrum_lis_flux: Sequence[float] | None) -> None:
        self.fold_lis_weight_ = None
        self.fold_toa_factor_ = None

        if spectrum_lis_flux is None:
            if self.spectrum_loss_weight > 0.0:
                raise ValueError("spectrum_lis_flux is required when spectrum_loss_weight > 0")
            return

        lis_flux = _as_1d_positive("spectrum_lis_flux", spectrum_lis_flux)
        if lis_flux.shape != self.elis_grid_.shape:
            raise ValueError("spectrum_lis_flux must match elis_grid")

        p_lis = ekin_to_momentum(self.elis_grid_, A=int(self.spectrum_A))
        p_toa = ekin_to_momentum(self.etoa_grid_, A=int(self.spectrum_A))
        fold_lis_weight = lis_flux / np.maximum(p_lis * p_lis, 1.0e-30)
        fold_toa_factor = p_toa * p_toa
        if np.any(~np.isfinite(fold_lis_weight)) or np.any(fold_lis_weight <= 0.0):
            raise ValueError("folded-spectrum LIS weights must be finite and positive")

        self.fold_lis_weight_ = fold_lis_weight.astype(np.float32)
        self.fold_toa_factor_ = fold_toa_factor.astype(np.float32)

    def _loss_components(self, logits, target, torch, F):
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        normalized_target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0e-30)
        matrix_cross_entropy = -(normalized_target * log_probs).sum(dim=-1).mean()
        matrix_probability_loss = ((probs - normalized_target) ** 2).mean()
        matrix_loss = (
            float(self.matrix_cross_entropy_weight) * matrix_cross_entropy
            + float(self.matrix_probability_loss_weight) * matrix_probability_loss
        )

        spectrum_loss = logits.new_tensor(0.0)
        if self.spectrum_loss_weight > 0.0:
            pred_spectrum = self._folded_spectrum_tensor(probs, torch)
            true_spectrum = self._folded_spectrum_tensor(normalized_target, torch)
            relative_delta = (pred_spectrum - true_spectrum) / true_spectrum.abs().clamp_min(1.0e-30)
            relative_error = torch.sqrt(relative_delta ** 2 + 1.0e-12)
            temperature = float(self.spectrum_max_error_temperature_percent) / 100.0
            threshold = float(self.spectrum_max_error_percent) / 100.0
            huber_delta = float(self.spectrum_huber_delta_percent) / 100.0
            per_spectrum_loss = _piecewise_bin_error_loss(
                relative_error,
                threshold,
                temperature,
                huber_delta,
                torch,
            )
            top_k = min(int(self.spectrum_top_k), int(per_spectrum_loss.shape[0]))
            spectrum_loss = torch.topk(
                per_spectrum_loss, k=top_k, dim=0,
            ).values.mean()

        loss = matrix_loss + float(self.spectrum_loss_weight) * spectrum_loss
        components = {
            "loss": loss,
            "matrix_loss": matrix_loss,
            "matrix_cross_entropy_loss": matrix_cross_entropy,
            "matrix_probability_loss": matrix_probability_loss,
            "spectrum_loss": spectrum_loss,
        }
        return loss, components

    def _folded_spectrum_tensor(self, matrices, torch):
        if self.fold_lis_weight_ is None or self.fold_toa_factor_ is None:
            raise RuntimeError("folded-spectrum loss is not configured")
        lis_weight = torch.as_tensor(self.fold_lis_weight_, dtype=torch.float32, device=matrices.device)
        toa_factor = torch.as_tensor(self.fold_toa_factor_, dtype=torch.float32, device=matrices.device)
        return torch.matmul(matrices, lis_weight) * toa_factor[None, :]

    def _evaluate_arrays(self, theta: np.ndarray, matrices: np.ndarray, torch, F) -> dict[str, float]:
        self.model_.eval()
        totals = {
            "loss": 0.0,
            "matrix_loss": 0.0,
            "matrix_cross_entropy_loss": 0.0,
            "matrix_probability_loss": 0.0,
            "spectrum_loss": 0.0,
            "max_spectrum_percentage_error": 0.0,
        }
        preds = []
        n_seen = 0
        max_spectrum_error = 0.0
        with torch.no_grad():
            for start in range(0, theta.shape[0], self.batch_size):
                stop = start + self.batch_size
                theta_tensor = torch.as_tensor(theta[start:stop], dtype=torch.float32, device=self.device_)
                target = torch.as_tensor(matrices[start:stop], dtype=torch.float32, device=self.device_)
                logits = self.model_(self._input_grid(theta_tensor, torch))
                _, components = self._loss_components(logits, target, torch, F)
                batch_size = int(target.shape[0])
                n_seen += batch_size
                for name, value in components.items():
                    totals[name] += float(value.detach().cpu()) * batch_size
                if self.spectrum_loss_weight > 0.0:
                    pred_spectrum = self._folded_spectrum_tensor(
                        F.softmax(logits, dim=-1), torch
                    )
                    true_spectrum = self._folded_spectrum_tensor(
                        target / target.sum(dim=-1, keepdim=True).clamp_min(1.0e-30), torch
                    )
                    exact_relative_error = (
                        (pred_spectrum - true_spectrum).abs()
                        / true_spectrum.abs().clamp_min(1.0e-30)
                    )
                    max_spectrum_error = max(
                        max_spectrum_error,
                        float(exact_relative_error.amax(dim=-1).max().detach().cpu()) * 100.0,
                    )
                preds.append(F.softmax(logits, dim=-1).detach().cpu().numpy())
        pred = np.concatenate(preds, axis=0)
        metrics = {name: total / max(n_seen, 1) for name, total in totals.items()}
        metrics["max_spectrum_percentage_error"] = max_spectrum_error
        metrics["rmse"] = float(np.sqrt(np.mean((pred - matrices) ** 2)))
        self.model_.train()
        return metrics

    def _model_for_inference(self, torch):
        if not hasattr(self, "model_"):
            _, nn, F = _require_torch()
            boundary_padding = getattr(self, "boundary_padding", 0)
            self.model_ = _make_fno_model(
                input_channels=len(self.param_names) + 2,
                width=self.width,
                modes_etoa=self.modes_etoa,
                modes_elis=self.modes_elis,
                n_layers=self.n_layers,
                projection_size=self.projection_size,
                boundary_padding=boundary_padding,
                dropout=self.dropout,
                torch=torch,
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

    def _save_torch_checkpoint(self, path: Path, epoch: int, torch) -> None:
        state = {
            "epoch": int(epoch),
            "model_state": {key: value.detach().cpu() for key, value in self.model_.state_dict().items()},
            "param_names": self.param_names,
            "etoa_grid": self.etoa_grid_,
            "elis_grid": self.elis_grid_,
            "history": getattr(self, "history_", []),
            "best_epoch": getattr(self, "best_epoch_", None),
            "best_loss": getattr(self, "best_loss_", None),
            "optimizer": getattr(self, "optimizer_name_", "AdamW"),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "matrix_probability_loss_weight": float(self.matrix_probability_loss_weight),
            "matrix_cross_entropy_weight": float(self.matrix_cross_entropy_weight),
            "spectrum_loss_weight": float(self.spectrum_loss_weight),
            "spectrum_max_error_percent": float(self.spectrum_max_error_percent),
            "spectrum_max_error_temperature_percent": float(self.spectrum_max_error_temperature_percent),
            "spectrum_huber_delta_percent": float(self.spectrum_huber_delta_percent),
            "boundary_padding": int(self.boundary_padding),
                "spectrum_top_k": int(self.spectrum_top_k),
            "lr_scheduler": self.lr_scheduler,
            "lr_scheduler_factor": float(self.lr_scheduler_factor),
            "lr_scheduler_patience": int(self.lr_scheduler_patience),
            "lr_scheduler_cooldown": int(self.lr_scheduler_cooldown),
            "lr_scheduler_min_lr": float(self.lr_scheduler_min_lr),
            "early_stopping": bool(self.early_stopping),
            "early_stopping_patience": int(self.early_stopping_patience),
            "early_stopping_min_delta": float(self.early_stopping_min_delta),
            "early_stopping_min_epochs": int(self.early_stopping_min_epochs),
            "stopped_epoch": int(getattr(self, "stopped_epoch_", epoch)),
            "spectrum_A": int(self.spectrum_A),
        }
        torch.save(state, path)

    def _require_fit(self) -> None:
        if not self._is_fit:
            raise RuntimeError("TorchFNOTransferMatrixSurrogate.fit must be called first")


def _make_fno_model(
    input_channels: int,
    width: int,
    modes_etoa: int,
    modes_elis: int,
    n_layers: int,
    projection_size: int,
    boundary_padding: int,
    dropout: float,
    torch,
    nn,
    F,
):
    class SpectralConv2d(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            scale = 1.0 / max(in_channels * out_channels, 1)
            self.weights_pos = nn.Parameter(
                scale
                * torch.randn(
                    in_channels,
                    out_channels,
                    modes_etoa,
                    modes_elis,
                    dtype=torch.cfloat,
                )
            )
            self.weights_neg = nn.Parameter(
                scale
                * torch.randn(
                    in_channels,
                    out_channels,
                    modes_etoa,
                    modes_elis,
                    dtype=torch.cfloat,
                )
            )

        def forward(self, x):
            batch, _, height, width_ = x.shape
            x_ft = torch.fft.rfft2(x)
            out_ft = torch.zeros(
                batch,
                self.weights_pos.shape[1],
                height,
                width_ // 2 + 1,
                dtype=torch.cfloat,
                device=x.device,
            )
            mode_h = min(modes_etoa, max(1, height // 2), self.weights_pos.shape[2])
            mode_w = min(modes_elis, width_ // 2 + 1, self.weights_pos.shape[3])
            out_ft[:, :, :mode_h, :mode_w] = torch.einsum(
                "bixy,ioxy->boxy",
                x_ft[:, :, :mode_h, :mode_w],
                self.weights_pos[:, :, :mode_h, :mode_w],
            )
            out_ft[:, :, -mode_h:, :mode_w] = torch.einsum(
                "bixy,ioxy->boxy",
                x_ft[:, :, -mode_h:, :mode_w],
                self.weights_neg[:, :, :mode_h, :mode_w],
            )
            return torch.fft.irfft2(out_ft, s=(height, width_))

    class FNO2d(nn.Module):
        def __init__(self):
            super().__init__()
            self.lift = nn.Conv2d(input_channels, width, kernel_size=1)
            self.spectral = nn.ModuleList([SpectralConv2d(width, width) for _ in range(n_layers)])
            self.pointwise = nn.ModuleList([nn.Conv2d(width, width, kernel_size=1) for _ in range(n_layers)])
            self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
            self.proj1 = nn.Conv2d(width, projection_size, kernel_size=1)
            self.proj2 = nn.Conv2d(projection_size, 1, kernel_size=1)
            self.boundary_padding = int(boundary_padding)

        def forward(self, x):
            x = self.lift(x)
            height, width_ = x.shape[-2:]
            pad = min(self.boundary_padding, height - 1, width_ - 1)
            if pad:
                x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
            for spectral, pointwise in zip(self.spectral, self.pointwise):
                x = spectral(x) + pointwise(x)
                x = F.gelu(x)
                x = self.dropout(x)
            if pad:
                x = x[..., pad:-pad, pad:-pad]
            x = F.gelu(self.proj1(x))
            return self.proj2(x).squeeze(1)

    return FNO2d()


def matrix_rmse(true_matrix: np.ndarray, pred_matrix: np.ndarray) -> float:
    """Return RMSE between two matrix batches or matrices."""
    true = np.asarray(true_matrix, dtype=float)
    pred = np.asarray(pred_matrix, dtype=float)
    if true.shape != pred.shape:
        raise ValueError("matrix shapes do not match")
    return float(np.sqrt(np.mean((pred - true) ** 2)))
