"""PyTorch Fourier Neural Operator surrogate for HelProp transfer matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..kernel import DEFAULT_PARAM_NAMES, fold_lis
from ..neural import _as_1d_positive, _theta_vector


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


@dataclass
class TorchFNOTransferMatrixSurrogate:
    """Fixed-grid 2D FNO mapping HelProp parameters to transfer matrices."""

    param_names: tuple[str, ...] = DEFAULT_PARAM_NAMES
    width: int = 32
    modes_etoa: int = 10
    modes_elis: int = 10
    n_layers: int = 4
    projection_size: int = 64
    dropout: float = 0.05
    learning_rate: float = 1.0e-3
    epochs: int = 300
    batch_size: int = 16
    weight_decay: float = 1.0e-4
    random_state: int = 123
    device: str = "auto"
    verbose: bool = False

    def __post_init__(self) -> None:
        self.param_names = tuple(self.param_names)
        if not self.param_names:
            raise ValueError("param_names must not be empty")
        if self.width <= 0 or self.n_layers <= 0 or self.projection_size <= 0:
            raise ValueError("width, n_layers, and projection_size must be positive")
        if self.modes_etoa <= 0 or self.modes_elis <= 0:
            raise ValueError("Fourier modes must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
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
        checkpoint_dir: str | Path | None = None,
        checkpoint_every: int = 0,
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

        checkpoint_dir_path = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if checkpoint_dir_path is not None:
            checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

        best_loss = float("inf")
        best_epoch = 0
        best_state = None
        self.history_ = []
        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            for theta_batch, target_batch in loader:
                theta_batch = theta_batch.to(self.device_)
                target_batch = target_batch.to(self.device_)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(self._input_grid(theta_batch, torch))
                loss = _row_cross_entropy(logits, target_batch, F)
                loss.backward()
                optimizer.step()

            train_loss, train_rmse = self._evaluate_arrays(theta_array, target, torch, F)
            if val_theta_array is not None and val_target is not None:
                val_loss, val_rmse = self._evaluate_arrays(val_theta_array, val_target, torch, F)
                selection_loss = val_loss
            else:
                val_loss = float("nan")
                val_rmse = float("nan")
                selection_loss = train_loss

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            self.history_.append(row)
            if self.verbose and (epoch == 1 or epoch % max(self.epochs // 10, 1) == 0):
                print(
                    f"epoch {epoch}/{self.epochs} "
                    f"train_loss={train_loss:.6g} val_loss={val_loss:.6g} "
                    f"train_rmse={train_rmse:.6g} val_rmse={val_rmse:.6g}"
                )

            if selection_loss < best_loss:
                best_loss = selection_loss
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model_.state_dict().items()
                }

            if checkpoint_dir_path is not None and checkpoint_every > 0 and epoch % checkpoint_every == 0:
                self._save_torch_checkpoint(checkpoint_dir_path / f"epoch_{epoch:04d}.pt", epoch, torch)

        if best_state is None:
            raise RuntimeError("FNO training did not produce a model state")
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.model_state_ = best_state
        self.best_epoch_ = int(best_epoch)
        self.best_loss_ = float(best_loss)
        self._is_fit = True

        if checkpoint_dir_path is not None:
            self.model_state_ = {
                key: value.detach().cpu().clone()
                for key, value in self.model_.state_dict().items()
            }
            self._save_torch_checkpoint(checkpoint_dir_path / "best.pt", self.best_epoch_, torch)
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

    def _evaluate_arrays(self, theta: np.ndarray, matrices: np.ndarray, torch, F) -> tuple[float, float]:
        self.model_.eval()
        losses = []
        preds = []
        with torch.no_grad():
            for start in range(0, theta.shape[0], self.batch_size):
                stop = start + self.batch_size
                theta_tensor = torch.as_tensor(theta[start:stop], dtype=torch.float32, device=self.device_)
                target = torch.as_tensor(matrices[start:stop], dtype=torch.float32, device=self.device_)
                logits = self.model_(self._input_grid(theta_tensor, torch))
                losses.append(float(_row_cross_entropy(logits, target, F).detach().cpu()))
                preds.append(F.softmax(logits, dim=-1).detach().cpu().numpy())
        pred = np.concatenate(preds, axis=0)
        rmse = float(np.sqrt(np.mean((pred - matrices) ** 2)))
        self.model_.train()
        return float(np.mean(losses)), rmse

    def _model_for_inference(self, torch):
        if not hasattr(self, "model_"):
            _, nn, F = _require_torch()
            self.model_ = _make_fno_model(
                input_channels=len(self.param_names) + 2,
                width=self.width,
                modes_etoa=self.modes_etoa,
                modes_elis=self.modes_elis,
                n_layers=self.n_layers,
                projection_size=self.projection_size,
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
        }
        torch.save(state, path)

    def _require_fit(self) -> None:
        if not self._is_fit:
            raise RuntimeError("TorchFNOTransferMatrixSurrogate.fit must be called first")


def _row_cross_entropy(logits, target, F):
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0e-30)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target * log_probs).sum(dim=-1).mean()


def _make_fno_model(
    input_channels: int,
    width: int,
    modes_etoa: int,
    modes_elis: int,
    n_layers: int,
    projection_size: int,
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

        def forward(self, x):
            x = self.lift(x)
            for spectral, pointwise in zip(self.spectral, self.pointwise):
                x = spectral(x) + pointwise(x)
                x = F.gelu(x)
                x = self.dropout(x)
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
