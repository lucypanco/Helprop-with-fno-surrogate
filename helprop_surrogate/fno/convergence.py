"""Training convergence and early-stopping utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

DEFAULT_EARLY_STOPPING_PATIENCE = 50
DEFAULT_EARLY_STOPPING_MIN_DELTA_PERCENT = 0.1
# Backward-compatible alias for callers using the original constant name.
DEFAULT_EARLY_STOPPING_MIN_DELTA = DEFAULT_EARLY_STOPPING_MIN_DELTA_PERCENT
DEFAULT_EARLY_STOPPING_MIN_EPOCHS = 100


@dataclass
class ConvergenceMonitor:
    """Stop after validation stops improving; test loss is diagnostic only.

    For the spectrum maximum-error metric, ``min_delta`` is in percentage
    points, so the default 0.1 means 0.1%.
    """

    patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA
    min_epochs: int = DEFAULT_EARLY_STOPPING_MIN_EPOCHS
    best_value: float | None = None
    bad_epochs: int = 0
    stopped_epoch: int | None = None
    history: list[dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.patience < 0:
            raise ValueError("patience must be non-negative")
        if self.min_delta < 0.0:
            raise ValueError("min_delta must be non-negative")
        if self.min_epochs < 1:
            raise ValueError("min_epochs must be positive")

    def update(
        self,
        *,
        epoch: int,
        train_loss: float,
        val_loss: float,
        test_loss: float | None = None,
        monitor_value: float | None = None,
    ) -> bool:
        """Record losses and return whether training should stop.

        ``monitor_value`` normally is validation maximum spectrum error. Test
        loss is recorded when available but never controls early stopping.
        """
        value = val_loss if monitor_value is None else monitor_value
        if not isfinite(value):
            raise ValueError("monitor_value must be finite")
        record = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "monitor_value": float(value),
        }
        if test_loss is not None:
            record["test_loss"] = float(test_loss)
        self.history.append(record)

        if self.best_value is None or value < self.best_value - self.min_delta:
            self.best_value = float(value)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        if epoch >= self.min_epochs and self.bad_epochs >= self.patience:
            self.stopped_epoch = int(epoch)
            return True
        return False
