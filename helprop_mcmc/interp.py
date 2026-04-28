"""Log-log interpolation matching HelProp's LogInterp."""

import numpy as np


class LogInterp:
    """Log-log linear interpolation.

    Replicates HelProp's LogInterp to interpolate the modulated output
    at observed energy points.
    """

    def __init__(self, x, y):
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        if len(x) < 2:
            raise ValueError("Need at least 2 points for interpolation")
        if np.any(x <= 0) or np.any(y <= 0):
            raise ValueError("x and y must be positive for log interpolation")
        self.xlog = np.log(x)
        self.ylog = np.log(y)

    def __call__(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        log_x = np.log(x)
        idx = np.searchsorted(self.xlog, log_x)
        idx = np.clip(idx, 1, len(self.xlog) - 1)
        i0, i1 = idx - 1, idx
        dx = self.xlog[i1] - self.xlog[i0]
        m = (self.ylog[i1] - self.ylog[i0]) / dx
        return np.exp(self.ylog[i0] + m * (log_x - self.xlog[i0]))
