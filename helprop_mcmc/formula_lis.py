"""Analytic local-interstellar spectrum models.

The Shen et al. (2025) LIS parameterization (their Equation 5) is

    J(E) = a0 (E / Ec)**a1
           [1 + (E / a2)**a3]**a4
           [1 + (E / a5)**a6]**a7,

with Ec = 1 GeV/n.  The ``lis_`` prefix used here keeps LIS parameters
separate from HelProp's diffusion parameter ``indexA`` and from the
article's unprefixed notation.
"""

from __future__ import annotations

import numpy as np


SHEN_LIS_SHAPE_NAMES = tuple(f"lis_a{i}" for i in range(1, 8))
SHEN_LIS_ALL_NAMES = ("lis_a0",) + SHEN_LIS_SHAPE_NAMES
SHEN_LIS_EC_GEV = 1.0

# Shen et al. (2025), Table 2, proton row (H).  These defaults are for the
# requested proton verification run; other nuclei require their own row.
DEFAULT_SHEN_LIS_PARAMS = {
    "lis_a0": 5.6603e3,
    "lis_a1": -7.0814e-1,
    "lis_a2": 6.7728e-2,
    "lis_a3": -1.7053,
    "lis_a4": -5.4184e-1,
    "lis_a5": 1.6393,
    "lis_a6": 1.5554,
    "lis_a7": -1.3292,
}

# Practical verification bounds around the Table 2 proton fit.  Shen et al.
# give best-fit values here, not a covariance matrix or confidence interval.
DEFAULT_SHEN_LIS_RANGES = {
    "lis_a0": (1.0e3, 1.0e4),
    "lis_a1": (-1.0, -0.4),
    "lis_a2": (3.0e-2, 1.2e-1),
    "lis_a3": (-3.0, -0.8),
    "lis_a4": (-1.2, 0.0),
    "lis_a5": (0.8, 3.0),
    "lis_a6": (0.8, 2.5),
    "lis_a7": (-2.5, -0.5),
}


def _parameter(params, name):
    """Read ``lis_aN`` parameters and accept bare ``aN`` as a convenience."""
    if name in params:
        return float(params[name])
    return float(params[name.replace("lis_", "")])


def shen_lis_flux(energy_gev, params, ec_gev=SHEN_LIS_EC_GEV):
    """Evaluate Shen et al. Eq. (5) at kinetic energy per nucleon in GeV.

    Evaluation is performed in log space so broad MCMC proposals do not
    overflow in the two smooth-break terms.  The returned flux is positive
    and has the same arbitrary/physical units as ``a0``.
    """
    energy = np.asarray(energy_gev, dtype=float)
    if np.any(~np.isfinite(energy)) or np.any(energy <= 0.0):
        raise ValueError("LIS energies must be finite and positive")
    if not np.isfinite(ec_gev) or ec_gev <= 0.0:
        raise ValueError("ec_gev must be finite and positive")

    a0 = _parameter(params, "lis_a0")
    a1 = _parameter(params, "lis_a1")
    a2 = _parameter(params, "lis_a2")
    a3 = _parameter(params, "lis_a3")
    a4 = _parameter(params, "lis_a4")
    a5 = _parameter(params, "lis_a5")
    a6 = _parameter(params, "lis_a6")
    a7 = _parameter(params, "lis_a7")
    values = (a0, a1, a2, a3, a4, a5, a6, a7)
    if np.any(~np.isfinite(values)):
        raise ValueError("all LIS parameters must be finite")
    if a0 <= 0.0 or a2 <= 0.0 or a5 <= 0.0:
        raise ValueError("a0, a2, and a5 must be positive")

    log_energy = np.log(energy)
    log_flux = np.log(a0) + a1 * (log_energy - np.log(ec_gev))
    # log(1 + exp(x)) is stable for both very small and very large x.
    log_flux += a4 * np.logaddexp(0.0, a3 * (log_energy - np.log(a2)))
    log_flux += a7 * np.logaddexp(0.0, a6 * (log_energy - np.log(a5)))
    if np.any(~np.isfinite(log_flux)) or np.any(log_flux >= np.log(np.finfo(float).max)):
        raise ValueError("LIS formula produced non-finite or overflowing flux")
    return np.exp(log_flux)


def write_shen_lis(path, energy_gev, params):
    """Write a two-column HelProp LIS input file."""
    energy = np.asarray(energy_gev, dtype=float)
    if energy.ndim != 1 or energy.size < 2 or np.any(np.diff(energy) <= 0.0):
        raise ValueError("LIS energy grid must be one-dimensional and increasing")
    flux = shen_lis_flux(energy, params)
    np.savetxt(path, np.column_stack((energy, flux)), header="E F")
