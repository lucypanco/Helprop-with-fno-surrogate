"""Placeholder observed data interface.

Replace the functions in this module with your own data loading logic.
The placeholder generates synthetic observed data by running HelProp at
a known (D0, m) point and adding 10% fractional uncertainty.
"""

import numpy as np
from pathlib import Path

from bayesian.config import M_PROTON, PROJECT_ROOT
from bayesian.io import read_spec


def ekin2p(ekin: float, A: int = 1) -> float:
    return (ekin * (ekin + 2 * M_PROTON)) ** 0.5 * A


def get_observed_data():
    """Return (E_obs, F_obs, F_err) in GeV-compatible units.

    Returns:
        E_obs: 1D array of observed kinetic energy bins [GeV]
        F_obs: 1D array of observed flux values [1/GeV]
        F_err: 1D array of fractional uncertainties (e.g. 0.1 for 10%)
    """
    # Try to load from the Output file in the project root
    output_file = PROJECT_ROOT / "Output"
    if output_file.exists():
        return _load_from_output(output_file)

    # Fallback: synthetic data
    return _synthetic_placeholder()


def _load_from_output(path: Path):
    """Load observed spectrum from the project Output file.

    Tries TXT format first (IO_TXT spec format: "# E F" header),
    then CSV, then falls back to permissive two-column reading.
    """
    for iotype in ("TXT", "CSV"):
        try:
            E_obs, F_obs = read_spec(str(path), iotype=iotype)
            F_err = np.full_like(F_obs, 0.1)  # 10% fractional uncertainty as placeholder
            return E_obs, F_obs, F_err
        except (ValueError, IndexError):
            continue

    # Fallback: permissive two-column reading (any whitespace/comma separated)
    E, F = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                E.append(float(parts[0]))
                F.append(float(parts[1]))

    E_obs = np.array(E)
    F_obs = np.array(F)
    F_err = np.full_like(F_obs, 0.1)  # 10% fractional uncertainty as placeholder

    return E_obs, F_obs, F_err


def _synthetic_placeholder():
    """Generate a simple synthetic observed spectrum with 10% uncertainty."""
    E_obs = np.geomspace(0.5, 50, 15)  # GeV, typical AMS-02 range

    # Simple power-law placeholder: F ~ 1.8 * E^-2.7
    F_obs = 1.8 * E_obs ** (-2.7)
    F_err = np.full_like(E_obs, 0.1)  # 10% fractional uncertainty

    return E_obs, F_obs, F_err


def get_lis_flux(ELIS: np.ndarray):
    """Return LIS flux at the given ELIS energy grid points.

    This is a placeholder using a simple power-law LIS.
    Replace with your own LIS model or file reader.

    Args:
        ELIS: LIS energy grid [GeV]

    Returns:
        F_LIS: LIS flux [1/GeV] at each ELIS point
    """
    # Simple LIS placeholder: F ~ 2.0 * E^-2.75 (rough Voyager-level unmodulated)
    F_LIS = 2.0 * ELIS ** (-2.75)
    return F_LIS
