"""Data loading interface for observed data and LIS spectra.

Supports loading from files (TXT/CSV formats) or falling back to
synthetic placeholders when no file is provided.
"""

import numpy as np
from pathlib import Path

from bayesian.config import M_PROTON, PROJECT_ROOT
from bayesian.io import read_spec


def get_observed_data(file_path: str = None, iotype: str = None):
    """Return (E_obs, F_obs, F_err) in GeV-compatible units.

    Args:
        file_path: path to observed spectrum file. If None, tries the
                   project-root ``Output`` file, then falls back to synthetic.
        iotype:    file format ("TXT", "CSV", or None for auto-detect).

    Returns:
        E_obs: 1D array of observed kinetic energy bins [GeV]
        F_obs: 1D array of observed flux values [1/GeV]
        F_err: 1D array of fractional uncertainties
    """
    if file_path is not None:
        return _load_from_file(Path(file_path), iotype)

    # Try the project-root Output file
    output_file = PROJECT_ROOT / "Output"
    if output_file.exists():
        return _load_from_file(output_file, iotype)

    # Fallback: synthetic data
    return _synthetic_placeholder()


def _load_from_file(path: Path, iotype: str = None):
    """Load observed spectrum from a file.

    Supports 2-column (E, F) and 3-column (E, F, F_err) formats.
    When only 2 columns are present, 10% fractional uncertainty is used.
    """
    # Try structured formats first
    for fmt in ([iotype] if iotype else ["TXT", "CSV"]):
        if fmt is None:
            continue
        try:
            result = read_spec(str(path), iotype=fmt)
            if len(result) == 3 and isinstance(result[2], dict):
                # BSON returned params dict as third element
                E_obs, F_obs = result[0], result[1]
            else:
                E_obs, F_obs = result[0], result[1]
            F_err = np.full_like(F_obs, 0.1)
            return E_obs, F_obs, F_err
        except (ValueError, IndexError):
            continue

    # Permissive reading: try 3-column then 2-column
    E, F, F_err_vals = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 3:
                E.append(float(parts[0]))
                F.append(float(parts[1]))
                F_err_vals.append(float(parts[2]))
            elif len(parts) >= 2:
                E.append(float(parts[0]))
                F.append(float(parts[1]))
                F_err_vals.append(0.1)

    E_obs = np.array(E)
    F_obs = np.array(F)
    F_err = np.array(F_err_vals)

    return E_obs, F_obs, F_err


def _synthetic_placeholder():
    """Generate a simple synthetic observed spectrum with 10% uncertainty."""
    E_obs = np.geomspace(0.5, 50, 15)  # GeV
    F_obs = 1.8 * E_obs ** (-2.7)
    F_err = np.full_like(E_obs, 0.1)
    return E_obs, F_obs, F_err


def get_lis_flux(ELIS: np.ndarray, file_path: str = None, iotype: str = None):
    """Return LIS flux at the given ELIS energy grid points.

    Args:
        ELIS:     LIS energy grid [GeV]
        file_path: path to LIS spectrum file. If None, uses placeholder.
        iotype:   file format ("TXT", "CSV", or None for auto-detect).

    Returns:
        F_LIS: LIS flux [1/GeV] at each ELIS point
    """
    if file_path is not None:
        return _load_lis_from_file(ELIS, Path(file_path), iotype)

    # Placeholder: simple power-law LIS
    F_LIS = 2.0 * ELIS ** (-2.75)
    return F_LIS


def _load_lis_from_file(ELIS: np.ndarray, path: Path, iotype: str = None):
    """Load LIS from a file and interpolate onto the ELIS grid."""
    from scipy.interpolate import interp1d

    for fmt in ([iotype] if iotype else ["TXT", "CSV"]):
        if fmt is None:
            continue
        try:
            result = read_spec(str(path), iotype=fmt)
            if len(result) == 3 and isinstance(result[2], dict):
                E_lis, F_lis = result[0], result[1]
            else:
                E_lis, F_lis = result[0], result[1]
            interp = interp1d(np.log(E_lis), np.log(F_lis),
                              kind='linear', fill_value='extrapolate')
            return np.exp(interp(np.log(ELIS)))
        except (ValueError, IndexError):
            continue

    # Permissive reading
    E_lis, F_lis = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                E_lis.append(float(parts[0]))
                F_lis.append(float(parts[1]))

    E_lis = np.array(E_lis)
    F_lis = np.array(F_lis)
    interp = interp1d(np.log(E_lis), np.log(F_lis),
                      kind='linear', fill_value='extrapolate')
    return np.exp(interp(np.log(ELIS)))
