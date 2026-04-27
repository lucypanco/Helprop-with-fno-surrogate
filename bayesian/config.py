"""Shared configuration for the Bayesian inference pipeline."""

import math
from pathlib import Path

# Project root (one level up from bayesian/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
HELPROP_BIN = PROJECT_ROOT / "HelProp"
OUTPUT_DIR = PROJECT_ROOT / "bayesian" / "output"
EMULATOR_DIR = PROJECT_ROOT / "bayesian" / "emulators"
STAN_DIR = PROJECT_ROOT / "bayesian" / "stan"

# Parameter ranges for the design of experiments
PARAM_RANGES = {
    "D0":    {"min": 1.0,  "max": 20.0},   # 1e22 cm^2/s
    "m":     {"min": -1.0, "max": 2.0},     # dimensionless co-rotation factor
    "B0":    {"min": 2.0,  "max": 10.0},    # nT
    "angle": {"min": 5.0,  "max": 35.0},    # degrees
}

# Parameter transforms: map raw parameter value to the space used for
# GP training and polynomial fitting.  D0 uses log-space; the rest are
# linear (identity).
PARAM_TRANSFORMS = {
    "D0":    lambda x: math.log(x),
    "m":     lambda x: x,
    "B0":    lambda x: x,
    "angle": lambda x: x,
}

# Inverse transforms (emulator space -> physical space)
PARAM_INV_TRANSFORMS = {
    "D0":    lambda x: math.exp(x),
    "m":     lambda x: x,
    "B0":    lambda x: x,
    "angle": lambda x: x,
}

# Priors for Stan model
PRIORS = {
    "D0":    {"dist": "lognormal", "mu": 1.6,  "sigma": 0.7},   # log(D0) ~ N(1.6, 0.7), center ~5
    "m":     {"dist": "normal",    "mu": 0.0,  "sigma": 1.0},   # m ~ N(0, 1)
    "B0":    {"dist": "normal",    "mu": 5.0,  "sigma": 1.5},   # B0 ~ N(5, 1.5) nT
    "angle": {"dist": "normal",    "mu": 15.0, "sigma": 10.0},  # angle ~ N(15, 10) deg
}

# Parameters that can be inferred (subset of PARAM_RANGES keys)
INFERABLE_PARAMS = ["D0", "m", "B0", "angle"]

# Simulation defaults (fixed params passed to HelProp CLI)
SIM_DEFAULTS = {
    "number": 1000,        # particles per TOA energy bin
    "nthread": 4,
    "A": 1,                # proton
    "Z": 1,
    "B0": 5,               # nT
    "polarity": -1,
    "angle": 15,           # deg
    "R0": 1,               # GV
    "indexA": 1,
    "indexB": 1,
    "iotype": "CSV",
    "etoa": "0.1,100,20",  # min,max,nbin in GeV
    "elis": "0.1,200,30",  # min,max,nbin in GeV
}

# Grid design defaults
GRID_DEFAULTS = {
    "n_points": 4,          # points per inferred dimension
    "method": "lhs",        # "lhs" or "grid"
}

# Emulator defaults
EMULATOR_DEFAULTS = {
    "poly_degree": 2,       # polynomial degree for Stan export
    "n_predict": 50,        # dense prediction grid points per dimension
}

# Proton rest mass in GeV (must match Unit.h / HelProp.cc)
M_PROTON = 0.938272


def n_poly_coeffs(degree: int, n_vars: int) -> int:
    """Number of coefficients in an N-variate polynomial of given degree.

    C(degree + n_vars, n_vars) = (degree + n_vars)! / (degree! * n_vars!)
    """
    from math import comb
    return comb(degree + n_vars, n_vars)


def ekin2p(ekin: float, A: int = 1) -> float:
    """Convert kinetic energy per nucleon [GeV] to total momentum [GeV/c].

    Replicates HelProp.cc: ekin2p(E, A) = sqrt(E * (E + 2 * m_proton)) * A
    """
    return (ekin * (ekin + 2 * M_PROTON)) ** 0.5 * A


def build_exponent_table(degree: int, n_vars: int) -> list:
    """Build the exponent table for an N-variate polynomial of given degree.

    Returns a list of N-tuples (e1, ..., eN) with sum(ei) <= degree,
    ordered by total degree then lexicographic.

    Example: degree=2, n_vars=4 -> 15 coefficients
    [[0,0,0,0], [1,0,0,0], [0,1,0,0], ..., [0,0,0,2]]
    """
    table = []

    def _recurse(dim, remaining, partial):
        if dim == n_vars - 1:
            table.append(tuple(partial + [remaining]))
        else:
            for e in range(remaining + 1):
                _recurse(dim + 1, remaining - e, partial + [e])

    for d in range(degree + 1):
        _recurse(0, d, [])

    return table
