"""Shared configuration for the Bayesian inference pipeline."""

import os
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
    "D0": {"min": 1.0, "max": 20.0},   # 1e22 cm^2/s (same unit as --D0 CLI arg)
    "m":  {"min": -1.0, "max": 2.0},    # dimensionless co-rotation factor
}

# Simulation defaults
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
    "n_d0": 5,            # number of D0 points
    "n_m": 5,             # number of m points
    "method": "lhs",       # "lhs" (Latin Hypercube) or "grid" (regular)
}

# Emulator defaults
EMULATOR_DEFAULTS = {
    "poly_degree": 3,      # polynomial degree for Stan export
    "n_predict": 50,       # dense prediction grid points per dimension
}

# Priors
PRIORS = {
    "D0": {"dist": "lognormal", "mu": 1.6, "sigma": 0.7},   # log(D0) ~ Normal(1.6, 0.7), center ~5
    "m":  {"dist": "normal", "mu": 0.0, "sigma": 1.0},       # m ~ Normal(0, 1)
}

# Proton rest mass in GeV (must match Unit.h / HelProp.cc)
M_PROTON = 0.938272


def ekin2p(ekin: float, A: int = 1) -> float:
    """Convert kinetic energy per nucleon [GeV] to total momentum [GeV/c].

    Replicates HelProp.cc: ekin2p(E, A) = sqrt(E * (E + 2 * m_proton)) * A
    """
    return (ekin * (ekin + 2 * M_PROTON)) ** 0.5 * A
