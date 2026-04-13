"""Emulator training: GP fit + polynomial export for CmdStan.

Two-stage approach:
  1. Fit independent GPs (RBF kernel) in log-space for each Green function matrix element.
  2. Evaluate GPs on a dense grid, then fit polynomial surfaces in (log(D0), m)
     and export coefficients to JSON for the Stan model.

Usage:
    python -m bayesian.train_emulator [options]
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from bayesian.config import (
    EMULATOR_DEFAULTS,
    EMULATOR_DIR,
    OUTPUT_DIR,
    PARAM_RANGES,
)


def load_matrix_csv(path: Path):
    """Parse a CSV matrix file written by HelProp --iotype CSV.

    Returns (ETOA, ELIS, weight_matrix) where weight_matrix[i_toa, i_lis].
    """
    with open(path) as f:
        lines = f.readlines()

    # Find #ELIS line
    elis_line = None
    etoa_line = None
    matrix_start = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#ELIS,"):
            elis_line = stripped[6:]  # after "#ELIS,"
        elif stripped.startswith("#ETOA,"):
            etoa_line = stripped[6:]
        elif stripped.startswith("#Matrix"):
            matrix_start = idx + 1
            break

    if elis_line is None or etoa_line is None or matrix_start is None:
        raise ValueError(f"Could not parse CSV matrix file: {path}")

    ELIS = np.array([float(x) for x in elis_line.split(",") if x.strip()])
    ETOA = np.array([float(x) for x in etoa_line.split(",") if x.strip()])

    rows = []
    for line in lines[matrix_start:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            break
        rows.append([float(x) for x in stripped.split(",") if x.strip()])

    weight = np.array(rows)
    assert weight.shape == (len(ETOA), len(ELIS)), \
        f"Matrix shape mismatch: {weight.shape} vs ({len(ETOA)}, {len(ELIS)})"

    return ETOA, ELIS, weight


def load_manifest(output_dir: Path):
    """Load manifest.json and all referenced matrix files.

    Returns:
        X: (N, 2) array of [D0, m]
        matrices: list of (N_toa, N_lis) weight matrices
        ETOA, ELIS: energy grids (same for all runs)
    """
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    X = []
    matrices = []
    ETOA = ELIS = None

    for entry in manifest:
        d0, m = entry["D0"], entry["m"]
        mat_path = output_dir / entry["file"]
        if not mat_path.exists():
            print(f"Warning: missing {mat_path}, skipping", file=sys.stderr)
            continue

        etoa, elis, weight = load_matrix_csv(mat_path)
        if ETOA is None:
            ETOA, ELIS = etoa, elis
        else:
            assert np.allclose(ETOA, etoa, atol=1e-10), "ETOA mismatch across runs"
            assert np.allclose(ELIS, elis, atol=1e-10), "ELIS mismatch across runs"

        X.append([d0, m])
        matrices.append(weight)

    return np.array(X), matrices, ETOA, ELIS


def train_gps(X, matrices, output_dir: Path):
    """Stage 1: fit independent GPs for each matrix element in log-space.

    Args:
        X: (N, 2) array of [D0, m] design points
        matrices: list of (N_toa, N_lis) weight arrays
        output_dir: directory to save emulator pickle

    Returns:
        gps: dict mapping (i_toa, i_lis) -> fitted GP regressor
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    n_toa = matrices[0].shape[0]
    n_lis = matrices[0].shape[1]

    # Transform inputs: log(D0), keep m as-is
    X_gp = X.copy()
    X_gp[:, 0] = np.log(X_gp[:, 0])

    gps = {}
    total = n_toa * n_lis
    count = 0

    for i_toa in range(n_toa):
        for i_lis in range(n_lis):
            count += 1
            y = np.array([mat[i_toa, i_lis] for mat in matrices])

            # Work in log-space, but handle zeros (clip to small positive)
            y_safe = np.clip(y, 1e-12, None)
            y_log = np.log(y_safe)

            kernel = ConstantKernel(1.0) * RBF(length_scale=[1.0, 1.0]) + WhiteKernel(noise_level=0.01)
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
            gp.fit(X_gp, y_log)
            gps[(i_toa, i_lis)] = gp

            if count % 50 == 0 or count == total:
                print(f"  GP training: {count}/{total}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = output_dir / "emulators.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"gps": gps, "n_toa": n_toa, "n_lis": n_lis}, f)
    print(f"Saved {len(gps)} GPs to {pkl_path}")

    return gps, n_toa, n_lis


def export_polynomials(gps, n_toa, n_lis, poly_degree, n_predict, output_dir: Path):
    """Stage 2: evaluate GPs on dense grid, fit polynomials, export for Stan.

    The polynomial is in (log(D0), m), degree poly_degree.
    Coefficients are stored in a format the Stan model can consume directly.
    """
    d0_lo, d0_hi = PARAM_RANGES["D0"]["min"], PARAM_RANGES["D0"]["max"]
    m_lo, m_hi = PARAM_RANGES["m"]["min"], PARAM_RANGES["m"]["max"]

    # Dense prediction grid
    log_d0_pred = np.linspace(np.log(d0_lo), np.log(d0_hi), n_predict)
    m_pred = np.linspace(m_lo, m_hi, n_predict)
    LOG_D0_GRID, M_GRID = np.meshgrid(log_d0_pred, m_pred)
    X_pred = np.column_stack([LOG_D0_GRID.ravel(), M_GRID.ravel()])

    # Build polynomial feature matrix
    # Features: [1, logD0, m, logD0^2, logD0*m, m^2, logD0^3, logD0^2*m, logD0*m^2, m^3, ...]
    # Number of coefficients for 2-variable polynomial of degree d: (d+1)(d+2)/2
    n_coeffs = (poly_degree + 1) * (poly_degree + 2) // 2
    feature_names = _poly_feature_names(poly_degree)

    def build_poly_features(x1, x2):
        """Build polynomial features from (logD0, m)."""
        features = []
        for total_deg in range(poly_degree + 1):
            for d1 in range(total_deg + 1):
                d2 = total_deg - d1
                features.append(x1 ** d1 * x2 ** d2)
        return np.column_stack(features)

    X_poly_pred = build_poly_features(X_pred[:, 0], X_pred[:, 1])

    # For each matrix element: predict with GP, fit polynomial
    all_coeffs = np.zeros((n_toa, n_lis, n_coeffs))

    total = n_toa * n_lis
    count = 0

    for i_toa in range(n_toa):
        for i_lis in range(n_lis):
            count += 1
            gp = gps[(i_toa, i_lis)]

            # GP prediction on dense grid (log-space)
            y_pred_log, _ = gp.predict(X_pred, return_std=True)

            # Fit polynomial in (logD0, m) space
            coeffs, _, _, _ = np.linalg.lstsq(X_poly_pred, y_pred_log, rcond=None)
            all_coeffs[i_toa, i_lis, :] = coeffs

            if count % 50 == 0 or count == total:
                print(f"  Polynomial fit: {count}/{total}")

    # Export as JSON for Stan
    stan_data = {
        "poly_degree": poly_degree,
        "n_coeffs": n_coeffs,
        "n_toa": n_toa,
        "n_lis": n_lis,
        "coeffs": all_coeffs.tolist(),  # [n_toa][n_lis][n_coeffs]
        "feature_names": feature_names,
        "log_d0_range": [float(np.log(d0_lo)), float(np.log(d0_hi))],
        "m_range": [float(m_lo), float(m_hi)],
    }

    json_path = output_dir / "stan_data_poly.json"
    with open(json_path, "w") as f:
        json.dump(stan_data, f, indent=2)
    print(f"Saved polynomial coefficients to {json_path}")

    return all_coeffs, stan_data


def _poly_feature_names(degree: int) -> list:
    """Generate human-readable names for polynomial features."""
    names = []
    for total_deg in range(degree + 1):
        for d1 in range(total_deg + 1):
            d2 = total_deg - d1
            parts = []
            if d1 == 0 and d2 == 0:
                parts.append("1")
            else:
                if d1 > 0:
                    parts.append(f"logD0^{d1}" if d1 > 1 else "logD0")
                if d2 > 0:
                    parts.append(f"m^{d2}" if d2 > 1 else "m")
            names.append("*".join(parts))
    return names


def main():
    parser = argparse.ArgumentParser(description="Train GP emulators and export polynomial fits for Stan")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory with manifest.json and matrix files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for emulator output")
    parser.add_argument("--poly-degree", type=int,
                        default=EMULATOR_DEFAULTS["poly_degree"],
                        help="Polynomial degree for Stan export")
    parser.add_argument("--n-predict", type=int,
                        default=EMULATOR_DEFAULTS["n_predict"],
                        help="Dense prediction grid points per dimension")
    parser.add_argument("--skip-gp", action="store_true",
                        help="Skip GP training (load existing emulators.pkl)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else OUTPUT_DIR
    output_dir = Path(args.output_dir) if args.output_dir else EMULATOR_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_gp:
        pkl_path = output_dir / "emulators.pkl"
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        gps = data["gps"]
        n_toa = data["n_toa"]
        n_lis = data["n_lis"]
        print(f"Loaded {len(gps)} GPs from {pkl_path}")
    else:
        print("Loading simulation data...")
        X, matrices, ETOA, ELIS = load_manifest(input_dir)
        print(f"Loaded {len(X)} design points, matrix shape ({len(ETOA)}, {len(ELIS)})")

        print("Training GPs...")
        gps, n_toa, n_lis = train_gps(X, matrices, output_dir)

    print("Exporting polynomial coefficients...")
    export_polynomials(gps, n_toa, n_lis, args.poly_degree, args.n_predict, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
