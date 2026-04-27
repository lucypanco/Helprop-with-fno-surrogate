"""Emulator training: GP fit + polynomial export for CmdStan.

Two-stage approach:
  1. Fit independent GPs (RBF kernel) in transformed parameter space
     for each Green function matrix element.
  2. Evaluate GPs on a dense grid, then fit polynomial surfaces and
     export coefficients + exponent table to JSON for the Stan model.

Supports any number of inferred parameters via PARAM_TRANSFORMS.

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
    PARAM_TRANSFORMS,
    build_exponent_table,
    n_poly_coeffs,
)
from bayesian.io import read_matrix


def load_manifest(output_dir: Path, param_names: list):
    """Load manifest.json and all referenced matrix files.

    Returns:
        X: (N, P) array of parameter values (original scale)
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
        params = entry.get("params", {})
        if not params:
            # Backward compat: flat keys in manifest
            params = {p: entry[p] for p in param_names if p in entry}

        mat_path = output_dir / entry["file"]
        if not mat_path.exists():
            print(f"Warning: missing {mat_path}, skipping", file=sys.stderr)
            continue

        iotype = entry.get("iotype", "CSV")
        result = read_matrix(str(mat_path), iotype=iotype)
        etoa, elis, weight = result[0], result[1], result[2]

        if ETOA is None:
            ETOA, ELIS = etoa, elis
        else:
            assert np.allclose(ETOA, etoa, atol=1e-10), "ETOA mismatch across runs"
            assert np.allclose(ELIS, elis, atol=1e-10), "ELIS mismatch across runs"

        X.append([params[p] for p in param_names])
        matrices.append(weight)

    return np.array(X), matrices, ETOA, ELIS


def _transform_X(X: np.ndarray, param_names: list) -> np.ndarray:
    """Transform parameter matrix to the space used for GP/polynomial fitting."""
    X_t = np.empty_like(X)
    for j, p in enumerate(param_names):
        X_t[:, j] = [PARAM_TRANSFORMS[p](v) for v in X[:, j]]
    return X_t


def train_gps(X: np.ndarray, matrices: list, param_names: list,
              output_dir: Path):
    """Stage 1: fit independent GPs for each matrix element in transformed space.

    Args:
        X: (N, P) array of parameter values (original scale)
        matrices: list of (N_toa, N_lis) weight arrays
        param_names: list of parameter names
        output_dir: directory to save emulator pickle

    Returns:
        gps: dict mapping (i_toa, i_lis) -> fitted GP regressor
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    n_params = len(param_names)
    n_toa = matrices[0].shape[0]
    n_lis = matrices[0].shape[1]

    # Transform inputs
    X_gp = _transform_X(X, param_names)

    gps = {}
    total = n_toa * n_lis
    count = 0

    for i_toa in range(n_toa):
        for i_lis in range(n_lis):
            count += 1
            y = np.array([mat[i_toa, i_lis] for mat in matrices])

            # Work in log-space, but handle zeros
            y_safe = np.clip(y, 1e-12, None)
            y_log = np.log(y_safe)

            length_scale = [1.0] * n_params
            kernel = (ConstantKernel(1.0) * RBF(length_scale=length_scale)
                      + WhiteKernel(noise_level=0.01))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                          normalize_y=True)
            gp.fit(X_gp, y_log)
            gps[(i_toa, i_lis)] = gp

            if count % 50 == 0 or count == total:
                print(f"  GP training: {count}/{total}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = output_dir / "emulators.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "gps": gps,
            "n_toa": n_toa,
            "n_lis": n_lis,
            "param_names": param_names,
        }, f)
    print(f"Saved {len(gps)} GPs to {pkl_path}")

    return gps, n_toa, n_lis


def export_polynomials(gps, n_toa: int, n_lis: int, param_names: list,
                       poly_degree: int, n_predict: int, output_dir: Path):
    """Stage 2: evaluate GPs on dense grid, fit polynomials, export for Stan.

    The polynomial is in the transformed parameter space.
    Coefficients and the exponent table are stored in a format the Stan model
    can consume directly.
    """
    n_params = len(param_names)

    # Build exponent table for N-variate polynomial of given degree
    exp_table = build_exponent_table(poly_degree, n_params)
    n_coeffs = len(exp_table)

    # Dense prediction grid in transformed space
    grids = []
    for j, p in enumerate(param_names):
        lo = PARAM_TRANSFORMS[p](PARAM_RANGES[p]["min"])
        hi = PARAM_TRANSFORMS[p](PARAM_RANGES[p]["max"])
        grids.append(np.linspace(lo, hi, n_predict))

    mesh = np.meshgrid(*grids, indexing='ij')
    X_pred = np.column_stack([m.ravel() for m in mesh])

    # Build polynomial feature matrix from exponent table
    def build_poly_features(X_t: np.ndarray) -> np.ndarray:
        """Build polynomial features using the exponent table."""
        features = np.ones((X_t.shape[0], n_coeffs))
        for k, exponents in enumerate(exp_table):
            for j, e in enumerate(exponents):
                if e > 0:
                    features[:, k] *= X_t[:, j] ** e
        return features

    X_poly_pred = build_poly_features(X_pred)

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

            # Fit polynomial in transformed parameter space
            coeffs, _, _, _ = np.linalg.lstsq(X_poly_pred, y_pred_log, rcond=None)
            all_coeffs[i_toa, i_lis, :] = coeffs

            if count % 50 == 0 or count == total:
                print(f"  Polynomial fit: {count}/{total}")

    # Export as JSON for Stan
    stan_data = {
        "n_params": n_params,
        "param_names": param_names,
        "poly_degree": poly_degree,
        "n_coeffs": n_coeffs,
        "n_toa": n_toa,
        "n_lis": n_lis,
        "exponent_table": [list(row) for row in exp_table],
        "coeffs": all_coeffs.tolist(),  # [n_toa][n_lis][n_coeffs]
    }

    json_path = output_dir / "stan_data_poly.json"
    with open(json_path, "w") as f:
        json.dump(stan_data, f, indent=2)
    print(f"Saved polynomial coefficients to {json_path}")

    return all_coeffs, stan_data


def main():
    parser = argparse.ArgumentParser(
        description="Train GP emulators and export polynomial fits for Stan")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory with manifest.json and matrix files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for emulator output")
    parser.add_argument("--infer", type=str, default=None,
                        help="Comma-separated list of inferred parameters "
                             "(e.g. D0,m,B0,angle). Read from manifest if omitted.")
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

    # Determine parameter names
    if args.infer:
        param_names = [p.strip() for p in args.infer.split(",")]
    elif args.skip_gp:
        # Read from existing pickle
        pkl_path = output_dir / "emulators.pkl"
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        param_names = data["param_names"]
        print(f"Read param_names={param_names} from existing pickle")
    else:
        # Read from manifest
        manifest_path = input_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest and "params" in manifest[0]:
            param_names = list(manifest[0]["params"].keys())
        else:
            # Backward compat
            param_names = [p for p in ["D0", "m", "B0", "angle"]
                          if p in manifest[0]]
        print(f"Read param_names={param_names} from manifest")

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
        X, matrices, ETOA, ELIS = load_manifest(input_dir, param_names)
        print(f"Loaded {len(X)} design points, "
              f"matrix shape ({len(ETOA)}, {len(ELIS)})")

        print("Training GPs...")
        gps, n_toa, n_lis = train_gps(X, matrices, param_names, output_dir)

    print("Exporting polynomial coefficients...")
    export_polynomials(gps, n_toa, n_lis, param_names,
                       args.poly_degree, args.n_predict, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
