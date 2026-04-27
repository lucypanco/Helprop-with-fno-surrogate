"""Grid runner: generate parameter design and run HelProp for each point.

Supports any subset of inferable parameters (D0, m, B0, angle).

Usage:
    python -m bayesian.run_grid [options]

Examples:
    python -m bayesian.run_grid --infer D0,m --n-points 5 --method lhs
    python -m bayesian.run_grid --infer D0,m,B0,angle --n-points 4
    python -m bayesian.run_grid --infer D0,m --n-points 3 --method grid
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from bayesian.config import (
    GRID_DEFAULTS,
    HELPROP_BIN,
    INFERABLE_PARAMS,
    OUTPUT_DIR,
    PARAM_RANGES,
    SIM_DEFAULTS,
)
from bayesian.io import EXTENSIONS


def generate_design(param_names: list, n_points: int, method: str,
                    seed: int = 42) -> np.ndarray:
    """Generate a (N, P) design matrix of parameter values.

    Args:
        param_names: list of parameter names (subset of INFERABLE_PARAMS)
        n_points:    points per dimension
        method:      "grid" for regular grid, "lhs" for Latin Hypercube Sampling
        seed:        random seed for LHS

    Returns:
        Array of shape (N, P) with columns for each parameter
    """
    n_params = len(param_names)
    ranges = [(PARAM_RANGES[p]["min"], PARAM_RANGES[p]["max"]) for p in param_names]

    if method == "grid":
        grids = []
        for p, (lo, hi) in zip(param_names, ranges):
            grids.append(np.linspace(lo, hi, n_points))
        # Full factorial meshgrid
        mesh = np.meshgrid(*grids, indexing='ij')
        design = np.column_stack([m.ravel() for m in mesh])
    elif method == "lhs":
        from scipy.stats.qmc import LatinHypercube

        n_total = n_points ** n_params
        sampler = LatinHypercube(d=n_params, seed=seed)
        unit_samples = sampler.random(n=n_total)
        design = np.empty_like(unit_samples)
        for j, (lo, hi) in enumerate(ranges):
            design[:, j] = lo + unit_samples[:, j] * (hi - lo)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'grid' or 'lhs'.")

    return design


def run_helprop(param_names: list, param_values: dict, output_path: Path,
                sim_overrides: dict) -> bool:
    """Run HelProp for a single parameter point, writing matrix to output_path.

    Args:
        param_names: list of inferred parameter names
        param_values: dict mapping param name to value
        output_path: path for output matrix file
        sim_overrides: dict of HelProp CLI overrides (from SIM_DEFAULTS)

    Returns True on success.
    """
    cmd = [str(HELPROP_BIN)]

    # Add inferred parameters as CLI args
    for p in param_names:
        cmd.extend([f"--{p}", str(param_values[p])])

    # Add fixed simulation parameters
    fixed_keys = ["A", "Z", "polarity", "R0", "indexA", "indexB",
                  "etoa", "elis", "number", "nthread", "iotype"]
    for key in fixed_keys:
        if key in sim_overrides:
            cmd.extend([f"--{key}", str(sim_overrides[key])])

    # Add other fixed params (B0, angle) if they are not being inferred
    for key in ["B0", "angle"]:
        if key not in param_names and key in sim_overrides:
            cmd.extend([f"--{key}", str(sim_overrides[key])])

    cmd.append(str(output_path))

    params_str = ", ".join(f"{p}={param_values[p]:.3f}" for p in param_names)
    print(f"  Running: {params_str} -> {output_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    if result.returncode != 0:
        print(f"  FAILED (rc={result.returncode}): {result.stderr[:500]}", file=sys.stderr)
        return False

    return True


def _make_filename(param_names: list, param_values: dict, ext: str) -> str:
    """Create a unique filename from parameter values."""
    parts = "_".join(f"{p}_{v:.4f}" for p, v in zip(param_names, param_values.values()))
    return f"matrix_{parts}{ext}"


def main():
    parser = argparse.ArgumentParser(description="Run HelProp over a parameter design grid")
    parser.add_argument("--infer", type=str, default="D0,m",
                        help="Comma-separated list of parameters to infer (e.g. D0,m,B0,angle)")
    parser.add_argument("--n-points", type=int, default=GRID_DEFAULTS["n_points"],
                        help="Points per dimension")
    parser.add_argument("--method", choices=["grid", "lhs"], default=GRID_DEFAULTS["method"],
                        help="Design method")
    parser.add_argument("--number", type=int, default=None,
                        help="Override particle count per bin")
    parser.add_argument("--nthread", type=int, default=None,
                        help="Override thread count")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for LHS")
    args = parser.parse_args()

    param_names = [p.strip() for p in args.infer.split(",")]
    for p in param_names:
        if p not in INFERABLE_PARAMS:
            print(f"Error: '{p}' is not an inferable parameter. "
                  f"Choose from: {INFERABLE_PARAMS}", file=sys.stderr)
            sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    design = generate_design(param_names, args.n_points, args.method, args.seed)
    n_total = len(design)
    print(f"Design: {n_total} points ({args.n_points}^{len(param_names)}, "
          f"params={param_names}, method={args.method})")

    sim_overrides = dict(SIM_DEFAULTS)
    if args.number is not None:
        sim_overrides["number"] = args.number
    if args.nthread is not None:
        sim_overrides["nthread"] = args.nthread

    iotype = sim_overrides.get("iotype", SIM_DEFAULTS["iotype"]).upper()
    ext = EXTENSIONS.get(iotype, ".csv")

    manifest = []
    n_ok = 0
    n_fail = 0

    for i, row in enumerate(design):
        param_values = {p: float(row[j]) for j, p in enumerate(param_names)}
        fname = _make_filename(param_names, param_values, ext)
        outpath = output_dir / fname

        if outpath.exists():
            print(f"  [{i+1}/{n_total}] Skipping (exists): {fname}")
            ok = True
        else:
            print(f"  [{i+1}/{n_total}]", end="")
            ok = run_helprop(param_names, param_values, outpath, sim_overrides)

        if ok:
            n_ok += 1
            entry = {"params": param_values, "file": fname, "iotype": iotype}
            # Also store flat keys for backward compatibility
            for p, v in param_values.items():
                entry[p] = v
            manifest.append(entry)
        else:
            n_fail += 1

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone: {n_ok} succeeded, {n_fail} failed. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
