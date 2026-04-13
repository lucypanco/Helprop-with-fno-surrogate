"""Grid runner: generate (D0, m) design and run HelProp for each point.

Usage:
    python -m bayesian.run_grid [options]

Examples:
    python -m bayesian.run_grid --n-d0 3 --n-m 3 --method grid
    python -m bayesian.run_grid --n-d0 10 --n-m 8 --method lhs --number 2000
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
    OUTPUT_DIR,
    PARAM_RANGES,
    SIM_DEFAULTS,
)


def generate_design(n_d0: int, n_m: int, method: str, seed: int = 42) -> np.ndarray:
    """Generate a (N, 2) design matrix of (D0, m) values.

    Args:
        n_d0: number of D0 levels
        n_m: number of m levels
        method: "grid" for regular grid, "lhs" for Latin Hypercube Sampling
        seed: random seed for LHS

    Returns:
        Array of shape (N, 2) with columns [D0, m]
    """
    d0_lo, d0_hi = PARAM_RANGES["D0"]["min"], PARAM_RANGES["D0"]["max"]
    m_lo, m_hi = PARAM_RANGES["m"]["min"], PARAM_RANGES["m"]["max"]

    if method == "grid":
        d0_vals = np.linspace(d0_lo, d0_hi, n_d0)
        m_vals = np.linspace(m_lo, m_hi, n_m)
        D0_grid, M_grid = np.meshgrid(d0_vals, m_vals)
        design = np.column_stack([D0_grid.ravel(), M_grid.ravel()])
    elif method == "lhs":
        from scipy.stats.qmc import LatinHypercube

        n_total = n_d0 * n_m
        sampler = LatinHypercube(d=2, seed=seed)
        unit_samples = sampler.random(n=n_total)
        # Scale to parameter ranges
        design = np.empty_like(unit_samples)
        design[:, 0] = d0_lo + unit_samples[:, 0] * (d0_hi - d0_lo)
        design[:, 1] = m_lo + unit_samples[:, 1] * (m_hi - m_lo)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'grid' or 'lhs'.")

    return design


def run_helprop(d0: float, m: float, output_path: Path, sim_overrides: dict) -> bool:
    """Run HelProp for a single (D0, m) point, writing matrix CSV to output_path.

    Returns True on success.
    """
    cmd = [
        str(HELPROP_BIN),
        "--D0", str(d0),
        "--m", str(m),
        "--iotype", sim_overrides.get("iotype", SIM_DEFAULTS["iotype"]),
        "--number", str(sim_overrides.get("number", SIM_DEFAULTS["number"])),
        "--nthread", str(sim_overrides.get("nthread", SIM_DEFAULTS["nthread"])),
        "--A", str(sim_overrides.get("A", SIM_DEFAULTS["A"])),
        "--Z", str(sim_overrides.get("Z", SIM_DEFAULTS["Z"])),
        "--B0", str(sim_overrides.get("B0", SIM_DEFAULTS["B0"])),
        "--polarity", str(sim_overrides.get("polarity", SIM_DEFAULTS["polarity"])),
        "--angle", str(sim_overrides.get("angle", SIM_DEFAULTS["angle"])),
        "--R0", str(sim_overrides.get("R0", SIM_DEFAULTS["R0"])),
        "--indexA", str(sim_overrides.get("indexA", SIM_DEFAULTS["indexA"])),
        "--indexB", str(sim_overrides.get("indexB", SIM_DEFAULTS["indexB"])),
        "--etoa", sim_overrides.get("etoa", SIM_DEFAULTS["etoa"]),
        "--elis", sim_overrides.get("elis", SIM_DEFAULTS["elis"]),
        str(output_path),  # positional <outmatrix>
    ]

    print(f"  Running: D0={d0:.3f}, m={m:.3f} -> {output_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    if result.returncode != 0:
        print(f"  FAILED (rc={result.returncode}): {result.stderr[:500]}", file=sys.stderr)
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Run HelProp over a (D0, m) design grid")
    parser.add_argument("--n-d0", type=int, default=GRID_DEFAULTS["n_d0"],
                        help="Number of D0 points")
    parser.add_argument("--n-m", type=int, default=GRID_DEFAULTS["n_m"],
                        help="Number of m points")
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

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    design = generate_design(args.n_d0, args.n_m, args.method, args.seed)
    n_total = len(design)
    print(f"Design: {n_total} points ({args.n_d0} x {args.n_m}, method={args.method})")

    sim_overrides = dict(SIM_DEFAULTS)
    if args.number is not None:
        sim_overrides["number"] = args.number
    if args.nthread is not None:
        sim_overrides["nthread"] = args.nthread

    manifest = []
    n_ok = 0
    n_fail = 0

    for i, (d0, m) in enumerate(design):
        fname = f"matrix_D0_{d0:.4f}_m_{m:.4f}.csv"
        outpath = output_dir / fname

        if outpath.exists():
            print(f"  [{i+1}/{n_total}] Skipping (exists): {fname}")
            ok = True
        else:
            print(f"  [{i+1}/{n_total}]", end="")
            ok = run_helprop(d0, m, outpath, sim_overrides)

        if ok:
            n_ok += 1
            manifest.append({"D0": float(d0), "m": float(m), "file": fname})
        else:
            n_fail += 1

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone: {n_ok} succeeded, {n_fail} failed. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
