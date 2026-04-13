"""End-to-end Bayesian inference pipeline for HelProp (D0, m).

Runs the grid simulation, trains emulators, prepares Stan data,
and runs CmdStan for posterior inference.

Usage:
    python -m bayesian.run_inference [options]

Examples:
    python -m bayesian.run_inference --placeholder --skip-grid
    python -m bayesian.run_inference --placeholder --n-d0 3 --n-m 3 --method grid
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from bayesian.config import (
    EMULATOR_DIR,
    OUTPUT_DIR,
    STAN_DIR,
)


def run_grid(args):
    """Phase 2: Run the simulation grid (or skip)."""
    from bayesian.run_grid import main as grid_main
    # Reconstruct sys.argv for the grid runner
    grid_argv = ["run_grid"]
    grid_argv.extend(["--n-d0", str(args.n_d0)])
    grid_argv.extend(["--n-m", str(args.n_m)])
    grid_argv.extend(["--method", args.method])
    if args.number:
        grid_argv.extend(["--number", str(args.number)])
    if args.nthread:
        grid_argv.extend(["--nthread", str(args.nthread)])
    if args.output_dir:
        grid_argv.extend(["--output-dir", args.output_dir])

    old_argv = sys.argv
    sys.argv = grid_argv
    try:
        grid_main()
    finally:
        sys.argv = old_argv


def train_emulators(args, input_dir: Path, output_dir: Path):
    """Phase 3: Train GP emulators and export polynomials."""
    from bayesian.train_emulator import main as train_main
    train_argv = ["train_emulator"]
    train_argv.extend(["--input-dir", str(input_dir)])
    train_argv.extend(["--output-dir", str(output_dir)])
    train_argv.extend(["--poly-degree", str(args.poly_degree)])
    if args.skip_gp:
        train_argv.append("--skip-gp")

    old_argv = sys.argv
    sys.argv = train_argv
    try:
        train_main()
    finally:
        sys.argv = old_argv


def prepare_stan_data(emulator_dir: Path, stan_dir: Path, use_placeholder: bool):
    """Phase 4: Prepare the full Stan data JSON with observed data + LIS."""
    from bayesian.obs_placeholder import get_observed_data, get_lis_flux

    # Load polynomial coefficients from emulator output
    poly_path = emulator_dir / "stan_data_poly.json"
    with open(poly_path) as f:
        poly_data = json.load(f)

    # Load observed data
    E_obs, F_obs, F_err = get_observed_data()

    # The poly_data has ETOA/ELIS from the emulator training;
    # we need those same grids. Load them from the first matrix file.
    input_dir = OUTPUT_DIR
    manifest_path = input_dir / "manifest.json"
    ETOA = ELIS = None
    if manifest_path.exists():
        from bayesian.train_emulator import load_matrix_csv
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest:
            etoa, elis, _ = load_matrix_csv(input_dir / manifest[0]["file"])
            ETOA, ELIS = etoa, elis

    if ETOA is None:
        print("Error: Could not determine ETOA/ELIS grids. Run the grid first.", file=sys.stderr)
        sys.exit(1)

    # LIS flux at ELIS points
    F_LIS = get_lis_flux(ELIS)

    # Build full Stan data dict
    stan_data = {
        "n_coeffs": poly_data["n_coeffs"],
        "n_toa": poly_data["n_toa"],
        "n_lis": poly_data["n_lis"],
        "poly_degree": poly_data["poly_degree"],
        "coeffs": poly_data["coeffs"],
        "ETOA": ETOA.tolist(),
        "ELIS": ELIS.tolist(),
        "n_obs": len(E_obs),
        "E_obs": E_obs.tolist(),
        "F_obs": F_obs.tolist(),
        "F_err": F_err.tolist(),
        "F_LIS": F_LIS.tolist(),
    }

    data_path = stan_dir / "helprop_data.json"
    with open(data_path, "w") as f:
        json.dump(stan_data, f, indent=2)
    print(f"Stan data written to {data_path}")
    return data_path


def run_stan(stan_dir: Path, data_path: Path, chains: int, iterations: int):
    """Phase 5: Compile and run CmdStan."""
    stan_model = stan_dir / "helprop_bayes.stan"

    # Try cmdstanpy first
    try:
        from cmdstanpy import CmdStanModel
        print("Using cmdstanpy...")

        model = CmdStanModel(stan_file=str(stan_model))
        fit = model.sample(
            data=str(data_path),
            chains=chains,
            iter_warmup=iterations // 2,
            iter_sampling=iterations,
            show_console=True,
        )

        # Print summary
        print("\n=== Posterior Summary ===")
        print(fit.summary())

        # Save samples
        output_csv = stan_dir / "posterior_samples.csv"
        fit.save_csvfiles(dir=str(stan_dir))
        print(f"Samples saved to {stan_dir}")

        return fit

    except ImportError:
        print("cmdstanpy not found, falling back to subprocess...")

    # Fallback: direct CmdStan invocation
    cmdstan_path = Path.home() / ".cmdstan"
    cmdstan_dirs = sorted(cmdstan_path.glob("cmdstan-*")) if cmdstan_path.exists() else []
    if not cmdstan_dirs:
        print("Error: CmdStan not found. Install with: install_cmdstan", file=sys.stderr)
        sys.exit(1)

    cmdstan_bin = cmdstan_dirs[-1] / "bin"
    exe_path = stan_dir / "helprop_bayes"

    # Compile
    print("Compiling Stan model...")
    compile_cmd = [str(cmdstan_bin / "stanc"), str(stan_model)]
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"stanc failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    compile_cmd2 = [str(cmdstan_bin / "stan_compile"), str(exe_path)]
    # Actually, use make
    make_cmd = ["make", str(exe_path)]
    result = subprocess.run(make_cmd, capture_output=True, text=True, cwd=str(cmdstan_dirs[-1]))
    if result.returncode != 0:
        print(f"make failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Sample
    print("Running MCMC sampling...")
    sample_cmd = [
        str(exe_path),
        "sample",
        f"num_chains={chains}",
        f"num_warmup={iterations // 2}",
        f"num_samples={iterations}",
        f"data file={data_path}",
        "output", f"file={stan_dir / 'output.csv'}",
    ]
    result = subprocess.run(sample_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Sampling failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    print(result.stdout)
    print(f"Output saved to {stan_dir / 'output.csv'}")


def main():
    parser = argparse.ArgumentParser(description="End-to-end Bayesian inference for HelProp (D0, m)")
    parser.add_argument("--placeholder", action="store_true",
                        help="Use placeholder observed data instead of real observations")
    parser.add_argument("--skip-grid", action="store_true",
                        help="Skip grid simulation (use existing output)")
    parser.add_argument("--skip-gp", action="store_true",
                        help="Skip GP training (use existing emulators.pkl)")
    parser.add_argument("--n-d0", type=int, default=5, help="Number of D0 design points")
    parser.add_argument("--n-m", type=int, default=5, help="Number of m design points")
    parser.add_argument("--method", choices=["grid", "lhs"], default="lhs",
                        help="Design method for grid runner")
    parser.add_argument("--number", type=int, default=None,
                        help="Override particle count per bin")
    parser.add_argument("--nthread", type=int, default=None,
                        help="Override thread count")
    parser.add_argument("--poly-degree", type=int, default=3,
                        help="Polynomial degree for emulator export")
    parser.add_argument("--chains", type=int, default=4,
                        help="Number of MCMC chains")
    parser.add_argument("--iterations", type=int, default=2000,
                        help="Number of MCMC iterations per chain")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override simulation output directory")
    parser.add_argument("--emulator-dir", type=str, default=None,
                        help="Override emulator output directory")
    args = parser.parse_args()

    input_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    emulator_dir = Path(args.emulator_dir) if args.emulator_dir else EMULATOR_DIR
    emulator_dir.mkdir(parents=True, exist_ok=True)
    stan_dir = STAN_DIR
    stan_dir.mkdir(parents=True, exist_ok=True)

    # Phase 2: Grid simulation
    if not args.skip_grid:
        print("=" * 50)
        print("Phase 2: Running simulation grid")
        print("=" * 50)
        run_grid(args)
    else:
        print("Skipping grid simulation (--skip-grid)")

    # Phase 3: Train emulators
    print("\n" + "=" * 50)
    print("Phase 3: Training emulators")
    print("=" * 50)
    train_emulators(args, input_dir, emulator_dir)

    # Phase 4: Prepare Stan data
    print("\n" + "=" * 50)
    print("Phase 4: Preparing Stan data")
    print("=" * 50)
    data_path = prepare_stan_data(emulator_dir, stan_dir, args.placeholder)

    # Phase 5: Run Stan
    print("\n" + "=" * 50)
    print("Phase 5: Running CmdStan inference")
    print("=" * 50)
    run_stan(stan_dir, data_path, args.chains, args.iterations)

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
