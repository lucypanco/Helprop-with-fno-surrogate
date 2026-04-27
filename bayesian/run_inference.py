"""End-to-end Bayesian inference pipeline for HelProp.

Supports inference over any subset of parameters (D0, m, B0, angle).
Runs the grid simulation, trains emulators, prepares Stan data,
runs CmdStan for posterior inference, and generates post-MCMC diagnostics.

Usage:
    python -m bayesian.run_inference [options]

Examples:
    # Full 4-parameter inference with real data
    python -m bayesian.run_inference \
        --obs-file data/ams02_proton.csv --lis-file data/vos_potgieter_lis.csv \
        --infer D0,m,B0,angle --n-points 4

    # 2-parameter inference with placeholder data
    python -m bayesian.run_inference --infer D0,m --n-points 5

    # Resume from existing grid (skip simulation)
    python -m bayesian.run_inference --skip-grid --infer D0,m,B0,angle
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from bayesian.config import (
    EMULATOR_DIR,
    INFERABLE_PARAMS,
    OUTPUT_DIR,
    PARAM_RANGES,
    PRIORS,
    SIM_DEFAULTS,
    STAN_DIR,
    build_exponent_table,
    n_poly_coeffs,
)


def run_grid(args, infer_params: list):
    """Phase 2: Run the simulation grid (or skip)."""
    from bayesian.run_grid import main as grid_main
    grid_argv = ["run_grid"]
    grid_argv.extend(["--infer", ",".join(infer_params)])
    grid_argv.extend(["--n-points", str(args.n_points)])
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


def train_emulators(args, infer_params: list, input_dir: Path, output_dir: Path):
    """Phase 3: Train GP emulators and export polynomials."""
    from bayesian.train_emulator import main as train_main
    train_argv = ["train_emulator"]
    train_argv.extend(["--input-dir", str(input_dir)])
    train_argv.extend(["--output-dir", str(output_dir)])
    train_argv.extend(["--infer", ",".join(infer_params)])
    train_argv.extend(["--poly-degree", str(args.poly_degree)])
    if args.skip_gp:
        train_argv.append("--skip-gp")

    old_argv = sys.argv
    sys.argv = train_argv
    try:
        train_main()
    finally:
        sys.argv = old_argv


def prepare_stan_data(emulator_dir: Path, stan_dir: Path, infer_params: list,
                      obs_file: str, lis_file: str, fixed_overrides: dict):
    """Phase 4: Prepare the full Stan data JSON.

    Builds the exponent table with all 4 parameter columns (filling 0 for
    non-inferred params), sets tight priors for fixed parameters, and
    combines emulator coefficients with observed data and LIS.
    """
    from bayesian.data_loader import get_observed_data, get_lis_flux
    from bayesian.io import read_matrix

    # Load polynomial coefficients from emulator output
    poly_path = emulator_dir / "stan_data_poly.json"
    with open(poly_path) as f:
        poly_data = json.load(f)

    # Load observed data
    E_obs, F_obs, F_err = get_observed_data(file_path=obs_file)

    # Load ETOA/ELIS grids from manifest
    input_dir = OUTPUT_DIR
    manifest_path = input_dir / "manifest.json"
    ETOA = ELIS = None
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest:
            first_entry = manifest[0]
            iotype = first_entry.get("iotype", "CSV")
            result = read_matrix(str(input_dir / first_entry["file"]), iotype=iotype)
            ETOA, ELIS = result[0], result[1]

    if ETOA is None:
        print("Error: Could not determine ETOA/ELIS grids. Run the grid first.",
              file=sys.stderr)
        sys.exit(1)

    # LIS flux at ELIS points
    F_LIS = get_lis_flux(ELIS, file_path=lis_file)

    # Build full 4-column exponent table from the emulator's per-inferred-param
    # exponent table. Columns correspond to [D0, m, B0, angle].
    all_params = ["D0", "m", "B0", "angle"]
    n_all = 4
    emulator_exp = poly_data["exponent_table"]  # [n_coeffs][n_inferred]
    emulator_params = poly_data["param_names"]
    n_coeffs = poly_data["n_coeffs"]

    full_exp_table = []
    for row in emulator_exp:
        full_row = [0] * n_all
        for j, pname in enumerate(emulator_params):
            idx = all_params.index(pname)
            full_row[idx] = row[j]
        full_exp_table.append(full_row)

    # Fixed parameter values for tight priors
    fixed_B0 = fixed_overrides.get("B0", SIM_DEFAULTS["B0"])
    fixed_angle = fixed_overrides.get("angle", SIM_DEFAULTS["angle"])

    # Determine prior widths for B0 and angle
    if "B0" in infer_params:
        prior_mu_B0 = PRIORS["B0"]["mu"]
        prior_sigma_B0 = PRIORS["B0"]["sigma"]
    else:
        prior_mu_B0 = fixed_B0
        prior_sigma_B0 = 0.001  # effectively fixed

    if "angle" in infer_params:
        prior_mu_angle = PRIORS["angle"]["mu"]
        prior_sigma_angle = PRIORS["angle"]["sigma"]
    else:
        prior_mu_angle = fixed_angle
        prior_sigma_angle = 0.001

    # Build full Stan data dict
    stan_data = {
        "n_params": n_all,
        "n_coeffs": n_coeffs,
        "n_toa": poly_data["n_toa"],
        "n_lis": poly_data["n_lis"],
        "poly_degree": poly_data["poly_degree"],
        "exponent_table": full_exp_table,
        "coeffs": poly_data["coeffs"],
        "ETOA": ETOA.tolist(),
        "ELIS": ELIS.tolist(),
        "n_obs": len(E_obs),
        "E_obs": E_obs.tolist(),
        "F_obs": F_obs.tolist(),
        "F_err": F_err.tolist(),
        "F_LIS": F_LIS.tolist(),
        "prior_mu_B0": prior_mu_B0,
        "prior_sigma_B0": prior_sigma_B0,
        "prior_mu_angle": prior_mu_angle,
        "prior_sigma_angle": prior_sigma_angle,
    }

    data_path = stan_dir / "helprop_data.json"
    with open(data_path, "w") as f:
        json.dump(stan_data, f, indent=2)
    print(f"Stan data written to {data_path}")
    return data_path


def run_stan(stan_dir: Path, data_path: Path, chains: int, iterations: int):
    """Phase 5: Compile and run CmdStan."""
    stan_model = stan_dir / "helprop_bayes.stan"

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

        print("\n=== Posterior Summary ===")
        print(fit.summary())

        output_csv = stan_dir / "posterior_samples.csv"
        fit.save_csvfiles(dir=str(stan_dir))
        print(f"Samples saved to {stan_dir}")

        return fit

    except ImportError:
        print("cmdstanpy not found, falling back to subprocess...")
        _run_stan_subprocess(stan_dir, data_path, chains, iterations)


def _run_stan_subprocess(stan_dir: Path, data_path: Path,
                         chains: int, iterations: int):
    """Fallback: direct CmdStan invocation via subprocess."""
    import subprocess

    cmdstan_path = Path.home() / ".cmdstan"
    cmdstan_dirs = sorted(cmdstan_path.glob("cmdstan-*")) if cmdstan_path.exists() else []
    if not cmdstan_dirs:
        print("Error: CmdStan not found. Install with: install_cmdstan",
              file=sys.stderr)
        sys.exit(1)

    cmdstan_bin = cmdstan_dirs[-1] / "bin"
    exe_path = stan_dir / "helprop_bayes"

    print("Compiling Stan model...")
    result = subprocess.run(
        [str(cmdstan_bin / "stanc"), str(stan_dir / "helprop_bayes.stan")],
        capture_output=True, text=True)
    if result.returncode != 0:
        print(f"stanc failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    print("Running MCMC sampling...")
    sample_cmd = [
        str(exe_path), "sample",
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


def run_postprocess(stan_dir: Path, infer_params: list):
    """Phase 6: Post-MCMC diagnostics and plots."""
    from bayesian.postprocess import main as postprocess_main
    post_argv = ["postprocess"]
    post_argv.extend(["--stan-dir", str(stan_dir)])
    post_argv.extend(["--infer", ",".join(infer_params)])
    old_argv = sys.argv
    sys.argv = post_argv
    try:
        postprocess_main()
    finally:
        sys.argv = old_argv


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end Bayesian inference for HelProp")
    parser.add_argument("--obs-file", type=str, default=None,
                        help="Path to observed spectrum file")
    parser.add_argument("--lis-file", type=str, default=None,
                        help="Path to LIS spectrum file")
    parser.add_argument("--infer", type=str, default="D0,m",
                        help="Comma-separated list of parameters to infer "
                             "(e.g. D0,m,B0,angle)")
    parser.add_argument("--skip-grid", action="store_true",
                        help="Skip grid simulation (use existing output)")
    parser.add_argument("--skip-gp", action="store_true",
                        help="Skip GP training (use existing emulators.pkl)")
    parser.add_argument("--n-points", type=int, default=4,
                        help="Design points per inferred dimension")
    parser.add_argument("--method", choices=["grid", "lhs"], default="lhs",
                        help="Design method for grid runner")
    parser.add_argument("--number", type=int, default=None,
                        help="Override particle count per bin")
    parser.add_argument("--nthread", type=int, default=None,
                        help="Override thread count")
    parser.add_argument("--poly-degree", type=int, default=2,
                        help="Polynomial degree for emulator export")
    parser.add_argument("--chains", type=int, default=4,
                        help="Number of MCMC chains")
    parser.add_argument("--iterations", type=int, default=2000,
                        help="Number of MCMC iterations per chain")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override simulation output directory")
    parser.add_argument("--emulator-dir", type=str, default=None,
                        help="Override emulator output directory")
    parser.add_argument("--skip-postprocess", action="store_true",
                        help="Skip Phase 6 post-MCMC diagnostics")
    # Fixed parameter overrides
    parser.add_argument("--B0", type=float, default=None,
                        help="Fixed B0 value (nT) when not inferred")
    parser.add_argument("--angle", type=float, default=None,
                        help="Fixed HCS tilt angle (deg) when not inferred")
    parser.add_argument("--A", type=int, default=None,
                        help="Atomic mass number")
    parser.add_argument("--Z", type=int, default=None,
                        help="Atomic number")
    parser.add_argument("--polarity", type=int, default=None,
                        help="Solar magnetic polarity (+1 or -1)")
    parser.add_argument("--R0", type=float, default=None,
                        help="Rigidity at 1 AU (GV)")
    parser.add_argument("--indexA", type=int, default=None,
                        help="Diffusion index A")
    parser.add_argument("--indexB", type=int, default=None,
                        help="Diffusion index B")
    args = parser.parse_args()

    # Validate infer params
    infer_params = [p.strip() for p in args.infer.split(",")]
    for p in infer_params:
        if p not in INFERABLE_PARAMS:
            print(f"Error: '{p}' is not inferable. Choose from: {INFERABLE_PARAMS}",
                  file=sys.stderr)
            sys.exit(1)

    # Build fixed parameter overrides
    fixed_overrides = {}
    for key in ["A", "Z", "polarity", "R0", "indexA", "indexB", "B0", "angle"]:
        val = getattr(args, key)
        if val is not None:
            fixed_overrides[key] = val

    # Apply overrides to SIM_DEFAULTS for downstream use
    sim = dict(SIM_DEFAULTS)
    sim.update(fixed_overrides)

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
        run_grid(args, infer_params)
    else:
        print("Skipping grid simulation (--skip-grid)")

    # Phase 3: Train emulators
    print("\n" + "=" * 50)
    print("Phase 3: Training emulators")
    print("=" * 50)
    train_emulators(args, infer_params, input_dir, emulator_dir)

    # Phase 4: Prepare Stan data
    print("\n" + "=" * 50)
    print("Phase 4: Preparing Stan data")
    print("=" * 50)
    data_path = prepare_stan_data(
        emulator_dir, stan_dir, infer_params,
        args.obs_file, args.lis_file, fixed_overrides)

    # Phase 5: Run Stan
    print("\n" + "=" * 50)
    print("Phase 5: Running CmdStan inference")
    print("=" * 50)
    run_stan(stan_dir, data_path, args.chains, args.iterations)

    # Phase 6: Post-processing
    if not args.skip_postprocess:
        print("\n" + "=" * 50)
        print("Phase 6: Post-MCMC diagnostics")
        print("=" * 50)
        run_postprocess(stan_dir, infer_params)
    else:
        print("Skipping post-processing (--skip-postprocess)")

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
