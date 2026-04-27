"""Post-MCMC diagnostics and plots for the Bayesian inference pipeline.

Loads CmdStan CSV output, computes posterior summaries, convergence
diagnostics (R-hat, ESS), generates corner plots and posterior predictive
checks, and saves results as JSON.

Usage:
    python -m bayesian.postprocess --stan-dir bayesian/stan --infer D0,m,B0,angle
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _find_csv_files(stan_dir: Path):
    """Find CmdStan output CSV files in the stan directory."""
    # cmdstanpy names files like output-1.csv, output-2.csv, etc.
    # or the user may have posterior_samples.csv from save_csvfiles()
    csv_files = sorted(stan_dir.glob("output-*.csv"))
    if not csv_files:
        csv_files = sorted(stan_dir.glob("*.csv"))
        # Exclude helprop_data.json-named files
        csv_files = [f for f in csv_files if f.name != "helprop_data.csv"]
    return csv_files


def _parse_cmdstan_csv(filepath: Path):
    """Parse a CmdStan output CSV file, returning a dict of column -> array.

    Skips comment lines (starting with #) and the initial columns that
    CmdStan uses for iteration info (lp__, accept_stat__, stepsize__,
    treedepth__, n_leapfrog__, divergent__, energy__).
    """
    columns = {}
    with open(filepath) as f:
        header = None
        data_lines = []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if header is None:
                header = parts
                for h in header:
                    columns[h] = []
            else:
                data_lines.append(parts)

    # Convert to numpy arrays
    n_cols = len(header)
    for i, h in enumerate(header):
        values = []
        for row in data_lines:
            if i < len(row):
                values.append(float(row[i]))
        columns[h] = np.array(values)

    return columns


def _compute_summary(samples_dict: dict, param_names: list) -> dict:
    """Compute posterior summary for specified parameters.

    Returns dict with mean, median, std, 16th/84th percentiles.
    """
    summary = {}
    for p in param_names:
        stan_name = {"D0": "D0", "m": "m_param", "B0": "B0",
                     "angle": "angle"}.get(p, p)
        if stan_name not in samples_dict:
            print(f"Warning: parameter '{stan_name}' not found in samples",
                  file=sys.stderr)
            continue
        vals = samples_dict[stan_name]
        summary[p] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "q16": float(np.percentile(vals, 16)),
            "q84": float(np.percentile(vals, 84)),
        }
    return summary


def _compute_convergence(all_samples: list, param_names: list) -> dict:
    """Compute R-hat and ESS for specified parameters across chains.

    Args:
        all_samples: list of dicts (one per chain) from _parse_cmdstan_csv
        param_names: list of parameter short names

    Returns dict with R-hat and ESS per parameter.
    """
    diagnostics = {}
    stan_names = {"D0": "D0", "m": "m_param", "B0": "B0", "angle": "angle"}

    for p in param_names:
        stan_name = stan_names.get(p, p)
        chains = []
        for s in all_samples:
            if stan_name in s:
                chains.append(s[stan_name])

        if len(chains) < 2:
            diagnostics[p] = {"Rhat": float("nan"), "ESS": float("nan"),
                              "note": "fewer than 2 chains"}
            continue

        n_chains = len(chains)
        n = len(chains[0])

        # Split-R-hat (Gelman-Rubin)
        chain_means = [np.mean(c) for c in chains]
        chain_vars = [np.var(c, ddof=1) for c in chains]
        grand_mean = np.mean(chain_means)
        B = n * np.var(chain_means, ddof=1)  # between-chain variance
        W = np.mean(chain_vars)               # within-chain variance
        var_hat = (1 - 1/n) * W + (1/n) * B
        Rhat = np.sqrt(var_hat / W) if W > 0 else float("nan")

        # Bulk ESS (using autocorrelation)
        # Simple estimate: min across chains of (n / (1 + 2*sum(rho_k)))
        # Using FFT-based autocorrelation
        ess_vals = []
        for c in chains:
            x = c - np.mean(c)
            n_c = len(x)
            if n_c < 2:
                ess_vals.append(1.0)
                continue
            # FFT autocorrelation
            fft_x = np.fft.fft(x, n=2 * n_c)
            acf = np.fft.ifft(fft_x * np.conj(fft_x))[:n_c].real
            acf /= acf[0]
            # Sum consecutive pairs until negative
            rho_sum = 0.0
            for k in range(1, n_c - 1, 2):
                pair_sum = acf[k] + (acf[k + 1] if k + 1 < n_c else 0.0)
                if pair_sum < 0:
                    break
                rho_sum += pair_sum
            tau = 1.0 + 2.0 * rho_sum
            ess_vals.append(max(1.0, n_c / tau))

        ESS = min(ess_vals) * n_chains

        diagnostics[p] = {
            "Rhat": round(float(Rhat), 4),
            "ESS": round(float(ESS), 1),
        }

    return diagnostics


def _plot_corner(all_samples: list, param_names: list, stan_dir: Path):
    """Generate corner plot of posterior distributions."""
    stan_names = {"D0": "D0", "m": "m_param", "B0": "B0", "angle": "angle"}
    labels = {"D0": r"$D_0\ (10^{22}\ \mathrm{cm}^2/\mathrm{s})$",
              "m": r"$m$",
              "B0": r"$B_0\ (\mathrm{nT})$",
              "angle": r"$\theta\ (\mathrm{deg})$"}

    # Combine all chains
    data = {}
    for p in param_names:
        stan_name = stan_names.get(p, p)
        vals = np.concatenate([s[stan_name] for s in all_samples
                               if stan_name in s])
        data[p] = vals

    n_params = len(param_names)
    if n_params < 2:
        # Simple 1D histogram
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        p = param_names[0]
        ax.hist(data[p], bins=50, density=True, alpha=0.7)
        ax.set_xlabel(labels.get(p, p))
        ax.set_ylabel("Density")
        ax.set_title(f"Posterior: {p}")
        fig.tight_layout()
        fig.savefig(stan_dir / "corner_plot.png", dpi=150)
        plt.close(fig)
        print(f"  Saved corner_plot.png (1D histogram)")
        return

    try:
        import corner
    except ImportError:
        # Fall back to manual pair plot
        _manual_corner(data, param_names, labels, stan_dir)
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    corner_data = np.column_stack([data[p] for p in param_names])
    corner_labels = [labels.get(p, p) for p in param_names]

    fig = corner.corner(corner_data, labels=corner_labels,
                        quantiles=[0.16, 0.5, 0.84], show_titles=True,
                        title_fmt=".3f")
    fig.savefig(stan_dir / "corner_plot.png", dpi=150)
    plt.close(fig)
    print(f"  Saved corner_plot.png")


def _manual_corner(data: dict, param_names: list, labels: dict,
                   stan_dir: Path):
    """Manual corner plot without corner.py."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(param_names)
    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(data[param_names[i]], bins=50, density=True,
                        alpha=0.7, color="steelblue")
                ax.set_xlabel(labels.get(param_names[i], param_names[i]))
            elif i > j:
                ax.scatter(data[param_names[j]][:500],
                           data[param_names[i]][:500],
                           s=1, alpha=0.3, c="steelblue")
                ax.set_xlabel(labels.get(param_names[j], param_names[j]))
                ax.set_ylabel(labels.get(param_names[i], param_names[i]))
            else:
                ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(stan_dir / "corner_plot.png", dpi=150)
    plt.close(fig)
    print(f"  Saved corner_plot.png (manual pair plot)")


def _plot_posterior_predictive(all_samples: list, stan_dir: Path):
    """Generate posterior predictive plot: predicted vs observed spectrum."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load Stan data for observed values
    data_path = stan_dir / "helprop_data.json"
    with open(data_path) as f:
        stan_data = json.load(f)

    E_obs = np.array(stan_data["E_obs"])
    F_obs = np.array(stan_data["F_obs"])
    F_err = np.array(stan_data["F_err"])
    n_obs = stan_data["n_obs"]

    # Collect F_pred_gen from all chains
    f_pred_all = []
    for s in all_samples:
        for col_name in s:
            if "F_pred_gen" in col_name:
                f_pred_all.append(s[col_name])
                break

    if not f_pred_all:
        print("  Warning: F_pred_gen not found in samples, skipping "
              "posterior predictive plot", file=sys.stderr)
        return

    f_pred_all = np.array(f_pred_all)  # (n_total_samples, n_obs)

    # Compute median and credible bands
    f_pred_median = np.median(f_pred_all, axis=0)
    f_pred_lo = np.percentile(f_pred_all, 16, axis=0)
    f_pred_hi = np.percentile(f_pred_all, 84, axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(E_obs, F_obs, yerr=F_err * F_obs,
                fmt="o", markersize=4, capsize=3,
                label="Observed", color="black", zorder=5)
    ax.plot(E_obs, f_pred_median, "-", lw=2,
            label="Posterior median", color="steelblue")
    ax.fill_between(E_obs, f_pred_lo, f_pred_hi,
                    alpha=0.3, color="steelblue",
                    label="68% credible interval")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Energy (GeV)")
    ax.set_ylabel("Flux (1/GeV)")
    ax.set_title("Posterior Predictive Check")
    ax.legend()
    fig.tight_layout()
    fig.savefig(stan_dir / "posterior_predictive.png", dpi=150)
    plt.close(fig)
    print(f"  Saved posterior_predictive.png")


def main():
    parser = argparse.ArgumentParser(
        description="Post-MCMC diagnostics for HelProp Bayesian inference")
    parser.add_argument("--stan-dir", type=str, default=None,
                        help="Stan output directory")
    parser.add_argument("--infer", type=str, default="D0,m",
                        help="Comma-separated list of inferred parameters")
    args = parser.parse_args()

    from bayesian.config import STAN_DIR

    stan_dir = Path(args.stan_dir) if args.stan_dir else STAN_DIR
    infer_params = [p.strip() for p in args.infer.split(",")]

    csv_files = _find_csv_files(stan_dir)
    if not csv_files:
        print(f"Error: No CmdStan output CSV files found in {stan_dir}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(csv_files)} chain CSV files")
    all_samples = [_parse_cmdstan_csv(f) for f in csv_files]

    # Combine all chains for summary statistics
    combined = {}
    for s in all_samples:
        for k, v in s.items():
            if k not in combined:
                combined[k] = []
            combined[k].extend(v)
    for k in combined:
        combined[k] = np.array(combined[k])

    print(f"Total samples: {len(next(iter(combined.values())))}")

    # Posterior summary
    print("\n--- Posterior Summary ---")
    summary = _compute_summary(combined, infer_params)
    for p in infer_params:
        if p in summary:
            s = summary[p]
            print(f"  {p:6s}: {s['mean']:.4f} +/- {s['std']:.4f}  "
                  f"median={s['median']:.4f}  "
                  f"68% CI=[{s['q16']:.4f}, {s['q84']:.4f}]")

    # Convergence diagnostics
    print("\n--- Convergence Diagnostics ---")
    convergence = _compute_convergence(all_samples, infer_params)
    for p in infer_params:
        if p in convergence:
            c = convergence[p]
            note = ""
            if c["Rhat"] > 1.1:
                note = " *** WARNING: R-hat > 1.1 ***"
            print(f"  {p:6s}: R-hat={c['Rhat']:.4f}  ESS={c['ESS']:.0f}{note}")

    # Save summary JSON
    results = {
        "parameters": summary,
        "convergence": convergence,
        "inferred": infer_params,
        "n_chains": len(csv_files),
    }
    summary_path = stan_dir / "posterior_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved {summary_path}")

    # Corner plot
    print("\n--- Generating Corner Plot ---")
    try:
        _plot_corner(all_samples, infer_params, stan_dir)
    except Exception as e:
        print(f"  Warning: corner plot failed: {e}", file=sys.stderr)

    # Posterior predictive
    print("\n--- Generating Posterior Predictive Plot ---")
    try:
        _plot_posterior_predictive(all_samples, stan_dir)
    except Exception as e:
        print(f"  Warning: posterior predictive plot failed: {e}",
              file=sys.stderr)

    print("\nPost-processing complete.")


if __name__ == "__main__":
    main()
