#!/usr/bin/env python3
"""
MCMC Bayesian Analysis of HelProp modulation parameters.

Fits D0 (diffusion coefficient)  and m_corot (co-rotation factor)
to observed cosmic ray spectra using Bayesian inference.

Samplers
--------
  pt      Parallel Tempering MCMC (default, best for multi-modal posteriors)
  emcee   Ensemble MCMC with differential-evolution moves
  dynesty Dynamic nested sampling (excellent for multi-modal, no emcee needed)

Usage
-----
  python mcmc_analysis.py --helprop ../HelProp --lis ../Proton_spectrum.txt --obs ../ProtonModulated_ekin.txt --sampler emcee --nwalkers 8 --nsteps 30 --nburn 10 --nproc 0
  python mcmc_analysis.py --sampler pt   --ntemps 10 --nwalkers 20 --nsteps 2000
  python mcmc_analysis.py --sampler emcee --nwalkers 32 --nsteps 5000
 python helprop_mcmc/mcmc_analysis.py \
    --backend surrogate \
    --surrogate-low-model surrogate_runs/run_0002/kernel_fno.pkl \
    --surrogate-high-model fno_runs/run_0003/kernel_fno.pkl \
    --surrogate-split-energy 1.0 \
    --surrogate-blend-dex 0.2 \
    --lis ./Proton_spectrum.txt \
    --obs ./ProtonModulated_ekin.txt \
    --sampler dynesty \
    --nproc 1 \
    --A 1 --Z 1 --polarity -1 --R0 1 --B0 5 \
    --hcs-osc-phase 0 \
    --sample-param D0 \
    --sample-param m \
    --sample-param indexA \
    --sample-param indexB \
    --sample-param angle \
    --sample-param hcs-osc-amp \
    --sample-param hcs-omega \
    --sample-range m:-3:3 \
    --sample-range angle:5:45 \
    --sample-range hcs-osc-amp:0:10 \
    --sample-range hcs-omega:0:4 \
    --outdir chains_dual_7d_phase0 \
    --verbose


Argument Formats
----------------
  --helprop <path>       Path to HelProp executable (required)
  --lis <path>          Path to LIS (Local Proton Spectrum) input file (required)
                        Format: two columns -> E(GeV)  Flux
                        Must match energy units of --obs data
  --obs <path>          Path to observed data file (required)
                        Format: three columns -> E(GeV)  Flux  err
                        Must use SAME energy units as --lis file!
                        Energy units must be consistent between:
                          - LIS file (column 1, in GeV)
                          - Observed data file (column 1, must match LIS units)

Common Issues
-------------
  1. Energy unit mismatch between LIS and obs data -> NaN errors, swap rate = 0
  2. LIS/obs energy range doesn't cover needed range -> invalid likelihoods
  3. Third column (err) missing in obs file -> "not enough values to unpack"
  4. All helprop calls returning None -> check D0/m ranges are valid
"""

import argparse
import os
import sys
import time
import numpy as np
from multiprocessing import Pool, cpu_count

from interp import LogInterp
from helprop_runner import HelPropRunner
from surrogate_runner import CompositeSurrogateRunner, SurrogateRunner

# ---------- optional deps ----------
try:
    import emcee
    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False

try:
    import dynesty
    HAS_DYNESTY = True
except ImportError:
    HAS_DYNESTY = False

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ==================================================================
# Prior  (callable classes — picklable for multiprocessing)
# ==================================================================

DEFAULT_PARAM_RANGES = {
    "D0": (0.1, 50.0),
    "m": (-2.0, 2.0),
    "indexA": (0.5, 2.0),
    "indexB": (0.5, 2.0),
    "angle": (5.0, 30.0),
    "hcs-osc-amp": (0.0, 10.0),
    "hcs-osc-phase": (0.0, 360.0),
    "hcs-omega": (0.0, 4.0),
}
DEFAULT_PARAM_LABELS = {
    "D0": r"$D_0\;(10^{22}\,\mathrm{cm^2/s})$",
    "m": r"$m_{\mathrm{corot}}$",
    "indexA": r"$a$",
    "indexB": r"$b$",
    "angle": r"$\alpha_{\mathrm{HCS}}$",
    "hcs-osc-amp": r"$\Delta\alpha_{\mathrm{HCS}}$",
    "hcs-osc-phase": r"$\phi_{\mathrm{HCS}}$",
    "hcs-omega": r"$\omega_{\mathrm{HCS}}/\Omega$",
}
NDIM = 2
LABELS = [DEFAULT_PARAM_LABELS["D0"], DEFAULT_PARAM_LABELS["m"]]
PARAM_NAMES = ("D0", "m")
PARAM_RANGES = {
    "D0": DEFAULT_PARAM_RANGES["D0"],
    "m": DEFAULT_PARAM_RANGES["m"],
}


class _LogPrior:
    """Picklable uniform log-prior over named parameter ranges."""
    def __init__(self, param_names, param_ranges):
        self.param_names = tuple(param_names)
        self.param_ranges = {name: tuple(param_ranges[name]) for name in self.param_names}

    def __call__(self, theta):
        values = np.asarray(theta, dtype=float)
        if values.shape != (len(self.param_names),):
            return -np.inf
        if np.any(~np.isfinite(values)):
            return -np.inf
        for name, value in zip(self.param_names, values):
            low, high = self.param_ranges[name]
            if value < low or value > high:
                return -np.inf
        return 0.0


class _PriorTransform:
    """Picklable unit-cube -> named parameter transform for nested sampling."""
    def __init__(self, param_names, param_ranges):
        self.param_names = tuple(param_names)
        self.param_ranges = {name: tuple(param_ranges[name]) for name in self.param_names}

    def __call__(self, u):
        values = []
        for name, unit_value in zip(self.param_names, u):
            low, high = self.param_ranges[name]
            values.append(low + unit_value * (high - low))
        return np.asarray(values, dtype=float)


class _LogLikelihood:
    """Picklable log-likelihood with internal result cache."""
    def __init__(self, runner, E_obs, F_obs, err_obs, param_names):
        self.runner = runner
        self.E_obs = E_obs
        self.F_obs = F_obs
        self.err_obs = err_obs
        self.param_names = tuple(param_names)
        self._cache = {}

    def __call__(self, theta):
        theta = np.asarray(theta, dtype=float)
        key = tuple(round(float(value), 8) for value in theta)
        if key in self._cache:
            return self._cache[key]

        theta_options = {
            name: float(value)
            for name, value in zip(self.param_names, theta)
        }
        f_mod = self.runner.run(theta_options)
        if f_mod is None:
            self._cache[key] = -np.inf
            return -np.inf

        try:
            F_sim = f_mod(self.E_obs)
            if np.any(F_sim <= 0) or np.any(~np.isfinite(F_sim)):
                self._cache[key] = -np.inf
                return -np.inf
        except (ValueError, IndexError):
            self._cache[key] = -np.inf
            return -np.inf

        sigma = self.err_obs / self.F_obs
        resid = (np.log(self.F_obs) - np.log(F_sim)) / sigma
        ll = -0.5 * np.sum(resid ** 2)
        self._cache[key] = ll
        return ll

    @property
    def cache(self):
        return self._cache


class _TemperedProb:
    """Picklable tempered posterior: log_prior + beta * log_likelihood."""
    def __init__(self, log_likelihood, log_prior, beta=1.0):
        self._ll = log_likelihood
        self._lp = log_prior
        self._beta = beta

    def __call__(self, theta):
        lp = self._lp(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._beta * self._ll(theta)


def make_log_prior(param_names, param_ranges):
    return _LogPrior(param_names, param_ranges)


def make_prior_transform(param_names, param_ranges):
    return _PriorTransform(param_names, param_ranges)


def make_log_likelihood(runner, E_obs, F_obs, err_obs, param_names):
    return _LogLikelihood(runner, E_obs, F_obs, err_obs, param_names)


def _parse_fixed_option_items(items):
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected name=value, got {item!r}")
        name, value = item.split("=", 1)
        parsed[name] = float(value)
    return parsed


def _parse_range_option_items(items):
    parsed = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"expected name:min:max, got {item!r}")
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"expected name:min:max, got {item!r}")
        name, low, high = parts
        low_value = float(low)
        high_value = float(high)
        if high_value <= low_value:
            raise ValueError(f"range upper bound must exceed lower bound for {name}")
        parsed[name] = (low_value, high_value)
    return parsed


def _default_ranges_for(param_names):
    return {
        name: DEFAULT_PARAM_RANGES[name]
        for name in param_names
        if name in DEFAULT_PARAM_RANGES
    }


def _warm_start(log_prob, n_walkers, param_names, param_ranges, oversample=20):
    """Find n_walkers valid starting positions with finite log-probability.

    Generates candidates in random batches and keeps only those where
    the log-probability is finite.  Once enough valid points are found,
    the remaining slots are filled with small perturbations around the
    valid ones (so all walkers start in a good region).
    """
    valid = []
    lows = np.asarray([param_ranges[name][0] for name in param_names], dtype=float)
    highs = np.asarray([param_ranges[name][1] for name in param_names], dtype=float)
    for _ in range(100):  # at most 100 batches
        n_try = max(n_walkers * oversample - len(valid), 64)
        candidates = lows + np.random.random((n_try, len(param_names))) * (highs - lows)
        for c in candidates:
            lp = log_prob(c)
            if np.isfinite(lp):
                valid.append(c.copy())
                if len(valid) >= n_walkers:
                    break
        if len(valid) >= n_walkers:
            break

    if not valid:
        sys.exit("ERROR: could not find any starting point with finite "
                 "log-probability.  Check that HelProp works for the "
                 "given parameter ranges.")

    valid = np.array(valid)
    if len(valid) < n_walkers:
        # Fill remaining slots with small perturbations around valid points
        print(f"  Warm-start: found {len(valid)} valid points, "
              f"perturbing to fill {n_walkers - len(valid)} more")
        while len(valid) < n_walkers:
            idx = np.random.randint(len(valid))
            perturbed = valid[idx] + 0.01 * (highs - lows) * np.random.randn(len(param_names))
            perturbed = np.clip(perturbed, lows, highs)
            lp = log_prob(perturbed)
            if np.isfinite(lp):
                valid = np.vstack([valid, perturbed])

    np.random.shuffle(valid)
    print(f"  Warm-start: {min(len(valid), n_walkers)}/{n_walkers} "
          f"walkers initialized at finite-log-prob positions")
    return valid[:n_walkers]


# ==================================================================
# Parallel Tempering Sampler  (built on emcee)
# ==================================================================

class PTSampler:
    """Parallel Tempering MCMC using emcee at each temperature level.

    * Hot chains (large T) explore broadly and can cross energy barriers
      between modes.
    * Cold chain (T = 1) samples the target posterior.
    * Replica-exchange (swap) moves transfer information between levels.

    This is the recommended sampler when the posterior may be multi-modal.
    """

    def __init__(self, n_temps, n_walkers, ndim,
                 log_likelihood, log_prior, Tmax=1e3, pool=None):
        self.n_temps   = n_temps
        self.n_walkers  = n_walkers
        self.ndim       = ndim
        self._log_like  = log_likelihood
        self._log_prior = log_prior

        # geometric temperature schedule:  beta_0 = 1 (cold) ... beta_{T-1} ~ 1/Tmax (hot)
        self.betas = np.logspace(0, np.log10(1.0 / Tmax), n_temps)

        # one emcee sampler per temperature
        self._samplers = []
        for t in range(n_temps):
            moves = [emcee.moves.DEMove()]
            try:
                moves.append(emcee.moves.DESnookerMove())
            except AttributeError:
                pass

            s = emcee.EnsembleSampler(
                n_walkers, ndim,
                _TemperedProb(self._log_like, self._log_prior, self.betas[t]),
                moves=moves, pool=pool)
            self._samplers.append(s)

        # swap bookkeeping
        self._swap_accepted = np.zeros(max(n_temps - 1, 0))
        self._swap_proposed = np.zeros(max(n_temps - 1, 0))

        # cold-chain storage
        self._cold_chain   = None
        self._cold_logprob = None

    # ----------------------------------------------------------
    def run_mcmc(self, p0, n_steps, swap_interval=5, progress=True):
        """Run PT-MCMC.

        Parameters
        ----------
        p0 : ndarray, shape (n_temps, n_walkers, ndim)
            Initial walker positions.
        n_steps : int
            Total MCMC steps per temperature.
        swap_interval : int
            Steps between replica-exchange proposals.
        progress : bool
            Print progress every ~100 steps.
        """
        positions = [p0[t].copy() for t in range(self.n_temps)]
        all_chains   = []
        all_logprobs = []

        step = 0
        t_start = time.time()

        while step < n_steps:
            n_sub = min(swap_interval, n_steps - step)

            for t in range(self.n_temps):
                self._samplers[t].reset()
                state = self._samplers[t].run_mcmc(positions[t], n_sub,
                                                   progress=False)
                positions[t] = state.coords.copy()

            step += n_sub

            # store cold-chain segment
            all_chains.append(self._samplers[0].get_chain().copy())
            all_logprobs.append(self._samplers[0].get_log_prob().copy())

            # replica exchange
            self._propose_swaps(positions)

            if progress and step % max(100, swap_interval * 10) == 0:
                elapsed = time.time() - t_start
                rate = step / elapsed if elapsed > 0 else 0
                rates = self.swap_acceptance_rates
                print(f"  step {step}/{n_steps}  "
                      f"({rate:.1f} steps/s)  "
                      f"swap rate mean={np.mean(rates):.2f}")

        self._cold_chain   = np.concatenate(all_chains,   axis=0)
        self._cold_logprob = np.concatenate(all_logprobs, axis=0)

        elapsed = time.time() - t_start
        print(f"  PT-MCMC done: {n_steps} steps in {elapsed:.1f}s")

    # ----------------------------------------------------------
    def _propose_swaps(self, positions):
        """Propose replica exchanges between adjacent temperature levels."""
        for t in range(self.n_temps - 1):
            for w in range(self.n_walkers):
                self._swap_proposed[t] += 1

                lp0 = self._log_prior(positions[t][w])
                lp1 = self._log_prior(positions[t + 1][w])
                if not (np.isfinite(lp0) and np.isfinite(lp1)):
                    continue

                ll0 = self._log_like(positions[t][w])
                ll1 = self._log_like(positions[t + 1][w])
                if not (np.isfinite(ll0) and np.isfinite(ll1)):
                    continue

                log_alpha = (self.betas[t] - self.betas[t + 1]) * (ll1 - ll0)

                if np.log(np.random.uniform()) < log_alpha:
                    positions[t][w], positions[t + 1][w] = \
                        positions[t + 1][w].copy(), positions[t][w].copy()
                    self._swap_accepted[t] += 1

    # ----------------------------------------------------------
    @property
    def swap_acceptance_rates(self):
        return self._swap_accepted / np.maximum(self._swap_proposed, 1)

    def get_chain(self, flat=False, discard=0, thin=1):
        c = self._cold_chain[discard::thin]
        return c.reshape(-1, self.ndim) if flat else c

    def get_log_prob(self, flat=False, discard=0, thin=1):
        lp = self._cold_logprob[discard::thin]
        return lp.reshape(-1) if flat else lp


# ==================================================================
# emcee-only sampler
# ==================================================================

def run_emcee_sampler(n_walkers, n_steps, ndim, log_likelihood, log_prior, p0,
                      pool=None):
    moves = [emcee.moves.DEMove()]
    try:
        moves.append(emcee.moves.DESnookerMove())
    except AttributeError:
        pass

    log_prob = _TemperedProb(log_likelihood, log_prior)

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob,
                                    moves=moves, pool=pool)

    t0 = time.time()
    sampler.run_mcmc(p0, n_steps, progress=True)
    print(f"  emcee done in {time.time() - t0:.1f}s")
    return sampler


# ==================================================================
# dynesty sampler
# ==================================================================

def run_dynesty_sampler(log_likelihood, prior_transform_fn, ndim,
                        nlive=500, dlogz=0.5, pool=None):
    sampler = dynesty.DynamicNestedSampler(
        log_likelihood, prior_transform_fn, ndim=ndim,
        bound="multi",      # multi-ellipsoid  -> tracks multiple modes
        sample="rwalk",
        pool=pool,
    )
    sampler.run_nested(nlive_init=nlive, dlogz_init=dlogz,
                       print_progress=True)
    return sampler


# ==================================================================
# Post-processing
# ==================================================================

def analyze_results(samples, outdir, param_names, labels, sampler_name="pt"):
    """Generate posterior analysis and save all outputs."""
    os.makedirs(outdir, exist_ok=True)
    param_names = tuple(param_names)
    labels = list(labels)
    ndim = len(param_names)

    # --- raw samples ---
    np.savetxt(os.path.join(outdir, "samples.dat"), samples,
               header="  ".join(param_names))

    # --- statistics ---
    lines = [
        f"Posterior Summary  (sampler: {sampler_name})",
        f"N samples : {len(samples)}",
        "",
        "Parameter estimates  (median  + 68 % credible interval):",
    ]
    for index, name in enumerate(param_names):
        vals = samples[:, index]
        q16, q50, q84 = np.percentile(vals, [16, 50, 84])
        lines.append(f"  {name:10s} = {q50:.4f}  "
                     f"(+{q84 - q50:.4f} / -{q50 - q16:.4f})  "
                     f"[16%: {q16:.4f}  50%: {q50:.4f}  84%: {q84:.4f}]")
    lines.append("")
    for index, name in enumerate(param_names):
        vals = samples[:, index]
        lines.append(f"{name} range in samples : [{vals.min():.4f}, {vals.max():.4f}]")
    summary = "\n".join(lines)
    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(summary)

    # --- corner plot ---
    if HAS_CORNER and HAS_MPL:
        fig = corner.corner(samples, labels=labels,
                            quantiles=[0.16, 0.5, 0.84],
                            show_titles=True, title_kwargs={"fontsize": 12})
        fig.savefig(os.path.join(outdir, "corner.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {outdir}/corner.png")

    # --- trace plot ---
    if HAS_MPL:
        fig, axes = plt.subplots(ndim, 1, figsize=(10, 3 * ndim),
                                 squeeze=False)
        for i, ax in enumerate(axes.flat):
            ax.plot(samples[:, i], alpha=0.3, lw=0.5)
            ax.set_ylabel(labels[i])
        axes[-1, 0].set_xlabel("Sample index")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "trace.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {outdir}/trace.png")

    # --- 1-D posterior histograms ---
    if HAS_MPL:
        fig, axes = plt.subplots(1, ndim, figsize=(4 * ndim, 4),
                                 squeeze=False)
        for i, ax in enumerate(axes.flat):
            ax.hist(samples[:, i], bins=60, density=True, alpha=0.7,
                    color="steelblue")
            q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
            ax.axvline(q50, color="crimson", lw=2,
                       label=f"median = {q50:.3f}")
            ax.axvline(q16, color="crimson", ls="--", lw=1)
            ax.axvline(q84, color="crimson", ls="--", lw=1)
            ax.set_xlabel(labels[i])
            ax.set_ylabel("Posterior density")
            ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "posterior.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {outdir}/posterior.png")

    # --- binned 1-D posteriors as text (for external plotting) ---
    for index, name in enumerate(param_names):
        vals = samples[:, index]
        hist, edges = np.histogram(vals, bins=80, density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        np.savetxt(os.path.join(outdir, f"posterior_{name}.dat"),
                   np.column_stack([centres, hist]),
                   header=f"{name}  density")


# ==================================================================
# Main
# ==================================================================

def main():
    global NDIM, LABELS, PARAM_NAMES, PARAM_RANGES

    ap = argparse.ArgumentParser(
        description="MCMC Bayesian Analysis of HelProp modulation parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    ap.add_argument("--helprop", default="./HelProp",
                    help="Path to HelProp executable")
    ap.add_argument("--lis", default="../LocalProton",
                    help="LIS input spectrum file")
    ap.add_argument("--obs", default="data/obs_data.dat",
                    help="Observed data file  (E  F  err)")
    ap.add_argument("-o", "--outdir", default="chains",
                    help="Output directory for posterior samples & plots")

    # sampler choice
    ap.add_argument("--sampler", choices=["pt", "emcee", "dynesty"],
                    default="pt",
                    help="Sampler:  pt = parallel tempering (multi-peak), "
                         "emcee = ensemble MCMC, dynesty = nested sampling")

    # MCMC parameters
    ap.add_argument("--nwalkers", type=int, default=20,
                    help="Number of walkers per temperature")
    ap.add_argument("--nsteps",   type=int, default=2000,
                    help="Total MCMC steps")
    ap.add_argument("--nburn",    type=int, default=500,
                    help="Burn-in steps to discard")

    # PT-specific
    ap.add_argument("--ntemps",       type=int,   default=10,
                    help="Number of temperature levels (pt)")
    ap.add_argument("--Tmax",         type=float, default=1e3,
                    help="Maximum temperature (pt)")
    ap.add_argument("--swap-interval", type=int,   default=5,
                    help="Steps between swap proposals (pt)")

    # HelProp physics parameters (fixed across MCMC)
    ap.add_argument("--A",         type=int, default=1,
                    help="Nucleon number A")
    ap.add_argument("--Z",         type=int, default=1,
                    help="Charge number Z")
    ap.add_argument("--B0",        type=float, default=5.0,
                    help="Magnetic field at Earth (nT)")
    ap.add_argument("--polarity",  type=int, default=-1,
                    help="Magnetic polarity (-1 or +1)")
    ap.add_argument("--angle",     type=float, default=15.0,
                    help="HCS tilt angle (deg)")
    ap.add_argument("--hcs-osc-amp", type=float, default=0.0,
                    help="HCS tilt perturbation amplitude (deg)")
    ap.add_argument("--hcs-osc-phase", type=float, default=0.0,
                    help="HCS tilt perturbation phase (deg)")
    ap.add_argument("--hcs-omega", type=float, default=1.0,
                    help="HCS tilt perturbation angular frequency in units of HCS::Omega")
    ap.add_argument("--R0",        type=float, default=1.0,
                    help="Reference rigidity (GV)")
    ap.add_argument("--indexA",    type=float, default=1.0,
                    help="Diffusion index a")
    ap.add_argument("--indexB",    type=float, default=1.0,
                    help="Diffusion index b")

    # HelProp simulation parameters
    ap.add_argument("--nparticles", type=int, default=200,
                    help="HelProp --number  (particles per energy bin)")
    ap.add_argument("--nthread",    type=int, default=4,
                    help="HelProp --nthread")
    ap.add_argument("--hcs-table",  default="",
                    help="Path to precomputed HCS distance table")
    ap.add_argument("--extra-opts", nargs="*", default=[],
                    help="Any additional HelProp flags")
    ap.add_argument("--no-cache", action="store_true",
                    help="Disable HelProp result caching in the runner")
    ap.add_argument("--verbose", action="store_true",
                    help="Show HelProp command and stderr output")
    ap.add_argument("--backend", choices=["helprop", "surrogate"],
                    default="helprop",
                    help="Forward model backend for likelihood calls")
    ap.add_argument("--surrogate-model", default="",
                    help="Saved helprop_surrogate model for --backend surrogate")
    ap.add_argument("--surrogate-low-model", default="",
                    help="Low-energy saved surrogate model for dual-kernel surrogate mode")
    ap.add_argument("--surrogate-high-model", default="",
                    help="High-energy saved surrogate model for dual-kernel surrogate mode")
    ap.add_argument("--surrogate-split-energy", type=float, default=1.0,
                    help="Energy in GeV where dual surrogate kernels are joined")
    ap.add_argument("--surrogate-blend-dex", type=float, default=0.2,
                    help="Log10 energy width for smooth dual-kernel blending")
    ap.add_argument("--surrogate-param", action="append", default=[],
                    help="Extra learned surrogate parameter as name=value")
    ap.add_argument(
        "--sample-param",
        action="append",
        default=[],
        help="Parameter name to sample; repeat to override inferred/default names",
    )
    ap.add_argument(
        "--sample-range",
        action="append",
        default=[],
        help="Sampled parameter range as name:min:max; repeat to override defaults",
    )

    # parallelism
    ap.add_argument("--nproc", type=int, default=1,
                    help="Number of parallel worker processes "
                         "(1 = serial, 0 = use all CPU cores)")

    # observed data
    ap.add_argument("--rel-err",    type=float, default=0.05,
                    help="Relative error when obs file has only (E,F) columns")

    # reproducibility
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")

    # prior ranges
    ap.add_argument("--D0-range", type=float, nargs=2,
                    default=list(DEFAULT_PARAM_RANGES["D0"]), help="D0 prior bounds")
    ap.add_argument("--m-range",  type=float, nargs=2,
                    default=list(DEFAULT_PARAM_RANGES["m"]),  help="m_corot prior bounds")

    args = ap.parse_args()

    # dependency checks
    if args.sampler in ("pt", "emcee") and not HAS_EMCEE:
        sys.exit("ERROR: emcee is required for pt/emcee samplers.  "
                 "pip install emcee")
    if args.sampler == "dynesty" and not HAS_DYNESTY:
        sys.exit("ERROR: dynesty is required.  pip install dynesty")

    # load observed data — supports 2-col (E, F) or 3-col (E, F, err)
    print(f"Loading observed data from {args.obs} ...")
    raw = np.loadtxt(args.obs, unpack=True)
    if raw.shape[0] == 2:
        E_obs, F_obs = raw
        err_obs = args.rel_err * F_obs
        print(f"  No error column found — using {args.rel_err:.0%} relative errors")
    elif raw.shape[0] >= 3:
        E_obs, F_obs, err_obs = raw[0], raw[1], raw[2]
        print(f"  Error column found in file")
    else:
        sys.exit(f"ERROR: expected 2 or 3 columns in {args.obs}, got {raw.shape[0]}")
    print(f"  {len(E_obs)} data points, "
          f"E in [{E_obs.min():.3f}, {E_obs.max():.3f}] GeV")

    explicit_ranges = _parse_range_option_items(args.sample_range)

    if args.backend == "surrogate":
        dual_surrogate = bool(args.surrogate_low_model or args.surrogate_high_model)
        if dual_surrogate and not (args.surrogate_low_model and args.surrogate_high_model):
            sys.exit("ERROR: both --surrogate-low-model and --surrogate-high-model are required for dual-kernel surrogate mode")
        if not dual_surrogate and not args.surrogate_model:
            sys.exit("ERROR: --surrogate-model is required with --backend surrogate")
        surrogate_options = {
            "B0": args.B0,
            "angle": args.angle,
            "hcs-osc-amp": args.hcs_osc_amp,
            "hcs-osc-phase": args.hcs_osc_phase,
            "hcs-omega": args.hcs_omega,
            "indexA": args.indexA,
            "indexB": args.indexB,
        }
        surrogate_options.update(_parse_fixed_option_items(args.surrogate_param))
        if dual_surrogate:
            runner = CompositeSurrogateRunner(
                args.surrogate_low_model,
                args.surrogate_high_model,
                args.lis,
                split_energy=args.surrogate_split_energy,
                blend_dex=args.surrogate_blend_dex,
                fixed_options=surrogate_options,
                verbose=args.verbose,
            )
        else:
            runner = SurrogateRunner(args.surrogate_model, args.lis,
                                     fixed_options=surrogate_options,
                                     verbose=args.verbose)
        inferred_names = tuple(runner.learned_parameters())
    else:
        # set up runner — fixed physics + simulation options
        common_opts = [
            f"--A={args.A}", f"--Z={args.Z}",
            f"--B0={args.B0}", f"--polarity={args.polarity}",
            f"--angle={args.angle}", f"--R0={args.R0}",
            f"--hcs-osc-amp={args.hcs_osc_amp}",
            f"--hcs-osc-phase={args.hcs_osc_phase}",
            f"--hcs-omega={args.hcs_omega}",
            f"--indexA={args.indexA}", f"--indexB={args.indexB}",
            f"--number={args.nparticles}",
            f"--nthread={args.nthread}",
            f"--iotype=TXT",
        ]
        if args.hcs_table:
            common_opts.append(f"--hcs-table={args.hcs_table}")
        common_opts.extend(args.extra_opts)
        runner = HelPropRunner(args.helprop, args.lis, common_opts,
                               verbose=args.verbose, timeout=600,
                               use_cache=not args.no_cache)
        inferred_names = ("D0", "m")

    param_names = tuple(args.sample_param) if args.sample_param else inferred_names
    if not param_names:
        sys.exit("ERROR: no sampled parameters were selected")
    if args.backend == "helprop" and param_names != ("D0", "m"):
        sys.exit("ERROR: --backend helprop currently supports sampling only D0 and m")
    repeated_params = [name for name in param_names if param_names.count(name) > 1]
    if repeated_params:
        sys.exit(f"ERROR: duplicate sampled parameters: {', '.join(sorted(set(repeated_params)))}")
    param_ranges = _default_ranges_for(param_names)
    param_ranges["D0"] = tuple(args.D0_range)
    param_ranges["m"] = tuple(args.m_range)
    param_ranges.update(explicit_ranges)
    missing_ranges = [name for name in param_names if name not in param_ranges]
    if missing_ranges:
        sys.exit(f"ERROR: missing ranges for sampled parameters: {', '.join(missing_ranges)}")
    unused_ranges = set(param_ranges).difference(param_names)
    for name in unused_ranges:
        param_ranges.pop(name, None)

    PARAM_NAMES = param_names
    PARAM_RANGES = param_ranges
    NDIM = len(param_names)
    LABELS = [DEFAULT_PARAM_LABELS.get(name, name) for name in param_names]

    print("  Sampled parameters:")
    for name in param_names:
        low, high = param_ranges[name]
        print(f"    {name}: [{low:g}, {high:g}]")

    log_prior = make_log_prior(param_names, param_ranges)
    prior_transform = make_prior_transform(param_names, param_ranges)
    log_likelihood = make_log_likelihood(runner, E_obs, F_obs, err_obs, param_names)

    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # ---------- worker pool ----------
    nproc = args.nproc if args.nproc > 0 else cpu_count()
    pool = Pool(nproc) if nproc > 1 else None
    if pool is not None:
        print(f"  Parallel pool: {nproc} worker processes "
              f"(out of {cpu_count()} CPU cores)")

    # ---------- run sampler ----------
    print(f"\n{'=' * 50}")
    print(f"  Sampler : {args.sampler.upper()}")
    print(f"  Walkers : {args.nwalkers}    Steps : {args.nsteps}    "
          f"Burn-in : {args.nburn}"
          f"{'    Workers : ' + str(nproc) if pool else ''}")
    print(f"{'=' * 50}\n")

    try:
        # build a combined log_prob for warm-start screening
        _warm_prob = _TemperedProb(log_likelihood, log_prior, beta=1.0)

        if args.sampler == "pt":
            p0 = np.zeros((args.ntemps, args.nwalkers, NDIM))
            # warm-start each temperature independently
            for t in range(args.ntemps):
                p0[t] = _warm_start(_warm_prob, args.nwalkers,
                                    param_names, param_ranges)

            sampler = PTSampler(args.ntemps, args.nwalkers, NDIM,
                                log_likelihood, log_prior, Tmax=args.Tmax,
                                pool=pool)
            sampler.run_mcmc(p0, args.nsteps,
                             swap_interval=args.swap_interval, progress=True)

            samples = sampler.get_chain(flat=True, discard=args.nburn)

            print("\nSwap acceptance rates:")
            for t, r in enumerate(sampler.swap_acceptance_rates):
                print(f"  T{t} <-> T{t+1} : {r:.3f}")

        elif args.sampler == "emcee":
            p0 = _warm_start(_warm_prob, args.nwalkers,
                             param_names, param_ranges)

            sampler = run_emcee_sampler(args.nwalkers, args.nsteps, NDIM,
                                        log_likelihood, log_prior, p0,
                                        pool=pool)
            samples = sampler.get_chain(flat=True, discard=args.nburn)

            try:
                tau = sampler.get_autocorr_time(quiet=True)
                tau_summary = "  ".join(
                    f"{name}={value:.1f}" for name, value in zip(param_names, tau)
                )
                print(f"  Autocorrelation times: {tau_summary}")
            except Exception:
                pass

        elif args.sampler == "dynesty":
            sampler = run_dynesty_sampler(log_likelihood, prior_transform, NDIM,
                                          pool=pool)
            from dynesty import utils as dyfunc
            weights = np.exp(sampler.results.logwt - sampler.results.logz[-1])
            samples = dyfunc.resample_equal(sampler.results.samples, weights)

        # ---------- results ----------
        rstats = runner.stats()
        print(f"\nRunner stats:  calls={rstats['total_calls']}  "
              f"cache_hits={rstats['cache_hits']}  "
              f"hit_rate={rstats['hit_rate']:.2%}")

        print(f"\nAnalyzing {len(samples)} posterior samples ...")
        analyze_results(samples, args.outdir, param_names, LABELS, sampler_name=args.sampler)

    finally:
        # clean up worker pool (even on exception)
        if pool is not None:
            pool.close()
            pool.join()

    print(f"\nAll output written to  {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()
