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
  python mcmc_analysis.py --helprop ../HelProp --lis ../LocalProton --obs data/obs_data.dat
  python mcmc_analysis.py --sampler pt   --ntemps 10 --nwalkers 20 --nsteps 2000
  python mcmc_analysis.py --sampler emcee --nwalkers 32 --nsteps 5000
  python mcmc_analysis.py --sampler dynesty

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

from interp import LogInterp
from helprop_runner import HelPropRunner

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
# Prior
# ==================================================================

D0_RANGE = (0.1, 50.0)
M_RANGE  = (-2.0, 2.0)
NDIM     = 2
LABELS   = [r"$D_0\;(10^{22}\,\mathrm{cm^2/s})$", r"$m_{\mathrm{corot}}$"]


def log_prior(theta):
    D0, m = theta
    if D0_RANGE[0] <= D0 <= D0_RANGE[1] and M_RANGE[0] <= m <= M_RANGE[1]:
        return 0.0
    return -np.inf


def prior_transform(u):
    """Unit-cube -> parameter space (uniform prior, for nested sampling)."""
    D0 = D0_RANGE[0] + u[0] * (D0_RANGE[1] - D0_RANGE[0])
    m  = M_RANGE[0]  + u[1] * (M_RANGE[1]  - M_RANGE[0])
    return np.array([D0, m])


# ==================================================================
# Likelihood
# ==================================================================

def make_log_likelihood(runner, E_obs, F_obs, err_obs):
    """Return a log-likelihood closure with an internal value cache."""
    _cache = {}

    def log_likelihood(theta):
        D0, m = theta
        key = (round(D0, 8), round(m, 8))
        if key in _cache:
            return _cache[key]

        f_mod = runner.run(D0, m)
        if f_mod is None:
            _cache[key] = -np.inf
            return -np.inf

        try:
            F_sim = f_mod(E_obs)
            if np.any(F_sim <= 0) or np.any(~np.isfinite(F_sim)):
                _cache[key] = -np.inf
                return -np.inf
        except (ValueError, IndexError):
            _cache[key] = -np.inf
            return -np.inf

        # Gaussian log-likelihood on log-flux (multiplicative errors)
        sigma = err_obs / F_obs          # relative error
        resid = (np.log(F_obs) - np.log(F_sim)) / sigma
        ll = -0.5 * np.sum(resid ** 2)
        _cache[key] = ll
        return ll

    log_likelihood.cache = _cache        # expose for introspection
    return log_likelihood


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
                 log_likelihood, log_prior, Tmax=1e3):
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
            beta = self.betas[t]

            def _make_tp(b):
                def tp(theta):
                    lp = self._log_prior(theta)
                    if not np.isfinite(lp):
                        return -np.inf
                    ll = self._log_like(theta)
                    return lp + b * ll
                return tp

            moves = [emcee.moves.DEMove()]
            try:
                moves.append(emcee.moves.DESnookerMove())
            except AttributeError:
                pass

            s = emcee.EnsembleSampler(n_walkers, ndim, _make_tp(beta),
                                      moves=moves)
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

def run_emcee_sampler(n_walkers, n_steps, log_likelihood, log_prior, p0):
    moves = [emcee.moves.DEMove()]
    try:
        moves.append(emcee.moves.DESnookerMove())
    except AttributeError:
        pass

    def log_prob(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    sampler = emcee.EnsembleSampler(n_walkers, NDIM, log_prob, moves=moves)

    t0 = time.time()
    sampler.run_mcmc(p0, n_steps, progress=True)
    print(f"  emcee done in {time.time() - t0:.1f}s")
    return sampler


# ==================================================================
# dynesty sampler
# ==================================================================

def run_dynesty_sampler(log_likelihood, prior_transform_fn,
                        nlive=500, dlogz=0.5):
    sampler = dynesty.DynamicNestedSampler(
        log_likelihood, prior_transform_fn, ndim=NDIM,
        bound="multi",      # multi-ellipsoid  -> tracks multiple modes
        sample="rwalk",
    )
    sampler.run_nested(nlive_init=nlive, dlogz_init=dlogz,
                       print_progress=True)
    return sampler


# ==================================================================
# Post-processing
# ==================================================================

def analyze_results(samples, outdir, sampler_name="pt"):
    """Generate posterior analysis and save all outputs."""
    os.makedirs(outdir, exist_ok=True)

    # --- raw samples ---
    np.savetxt(os.path.join(outdir, "samples.dat"), samples,
               header="D0  m_corot")

    D0_s = samples[:, 0]
    m_s  = samples[:, 1]

    # --- statistics ---
    lines = [
        f"Posterior Summary  (sampler: {sampler_name})",
        f"N samples : {len(samples)}",
        "",
        "Parameter estimates  (median  + 68 % credible interval):",
    ]
    for name, vals in [("D0", D0_s), ("m_corot", m_s)]:
        q16, q50, q84 = np.percentile(vals, [16, 50, 84])
        lines.append(f"  {name:10s} = {q50:.4f}  "
                     f"(+{q84 - q50:.4f} / -{q50 - q16:.4f})  "
                     f"[16%: {q16:.4f}  50%: {q50:.4f}  84%: {q84:.4f}]")
    lines += [
        "",
        f"D0 range in samples     : [{D0_s.min():.4f}, {D0_s.max():.4f}]",
        f"m_corot range in samples : [{m_s.min():.4f}, {m_s.max():.4f}]",
    ]
    summary = "\n".join(lines)
    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(summary)

    # --- corner plot ---
    if HAS_CORNER and HAS_MPL:
        fig = corner.corner(samples, labels=LABELS,
                            quantiles=[0.16, 0.5, 0.84],
                            show_titles=True, title_kwargs={"fontsize": 12})
        fig.savefig(os.path.join(outdir, "corner.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {outdir}/corner.png")

    # --- trace plot ---
    if HAS_MPL:
        fig, axes = plt.subplots(NDIM, 1, figsize=(10, 3 * NDIM),
                                 squeeze=False)
        for i, ax in enumerate(axes.flat):
            ax.plot(samples[:, i], alpha=0.3, lw=0.5)
            ax.set_ylabel(LABELS[i])
        axes[-1, 0].set_xlabel("Sample index")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "trace.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {outdir}/trace.png")

    # --- 1-D posterior histograms ---
    if HAS_MPL:
        fig, axes = plt.subplots(1, NDIM, figsize=(4 * NDIM, 4),
                                 squeeze=False)
        for i, ax in enumerate(axes.flat):
            ax.hist(samples[:, i], bins=60, density=True, alpha=0.7,
                    color="steelblue")
            q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
            ax.axvline(q50, color="crimson", lw=2,
                       label=f"median = {q50:.3f}")
            ax.axvline(q16, color="crimson", ls="--", lw=1)
            ax.axvline(q84, color="crimson", ls="--", lw=1)
            ax.set_xlabel(LABELS[i])
            ax.set_ylabel("Posterior density")
            ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "posterior.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {outdir}/posterior.png")

    # --- binned 1-D posteriors as text (for external plotting) ---
    for name, vals in [("D0", D0_s), ("m_corot", m_s)]:
        hist, edges = np.histogram(vals, bins=80, density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        np.savetxt(os.path.join(outdir, f"posterior_{name}.dat"),
                   np.column_stack([centres, hist]),
                   header=f"{name}  density")


# ==================================================================
# Main
# ==================================================================

def main():
    global D0_RANGE, M_RANGE

    ap = argparse.ArgumentParser(
        description="MCMC Bayesian Analysis of HelProp (D0, m_corot)",
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

    # observed data
    ap.add_argument("--rel-err",    type=float, default=0.05,
                    help="Relative error when obs file has only (E,F) columns")

    # prior ranges
    ap.add_argument("--D0-range", type=float, nargs=2,
                    default=list(D0_RANGE), help="D0 prior bounds")
    ap.add_argument("--m-range",  type=float, nargs=2,
                    default=list(M_RANGE),  help="m_corot prior bounds")

    args = ap.parse_args()

    # update global prior bounds
    D0_RANGE = tuple(args.D0_range)
    M_RANGE  = tuple(args.m_range)

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

    # set up runner — fixed physics + simulation options
    common_opts = [
        f"--A={args.A}", f"--Z={args.Z}",
        f"--B0={args.B0}", f"--polarity={args.polarity}",
        f"--angle={args.angle}", f"--R0={args.R0}",
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
    log_likelihood = make_log_likelihood(runner, E_obs, F_obs, err_obs)

    np.random.seed(args.seed if args.seed else 42)
    os.makedirs(args.outdir, exist_ok=True)

    # ---------- run sampler ----------
    print(f"\n{'=' * 50}")
    print(f"  Sampler : {args.sampler.upper()}")
    print(f"  Walkers : {args.nwalkers}    Steps : {args.nsteps}    "
          f"Burn-in : {args.nburn}")
    print(f"{'=' * 50}\n")

    if args.sampler == "pt":
        p0 = np.zeros((args.ntemps, args.nwalkers, NDIM))
        for t in range(args.ntemps):
            p0[t, :, 0] = np.random.uniform(*D0_RANGE, size=args.nwalkers)
            p0[t, :, 1] = np.random.uniform(*M_RANGE,  size=args.nwalkers)

        sampler = PTSampler(args.ntemps, args.nwalkers, NDIM,
                            log_likelihood, log_prior, Tmax=args.Tmax)
        sampler.crun_mcm(p0, args.nsteps,
                         swap_interval=args.swap_interval, progress=True)

        samples = sampler.get_chain(flat=True, discard=args.nburn)

        print("\nSwap acceptance rates:")
        for t, r in enumerate(sampler.swap_acceptance_rates):
            print(f"  T{t} <-> T{t+1} : {r:.3f}")

    elif args.sampler == "emcee":
        p0 = np.zeros((args.nwalkers, NDIM))
        p0[:, 0] = np.random.uniform(*D0_RANGE, size=args.nwalkers)
        p0[:, 1] = np.random.uniform(*M_RANGE,  size=args.nwalkers)

        sampler = run_emcee_sampler(args.nwalkers, args.nsteps,
                                    log_likelihood, log_prior, p0)
        samples = sampler.get_chain(flat=True, discard=args.nburn)

        try:
            tau = sampler.get_autocorr_time(quiet=True)
            print(f"  Autocorrelation times: D0={tau[0]:.1f}  "
                  f"m_corot={tau[1]:.1f}")
        except Exception:
            pass

    elif args.sampler == "dynesty":
        sampler = run_dynesty_sampler(log_likelihood, prior_transform)
        from dynesty import utils as dyfunc
        weights = np.exp(sampler.results.logwt - sampler.results.logz[-1])
        samples = dyfunc.resample_equal(sampler.results.samples, weights)

    # ---------- results ----------
    rstats = runner.stats()
    print(f"\nRunner stats:  calls={rstats['total_calls']}  "
          f"cache_hits={rstats['cache_hits']}  "
          f"hit_rate={rstats['hit_rate']:.2%}")

    print(f"\nAnalyzing {len(samples)} posterior samples ...")
    analyze_results(samples, args.outdir, sampler_name=args.sampler)

    print(f"\nAll output written to  {os.path.abspath(args.outdir)}/")


if __name__ == "__main__":
    main()
