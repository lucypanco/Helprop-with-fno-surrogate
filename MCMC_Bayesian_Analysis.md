# MCMC Bayesian Analysis of HelProp

## 1. Objective

Fit HelProp modulation parameters **D₀** (diffusion coefficient) and **m_corot** (co-rotation factor) to observed cosmic ray spectra using MCMC Bayesian inference.

### Target Parameters

| Parameter     | HelProp Flag | Default | Unit                | Physical Meaning                          |
|---------------|-------------|---------|---------------------|-------------------------------------------|
| D₀            | `--D0`      | 5       | 10²² cm²/s         | Reference diffusion coefficient           |
| m_corot       | `--m`       | 0       | dimensionless       | Co-rotation factor in azimuthal drift     |

## 2. Overview

The workflow is entirely **Python-driven**. Each MCMC iteration:

1. Proposes a (D₀, m_corot) pair.
2. Runs `HelProp` as a subprocess with those parameters, using the **same given LIS input by inspec args** (no `--etoa` flag).
3. Reads the modulated output spectrum.
4. Uses **log-linear interpolation** (matching HelProp's `LogInterp`) to evaluate the model at the **observed energy points**.
5. Computes the **log-likelihood** from the residuals.
6. Accepts/rejects via the MCMC sampler.

No changes to HelProp source code are required.

## 3. Prerequisites

- **HelProp** built and executable (`./HelProp`). See `AGENTS.md` for build instructions.
- **Python 3.9+** with:
  ```
  numpy scipy emcee corner matplotlib
  ```
  Install via:
  ```bash
  pip install numpy scipy emcee corner matplotlib
  ```

## 4. HelProp Invocation

Each MCMC step calls HelProp with:

```bash
./HelProp [common_opts] <LIS_input> <output_spec>
```

Where **common_opts** must include the varying parameters:

```bash
--D0 <D0_val> --m <m_corot_val> [other_fixed_opts]
```

### Key Notes

- **No `--etoa` flag** is passed. HelProp uses the LIS energy grid as both TOA and ELIS (line 220-223 in `HelProp.cc`), producing output at the same energy points as the LIS input.
- **The LIS input is fixed** across all MCMC iterations. It defines the energy grid for the modulated output.
- **Output format**: Use `--iotype TXT` (default) or `--iotype CSV` for easy parsing in Python.

### Fixed Options (should match your experiment)

These should be set once and kept constant across all MCMC runs:

| Flag             | Description                                      |
|------------------|--------------------------------------------------|
| `--A`            | Nucleon number (e.g., 1 for protons)             |
| `--Z`            | Charge number (e.g., 1 for protons)              |
| `--B0`           | Magnetic field strength at Earth (nT)            |
| `--polarity`     | Magnetic field polarity (-1 or +1)               |
| `--angle`        | HCS tilt angle (deg)                            |
| `--R0`           | Reference rigidity (GV)                          |
| `--indexA`       | Diffusion index a                               |
| `--indexB`       | Diffusion index b                               |
| `--number`       | Simulation particles per bin                     |
| `--nthread`      | Number of threads                                |
| `--seed`         | Global random seed (for reproducibility)         |
| `--hcs-table`    | Path to HCS distance table (if used)             |

## 5. Python Architecture

```
helprop_mcmc/
├── mcmc_analysis.py     # Main script: MCMC sampler + likelihood
├── helprop_runner.py   # Wrapper: builds cmd, runs subprocess, parses output
├── interp.py           # Log-linear interpolation (replicate LogInterp)
├── data/
│   ├── lis_input.dat   # LIS spectrum (E, F) in same format as HelProp input
│   └── obs_data.dat    # Observed TOA spectrum (E_obs, F_obs, err_obs) in GeV
└── chains/             # Output directory for MCMC chains
```

### 5.1 `interp.py` — Log-Linear Interpolation

Replicates HelProp's `LogInterp` to interpolate the modulated output at observed energies.

```python
import numpy as np

class LogInterp:
    def __init__(self, x, y):
        self.xlog = np.log(x)
        self.ylog = np.log(y)

    def __call__(self, x):
        x = np.atleast_1d(np.asarray(x))
        idx = np.searchsorted(self.xlog, np.log(x))
        idx = np.clip(idx, 1, len(self.xlog) - 1)
        i0, i1 = idx - 1, idx
        m = (self.ylog[i1] - self.ylog[i0]) / (self.xlog[i1] - self.xlog[i0])
        return np.exp(self.ylog[i0] + m * (np.log(x) - self.xlog[i0]))
```

**Why?** HelProp's output energies match the LIS grid (no `--etoa`), but observed data points rarely align exactly. Interpolation is needed to evaluate the model at `E_obs`.

### 5.2 `helprop_runner.py` — Subprocess Wrapper

```python
import subprocess
import numpy as np
from interp import LogInterp

def run_helprop(D0, m_corot, lis_input, out_spec,
                common_opts=None, helprop_bin="./HelProp"):
    """
    Run HelProp with given (D0, m_corot), return modulated spectrum.

    Returns: LogInterp object f_mod(E) representing the modulated spectrum.
    """
    cmd = [
        helprop_bin,
        f"--D0={D0}", f"--m={m_corot}",
        lis_input, out_spec
    ]
    if common_opts:
        cmd = cmd[:1] + common_opts + cmd[1:]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"HelProp failed: {result.stderr}")

    # Parse output: E and F columns
    E_mod, F_mod = np.loadtxt(out_spec, unpack=True)
    return LogInterp(E_mod, F_mod)
```

**Notes:**
- `common_opts` is a list of fixed flags, e.g. `["--A=1", "--Z=1", "--B0=5", "--polarity=-1", "--angle=15", ...]`.
- `out_spec` is a temporary file path (use `tempfile` module or overwrite the same file each call).
- HelProp writes output with energies in GeV by default (unit conversion happens inside HelProp, `eunit = GeV`).

### 5.3 `mcmc_analysis.py` — MCMC Core

```python
import numpy as np
import emcee
from helprop_runner import run_helprop
from interp import LogInterp

# Load observed data: E_obs (GeV), F_obs, err_obs
E_obs, F_obs, err_obs = np.loadtxt("data/obs_data.dat", unpack=True)

# Fixed HelProp options
COMMON_OPTS = [
    "--A=1", "--Z=1", "--B0=5", "--polarity=-1", "--angle=15",
    "--R0=1", "--indexA=1", "--indexB=1",
    "--number=1000", "--nthread=4", "--seed=42",
    "--iotype=TXT"
]

def log_prior(theta):
    D0, m_corot = theta
    if 0.1 <= D0 <= 50 and -2.0 <= m_corot <= 2.0:
        return 0.0
    return -np.inf

def log_likelihood(theta):
    D0, m_corot = theta
    try:
        f_mod = run_helprop(D0, m_corot,
                            lis_input="data/lis_input.dat",
                            out_spec="/tmp/helprop_out.dat",
                            common_opts=COMMON_OPTS)
    except Exception:
        return -np.inf

    F_sim = f_mod(E_obs)
    resid = (np.log(F_obs) - np.log(F_sim)) / (err_obs / F_obs)
    return -0.5 * np.sum(resid**2)

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

# MCMC setup
ndim, nwalkers = 2, 16
p0 = np.random.uniform([0.1, -2.0], [50.0, 2.0], size=(nwalkers, ndim))

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
sampler.run_mcmc(p0, 10000, progress=True)

# Analysis
samples = sampler.get_chain(flat=True)
np.savetxt("chains/samples.dat", samples)
```

### 5.4 Log-Likelihood Choice

The above uses:

```
log L = -0.5 * Σ [(log F_obs - log F_sim) / (err/F)]²
```

This is a **Gaussian log-likelihood on log-flux**, which is appropriate when errors are multiplicative (relative errors). Adjust based on your error model:

- **Absolute errors**: `resid = (F_obs - F_sim) / err_obs`
- **Asymmetric errors**: use `scipy.stats.norm` with different sigmas above/below

## 6. Performance Considerations

### 6.1 HelProp is the Bottleneck

Each MCMC step runs the full particle simulation. Key optimizations:

| Strategy                        | Expected Speedup |
|---------------------------------|-----------------|
| Use `--nthread > 1`              | ~linear in threads |
| Precompute Green's Function     | See Section 6.2 |
| Reduce `--number` during burn-in | ~linear in particles |
| Cache Liss input read           | N/A (fixed) |

### 6.2 Precomputed Green's Function (Advanced)

If D₀ and m_corot are the **only** varying parameters, the Green's function matrix `weight[i][j]` in `HelProp.cc` (line 246) depends on both `D0` and `m_corot` via the particle trajectories. It **cannot** be cached separately — it must be recomputed for each parameter set.

However, `--number` can be **reduced** during burn-in (e.g., 100-500 particles) and **increased** for the final production run (1000+).

### 6.3 Parallel MCMC

`emcee` runs multiple walkers in parallel by default. For N walkers on M threads:

```
Total threads used = N * M  (HelProp threads per walker)
```

Example: 16 walkers × 4 threads = 64 total threads. Set `OMP_NUM_THREADS=1` inside each subprocess call to avoid oversubscription:

```python
import os
os.environ["OMP_NUM_THREADS"] = "1"
cmd = [...]  # HelProp inherits this env var
```

### 6.4 Temporary File I/O

Writing/reading a single output file per MCMC step is cheap relative to the simulation cost. No additional optimization needed.

## 7. Workflow Summary

```
1. Prepare data files:
   data/lis_input.dat   — LIS spectrum (E, F)
   data/obs_data.dat    — Observed spectrum (E_obs, F_obs, err_obs)

2. Configure COMMON_OPTS in mcmc_analysis.py

3. Run burn-in (low statistics):
   --number=200, nwalkers=16, steps=1000

4. Run production (high statistics):
   --number=2000, nwalkers=32, steps=10000

5. Analyze chains:
   python -c "import corner; ..."
```

## 8. Example Data Files

### `data/lis_input.dat` (TXT format, matches HelProp spec format)

```
# E F
0.1  1.0e4
0.2  5.0e3
0.5  2.0e3
1.0  1.0e3
2.0  5.0e2
5.0  2.0e2
```

### `data/obs_data.dat` (E in GeV, F in same units as LIS, err as absolute error)

```
# E_obs  F_obs  err_obs
0.15    800.0  50.0
0.5     400.0  30.0
1.0     200.0  20.0
2.0     100.0  15.0
```

## 9. Expected Output

- `chains/samples.dat` — flattened chain of shape `(n_steps * nwalkers, 2)`
- Corner plot: `corner.corner(samples, labels=["D0", "m_corot"])`
- Print summary: mean, std, 16th/50th/84th percentiles of the posterior

## 10. Troubleshooting

| Symptom                              | Likely Cause                             | Fix                                     |
|--------------------------------------|------------------------------------------|----------------------------------------|
| HelProp segfaults                    | `--number` too high for available RAM     | Reduce `--number`                      |
| Interpolation returns NaN             | Observed E outside LIS energy range      | Extend LIS grid or filter obs data     |
| All log_probability = -inf            | Prior rejection or HelProp failure       | Check stderr of subprocess             |
| MCMC doesn't converge                | Too few walkers or steps                 | Increase walkers/steps; check trace   |
| Slow chains (hours+)                 | Too many particles per step              | Reduce `--number` for burn-in          |