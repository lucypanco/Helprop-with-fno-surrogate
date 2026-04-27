# HelProp Bayesian Inference Module

This module performs Bayesian inference of HelProp parameters -- **D0** (diffusion coefficient), **m** (co-rotation factor), **B0** (magnetic field strength), and **angle** (HCS tilt angle) -- from observed cosmic ray data. Because HelProp's SDE integrator is stochastic and non-differentiable, the pipeline uses an **emulation-based** approach: pre-compute Green function matrices on a parameter grid, fit surrogate models (GPs + polynomials), and run MCMC inside CmdStan using the fast polynomial emulator as the forward model.

Any subset of the four parameters can be inferred. Non-inferred parameters are fixed via tight priors at their default values.

## Prerequisites

### Build HelProp

```bash
cmake -B build
cmake --build build
```

The `HelProp` binary must exist at the project root.

### Python dependencies

```bash
pip install numpy scipy scikit-learn matplotlib
```

### CmdStan (for MCMC inference)

**Option A -- cmdstanpy (recommended):**

```bash
pip install cmdstanpy
python -m cmdstanpy install_cmdstan
```

**Option B -- manual CmdStan installation:**

Install CmdStan following [https://mc-stan.org/users/interfaces/cmdstan](https://mc-stan.org/users/interfaces/cmdstan), and make sure `~/.cmdstan/cmdstan-*/bin` is on your `PATH`.

### BSON I/O (optional)

```bash
pip install pymongo
```

### Corner plots (optional)

```bash
pip install corner
```

## Pipeline Overview

```
  Phase 1: Prepare input data
           (observed spectrum + LIS)
                  |
  Phase 2: Grid simulation
           (run HelProp at many parameter points)
                  |
  Phase 3: Emulator training
           (GP fit -> polynomial coefficients for Stan)
                  |
  Phase 4: Prepare Stan data
           (combine emulator + observations + LIS into one JSON)
                  |
  Phase 5: MCMC inference
           (CmdStan samples the posterior)
                  |
  Phase 6: Post-processing
           (corner plots, posterior predictive, summary JSON)
```

You can run the entire pipeline in one command, or execute each phase individually.

## Quick Start (One Command)

```bash
# 4-parameter inference with placeholder data
python -m bayesian.run_inference --infer D0,m,B0,angle --n-points 3

# 2-parameter inference with real data
python -m bayesian.run_inference \
    --obs-file data/ams02_proton.csv --lis-file data/vos_potgieter_lis.csv \
    --infer D0,m --n-points 5

# Skip grid simulation (reuse existing output), just re-run MCMC
python -m bayesian.run_inference --skip-grid --infer D0,m

# Full inference with more samples
python -m bayesian.run_inference \
    --infer D0,m,B0,angle --n-points 4 --chains 4 --iterations 4000
```

## CLI Options for `run_inference`

| Flag | Default | Description |
|------|---------|-------------|
| `--infer` | `D0,m` | Comma-separated list of parameters to infer (D0, m, B0, angle) |
| `--obs-file` | None | Path to observed spectrum file (CSV/TXT, 2 or 3 columns) |
| `--lis-file` | None | Path to LIS spectrum file |
| `--n-points` | 4 | Design points per inferred dimension |
| `--method` | `lhs` | Design method: `lhs` or `grid` |
| `--number` | 1000 | Particles per TOA energy bin |
| `--nthread` | 4 | Thread count |
| `--poly-degree` | 2 | Polynomial degree for emulator |
| `--chains` | 4 | MCMC chains |
| `--iterations` | 2000 | MCMC iterations per chain |
| `--skip-grid` | false | Skip grid simulation |
| `--skip-gp` | false | Skip GP training |
| `--skip-postprocess` | false | Skip post-MCMC diagnostics |
| `--B0` | 5 | Fixed B0 value when not inferred |
| `--angle` | 15 | Fixed HCS tilt angle when not inferred |
| `--A` | 1 | Atomic mass number |
| `--Z` | 1 | Atomic number |
| `--polarity` | -1 | Solar magnetic polarity |
| `--R0` | 1 | Rigidity at 1 AU (GV) |
| `--indexA` | 1 | Diffusion index A |
| `--indexB` | 1 | Diffusion index B |

## Observed Data Formats

### 2-column (E, F)
```
# E F
1.00000000e-01 2.50000000e+01
2.00000000e-01 1.80000000e+01
```

### 3-column (E, F, F_err)
```
# E F F_err
1.00000000e-01 2.50000000e+01 2.50000000e+00
2.00000000e-01 1.80000000e+01 1.80000000e+00
```

CSV and TXT formats are both supported. If no `--obs-file` is provided, the pipeline looks for a project-root `Output` file, then falls back to synthetic data.

## Step-by-Step Usage

### Phase 2 -- Grid Simulation

```bash
# 4-param LHS: 4^4 = 256 design points
python -m bayesian.run_grid --infer D0,m,B0,angle --n-points 4

# 2-param regular grid: 5^2 = 25 points
python -m bayesian.run_grid --infer D0,m --n-points 5 --method grid
```

### Phase 3 -- Emulator Training

```bash
python -m bayesian.train_emulator --infer D0,m,B0,angle
python -m bayesian.train_emulator --infer D0,m --skip-gp  # re-export only
```

### Phase 4 -- Prepare Stan Data

Done automatically by `run_inference`. Output: `bayesian/stan/helprop_data.json`.

### Phase 5 -- MCMC Inference

Done automatically by `run_inference`. Output: posterior CSV files in `bayesian/stan/`.

### Phase 6 -- Post-Processing

```bash
python -m bayesian.postprocess --stan-dir bayesian/stan --infer D0,m,B0,angle
```

Outputs:
- `posterior_summary.json` -- parameter estimates + credible intervals + R-hat/ESS
- `corner_plot.png` -- posterior joint distribution
- `posterior_predictive.png` -- predicted vs observed spectrum

## Configuration

All configurable parameters live in `bayesian/config.py`:

| Variable | Description |
|----------|-------------|
| `PARAM_RANGES` | Min/max for D0, m, B0, angle |
| `PARAM_TRANSFORMS` | Transforms to emulator space (log for D0, identity for rest) |
| `PRIORS` | Prior distributions for Stan model |
| `INFERABLE_PARAMS` | List of all inferable parameter names |
| `SIM_DEFAULTS` | Default HelProp CLI arguments |
| `GRID_DEFAULTS` | Grid design settings |
| `EMULATOR_DEFAULTS` | Emulator settings (poly_degree, n_predict) |

## Stan Model

The Stan model at `bayesian/stan/helprop_bayes.stan` implements:

1. **Priors:** D0 ~ LogNormal(1.6, 0.7), m ~ Normal(0, 1), B0 ~ Normal(5, 1.5), angle ~ Normal(15, 10). Non-inferred parameters get very tight priors (sigma=0.001) to fix them.
2. **Forward model:** N-variate polynomial emulator with exponent table (passed as data). Evaluates the Green function weight matrix, then convolves with the LIS using the momentum-space Jacobian.
3. **Energy interpolation:** Log-linear interpolation between bracketing ETOA bins (binary search) instead of nearest-neighbor.
4. **Likelihood:** Log-normal: `log(F_obs) ~ Normal(log(F_model), F_err)`
5. **Generated quantities:** Posterior predictive flux at observed energies

## File Layout

```
project_root/
  HelProp                          # compiled binary
  bayesian/
    __init__.py
    config.py                      # all configurable parameters
    io.py                          # format-aware readers/writers (TXT/CSV/BSON)
    data_loader.py                 # observed data + LIS loading (file or placeholder)
    run_grid.py                    # Phase 2: design + HelProp grid runner
    train_emulator.py              # Phase 3: GP fit + polynomial export
    run_inference.py               # End-to-end pipeline orchestrator
    postprocess.py                 # Phase 6: post-MCMC diagnostics + plots
    output/
      manifest.json                # grid index: params, filename, iotype
      matrix_*.csv                 # Green function matrices
    emulators/
      emulators.pkl                # serialized GP models
      stan_data_poly.json          # polynomial coefficients + exponent table
    stan/
      helprop_bayes.stan           # Stan model
      helprop_data.json            # assembled Stan data
      posterior_summary.json       # parameter estimates + convergence
      corner_plot.png              # posterior joint distribution
      posterior_predictive.png     # predicted vs observed spectrum
```

## Tips

- **Start small.** Use `--infer D0,m --n-points 3` for a quick test before scaling to 4 parameters.
- **4-param grids are large.** `--n-points 4` with 4 params = 256 simulations. Plan accordingly.
- **Polynomial degree.** Degree 2 is the new default (15 coeffs for 4 params). Higher degrees may cause Stan sampling issues.
- **Resume interrupted grids.** The grid runner skips existing files. Just re-run the same command.
- **Check convergence.** After MCMC, verify R-hat < 1.1 and ESS > 100 for all parameters.
