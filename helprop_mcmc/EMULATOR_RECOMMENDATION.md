# HelProp Surrogate Model Recommendation

## Problem

HelProp is a Monte Carlo SDE integrator. One evaluation takes **minutes**.
MCMC needs **10^4–10^5** evaluations. Direct coupling is impractical.

Build a fast emulator f(D0, m, [B0, angle]) → modulated spectrum, then run MCMC
against the emulator (microseconds per evaluation).

---

## Recommended Architecture

### Gaussian Process with PCA Output Compression

This is the standard approach in astrophysics surrogate modeling (PICO for CMB,
CR propagation emulators). It is the best match for HelProp because:

- **Input is low-dimensional** (2–4 parameters) → GP excels here
- **Output is correlated** (flux at 20–30 energy bins along a smooth curve) → PCA
  compresses this to 3–5 coefficients with <1% reconstruction error
- **GP gives uncertainty for free** → can be folded into the likelihood as
  emulator noise, keeping the Bayesian framework honest
- **Training data is expensive** → GP is sample-efficient (50–500 runs suffice)
- **No GPU required** → runs on any server with CPU

```
                    PCA compression
  HelProp grid  ──────────────────────>  PCA coefficients
  (N runs × 20 bins)                    (N runs × K components, K≈3–5)

                    independent GP per component
  (D0, m, ...)   ──────────────────────>  predicted PCA coefficients  ŷ_k(x)

                    PCA reconstruction + GP variance
  ŷ_k(x)        ──────────────────────>  F_pred(E), σ_emu(E)
```

### Why not a neural network?

| Aspect | GP + PCA | Neural Network |
|--------|----------|----------------|
| Sample efficiency | 50–500 runs | 500–5000 runs |
| Uncertainty | Native (posterior variance) | Needs ensemble / MC-dropout / BNN |
| Training time | Seconds | Minutes–hours + tuning |
| Hyperparameter tuning | Kernel + likelihood, automatic | Architecture, LR, schedule, epochs |
| Extrapolation | Honest (variance explodes) | Silent overconfidence |
| Best for | <10 input params | >10 input params |

A neural network becomes competitive when you have >10 parameters or >10^4
training samples. For 2–4 HelProp parameters, GP is strictly better.

### Why not the polynomial emulator already in `bayesian/`?

The existing `train_emulator.py` already fits per-element GPs then exports a
polynomial approximation for Stan. This works but has limitations:

- Polynomial degree must be chosen a priori and is fixed globally
- No uncertainty quantification on the emulator itself
- Accuracy degrades at the edges of parameter space

The GP + PCA approach replaces the polynomial with a direct GP prediction
that carries uncertainty and adapts to the local curvature of the response
surface.

---

## Implementation Plan

### Phase 1 — Data Collection

Reuse `run_grid.py`. Generate a Latin Hypercube design over the parameter space.

```
2 params (D0, m):          5^2 = 25 runs  →  enough for GP
4 params (D0, m, B0, angle): 4^4 = 256 runs →  enough for GP
```

Each run produces a modulated spectrum (20–30 energy bins).
Save all spectra into a single matrix `Y` of shape `(N_runs, N_energy_bins)`,
along with the parameter matrix `X` of shape `(N_runs, N_params)`.

### Phase 2 — PCA Compression

```python
from sklearn.decomposition import PCA
import numpy as np

# Y: (N_runs, N_bins) — log-flux at each energy bin
log_Y = np.log(Y)
pca = PCA(n_components=0.999)   # keep 99.9% variance, typically K=3–5
Z = pca.fit_transform(log_Y)    # (N_runs, K) PCA scores
```

Record:
- `pca.mean_`, `pca.components_` for reconstruction
- `K` (number of retained components)
- Reconstruction error on a held-out test set

### Phase 3 — GP Training

One independent GP per PCA component, using GPyTorch (GPU-accelerated):

```python
import gpytorch
import torch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1])
        )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

# Train K independent GPs on columns of Z
# Input: X (parameter values, standardized)
# Output: Z[:, k] for each component k
```

ARD (Automatic Relevance Determination) RBF kernel automatically learns
which parameters matter most.

**Transform D0 to log-space before training** (flux scales roughly as a
power of D0, so log(D0) is closer to linear).

### Phase 4 — Emulator Prediction (used inside MCMC)

```python
def predict_spectrum(params_array):
    """params_array: [D0, m, B0, angle] → (F_pred, F_std)"""
    x = torch.tensor(standardize(params_array)).unsqueeze(0)
    z_pred = np.zeros(K)
    z_std  = np.zeros(K)
    for k in range(K):
        gps[k].eval()
        likelihoods[k].eval()
        with torch.no_grad():
            pred = likelihoods[k](gps[k](x))
        z_pred[k] = pred.mean.item()
        z_std[k]  = pred.stddev.item()

    log_flux = pca.inverse_transform(z_pred.reshape(1, -1))[0]

    # Uncertainty via error propagation through PCA
    # σ²_emu = Σ_k (σ_k² · PC_k²)
    components = pca.components_      # (K, N_bins)
    log_flux_var = np.sum(z_std[:, None]**2 * components**2, axis=0)
    return np.exp(log_flux), np.exp(log_flux) * np.sqrt(log_flux_var)
```

### Phase 5 — MCMC Likelihood with Emulator Uncertainty

```python
def log_likelihood(theta):
    F_pred, F_emu_std = predict_spectrum(theta)

    # Total uncertainty = observational + emulator
    sigma_total = np.sqrt(F_obs_err**2 + F_emu_std**2)

    resid = (np.log(F_obs) - np.log(F_pred)) / sigma_total
    return -0.5 * np.sum(resid**2)
```

The emulator uncertainty enters as an additive noise term. In regions where
the GP is uncertain (sparse training data), the likelihood is automatically
widened, preventing overconfident posterior inference.

---

## Software Stack

| Package | Role | Install |
|---------|------|---------|
| **gpytorch** | GP training + prediction | `pip install gpytorch` |
| **torch** | Tensor backend (CPU or GPU) | bundled with gpytorch |
| **scikit-learn** | PCA + preprocessing | `pip install scikit-learn` |
| **emcee** | MCMC sampler | `pip install emcee` |
| **numpy** | Array ops | already installed |

All are pure-Python with optional CUDA. No compilation needed.

### Minimal alternative (no gpytorch)

If you want zero non-standard dependencies beyond numpy/scipy:

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
```

sklearn's GP is CPU-only and slower for large training sets, but perfectly
adequate for N < 500 training points.

---

## Validation Protocol

Before trusting any MCMC results from the emulator:

1. **Leave-one-out cross-validation** on the GP: predict each training point
   using the remaining N-1 points. Plot predicted vs true log-flux. Require
   R² > 0.99.

2. **Test set validation**: hold out 10% of runs, train on 90%. Predict the
   held-out spectra. Plot residuals. The GP predictive variance should
   cover 95% of residuals within 2σ.

3. **Recovery test**: generate synthetic observed data from a known parameter
   vector using HelProp directly, then run MCMC with the emulator. The true
   parameters should lie within the 68% credible region.

4. **Convergence check**: if the MCMC posterior extends into regions of
   parameter space where GP uncertainty is large (> 5% of flux), add more
   training points in that region and retrain.

---

## File Layout

```
helprop_mcmc/
  emulator/
    train.py          # Phase 1–3: run grid, PCA, train GPs
    predict.py         # Phase 4: fast prediction API
    validate.py        # cross-validation and test-set checks
    saved/
      pca.pkl          # fitted PCA
      gp_k0.pth        # saved GP model for PCA component 0
      gp_k1.pth        # ...
      scaler.pkl       # input standardizer
  mcmc_analysis.py     # Phase 5: MCMC using emulator (modified)
```

---

## Summary

| What | Choice |
|------|--------|
| **Model** | GP (RBF + ARD kernel) per PCA component |
| **Output handling** | PCA on log-flux (K ≈ 3–5 components) |
| **Framework** | GPyTorch (GPU optional) or sklearn GP (CPU only) |
| **Training data** | 25–256 HelProp runs via LHS |
| **Uncertainty** | GP posterior variance → additive likelihood noise |
| **MCMC sampler** | emcee (existing), now microseconds per evaluation |
| **Validation** | LOO-CV + test set + parameter recovery |
