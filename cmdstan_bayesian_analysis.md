# Bayesian Inference of m and D0 with CmdStan

## The Parameters

### D0 (Reference Diffusion Coefficient)
- Already a `particle` member variable (`particle.h:28`)
- Already configurable via `--D0` CLI option in units of 1e22 cm²/s (default: 5)
- Directly scales `Kpara0()`: `Kpara0 = D0 * V_p / c_speed * ...`

### m (Rigidity Smoothness Index)
- **Hardcoded** as `const double m = 3` in `Kpara0()` (`particle.cc:69`)
- Controls the smoothness of the transition between rigidity regimes in the diffusion coefficient:
  ```
  Kpara0 = D0 * V_p / c_speed * (r0^indexA) * ((r0^m + rk^m) / (1 + rk^m))^((indexB - indexA) / m)
  ```
  where `r0 = rigidity / rigidity0`, `rk = 3*GeV / e / rigidity0`
- When `r0 >> rk`: `Kpara0 ~ D0 * V_p / c * r0^indexB` (high-rigidity regime)
- When `r0 << rk`: `Kpara0 ~ D0 * V_p / c * r0^indexA` (low-rigidity regime)
- `m` controls how sharply the transition occurs; larger m = sharper transition
- **Note**: There is also a separate `double m = 0;` in `step()` (particle.cc:130) used in `dphi += m * HCS::Omega * dt` (line 277) — this is always zero and unrelated to the `m` in `Kpara0()`

## Why CmdStan Cannot Be Used Directly (HMC/NUTS)

CmdStan's default sampler (NUTS/HMC) requires a **differentiable log-posterior**. The HelProp simulation has three properties that make this impossible:

1. **Stochastic**: `particle::step()` uses `mt19937` + `normal_distribution` for random walks. Each evaluation with the same parameters gives different results (unless `--seed` is fixed), so the log-likelihood is itself a random variable.

2. **Non-differentiable**: The SDE integration in `step()` contains if/else branches (`if (d_HCS < 2*Rg)`, `if (force_outward && dwr < 0)`, `if (r < 0.5*AU)`), `fmin/fmax`, `fabs`, and assertions. No gradient ∂Ospec/∂D0 or ∂Ospec/∂m can be computed through this.

3. **Computationally expensive**: Each parameter evaluation requires simulating `number × len(ETOA)` particles, each running thousands of SDE steps. A single HelProp run takes seconds to minutes; HMC typically needs 10³-10⁴ evaluations.

## Recommended Approach: Emulation-Based Bayesian Inference

The most practical path uses HelProp as a **pre-computation step**, not inside the sampler.

### Step 1: Make `m` Configurable

Before anything else, `m` must be promoted from a hardcoded constant to a parameter. In `particle.h`, add a member:
```cpp
double m_smooth;  // rigidity smoothness index in Kpara0
```

In `Kpara0()`, replace `const double m = 3;` with `m_smooth`. Initialize it in the constructor and add a `--indexM` CLI option in `HelProp.cc`.

### Step 2: Pre-compute Green Function Matrices

Run HelProp on a **design of experiments** (Latin Hypercube or grid) over (D0, m):

```bash
for D0 in 1 2 3 5 8 10; do
  for m in 1 2 3 5 8; do
    ./HelProp --D0 $D0 --indexM $m --number 10000 \
      --etoa 0.1,100,20 --elis 0.1,100,20 \
      --iotype BSON --append outmatrix_D${D0}_m${m}.bson
  done
done
```

Each run produces a Green function matrix `weight[i_toa][i_lis]`. Store all (D0, m, weight) tuples.

### Step 3: Build an Emulator

For each element `weight[i_toa][i_lis]`, fit an emulator (Gaussian Process or polynomial) as a function of (D0, m):

```
weight[i_toa][i_lis] ~ GP(D0, m)
```

The emulator provides:
- A fast, deterministic prediction `weight_hat(D0, m)` for any (D0, m)
- Analytical gradients ∂weight_hat/∂D0 and ∂weight_hat/∂m (for GP with RBF kernel)

This is the critical step that makes CmdStan viable.

### Step 4: Write the Stan Model

Given observed TOA data `(E_obs, F_obs)` and a known LIS, the forward model is (from `HelProp.cc:270-277`):

```stan
data {
  int<lower=1> N_toa;       // number of TOA energy bins
  int<lower=1> N_lis;       // number of LIS energy bins
  vector[N_toa] ETOA;       // TOA energies [GeV]
  vector[N_lis] ELIS;       // LIS energies [GeV]
  vector[N_lis] FLIS;       // LIS differential flux
  vector[N_toa] F_obs;      // observed TOA flux
  vector[N_toa] F_err;      // observed TOA uncertainties
}
parameters {
  real<lower=0.1, upper=20> D0;   // in units of 1e22 cm²/s
  real<lower=0.5, upper=10> m;    // smoothness index
}
model {
  // Priors
  D0 ~ lognormal(log(5), 0.5);
  m ~ lognormal(log(3), 0.3);

  // Emulator prediction for weight matrix
  // (implemented as functions in Stan, or via external C++)
  matrix[N_toa, N_lis] weight = predict_weight(D0, m);

  // Forward model: LIS -> TOA (matching HelProp.cc:270-277)
  vector[N_toa] pTOA;
  vector[N_lis] pLIS;
  for (i in 1:N_toa) pTOA[i] = sqrt(ETOA[i] * (ETOA[i] + 2*0.938272)) * 1; // A=1
  for (j in 1:N_lis) pLIS[j] = sqrt(ELIS[j] * (ELIS[j] + 2*0.938272)) * 1;

  vector[N_toa] F_model;
  for (i in 1:N_toa) {
    real value = 0;
    for (j in 1:N_lis)
      value += weight[i,j] * FLIS[j] / (pLIS[j]^2) * (pTOA[i]^2);
    F_model[i] = value;
  }

  // Likelihood
  F_obs ~ lognormal(log(F_model), F_err);
}
```

The `predict_weight(D0, m)` function is where the emulator lives. Options for implementing it:
- **GP emulator in Stan**: Define GP mean + covariance functions directly in the Stan model using the pre-computed training data. This is self-contained but verbose.
- **Polynomial emulator**: Fit a polynomial surface to each weight element offline, hard-code the coefficients as constants in Stan. Fastest to evaluate.
- **External C++**: Use CmdStan's `allow_undefined` + external C++ to call a pre-built interpolator. Most flexible.

### Step 5: Run Inference

```bash
cmdstan_model bayesian_helprop.stan
./bayesian_helprop sample data=helprop_data.json
```

## Alternative Approaches

### Approximate Bayesian Computation (ABC)
If building an emulator is too much effort, ABC bypasses the likelihood entirely:
1. Sample (D0, m) from the prior
2. Run HelProp to get predicted TOA spectrum
3. Compare with observed data using a distance metric (e.g., chi-squared)
4. Accept if distance < epsilon

This is simple but requires many HelProp runs (10⁴-10⁶) and is very slow unless you reduce particle count drastically. No CmdStan needed — any ABC library works (e.g., `pyABC`, `ABC-SMC`).

### Synthetic Likelihood
For each proposed (D0, m), run HelProp R times with different seeds, compute the mean and covariance of the output spectra, and use:
```
log L = -0.5 * (F_obs - mean(F_sim))' * Cov(F_sim)^{-1} * (F_obs - mean(F_sim))
```
This is a valid likelihood for CmdStan but requires R ≥ 30 simulations per MCMC step, making it extremely expensive.

## Summary

| Approach | CmdStan Compatible | Effort | Speed | Accuracy |
|----------|-------------------|--------|-------|----------|
| Direct HMC through simulator | No (non-differentiable, stochastic) | — | — | — |
| Emulation + CmdStan | Yes | High (emulator build) | Fast | Good (if emulator is accurate) |
| ABC | No (different tool) | Low | Very slow | Good (if epsilon is small) |
| Synthetic Likelihood + CmdStan | Yes | Medium | Very slow | Moderate |

**Recommendation**: The emulation-based approach is the most practical. The key prerequisite is promoting `m` from `const double m = 3` to a configurable parameter, then running a grid of simulations to train the emulator. The existing Green function matrix output (`--iotype BSON <outmatrix>`) already provides the data you need for training.
