# HelProp Surrogate Model Recommendation

## Goal

HelProp is too expensive to call directly at every MCMC likelihood evaluation. The surrogate should replace the slow forward solar-modulation calculation while preserving the Bayesian inference structure.

The recommended target is not the posterior distribution itself. The neural model should emulate the HelProp modulation kernel or transfer matrix, then the MCMC sampler should still compute the posterior from the likelihood and priors.

## HelProp Forward Model

From `src/HelProp.cc`, HelProp simulates particles backward from each TOA kinetic-energy bin and bins their final boundary energies into a Green-function / transfer matrix row:

```text
M_ij(theta) = probability that a particle started at TOA energy i exits with LIS energy j
```

Then HelProp folds this matrix with the LIS spectrum:

```text
J_TOA[i] = p_TOA[i]^2 * sum_j M[i,j] * J_LIS[j] / p_LIS[j]^2
```

So the reusable physics object is the transfer kernel/matrix, not the final modulated spectrum.

## Recommended 2026 Approach

Use a conditional probabilistic kernel surrogate:

```text
p(log E_LIS | log E_TOA, theta)
```

where `theta` contains the modulation parameters varied in the fit, for example:

```text
theta = [log10(D0), m]
```

or, for a larger modulation fit:

```text
theta = [log10(D0), m, angle, B0, indexA, indexB, R0, polarity, Z/A]
```

The best practical model is:

```text
Conditional Neural Spline Flow
+ ensemble uncertainty
+ active high-fidelity retraining
+ delayed-acceptance MCMC
```

This is newer and more suitable than a plain neural network that directly regresses a fixed transfer matrix.

## Why A Conditional Flow

HelProp is Monte Carlo. A binned transfer matrix is a noisy estimate of an underlying continuous transition density. A conditional flow learns that density directly:

```text
(theta, log E_TOA) -> distribution over log E_LIS
```

Advantages:

- The transfer matrix is automatically non-negative.
- Each row can be normalized by construction.
- The model can work on different energy grids.
- Raw particle samples are more informative than only binned matrices.
- Emulator uncertainty can be estimated with an ensemble.
- It remains compatible with standard MCMC likelihoods.

## Building The Matrix From The Flow

After training, construct the matrix by integrating the learned density over each LIS energy bin:

```text
M_ij(theta) = integral over LIS bin j of p(log E_LIS | log E_TOA_i, theta) d log E_LIS
```

Then use the same folding equation as HelProp:

```text
J_TOA[i] = p_TOA[i]^2 * sum_j M_ij(theta) * J_LIS[j] / p_LIS[j]^2
```

This keeps the physical interpretation of HelProp's transfer matrix while avoiding a full simulation at every MCMC step.

## Training Data

Use HelProp to generate particle-level transition samples:

```text
(theta, E_TOA) -> E_LIS sample
```

The most useful HelProp output mode is BSON with `--sample`, because `src/IO.cc` stores:

```text
seed
etoa
elis
params
M
```

A training row should include:

```text
input:  [log E_TOA, log10(D0), m, fixed/varying physics parameters...]
target: log E_LIS
```

For numerical stability, use:

```text
target = log(E_LIS / E_TOA)
```

or:

```text
target = log E_LIS
```

## Multi-Fidelity Strategy

Use cheap HelProp runs first, then expensive ones only where needed.

Low-fidelity stage:

```text
--number 50-200
coarse ETOA/ELIS grid
wide parameter sampling
```

High-fidelity stage:

```text
--number 1000-10000
final energy grid
posterior-relevant parameter region
```

Sampling strategy:

1. Start with Sobol or Latin-hypercube points over the prior range.
2. Train an ensemble of conditional flows.
3. Run a preliminary MCMC with the surrogate.
4. Identify high-posterior or high-disagreement regions.
5. Run true HelProp there.
6. Retrain or fine-tune the ensemble.

## MCMC Integration

Use the surrogate inside the likelihood:

```text
theta
-> conditional flow kernel
-> transfer matrix M(theta)
-> folded TOA spectrum
-> likelihood(data | theta)
-> MCMC posterior
```

For correctness, use delayed-acceptance MCMC:

1. First-stage accept/reject with the fast surrogate likelihood.
2. For accepted candidates, periodically evaluate true HelProp.
3. Correct or validate the chain with true-model likelihoods.
4. Add failed/high-error points to the training set.

This prevents the posterior from being controlled blindly by emulator bias.

## Loss Function

For particle-level flow training, maximize conditional likelihood:

```text
-log p_phi(log E_LIS | log E_TOA, theta)
```

For matrix-level validation, check:

```text
row-wise KL divergence between M_true and M_pred
```

For inference-level validation, check:

```text
error in log-likelihood
error in folded TOA spectrum
posterior shift after true HelProp correction
```

The final acceptance criterion should be based on likelihood error, not only matrix MSE.

## Model Choice

Best first implementation:

```text
Conditional Neural Spline Flow
```

Condition network:

```text
MLP([log E_TOA, log10(D0), m, ...]) -> spline parameters
```

Output distribution:

```text
log(E_LIS / E_TOA)
```

Use 3 to 5 independently trained models as an ensemble.

If the parameter dimension becomes large or time-dependent, upgrade to:

```text
conditional flow matching
```

If you insist on fixed-grid matrix prediction, use:

```text
residual row-normalized matrix decoder
```

but this is less flexible than the conditional kernel flow.

## What Not To Do

Do not train the network to directly output the Bayesian posterior unless you want to replace MCMC entirely with simulation-based inference. That is possible, but harder to validate for this project.

Do not train only on final modulated spectra unless the LIS is fixed forever. Spectrum-level surrogates are less reusable than transfer-kernel surrogates.

Do not use a plain unconstrained MLP matrix regressor without row normalization. It can produce negative probabilities or rows that do not sum to one.

## Practical Recommendation

For the current HelProp MCMC problem:

```text
Use a conditional neural spline flow to emulate
p(log(E_LIS / E_TOA) | log E_TOA, log10(D0), m)
```

Then:

```text
flow -> transfer matrix -> LIS folding -> likelihood -> MCMC
```

Add:

```text
ensemble uncertainty
active learning near posterior
delayed-acceptance true HelProp checks
```

This is the most appropriate modern approach because it uses HelProp's Monte Carlo particle samples directly, preserves the transfer-matrix physics, and keeps the Bayesian posterior calculation explicit.

