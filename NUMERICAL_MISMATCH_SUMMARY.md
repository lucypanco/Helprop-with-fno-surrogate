# Numerical Parameter Mismatch Summary

Comparison scope: `src` implementation versus `Unified HelProp.pdf`.

No source-code changes are included in this document.

## 1. Termination-shock radius (`r_TS`)

The paper's Eq. (5) defines the termination-shock radius as:

```text
r_TS = 90 AU
```

The current implementation uses `120 AU` in both solar-wind branches:

```cpp
(r - 120 * AU) / 1.2 / AU
```

Location: [`src/particle.cc:40-42`](src/particle.cc:40)

This changes the radial transition of the solar-wind profile and is a definite mismatch with the paper.

## 2. Polar perpendicular-diffusion factor

The paper's Eq. (8) defines:

```text
f_perp_theta = A+ -/+ A- tanh[-8(|90 deg - theta| -/+ theta_F)]
A+ = (d + 1) / 2
A- = (d - 1) / 2
d = 3
theta_F = 35 deg
```

Therefore, the paper gives approximately:

- `f_perp_theta = 1` near the equator;
- `f_perp_theta = 3` toward the poles.

The current implementation is:

```cpp
return 1.0 - 0.5 * tanh(8 * (theta - pi / 2 + 35 * deg));
```

Location: [`src/particle.cc:77-79`](src/particle.cc:77)

Numerically, this gives approximately:

- `0.5` near the equator;
- `1.5` toward the poles.

Thus, the current polar perpendicular diffusion is lower than the paper's expression by approximately a factor of two. The radial perpendicular factor `K_perp_r = 0.02 K_parallel` is separate and matches the paper.

## 3. No-argument constructor: `indexA`

The HelProp command-line default is:

```text
indexA = 1
```

The argument-based constructor correctly reads `--indexA`:

```cpp
indexA(stod(args.at("--indexA").asString()))
```

However, the no-argument constructor hard-codes:

```cpp
indexA(2)
```

Location: [`src/particle.cc:21`](src/particle.cc:21)

As a result, `particle()` does not use the same default as `HelProp` when no explicit command-line value is supplied. This is a definite constructor-default mismatch.

## Summary table

| Parameter | Current `src` | Paper / HelProp intended value | Status |
|---|---:|---:|---|
| `r_TS` | `120 AU` | `90 AU` | Mismatch |
| `f_perp_theta` at equator | `~0.5` | `~1` | Mismatch |
| `f_perp_theta` at poles | `~1.5` | `~3` | Mismatch |
| `particle()::indexA` | `2` | `1` | Mismatch |

