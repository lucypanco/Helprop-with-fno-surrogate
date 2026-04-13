# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HelProp is a C++20 Monte Carlo simulation of cosmic ray propagation through the heliosphere. It computes the modulated spectrum at Earth (TOA) from an input local interstellar spectrum (LIS) by tracking particles through the heliospheric magnetic field and solar wind using stochastic differential equations. The output is a Green function matrix or a modulated spectrum.

## Build

```bash
cmake -B build
cmake --build build
```

Dependencies (libbson, reflect-cpp) are built automatically by `extern/install_extern.sh` during cmake configure. The project also requires NLOPT installed on the system.

Profiling build: `cmake -B build -DPROFILE_DEBUG=ON` (compiles with `-O0 -pg`).

Executables are output to the project root directory. The shared library `libhelprop.so` is output to `lib/`.

## Executables

- `./HelProp` — Main simulation (see USAGE string in `src/HelProp.cc` for all CLI options)
- `./ParticleTest` — Single particle trajectory test with logging
- `./IOTest` — I/O round-trip test for BSON spec/matrix files
- `./gen_HCS_distance_map` — Pre-compute HCS distance lookup tables
- `./get_HCS_distance_test` — HCS distance map test

No test framework is used; tests are standalone executables.

## Architecture

### Core Physics

- **`particle`** (`src/particle.h`/`.cc`) — Central class representing a charged cosmic ray particle. Handles the stochastic propagation step (SDE integration) in spherical coordinates (r, theta, phi). Computes diffusion tensor K, gradient/drift velocities, solar wind advection, energy losses, and HCS drift. The `step()` method runs the full backward-in-time propagation from Earth to the heliospheric boundary.
- **`HCS`** (`src/HCS.h`/`.cc`) — Heliospheric Current Sheet model. Supports two analytical forms: `Jokipii_Thomas` and `Kota_Jokipii` (selected via `HCS::hcsform`). Computes the signed distance from any point to the HCS using iterative methods (`spiral_iterate`, `wave_iterate`, `point_iterate`) or pre-computed interpolation tables (`KDInterp`). The distance calculation is the most computationally expensive part and is optionally accelerated by `--hcs-table`.
- **`Unit`** (`src/Unit.h`) — Namespace with physical constants and unit conversions. All internal calculations use natural units defined here (e.g., `GeV`, `AU`, `nT`). Note: `sec` is defined as `2.99792e8` (not 1), so time units are scaled by c.

### Supporting Math

- **`Vec`** (`src/Vec.hh`) — Lightweight 3D vector struct with dot/cross products, spherical coordinate conversion, and rotations.
- **`LogInterp`** (`src/loginterp.h`/`.cc`) — Log-log interpolation of 1D data (used for interpolating input LIS spectra).
- **`root_finding`** (`src/root_finding.h`/`.cc`) — Ridders' and Brent's root-finding methods (used in HCS distance iteration).
- **`newton_solver`** (`src/newton_solver.h`/`.cc`) — Newton's method solver.
- **`fcache`** (`src/fcache.h`/`.cc`) — 1D function caching with uniform binning.
- **`KDInterp`** (`src/KDInterp.h`/`.cc`) — Adaptive KD-tree based N-dimensional interpolation with error-driven refinement. Used for HCS distance pre-computation.
- **`NDIndex`** (`src/NDIndex.h`/`.cc`) — N-dimensional index utility for KDInterp.
- **`hcs_interp`** (`src/hcs_interp.h`/`.cc`) — HCS-specific interpolation building and evaluation using KDInterp.

### I/O System

- **`IO`** (`src/IO.h`/`.cc`) — Abstract base class for spectrum and matrix I/O with three implementations:
  - `IO_TXT` — Space-separated text files
  - `IO_CSV` — Comma-separated text files
  - `IO_BSON` — Binary BSON format (uses reflect-cpp and libbson)
- All formats handle unit conversion via `eunit` (set to `GeV`). Spectra store E and F (flux); matrices store ETOA, ELIS energy axes and a 2D Green function weight matrix.
- **`docopt`** (`docopt/`) — Vendored command-line argument parser.

### Main Simulation Flow (`HelProp.cc`)

1. Parse CLI args with docopt
2. Select I/O backend (TXT/CSV/BSON)
3. Read input LIS spectrum
4. For each TOA energy bin, simulate N particles backward via `simulating()` (multi-threaded)
5. Bin results into Green function matrix via `count_GreenFunction()`
6. Output: either the raw Green function matrix (`<outmatrix>`) or convolve with LIS to produce modulated spectrum (`<outspec>`)

### Libraries

Two shared libraries are built:
- `helprop` — Full library (all source files, linked with NLOPT)
- `helprop4externel` — Minimal library (particle, HCS, solvers only, no I/O) for external coupling

## Key Patterns

- `HCS::angle` and `HCS::hcsform` are static class members set globally before simulation
- Particle positions are in spherical coordinates; the `particle::step()` method handles boundary conditions (inner reflection at 0.5 AU, outer boundary at 100 AU)
- Threading is done manually with `std::thread` and a mutex-protected particle index counter
- BSON I/O entries are 1-indexed (ientry starts at 1)
- The `particle` constructor reads all physics parameters from the docopt args map directly
