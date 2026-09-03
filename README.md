# HelProp

HelProp is a C++ Monte Carlo solver for charged-particle transport in the
heliosphere. It solves a three-dimensional backward stochastic differential
equation with Parker magnetic fields, solar-wind convection, adiabatic energy
changes, diffusion, gradient-curvature drift, and a wavy heliospheric current
sheet (HCS).

The repository also contains tools for generating HelProp transfer matrices
and training a Fourier Neural Operator (FNO) surrogate.

## Repository layout

- `src/` - HelProp C++ source, model equations, tests, and utilities.
- `CMakeLists.txt` - CMake build configuration.
- `helprop_surrogate/` - matrix generation, FNO training, prediction, and reports.
- `helprop_surrogate/MANUAL.md` - detailed surrogate workflow.
- `helprop_mcmc/` - optional surrogate-based MCMC tools.
- `extern/` and `docopt/` - bundled dependencies and command-line parsing.

## Requirements

- CMake 3.13 or newer.
- A C++20-capable compiler.
- Bash and `make`; configuration runs `extern/install_extern.sh`.
- NLOPT available to CMake.
- Python 3 with `numpy` for surrogate data tools.
- `torch` and `pymongo` for FNO training and BSON workflows.

On Windows, use WSL, MSYS2, or Git Bash with a compatible C++ toolchain.

## Build

Run from the repository root:

```bash
cmake -B build
cmake --build build
```

The main executables are written to the project root:

```text
HelProp
IOTest
ParticleTest
KinematicsTest
gen_HCS_distance_map
get_HCS_distance_test
```

The shared library is written to `lib/`.

## Run HelProp

HelProp accepts a local interstellar spectrum and writes a modulated spectrum:

```bash
./HelProp [options] <inspec> <outspec>
```

The text input format is two columns in GeV:

```text
# E F
0.001 1.0e4
0.010 2.0e3
0.100 1.0e2
```

Example proton run:

```bash
./HelProp \
  --seed=123 \
  --number=1000 \
  --A=1 --Z=1 --polarity=-1 \
  --B0=5 --angle=15 \
  --D0=5 --R0=1 --indexA=1 --indexB=1 --m=0 \
  input.txt output.txt
```

The output contains the TOA energy and modulated flux in the same two-column
text format.

### Matrix-only mode

To generate a transfer matrix without folding a spectrum:

```bash
./HelProp \
  --A=1 --Z=1 --polarity=-1 \
  --B0=5 --angle=15 --D0=5 --R0=1 --indexA=1 --indexB=1 \
  --etoa=0.1,120,80 \
  --elis=0.1,150,80 \
  --number=800 \
  matrix.txt
```

Matrix rows correspond to ETOA bins and columns correspond to ELIS bins. Each
row is normalized as a transition probability.

### Electrons and positrons

Use `A=0` for leptons:

```bash
# electron
./HelProp --A=0 --Z=-1 --polarity=-1 input.txt electron_output.txt

# positron
./HelProp --A=0 --Z=1 --polarity=-1 input.txt positron_output.txt
```

For `A=0`, energies are kinetic energy per particle. Diffusion uses the
absolute rigidity, while gradient-curvature and HCS drifts change sign with
the charge `Z`.

## Main parameters

| Option | Default | Meaning |
|---|---:|---|
| `--A` | `1` | Nucleon number; use `0` for an electron or positron |
| `--Z` | `1` | Charge number; use `-1` for an electron |
| `--B0` | `5` | Magnetic-field strength near Earth, nT |
| `--polarity` | `-1` | HMF polarity |
| `--angle` | `15` | HCS tilt angle, degrees |
| `--D0` | `5` | Reference diffusion coefficient, `10^22 cm^2/s` |
| `--R0` | `1` | Reference rigidity, GV |
| `--indexA` | `1` | Low-rigidity diffusion index |
| `--indexB` | `1` | High-rigidity diffusion index |
| `--m` | `0` | Co-rotation factor |
| `--number` | `1000` | Particles per energy bin |
| `--nthread` | `1` | Threads used inside a run |
| `--seed` | random | Reproducible random seed |
| `--hcs-table` | empty | Optional precomputed HCS distance-map directory |

HCS oscillation options are also available: `--hcs-osc-amp`,
`--hcs-osc-phase`, and `--hcs-omega`. Use `./HelProp --help` for the complete
option list.

## HCS distance maps

Generate interpolation tables for HCS distances with:

```bash
./gen_HCS_distance_map dmap
```

Use the generated directory in a simulation:

```bash
./HelProp --hcs-table=dmap input.txt output.txt
```

## Tests

After building:

```bash
ctest --test-dir build
./KinematicsTest
./IOTest
./ParticleTest --help
```

`ParticleTest` writes a particle trajectory log when given an output path:

```bash
./ParticleTest --seed=123 --ekin=0.5 particle_log.csv
```

## Surrogate workflow

Install the Python dependencies:

```bash
python -m pip install numpy torch pymongo
```

Generate transfer-matrix data:

```bash
python -m helprop_surrogate.matrix_data \
  --helprop ./HelProp \
  --run-dir surrogate_runs/run_proton_A1 \
  --n-runs 30000 \
  --learn D0 indexA indexB angle \
  --range D0:0.1:15 \
  --range indexA:0:3 \
  --range indexB:0:3 \
  --range angle:5:45 \
  --fixed A=1 --fixed Z=-1 --fixed polarity=1 \
  --fixed R0=1 --fixed B0=5 \
  --etoa 0.1,120,80 \
  --elis 0.1,150,80 \
  --number 800 --nthread 5 --jobs 3
```

Resume an interrupted data-generation run:

```bash
python -m helprop_surrogate.matrix_data \
  --continue surrogate_runs/run_proton_A1
```

Train an FNO surrogate:

```bash
python -m helprop_surrogate.fno.train \
  --dataset surrogate_runs/run_proton_A1/data/matrices.npz \
  --outdir surrogate_runs/run_proton_A1 \
  --lis Proton_spectrum.txt \
  --epochs 700 --batch-size 36 \
  --width 96 --layers 6 \
  --modes-etoa 32 --modes-elis 40 \
  --projection-size 192 \
  --device cuda --seed 123 \
  --fixed A=1 --fixed Z=-1 --fixed polarity=-1 \
  --fixed R0=1 --fixed B0=5 \
  --no-checkpoints
```

The detailed training, validation, prediction, folding-report, resume, and
dataset-merge instructions are in [`helprop_surrogate/MANUAL.md`](helprop_surrogate/MANUAL.md).

## Units and conventions

- Distances are internally stored in SI units; input/output energies are GeV.
- Magnetic field strength is specified in nT.
- Rigidity is in GV when supplied through command-line options.
- Positive `A` denotes a nucleus; `A=0` denotes an electron or positron.
- `Z` controls the charge-dependent drift direction.

