# HelProp - Cosmic Ray Propagation

## Build
```powershell
cmake -B build
cmake --build build
```

Dependencies are built automatically by `extern/install_extern.sh` during cmake configure.

## Run Executables
```powershell
./HelProp --help              # Main simulation (see src/HelProp.cc for options)
./IOTest                    # I/O test binary
./ParticleTest -s 123 out.bin  # Single particle test
./get_HCS_distance_test      # Distance map test
python FIgure.py             # Plot output
```

## Key Files
- `src/HelProp.cc`: Main simulation entrypoint
- `src/particle.cc`, `src/HCS.cc`: Core physics
- `docopt/`: Vendored command-line parser
- `src/IO.cc`: Custom binary format (spec/matrix files)

## Notes
- C++20, uses NLOPT, libbson, reflect-cpp
- Windows paths, PowerShell used for build
- Tests run as standalone binaries (no test framework)