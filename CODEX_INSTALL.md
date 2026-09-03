# Codex Install Handoff for HelProp

Use this file after moving the HelProp project folder to another device. Launch Codex from the project root, then paste the prompt below.

## Prompt to Paste into Codex

```text
You are in the HelProp project root. Please install and build this project on this machine.

Follow CODEX_INSTALL.md and AGENTS.md. First inspect the local OS, compiler, CMake, Bash/make availability, and whether NLOPT is installed. Then configure and build the project. Ask before installing system packages or downloading dependencies. Keep any edits minimal and focused on setup/build fixes.

Expected build commands:

cmake -B build
cmake --build build

After a successful build, verify with:

./HelProp --help
./IOTest
./ParticleTest -s 123 out.bin
./get_HCS_distance_test

If a command needs Windows PowerShell syntax, adapt it for the current shell. If the build fails, diagnose the first real error, fix only what is necessary, and rerun the relevant command.
```

## Project Requirements

- CMake 3.13 or newer.
- A C++20-capable compiler.
- Bash and `make`, because `CMakeLists.txt` runs `extern/install_extern.sh` during configure.
- NLOPT installed on the system so `CMakeModules/FindNLOPT.cmake` can find it.
- The vendored directories `docopt/`, `extern/mongo-c-driver/`, and `extern/reflect-cpp/` must be present.

## Normal Build

From the project root:

```powershell
cmake -B build
cmake --build build
```

During `cmake -B build`, the project runs:

```bash
extern/install_extern.sh
```

That script builds and installs the vendored `mongo-c-driver` BSON library and `reflect-cpp` into `extern/`.

## Expected Outputs

Executables are written to the project root:

- `HelProp`
- `IOTest`
- `ParticleTest`
- `gen_HCS_distance_map`
- `get_HCS_distance_test`

The shared library output goes to:

- `lib/`

## Transfer Notes

- Transfer the source tree, not just the existing `build/` folder.
- It is safe to omit generated build outputs such as `build/`, `lib/`, `extern/build/`, `extern/lib/`, and `extern/lib64/`; they can be regenerated.
- Do not omit `extern/mongo-c-driver/`, `extern/reflect-cpp/`, `docopt/`, `src/`, `CMakeModules/`, `CMakeLists.txt`, `AGENTS.md`, or this file.
- On Windows, use a shell/toolchain that can run Bash scripts and `make` for the external dependency build, such as WSL, MSYS2/MinGW, or Git Bash with a compatible compiler toolchain.

## Useful Runtime Commands

```powershell
./HelProp --help
./IOTest
./ParticleTest -s 123 out.bin
./get_HCS_distance_test
python rigidity_to_ekin.py
```
