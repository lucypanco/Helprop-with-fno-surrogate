"""Direct HelProp hcs-osc-amp scan over seeds and ETOA bin counts."""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helprop_surrogate.file_safety import (
    atomic_promote,
    atomic_replace_text,
    ensure_unique_outputs,
    temp_output_path,
)


DEFAULT_HCS_OSC_MIN = -50.0
DEFAULT_HCS_OSC_MAX = 50.0
DEFAULT_HCS_OSC_STEP = 2.0
DEFAULT_SEED_VALUES = (12345, 67890)
DEFAULT_BIN_VALUES = (150, 300)
DEFAULT_SELECTED_HCS_OSC_VALUES = (6.0, 16.0, 36.0)
DEFAULT_NUMBER = 500
DEFAULT_ETOA_MIN = 1.0e-1
DEFAULT_ETOA_MAX = 1.0e5
FIXED_M = 0.0
FIXED_HCS_OSC_PHASE = 0.0


@dataclass(frozen=True)
class SpectrumScanRun:
    """One HelProp spectrum run in the hcs-osc-amp scan."""

    index: int
    seed: int
    etoa_bins: int
    hcs_osc_amp: float
    output: Path
    command: list[str]


@dataclass(frozen=True)
class FluxAtEnergy:
    """Interpolated flux for one run at a single energy."""

    seed: int
    etoa_bins: int
    hcs_osc_amp: float
    energy: float
    flux: float
    percent: float
    output: Path


def default_scan_values() -> tuple[tuple[float, ...], tuple[int, ...], tuple[int, ...]]:
    return (
        build_hcs_osc_values(DEFAULT_HCS_OSC_MIN, DEFAULT_HCS_OSC_MAX, DEFAULT_HCS_OSC_STEP),
        DEFAULT_SEED_VALUES,
        DEFAULT_BIN_VALUES,
    )


def parse_float_list(spec: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in spec.split(",") if item.strip())
    if not values:
        raise ValueError("value list must not be empty")
    if any(not np.isfinite(value) for value in values):
        raise ValueError("value list contains a non-finite value")
    return values


def parse_int_list(spec: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in spec.split(",") if item.strip())
    if not values:
        raise ValueError("integer list must not be empty")
    if any(value <= 0 for value in values):
        raise ValueError("integer list values must be positive")
    return values


def build_hcs_osc_values(hcs_osc_min: float, hcs_osc_max: float, hcs_osc_step: float) -> tuple[float, ...]:
    """Build an inclusive hcs-osc-amp grid and force the standard value 0 into it."""
    hcs_osc_min = float(hcs_osc_min)
    hcs_osc_max = float(hcs_osc_max)
    hcs_osc_step = float(hcs_osc_step)
    if not all(np.isfinite(value) for value in (hcs_osc_min, hcs_osc_max, hcs_osc_step)):
        raise ValueError("hcs-osc grid bounds and step must be finite")
    if hcs_osc_step <= 0.0:
        raise ValueError("hcs-osc step must be positive")
    if hcs_osc_max < hcs_osc_min:
        raise ValueError("hcs-osc max must be greater than or equal to hcs-osc min")
    count = int(np.floor((hcs_osc_max - hcs_osc_min) / hcs_osc_step + 0.5)) + 1
    values = [hcs_osc_min + index * hcs_osc_step for index in range(count)]
    if values[-1] < hcs_osc_max - 1.0e-9:
        values.append(hcs_osc_max)
    if not any(np.isclose(value, 0.0) for value in values):
        values.append(0.0)
    return tuple(sorted({round(float(value), 12) for value in values}))


def value_token(value: float | int) -> str:
    """Convert a numeric value to a stable file-name token."""
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.12g}".replace("-", "m").replace(".", "p")


def build_fixed_options(args: argparse.Namespace) -> list[str]:
    """Build common HelProp options that do not vary over the scan."""
    options = [
        f"--A={args.A}",
        f"--Z={args.Z}",
        f"--B0={args.B0:.17g}",
        f"--polarity={args.polarity}",
        f"--angle={args.angle:.17g}",
        f"--hcs-omega={args.hcs_omega:.17g}",
        f"--D0={args.D0:.17g}",
        f"--R0={args.R0:.17g}",
        f"--indexA={args.indexA:.17g}",
        f"--indexB={args.indexB:.17g}",
        f"--number={args.number}",
        f"--nthread={args.nthread}",
        "--iotype=TXT",
    ]
    if args.elis:
        options.append(f"--elis={args.elis}")
    if args.hcs_table:
        options.append(f"--hcs-table={args.hcs_table}")
    options.extend(args.extra_opts)
    return options


def build_helprop_spectrum_command(
    helprop: str | Path,
    lis_input: str | Path,
    output: str | Path,
    fixed_options: Sequence[str],
    *,
    seed: int,
    etoa_bins: int,
    etoa_min: float,
    etoa_max: float,
    hcs_osc_amp: float,
) -> list[str]:
    """Build one direct HelProp spectrum command."""
    return [
        str(helprop),
        *[str(option) for option in fixed_options],
        f"--seed={int(seed)}",
        f"--etoa={float(etoa_min):.12g},{float(etoa_max):.12g},{int(etoa_bins)}",
        f"--m={FIXED_M:.17g}",
        f"--hcs-osc-amp={float(hcs_osc_amp):.17g}",
        f"--hcs-osc-phase={FIXED_HCS_OSC_PHASE:.17g}",
        str(lis_input),
        str(output),
    ]


def make_scan_runs(
    helprop: str | Path,
    lis_input: str | Path,
    outdir: str | Path,
    fixed_options: Sequence[str],
    hcs_osc_values: Sequence[float],
    seed_values: Sequence[int],
    bin_values: Sequence[int],
    etoa_min: float = DEFAULT_ETOA_MIN,
    etoa_max: float = DEFAULT_ETOA_MAX,
) -> list[SpectrumScanRun]:
    """Create run descriptors in seed, ETOA bin count, then hcs-osc-amp order."""
    outdir = Path(outdir)
    spectra_dir = outdir / "spectra"
    runs = []
    index = 0
    for seed in seed_values:
        for etoa_bins in bin_values:
            run_dir = spectra_dir / f"seed_{int(seed)}" / f"bins_{int(etoa_bins)}"
            for hcs_osc_amp in hcs_osc_values:
                output = run_dir / f"hcs_osc_{value_token(hcs_osc_amp)}.txt"
                command = build_helprop_spectrum_command(
                    helprop,
                    lis_input,
                    output,
                    fixed_options,
                    seed=int(seed),
                    etoa_bins=int(etoa_bins),
                    etoa_min=etoa_min,
                    etoa_max=etoa_max,
                    hcs_osc_amp=hcs_osc_amp,
                )
                runs.append(
                    SpectrumScanRun(
                        index=index,
                        seed=int(seed),
                        etoa_bins=int(etoa_bins),
                        hcs_osc_amp=float(hcs_osc_amp),
                        output=output,
                        command=command,
                    )
                )
                index += 1
    return runs


def write_manifest(path: str | Path, runs: Sequence[SpectrumScanRun]) -> None:
    """Write the ordered scan manifest."""
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["index", "seed", "etoa_bins", "hcs-osc-amp", "output", "command"])
    for run in runs:
        writer.writerow(
            [
                run.index,
                run.seed,
                run.etoa_bins,
                run.hcs_osc_amp,
                run.output,
                " ".join(run.command),
            ]
        )
    atomic_replace_text(path, stream.getvalue())


def run_helprop_spectra(
    runs: Sequence[SpectrumScanRun],
    *,
    timeout: float | None = None,
    dry_run: bool = False,
    jobs: int = 1,
) -> None:
    """Execute all scan commands, writing each output atomically."""
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    if dry_run:
        for run in runs:
            print(" ".join(run.command))
        return
    if jobs == 1:
        for ordinal, run in enumerate(runs, start=1):
            print(f"[{ordinal}/{len(runs)}] {run.output}")
            _run_one_helprop_spectrum(run, timeout=timeout)
        return

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(_run_one_helprop_spectrum, run, timeout): run
            for run in runs
        }
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            print(f"[{completed}/{len(runs)}] {futures[future].output}")


def _run_one_helprop_spectrum(run: SpectrumScanRun, timeout: float | None = None) -> None:
    temporary_output = temp_output_path(run.output)
    command = list(run.command)
    command[-1] = str(temporary_output)
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"HelProp failed for run {run.index} with exit code {result.returncode}\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        atomic_promote(temporary_output, run.output)
    finally:
        if temporary_output.exists():
            temporary_output.unlink()


def load_spectrum(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a HelProp TXT spectrum file."""
    data = np.loadtxt(path)
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError(f"spectrum file must contain at least two columns: {path}")
    energy = np.asarray(data[:, 0], dtype=float)
    flux = np.asarray(data[:, 1], dtype=float)
    if energy.size < 2:
        raise ValueError(f"spectrum file must contain at least two rows: {path}")
    if np.any(~np.isfinite(energy)) or np.any(~np.isfinite(flux)):
        raise ValueError(f"spectrum contains non-finite values: {path}")
    if np.any(energy <= 0.0) or np.any(flux <= 0.0):
        raise ValueError(f"spectrum energy and flux must be positive: {path}")
    if np.any(np.diff(energy) <= 0.0):
        raise ValueError(f"spectrum energy grid must be strictly increasing: {path}")
    return energy, flux


def interpolate_loglog(
    source_energy: Sequence[float],
    source_flux: Sequence[float],
    target_energy: Sequence[float],
) -> np.ndarray:
    """Interpolate positive flux values on log-log axes."""
    source_energy = np.asarray(source_energy, dtype=float)
    source_flux = np.asarray(source_flux, dtype=float)
    target_energy = np.asarray(target_energy, dtype=float)
    if np.any(source_energy <= 0.0) or np.any(source_flux <= 0.0) or np.any(target_energy <= 0.0):
        raise ValueError("log-log interpolation requires positive inputs")
    if np.any(np.diff(source_energy) <= 0.0):
        raise ValueError("source energy grid must be strictly increasing")
    eps = 1.0e-12
    if target_energy[0] < source_energy[0] * (1.0 - eps) or target_energy[-1] > source_energy[-1] * (1.0 + eps):
        raise ValueError(
            "target energy grid is outside source range "
            f"[{source_energy[0]}, {source_energy[-1]}]"
        )
    clipped = np.clip(target_energy, source_energy[0], source_energy[-1])
    return np.exp(np.interp(np.log(clipped), np.log(source_energy), np.log(source_flux)))


def collect_flux_at_energy(
    runs: Sequence[SpectrumScanRun],
    energy: float,
) -> list[FluxAtEnergy]:
    """Read all spectra and interpolate flux and percent change at one energy."""
    energy = float(energy)
    if not np.isfinite(energy) or energy <= 0.0:
        raise ValueError("energy must be positive and finite")
    baseline_flux = {}
    for run in runs:
        if np.isclose(run.hcs_osc_amp, 0.0):
            source_energy, source_flux = load_spectrum(run.output)
            baseline_flux[(run.seed, run.etoa_bins)] = float(
                interpolate_loglog(source_energy, source_flux, [energy])[0]
            )

    records = []
    for run in runs:
        source_energy, source_flux = load_spectrum(run.output)
        flux = float(interpolate_loglog(source_energy, source_flux, [energy])[0])
        base_flux = baseline_flux[(run.seed, run.etoa_bins)]
        records.append(
            FluxAtEnergy(
                seed=run.seed,
                etoa_bins=run.etoa_bins,
                hcs_osc_amp=run.hcs_osc_amp,
                energy=energy,
                flux=flux,
                percent=100.0 * (flux / base_flux - 1.0),
                output=run.output,
            )
        )
    return records


def save_flux_records(path: str | Path, records: Sequence[FluxAtEnergy]) -> None:
    """Save the interpolated single-energy flux table."""
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["seed", "etoa_bins", "hcs-osc-amp", "energy", "flux", "percent_vs_hcs_osc0", "output"])
    for record in records:
        writer.writerow(
            [
                record.seed,
                record.etoa_bins,
                record.hcs_osc_amp,
                record.energy,
                record.flux,
                record.percent,
                record.output,
            ]
        )
    atomic_replace_text(path, stream.getvalue())


def plot_selected_flux(
    runs: Sequence[SpectrumScanRun],
    outdir: str | Path,
    *,
    seed: int,
    etoa_bins: int,
    selected_hcs_osc_values: Sequence[float],
) -> Path:
    """Plot selected full-spectrum percent differences for the first seed/bin count."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plotting requires matplotlib") from exc

    by_key = {
        (run.seed, run.etoa_bins, round(run.hcs_osc_amp, 12)): run
        for run in runs
    }
    baseline_key = (int(seed), int(etoa_bins), 0.0)
    if baseline_key not in by_key:
        raise KeyError(f"missing hcs-osc-amp=0 run for seed={seed}, bins={etoa_bins}")
    baseline_energy, baseline_flux = load_spectrum(by_key[baseline_key].output)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for hcs_osc_amp in selected_hcs_osc_values:
        key = (int(seed), int(etoa_bins), round(float(hcs_osc_amp), 12))
        if key not in by_key:
            raise KeyError(f"missing run for seed={seed}, bins={etoa_bins}, hcs-osc-amp={hcs_osc_amp:g}")
        energy, flux = load_spectrum(by_key[key].output)
        flux_on_baseline_grid = interpolate_loglog(energy, flux, baseline_energy)
        percent = 100.0 * (flux_on_baseline_grid / baseline_flux - 1.0)
        ax.plot(baseline_energy, percent, lw=1.4, label=f"hcs-osc={hcs_osc_amp:g}")
    ax.set_xscale("log")
    ax.set_xlabel("E_TOA (GeV)")
    ax.set_ylabel("Flux change vs hcs-osc=0 (%)")
    ax.set_title(f"seed={seed}, ETOA bins={etoa_bins}")
    ax.axhline(0.0, color="0.35", lw=0.7)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = Path(outdir) / f"selected_flux_percent_seed_{seed}_bins_{etoa_bins}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_flux_matrix(
    records: Sequence[FluxAtEnergy],
    outdir: str | Path,
    *,
    seed_values: Sequence[int],
    bin_values: Sequence[int],
    hcs_osc_values: Sequence[float],
    energy: float,
) -> Path:
    """Plot a seed x ETOA-bin matrix of flux percent at one energy versus hcs-osc-amp."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plotting requires matplotlib") from exc

    by_key = {
        (record.seed, record.etoa_bins, round(record.hcs_osc_amp, 12)): record.percent
        for record in records
    }
    nrows = len(seed_values)
    ncols = len(bin_values)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(8.0, 4.2 * ncols), max(5.0, 3.0 * nrows)),
        sharex=True,
        squeeze=False,
    )
    hcs_osc_array = np.asarray(hcs_osc_values, dtype=float)
    for irow, seed in enumerate(seed_values):
        for icol, etoa_bins in enumerate(bin_values):
            ax = axes[irow, icol]
            percent = np.asarray(
                [by_key[(int(seed), int(etoa_bins), round(float(hcs_osc_amp), 12))] for hcs_osc_amp in hcs_osc_array],
                dtype=float,
            )
            ax.plot(hcs_osc_array, percent, color="tab:blue", marker=".", ms=3.0, lw=1.0)
            ax.axhline(0.0, color="0.35", lw=0.7)
            ax.axvline(0.0, color="0.4", lw=0.7)
            ax.grid(True, alpha=0.25)
            ax.set_title(f"seed={seed}, ETOA bins={etoa_bins}", fontsize=10)
            if irow == nrows - 1:
                ax.set_xlabel("hcs-osc-amp")
            if icol == 0:
                ax.set_ylabel(f"Flux change at {energy:g} GeV (%)")
    fig.suptitle(f"Flux at {energy:g} GeV vs hcs-osc=0", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    path = Path(outdir) / f"flux_{value_token(energy)}gev_percent_matrix.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan direct HelProp spectra over hcs-osc-amp.")
    parser.add_argument("lis_input", type=Path, help="LIS input spectrum passed to HelProp")
    parser.add_argument("--helprop", default="./HelProp", help="Path to the HelProp executable")
    parser.add_argument("--outdir", type=Path, default=REPO_ROOT / "effect_scan")
    parser.add_argument("--hcs-osc-min", type=float, default=DEFAULT_HCS_OSC_MIN)
    parser.add_argument("--hcs-osc-max", type=float, default=DEFAULT_HCS_OSC_MAX)
    parser.add_argument("--hcs-osc-step", type=float, default=DEFAULT_HCS_OSC_STEP)
    parser.add_argument(
        "--seed-values",
        default=",".join(str(value) for value in DEFAULT_SEED_VALUES),
        help="Comma-separated fixed seeds",
    )
    parser.add_argument(
        "--bin-values",
        default=",".join(str(value) for value in DEFAULT_BIN_VALUES),
        help="Comma-separated ETOA bin counts",
    )
    parser.add_argument(
        "--selected-hcs-osc-values",
        default=",".join(f"{value:g}" for value in DEFAULT_SELECTED_HCS_OSC_VALUES),
        help="Comma-separated hcs-osc-amp values for the full-spectrum percentage plot",
    )
    parser.add_argument("--flux-energy", type=float, default=1.0)
    parser.add_argument("--A", type=int, default=1)
    parser.add_argument("--Z", type=int, default=1)
    parser.add_argument("--B0", type=float, default=5.0)
    parser.add_argument("--polarity", type=int, default=-1)
    parser.add_argument("--angle", type=float, default=15.0)
    parser.add_argument("--hcs-omega", type=float, default=1.0)
    parser.add_argument("--D0", type=float, default=5.0)
    parser.add_argument("--R0", type=float, default=1.0)
    parser.add_argument("--indexA", type=float, default=1.0)
    parser.add_argument("--indexB", type=float, default=1.0)
    parser.add_argument("--number", type=int, default=DEFAULT_NUMBER)
    parser.add_argument("--nthread", type=int, default=1)
    parser.add_argument("--etoa-min", type=float, default=DEFAULT_ETOA_MIN)
    parser.add_argument("--etoa-max", type=float, default=DEFAULT_ETOA_MAX)
    parser.add_argument("--elis", default="", help="Optional HelProp --elis min,max,n")
    parser.add_argument("--hcs-table", default="")
    parser.add_argument("--extra-opts", nargs="*", default=[])
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Write manifest and print commands only")
    parser.add_argument("--plot-only", action="store_true", help="Read existing spectra and rebuild plots")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    hcs_osc_values = build_hcs_osc_values(args.hcs_osc_min, args.hcs_osc_max, args.hcs_osc_step)
    seed_values = parse_int_list(args.seed_values)
    bin_values = parse_int_list(args.bin_values)
    selected_hcs_osc_values = parse_float_list(args.selected_hcs_osc_values)
    fixed_options = build_fixed_options(args)
    outdir = Path(args.outdir)

    runs = make_scan_runs(
        helprop=args.helprop,
        lis_input=args.lis_input,
        outdir=outdir,
        fixed_options=fixed_options,
        hcs_osc_values=hcs_osc_values,
        seed_values=seed_values,
        bin_values=bin_values,
        etoa_min=args.etoa_min,
        etoa_max=args.etoa_max,
    )
    manifest = outdir / "manifest.csv"
    flux_out = outdir / "flux_at_energy.csv"
    ensure_unique_outputs([manifest, flux_out, *[run.output for run in runs]])
    write_manifest(manifest, runs)
    print(f"manifest: {manifest}")
    print(f"runs: {len(runs)}")

    if not args.plot_only:
        run_helprop_spectra(
            runs,
            timeout=args.timeout,
            dry_run=args.dry_run,
            jobs=args.jobs,
        )
    if args.dry_run:
        return 0

    records = collect_flux_at_energy(runs, args.flux_energy)
    save_flux_records(flux_out, records)
    print(f"flux table: {flux_out}")
    if not args.no_plot:
        selected_path = plot_selected_flux(
            runs,
            outdir,
            seed=seed_values[0],
            etoa_bins=bin_values[0],
            selected_hcs_osc_values=selected_hcs_osc_values,
        )
        matrix_path = plot_flux_matrix(
            records,
            outdir,
            seed_values=seed_values,
            bin_values=bin_values,
            hcs_osc_values=hcs_osc_values,
            energy=args.flux_energy,
        )
        print(f"selected flux plot: {selected_path}")
        print(f"matrix plot: {matrix_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
