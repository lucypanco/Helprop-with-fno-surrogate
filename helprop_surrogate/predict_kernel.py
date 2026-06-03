"""Use a saved conditional kernel surrogate to build matrices or spectra."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .file_safety import prepare_output_path
from .kernel import fold_lis
from .model import HelPropKernelModel, load_model, parse_key_value_options


def _read_energy_grid(spec: str) -> np.ndarray:
    path = Path(spec)
    if path.exists():
        data = np.loadtxt(path)
        if data.ndim == 1:
            return np.asarray(data, dtype=float)
        return np.asarray(data[:, 0], dtype=float)

    parts = [float(item) for item in spec.split(",")]
    if len(parts) != 3:
        raise ValueError("energy grid must be a file path or min,max,n")
    emin, emax, nbin = parts
    if emin <= 0.0 or emax <= 0.0 or emax <= emin or nbin < 1:
        raise ValueError("energy grid min,max,n must be positive and increasing")
    return np.geomspace(emin, emax, int(nbin))


def _read_lis(path: Path, elis_grid: np.ndarray) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("LIS file must contain at least two columns: E flux")
    energy = data[:, 0]
    flux = data[:, 1]
    if np.any(energy <= 0.0) or np.any(flux <= 0.0):
        raise ValueError("LIS energies and fluxes must be positive")
    return np.exp(np.interp(np.log(elis_grid), np.log(energy), np.log(flux)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict HelProp transfer matrix or folded spectrum with a kernel surrogate.",
        allow_abbrev=False,
    )
    parser.add_argument("model", type=Path, help="Pickled HelPropKernelModel")
    parser.add_argument("--etoa", default=None, help="TOA grid file or min,max,n")
    parser.add_argument("--elis", default=None, help="LIS grid file or min,max,n")
    parser.add_argument("--param", action="append", default=[], help="Learned parameter as name=value")
    parser.add_argument("--strict-ranges", action="store_true", help="Fail outside saved training ranges")
    parser.add_argument("--matrix-out", type=Path, help="Save transfer matrix as text")
    parser.add_argument("--lis", type=Path, help="LIS spectrum file for folded prediction")
    parser.add_argument("--spectrum-out", type=Path, help="Save folded TOA spectrum as text")
    parser.add_argument(
        "--spectrum-etoa",
        default=None,
        help="Optional final spectrum output grid file or min,max,n; does not change the model matrix grid",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    model = load_model(args.model)
    options = _runtime_options_from_args(model, args.param, unknown)

    etoa_grid = _read_energy_grid(args.etoa) if args.etoa is not None else None
    elis_grid = _read_energy_grid(args.elis) if args.elis is not None else None
    matrix = model.matrix(
        options,
        etoa_grid,
        elis_grid,
        strict_ranges=args.strict_ranges,
    )

    if args.matrix_out is not None:
        prepare_output_path(args.matrix_out)
        np.savetxt(args.matrix_out, matrix, header="rows=ETOA cols=ELIS")
        print(f"matrix saved: {args.matrix_out}")

    if args.spectrum_out is not None:
        if args.lis is None:
            raise ValueError("--lis is required when --spectrum-out is used")
        if elis_grid is None:
            if model.elis_grid is None:
                raise ValueError("--elis is required when model has no stored elis grid")
            elis_grid = np.asarray(model.elis_grid, dtype=float)
        if etoa_grid is None:
            if model.etoa_grid is None:
                raise ValueError("--etoa is required when model has no stored etoa grid")
            etoa_grid = np.asarray(model.etoa_grid, dtype=float)
        lis_flux = _read_lis(args.lis, elis_grid)
        spectrum = fold_lis(matrix, etoa_grid, elis_grid, lis_flux, A=int(model.fixed.get("A", 1)))
        output_etoa_grid = (
            _read_energy_grid(args.spectrum_etoa)
            if args.spectrum_etoa is not None
            else etoa_grid
        )
        output_spectrum = _interpolate_spectrum_output(etoa_grid, spectrum, output_etoa_grid)
        payload = np.column_stack([output_etoa_grid, output_spectrum])
        prepare_output_path(args.spectrum_out)
        np.savetxt(args.spectrum_out, payload, header="E_TOA flux")
        print(f"spectrum saved: {args.spectrum_out}")

    if args.matrix_out is None and args.spectrum_out is None:
        print(matrix)
    return 0


def _runtime_options_from_args(
    model: HelPropKernelModel,
    param_items: list[str],
    unknown: list[str],
) -> dict[str, float]:
    options = parse_key_value_options(param_items)
    index = 0
    while index < len(unknown):
        token = unknown[index]
        if not token.startswith("--"):
            raise ValueError(f"unexpected argument: {token}")
        name_value = token[2:]
        if "=" in name_value:
            name, value = name_value.split("=", 1)
            index += 1
        else:
            name = name_value
            if index + 1 >= len(unknown):
                raise ValueError(f"missing value for --{name}")
            value = unknown[index + 1]
            index += 2
        if name in options:
            raise ValueError(f"duplicate parameter: {name}")
        options[name] = float(value)

    model.theta_from_options(options)
    return options


def _interpolate_spectrum_output(
    source_etoa: np.ndarray,
    source_spectrum: np.ndarray,
    output_etoa: np.ndarray,
) -> np.ndarray:
    source_etoa = np.asarray(source_etoa, dtype=float)
    source_spectrum = np.asarray(source_spectrum, dtype=float)
    output_etoa = np.asarray(output_etoa, dtype=float)
    if source_etoa.ndim != 1 or source_spectrum.ndim != 1 or output_etoa.ndim != 1:
        raise ValueError("spectrum interpolation grids must be one-dimensional")
    if source_etoa.size != source_spectrum.size:
        raise ValueError("source ETOA grid and spectrum length do not match")
    if source_etoa.size == 0 or output_etoa.size == 0:
        raise ValueError("spectrum interpolation grids must not be empty")
    if np.any(source_etoa <= 0.0) or np.any(output_etoa <= 0.0):
        raise ValueError("spectrum ETOA grids must be positive")
    if np.any(~np.isfinite(source_etoa)) or np.any(~np.isfinite(source_spectrum)) or np.any(~np.isfinite(output_etoa)):
        raise ValueError("spectrum interpolation inputs must be finite")
    if np.any(np.diff(source_etoa) <= 0.0) or np.any(np.diff(output_etoa) <= 0.0):
        raise ValueError("spectrum ETOA grids must be strictly increasing")
    if np.any(source_spectrum <= 0.0):
        raise ValueError("spectrum fluxes must be positive for log interpolation")
    if output_etoa[0] < source_etoa[0] or output_etoa[-1] > source_etoa[-1]:
        raise ValueError(
            "--spectrum-etoa must stay inside the trained ETOA range "
            f"[{source_etoa[0]}, {source_etoa[-1]}]"
        )
    return np.exp(
        np.interp(
            np.log(output_etoa),
            np.log(source_etoa),
            np.log(source_spectrum),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
