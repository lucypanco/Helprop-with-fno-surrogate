"""Surrogate-backed runner with the same call shape as HelPropRunner."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping

import numpy as np

from interp import LogInterp


def _ensure_repo_root_on_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _load_lis(lis_input):
    lis_path = os.path.abspath(lis_input)
    lis_data = np.loadtxt(lis_path)
    if lis_data.ndim != 2 or lis_data.shape[1] < 2:
        raise ValueError("LIS file must contain at least two columns: E flux")
    lis_energy = lis_data[:, 0]
    lis_flux = lis_data[:, 1]
    if np.any(lis_energy <= 0.0) or np.any(lis_flux <= 0.0):
        raise ValueError("LIS energies and fluxes must be positive")
    if np.any(np.diff(lis_energy) <= 0.0):
        raise ValueError("LIS energies must be strictly increasing")
    return lis_path, lis_energy, lis_flux


def _resolve_grid(model, name, fallback_grid):
    stored = getattr(model, f"{name}_grid")
    if stored is not None:
        return np.asarray(stored, dtype=float)
    return np.asarray(fallback_grid, dtype=float)


def _interpolate_positive(source_energy, source_flux, target_energy):
    source_energy = np.asarray(source_energy, dtype=float)
    source_flux = np.asarray(source_flux, dtype=float)
    target_energy = np.asarray(target_energy, dtype=float)
    if target_energy.size == 0:
        return np.asarray([], dtype=float)
    if np.any(source_energy <= 0.0) or np.any(source_flux <= 0.0) or np.any(target_energy <= 0.0):
        raise ValueError("energies and fluxes must be positive for log interpolation")
    if np.any(~np.isfinite(source_energy)) or np.any(~np.isfinite(source_flux)) or np.any(~np.isfinite(target_energy)):
        raise ValueError("energies and fluxes must be finite")
    if np.any(np.diff(source_energy) <= 0.0):
        raise ValueError("source energy grid must be strictly increasing")
    eps = 1.0e-12
    if target_energy[0] < source_energy[0] * (1.0 - eps) or target_energy[-1] > source_energy[-1] * (1.0 + eps):
        raise ValueError("target energy grid is outside source energy range")
    clipped = np.clip(target_energy, source_energy[0], source_energy[-1])
    return np.exp(np.interp(np.log(clipped), np.log(source_energy), np.log(source_flux)))


def _interpolate_lis_flux(source_energy, source_flux, target_energy):
    source_energy = np.asarray(source_energy, dtype=float)
    source_flux = np.asarray(source_flux, dtype=float)
    target_energy = np.asarray(target_energy, dtype=float)
    return np.exp(np.interp(np.log(target_energy), np.log(source_energy), np.log(source_flux)))


def _options_for_model(model, fixed_options, D0=None, m_corot=None, theta_options=None):
    options = {
        name: value
        for name, value in fixed_options.items()
        if name in model.learned
    }
    if theta_options:
        options.update(
            (name, value)
            for name, value in theta_options.items()
            if name in model.learned
        )
    if D0 is not None and "D0" in model.learned:
        options["D0"] = D0
    if m_corot is not None and "m" in model.learned:
        options["m"] = m_corot
    return options


def _theta_options_from_args(D0=None, m_corot=None, theta=None):
    if theta is not None:
        return dict(theta)
    if isinstance(D0, Mapping):
        return dict(D0)
    return None


def _cache_key(model, options):
    return tuple((name, round(float(options[name]), 8)) for name in model.learned)


class SurrogateRunner:
    """Return modulated spectra from a saved HelProp surrogate model."""

    def __init__(self, model_path, lis_input, fixed_options=None, verbose=False):
        _ensure_repo_root_on_path()

        from helprop_surrogate.model import load_model

        self.model = load_model(model_path)
        self.lis_input, self._lis_energy, self._lis_flux = _load_lis(lis_input)
        self.fixed_options = dict(fixed_options or {})
        self.verbose = verbose
        self._cache = {}
        self._call_count = 0
        self._cache_hits = 0

    def learned_parameters(self):
        return tuple(self.model.learned)

    def run(self, D0=None, m_corot=None, theta=None):
        try:
            theta_options = _theta_options_from_args(D0, m_corot, theta)
            positional_D0 = None if theta_options is not None else D0
            positional_m = None if theta_options is not None else m_corot
            options = _options_for_model(
                self.model,
                self.fixed_options,
                positional_D0,
                positional_m,
                theta_options=theta_options,
            )
            key = _cache_key(self.model, options)
        except Exception as exc:
            if self.verbose:
                print(f"  Surrogate options failed: {exc}")
            return None

        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]

        self._call_count += 1
        try:
            etoa = self._resolve_grid("etoa")
            elis = self._resolve_grid("elis")
            lis_flux = self._interpolate_lis(elis)
            flux = self.model.predict_spectrum(options, lis_flux, etoa, elis)
            if np.any(~np.isfinite(flux)) or np.any(flux <= 0.0):
                return None
            interp = LogInterp(etoa, flux)
            self._cache[key] = interp
            return interp
        except Exception as exc:
            if self.verbose:
                print(f"  Surrogate failed: {exc}")
            return None

    def stats(self):
        total = self._call_count + self._cache_hits
        return {
            "total_calls": self._call_count,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache),
            "hit_rate": self._cache_hits / max(total, 1),
        }

    def _resolve_grid(self, name):
        return _resolve_grid(self.model, name, self._lis_energy)

    def _interpolate_lis(self, elis_grid):
        return _interpolate_lis_flux(self._lis_energy, self._lis_flux, elis_grid)


class CompositeSurrogateRunner:
    """Splice two saved surrogate spectra into one MCMC forward model."""

    def __init__(
        self,
        low_model_path,
        high_model_path,
        lis_input,
        split_energy=1.0,
        blend_dex=0.2,
        fixed_options=None,
        verbose=False,
    ):
        _ensure_repo_root_on_path()

        from helprop_surrogate.model import load_model

        self.low_model = load_model(low_model_path)
        self.high_model = load_model(high_model_path)
        self.lis_input, self._lis_energy, self._lis_flux = _load_lis(lis_input)
        self.split_energy = float(split_energy)
        self.blend_dex = float(blend_dex)
        self.fixed_options = dict(fixed_options or {})
        self.verbose = verbose
        self._cache = {}
        self._call_count = 0
        self._cache_hits = 0

        if not np.isfinite(self.split_energy) or self.split_energy <= 0.0:
            raise ValueError("split_energy must be positive and finite")
        if not np.isfinite(self.blend_dex) or self.blend_dex < 0.0:
            raise ValueError("blend_dex must be finite and non-negative")
        self._check_blend_coverage()

    def learned_parameters(self):
        names = list(self.low_model.learned)
        names.extend(name for name in self.high_model.learned if name not in names)
        return tuple(names)

    def run(self, D0=None, m_corot=None, theta=None):
        try:
            theta_options = _theta_options_from_args(D0, m_corot, theta)
            positional_D0 = None if theta_options is not None else D0
            positional_m = None if theta_options is not None else m_corot
            low_options = _options_for_model(
                self.low_model,
                self.fixed_options,
                positional_D0,
                positional_m,
                theta_options=theta_options,
            )
            high_options = _options_for_model(
                self.high_model,
                self.fixed_options,
                positional_D0,
                positional_m,
                theta_options=theta_options,
            )
            key = (
                ("low", _cache_key(self.low_model, low_options)),
                ("high", _cache_key(self.high_model, high_options)),
            )
        except Exception as exc:
            if self.verbose:
                print(f"  Composite surrogate options failed: {exc}")
            return None

        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]

        self._call_count += 1
        try:
            low_etoa, low_flux = self._predict_model(self.low_model, low_options)
            high_etoa, high_flux = self._predict_model(self.high_model, high_options)
            etoa, flux = self._splice(low_etoa, low_flux, high_etoa, high_flux)
            if np.any(~np.isfinite(flux)) or np.any(flux <= 0.0):
                return None
            interp = LogInterp(etoa, flux)
            self._cache[key] = interp
            return interp
        except Exception as exc:
            if self.verbose:
                print(f"  Composite surrogate failed: {exc}")
            return None

    def stats(self):
        total = self._call_count + self._cache_hits
        return {
            "total_calls": self._call_count,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache),
            "hit_rate": self._cache_hits / max(total, 1),
        }

    def _predict_model(self, model, options):
        etoa = _resolve_grid(model, "etoa", self._lis_energy)
        elis = _resolve_grid(model, "elis", self._lis_energy)
        lis_flux = _interpolate_lis_flux(self._lis_energy, self._lis_flux, elis)
        flux = model.predict_spectrum(options, lis_flux, etoa, elis)
        if np.any(~np.isfinite(flux)) or np.any(flux <= 0.0):
            raise ValueError("model returned non-positive or non-finite flux")
        return etoa, flux

    def _blend_bounds(self):
        if self.blend_dex == 0.0:
            return self.split_energy, self.split_energy
        half_width = 10.0 ** (self.blend_dex / 2.0)
        return self.split_energy / half_width, self.split_energy * half_width

    def _check_blend_coverage(self):
        blend_lo, blend_hi = self._blend_bounds()
        for label, model in (("low", self.low_model), ("high", self.high_model)):
            etoa = _resolve_grid(model, "etoa", self._lis_energy)
            if etoa.ndim != 1 or etoa.size < 2:
                raise ValueError(f"{label} model ETOA grid must contain at least two points")
            if np.any(etoa <= 0.0) or np.any(~np.isfinite(etoa)) or np.any(np.diff(etoa) <= 0.0):
                raise ValueError(f"{label} model ETOA grid must be finite, positive, and increasing")
            if etoa[0] > blend_lo or etoa[-1] < blend_hi:
                raise ValueError(
                    f"{label} model ETOA grid [{etoa[0]}, {etoa[-1]}] "
                    f"does not cover blend window [{blend_lo}, {blend_hi}]"
                )

    def _splice(self, low_etoa, low_flux, high_etoa, high_flux):
        blend_lo, blend_hi = self._blend_bounds()
        if self.blend_dex == 0.0:
            grid = np.concatenate([
                low_etoa[low_etoa < self.split_energy],
                np.asarray([self.split_energy]),
                high_etoa[high_etoa > self.split_energy],
            ])
            etoa = np.unique(grid)
            low_mask = etoa < self.split_energy
            flux = np.empty_like(etoa)
            flux[low_mask] = _interpolate_positive(low_etoa, low_flux, etoa[low_mask])
            flux[~low_mask] = _interpolate_positive(high_etoa, high_flux, etoa[~low_mask])
            return etoa, flux

        grid = np.concatenate([
            low_etoa[low_etoa < blend_hi],
            high_etoa[high_etoa > blend_lo],
            np.asarray([blend_lo, self.split_energy, blend_hi]),
        ])
        etoa = np.unique(grid)
        flux = np.empty_like(etoa)

        low_mask = etoa < blend_lo
        high_mask = etoa > blend_hi
        blend_mask = ~(low_mask | high_mask)

        flux[low_mask] = _interpolate_positive(low_etoa, low_flux, etoa[low_mask])
        flux[high_mask] = _interpolate_positive(high_etoa, high_flux, etoa[high_mask])

        blend_etoa = etoa[blend_mask]
        low_blend = _interpolate_positive(low_etoa, low_flux, blend_etoa)
        high_blend = _interpolate_positive(high_etoa, high_flux, blend_etoa)
        weights = (np.log(blend_etoa) - np.log(blend_lo)) / (np.log(blend_hi) - np.log(blend_lo))
        flux[blend_mask] = np.exp((1.0 - weights) * np.log(low_blend) + weights * np.log(high_blend))
        return etoa, flux
