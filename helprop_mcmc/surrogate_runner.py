"""Surrogate-backed runner with the same call shape as HelPropRunner."""

from __future__ import annotations

import os
import sys

import numpy as np

from interp import LogInterp


class SurrogateRunner:
    """Return modulated spectra from a saved HelProp surrogate model."""

    def __init__(self, model_path, lis_input, fixed_options=None, verbose=False):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from helprop_surrogate.model import load_model

        self.model = load_model(model_path)
        self.lis_input = os.path.abspath(lis_input)
        self.fixed_options = dict(fixed_options or {})
        self.verbose = verbose
        self._cache = {}
        self._call_count = 0
        self._cache_hits = 0

        lis_data = np.loadtxt(self.lis_input)
        if lis_data.ndim != 2 or lis_data.shape[1] < 2:
            raise ValueError("LIS file must contain at least two columns: E flux")
        self._lis_energy = lis_data[:, 0]
        self._lis_flux = lis_data[:, 1]
        if np.any(self._lis_energy <= 0.0) or np.any(self._lis_flux <= 0.0):
            raise ValueError("LIS energies and fluxes must be positive")

    def run(self, D0, m_corot):
        options = {
            name: value
            for name, value in self.fixed_options.items()
            if name in self.model.learned
        }
        if "D0" in self.model.learned:
            options["D0"] = D0
        if "m" in self.model.learned:
            options["m"] = m_corot

        key = tuple((name, round(float(options[name]), 8)) for name in self.model.learned)
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
        stored = getattr(self.model, f"{name}_grid")
        if stored is not None:
            return np.asarray(stored, dtype=float)
        return self._lis_energy

    def _interpolate_lis(self, elis_grid):
        return np.exp(
            np.interp(
                np.log(elis_grid),
                np.log(self._lis_energy),
                np.log(self._lis_flux),
            )
        )
