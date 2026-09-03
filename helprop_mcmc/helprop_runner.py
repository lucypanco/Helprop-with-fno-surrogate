"""Wrapper for running HelProp as a subprocess with result caching."""

import subprocess
import os
import sys
import tempfile
import numpy as np
from interp import LogInterp
from formula_lis import (
    DEFAULT_SHEN_LIS_PARAMS,
    SHEN_LIS_ALL_NAMES,
    write_shen_lis,
)


def _theta_options(D0=None, m_corot=None, theta=None):
    """Normalize the runner's positional and full-theta call conventions."""
    if theta is not None:
        return dict(theta)
    if isinstance(D0, dict):
        return dict(D0)
    return {"D0": float(D0), "m": float(m_corot)}


class HelPropRunner:
    """Run HelProp with given (D0, m_corot) and return the modulated spectrum.

    Features:
      - In-memory cache keyed on (D0, m_corot) to avoid redundant runs.
      - Automatic temporary output file management.
      - Configurable timeout and error handling.
    """

    def __init__(self, helprop_bin, lis_input, common_opts,
                 work_dir=None, verbose=False, timeout=600,
                 use_cache=True):
        self.helprop_bin = os.path.abspath(helprop_bin)
        self.lis_input = os.path.abspath(lis_input)
        self.common_opts = list(common_opts)
        self.verbose = verbose
        self.timeout = timeout
        self.use_cache = use_cache
        self._cache = {}
        self._call_count = 0
        self._cache_hits = 0

        if work_dir is None:
            self._work_dir = tempfile.mkdtemp(prefix="helprop_mcmc_")
        else:
            self._work_dir = work_dir
            os.makedirs(work_dir, exist_ok=True)

    def run(self, D0=None, m_corot=None, theta=None):
        """Run HelProp with a two-parameter or full-theta call."""
        options = _theta_options(D0, m_corot, theta)
        return self._run_with_lis(options, self.lis_input,
                                  (round(options["D0"], 8), round(options["m"], 8)))

    def _run_with_lis(self, options, lis_input, key):
        """Run HelProp using ``lis_input`` and cache by ``key``."""
        key = tuple(key)
        if self.use_cache and key in self._cache:
            self._cache_hits += 1
            if self.verbose:
                print(f"  [cache hit] key={key}")
            return self._cache[key]

        self._call_count += 1
        out_spec = os.path.join(self._work_dir,
                                f"out_{os.getpid()}_{self._call_count}.dat")

        cmd = [self.helprop_bin]
        cmd.extend(self.common_opts)
        cmd.extend([f"--D0={options['D0']}", f"--m={options['m']}"])
        cmd.extend([lis_input, out_spec])

        print(f"  [run #{self._call_count}]  key={key}  -> {out_spec}")
        if self.verbose:
            print("    cmd:", " ".join(cmd))
        sys.stdout.flush()

        try:
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=self.timeout
            )
            if self.verbose:
                if result.stdout:
                    print("    HelProp stdout:", result.stdout.strip())
                if result.stderr:
                    print("    HelProp stderr:", result.stderr.strip())
            if result.returncode != 0:
                if result.stderr:
                    print(f"  HelProp stderr: {result.stderr[:500]}")
                return None

            data = np.loadtxt(out_spec)
            if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
                return None

            E_mod, F_mod = data[:, 0], data[:, 1]
            if np.any(F_mod <= 0) or np.any(E_mod <= 0):
                return None

            interp = LogInterp(E_mod, F_mod)
            self._cache[key] = interp
            return interp

        except (subprocess.TimeoutExpired, FileNotFoundError,
                ValueError, OSError) as e:
            if self.verbose:
                print(f"  HelProp failed: {e}")
            return None


class FormulaLISRunner(HelPropRunner):
    """HelProp runner whose LIS is generated from Shen et al. Equation (5).

    The file is an implementation detail for the existing HelProp executable;
    callers provide only MCMC parameters, never a user LIS file.  Each worker
    process gets its own generated file to avoid concurrent writes.
    """

    def __init__(self, helprop_bin, common_opts, lis_energy,
                 fixed_lis_params=None, work_dir=None, verbose=False,
                 timeout=600, use_cache=True):
        super().__init__(helprop_bin, "", common_opts, work_dir=work_dir,
                         verbose=verbose, timeout=timeout,
                         use_cache=use_cache)
        self.lis_energy = np.asarray(lis_energy, dtype=float)
        if self.lis_energy.ndim != 1 or self.lis_energy.size < 2:
            raise ValueError("formula LIS energy grid must contain at least two points")
        if np.any(~np.isfinite(self.lis_energy)) or np.any(self.lis_energy <= 0.0):
            raise ValueError("formula LIS energy grid must be finite and positive")
        if np.any(np.diff(self.lis_energy) <= 0.0):
            raise ValueError("formula LIS energy grid must be strictly increasing")
        self.fixed_lis_params = dict(DEFAULT_SHEN_LIS_PARAMS)
        self.fixed_lis_params.update(fixed_lis_params or {})

    def run(self, D0=None, m_corot=None, theta=None):
        options = _theta_options(D0, m_corot, theta)
        params = dict(self.fixed_lis_params)
        for name in SHEN_LIS_ALL_NAMES:
            if name in options:
                params[name] = options[name]

        key = tuple(
            [(name, round(float(options[name]), 8))
             for name in ("D0", "m")]
            + [(name, round(float(params[name]), 8))
               for name in SHEN_LIS_ALL_NAMES]
        )
        if self.use_cache and key in self._cache:
            self._cache_hits += 1
            return self._cache[key]

        # The PID prevents collisions when a multiprocessing MCMC uses a
        # shared runner work directory.
        lis_path = os.path.join(self._work_dir, f"lis_formula_{os.getpid()}.dat")
        try:
            write_shen_lis(lis_path, self.lis_energy, params)
        except (ValueError, OSError) as exc:
            if self.verbose:
                print(f"  Formula LIS failed: {exc}")
            return None
        return self._run_with_lis(options, lis_path, key)

    def stats(self):
        total = self._call_count + self._cache_hits
        return {
            "total_calls": self._call_count,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache),
            "hit_rate": self._cache_hits / max(total, 1),
        }
