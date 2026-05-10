"""Wrapper for running HelProp as a subprocess with result caching."""

import subprocess
import os
import sys
import tempfile
import numpy as np
from interp import LogInterp


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

    def run(self, D0, m_corot):
        """Run HelProp with (D0, m_corot). Returns LogInterp or None on failure."""
        key = (round(D0, 8), round(m_corot, 8))
        if self.use_cache and key in self._cache:
            self._cache_hits += 1
            if self.verbose:
                print(f"  [cache hit] D0={D0}  m={m_corot}")
            return self._cache[key]

        self._call_count += 1
        out_spec = os.path.join(self._work_dir,
                                f"out_{os.getpid()}_{self._call_count}.dat")

        cmd = [self.helprop_bin]
        cmd.extend(self.common_opts)
        cmd.extend([f"--D0={D0}", f"--m={m_corot}"])
        cmd.extend([self.lis_input, out_spec])

        print(f"  [run #{self._call_count}]  D0={D0}  m={m_corot}  -> {out_spec}")
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

    def stats(self):
        total = self._call_count + self._cache_hits
        return {
            "total_calls": self._call_count,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache),
            "hit_rate": self._cache_hits / max(total, 1),
        }
