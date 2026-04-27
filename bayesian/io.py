"""IO functions for the HelProp Bayesian pipeline.

Provides read/write for spectrum and Green function matrix files in TXT, CSV,
and BSON formats, matching the C++ IO.cc interface.

File formats (matching IO.cc):
  TXT spec:    header "# E F", then space-separated "E F" data lines
  CSV spec:    header "#E,F", then comma-separated "E,F" data lines
  BSON spec:   concatenated BSON documents (SpecBson structs)

  TXT matrix:  header "# ELIS[0] ELIS[1] ...", then "ETOA[i] M[i][0] M[i][1] ..." rows
  CSV matrix:  "#ELIS,val0,val1,...", "#ETOA,val0,val1,...", "#Matrix", then CSV rows
  BSON matrix: concatenated BSON documents (MatrixBson structs)

All formats support multi-entry files (1-indexed via ientry).
Unit conversion via eunit parameter (default 1.0 = GeV).
"""

import struct
from pathlib import Path
from typing import Tuple

import numpy as np

# File extension convention per iotype
EXTENSIONS = {"TXT": ".txt", "CSV": ".csv", "BSON": ".bson"}


# ============================================================
# Internal helpers
# ============================================================

def _iter_bson_entries(raw: bytes):
    """Yield raw BSON documents from concatenated BSON data.

    Each BSON document is self-sized (4-byte LE int32 at offset 0).
    Matches the C++ readbson() logic in IO.cc.
    """
    offset = 0
    while offset < len(raw):
        if offset + 4 > len(raw):
            break
        doc_size = struct.unpack_from("<i", raw, offset)[0]
        if doc_size < 4 or offset + doc_size > len(raw):
            break
        yield raw[offset:offset + doc_size]
        offset += doc_size


# ============================================================
# Spec readers
# ============================================================

def read_spec_txt(filename: str, ientry: int = 1, eunit: float = 1.0):
    """Read spectrum from TXT file. Matches IO_TXT::readspec.

    Format:
        # E F
        E[0] F[0]
        E[1] F[1]
        ...
    Entries are separated by "# E F" markers.

    Args:
        filename: path to the file
        ientry: 1-based entry index
        eunit: energy unit conversion factor (default 1.0 = GeV)

    Returns:
        (E, F) as numpy arrays in internal units
    """
    with open(filename) as f:
        lines = f.readlines()

    idata = 0
    data_start = None
    for idx, line in enumerate(lines):
        if line.strip() == "# E F":
            idata += 1
            if idata == ientry:
                data_start = idx + 1
                break

    if data_start is None:
        raise ValueError(f"Entry {ientry} not found in {filename}")

    E, F = [], []
    for line in lines[data_start:]:
        if line.startswith("#"):
            break
        parts = line.split()
        if len(parts) >= 2:
            E.append(float(parts[0]))
            F.append(float(parts[1]))

    return np.array(E) * eunit, np.array(F) / eunit


def read_spec_csv(filename: str, ientry: int = 1, eunit: float = 1.0):
    """Read spectrum from CSV file. Matches IO_CSV::readspec.

    Format:
        #E,F
        E[0],F[0]
        E[1],F[1]
        ...
    Entries are separated by "#E,F" markers.

    Args:
        filename: path to the file
        ientry: 1-based entry index
        eunit: energy unit conversion factor (default 1.0 = GeV)

    Returns:
        (E, F) as numpy arrays in internal units
    """
    with open(filename) as f:
        lines = f.readlines()

    idata = 0
    data_start = None
    for idx, line in enumerate(lines):
        if line.strip() == "#E,F":
            idata += 1
            if idata == ientry:
                data_start = idx + 1
                break

    if data_start is None:
        raise ValueError(f"Entry {ientry} not found in {filename}")

    E, F = [], []
    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped:
            break
        parts = stripped.split(",")
        if len(parts) >= 2:
            E.append(float(parts[0]))
            F.append(float(parts[1]))

    return np.array(E) * eunit, np.array(F) / eunit


def read_spec_bson(filename: str, ientry: int = 1, eunit: float = 1.0):
    """Read spectrum from BSON file. Matches IO_BSON::readspec.

    Format: concatenated BSON documents (SpecBson structs).
    Each document has keys: params, E, F, seed, etoa, elis.

    Args:
        filename: path to the file
        ientry: 1-based entry index
        eunit: energy unit conversion factor (default 1.0 = GeV)

    Returns:
        (E, F, params) where params is a dict of simulation parameters
    """
    try:
        import bson
    except ImportError:
        raise ImportError(
            "bson package required for BSON I/O. Install with: pip install pymongo"
        )

    with open(filename, "rb") as f:
        raw = f.read()

    docs = list(_iter_bson_entries(raw))
    if ientry < 1 or ientry > len(docs):
        raise ValueError(f"Entry {ientry} not found in {filename} ({len(docs)} entries)")

    doc = bson.decode(docs[ientry - 1])
    E = np.array(doc["E"], dtype=float) * eunit
    F = np.array(doc["F"], dtype=float) / eunit
    params = {k: float(v) for k, v in doc.get("params", {}).items()}

    return E, F, params


def read_spec(filename: str, iotype: str = "CSV", ientry: int = 1, eunit: float = 1.0):
    """Read spectrum from file, dispatching by format.

    For TXT/CSV: returns (E, F)
    For BSON: returns (E, F, params)
    """
    iotype = iotype.upper()
    if iotype == "TXT":
        return read_spec_txt(filename, ientry, eunit)
    elif iotype == "CSV":
        return read_spec_csv(filename, ientry, eunit)
    elif iotype == "BSON":
        return read_spec_bson(filename, ientry, eunit)
    else:
        raise ValueError(f"Unknown iotype: {iotype!r}. Use 'TXT', 'CSV', or 'BSON'.")


# ============================================================
# Spec writers
# ============================================================

def write_spec_txt(filename: str, E: np.ndarray, F: np.ndarray,
                   eunit: float = 1.0, mode: str = "recreate"):
    """Write spectrum to TXT file. Matches IO_TXT::writespec.

    Args:
        filename: output path
        E: energy array (internal units)
        F: flux array (internal units)
        eunit: energy unit conversion factor
        mode: "recreate" to overwrite, "append" to append
    """
    E_out = E / eunit
    F_out = F * eunit

    with open(filename, "a" if mode == "append" else "w") as f:
        f.write("# E F\n")
        for e, fl in zip(E_out, F_out):
            f.write(f"{e:.8e} {fl:.8e}\n")


def write_spec_csv(filename: str, E: np.ndarray, F: np.ndarray,
                   eunit: float = 1.0, mode: str = "recreate"):
    """Write spectrum to CSV file. Matches IO_CSV::writespec."""
    E_out = E / eunit
    F_out = F * eunit

    with open(filename, "a" if mode == "append" else "w") as f:
        f.write("#E,F\n")
        for e, fl in zip(E_out, F_out):
            f.write(f"{e:.8e},{fl:.8e}\n")


def write_spec(filename: str, E: np.ndarray, F: np.ndarray,
               iotype: str = "CSV", eunit: float = 1.0, mode: str = "recreate"):
    """Write spectrum to file, dispatching by format."""
    iotype = iotype.upper()
    if iotype == "TXT":
        write_spec_txt(filename, E, F, eunit, mode)
    elif iotype == "CSV":
        write_spec_csv(filename, E, F, eunit, mode)
    elif iotype == "BSON":
        raise NotImplementedError("BSON spec write not implemented in Python "
                                  "(use HelProp C++ for BSON output)")
    else:
        raise ValueError(f"Unknown iotype: {iotype!r}. Use 'TXT', 'CSV', or 'BSON'.")


# ============================================================
# Matrix readers
# ============================================================

def read_matrix_txt(filename: str, ientry: int = 1, eunit: float = 1.0):
    """Read Green function matrix from TXT file. Matches IO_TXT::readmatrix.

    Format:
        # ELIS[0] ELIS[1] ... ELIS[N-1]
        ETOA[0] M[0][0] M[0][1] ... M[0][N-1]
        ETOA[1] M[1][0] M[1][1] ... M[1][N-1]
        ...
    Entries are separated by '#' header lines.

    Args:
        filename: path to the file
        ientry: 1-based entry index
        eunit: energy unit conversion factor (default 1.0 = GeV)

    Returns:
        (ETOA, ELIS, M) as numpy arrays; M is 2D with shape [n_toa, n_lis]
    """
    with open(filename) as f:
        lines = f.readlines()

    # Find entry start: entries separated by lines starting with '#'
    idata = 0
    entry_start = None
    for idx, line in enumerate(lines):
        if line.startswith("#"):
            idata += 1
            if idata == ientry:
                entry_start = idx
                break

    if entry_start is None:
        raise ValueError(f"Entry {ientry} not found in {filename}")

    # Parse header: "# ELIS[0] ELIS[1] ..." -> skip '#' and split
    header = lines[entry_start].strip()
    elis_str = header[1:].strip()  # remove leading '#'
    ELIS = np.array([float(x) for x in elis_str.split()]) * eunit

    # Parse data rows until next '#' or EOF
    ETOA_list = []
    rows = []
    for line in lines[entry_start + 1:]:
        if line.startswith("#"):
            break
        parts = line.split()
        if len(parts) < 2:
            continue
        ETOA_list.append(float(parts[0]))
        rows.append([float(x) for x in parts[1:]])

    ETOA = np.array(ETOA_list) * eunit
    M = np.array(rows)

    return ETOA, ELIS, M


def read_matrix_csv(filename: str, ientry: int = 1, eunit: float = 1.0):
    """Read Green function matrix from CSV file. Matches IO_CSV::readmatrix.

    Format:
        #ELIS,val0,val1,...,valN
        #ETOA,val0,val1,...,valN
        #Matrix
        M[0][0],M[0][1],...,M[0][N-1]
        M[1][0],M[1][1],...,M[1][N-1]
        ...
    Entries are separated by '#ELIS,' markers.

    Args:
        filename: path to the file
        ientry: 1-based entry index
        eunit: energy unit conversion factor (default 1.0 = GeV)

    Returns:
        (ETOA, ELIS, M) as numpy arrays; M is 2D with shape [n_toa, n_lis]
    """
    with open(filename) as f:
        lines = f.readlines()

    # Find the ientry-th "#ELIS," line
    idata = 0
    elis_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("#ELIS,"):
            idata += 1
            if idata == ientry:
                elis_idx = idx
                break

    if elis_idx is None:
        raise ValueError(f"Entry {ientry} not found in {filename}")

    # Parse ELIS: "#ELIS,val0,val1,..."
    elis_line = lines[elis_idx].strip()[6:]  # skip "#ELIS,"
    ELIS = np.array([float(x) for x in elis_line.split(",") if x.strip()]) * eunit

    # Parse ETOA: "#ETOA,val0,val1,..."
    etoa_line = lines[elis_idx + 1].strip()[6:]  # skip "#ETOA,"
    ETOA = np.array([float(x) for x in etoa_line.split(",") if x.strip()]) * eunit

    # Skip "#Matrix" line, read data rows
    matrix_start = elis_idx + 3  # +1 ETOA line, +1 "#Matrix" line, +1 first data line
    rows = []
    for line in lines[matrix_start:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            break
        rows.append([float(x) for x in stripped.split(",") if x.strip()])

    M = np.array(rows)

    return ETOA, ELIS, M


def read_matrix_bson(filename: str, ientry: int = 1, eunit: float = 1.0):
    """Read Green function matrix from BSON file. Matches IO_BSON::readmatrix.

    Format: concatenated BSON documents (MatrixBson structs).
    Each document has keys: params, seed, ETOA, ELIS, etoa, elis, M.

    Args:
        filename: path to the file
        ientry: 1-based entry index
        eunit: energy unit conversion factor (default 1.0 = GeV)

    Returns:
        (ETOA, ELIS, M, params) where params is a dict of simulation parameters
    """
    try:
        import bson
    except ImportError:
        raise ImportError(
            "bson package required for BSON I/O. Install with: pip install pymongo"
        )

    with open(filename, "rb") as f:
        raw = f.read()

    docs = list(_iter_bson_entries(raw))
    if ientry < 1 or ientry > len(docs):
        raise ValueError(f"Entry {ientry} not found in {filename} ({len(docs)} entries)")

    doc = bson.decode(docs[ientry - 1])
    ETOA = np.array(doc["ETOA"], dtype=float) * eunit
    ELIS = np.array(doc["ELIS"], dtype=float) * eunit
    M = np.array(doc["M"], dtype=float)
    params = {k: float(v) for k, v in doc.get("params", {}).items()}

    return ETOA, ELIS, M, params


def read_matrix(filename: str, iotype: str = "CSV", ientry: int = 1, eunit: float = 1.0):
    """Read matrix from file, dispatching by format.

    For TXT/CSV: returns (ETOA, ELIS, M)
    For BSON: returns (ETOA, ELIS, M, params)
    """
    iotype = iotype.upper()
    if iotype == "TXT":
        return read_matrix_txt(filename, ientry, eunit)
    elif iotype == "CSV":
        return read_matrix_csv(filename, ientry, eunit)
    elif iotype == "BSON":
        return read_matrix_bson(filename, ientry, eunit)
    else:
        raise ValueError(f"Unknown iotype: {iotype!r}. Use 'TXT', 'CSV', or 'BSON'.")


# ============================================================
# Matrix writers
# ============================================================

def write_matrix_txt(filename: str, ETOA: np.ndarray, ELIS: np.ndarray,
                     M: np.ndarray, eunit: float = 1.0, mode: str = "recreate"):
    """Write Green function matrix to TXT file. Matches IO_TXT::writematrix.

    Args:
        filename: output path
        ETOA: TOA energy array (internal units)
        ELIS: LIS energy array (internal units)
        M: 2D weight matrix [n_toa, n_lis]
        eunit: energy unit conversion factor
        mode: "recreate" to overwrite, "append" to append
    """
    ETOA_out = ETOA / eunit
    ELIS_out = ELIS / eunit

    with open(filename, "a" if mode == "append" else "w") as f:
        f.write("#")
        for el in ELIS_out:
            f.write(f" {el:.8e}")
        f.write("\n")
        for i in range(M.shape[0]):
            f.write(f"{ETOA_out[i]:.8e}")
            for j in range(M.shape[1]):
                f.write(f" {M[i, j]:.8e}")
            f.write("\n")


def write_matrix_csv(filename: str, ETOA: np.ndarray, ELIS: np.ndarray,
                     M: np.ndarray, eunit: float = 1.0, mode: str = "recreate"):
    """Write Green function matrix to CSV file. Matches IO_CSV::writematrix."""
    ETOA_out = ETOA / eunit
    ELIS_out = ELIS / eunit

    with open(filename, "a" if mode == "append" else "w") as f:
        f.write("#ELIS,")
        f.write(",".join(f"{x:.8e}" for x in ELIS_out))
        f.write("\n")

        f.write("#ETOA,")
        f.write(",".join(f"{x:.8e}" for x in ETOA_out))
        f.write("\n")

        f.write("#Matrix\n")

        for i in range(M.shape[0]):
            f.write(",".join(f"{M[i, j]:.8e}" for j in range(M.shape[1])))
            f.write("\n")


def write_matrix(filename: str, ETOA: np.ndarray, ELIS: np.ndarray,
                 M: np.ndarray, iotype: str = "CSV", eunit: float = 1.0,
                 mode: str = "recreate"):
    """Write matrix to file, dispatching by format."""
    iotype = iotype.upper()
    if iotype == "TXT":
        write_matrix_txt(filename, ETOA, ELIS, M, eunit, mode)
    elif iotype == "CSV":
        write_matrix_csv(filename, ETOA, ELIS, M, eunit, mode)
    elif iotype == "BSON":
        raise NotImplementedError("BSON matrix write not implemented in Python "
                                  "(use HelProp C++ for BSON output)")
    else:
        raise ValueError(f"Unknown iotype: {iotype!r}. Use 'TXT', 'CSV', or 'BSON'.")
