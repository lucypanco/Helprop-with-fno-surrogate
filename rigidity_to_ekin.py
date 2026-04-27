import numpy as np
import os
import sys

# Rest mass energies (GeV) — fill in by user
m_proton  = 0.93827   # GeV
m_electron = 5.10998e-4  # GeV

def rigidity_to_ekin(R, Z, A):
    """Convert rigidity R (GV) to kinetic energy per nucleon (GeV/nuc).

    Parameters
    ----------
    R : array_like
        Rigidity in GV (GeV/e for charge Z).
    Z : int or float
        Charge number.
    A : int or float
        Nucleon number. Use A <= 0 for electrons.

    Returns
    -------
    ekin_nuc : ndarray
        Kinetic energy per nucleon in GeV/nuc.
        For electrons (A <= 0), returns kinetic energy in GeV.
    """
    R = np.asarray(R, dtype=float)
    p = R * Z  # momentum in GeV/c (since GV * Z = GeV/c for charge Z)

    if A > 0:
        m = A * m_proton
        E = np.sqrt(p**2 + m**2)
        return (E - m) / A
    else:
        m = m_electron
        E = np.sqrt(p**2 + m**2)
        return E - m

if __name__ == "__main__":
    usage = (
        f"Usage:\n"
        f"  {sys.argv[0]} Z A <input_file> [output_file]\n"
        f"  {sys.argv[0]} Z A rigidity_1 rigidity_2 ...\n"
        f"\n"
        f"  Z            charge number\n"
        f"  A            nucleon number (<= 0 for electrons)\n"
        f"  input_file   two-column file (rigidity, value) in GV\n"
        f"  output_file  optional output path (default: stdout)"
    )

    if len(sys.argv) < 4:
        print(usage)
        sys.exit(1)

    Z = float(sys.argv[1])
    A = float(sys.argv[2])

    # Check if the third argument is an existing file
    input_path = sys.argv[3]
    if os.path.isfile(input_path):
        data = np.loadtxt(input_path)
        R = data[:, 0]

        if data.shape[1] == 2:
            values = data[:, 1]
            ekin = rigidity_to_ekin(R, Z, A)
            output_lines = [f"{e:.6e}\t{v:.6e}" for e, v in zip(ekin, values)]
        else:
            ekin = rigidity_to_ekin(R, Z, A)
            output_lines = [f"{e:.6e}" for e in ekin]

        output = "\n".join(output_lines) + "\n"

        if len(sys.argv) >= 5:
            output_path = sys.argv[4]
            with open(output_path, "w") as f:
                f.write(output)
            print(f"Wrote {len(output_lines)} rows to {output_path}")
        else:
            sys.stdout.write(output)
    else:
        # Legacy mode: individual rigidity values as CLI args
        R = [float(x) for x in sys.argv[3:]]
        ekin = rigidity_to_ekin(R, Z, A)
        for r, e in zip(R, ekin):
            print(f"{r:.6e}  {e:.6e}")
