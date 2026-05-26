import os

# ── MKL / block2 ──────────────────────────────────────────────────────────────
BLOCKEXE_WRAPPER = os.path.expanduser("~/block2main_wrapper.sh")

# ── Molecule ──────────────────────────────────────────────────────────────────
MOLECULE = "FeN6"

geometries = {
    "LiH" : [("Li", (0., 0., 0.00)), ("H",  (0., 0., 1.60))],
    "H2O" : [("O",  (0., 0., 0.00)), ("H",  (0.,  0.757,  0.586)),
                                      ("H",  (0., -0.757,  0.586))],
    "N2"  : [("N",  (0., 0., 0.00)), ("N",  (0., 0., 1.098))],
    "H6"  : [("H",  (0., 0., i*0.74)) for i in range(6)],
    "FeN6": [
        ("Fe", (0.000,  0.000,  0.000)),
        ("N",  (0.000,  0.000,  2.000)),
        ("N",  (0.000,  0.000, -2.000)),
        ("N",  (0.000,  2.000,  0.000)),
        ("N",  (0.000, -2.000,  0.000)),
        ("N",  (2.000,  0.000,  0.000)),
        ("N",  (-2.000, 0.000,  0.000)),
    ],
}

GEOMETRY  = geometries[MOLECULE]
ATOM_SYMS = [a[0] for a in GEOMETRY]
N_ATOMS   = len(GEOMETRY)

# ── Basis ─────────────────────────────────────────────────────────────────────
BASIS = "sto-3g"

# ── ASF settings ──────────────────────────────────────────────────────────────
ENTROPY_THRESHOLD = 0.15
MAX_NORB          = 8
MIN_NORB          = 2

# ── DMET settings ─────────────────────────────────────────────────────────────
# Fragment 0 = Fe (1 atom) | Fragment 1 = all 6 N atoms
# Why CCSD for both?
#   DMET Schmidt decomposition gives Fe a ~30-orbital embedding.
#   FCI on 30 orbitals requires 2.62 TiB — impossible.
#   CCSD is polynomial and accurate enough for DMET self-consistency.
FRAGMENT_ATOMS   = [1, 6]
FRAGMENT_SOLVERS = ["ccsd", "ccsd"]
MOST_ACTIVE_FRAG = 0                 # Fe is always fragment index 0

# ── SQD active space settings ─────────────────────────────────────────────────
# After DMET, the Fe embedding has ~30 orbitals.
# We select SQD_MAX_ORBS frontier orbitals (HOMO ± window) for the quantum step.
# SQD_MAX_ORBS → n_qubits = 2 * SQD_MAX_ORBS (keep ≤ 16 for fast simulation)
SQD_MAX_ORBS = 8

# ── SQD circuit / sampling settings ──────────────────────────────────────────
N_SHOTS     = 500_000
SQD_ITERS   = 10
ANSATZ_REPS = 3

# ── Results paths ─────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
STEP1_FILE  = os.path.join(RESULTS_DIR, "step1_asf.pkl")
STEP2_FILE  = os.path.join(RESULTS_DIR, "step2_dmet.pkl")