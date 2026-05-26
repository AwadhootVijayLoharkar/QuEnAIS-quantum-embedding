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

# ── DMET embedding settings ───────────────────────────────────────────────────
# Bath orbitals with Schmidt value below this are discarded
# Lower  → more bath orbitals (larger embedding, more accurate, more qubits)
# Higher → fewer bath orbitals (smaller embedding, faster, fewer qubits)
BATH_TOLERANCE = 1e-8

# Hard cap on total embedding orbitals: n_imp + n_bath <= MAX_EMBED_ORBS
# This controls how many qubits SQD will use: n_qubits = 2 * MAX_EMBED_ORBS
# 16 → 32 qubits  (good for statevector simulation)
# 12 → 24 qubits  (faster)
#  8 → 16 qubits  (fastest, less accurate bath)
MAX_EMBED_ORBS = 10

# ── SQD settings ─────────────────────────────────────────────────────────────
N_SHOTS     = 5000
SQD_ITERS   = 10
ANSATZ_REPS = 2

# ── Results paths ─────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
STEP1_FILE  = os.path.join(RESULTS_DIR, "step1_asf.pkl")
STEP2_FILE  = os.path.join(RESULTS_DIR, "step2_hamiltonian.pkl")