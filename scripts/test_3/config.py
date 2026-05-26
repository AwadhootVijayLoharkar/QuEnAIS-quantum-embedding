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

# ── Complexity Classification ─────────────────────────────────────────────────
#
# Step 1 automatically classifies every molecule into one of three tiers:
#
#   Tier 1  Simple closed-shell organic, large HOMO-LUMO gap, no TM
#           Example: H2O, LiH, benzene
#           Cost: ~30 seconds
#
#   Tier 2  Open-shell, small gap, or moderate correlation
#           Example: N2, radicals, strained organics
#           Cost: ~1-2 minutes
#
#   Tier 3  Transition metal / lanthanide / strongly correlated
#           Example: FeN6, Fe-porphyrin, Fe-S clusters
#           Cost: ~3-5 minutes  (ASF uses block2 DMRG internally)
#
# Classification is based on three fast-to-compute indicators:
#   1. Presence of d/f-block elements                → always Tier 3
#   2. UHF spin contamination ratio                  → Tier 2+ if large
#   3. HOMO-LUMO gap                                 → Tier 2+ if small

# d/f-block elements (transition metals, lanthanides, actinides)
TM_ELEMENTS = {
    'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
    'Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',
    'La','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg',
    'Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu',
    'Ac','Th','Pa','U','Np','Pu',
}

# ⟨S²⟩_actual / ⟨S²⟩_expected above this → Tier 2+
# (1.0 = perfect, 1.3 = 30% contamination → open-shell character)
SPIN_CONTAMINATION_TIER2_THRESHOLD = 1.3

# HOMO-LUMO gap below this (eV) → Tier 2+
# (small gap = near-degeneracy = correlation important)
HOMO_LUMO_TIER2_THRESHOLD_EV = 1.0

# ── ASF Parameters Per Tier ───────────────────────────────────────────────────
#
# We ALWAYS use entropy_threshold = 0.01 (very low) so ASF returns a BROAD
# pool of candidate orbitals. The adaptive gap detection (Phase C) then
# selects the final active space — no fixed threshold needed.
#
# max_norb controls the candidate pool size per tier:
#   Tier 1: 12 candidates → gap detection picks 2-8
#   Tier 2: 14 candidates → gap detection picks 2-8
#   Tier 3: 16 candidates → gap detection picks 2-8
#
# The higher max_norb for harder molecules gives gap detection more to
# work with and avoids missing the natural grouping of correlated orbitals.

ASF_PARAMS = {
    1: {"entropy_threshold": 0.01, "max_norb": 12, "min_norb": 2},
    2: {"entropy_threshold": 0.01, "max_norb": 14, "min_norb": 2},
    3: {"entropy_threshold": 0.01, "max_norb": 16, "min_norb": 2},
}

# ── Adaptive Gap Detection ────────────────────────────────────────────────────
#
# After ASF returns candidates, gap detection finds the LARGEST natural gap
# in the deviation spectrum and uses it as the cutoff.
#
# Final active space will have between GAP_MIN_NORB and GAP_MAX_NORB orbitals.
# GAP_MAX_NORB = 8 → DMET embedding ≤ 16 orbitals → ≤ 32 qubits (manageable)
# Lower GAP_MAX_NORB → fewer qubits but possibly missing correlation
# Higher GAP_MAX_NORB → more accurate but more qubits

GAP_MIN_NORB = 2
GAP_MAX_NORB = 8

# NO occupation above this → treated as doubly occupied core (not active)
# Used for electron counting: nel = mol.nelectron - 2 * n_core_orbitals
CORE_OCC_THRESHOLD = 1.8

# ── DMET Embedding ────────────────────────────────────────────────────────────
BATH_TOLERANCE = 1e-8   # Schmidt singular value cutoff for bath orbitals
MAX_EMBED_ORBS = 16     # n_imp + n_bath <= this (controls qubit count)

# ── SQD ───────────────────────────────────────────────────────────────────────
N_SHOTS     = 8192
SQD_ITERS   = 10
ANSATZ_REPS = 3

# ── Execution Backend ─────────────────────────────────────────────────────────
# "local"  → StatevectorSampler   exact statevector, best for testing (≤ ~20 qubits)
# "mps"    → AerSimulator MPS     tensor network, handles larger circuits, approximate
# "ibm"    → IBM Quantum hardware real device, requires saved credentials
BACKEND = "ibm"

# ── MPS Tensor Network Settings (BACKEND = "mps") ─────────────────────────────
# MPS_MAX_BOND_DIM controls the accuracy/cost tradeoff:
#   32  → fast,  good for weakly entangled circuits (shallow EfficientSU2)
#   256 → balanced, recommended starting point
#   512 → high accuracy, slow for > 30 qubits
# Rule of thumb: if energy stops improving when you double bond dim, you are converged.
# MPS_TRUNC_THRESH: singular values below this are discarded during contraction.
#   1e-6  → default, good balance
#   1e-10 → near-exact, significantly slower
MPS_MAX_BOND_DIM = 256
MPS_TRUNC_THRESH = 1e-6

# ── IBM Quantum Settings (BACKEND = "ibm") ────────────────────────────────────
# IBM_BACKEND_NAME : None = auto-select least_busy with enough qubits
#                   or pin to a device e.g. "ibm_brisbane", "ibm_kyiv"
# IBM_OPTIMIZATION_LEVEL:
#   0 → fastest transpile, no guarantees on gate count
#   1 → default, good balance                        ← recommended start
#   2 → deeper optimization, slower transpile
#   3 → most aggressive, slowest compile, lowest gate count
IBM_BACKEND_NAME       = None
IBM_OPTIMIZATION_LEVEL = 1
IBM_MAX_CIRCUIT_DEPTH  = 3000

# ── Quantum Solver Selection ──────────────────────────────────────────────────
# "sqd"  : Sampling-based Quantum Diagonalization (original)
# "skqd" : Sampling-based Krylov Quantum Diagonalization (new)
QUANTUM_SOLVER = "skqd"

# ── SKQD Parameters ───────────────────────────────────────────────────────────
SKQD_KRYLOV_DIM   = 10     # number of Krylov vectors to build
SKQD_DT           = 0.5    # time step per Krylov evolution (Ha^-1)
SKQD_TROTTER_REPS = 2      # Trotter steps per evolution gate (accuracy vs depth)
SKQD_SHOTS        = 8192   # shots per Krylov circuit

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
STEP1_FILE  = os.path.join(RESULTS_DIR, "step1_asf.pkl")
STEP2_FILE  = os.path.join(RESULTS_DIR, "step2_hamiltonian.pkl")