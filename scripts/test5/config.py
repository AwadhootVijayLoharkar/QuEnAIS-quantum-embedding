# config.py — Strongly Correlated Molecules Pipeline
# (Full updated file — Phase 1 fixes + Phase 2/3 additions)

import os
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════════
PROJECT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(PROJECT_DIR, "results")
CIF_DIR      = os.path.join(PROJECT_DIR, "cif_files")
STEP0_FILE   = os.path.join(RESULTS_DIR, "step0_classical.pkl")
STEP1_FILE   = os.path.join(RESULTS_DIR, "step1_asf.pkl")
STEP2_FILE   = os.path.join(RESULTS_DIR, "step2_hamiltonian.pkl")
STEP3_FILE   = os.path.join(RESULTS_DIR, "step3_results.pkl")
PLOTS_DIR    = os.path.join(RESULTS_DIR, "plots")

# ═══════════════════════════════════════════════════════════════════════════════
# Block2 / DMRG
# ═══════════════════════════════════════════════════════════════════════════════
BLOCKEXE_WRAPPER = os.path.expanduser("~/block2main_wrapper.sh")

# ═══════════════════════════════════════════════════════════════════════════════
# Molecule Selection
# ═══════════════════════════════════════════════════════════════════════════════
MOLECULE = "TiO2"
CHARGE   = 0
SPIN     = 0        # 2S: 0=singlet, 2=triplet, 4=quintet
BASIS    = "def2-svp"

# ═══════════════════════════════════════════════════════════════════════════════
# CIF Parsing
# ═══════════════════════════════════════════════════════════════════════════════
EXTRACT_MOLECULE = True


def load_geometry(molecule_name):
    """
    Load geometry from a CIF file in CIF_DIR.
    Parses fractional coordinates + cell parameters → Cartesian coords.
    Only parses loops containing _atom_site_fract_x/y/z (ignores aniso loops).
    """
    cif_path = os.path.join(CIF_DIR, f"{molecule_name}.cif")
    if not os.path.exists(cif_path):
        raise FileNotFoundError(
            f"CIF file not found: {cif_path}\n"
            f"Place your .cif files in: {CIF_DIR}/"
        )

    cell_a = cell_b = cell_c = 1.0
    cell_alpha = cell_beta = cell_gamma = 90.0
    atoms = []

    FRAC_KEYS    = {"_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"}
    in_atom_loop = False
    loop_has_frac= False
    atom_keys    = []
    in_multiline = False

    with open(cif_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(";"):
                in_multiline = not in_multiline
                continue
            if in_multiline:
                continue
            if not line or line.startswith("#"):
                continue

            if line.startswith("_cell_length_a"):
                cell_a = _parse_cif_number(line.split()[-1])
            elif line.startswith("_cell_length_b"):
                cell_b = _parse_cif_number(line.split()[-1])
            elif line.startswith("_cell_length_c"):
                cell_c = _parse_cif_number(line.split()[-1])
            elif line.startswith("_cell_angle_alpha"):
                cell_alpha = _parse_cif_number(line.split()[-1])
            elif line.startswith("_cell_angle_beta"):
                cell_beta = _parse_cif_number(line.split()[-1])
            elif line.startswith("_cell_angle_gamma"):
                cell_gamma = _parse_cif_number(line.split()[-1])
            elif line == "loop_":
                in_atom_loop = False; loop_has_frac = False; atom_keys = []
            elif line.startswith("_atom_site_"):
                atom_keys.append(line)
                if line in FRAC_KEYS:
                    loop_has_frac = True
                in_atom_loop = True
            elif in_atom_loop and line and not line.startswith("_"):
                if line.startswith("loop_"):
                    in_atom_loop = False; loop_has_frac = False; atom_keys = []
                    continue
                if not loop_has_frac:
                    continue
                tokens = line.split()
                if len(tokens) < len(atom_keys):
                    continue
                row    = dict(zip(atom_keys, tokens))
                symbol = _extract_element(
                    row.get("_atom_site_type_symbol",
                            row.get("_atom_site_label", "X"))
                )
                if symbol in ("X", ""):
                    continue
                fx = _parse_cif_number(row.get("_atom_site_fract_x", "0"))
                fy = _parse_cif_number(row.get("_atom_site_fract_y", "0"))
                fz = _parse_cif_number(row.get("_atom_site_fract_z", "0"))
                atoms.append((symbol, fx, fy, fz))

    if not atoms:
        raise ValueError(
            f"No atoms parsed from {cif_path}\n"
            f"Check that the CIF contains _atom_site_fract_x/y/z fields."
        )

    for i, (s1, fx1, fy1, fz1) in enumerate(atoms):
        for j, (s2, fx2, fy2, fz2) in enumerate(atoms):
            if i >= j:
                continue
            if abs(fx1-fx2) + abs(fy1-fy2) + abs(fz1-fz2) < 1e-4:
                raise ValueError(
                    f"Atoms {i}({s1}) and {j}({s2}) have identical fractional "
                    f"coordinates — likely a CIF parsing error."
                )

    frac_to_cart = _build_cell_matrix(
        cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma
    )
    geometry = []
    for symbol, fx, fy, fz in atoms:
        cart = frac_to_cart @ np.array([fx, fy, fz])
        geometry.append((symbol, tuple(cart)))

    for i, (s1, c1) in enumerate(geometry):
        for j, (s2, c2) in enumerate(geometry):
            if i >= j:
                continue
            dist = np.linalg.norm(np.array(c1) - np.array(c2))
            if dist < 0.5:
                raise ValueError(
                    f"Atoms {i}({s1}) and {j}({s2}) are {dist:.3f} Å apart "
                    f"— likely a CIF parsing error."
                )
    return geometry


def _parse_cif_number(s):
    return float(s.split("(")[0])


def _extract_element(s):
    elem = ""
    for ch in s:
        if ch.isalpha():
            elem += ch
        else:
            break
    if not elem:
        return "X"
    return elem[0].upper() + elem[1:].lower() if len(elem) > 1 else elem.upper()


def _build_cell_matrix(a, b, c, alpha, beta, gamma):
    alpha_r = np.radians(alpha)
    beta_r  = np.radians(beta)
    gamma_r = np.radians(gamma)
    cos_a, cos_b, cos_g = np.cos(alpha_r), np.cos(beta_r), np.cos(gamma_r)
    sin_g = np.sin(gamma_r)
    ax = a
    bx = b * cos_g;  by = b * sin_g
    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz = np.sqrt(max(0.0, c**2 - cx**2 - cy**2))
    return np.array([[ax, bx, cx], [0., by, cy], [0., 0., cz]])


# ═══════════════════════════════════════════════════════════════════════════════
# Physical Constants
# ═══════════════════════════════════════════════════════════════════════════════
HARTREE_TO_EV       = 27.211386245988   # NIST 2018 CODATA
HARTREE_TO_KCAL_MOL = 627.5094740631

# ═══════════════════════════════════════════════════════════════════════════════
# Tier Classification
# ═══════════════════════════════════════════════════════════════════════════════
TM_ELEMENTS = {
    'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
    'Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',
    'La','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg',
    'Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu',
    'Ac','Th','Pa','U','Np','Pu',
}

SPIN_CONTAMINATION_TIER2_THRESHOLD   = 1.3
SPIN_CONTAMINATION_SINGLET_THRESHOLD = 0.05
HOMO_LUMO_TIER2_THRESHOLD_EV        = 1.0

# ═══════════════════════════════════════════════════════════════════════════════
# Active Space Finder (ASF)
# ═══════════════════════════════════════════════════════════════════════════════
ASF_PARAMS = {
    1: {"entropy_threshold": 0.05,  "max_norb": 12, "min_norb": 2},
    2: {"entropy_threshold": 0.02,  "max_norb": 14, "min_norb": 2},
    3: {"entropy_threshold": 0.005, "max_norb": 16, "min_norb": 4},
}

GAP_MIN_NORB       = 2
GAP_MAX_NORB       = 16
CORE_OCC_THRESHOLD = 1.95

# ═══════════════════════════════════════════════════════════════════════════════
# DMET Embedding
# ═══════════════════════════════════════════════════════════════════════════════
BATH_TOLERANCE = 1e-8
MIN_BATH_ORBS  = 0
MAX_EMBED_ORBS = 24

# ═══════════════════════════════════════════════════════════════════════════════
# Fermion-to-Qubit Mapping
# ═══════════════════════════════════════════════════════════════════════════════
# "jw"  — Jordan-Wigner:   O(N) Pauli string length, simpler
# "bk"  — Bravyi-Kitaev:   O(log N) Pauli string length, fewer gates for SKQD
# BK is preferred for larger systems (n_emb > 8) on real hardware or deep circuits.
FERMION_TO_QUBIT = "bk"    # "jw" | "bk"

# ═══════════════════════════════════════════════════════════════════════════════
# Quantum Solver
# ═══════════════════════════════════════════════════════════════════════════════
QUANTUM_SOLVER = "sqd"      # "sqd" | "skqd" | "sqdrift"
BACKEND        = "mps"      # "local" | "mps" | "ibm"

# --- Ansatz selection for SQD ---
# "su2"  — EfficientSU2: general, does NOT conserve particle number
#           ~30-60% of shots filtered out (expected)
# "lucj" — Local Unitary Cluster Jastrow: conserves particle number by construction
#           0% of shots wasted, physically motivated, faster convergence
ANSATZ = "su2"             # "su2" | "lucj"

# LUCJ parameters
LUCJ_NUM_LAYERS     = 3     # number of orbital rotation + Jastrow layers
LUCJ_MAX_ITERATIONS = 10    # optimization iterations (if variational)

# SU2 parameters (used only when ANSATZ="su2")
ANSATZ_REPS = 3

# Shared SQD
N_SHOTS   = 8192
SQD_ITERS = 10

# SKQD
SKQD_KRYLOV_DIM   = 5
SKQD_DT           = 0.9
SKQD_TROTTER_REPS = 1
SKQD_SHOTS        = 8192

# SqDRIFT
SQDRIFT_NUM_CIRCUITS = 70
SQDRIFT_NUM_GROUPS   = 100
SQDRIFT_TIME         = 2.0
SQDRIFT_ITERS        = 10
SQDRIFT_SHOTS        = 8192

# MPS backend
MPS_MAX_BOND_DIM = 256
MPS_TRUNC_THRESH = 1e-6

# IBM backend
IBM_BACKEND_NAME       = None
IBM_OPTIMIZATION_LEVEL = 1
IBM_MAX_CIRCUIT_DEPTH  = 3000

# ═══════════════════════════════════════════════════════════════════════════════
# Classical Reference Methods  (step0_classical.py)
# ═══════════════════════════════════════════════════════════════════════════════
# Which methods to run. Each adds compute time:
#   HF      — seconds
#   MP2     — seconds to minutes
#   CCSD    — minutes
#   CCSD_T  — minutes to hours (skip for large systems)
#   CASSCF  — minutes (uses active space from Step 1 if available)
#   NEVPT2  — minutes (requires CASSCF)
CLASSICAL_METHODS = ["HF", "MP2"]
# CLASSICAL_METHODS = ["HF", "MP2", "CCSD", "CCSD_T", "CASSCF", "NEVPT2"]  # full

# ═══════════════════════════════════════════════════════════════════════════════
# Geometry Scan  (step4_visualize.py)
# ═══════════════════════════════════════════════════════════════════════════════
# Set GEOMETRY_SCAN = True to enable a bond-length scan for potential energy curve.
# SCAN_ATOM_PAIR: indices of the two atoms whose distance is scanned.
# SCAN_DISTANCES: bond lengths in Angstrom to evaluate.
GEOMETRY_SCAN     = True
SCAN_ATOM_PAIR    = (0, 1)         # Ti(0) — O(1)
SCAN_DISTANCES    = np.linspace(0.9, 4.0, 20)   # Å
SCAN_METHOD       = "MP2"         # which classical method to use for scan

# ═══════════════════════════════════════════════════════════════════════════════
# Resolve geometry at import time
# ═══════════════════════════════════════════════════════════════════════════════
GEOMETRY  = load_geometry(MOLECULE)
ATOM_SYMS = [a[0] for a in GEOMETRY]
N_ATOMS   = len(GEOMETRY)

# ═══════════════════════════════════════════════════════════════════════════════
# Geometry Scan  (step4_visualize.py)
# ═══════════════════════════════════════════════════════════════════════════════
GEOMETRY_SCAN     = True
SCAN_ATOM_PAIR    = (0, 1)
SCAN_DISTANCES    = np.linspace(0.9, 3.25, 10)   # Å
SCAN_METHOD       = "MP2"

# ── Quantum PES scan (added) ──────────────────────────────────────────────────
# QUANTUM_SCAN=True runs SQD at each scan geometry alongside the classical method.
# WARNING: each geometry point reruns Steps 1-3.  Expect 5-30 min per point.
# QUANTUM_SCAN_FAST=True uses reduced shots/iters for speed; less accurate.
QUANTUM_SCAN          = True         # False to skip quantum curve entirely
QUANTUM_SCAN_FAST     = True         # True = fewer shots & iters (recommended)
QUANTUM_SCAN_SHOTS    = 2048         # shots per geometry (fast mode)
QUANTUM_SCAN_ITERS    = 4            # SQD iterations per geometry (fast mode)