# config.py — Strongly Correlated Molecules Pipeline

import os
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════════
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
CIF_DIR     = os.path.join(PROJECT_DIR, "cif_files")
STEP1_FILE  = os.path.join(RESULTS_DIR, "step1_asf.pkl")
STEP2_FILE  = os.path.join(RESULTS_DIR, "step2_hamiltonian.pkl")

# ═══════════════════════════════════════════════════════════════════════════════
# Block2 / DMRG
# ═══════════════════════════════════════════════════════════════════════════════
BLOCKEXE_WRAPPER = os.path.expanduser("~/block2main_wrapper.sh")

# ═══════════════════════════════════════════════════════════════════════════════
# Molecule Selection
# ═══════════════════════════════════════════════════════════════════════════════
# Name must match a .cif file in CIF_DIR (e.g., "CuCl2.cif")
MOLECULE = "TiO2"

CHARGE = 0
SPIN   = 0       # 2S: 0=singlet, 2=triplet, 4=quintet
BASIS  = "sto-3g"

# ═══════════════════════════════════════════════════════════════════════════════
# CIF Parsing
# ═══════════════════════════════════════════════════════════════════════════════
# CIF files contain crystal structures (periodic). We need a molecule.
# EXTRACT_MOLECULE = True → take one asymmetric unit as a molecular cluster.
EXTRACT_MOLECULE = True


def load_geometry(molecule_name):
    """
    Load geometry from a CIF file in CIF_DIR.

    Parses fractional coordinates + cell parameters → Cartesian coords.
    If EXTRACT_MOLECULE is True, extracts unique atoms (asymmetric unit).
    """
    cif_path = os.path.join(CIF_DIR, f"{molecule_name}.cif")
    if not os.path.exists(cif_path):
        raise FileNotFoundError(
            f"CIF file not found: {cif_path}\n"
            f"Place your .cif files in: {CIF_DIR}/"
        )

    # Parse CIF manually (no heavy dependencies required)
    cell_a = cell_b = cell_c = 1.0
    cell_alpha = cell_beta = cell_gamma = 90.0
    atoms = []  # list of (symbol, frac_x, frac_y, frac_z)

    in_atom_loop = False
    atom_keys = []

    with open(cif_path) as f:
        for line in f:
            line = line.strip()

            # Cell parameters
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

            # Atom site loop
            elif line == "loop_":
                in_atom_loop = False
                atom_keys = []
            elif line.startswith("_atom_site_"):
                atom_keys.append(line)
                in_atom_loop = True
            elif in_atom_loop and line and not line.startswith("_"):
                if line.startswith("loop_") or line.startswith("#"):
                    in_atom_loop = False
                    continue
                tokens = line.split()
                if len(tokens) >= len(atom_keys):
                    row = dict(zip(atom_keys, tokens))
                    symbol = _extract_element(
                        row.get("_atom_site_type_symbol",
                                row.get("_atom_site_label", "X"))
                    )
                    fx = _parse_cif_number(row.get("_atom_site_fract_x", "0"))
                    fy = _parse_cif_number(row.get("_atom_site_fract_y", "0"))
                    fz = _parse_cif_number(row.get("_atom_site_fract_z", "0"))
                    atoms.append((symbol, fx, fy, fz))

    if not atoms:
        raise ValueError(f"No atoms parsed from {cif_path}")

    # Fractional → Cartesian using cell parameters
    frac_to_cart = _build_cell_matrix(
        cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma
    )

    geometry = []
    for symbol, fx, fy, fz in atoms:
        cart = frac_to_cart @ np.array([fx, fy, fz])
        geometry.append((symbol, tuple(cart)))

    return geometry


def _parse_cif_number(s):
    """Parse a CIF numeric field, stripping uncertainty in parentheses."""
    # "0.250(5)" → 0.250, "90" → 90.0
    s = s.split("(")[0]
    return float(s)


def _extract_element(s):
    """Extract element symbol from CIF label like 'Cu2+' or 'Cl1'."""
    elem = ""
    for ch in s:
        if ch.isalpha():
            elem += ch
        else:
            break
    # Capitalize properly: first upper, rest lower
    return elem[0].upper() + elem[1:].lower() if len(elem) > 1 else elem.upper()


def _build_cell_matrix(a, b, c, alpha, beta, gamma):
    """Convert cell parameters (Å, degrees) to fractional→Cartesian matrix."""
    alpha_r = np.radians(alpha)
    beta_r  = np.radians(beta)
    gamma_r = np.radians(gamma)

    cos_a, cos_b, cos_g = np.cos(alpha_r), np.cos(beta_r), np.cos(gamma_r)
    sin_g = np.sin(gamma_r)

    # Standard crystallographic convention (a along x)
    ax = a
    bx = b * cos_g
    by = b * sin_g
    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz = np.sqrt(max(0, c**2 - cx**2 - cy**2))

    return np.array([
        [ax, bx, cx],
        [0., by, cy],
        [0., 0., cz],
    ])


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

SPIN_CONTAMINATION_TIER2_THRESHOLD = 1.3
HOMO_LUMO_TIER2_THRESHOLD_EV      = 1.0

# ═══════════════════════════════════════════════════════════════════════════════
# Active Space Finder (ASF)
# ═══════════════════════════════════════════════════════════════════════════════
ASF_PARAMS = {
    1: {"entropy_threshold": 0.01, "max_norb": 12, "min_norb": 2},
    2: {"entropy_threshold": 0.01, "max_norb": 14, "min_norb": 2},
    3: {"entropy_threshold": 0.01, "max_norb": 16, "min_norb": 2},
}

GAP_MIN_NORB       = 2
GAP_MAX_NORB       = 16
CORE_OCC_THRESHOLD = 1.8

# ═══════════════════════════════════════════════════════════════════════════════
# DMET Embedding
# ═══════════════════════════════════════════════════════════════════════════════
BATH_TOLERANCE = 1e-8
MAX_EMBED_ORBS = 10

# ═══════════════════════════════════════════════════════════════════════════════
# Quantum Solver
# ═══════════════════════════════════════════════════════════════════════════════
QUANTUM_SOLVER = "skqd"       # "sqd" | "skqd" | "sqdrift"
BACKEND        = "mps"     # "local" | "mps" | "ibm"

N_SHOTS     = 8192
ANSATZ_REPS = 3
SQD_ITERS   = 10

SKQD_KRYLOV_DIM   = 5
SKQD_DT           = 0.9
SKQD_TROTTER_REPS = 1
SKQD_SHOTS        = 8192

SQDRIFT_NUM_CIRCUITS = 70
SQDRIFT_NUM_GROUPS   = 100
SQDRIFT_TIME         = 2.0
SQDRIFT_ITERS        = 10
SQDRIFT_SHOTS        = 8192

MPS_MAX_BOND_DIM = 256
MPS_TRUNC_THRESH = 1e-6

IBM_BACKEND_NAME       = None
IBM_OPTIMIZATION_LEVEL = 1
IBM_MAX_CIRCUIT_DEPTH  = 3000

# ═══════════════════════════════════════════════════════════════════════════════
# Resolve geometry at import time
# ═══════════════════════════════════════════════════════════════════════════════
GEOMETRY  = load_geometry(MOLECULE)
ATOM_SYMS = [a[0] for a in GEOMETRY]
N_ATOMS   = len(GEOMETRY)