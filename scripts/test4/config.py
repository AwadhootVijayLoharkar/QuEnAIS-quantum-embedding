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

    Fixes vs original:
      - Only parses loops that contain _atom_site_fract_x/y/z (ignores aniso loops)
      - Skips comment lines and multi-line string blocks
      - Skips atoms with symbol 'X' (fallback symbol = parsing failure)
      - Validates no duplicate fractional coordinates
      - Validates no atoms unreasonably close in Cartesian space
    """
    cif_path = os.path.join(CIF_DIR, f"{molecule_name}.cif")
    if not os.path.exists(cif_path):
        raise FileNotFoundError(
            f"CIF file not found: {cif_path}\n"
            f"Place your .cif files in: {CIF_DIR}/"
        )

    cell_a = cell_b = cell_c = 1.0
    cell_alpha = cell_beta = cell_gamma = 90.0
    atoms = []  # list of (symbol, frac_x, frac_y, frac_z)

    # Keys that define a coordinate loop (all three must be present to parse it)
    FRAC_KEYS = {
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    }

    in_atom_loop  = False
    loop_has_frac = False   # True only when current loop has all frac coord keys
    atom_keys     = []
    in_multiline  = False   # track ; ... ; string blocks

    with open(cif_path) as f:
        for line in f:
            line = line.strip()

            # ── Multi-line string block handling ──────────────────────────────
            # CIF uses ; at start of line to open/close multi-line strings
            if line.startswith(";"):
                in_multiline = not in_multiline
                continue
            if in_multiline:
                continue

            # ── Skip blank lines and comments ─────────────────────────────────
            if not line or line.startswith("#"):
                continue

            # ── Cell parameters ───────────────────────────────────────────────
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

            # ── New loop_ block → reset all loop state ────────────────────────
            elif line == "loop_":
                in_atom_loop  = False
                loop_has_frac = False
                atom_keys     = []

            # ── Collect _atom_site_ keys ──────────────────────────────────────
            elif line.startswith("_atom_site_"):
                atom_keys.append(line)
                if line in FRAC_KEYS:
                    loop_has_frac = True   # this loop contains coordinate data
                in_atom_loop = True

            # ── Parse data rows ───────────────────────────────────────────────
            elif in_atom_loop and line and not line.startswith("_"):

                # A new loop_ appearing here closes the current one
                if line.startswith("loop_"):
                    in_atom_loop  = False
                    loop_has_frac = False
                    atom_keys     = []
                    continue

                # Skip loops that are NOT coordinate loops (e.g. aniso, symop)
                if not loop_has_frac:
                    continue

                tokens = line.split()
                if len(tokens) < len(atom_keys):
                    continue   # malformed row

                row    = dict(zip(atom_keys, tokens))
                symbol = _extract_element(
                    row.get("_atom_site_type_symbol",
                            row.get("_atom_site_label", "X"))
                )

                # Skip fallback/unknown symbols
                if symbol in ("X", ""):
                    continue

                fx = _parse_cif_number(row.get("_atom_site_fract_x", "0"))
                fy = _parse_cif_number(row.get("_atom_site_fract_y", "0"))
                fz = _parse_cif_number(row.get("_atom_site_fract_z", "0"))
                atoms.append((symbol, fx, fy, fz))

    # ── Post-parse validation ─────────────────────────────────────────────────
    if not atoms:
        raise ValueError(
            f"No atoms parsed from {cif_path}\n"
            f"Check that the CIF file contains _atom_site_fract_x/y/z fields."
        )

    # Check for duplicate fractional coordinates (likely double-counting)
    for i, (s1, fx1, fy1, fz1) in enumerate(atoms):
        for j, (s2, fx2, fy2, fz2) in enumerate(atoms):
            if i >= j:
                continue
            delta = abs(fx1 - fx2) + abs(fy1 - fy2) + abs(fz1 - fz2)
            if delta < 1e-4:
                raise ValueError(
                    f"Atoms {i}({s1}) and {j}({s2}) have identical fractional "
                    f"coordinates ({fx1:.4f},{fy1:.4f},{fz1:.4f}) — "
                    f"likely a CIF parsing error or duplicate site."
                )

    # ── Fractional → Cartesian ────────────────────────────────────────────────
    frac_to_cart = _build_cell_matrix(
        cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma
    )

    geometry = []
    for symbol, fx, fy, fz in atoms:
        cart = frac_to_cart @ np.array([fx, fy, fz])
        geometry.append((symbol, tuple(cart)))

    # Check for atoms unreasonably close in Cartesian space (< 0.5 Å)
    for i, (s1, c1) in enumerate(geometry):
        for j, (s2, c2) in enumerate(geometry):
            if i >= j:
                continue
            dist = np.linalg.norm(np.array(c1) - np.array(c2))
            if dist < 0.5:
                raise ValueError(
                    f"Atoms {i}({s1}) and {j}({s2}) are only {dist:.3f} Å apart.\n"
                    f"This is likely a CIF parsing error (minimum physical bond ~0.7 Å)."
                )

    return geometry


def _parse_cif_number(s):
    """Parse a CIF numeric field, stripping uncertainty in parentheses.
    e.g. '0.250(5)' → 0.250,  '90' → 90.0
    """
    s = s.split("(")[0]
    return float(s)


def _extract_element(s):
    """Extract element symbol from CIF label like 'Cu2+', 'Cl1', 'O1'.
    Returns only leading alphabetic characters, properly capitalised.
    """
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
    """Convert cell parameters (Å, degrees) to fractional→Cartesian matrix.
    Uses standard crystallographic convention (a along x).
    """
    alpha_r = np.radians(alpha)
    beta_r  = np.radians(beta)
    gamma_r = np.radians(gamma)

    cos_a, cos_b, cos_g = np.cos(alpha_r), np.cos(beta_r), np.cos(gamma_r)
    sin_g = np.sin(gamma_r)

    ax = a
    bx = b * cos_g
    by = b * sin_g
    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz = np.sqrt(max(0.0, c**2 - cx**2 - cy**2))

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

# --- Spin contamination thresholds ---
# For NON-SINGLETS: ratio  ⟨S²⟩_UHF / S(S+1)  — should be ~1.0 for pure state
# Values > threshold indicate contamination → Tier 2
SPIN_CONTAMINATION_TIER2_THRESHOLD   = 1.3

# For SINGLETS: absolute ⟨S²⟩_UHF  — should be exactly 0.0 for pure singlet
# Using ratio is undefined (0/0), so we use absolute deviation instead
# Values > threshold indicate contamination → Tier 2
SPIN_CONTAMINATION_SINGLET_THRESHOLD = 0.05

# HOMO-LUMO gap threshold (eV): gap below this → Tier 2
# Applied to the MINIMUM gap across both alpha and beta spin channels
HOMO_LUMO_TIER2_THRESHOLD_EV = 1.0

# NIST 2018 CODATA — used for unit conversion, never hardcode 27.2114 inline
HARTREE_TO_EV = 27.211386245988

# ═══════════════════════════════════════════════════════════════════════════════
# Active Space Finder (ASF)
# ═══════════════════════════════════════════════════════════════════════════════
# entropy_threshold: orbitals with entanglement entropy below this are excluded
# Tier 3 (metals) uses a lower threshold to capture weakly correlated d/f orbitals
ASF_PARAMS = {
    1: {"entropy_threshold": 0.05,  "max_norb": 12, "min_norb": 2},
    2: {"entropy_threshold": 0.02,  "max_norb": 14, "min_norb": 2},
    3: {"entropy_threshold": 0.005, "max_norb": 16, "min_norb": 4},
}

GAP_MIN_NORB = 2
GAP_MAX_NORB = 16

# Core orbital threshold: MOs with occupation > this AND outside active space
# are treated as frozen core. Set conservatively (standard CASSCF uses ~1.98).
# Lowering this risks misclassifying correlated orbitals as core.
CORE_OCC_THRESHOLD = 1.95

# ── DMET Embedding ────────────────────────────────────────────────────────────
# BATH_TOLERANCE: singular values BELOW this are treated as numerical noise.
# For minimal basis sets (sto-3g) legitimate bath SVs can be as small as ~1e-8.
# Setting this too high (e.g. 1e-6) filters out real bath orbitals.
# Rule: set at least 2 orders of magnitude below the smallest expected real SV.
BATH_TOLERANCE = 1e-8

# MIN_BATH_ORBS: minimum bath orbitals to require.
# If fewer are found, emit a warning but do NOT crash — proceed with available.
# Set to 0 to always allow pure-impurity embedding (no bath).
MIN_BATH_ORBS = 1

MAX_EMBED_ORBS = 24

# ═══════════════════════════════════════════════════════════════════════════════
# Quantum Solver
# ═══════════════════════════════════════════════════════════════════════════════
QUANTUM_SOLVER = "sqd"   # "sqd" | "skqd" | "sqdrift"
BACKEND        = "ibm"    # "local" | "mps" | "ibm"

N_SHOTS     = 4096
ANSATZ_REPS = 4
SQD_ITERS   = 5

SKQD_KRYLOV_DIM   = 5
SKQD_DT           = 1.5
SKQD_TROTTER_REPS = 1
SKQD_SHOTS        = 2048

SQDRIFT_NUM_CIRCUITS = 50
SQDRIFT_NUM_GROUPS   = 100
SQDRIFT_TIME         = 2.0
SQDRIFT_ITERS        = 10
SQDRIFT_SHOTS        = 2048

# MPS bond dimension scales with system: min(2^n_emb, cap)
# For n_emb ≤ 8  → exact at 256; for larger systems set higher or use cap
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