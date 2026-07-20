# config.py — test7
"""
Molecule-first configuration.

CHANGE FROM test5: crystal/mineral CIFs (TiO2, NiO, VO2, CuCl2, CuFeS2, ...)
are intentionally NOT auto-accepted anymore. A bulk solid has no natural
finite cutoff -- pulling its asymmetric unit (or even a symmetry-expanded
cell) out of the CIF and calling it "a molecule" produces a fragment with
dangling bonds and no real chemistry (that was test5's CIF bug: it silently
did exactly this). There is no automatic fix for that -- a periodic solid
genuinely isn't a molecule -- so this file instead:

  1. Makes explicit, hand-specified finite geometries (`geometries` dict,
     same pattern as FeN6) the primary and recommended path.
  2. Still allows loading a CIF, but only if it looks like a genuine
     discrete-molecule crystal structure (organic-capped compound with a
     real CCDC/PubChem entry) -- `assert_not_periodic_solid()` below is a
     heuristic tripwire that rejects anything that looks like a bare
     metal-oxide/halide extended solid instead of silently accepting it
     the way test5 did.

If you actually need TiO2/NiO/etc.-like local-site chemistry: that is a
periodic-embedding or hand-built-capped-cluster modeling problem, which is
a domain decision a chemist has to make explicitly -- it is not something
this loader should ever infer automatically from a mineral CIF.
"""

import os
import warnings
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
CIF_DIR     = os.path.join(PROJECT_DIR, "cif_files")
STEP0_FILE  = os.path.join(RESULTS_DIR, "step0_classical.pkl")
STEP1_FILE  = os.path.join(RESULTS_DIR, "step1_asf.pkl")
STEP2_FILE  = os.path.join(RESULTS_DIR, "step2_hamiltonian.pkl")
STEP3_FILE  = os.path.join(RESULTS_DIR, "step3_results.pkl")

BLOCKEXE_WRAPPER = os.path.expanduser("~/block2main_wrapper.sh")

# ═══════════════════════════════════════════════════════════════════════
# Physical constants
# ═══════════════════════════════════════════════════════════════════════
HARTREE_TO_EV       = 27.211386245988   # NIST 2018 CODATA
HARTREE_TO_KCAL_MOL = 627.5094740631

TM_ELEMENTS = {
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu',
}

SPIN_CONTAMINATION_TIER2_THRESHOLD   = 1.3
SPIN_CONTAMINATION_SINGLET_THRESHOLD = 0.05
HOMO_LUMO_TIER2_THRESHOLD_EV         = 1.0

# ═══════════════════════════════════════════════════════════════════════
# Molecule selection — explicit finite geometries (recommended path)
# ═══════════════════════════════════════════════════════════════════════
MOLECULE = "FeN6"
CHARGE   = 0
SPIN     = 4          # 2S; set per molecule (FeN6 model: high-spin Fe(II))
BASIS    = "sto-3g"

# Each entry: list of (symbol, (x, y, z)) in Angstrom -- a genuine,
# finite, chemically well-defined molecule. Add your own here.
geometries = {
    "LiH": [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.5949))],
    "H2O": [("O", (0.0, 0.0, 0.1173)),
            ("H", (0.0, 0.7572, -0.4692)),
            ("H", (0.0, -0.7572, -0.4692))],
    "N2":  [("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.0977))],
    # Hand-built octahedral model complex -- a real finite molecule.
    # Replace bond lengths / ligands with a literature-vetted geometry
    # for production use; this is a placeholder template.
    "FeN6": [
        ("Fe", (0.0, 0.0, 0.0)),
        ("N", (2.10, 0.0, 0.0)), ("N", (-2.10, 0.0, 0.0)),
        ("N", (0.0, 2.10, 0.0)), ("N", (0.0, -2.10, 0.0)),
        ("N", (0.0, 0.0, 2.10)), ("N", (0.0, 0.0, -2.10)),
    ],
}

# ═══════════════════════════════════════════════════════════════════════
# CIF loading — finite molecules only
# ═══════════════════════════════════════════════════════════════════════
FRAC_KEYS = {"_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"}


def assert_not_periodic_solid(cif_path, atoms, has_symmetry_block, z_units):
    """
    Heuristic tripwire against silently treating a bulk crystal (mineral /
    extended inorganic solid) as a molecule -- this is exactly the bug
    test5 had: it parsed only the CIF's asymmetric-unit atom loop (e.g. 3
    atoms for TiO2's Z=8 rutile/brookite cell) and fed that straight to
    PySCF's gto.M() as if it were a complete, finite molecule.

    This is NOT a rigorous classifier. It catches the common, obvious
    failure mode: a bare metal-oxide/halide composition (no organic
    capping atoms) with a 3D space group and Z > 1 formula units -- the
    signature of an extended solid, not a discrete compound. A genuine
    molecular-crystal CIF (an actual compound with ligands) will almost
    always include C/H/N/P/S atoms from those ligands.
    """
    symbols = {a[0] for a in atoms}
    organic_markers = {"C", "H", "N", "P", "S"} & symbols
    if has_symmetry_block and z_units and z_units >= 2 and not organic_markers:
        raise ValueError(
            f"'{cif_path}' looks like a periodic solid / mineral, not a "
            f"finite molecule (3D space group present, Z={z_units}, no "
            f"organic capping atoms found -- only {sorted(symbols)}).\n\n"
            f"A bulk solid has no natural finite cutoff: its asymmetric "
            f"unit is NOT a molecule, and expanding it by symmetry just "
            f"gives you a piece of an infinite lattice with dangling "
            f"bonds, not a chemically meaningful finite system.\n\n"
            f"Options:\n"
            f"  1. Add a real, explicit finite geometry to `geometries` "
            f"in config.py (see FeN6 for the pattern).\n"
            f"  2. Point MOLECULE at a genuine discrete-molecule CIF (an "
            f"actual compound with a CCDC/PubChem entry), not a mineral "
            f"CIF from a crystallography database.\n"
            f"  3. If you specifically need periodic-solid physics, this "
            f"pipeline (gto.M molecular PySCF) is the wrong tool -- that "
            f"requires periodic embedding methods, not this codepath."
        )


def load_geometry_from_cif(molecule_name):
    """
    Parse a genuine finite-molecule CIF into Cartesian atom coordinates.
    Raises via assert_not_periodic_solid() if the file looks like a bulk
    mineral/solid instead of a molecule.
    """
    cif_path = os.path.join(CIF_DIR, f"{molecule_name}.cif")
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"CIF file not found: {cif_path}")

    cell_a = cell_b = cell_c = 1.0
    cell_alpha = cell_beta = cell_gamma = 90.0
    z_units = None
    has_symmetry_block = False
    atoms = []

    in_atom_loop, loop_has_frac, atom_keys, in_multiline = False, False, [], False

    with open(cif_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(";"):
                in_multiline = not in_multiline
                continue
            if in_multiline or not line or line.startswith("#"):
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
            elif line.startswith("_cell_formula_units_Z"):
                try:
                    z_units = int(float(line.split()[-1]))
                except ValueError:
                    pass
            elif line.startswith("_symmetry_space_group_name") or \
                 line.startswith("_space_group_name") or \
                 line.startswith("_space_group_symop"):
                has_symmetry_block = True
            elif line == "loop_":
                in_atom_loop, loop_has_frac, atom_keys = False, False, []
            elif line.startswith("_atom_site_"):
                atom_keys.append(line)
                if line in FRAC_KEYS:
                    loop_has_frac = True
                in_atom_loop = True
            elif in_atom_loop and not line.startswith("_") and not line.startswith("loop_"):
                if not loop_has_frac:
                    continue
                tokens = line.split()
                if len(tokens) < len(atom_keys):
                    continue
                row = dict(zip(atom_keys, tokens))
                symbol = _extract_element(
                    row.get("_atom_site_type_symbol", row.get("_atom_site_label", "X"))
                )
                if symbol in ("X", ""):
                    continue
                fx = _parse_cif_number(row.get("_atom_site_fract_x", "0"))
                fy = _parse_cif_number(row.get("_atom_site_fract_y", "0"))
                fz = _parse_cif_number(row.get("_atom_site_fract_z", "0"))
                atoms.append((symbol, fx, fy, fz))

    if not atoms:
        raise ValueError(f"No atoms parsed from {cif_path}")

    # Tripwire BEFORE doing anything else with the geometry.
    assert_not_periodic_solid(cif_path, atoms, has_symmetry_block, z_units)

    frac_to_cart = _build_cell_matrix(cell_a, cell_b, cell_c,
                                       cell_alpha, cell_beta, cell_gamma)
    geometry = []
    for symbol, fx, fy, fz in atoms:
        cart = frac_to_cart @ np.array([fx, fy, fz])
        geometry.append((symbol, tuple(cart)))
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
    return (elem[0].upper() + elem[1:].lower()) if len(elem) > 1 else elem.upper()


def _build_cell_matrix(a, b, c, alpha, beta, gamma):
    alpha_r, beta_r, gamma_r = np.radians([alpha, beta, gamma])
    cos_a, cos_b, cos_g = np.cos(alpha_r), np.cos(beta_r), np.cos(gamma_r)
    sin_g = np.sin(gamma_r)
    ax = a
    bx, by = b * cos_g, b * sin_g
    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz = np.sqrt(max(0.0, c ** 2 - cx ** 2 - cy ** 2))
    return np.array([[ax, bx, cx], [0., by, cy], [0., 0., cz]])


def load_geometry(molecule_name):
    """Primary entry point: explicit geometry first, CIF as a fallback."""
    if molecule_name in geometries:
        return geometries[molecule_name]
    warnings.warn(
        f"'{molecule_name}' not found in config.geometries -- falling back "
        f"to CIF lookup in {CIF_DIR}. Prefer adding an explicit geometry.",
        RuntimeWarning,
    )
    return load_geometry_from_cif(molecule_name)


# ═══════════════════════════════════════════════════════════════════════
# Tier classification / ASF / embedding parameters
# ═══════════════════════════════════════════════════════════════════════
ASF_PARAMS = {
    1: {"entropy_threshold": 0.05,  "max_norb": 12, "min_norb": 2},
    2: {"entropy_threshold": 0.02,  "max_norb": 14, "min_norb": 2},
    3: {"entropy_threshold": 0.005, "max_norb": 16, "min_norb": 4},
}
GAP_MIN_NORB       = 2
GAP_MAX_NORB       = 16
CORE_OCC_THRESHOLD = 1.95

BATH_TOLERANCE = 1e-8
MIN_BATH_ORBS  = 0
MAX_EMBED_ORBS = 18

# ── NEW: reference density for the Schmidt decomposition ────────────────
# "mp2"   -- reuse Step 1's MP2 1-RDM (fast, but unreliable exactly where
#            static correlation is strong -- the systems this pipeline
#            targets). No recomputation: Step 2 reads Step 1's saved DM.
# "casci" -- run a plain CASCI (no orbital optimization) inside the
#            ASF-selected active space to get a correlated reference DM.
#            Bounded by the SAME active-space size (~10-16 orbitals) the
#            pipeline already treats as tractable elsewhere -- this does
#            not solve the impurity+bath problem, only refines what goes
#            into building the bath. Recommended default.
DMET_REFERENCE = "casci"   # "mp2" | "casci"

# ── NEW: one-shot grand-canonical chemical-potential correction ─────────
MU_CORRECTION       = True
MU_SEARCH_RANGE     = (-5.0, 5.0)   # Ha
MU_MAX_ITER         = 60
MU_TOL              = 1e-10

# ── NEW: embedding self-consistency diagnostic threshold ────────────────
# mismatch_score above this flags that the bath was built from a density
# that doesn't resemble what the correlated/quantum solver actually found
# -- a signal the one-shot approximation may be breaking down.
CONSISTENCY_MISMATCH_THRESHOLD = 0.10

# ═══════════════════════════════════════════════════════════════════════
# Quantum solver / GQE-for-QSCI
# ═══════════════════════════════════════════════════════════════════════
QUANTUM_SOLVER   = "gqe_qsci"   # driven separately via gqe_for_qsci.py
FERMION_TO_QUBIT = "jw"

# ═══════════════════════════════════════════════════════════════════════
# Classical reference methods (step0_classical.py)
# ═══════════════════════════════════════════════════════════════════════
CLASSICAL_METHODS = ["HF", "MP2", "CCSD", "CASSCF"]

# ═══════════════════════════════════════════════════════════════════════
# Resolve geometry at import time
# ═══════════════════════════════════════════════════════════════════════
GEOMETRY  = load_geometry(MOLECULE)
ATOM_SYMS = [a[0] for a in GEOMETRY]
N_ATOMS   = len(GEOMETRY)