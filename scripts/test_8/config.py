# config.py — test_8
"""
Same pipeline as test7, reorganized so everything lives in and runs from
one self-contained folder (test_8/) -- no more cd'ing into a separate
gqe-for-qsci checkout to run the GQE side.

CHANGES vs test7:

  1. OPENBLAS_NUM_THREADS / MKL_NUM_THREADS / NUMEXPR_NUM_THREADS are
     forced to "1" here, BEFORE numpy is ever imported anywhere in the
     pipeline. This removes:

        "OpenBLAS Warning : Detect OpenMP Loop and this application may
         hang. Please rebuild the library with USE_OPENMP=1 option."

     Root cause: block2 (the DMRG library ASF.py uses internally) is
     OpenMP-threaded. OpenBLAS is ALSO multi-threaded (via pthreads) by
     default. When OpenBLAS gets called from inside a live OpenMP
     parallel region, the two threading runtimes fight over CPU affinity
     -- that's what the warning describes, and it's a genuine hang risk,
     not just noise (it's the same family of bug behind the FeN6 DMRG
     hang). The fix isn't rebuilding OpenBLAS: it's making sure only ONE
     of the two libraries actually spawns threads. Since block2/pyscf's
     integral code already parallelizes via OpenMP, OpenBLAS is forced
     single-threaded here so it never spawns its own pool inside an
     OpenMP region. OMP_NUM_THREADS is left alone (defaults to
     cpu_count-1) so block2 keeps its own parallelism.

     IMPORTANT: this only works if config.py is imported BEFORE numpy in
     every entry script -- env vars set after numpy's first import don't
     reliably reach OpenBLAS's thread pool (it reads them once, lazily,
     the first time a thread pool is needed). Every script in this folder
     now does `import config` first, `import numpy` after -- test7 had
     this backwards in a couple of files (DMET.py, gqe_for_qsci.py,
     visualization.py all imported numpy before config).

  2. GQE_QSCI_REPO_PATH (+ GQE_TRAIN_ENTRYPOINT / GQE_TRAIN_ARGS): the
     external gqe_qsci package no longer needs to be on PYTHONPATH by
     convention (i.e. by cd-ing into its folder before running Python).
     gqe_for_qsci.py and run_gqe_training.py both use this path directly
     (sys.path bootstrap / subprocess cwd), so everything below runs from
     test_8/ regardless of where that repo actually lives on disk.
     EDIT GQE_QSCI_REPO_PATH to match your machine.
"""

import os

# ── Must run before numpy/scipy/pyscf are imported anywhere ─────────────
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) - 1)))

import warnings
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# Paths -- everything resolved relative to THIS file, i.e. test_8/
# ═══════════════════════════════════════════════════════════════════════
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
CIF_DIR     = os.path.join(PROJECT_DIR, "cif_files")
STEP0_FILE  = os.path.join(RESULTS_DIR, "step0_classical.pkl")
STEP1_FILE  = os.path.join(RESULTS_DIR, "step1_asf.pkl")
STEP2_FILE  = os.path.join(RESULTS_DIR, "step2_hamiltonian.pkl")
STEP3_FILE  = os.path.join(RESULTS_DIR, "step3_results.pkl")

# Point this at the .log file produced by `python train.py ... > this.log`
# (external gqe-for-qsci repo) -- visualization.py parses the
# "[epoch N] {...}" lines out of it. run_gqe_training.py (below) writes to
# this path automatically. Leave alone to skip GQE plots until you run it.
GQE_LOG_FILE = os.path.join(PROJECT_DIR, "gqe_train.log")

BLOCKEXE_WRAPPER = os.path.expanduser("~/block2main_wrapper.sh")

# EDIT to wherever you cloned the external gqe_qsci / GQE-for-QSCI repo
# (the one with GQE_README.md, train.py, activate_custom_mpi.sh). Can also
# be set via the GQE_QSCI_REPO_PATH environment variable instead of
# editing this file.
GQE_QSCI_REPO_PATH = os.environ.get(
    "GQE_QSCI_REPO_PATH", os.path.join(os.path.expanduser("~"), "gqe-for-qsci")
)
GQE_TRAIN_ENTRYPOINT = "train.py"     # relative to GQE_QSCI_REPO_PATH
GQE_TRAIN_ARGS       = []             # extra fixed CLI/Hydra overrides, if any

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
MOLECULE = "N2"
CHARGE   = 0
SPIN     = 0
BASIS    = "sto-3g"

geometries = {
    "LiH": [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.5949))],
    "H2O": [("O", (0.0, 0.0, 0.1173)),
            ("H", (0.0, 0.7572, -0.4692)),
            ("H", (0.0, -0.7572, -0.4692))],
    "N2":  [("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.0977))],
    # Placeholder template -- replace with a literature-vetted geometry
    # and the correct ground-state spin before trusting FeN6 results.
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
    symbols = {a[0] for a in atoms}
    organic_markers = {"C", "H", "N", "P", "S"} & symbols
    if has_symmetry_block and z_units and z_units >= 2 and not organic_markers:
        raise ValueError(
            f"'{cif_path}' looks like a periodic solid / mineral, not a "
            f"finite molecule (3D space group present, Z={z_units}, no "
            f"organic capping atoms found -- only {sorted(symbols)}).\n\n"
            f"Options:\n"
            f"  1. Add a real, explicit finite geometry to `geometries` "
            f"in config.py.\n"
            f"  2. Point MOLECULE at a genuine discrete-molecule CIF.\n"
            f"  3. If you need periodic-solid physics, this pipeline "
            f"(gto.M molecular PySCF) is the wrong tool."
        )


def load_geometry_from_cif(molecule_name):
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

    assert_not_periodic_solid(cif_path, atoms, has_symmetry_block, z_units)

    frac_to_cart = _build_cell_matrix(cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma)
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
GAP_DEGENERACY_TOL = 1e-3   # orbitals within this deviation-value tolerance
                             # of the chosen cutoff are treated as a
                             # degenerate group and kept/dropped together,
                             # instead of split across the cutoff -- fixes
                             # the N2 case where gap detection picked one
                             # of a true degenerate pi-orbital pair but
                             # not the other.
CORE_OCC_THRESHOLD = 1.95

BATH_TOLERANCE = 1e-8
MIN_BATH_ORBS  = 0
MAX_EMBED_ORBS = 18

# Optional escape hatch: set to a list of MO indices (0-based, in the UHF
# alpha-MO basis -- the same basis compute_mp2_deviations already uses)
# to bypass ASF/DMRG entirely and use that exact active space. Useful for
# small, well-studied molecules (like N2) where hand-verifying the active
# space beats trusting automatic selection. Leave as None to use ASF's
# automatic selection (now with the degeneracy fix above).
FORCE_ACTIVE_SPACE = None   # e.g. [3, 4, 5, 6, 7, 8] for a manual N2 space

# "mp2" -- reuse Step 1's MP2 1-RDM (fast, unreliable exactly where static
#          correlation is strong).
# "casci" -- plain CASCI (no orbital optimization) inside the ASF active
#            space; recommended default. See dmet_lib.get_reference_density.
DMET_REFERENCE = "casci"   # "mp2" | "casci"

# One-shot grand-canonical chemical-potential correction.
MU_CORRECTION   = True
MU_SEARCH_RANGE = "auto"   # derives the bracket from h1e_emb's own
                            # eigenvalue spectrum -- see dmet_lib.py
MU_MAX_ITER     = 60
MU_TOL          = 1e-10

CONSISTENCY_MISMATCH_THRESHOLD = 0.10

# ═══════════════════════════════════════════════════════════════════════
# Quantum solver / GQE-for-QSCI
# ═══════════════════════════════════════════════════════════════════════
QUANTUM_SOLVER   = "gqe_qsci"
FERMION_TO_QUBIT = "jw"

# ═══════════════════════════════════════════════════════════════════════
# Classical reference methods (classical_methods.py)
# ═══════════════════════════════════════════════════════════════════════
CLASSICAL_METHODS = ["HF", "MP2", "CCSD", "CASSCF"]
# CLASSICAL_METHODS = ["HF", "MP2", "CCSD", "CCSD_T", "CASSCF", "NEVPT2"]  # full

# ═══════════════════════════════════════════════════════════════════════
# Resolve geometry at import time
# ═══════════════════════════════════════════════════════════════════════
GEOMETRY  = load_geometry(MOLECULE)
ATOM_SYMS = [a[0] for a in GEOMETRY]
N_ATOMS   = len(GEOMETRY)