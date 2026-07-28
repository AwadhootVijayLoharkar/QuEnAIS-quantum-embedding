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


def cached_result_is_current(path, verbose=True):
    """
    True only if `path` exists AND the pickle in it was generated for the
    CURRENT config.MOLECULE / config.BASIS.

    Why this exists: every step's cache file lives at a fixed path shared
    across molecules (step0_classical.pkl, step1_asf.pkl,
    step2_hamiltonian.pkl). Change MOLECULE from LiH to ScH and every
    script's `if os.path.exists(...)` cache check happily reuses LiH's
    results -- silently, with no error. That's the exact failure mode that
    already cost real debugging time three separate times in this project
    (a stale test5 step2 pickle wired into the external repo's
    dmet_embedding.yaml; two file-content mixups). A wrong-but-plausible
    number is far more expensive than a recomputation.

    Paths stay fixed on purpose -- the external gqe-for-qsci repo's
    configs/molecule/dmet_embedding.yaml hardcodes the step2 path, so
    making these molecule-specific would silently break THAT instead.
    Validating content is the fix that doesn't trade one stale-path bug
    for another.
    """
    import pickle as _pickle
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as fh:
            data = _pickle.load(fh)
    except Exception as exc:
        if verbose:
            print(f"  [cache] {os.path.basename(path)} unreadable ({exc}); recomputing.")
        return False

    # step0 stores molecule/basis at top level; step1/step2 nest it in mol_info.
    info = data.get("mol_info", data) if isinstance(data, dict) else {}
    cached_mol   = info.get("molecule")
    cached_basis = info.get("basis")

    if cached_mol is None:
        if verbose:
            print(f"  [cache] {os.path.basename(path)} has no molecule tag "
                  f"(pre-dates this check); recomputing to be safe.")
        return False

    if cached_mol != MOLECULE or (cached_basis is not None and cached_basis != BASIS):
        if verbose:
            print(f"  [cache] {os.path.basename(path)} was built for "
                  f"{cached_mol}/{cached_basis}, but config says "
                  f"{MOLECULE}/{BASIS} -- ignoring stale cache and recomputing.")
        return False
    return True

# EDIT to wherever you cloned the external gqe_qsci / GQE-for-QSCI repo
# (the one with GQE_README.md, train.py, activate_custom_mpi.sh). Can also
# be set via the GQE_QSCI_REPO_PATH environment variable instead of
# editing this file.
GQE_QSCI_REPO_PATH = os.environ.get(
    "GQE_QSCI_REPO_PATH",
    "/home/loharkar/QuEnAIS-quantum-embedding/gqe-for-qsci",
)
GQE_TRAIN_ENTRYPOINT = "train.py"     # relative to GQE_QSCI_REPO_PATH
# GQE_TRAIN_ARGS is built from the named GQE_* fields further down this
# file (see build_gqe_hydra_overrides()) -- assigned at the bottom of this
# file, after those fields exist.

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
# FIRST TRANSITION-METAL RUNG: ScH.
# Chosen deliberately as the smallest real step up from LiH toward the
# TM-complex target, changing as few things at once as possible:
#   - X(1)Sigma+ ground state => SPIN=0, so n_alpha == n_beta. That keeps
#     the pipeline inside the ONLY regime it has ever been validated in.
#     (Open-shell TM systems like TiO (X(3)Delta) or ScO (X(2)Sigma+)
#     trigger the n_alpha != n_beta warning branch in
#     dmet_lib.chemical_potential_correction(), which has never been
#     exercised -- don't stack that new failure mode on top of a new
#     molecule. Save those for rung 2+, after ScH validates clean.)
#   - Sc is the lightest 3d transition metal (21 electrons), so CASSCF/
#     NEVPT2 stays cheap enough to remain a trustworthy answer key --
#     which is exactly what caught every bug during the N2/LiH work.
#   - Still has genuine d-orbital character, unlike LiH. This is a real
#     test of ASF's active-space selection, not a rerun of a solved case.
# Note ScH has a low-lying (3)Delta state; if UHF converges to something
# with unexpected spin contamination, that's a real physical near-
# degeneracy, not necessarily a code bug.
# Back to ScH. The LiH control run PASSED: DMET+GQE reached -7.880890 vs
# DMET+CASCI -7.881246 (0.356 mHa error, inside chemical accuracy),
# recovering 18.9 of 19.2 mHa of correlation. That proves the
# DMET -> GQE handoff (molecule=dmet_embedding + dmet_pauli_evolution)
# works correctly, so ScH's stall at HF is a sampling-CAPACITY problem,
# not a wiring bug -- see the enlarged GQE sampling settings below.
# TO RE-RUN THE LiH CONTROL: set MOLECULE = "LiH" and
# FORCE_ACTIVE_SPACE = None.
MOLECULE = "ScH"
CHARGE   = 0
SPIN     = 0        # X(1)Sigma+ ground state (singlet)
BASIS    = "sto-3g"

geometries = {
    # X(1)Sigma+ ground state; r_e ~ 1.78 Ang (theory puts it close to
    # 3.4 a0 = 1.799 Ang). Verify against your own preferred reference
    # before quoting any absolute energy from this geometry.
    "ScH": [("Sc", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.7800))],
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
# ScH (4e, 6o): set deliberately, because BOTH the automatic paths gave a
# demonstrably inadequate active space on this first TM system --
# confirmed against classical references, which is exactly why we
# established them first:
#
#   CASSCF(2e,3o) recovered only -22.79 kcal/mol vs HF, while plain CCSD
#   recovered -44.01. Even CASSCF+NEVPT2 (-752.69944) landed 6.6 mHa
#   ABOVE CCSD(T) (-752.70989). A well-chosen active space should put
#   CASSCF+NEVPT2 at or below CCSD(T), so that reference could not be
#   trusted as an answer key.
#
# Why the automatic paths failed here:
#   1. ASF itself selected 4 orbitals [10,11,12,13] (entropies 0.316,
#      0.166, 0.166, 0.094) -- but only 2 ACTIVE ELECTRONS, treating MO 9
#      (S=0.055) as inert core. For a 3d transition metal that leaves
#      most of the interesting correlation outside the active space.
#   2. Phase C's gap detection then RE-CUT ASF's 4 orbitals down to 3,
#      dropping MO 13 -- second-guessing ASF's entanglement-entropy
#      selection using a cruder MP2 occupation-deviation metric. See the
#      warning now emitted in ASF.py Phase C.
#
# This space instead takes every orbital in ASF's own window with entropy
# >= 0.055, which is a clean break (MO 14: S=0.072 vs MO 15: S=0.023, and
# MOs 6-8: S=0.009). MOs 9 and 10 are the occupied valence pair, giving
# 4 active electrons -- matching ScH's actual valence count (Sc 3d(1)4s(2)
# + H 1s(1)); MOs 0-8 are genuine Sc core (1s2s2p3s3p). MOs 11 and 12 are
# a degenerate pair (identical S=0.166) and are kept together, which is
# the same symmetry-preservation concern that motivated GAP_DEGENERACY_TOL.
#
# Set back to None once ASF's TM behavior is fixed, and compare.
# Set to None when running the LiH control (these are ScH MO indices and
# are out of range for LiH's 6 AOs; LiH validates fine on ASF's own
# automatic (2e, 2o) selection).
FORCE_ACTIVE_SPACE = [9, 10, 11, 12, 13, 14]   # ScH: (4e, 6o)

# "mp2" -- reuse Step 1's MP2 1-RDM (fast, unreliable exactly where static
#          correlation is strong).
# "casci" -- plain CASCI (no orbital optimization) inside the ASF active
#            space; recommended default. See dmet_lib.get_reference_density.
DMET_REFERENCE = "casci"   # "mp2" | "casci"

# One-shot grand-canonical chemical-potential correction.
# Restored to True: the diagnostic with it off proved mu is mathematically
# INERT for CASCI-based total energies (a fixed-particle-number solver
# makes the h1e-mu*I / ecore+mu*N shift cancel exactly), confirmed by the
# embedding CASCI energy coming out bit-for-bit identical with mu on vs
# off. mu was never the cause of the large mismatch/unphysical energies --
# the real bug is being tracked down elsewhere (Phase C/D/E of DMET.py).
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

# ── GQE-for-QSCI training hyperparameters ───────────────────────────────
# These map onto keys in the external gqe-for-qsci repo's own Hydra
# configs (configs/default.yaml + configs/trainer/default.yaml). Edit the
# values here instead of hand-editing those yaml files -- run_gqe_training.py
# forwards whichever ones you set (anything left as None is skipped, so
# the external repo's own yaml default applies unchanged). This only
# controls the external repo's config; it does NOT touch anything in
# ASF.py/DMET.py/gqe_for_qsci.py.
#
# The GQE_* names below are deliberately flat for readability; the
# trainer-group ones get their required "trainer." Hydra prefix added
# automatically in build_gqe_hydra_overrides() -- see the prefix note
# there. Do NOT pass bare `load_checkpoint=false` on the command line;
# Hydra rejects it with "Key 'load_checkpoint' is not in struct".
# WHICH MOLECULE CONFIG train.py LOADS. This is a Hydra config-GROUP
# selection (configs/molecule/<name>.yaml), not a scalar value.
#
# CRITICAL -- this was silently wrong for every run before it was added.
# configs/default.yaml declares:
#     defaults:
#       - molecule: n2
# so unless something overrides it, train.py loads configs/molecule/n2.yaml
# and NEVER reads dmet_embedding.yaml (the one pointing at
# results/step2_hamiltonian.pkl). Proof it was happening: the LiH and ScH
# runs produced near-identical epoch logs (cx_count=52, total_gates=166,
# num_sampled_basis=25, energies matching to 5 decimals) with absolute
# energies around -107 Ha -- N2's energy scale, not LiH's (-7.9) or ScH's
# (-752). GQE was training on N2 the entire time.
#
# That also silently corrupted the reported "DMET+GQE" energy:
# visualization.py builds it as (true embedding CASCI) + (GQE's
# best_so_far error vs its OWN R-CASCI reference). With the wrong molecule
# loaded, that adds N2's convergence error to your molecule's CASCI --
# producing a plausible-looking number with no physical meaning.
GQE_MOLECULE_CONFIG          = "dmet_embedding"   # configs/molecule/<this>.yaml

# SAMPLING CAPACITY -- enlarged for ScH (22 qubits). The repo's defaults
# (num_samples=10, max_iters=50) were sized for a small system and are far
# too small here. Evidence, comparing the LiH control run to ScH:
#
#                        LiH            ScH
#   embedding orbitals   4  (8 qubits)  11 (22 qubits)
#   determinant space    C(4,2)^2 = 36  C(11,4)^2 = 108,900
#   num_sampled_basis    17             15
#   final subspace_dim   34             309
#   result               0.36 mHa err   stalled at HF (60 mHa err)
#
# The Hilbert space grew ~3000x while the sampling volume stayed flat, so
# ScH explored ~0.3% of its determinant space and the optimizer fell back
# to the only state it could reliably reach (HF). LiH's 34-configuration
# subspace covers most of its 36 determinants, which is why it nearly
# nails CASCI. Scaling the search to match the space:
GQE_SEED                     = None   # int, e.g. 32
GQE_MAX_ITERS                = 120    # 200 was wasteful: the ngates=20 run
                                       # stopped improving at epoch ~100 and
                                       # the last 100 epochs changed nothing.
                                       # Raise again only if the deeper
                                       # circuits are still improving at the end.
GQE_NUM_SAMPLES              = 100    # was 10 -- circuits sampled per iteration
GQE_BATCH_SIZE               = 100    # keep == GQE_NUM_SAMPLES (online training)
GQE_STEP_PER_EPOCH           = None   # int -- policy updates per iteration
GQE_WARMUP_SIZE              = 100    # keep in step with num_samples
GQE_BUFFER_SIZE              = 100    # keep == GQE_NUM_SAMPLES (online training)
GQE_LOAD_CHECKPOINT          = False  # bool -- True silently resumes from
                                       # whatever checkpoint dir the repo finds
                                       # under gqe-for-qsci/outputs/, which
                                       # isn't scoped per-molecule (this is
                                       # what caused the frozen ~-107 Ha
                                       # ensemble-energy numbers during LiH
                                       # debugging). Keep False unless you
                                       # specifically want to resume a run.
GQE_CHECKPOINT_EVERY_N_ITERS = None   # int

GQE_OPTIMIZER_LR             = None   # float, e.g. 5e-6
GQE_OPTIMIZER_CLS            = None   # str, e.g. "AdamW"
GQE_OPTIMIZER_WEIGHT_DECAY   = None   # float

GQE_LOSS_TYPE                = None   # str, e.g. "grpo"
GQE_LOSS_CLIP_GRPO_LOW       = None   # float
GQE_LOSS_CLIP_GRPO_HIGH      = None   # float

GQE_TEMP_SCHED_INITIAL       = None   # float
GQE_TEMP_SCHED_DELTA         = None   # float
GQE_TEMP_SCHED_TARGET_VAR    = None   # float

# Circuit depth -- now the binding constraint. At ngates=20 the ScH run
# broke the HF stall (error 0.0604 -> 0.0368, e/min -752.639 -> -752.660,
# 39% of correlation recovered) but then FROZE from epoch ~100 onward:
# Global-refined subspace_dim stuck at exactly 1116 and the error at
# exactly 0.036796 for the final 100 epochs.
#
# That plateau is a reachability limit, not an optimization failure --
# e/mean (-752.55) still sat well above e/min (-752.66), so the policy
# was still sampling varied circuits; they simply couldn't reach any new
# determinants. More epochs or more samples cannot fix that; only deeper
# circuits can. Doubling to 40.
GQE_NGATES                   = 40     # 10 = repo default, 20 = plateaued
GQE_REFERENCE_KEYS           = None   # list[str], e.g. ["R-CASCI", "R-CCSD"]

GQE_SAMPLER_MPI              = None   # bool
GQE_SAMPLER_SHOTS            = None   # int -- shots per circuit; only matters
                                       # once GQE_CUDAQ_TARGET below is no
                                       # longer an exact-statevector backend

# MUST be one of the DMET-aware pools when feeding a DMET embedding.
# The repo's factory.py registers four specs:
#     "pauli_evolution", "excitation"            -- stock, geometry-based
#     "dmet_pauli_evolution", "dmet_excitation"  -- DMET-aware
# The stock pools rebuild the molecule FROM ITS GEOMETRY to derive CCSD
# amplitudes (gqe_qsci/gqe/operator_pool.py:97 iterates
# self.molecule.geometry). A DMET embedding has no geometry -- it is just
# h1e/h2e/ecore in an abstract orbital basis -- so geometry is None and
# that line dies with "TypeError: 'NoneType' object is not iterable".
# The DMET pools instead take CCSD amplitudes from the embedding's own
# CCSD solve (molecule.ccsd_amplitude), which is the whole reason
# DMETUCCSDBasedPool exists. "dmet_pauli_evolution" is the direct
# analogue of the repo default "pauli_evolution" and accepts the same
# remove_z_ladder / only_use_first_pauli options below.
GQE_OPERATOR_POOL_SPEC             = "dmet_pauli_evolution"
GQE_OPERATOR_POOL_CCSD_THRESHOLD   = None  # float
# Set False (repo default is True) because the ScH run showed direct
# evidence of broken particle-number conservation: the epoch log reported
# 12 sampled basis states but only 8 symmetry-preserving -- a third of
# every sample discarded. The Jordan-Wigner Z-ladder encodes fermionic
# anticommutation; removing it makes exp(i*theta*P) no longer a
# particle-number-conserving excitation, so sampled states leak into the
# wrong electron-number sector. Consistent with GQE stalling at exactly
# the HF energy (-752.63874 vs HF -752.63870) while recovering none of
# the 60.8 mHa of correlation that the embedding's own CCSD finds.
#
# Keeping the ladder makes each circuit deeper (more gates per operator),
# so expect slower epochs. If this fixes the symmetry ratio but
# convergence is still poor, the next lever is
# GQE_OPERATOR_POOL_ONLY_FIRST_PAULI below.
GQE_OPERATOR_POOL_REMOVE_Z_LADDER  = False
GQE_OPERATOR_POOL_ONLY_FIRST_PAULI = None  # bool

GQE_QSCI_MAX_DIM              = None  # int -- max QSCI subspace dimension
GQE_QSCI_ENLARGE_METHOD       = None  # str, e.g. "symmetry_completion"
GQE_QSCI_MAX_CYCLE            = None  # int

# ── CUDA-Q backend selection ────────────────────────────────────────────
# "qpp-cpu"       -- local CPU statevector simulator (current behavior)
# "nvidia"        -- local GPU-accelerated statevector simulator
# "tensornet"     -- exact tensor-network contraction (still classical,
#                    different scaling than statevector)
# "tensornet-mps" -- bond-dimension-truncated MPS simulator (approximate,
#                    scales further at the cost of exactness)
# "quantinuum" / "ionq" / etc. -- real QPU hardware. NOT a drop-in: needs
#                    provider credentials, and GQE's energy evaluation
#                    likely assumes exact statevector expectation values
#                    today, so hardware also needs shots + error
#                    mitigation added to the energy-eval step, not just a
#                    target swap.
#
# NOTE: grep-ing the external repo's train.py/factory.py for
# set_target()/cudaq_target came back empty -- that code never calls
# cudaq.set_target() itself, so there's no Hydra config key to override
# here. Instead, run_gqe_training.py sets CUDA-Q's own
# CUDAQ_DEFAULT_SIMULATOR environment variable from this field before
# launching train.py -- the same mechanism you already use manually
# (`export CUDAQ_DEFAULT_SIMULATOR=qpp-cpu`) for GPU-architecture
# mismatches. That picks the backend at runtime with zero changes needed
# to the external repo's source.
GQE_CUDAQ_TARGET = "qpp-cpu"


def build_gqe_hydra_overrides():
    """
    Turns the GQE_* fields above into a list of Hydra CLI override strings
    ("key=value"), skipping anything left as None. GQE_CUDAQ_TARGET is NOT
    included here -- it's applied via the CUDAQ_DEFAULT_SIMULATOR env var
    in run_gqe_training.py instead, since the external repo's Hydra config
    has no key for it (confirmed empty grep for set_target/cudaq_target).
    """
    import json as _json

    def _fmt(v):
        if isinstance(v, bool):
            return str(v).lower()
        if isinstance(v, (list, tuple)):
            # Hydra list syntax is [a,b] -- NOT JSON. json.dumps would emit
            # ["a", "b"] (double quotes + spaces), which Hydra's override
            # grammar rejects.
            return "[" + ",".join(str(x) for x in v) + "]"
        if isinstance(v, dict):
            return _json.dumps(v)
        return str(v)

    # KEY PREFIXES MATTER. This repo splits its config across two files,
    # and Hydra addresses each by its config-GROUP path, not by a flat
    # name:
    #   configs/trainer/default.yaml -> needs a "trainer." prefix
    #       (seed, max_iters, num_samples, batch_size, step_per_epoch,
    #        warmup_size, buffer_size, load_checkpoint,
    #        checkpoint_every_n_iters, optimizer.*, loss.*,
    #        temperature_scheduler.*)
    #   configs/default.yaml         -> genuinely top-level, no prefix
    #       (ngates, reference_keys, sampler.*, operator_pool.*, qsci.*)
    # Getting this wrong fails loudly and harmlessly -- Hydra refuses with
    # "Key 'X' is not in struct" rather than silently ignoring the
    # override -- which is the good kind of failure.
    mapping = {
        # Config-GROUP override (selects configs/molecule/<name>.yaml).
        # Written as "molecule=dmet_embedding" -- no "+" prefix, because
        # the group already exists in the defaults list.
        "molecule": GQE_MOLECULE_CONFIG,
        "trainer.seed": GQE_SEED,
        "trainer.max_iters": GQE_MAX_ITERS,
        "trainer.num_samples": GQE_NUM_SAMPLES,
        "trainer.batch_size": GQE_BATCH_SIZE,
        "trainer.step_per_epoch": GQE_STEP_PER_EPOCH,
        "trainer.warmup_size": GQE_WARMUP_SIZE,
        "trainer.buffer_size": GQE_BUFFER_SIZE,
        "trainer.load_checkpoint": GQE_LOAD_CHECKPOINT,
        "trainer.checkpoint_every_n_iters": GQE_CHECKPOINT_EVERY_N_ITERS,
        "trainer.optimizer.lr": GQE_OPTIMIZER_LR,
        "trainer.optimizer.cls": GQE_OPTIMIZER_CLS,
        "trainer.optimizer.weight_decay": GQE_OPTIMIZER_WEIGHT_DECAY,
        "trainer.loss.type": GQE_LOSS_TYPE,
        "trainer.loss.clip_grpo_low": GQE_LOSS_CLIP_GRPO_LOW,
        "trainer.loss.clip_grpo_high": GQE_LOSS_CLIP_GRPO_HIGH,
        "trainer.temperature_scheduler.initial": GQE_TEMP_SCHED_INITIAL,
        "trainer.temperature_scheduler.delta": GQE_TEMP_SCHED_DELTA,
        "trainer.temperature_scheduler.target_var": GQE_TEMP_SCHED_TARGET_VAR,
        "ngates": GQE_NGATES,
        "reference_keys": GQE_REFERENCE_KEYS,
        "sampler.mpi": GQE_SAMPLER_MPI,
        "sampler.shots": GQE_SAMPLER_SHOTS,
        "operator_pool.spec": GQE_OPERATOR_POOL_SPEC,
        "operator_pool.ccsd_threshold": GQE_OPERATOR_POOL_CCSD_THRESHOLD,
        "operator_pool.remove_z_ladder": GQE_OPERATOR_POOL_REMOVE_Z_LADDER,
        "operator_pool.only_use_first_pauli": GQE_OPERATOR_POOL_ONLY_FIRST_PAULI,
        "qsci.max_dim": GQE_QSCI_MAX_DIM,
        "qsci.enlarge_method": GQE_QSCI_ENLARGE_METHOD,
        "qsci.max_cycle": GQE_QSCI_MAX_CYCLE,
    }
    overrides = []
    for key, val in mapping.items():
        if val is None:
            continue
        overrides.append(f"{key}={_fmt(val)}")
    return overrides


GQE_TRAIN_ARGS = build_gqe_hydra_overrides()

# ═══════════════════════════════════════════════════════════════════════
# Classical reference methods (classical_methods.py)
# ═══════════════════════════════════════════════════════════════════════
# NEVPT2 enabled for the TM rung: CASSCF alone is frequently NOT accurate
# enough to serve as ground truth for transition-metal energetics (it has
# static correlation but misses dynamic correlation, which is large for 3d
# systems). Since the whole point of this step is having a trustworthy
# answer key -- the thing that caught every bug during N2/LiH -- the extra
# cost is the point, not overhead. Drop back to the shorter list below if
# NEVPT2 turns out to be prohibitively slow for a bigger system later.
CLASSICAL_METHODS = ["HF", "MP2", "CCSD", "CCSD_T", "CASSCF", "NEVPT2"]
# CLASSICAL_METHODS = ["HF", "MP2", "CCSD", "CASSCF"]  # lighter (LiH-era default)

# ═══════════════════════════════════════════════════════════════════════
# Resolve geometry at import time
# ═══════════════════════════════════════════════════════════════════════
GEOMETRY  = load_geometry(MOLECULE)
ATOM_SYMS = [a[0] for a in GEOMETRY]
N_ATOMS   = len(GEOMETRY)