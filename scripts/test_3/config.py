import os

# ── MKL / block2 ──────────────────────────────────────────────────────────────
BLOCKEXE_WRAPPER = os.path.expanduser("~/block2main_wrapper.sh")

# ── Molecule ──────────────────────────────────────────────────────────────────
MOLECULE = "FeN6"


# ── Protein-Binding Ligand Mode ───────────────────────────────────────────────
# Set True when running graph-transformer-generated drug-like ligands
LIGAND_MODE = True

# Physiological protonation state of the ligand
# Most drug-like amines are protonated (charge=+1) at pH 7.4
# Most carboxylic acids are deprotonated (charge=-1) at pH 7.4
# Neutral molecules: charge=0
LIGAND_CHARGE = 0
LIGAND_SPIN   = 0   # 0 = singlet, 2 = triplet (radicals)

# Basis set for ligand mode — sto-3g is too small for accurate HOMO/LUMO
# 6-31G*  : good balance, ~2x more AOs than sto-3g, recommended start
# cc-pVDZ : higher accuracy, slower
# sto-3g  : only for rapid screening of many graph-transformer samples
LIGAND_BASIS = "sto-3g"#"6-31g*"

# ── Drug-like scoring thresholds ─────────────────────────────────────────────
# HOMO energy above this (eV) → molecule is easily oxidized → reactive/toxic risk
HOMO_REACTIVITY_THRESHOLD_EV = -7.0

# LUMO energy below this (eV) → molecule is easily reduced → electrophilic
LUMO_REACTIVITY_THRESHOLD_EV = -1.0

# HOMO-LUMO gap below this (eV) → high reactivity, potential instability
HL_GAP_DRUG_THRESHOLD_EV = 3.0

# Dipole moment above this (Debye) → strongly polar → may have solubility issues
DIPOLE_THRESHOLD_DEBYE = 5.0

# ── Binding pharmacophore atoms ───────────────────────────────────────────────
# Atoms that participate in protein-ligand interactions
# Used to weight Löwdin population analysis toward binding-relevant orbitals
PHARMACOPHORE_ATOMS = {"N", "O", "S", "F", "Cl"}

# ── Graph transformer scoring weights ─────────────────────────────────────────
# These weight the final grading score sent back to the graph transformer
# Higher weight = this property matters more for ranking candidates
SCORE_WEIGHTS = {
    "homo_lumo_gap_eV"       : 0.25,   # stability / selectivity
    "correlation_strength"   : 0.20,   # quantum accuracy needed
    "dipole_moment_debye"    : 0.15,   # polarity / solubility proxy
    "pharmacophore_fraction" : 0.20,   # fraction of active orbs on N/O/S
    "homo_energy_eV"         : 0.10,   # electron donation capacity
    "lumo_energy_eV"         : 0.10,   # electron acceptance capacity
}






# ── Paste these into your geometries dict in config.py ────────────────────────

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

    # ── Option 1: CuCl2 — SIMPLEST, start here ────────────────────────────────
    # Geometry  : linear D∞h
    # Metal     : Cu(II), d9, one unpaired electron
    # Why useful: smallest possible TM test — 3 atoms, well-studied d9 Jahn-Teller
    #             Active space almost always 1-2 orbitals → very fast FCI reference
    #             Good sanity check before running harder systems
    # Cu-Cl bond: 2.08 Å (experimental gas-phase)
    "CuCl2": [
        ("Cu", (0.000,  0.000,  0.000)),
        ("Cl", (0.000,  0.000,  2.080)),
        ("Cl", (0.000,  0.000, -2.080)),
    ],

    # ── Option 2: FeCl4 — TETRAHEDRAL d5, different geometry from FeN6 ────────
    # Geometry  : tetrahedral Td  (very different from FeN6 octahedral)
    # Metal     : Fe(III), d5, high-spin, half-filled shell
    # Why useful: same Fe as FeN6 but completely different ligand field
    #             d5 high-spin is strongly correlated → tests Tier 3 path
    #             Tetrahedral CF splitting is smaller than octahedral
    #             → more degenerate near-HOMO orbitals → harder active space
    # Fe-Cl bond: 2.19 Å (experimental FeCl4⁻)
    # Tetrahedral positions: vertices of a cube, scale = 2.19 / sqrt(3) = 1.264 Å
    "FeCl4": [
        ("Fe", ( 0.000,  0.000,  0.000)),
        ("Cl", ( 1.264,  1.264,  1.264)),
        ("Cl", ( 1.264, -1.264, -1.264)),
        ("Cl", (-1.264,  1.264, -1.264)),
        ("Cl", (-1.264, -1.264,  1.264)),
    ],

    # ── Option 3: MnF6 — CLOSEST ANALOG to FeN6 ──────────────────────────────
    # Geometry  : octahedral Oh  (identical coordination pattern to FeN6)
    # Metal     : Mn(IV), d3, half-filled t2g shell
    # Why useful: drop-in replacement for FeN6
    #             d3 is less correlated than d5/d6 → lower ΔE vs FCI expected
    #             F is more electronegative than N → stronger ligand field
    #             → larger HOMO-LUMO gap → potentially Tier 2 instead of Tier 3
    #             Direct comparison to FeN6 isolates metal vs ligand effect
    # Mn-F bond : 1.79 Å (experimental MnF6²⁻)
    "MnF6": [
        ("Mn", (0.000,  0.000,  0.000)),
        ("F",  (0.000,  0.000,  1.790)),
        ("F",  (0.000,  0.000, -1.790)),
        ("F",  (0.000,  1.790,  0.000)),
        ("F",  (0.000, -1.790,  0.000)),
        ("F",  (1.790,  0.000,  0.000)),
        ("F", (-1.790,  0.000,  0.000)),
    ],

    # ── Option 4: NiCO4 — TETRAHEDRAL with CO ligands, hardest ───────────────
    # Geometry  : tetrahedral Td
    # Metal     : Ni(0), d10, formally closed-shell but CO back-bonding
    #             creates significant correlation
    # Why useful: CO is a pi-acceptor ligand — very different bonding from
    #             ionic N/F/Cl ligands above
    #             Ni(0) d10 might look simple but CO backbonding mixes
    #             Ni 3d with CO π* → correlated multiconfigurational character
    #             Larger molecule (9 atoms) → tests scaling of DMET bath
    # Ni-C bond : 1.838 Å  C-O bond: 1.141 Å (experimental Ni(CO)4)
    # C position : 1.838 / sqrt(3) = 1.061 Å along each tetrahedral direction
    # O position : (1.838 + 1.141) / sqrt(3) = 1.720 Å along same direction
    "NiCO4": [
        ("Ni", ( 0.000,  0.000,  0.000)),
        ("C",  ( 1.061,  1.061,  1.061)),
        ("O",  ( 1.720,  1.720,  1.720)),
        ("C",  ( 1.061, -1.061, -1.061)),
        ("O",  ( 1.720, -1.720, -1.720)),
        ("C",  (-1.061,  1.061, -1.061)),
        ("O",  (-1.720,  1.720, -1.720)),
        ("C",  (-1.061, -1.061,  1.061)),
        ("O",  (-1.720, -1.720,  1.720)),
    ],
    "CO": [
    ("C", (0.000, 0.000, 0.000)),
    ("O", (0.000, 0.000, 1.128)),   # C-O = 1.128 Å experimental
],

"pyridine": [
    # C2v, N at top, ring in xy-plane
    # C-C = 1.394 Å, C-N = 1.337 Å, C-H = 1.086 Å
    ("N",  ( 0.000,  1.337,  0.000)),
    ("C",  ( 1.149,  0.690,  0.000)),
    ("C",  ( 1.197, -0.695,  0.000)),
    ("C",  ( 0.000, -1.392,  0.000)),
    ("C",  (-1.197, -0.695,  0.000)),
    ("C",  (-1.149,  0.690,  0.000)),
    ("H",  ( 2.071,  1.243,  0.000)),
    ("H",  ( 2.150, -1.248,  0.000)),
    ("H",  ( 0.000, -2.471,  0.000)),
    ("H",  (-2.150, -1.248,  0.000)),
    ("H",  (-2.071,  1.243,  0.000)),
],

"imidazole": [
    # 5-membered aromatic ring, two N atoms
    # Good model for histidine ligand in metalloenzymes
    ("N",  ( 0.000,  1.130,  0.000)),
    ("C",  ( 1.069,  0.442,  0.000)),
    ("N",  ( 0.660, -0.880,  0.000)),
    ("C",  (-0.660, -0.880,  0.000)),
    ("C",  (-1.069,  0.442,  0.000)),
    ("H",  ( 0.000,  2.145,  0.000)),
    ("H",  ( 2.075,  0.839,  0.000)),
    ("H",  ( 1.254, -1.671,  0.000)),
    ("H",  (-1.254, -1.671,  0.000)),
    ("H",  (-2.075,  0.839,  0.000)),
],

"bipyridine": [
    # 2,2'-bipyridine (bipy) — most common chelating ligand in TM chemistry
    # Two pyridine rings connected at C2-C2' bond, D2 symmetry approx
    # Ring 1 (left pyridine)
    ("N",  (-2.805,  1.126,  0.000)),
    ("C",  (-1.569,  1.390,  0.000)),
    ("C",  (-0.726,  0.300,  0.000)),
    ("C",  (-1.248, -0.988,  0.000)),
    ("C",  (-2.586, -1.210,  0.000)),
    ("C",  (-3.379, -0.107,  0.000)),
    ("H",  (-1.166,  2.394,  0.000)),
    ("H",  ( 0.348,  0.437,  0.000)),
    ("H",  (-0.637, -1.857,  0.000)),
    ("H",  (-2.982, -2.211,  0.000)),
    ("H",  (-4.457, -0.211,  0.000)),
    # Ring 2 (right pyridine, mirror of ring 1 across yz-plane)
    ("N",  ( 2.805,  1.126,  0.000)),
    ("C",  ( 1.569,  1.390,  0.000)),
    ("C",  ( 0.726,  0.300,  0.000)),
    ("C",  ( 1.248, -0.988,  0.000)),
    ("C",  ( 2.586, -1.210,  0.000)),
    ("C",  ( 3.379, -0.107,  0.000)),
    ("H",  ( 1.166,  2.394,  0.000)),
    ("H",  (-0.348,  0.437,  0.000)),
    ("H",  ( 0.637, -1.857,  0.000)),
    ("H",  ( 2.982, -2.211,  0.000)),
    ("H",  ( 4.457, -0.211,  0.000)),
],

"ozone": [
    # Classic strongly-correlated diradical character
    # Good stress-test: Tier 2-3 despite no TM, open-shell singlet
    # O-O bond = 1.272 Å, angle = 116.8°
    ("O",  ( 0.000,  0.000,  0.000)),
    ("O",  ( 1.090,  0.713,  0.000)),
    ("O",  (-1.090,  0.713,  0.000)),
],
# ── Tier 1 ligands (simple, fast screening) ───────────────────────────────

    # Urea — simplest H-bond donor/acceptor pharmacophore
    # Appears in kinase inhibitors, urea-based drugs
    # Tier 1: closed-shell, large gap
    "urea": [
        ("C", ( 0.000,  0.000,  0.000)),
        ("O", ( 0.000,  0.000,  1.220)),
        ("N", ( 1.153, -0.000, -0.663)),
        ("N", (-1.153,  0.000, -0.663)),
        ("H", ( 1.180,  0.813, -1.257)),
        ("H", ( 1.180, -0.813, -1.257)),
        ("H", (-1.180,  0.813, -1.257)),
        ("H", (-1.180, -0.813, -1.257)),
    ],

    # Acetamide — amide pharmacophore (backbone mimic)
    # Models peptide bond → protein backbone H-bond interactions
    "acetamide": [
        ("C", (-1.225,  0.418,  0.000)),
        ("C", ( 0.179,  0.012,  0.000)),
        ("O", ( 0.364, -1.189,  0.000)),
        ("N", ( 1.219,  0.903,  0.000)),
        ("H", (-1.850, -0.467,  0.000)),
        ("H", (-1.376,  1.047,  0.877)),
        ("H", (-1.376,  1.047, -0.877)),
        ("H", ( 1.075,  1.890,  0.000)),
        ("H", ( 2.168,  0.557,  0.000)),
    ],

    # ── Tier 2 ligands (moderate correlation, aromatic) ───────────────────────

    # Imidazole — histidine side chain model
    # Coordinates Zn2+, Fe2+, Cu2+ in metalloenzyme active sites
    # Tier 2: aromatic, moderate π correlation
    # N-H bond participates in H-bonding with Asp/Glu residues
    "imidazole": [
        ("N",  ( 0.000,  1.130,  0.000)),
        ("C",  ( 1.069,  0.442,  0.000)),
        ("N",  ( 0.660, -0.880,  0.000)),
        ("C",  (-0.660, -0.880,  0.000)),
        ("C",  (-1.069,  0.442,  0.000)),
        ("H",  ( 0.000,  2.145,  0.000)),
        ("H",  ( 2.075,  0.839,  0.000)),
        ("H",  ( 1.254, -1.671,  0.000)),
        ("H",  (-1.254, -1.671,  0.000)),
        ("H",  (-2.075,  0.839,  0.000)),
    ],

    # Indole — tryptophan side chain model
    # Key pharmacophore in many kinase inhibitors (ATP pocket π-stacking)
    # Tier 2: bicyclic aromatic, delocalized π system
    "indole": [
        ("C",  ( 0.000,  0.000,  0.000)),   # C2
        ("C",  ( 1.194,  0.694,  0.000)),   # C3
        ("C",  ( 2.374,  0.052,  0.000)),   # C3a
        ("C",  ( 3.564,  0.726,  0.000)),   # C4
        ("C",  ( 3.579,  2.116,  0.000)),   # C5
        ("C",  ( 2.393,  2.774,  0.000)),   # C6
        ("C",  ( 1.208,  2.090,  0.000)),   # C7
        ("C",  ( 0.000,  2.742,  0.000)),   # C7a
        ("N",  (-0.028,  1.390,  0.000)),   # N1
        ("H",  (-0.940, -0.534,  0.000)),   # H2
        ("H",  ( 1.159,  1.775,  0.000)),   # H3
        ("H",  ( 4.499,  0.185,  0.000)),   # H4
        ("H",  ( 4.511,  2.660,  0.000)),   # H5
        ("H",  ( 2.390,  3.858,  0.000)),   # H6
        ("H",  (-0.945,  3.263,  0.000)),   # H7
        ("H",  (-0.964,  1.034,  0.000)),   # HN
    ],

    # Pyrimidine — DNA base analog, common kinase hinge binder
    # Tier 1-2: aromatic, two N atoms provide H-bond acceptors
    # Present in imatinib, gefitinib, erlotinib scaffolds
    "pyrimidine": [
        ("N",  ( 0.000,  1.336,  0.000)),
        ("C",  ( 1.149,  0.672,  0.000)),
        ("C",  ( 1.149, -0.672,  0.000)),
        ("N",  ( 0.000, -1.336,  0.000)),
        ("C",  (-1.149, -0.672,  0.000)),
        ("C",  (-1.149,  0.672,  0.000)),
        ("H",  ( 2.092,  1.210,  0.000)),
        ("H",  ( 2.092, -1.210,  0.000)),
        ("H",  (-2.092, -1.210,  0.000)),
        ("H",  (-2.092,  1.210,  0.000)),
    ],

    # ── Tier 2-3 ligands (strong correlation, challenging) ────────────────────

    # Catechol — dopamine/norepinephrine pharmacophore
    # Chelates Fe3+ in transferrin, forms quinone under oxidation
    # Two adjacent OH groups → redox-active, near-degenerate orbitals
    "catechol": [
        ("C",  ( 0.000,  1.396,  0.000)),
        ("C",  ( 1.209,  0.698,  0.000)),
        ("C",  ( 1.209, -0.698,  0.000)),
        ("C",  ( 0.000, -1.396,  0.000)),
        ("C",  (-1.209, -0.698,  0.000)),
        ("C",  (-1.209,  0.698,  0.000)),
        ("O",  ( 0.000,  2.729,  0.000)),
        ("O",  ( 2.348,  1.358,  0.000)),
        ("H",  ( 2.152, -1.241,  0.000)),
        ("H",  ( 0.000, -2.479,  0.000)),
        ("H",  (-2.152, -1.241,  0.000)),
        ("H",  (-2.152,  1.241,  0.000)),
        ("H",  ( 0.862,  3.097,  0.000)),
        ("H",  ( 3.192,  0.861,  0.000)),
    ],

    # Porphine — simplest porphyrin (heme model without Fe)
    # Core pharmacophore of heme, chlorophyll, vitamin B12
    # Tier 3: 18π aromatic macrocycle, strongly correlated
    # Large active space: 4 inner N lone pairs + extensive π system
    # NOTE: Use BASIS="sto-3g" and GAP_MAX_NORB=12 for this molecule
    "porphine": [
        ("N",  ( 0.000,  2.040,  0.000)),
        ("C",  ( 1.082,  2.784,  0.000)),
        ("C",  ( 2.185,  2.040,  0.000)),
        ("C",  ( 2.185, -2.040,  0.000)),
        ("C",  ( 1.082, -2.784,  0.000)),
        ("N",  ( 0.000, -2.040,  0.000)),
        ("C",  (-1.082, -2.784,  0.000)),
        ("C",  (-2.185, -2.040,  0.000)),
        ("N",  (-2.040,  0.000,  0.000)),
        ("C",  (-2.784,  1.082,  0.000)),
        ("C",  (-2.040,  2.185,  0.000)),
        ("C",  ( 2.040,  2.185,  0.000)),   # meso-C alpha
        ("C",  ( 2.784,  1.082,  0.000)),
        ("N",  ( 2.040,  0.000,  0.000)),
        ("C",  ( 2.784, -1.082,  0.000)),
        ("C",  ( 2.040, -2.185,  0.000)),
        ("C",  (-2.040, -2.185,  0.000)),
        ("C",  (-2.784, -1.082,  0.000)),
        ("C",  ( 0.000,  3.468,  0.000)),   # meso-H carbons
        ("C",  ( 3.468,  0.000,  0.000)),
        ("C",  ( 0.000, -3.468,  0.000)),
        ("C",  (-3.468,  0.000,  0.000)),
        ("H",  ( 1.082,  3.868,  0.000)),
        ("H",  ( 3.868,  1.082,  0.000)),
        ("H",  ( 3.868, -1.082,  0.000)),
        ("H",  ( 1.082, -3.868,  0.000)),
        ("H",  (-1.082, -3.868,  0.000)),
        ("H",  (-3.868, -1.082,  0.000)),
        ("H",  (-3.868,  1.082,  0.000)),
        ("H",  (-1.082,  3.868,  0.000)),
        ("H",  ( 0.000,  4.552,  0.000)),
        ("H",  ( 4.552,  0.000,  0.000)),
        ("H",  ( 0.000, -4.552,  0.000)),
        ("H",  (-4.552,  0.000,  0.000)),
        ("H",  ( 2.040,  3.125,  0.000)),
        ("H",  (-2.040,  3.125,  0.000)),
        ("H",  ( 2.040, -3.125,  0.000)),
        ("H",  (-2.040, -3.125,  0.000)),
        ("H",  ( 0.000,  1.004,  0.000)),   # inner N-H
        ("H",  ( 0.000, -1.004,  0.000)),   # inner N-H
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
GAP_MAX_NORB = 16

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
BACKEND = "local"

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
QUANTUM_SOLVER = "sqdrift"

# ── SKQD Parameters ───────────────────────────────────────────────────────────
SKQD_KRYLOV_DIM   = 10     # number of Krylov vectors to build
SKQD_DT           = 0.5    # time step per Krylov evolution (Ha^-1)
SKQD_TROTTER_REPS = 2      # Trotter steps per evolution gate (accuracy vs depth)
SKQD_SHOTS        = 8192   # shots per Krylov circuit


# ── SqDRIFT Parameters ────────────────────────────────────────────────────────
SQDRIFT_NUM_CIRCUITS = 70    # was 10  → more ensemble diversity
SQDRIFT_NUM_GROUPS   = 100    # was 10  → covers ~13% of 2221 groups per circuit
SQDRIFT_TIME         = 2.0   # was 1.0 → larger time step mixes more states
SQDRIFT_ITERS        = 10     # was 10  → more recover+diagonalize rounds
SQDRIFT_SHOTS        = 8192  # was 8192 → double the sampling budget

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
STEP1_FILE  = os.path.join(RESULTS_DIR, "step1_asf.pkl")
STEP2_FILE  = os.path.join(RESULTS_DIR, "step2_hamiltonian.pkl")