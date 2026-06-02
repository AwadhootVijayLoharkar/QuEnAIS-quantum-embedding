"""
Step 5 — Binding-Relevant Quantum Scoring for Graph Transformer
===============================================================

Computes additional quantum chemical descriptors relevant to
protein-ligand binding. These descriptors are added to the existing
pipeline score vector and sent back as a grading signal to the
graph transformer.

Core ideology
─────────────
  Graph transformer generates candidate molecules (as graphs)
       ↓
  Steps 1-3: active space → DMET embedding → quantum solver
       ↓
  Step 4: visualization (optional)
       ↓
  Step 5 (this script): binding-relevant grading signal
       ↓
  Graph transformer receives scores → trains to generate better candidates

Why quantum descriptors for binding?
─────────────────────────────────────
  Classical docking scores (Glide, Vina) use force-field approximations.
  Quantum descriptors capture:
    - Charge transfer effects (π-stacking, halogen bonds)
    - Polarization of electron density at binding interface
    - Reactivity (covalent warheads, redox-active groups)
    - Tautomer stability (imidazole, guanidinium, amidine)
  These are systematically missed by classical scoring functions.

Descriptors computed
─────────────────────
  From UHF/MP2 (already available from Step 1 mf object):
    HOMO energy          → nucleophilicity, electron donation to metal
    LUMO energy          → electrophilicity, covalent warhead activity
    HOMO-LUMO gap        → chemical hardness, selectivity proxy
    Dipole moment        → desolvation penalty, electrostatic complementarity
    Molecular hardness   → (LUMO - HOMO) / 2  → resistance to deformation
    Fukui f+ index       → site of nucleophilic attack on protein
    Fukui f- index       → site of electrophilic attack from protein

  From Step 1 Löwdin analysis:
    Pharmacophore fraction → fraction of active orbs on N/O/S atoms
    H-bond donor count    → estimated from N-H and O-H bearing active orbs
    π-stacking score      → fraction of active orbs on aromatic carbons

  From Step 2 DMET:
    Correlation in binding site → how quantum the binding pocket needs to be
    Embedding accuracy          → sv² coverage already computed

  From Step 3 solver:
    Quantum correction          → FCI - HF energy in active space
    Convergence quality         → ΔE vs FCI → grading signal reliability

  Composite binding score (weighted sum for graph transformer):
    binding_score = Σ weight_i × normalized_descriptor_i

Requires : results/step1_asf.pkl
           results/step2_hamiltonian.pkl
           results/step3_results.pkl
           config.py (LIGAND_MODE, SCORE_WEIGHTS, PHARMACOPHORE_ATOMS)
Saves    : results/step5_binding_score.pkl
           results/step5_binding_score.json  ← for graph transformer interface
"""

import os
import sys
import json
import pickle
import numpy as np

import config

# ── Load all upstream results ─────────────────────────────────────────────────
RESULTS_DIR    = config.RESULTS_DIR
STEP5_PKL      = os.path.join(RESULTS_DIR, "step5_binding_score.pkl")
STEP5_JSON     = os.path.join(RESULTS_DIR, "step5_binding_score.json")

for fpath, label in [
    (config.STEP1_FILE, "Step 1"),
    (config.STEP2_FILE, "Step 2"),
    (os.path.join(RESULTS_DIR, "step3_results.pkl"), "Step 3"),
]:
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"[Step 5] {label} not found: {fpath}")

with open(config.STEP1_FILE, "rb") as f:              step1 = pickle.load(f)
with open(config.STEP2_FILE, "rb") as f:              step2 = pickle.load(f)
with open(os.path.join(RESULTS_DIR,
          "step3_results.pkl"), "rb") as f:            step3 = pickle.load(f)

# Unpack
mol_info         = step1["mol_info"]
scores_s1        = step1["scores"]
mo_list          = step1["mo_list"]
deviation        = step1["deviation"]
lowdin_weights   = step1["lowdin_weights"]
dominant_atoms   = step1["dominant_atoms"]

scores_s2        = step2["scores"]
n_emb            = step2["n_emb"]
n_alpha          = step2["n_alpha"]
n_beta           = step2["n_beta"]
fci_ref_e        = step2["fci_ref_e"]

solver_name      = step3["solver"]
final_energy     = step3["energy"]
error_vs_fci     = step3["error_vs_fci"]
converged        = step3["converged"]
pipeline_score   = step3.get("pipeline_score", {})

molecule  = mol_info["molecule"]
atom_syms = mol_info["atom_syms"]
n_atoms   = mol_info["n_atoms"]

print(f"\n{'='*65}")
print(f"[Step 5] Binding Score — {molecule}")
print(f"{'='*65}")

# ═════════════════════════════════════════════════════════════════════════════
# Phase A — Rerun UHF to get orbital energies + dipole moment
# These are not stored in step1 pkl but are fast to recompute
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Phase A: UHF orbital energies + dipole ──")

os.environ["BLOCKEXE"]            = config.BLOCKEXE_WRAPPER
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_DEBUG_CPU_TYPE"]  = "5"

from pyscf import gto, scf as pyscf_scf

# Use LIGAND_BASIS if in LIGAND_MODE, else fall back to config.BASIS
basis_to_use = (getattr(config, "LIGAND_BASIS", config.BASIS)
                if getattr(config, "LIGAND_MODE", False)
                else config.BASIS)
charge       = getattr(config, "LIGAND_CHARGE", 0)
spin         = getattr(config, "LIGAND_SPIN",   0)

mol = gto.M(
    atom    = config.GEOMETRY,
    basis   = basis_to_use,
    charge  = charge,
    spin    = spin,
    verbose = 0,
)

mf = pyscf_scf.UHF(mol)
mf.kernel()

print(f"  UHF energy  : {mf.e_tot:.8f} Ha  (converged = {mf.converged})")
print(f"  Charge      : {charge}  Spin : {spin}")
print(f"  Basis used  : {basis_to_use}")

# ── HOMO / LUMO energies ──────────────────────────────────────────────────────
# Use alpha channel for both UHF and RHF
HARTREE_TO_EV = 27.2114

if isinstance(mf.mo_energy, (tuple, list)):
    mo_e   = np.asarray(mf.mo_energy[0])
    mo_occ = np.asarray(mf.mo_occ[0])
else:
    mo_e   = np.asarray(mf.mo_energy)
    mo_occ = np.asarray(mf.mo_occ)

occ_mask  = mo_occ > 0.5
virt_mask = mo_occ < 0.5

homo_e_Ha  = float(mo_e[occ_mask][-1])  if occ_mask.any()  else 0.0
lumo_e_Ha  = float(mo_e[virt_mask][0])  if virt_mask.any() else 0.0
homo_ev    = homo_e_Ha * HARTREE_TO_EV
lumo_ev    = lumo_e_Ha * HARTREE_TO_EV
hl_gap_ev  = lumo_ev - homo_ev

# Koopmans' theorem approximations:
# IE  ≈ -HOMO (ionization energy  → H-bond donor strength proxy)
# EA  ≈ -LUMO (electron affinity  → electrophilicity proxy)
# η   = (LUMO - HOMO) / 2  (chemical hardness → selectivity)
# μ_e = (HOMO + LUMO) / 2  (chemical potential → reactivity direction)
ie_ev       = -homo_ev
ea_ev       = -lumo_ev
hardness_ev = hl_gap_ev / 2.0
chem_pot_ev = (homo_ev + lumo_ev) / 2.0

print(f"\n  HOMO         : {homo_ev:.4f} eV")
print(f"  LUMO         : {lumo_ev:.4f} eV")
print(f"  Gap          : {hl_gap_ev:.4f} eV")
print(f"  Ioniz. energy: {ie_ev:.4f} eV  (Koopmans)")
print(f"  Elec. affin. : {ea_ev:.4f} eV  (Koopmans)")
print(f"  Hardness η   : {hardness_ev:.4f} eV")
print(f"  Chem. pot. μ : {chem_pot_ev:.4f} eV")

# ── Dipole moment ─────────────────────────────────────────────────────────────
# Dipole moment from the UHF density matrix
# High dipole → strongly polar → desolvation penalty in binding
# Low dipole  → lipophilic → good membrane permeability
DEBYE_CONV = 2.5418   # a.u. → Debye

# FIXED — handles all three possible shapes PySCF can return
dm = mf.make_rdm1()

if isinstance(dm, (tuple, list)):
    # explicit Python tuple/list of two (nao, nao) arrays
    dm_total = np.asarray(dm[0]) + np.asarray(dm[1])
elif hasattr(dm, "ndim") and dm.ndim == 3:
    # numpy array shape (2, nao, nao) — most common UHF case
    dm_total = dm[0] + dm[1]
else:
    # RHF — single (nao, nao) array
    dm_total = dm

dm_total = 0.5 * (dm_total + dm_total.T)   # symmetrize numerical noise
dip_ao  = mol.intor('int1e_r')   # (3, nao, nao) position integrals
# Electronic contribution (negative charge): -e * Tr[r * dm]
dip_elec = np.array([-np.einsum('ij,ji->', dip_ao[k], dm_total)
                      for k in range(3)])
# Nuclear contribution
dip_nuc = np.zeros(3)
for i in range(mol.natm):
    z = mol.atom_charge(i)
    dip_nuc += z * np.array(mol.atom_coord(i))   # coords already in Bohr

dip_total = (dip_elec + dip_nuc) * DEBYE_CONV   # Debye
dipole_magnitude = float(np.linalg.norm(dip_total))

print(f"\n  Dipole moment: {dipole_magnitude:.4f} D  "
      f"(x={dip_total[0]:.3f}, y={dip_total[1]:.3f}, z={dip_total[2]:.3f})")

# ═════════════════════════════════════════════════════════════════════════════
# Phase B — Fukui Indices (condensed-to-atoms)
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Phase B: Fukui Indices ──")
print("  f+(r): site of nucleophilic attack from protein")
print("  f-(r): site of electrophilic attack from protein")

# Condensed Fukui indices via finite difference (N±1 electron systems)
# f+(k) = q(k, N+1) - q(k, N)  → high = site attacked by nucleophile
# f-(k) = q(k, N)   - q(k, N-1)→ high = site attacked by electrophile
# Löwdin charges used (more stable than Mulliken)

def lowdin_charges(mol_obj, mf_obj):
    S     = mol_obj.intor('int1e_ovlp')
    evals, evecs = np.linalg.eigh(S)
    mask  = evals > 1e-15
    S_sq  = (evecs[:, mask] * np.sqrt(evals[mask])) @ evecs[:, mask].T

    dm_raw = mf_obj.make_rdm1()

    # FIXED — same three-way check
    if isinstance(dm_raw, (tuple, list)):
        dm_tot = np.asarray(dm_raw[0]) + np.asarray(dm_raw[1])
    elif hasattr(dm_raw, "ndim") and dm_raw.ndim == 3:
        dm_tot = dm_raw[0] + dm_raw[1]
    else:
        dm_tot = dm_raw

    dm_tot = 0.5 * (dm_tot + dm_tot.T)

    dm_lo  = S_sq @ dm_tot @ S_sq

    ao_lbls = mol_obj.ao_labels(fmt=None)
    pop     = np.zeros(mol_obj.natm)

    for ao_j, (atom_idx, *_) in enumerate(ao_lbls):
        pop[atom_idx] += dm_lo[ao_j, ao_j]

    charges = np.array([mol_obj.atom_charge(i)
                        for i in range(mol_obj.natm)]) - pop
    return charges

# N-electron charges (already computed above)
q_N = lowdin_charges(mol, mf)

# N+1 electron (anion) — catches nucleophilic attack sites
try:
    mol_anion = gto.M(
        atom=config.GEOMETRY, basis=basis_to_use,
        charge=charge - 1, spin=1, verbose=0
    )
    mf_anion = pyscf_scf.UHF(mol_anion)
    mf_anion.kernel()
    q_Np1    = lowdin_charges(mol_anion, mf_anion)
    fukui_plus  = q_N - q_Np1     # positive where anion adds electron density
    fukui_plus  = np.maximum(fukui_plus, 0)
    fukui_ok    = True
    print(f"  N+1 (anion) UHF converged : {mf_anion.converged}")
except Exception as exc:
    print(f"  Anion calculation failed ({exc}) — Fukui f+ set to zeros")
    fukui_plus = np.zeros(n_atoms)
    fukui_ok   = False

# N-1 electron (cation) — catches electrophilic attack sites
try:
    mol_cation = gto.M(
        atom=config.GEOMETRY, basis=basis_to_use,
        charge=charge + 1, spin=1, verbose=0
    )
    mf_cation = pyscf_scf.UHF(mol_cation)
    mf_cation.kernel()
    q_Nm1    = lowdin_charges(mol_cation, mf_cation)
    fukui_minus = q_Nm1 - q_N     # positive where cation loses electron density
    fukui_minus = np.maximum(fukui_minus, 0)
    print(f"  N-1 (cation) UHF converged: {mf_cation.converged}")
except Exception as exc:
    print(f"  Cation calculation failed ({exc}) — Fukui f- set to zeros")
    fukui_minus = np.zeros(n_atoms)

print(f"\n  Atom | Symbol | Löwdin charge | f+ (nucl) | f- (elec)")
print(f"  {'─'*58}")
for i in range(n_atoms):
    print(f"  {i:4d} | {atom_syms[i]:6s} | {q_N[i]:+10.4f}   | "
          f"{fukui_plus[i]:8.4f}  | {fukui_minus[i]:8.4f}")

# Most reactive atoms
most_nucleophilic  = int(np.argmax(fukui_plus))
most_electrophilic = int(np.argmax(fukui_minus))
print(f"\n  Most nucleophilic  atom : {most_nucleophilic} ({atom_syms[most_nucleophilic]})  "
      f"f+ = {fukui_plus[most_nucleophilic]:.4f}")
print(f"  Most electrophilic atom : {most_electrophilic} ({atom_syms[most_electrophilic]})  "
      f"f- = {fukui_minus[most_electrophilic]:.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# Phase C — Pharmacophore-Aware Orbital Analysis
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Phase C: Pharmacophore Analysis ──")

pharmacophore_set = getattr(config, "PHARMACOPHORE_ATOMS", {"N", "O", "S", "F", "Cl"})
n_active          = len(mo_list)

# Fraction of active orbital weight on pharmacophore atoms (N/O/S/halogens)
# High fraction → quantum correlation is concentrated on binding-relevant atoms
# → the molecule's key binding interactions are quantum in nature
pharma_atom_idx = [i for i in range(n_atoms) if atom_syms[i] in pharmacophore_set]
aromatic_syms   = {"C"}   # will be refined below by deviation threshold

if len(pharma_atom_idx) > 0 and lowdin_weights.shape[0] > 0:
    pharma_weight_per_orb = lowdin_weights[:, pharma_atom_idx].sum(axis=1)
    pharmacophore_fraction = float(np.mean(pharma_weight_per_orb))
else:
    pharmacophore_fraction = 0.0

# π-stacking score: fraction of active orbital weight on sp2 carbons
# Proxy: carbon atoms with deviation > 0.05 (fractional occupation → aromatic)
# High π-stacking score → molecule can stack with Phe/Tyr/Trp in ATP pocket
carbon_idx = [i for i in range(n_atoms) if atom_syms[i] == "C"]
if len(carbon_idx) > 0 and lowdin_weights.shape[0] > 0:
    pi_weight_per_orb = lowdin_weights[:, carbon_idx].sum(axis=1)
    pi_stacking_score = float(np.mean(pi_weight_per_orb))
else:
    pi_stacking_score = 0.0

# H-bond capacity: count active orbitals predominantly on N or O atoms
# These orbitals participate in H-bond donation/acceptance with protein backbone
hbond_atoms  = [i for i in range(n_atoms) if atom_syms[i] in {"N", "O"}]
if len(hbond_atoms) > 0 and lowdin_weights.shape[0] > 0:
    hbond_weight = lowdin_weights[:, hbond_atoms].sum(axis=1)
    n_hbond_orbs = int(np.sum(hbond_weight > 0.5))
    hbond_score  = float(np.mean(hbond_weight))
else:
    n_hbond_orbs = 0
    hbond_score  = 0.0

print(f"\n  Pharmacophore atoms (N/O/S/hal) : {[f'{atom_syms[i]}({i})' for i in pharma_atom_idx]}")
print(f"  Pharmacophore fraction          : {pharmacophore_fraction:.4f}")
print(f"  π-stacking score                : {pi_stacking_score:.4f}")
print(f"  H-bond orbital count            : {n_hbond_orbs}")
print(f"  H-bond weight score             : {hbond_score:.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# Phase D — Reactivity Flags
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Phase D: Reactivity Flags ──")

# Flags relevant to medicinal chemistry filtering
# These are binary pass/fail criteria used in graph transformer feedback

homo_thresh   = getattr(config, "HOMO_REACTIVITY_THRESHOLD_EV", -7.0)
lumo_thresh   = getattr(config, "LUMO_REACTIVITY_THRESHOLD_EV", -1.0)
gap_thresh    = getattr(config, "HL_GAP_DRUG_THRESHOLD_EV",      3.0)
dipole_thresh = getattr(config, "DIPOLE_THRESHOLD_DEBYE",        5.0)

flags = {
    # True = flag raised = potential problem
    "high_reactivity"   : hl_gap_ev < gap_thresh,
    "strong_nucleophile": homo_ev > homo_thresh,
    "electrophilic"     : lumo_ev < lumo_thresh,
    "strongly_polar"    : dipole_magnitude > dipole_thresh,
    "strongly_correlated": float(scores_s1.get("correlation_strength", 0)) > 0.7,
    "open_shell"        : int(scores_s1.get("n_somo", 0)) > 0,
    "quantum_accurate"  : converged,
}

print(f"\n  {'Flag':<25} {'Raised?':>8}   Criterion")
print(f"  {'─'*65}")
flag_criteria = {
    "high_reactivity"    : f"gap {hl_gap_ev:.2f} eV < {gap_thresh} eV",
    "strong_nucleophile" : f"HOMO {homo_ev:.2f} eV > {homo_thresh} eV",
    "electrophilic"      : f"LUMO {lumo_ev:.2f} eV < {lumo_thresh} eV",
    "strongly_polar"     : f"dipole {dipole_magnitude:.2f} D > {dipole_thresh} D",
    "strongly_correlated": f"corr strength {scores_s1.get('correlation_strength',0):.2f} > 0.7",
    "open_shell"         : f"n_SOMO = {scores_s1.get('n_somo',0)}",
    "quantum_accurate"   : f"ΔE vs FCI = {error_vs_fci:.2e} Ha" if error_vs_fci else "N/A",
}
for fname, fval in flags.items():
    symbol = "⚠" if fval else "✓"
    print(f"  {fname:<25} {symbol} {str(fval):<6}   {flag_criteria[fname]}")

# ═════════════════════════════════════════════════════════════════════════════
# Phase E — Composite Binding Score (graph transformer grading signal)
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Phase E: Composite Binding Score ──")

# Each descriptor is normalized to [0, 1] before weighting
# The normalization references are physically motivated ranges:
#   HOMO: typical drug-like range -12 to -7 eV  → higher = better donor
#   LUMO: typical range -3 to +3 eV             → higher = less electrophilic
#   Gap:  typical range 3-10 eV                 → higher = more selective
#   Dipole: 0-8 D                                → lower = better membrane perm
#   Correlation: 0-1 already normalized
#   Pharmacophore fraction: 0-1 already normalized

def clamp01(x):
    return float(max(0.0, min(1.0, x)))

# Normalized descriptors (all in 0=bad, 1=good direction for binding)
norm = {
    "homo_energy_eV"       : clamp01((homo_ev - (-14.0)) / (-7.0 - (-14.0))),
    "lumo_energy_eV"       : clamp01((lumo_ev - (-3.0)) / (3.0 - (-3.0))),
    "homo_lumo_gap_eV"     : clamp01((hl_gap_ev - 3.0) / (10.0 - 3.0)),
    "dipole_moment_debye"  : clamp01(1.0 - dipole_magnitude / 10.0),
    "correlation_strength" : clamp01(float(scores_s1.get("correlation_strength", 0))),
    "pharmacophore_fraction": clamp01(pharmacophore_fraction),
}

# Weights from config (or defaults)
weights = getattr(config, "SCORE_WEIGHTS", {
    "homo_lumo_gap_eV"       : 0.25,
    "correlation_strength"   : 0.20,
    "dipole_moment_debye"    : 0.15,
    "pharmacophore_fraction" : 0.20,
    "homo_energy_eV"         : 0.10,
    "lumo_energy_eV"         : 0.10,
})

# Weighted sum → composite score in [0, 1]
composite_score = sum(
    weights.get(k, 0.0) * v
    for k, v in norm.items()
)
total_weight = sum(weights.get(k, 0.0) for k in norm)
if total_weight > 0:
    composite_score /= total_weight
composite_score = float(composite_score)

# Penalty for raised flags (reduces score)
n_flags_raised = sum(1 for f in flags.values()
                     if f and f != "quantum_accurate")
flag_penalty   = min(n_flags_raised * 0.05, 0.25)   # max 25% penalty
final_score    = max(0.0, composite_score - flag_penalty)

print(f"\n  {'Descriptor':<30} {'Raw':>10}  {'Norm':>6}  {'Weight':>7}")
print(f"  {'─'*60}")
descriptor_raw = {
    "homo_energy_eV"        : homo_ev,
    "lumo_energy_eV"        : lumo_ev,
    "homo_lumo_gap_eV"      : hl_gap_ev,
    "dipole_moment_debye"   : dipole_magnitude,
    "correlation_strength"  : float(scores_s1.get("correlation_strength", 0)),
    "pharmacophore_fraction": pharmacophore_fraction,
}
for k, nv in norm.items():
    raw = descriptor_raw.get(k, 0.0)
    w   = weights.get(k, 0.0)
    print(f"  {k:<30} {raw:10.4f}  {nv:6.3f}  {w:7.2f}")

print(f"\n  Composite score (pre-penalty)  : {composite_score:.4f}")
print(f"  Flags raised                   : {n_flags_raised}")
print(f"  Flag penalty                   : -{flag_penalty:.4f}")
print(f"  ── FINAL BINDING SCORE         : {final_score:.4f}  (0=poor, 1=excellent)")

# ═════════════════════════════════════════════════════════════════════════════
# Assemble full output for graph transformer
# ═════════════════════════════════════════════════════════════════════════════
binding_scores = {
    # ── Primary grading signal ────────────────────────────────────────────────
    "molecule"              : molecule,
    "final_binding_score"   : final_score,      # ← main signal to graph transformer
    "composite_pre_penalty" : composite_score,
    "n_flags_raised"        : n_flags_raised,
    "flag_penalty"          : flag_penalty,

    # ── Orbital energy descriptors ────────────────────────────────────────────
    "homo_energy_Ha"        : homo_e_Ha,
    "lumo_energy_Ha"        : lumo_e_Ha,
    "homo_energy_eV"        : homo_ev,
    "lumo_energy_eV"        : lumo_ev,
    "homo_lumo_gap_eV"      : hl_gap_ev,
    "ionization_energy_eV"  : ie_ev,
    "electron_affinity_eV"  : ea_ev,
    "hardness_eV"           : hardness_ev,
    "chemical_potential_eV" : chem_pot_ev,

    # ── Polarity ──────────────────────────────────────────────────────────────
    "dipole_moment_debye"   : float(dipole_magnitude),
    "dipole_x_debye"        : float(dip_total[0]),
    "dipole_y_debye"        : float(dip_total[1]),
    "dipole_z_debye"        : float(dip_total[2]),

    # ── Fukui reactivity indices ──────────────────────────────────────────────
    "fukui_plus"            : fukui_plus.tolist(),
    "fukui_minus"           : fukui_minus.tolist(),
    "most_nucleophilic_atom": int(most_nucleophilic),
    "most_electrophilic_atom": int(most_electrophilic),
    "fukui_ok"              : fukui_ok,

    # ── Pharmacophore analysis ────────────────────────────────────────────────
    "pharmacophore_fraction": pharmacophore_fraction,
    "pi_stacking_score"     : pi_stacking_score,
    "hbond_score"           : hbond_score,
    "n_hbond_active_orbs"   : n_hbond_orbs,

    # ── Quantum accuracy ──────────────────────────────────────────────────────
    "fci_ref_energy"        : float(fci_ref_e) if fci_ref_e is not None else None,
    "solver_energy"         : float(final_energy) if final_energy is not None else None,
    "error_vs_fci"          : float(error_vs_fci) if error_vs_fci is not None else None,
    "solver_converged"      : bool(converged),
    "solver_name"           : solver_name,

    # ── Reactivity flags (for graph transformer penalty) ──────────────────────
    "flags"                 : {k: bool(v) for k, v in flags.items()},

    # ── Full normalized descriptor vector (for ML input) ─────────────────────
    "normalized_descriptors": norm,
    "score_weights"         : weights,

    # ── Upstream pipeline scores (pass-through) ───────────────────────────────
    "pipeline_s1"           : scores_s1,
    "pipeline_s2"           : scores_s2,
}

# ── Save pickle ───────────────────────────────────────────────────────────────
with open(STEP5_PKL, "wb") as f:
    pickle.dump(binding_scores, f)
print(f"\n  ✓ Saved (pkl) → {STEP5_PKL}")

# ── Save JSON (graph transformer interface) ───────────────────────────────────
# JSON output strips non-serializable nested objects
json_output = {
    k: v for k, v in binding_scores.items()
    if k not in ("pipeline_s1", "pipeline_s2")
}
with open(STEP5_JSON, "w") as f:
    json.dump(json_output, f, indent=2, default=str)
print(f"  ✓ Saved (json) → {STEP5_JSON}")

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"[Step 5] Binding Score Summary — {molecule}")
print(f"{'='*65}")
print(f"  HOMO / LUMO      : {homo_ev:.3f} eV / {lumo_ev:.3f} eV")
print(f"  H-L gap          : {hl_gap_ev:.3f} eV")
print(f"  Dipole           : {dipole_magnitude:.3f} D")
print(f"  Hardness η       : {hardness_ev:.3f} eV")
print(f"  Pharma fraction  : {pharmacophore_fraction:.4f}")
print(f"  π-stacking score : {pi_stacking_score:.4f}")
print(f"  H-bond score     : {hbond_score:.4f}")
print(f"  Flags raised     : {n_flags_raised}")
print(f"\n  ► FINAL BINDING SCORE : {final_score:.4f}")
print(f"    (sent to graph transformer as grading signal)")
print(f"{'='*65}")