# step1_asf.py — Active Space Finder for Strongly Correlated Molecules
"""
Identifies the most correlated orbitals for quantum embedding.

Phases:
  A: UHF + tier classification (simple / moderate / strongly correlated)
  B: MP2 natural orbital deviations + ASF candidate pool
  C: Adaptive gap detection → final active space
  D: Löwdin population analysis → orbital-to-atom mapping

Requires: config.py, CIF file in cif_files/
Saves:    results/step1_asf.pkl
"""

import os
import sys
import pickle
import numpy as np

import config

# ── Setup ─────────────────────────────────────────────────────────────────────
FORCE_RERUN = True

os.makedirs(config.RESULTS_DIR, exist_ok=True)
if os.path.exists(config.STEP1_FILE) and not FORCE_RERUN:
    print(f"[Step 1] Cached: {config.STEP1_FILE}")
    sys.exit(0)

os.environ["BLOCKEXE"]            = config.BLOCKEXE_WRAPPER
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_DEBUG_CPU_TYPE"]  = "5"

from pyscf import gto, scf, mp as pyscf_mp
from pyscf.dmrgscf import dmrgci
from asf.wrapper import find_from_scf

dmrgci.settings.BLOCKEXE = config.BLOCKEXE_WRAPPER


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def run_uhf(mol):
    """Run UHF with DIIS + Newton fallback for difficult convergence."""
    mf = scf.UHF(mol)
    mf.max_cycle   = 400
    mf.level_shift = 0.5
    mf.kernel()

    if not mf.converged:
        print("  DIIS failed → Newton solver...")
        nw = mf.newton()
        nw.max_cycle = 400
        nw.kernel(mf.mo_coeff)
        if nw.converged:
            for attr in ("e_tot", "mo_coeff", "mo_energy", "mo_occ", "converged"):
                setattr(mf, attr, getattr(nw, attr))

    return mf


def classify(mol, mf):
    """
    Classify molecule into tiers based on:
      1. Presence of d/f-block elements → Tier 3
      2. Spin contamination > threshold → Tier 2
      3. Small HOMO-LUMO gap → Tier 2
      4. Otherwise → Tier 1
    """
    has_tm = any(mol.atom_symbol(i) in config.TM_ELEMENTS for i in range(mol.natm))

    # Spin contamination
    s2, _ = mf.spin_square()
    s_expected = (mol.spin / 2.0) * (mol.spin / 2.0 + 1.0)
    spin_cont = s2 / max(s_expected, 0.75)

    # HOMO-LUMO gap
    mo_e = np.asarray(mf.mo_energy[0])  # alpha channel
    mo_occ = np.asarray(mf.mo_occ[0])
    occ_e = mo_e[mo_occ > 0.5]
    vir_e = mo_e[mo_occ < 0.5]
    gap_ev = (vir_e[0] - occ_e[-1]) * 27.2114 if len(occ_e) and len(vir_e) else 10.0

    indicators = {
        "has_tm": has_tm,
        "spin_contamination": float(spin_cont),
        "s2": float(s2),
        "homo_lumo_gap_eV": float(gap_ev),
    }

    if has_tm:
        return 3, indicators
    if spin_cont > config.SPIN_CONTAMINATION_TIER2_THRESHOLD or gap_ev < config.HOMO_LUMO_TIER2_THRESHOLD_EV:
        return 2, indicators
    return 1, indicators


def compute_mp2_deviations(mf, mol):
    """
    Compute MP2 natural orbital deviations: dev_i = min(n_i, 2 - n_i).
      dev = 0 → doubly occupied or empty (uncorrelated)
      dev = 1 → half-filled (maximally correlated)
    """
    S = mol.intor("int1e_ovlp")
    evals, evecs = np.linalg.eigh(S)
    mask = evals > 1e-15
    S_invsqrt = (evecs[:, mask] / np.sqrt(evals[mask])) @ evecs[:, mask].T

    try:
        mymp = pyscf_mp.MP2(mf)
        mymp.verbose = 0
        e_corr, _ = mymp.kernel()
        dm1 = mymp.make_rdm1()

        if isinstance(dm1, (tuple, list)):
            Ca, Cb = np.asarray(mf.mo_coeff[0]), np.asarray(mf.mo_coeff[1])
            dm_ao = Ca @ dm1[0] @ Ca.T + Cb @ dm1[1] @ Cb.T
        else:
            dm_ao = mf.mo_coeff @ dm1 @ mf.mo_coeff.T
    except Exception:
        e_corr = 0.0
        dm_raw = mf.make_rdm1()
        dm_ao = (dm_raw[0] + dm_raw[1]) if isinstance(dm_raw, tuple) else dm_raw

    # Diagonalize in Löwdin basis → natural orbital occupations
    dm_lo = S_invsqrt @ dm_ao @ S_invsqrt.T
    dm_lo = 0.5 * (dm_lo + dm_lo.T)
    no_occ = np.clip(np.linalg.eigvalsh(dm_lo)[::-1], 0.0, 2.0)
    deviation = np.minimum(no_occ, 2.0 - no_occ)

    return deviation, no_occ, e_corr


def find_gap_cutoff(values, min_n, max_n):
    """
    Find largest gap in sorted values to determine active space size.
    Returns (n_selected, gap_size, indices_of_selected).
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    min_n = max(1, min(min_n, n))
    max_n = min(max_n, n)

    if min_n >= max_n:
        order = np.argsort(-values)
        return min_n, 0.0, list(order[:min_n])

    order = np.argsort(-values)
    sorted_v = values[order]

    best_gap, best_n = -1.0, min_n
    for k in range(min_n, max_n + 1):
        gap = sorted_v[k-1] - (sorted_v[k] if k < n else 0.0)
        if gap > best_gap:
            best_gap, best_n = gap, k

    return best_n, float(best_gap), list(order[:best_n])


def lowdin_population(mo_coeff, mo_list, S, ao_labels, n_atoms):
    """Löwdin population: weight of each active MO on each atom."""
    evals, evecs = np.linalg.eigh(S)
    mask = evals > 1e-15
    S_sqrt = (evecs[:, mask] * np.sqrt(evals[mask])) @ evecs[:, mask].T

    weights = np.zeros((len(mo_list), n_atoms))
    for k, mo_idx in enumerate(mo_list):
        c_lo = S_sqrt @ mo_coeff[:, mo_idx]
        for ao_j, (atom_idx, *_) in enumerate(ao_labels):
            weights[k, atom_idx] += c_lo[ao_j] ** 2

    return weights


def count_active_electrons(mol, no_occ, active_list):
    """Count electrons in active space = total - 2×core."""
    active_set = set(active_list)
    n_core = sum(1 for i, occ in enumerate(no_occ)
                 if i not in active_set and occ > config.CORE_OCC_THRESHOLD)
    nel = mol.nelectron - 2 * n_core
    nel = max(2, min(nel, 2 * len(active_list)))
    if nel % 2 != 0:
        nel -= 1
    return nel


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"[Step 1] Active Space Finder — {config.MOLECULE}")
print(f"{'='*60}")

# Build molecule
mol = gto.M(
    atom    = config.GEOMETRY,
    basis   = config.BASIS,
    charge  = config.CHARGE,
    spin    = config.SPIN,
    verbose = 3,
)

# ── Phase A: UHF + Classification ────────────────────────────────────────────
print("\n── Phase A: UHF + Classification ──")
mf = run_uhf(mol)
print(f"  UHF energy = {mf.e_tot:.8f} Ha  (converged: {mf.converged})")

tier, indicators = classify(mol, mf)
print(f"  Tier {tier}: TM={indicators['has_tm']}, "
      f"gap={indicators['homo_lumo_gap_eV']:.2f} eV, "
      f"⟨S²⟩={indicators['s2']:.3f}")

# ── Phase B: MP2 Deviations + ASF ────────────────────────────────────────────
print("\n── Phase B: MP2 Deviations + ASF ──")
deviation, no_occ, e_corr = compute_mp2_deviations(mf, mol)
print(f"  MP2 correlation energy: {e_corr:.6f} Ha")
print(f"  Orbitals with dev > 0.05: {int(np.sum(deviation > 0.05))}")

asf_p = config.ASF_PARAMS[tier]
print(f"  Running ASF (Tier {tier}, max_norb={asf_p['max_norb']})...")

active_space = find_from_scf(
    mf,
    entropy_threshold = asf_p["entropy_threshold"],
    max_norb          = asf_p["max_norb"],
    min_norb          = asf_p["min_norb"],
    verbose           = True,
)

mo_list  = list(active_space.mo_list)
mo_coeff = active_space.mo_coeff
n_cand   = len(mo_list)
print(f"  ASF candidates: {n_cand} orbitals → {mo_list}")

if n_cand == 0:
    raise RuntimeError("ASF returned 0 candidates. Lower entropy_threshold.")

# ── Phase C: Gap Detection ────────────────────────────────────────────────────
print("\n── Phase C: Gap Detection ──")
cand_devs = np.array([deviation[i] if i < len(deviation) else 0.0 for i in mo_list])

for mo_idx, dev in sorted(zip(mo_list, cand_devs), key=lambda x: -x[1]):
    print(f"  MO {mo_idx:3d}: dev={dev:.4f}  {'█' * int(dev * 20)}")

n_final, gap_val, selected_k = find_gap_cutoff(
    cand_devs, config.GAP_MIN_NORB, config.GAP_MAX_NORB
)
final_mo_list = sorted(mo_list[k] for k in selected_k)
nel = count_active_electrons(mol, no_occ, final_mo_list)

print(f"\n  Result: ({nel}e, {n_final}orb) active space")
print(f"  Orbitals: {final_mo_list}  |  Gap: {gap_val:.4f}")

# ── Phase D: Löwdin Population ────────────────────────────────────────────────
print("\n── Phase D: Löwdin Population ──")
S = mol.intor("int1e_ovlp")
ao_labels = mol.ao_labels(fmt=None)
weights = lowdin_population(mo_coeff, final_mo_list, S, ao_labels, config.N_ATOMS)
dominant_atoms = np.argmax(weights, axis=1).astype(int)

for k, mo_idx in enumerate(final_mo_list):
    atom = dominant_atoms[k]
    print(f"  MO {mo_idx:3d} → {config.ATOM_SYMS[atom]:2s} (atom {atom})")

# Correlation strength
final_devs = np.array([deviation[i] for i in final_mo_list if i < len(deviation)])
corr_strength = float(np.mean(final_devs)) if len(final_devs) > 0 else 0.0

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"[Step 1] Summary: {config.MOLECULE}")
print(f"  Tier               : {tier}")
print(f"  Active space       : ({nel}e, {n_final}orb)")
print(f"  Orbitals           : {final_mo_list}")
print(f"  Correlation strength: {corr_strength:.3f}")
print(f"{'='*60}")

# ── Save ──────────────────────────────────────────────────────────────────────
results = {
    "nel"             : nel,
    "mo_list"         : final_mo_list,
    "mo_coeff"        : mo_coeff,
    "n_active_orbs"   : n_final,
    "no_occ"          : no_occ,
    "deviation"       : deviation,
    "lowdin_weights"  : weights,
    "dominant_atoms"  : dominant_atoms,
    "tier"            : tier,
    "indicators"      : indicators,
    "corr_strength"   : corr_strength,
    "mol_info": {
        "molecule"    : config.MOLECULE,
        "basis"       : config.BASIS,
        "n_atoms"     : config.N_ATOMS,
        "atom_syms"   : config.ATOM_SYMS,
        "n_electrons" : mol.nelectron,
        "n_ao"        : mol.nao_nr(),},
    "uhf_energy"  : float(mf.e_tot),
    "mo_coeff_uhf": np.asarray(mf.mo_coeff),
    "mo_energy"   : np.asarray(mf.mo_energy),
    "mo_occ"      : np.asarray(mf.mo_occ),
    "converged"   : mf.converged,
}

with open(config.STEP1_FILE, "wb") as f:
    pickle.dump(results, f)

print(f"\n[Step 1] ✓ Saved → {config.STEP1_FILE}")