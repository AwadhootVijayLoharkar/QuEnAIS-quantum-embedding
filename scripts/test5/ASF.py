# step1_asf.py — Active Space Finder for Strongly Correlated Molecules
"""
Identifies the most correlated orbitals for quantum embedding.

Phases:
  A: UHF + tier classification (simple / moderate / strongly correlated)
  B: MP2 natural orbital deviations + ASF candidate pool
  C: Adaptive gap detection → final active space
  D: Löwdin population analysis → orbital-to-atom mapping

Fixes vs original:
  - classify(): singlet S²=0 handled separately (absolute threshold, not ratio)
  - classify(): HOMO-LUMO gap uses minimum across BOTH spin channels
  - classify(): HARTREE_TO_EV constant from config, not magic number inline
  - compute_mp2_deviations(): bare Exception catch replaced with specific types
  - count_active_electrons(): uses MO occupations (not Löwdin NO occupations)
    to avoid basis-space mismatch; adds bounds checks and diagnostics
  - FORCE_RERUN: controlled via CLI flag --force instead of hardcoded True

Requires: config.py, CIF file in cif_files/
Saves:    results/step1_asf.pkl
"""

import os
import sys
import pickle
import argparse
import warnings
import numpy as np

import config

# ── CLI argument: --force bypasses cache ──────────────────────────────────────
parser = argparse.ArgumentParser(description="Step 1: Active Space Finder")
parser.add_argument("--force", action="store_true",
                    help="Rerun even if cached result exists")
args = parser.parse_args()
FORCE_RERUN = args.force

# ── Setup ─────────────────────────────────────────────────────────────────────
os.makedirs(config.RESULTS_DIR, exist_ok=True)
if os.path.exists(config.STEP1_FILE) and not FORCE_RERUN:
    print(f"[Step 1] Using cached result: {config.STEP1_FILE}")
    print(f"         Run with --force to recompute.")
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
    """
    Run UHF with DIIS convergence acceleration.
    Falls back to Newton (second-order) solver if DIIS fails.
    level_shift=0.5 helps convergence for open-shell / near-degenerate systems.
    """
    mf = scf.UHF(mol)
    mf.max_cycle   = 400
    mf.level_shift = 0.5
    mf.kernel()

    if not mf.converged:
        print("  DIIS did not converge → trying Newton solver...")
        nw = mf.newton()
        nw.max_cycle = 400
        nw.kernel(mf.mo_coeff)
        if nw.converged:
            for attr in ("e_tot", "mo_coeff", "mo_energy", "mo_occ", "converged"):
                setattr(mf, attr, getattr(nw, attr))
        else:
            warnings.warn(
                "UHF did not converge with DIIS or Newton. "
                "Results may be unreliable. Consider a better initial guess "
                "or a different basis set.",
                RuntimeWarning,
            )

    return mf


def classify(mol, mf):
    """
    Classify molecule into correlation tiers:
      Tier 3 — has d/f-block transition metal elements
      Tier 2 — significant spin contamination OR small HOMO-LUMO gap
      Tier 1 — weakly correlated, standard treatment sufficient

    Fixes vs original:
      1. Singlet (S=0) uses absolute ⟨S²⟩ threshold, not a ratio.
         Ratio S²/S(S+1) is 0/0 for singlets — the original code patched
         this with max(..., 0.75) which gives a physically meaningless number.
      2. HOMO-LUMO gap uses the MINIMUM across alpha AND beta channels.
         Original used only alpha, missing cases where beta gap is smaller.
      3. Unit conversion uses config.HARTREE_TO_EV constant (NIST 2018),
         not an inline magic number.
    """
    has_tm = any(mol.atom_symbol(i) in config.TM_ELEMENTS
                 for i in range(mol.natm))

    # ── Spin contamination ────────────────────────────────────────────────────
    s2, _  = mf.spin_square()
    S_val  = mol.spin / 2.0          # PySCF mol.spin = 2S → S = mol.spin/2
    s_expected = S_val * (S_val + 1.0)

    is_singlet = (mol.spin == 0)

    if is_singlet:
        # For a singlet S(S+1)=0, so the ratio S²/S(S+1) is undefined.
        # Any nonzero ⟨S²⟩ is pure contamination — use it as an absolute metric.
        spin_cont         = float(s2)          # absolute deviation from 0
        spin_contaminated = spin_cont > config.SPIN_CONTAMINATION_SINGLET_THRESHOLD
        spin_metric_label = f"⟨S²⟩={s2:.4f} (singlet, expected 0)"
    else:
        # For non-singlets, ratio measures how far above ideal the state is.
        # Pure spin state → ratio = 1.0; contamination → ratio > 1.0
        spin_cont         = float(s2 / s_expected)
        spin_contaminated = spin_cont > config.SPIN_CONTAMINATION_TIER2_THRESHOLD
        spin_metric_label = (f"⟨S²⟩={s2:.4f} / expected {s_expected:.4f} "
                             f"= ratio {spin_cont:.3f}")

    # ── HOMO-LUMO gap (both spin channels) ───────────────────────────────────
    gaps    = {}
    gap_min = 10.0   # default (large) if no gap found

    for label, ch in [("alpha", 0), ("beta", 1)]:
        mo_e   = np.asarray(mf.mo_energy[ch])
        mo_occ = np.asarray(mf.mo_occ[ch])
        occ_e  = mo_e[mo_occ > 0.5]
        vir_e  = mo_e[mo_occ < 0.5]
        if len(occ_e) > 0 and len(vir_e) > 0:
            gaps[label] = float((vir_e[0] - occ_e[-1]) * config.HARTREE_TO_EV)

    if gaps:
        gap_min = min(gaps.values())   # use worst-case (smallest) gap

    gap_label = "  ".join(f"{k}={v:.3f} eV" for k, v in gaps.items())
    gap_label += f"  →  min={gap_min:.3f} eV"

    # ── Build indicators dict ─────────────────────────────────────────────────
    indicators = {
        "has_tm"              : has_tm,
        "is_singlet"          : is_singlet,
        "s2"                  : float(s2),
        "s_expected"          : float(s_expected),
        "spin_cont"           : float(spin_cont),
        "spin_contaminated"   : spin_contaminated,
        "homo_lumo_gap_eV"    : float(gap_min),
        "gap_alpha_eV"        : gaps.get("alpha"),
        "gap_beta_eV"         : gaps.get("beta"),
    }

    # ── Tier decision ─────────────────────────────────────────────────────────
    if has_tm:
        tier = 3
    elif spin_contaminated or gap_min < config.HOMO_LUMO_TIER2_THRESHOLD_EV:
        tier = 2
    else:
        tier = 1

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print(f"  Spin  : {spin_metric_label}")
    print(f"         contaminated = {'YES → Tier≥2' if spin_contaminated else 'no'}")
    print(f"  Gap   : {gap_label}")
    print(f"         below threshold = {'YES → Tier≥2' if gap_min < config.HOMO_LUMO_TIER2_THRESHOLD_EV else 'no'}")
    print(f"  TM    : {'YES → Tier 3' if has_tm else 'no'}")

    return tier, indicators


def compute_mp2_deviations(mf, mol):
    """
    Compute MP2 natural orbital deviations: dev_i = min(n_i, 2 - n_i).

      dev = 0 → doubly occupied or empty (uncorrelated)
      dev = 1 → half-filled (maximally correlated)

    Fix vs original: only catches specific numerical exceptions, not all
    exceptions. A broad 'except Exception' can silently mask import errors,
    memory errors, etc. and give misleading results.
    """
    S = mol.intor("int1e_ovlp")
    evals, evecs = np.linalg.eigh(S)
    mask      = evals > 1e-15
    S_invsqrt = (evecs[:, mask] / np.sqrt(evals[mask])) @ evecs[:, mask].T

    mp2_ok = False
    e_corr = 0.0

    try:
        mymp = pyscf_mp.MP2(mf)
        mymp.verbose = 0
        e_corr, _    = mymp.kernel()
        dm1          = mymp.make_rdm1()

        if isinstance(dm1, (tuple, list)):
            Ca, Cb = np.asarray(mf.mo_coeff[0]), np.asarray(mf.mo_coeff[1])
            dm_ao  = Ca @ dm1[0] @ Ca.T + Cb @ dm1[1] @ Cb.T
        else:
            dm_ao = mf.mo_coeff @ dm1 @ mf.mo_coeff.T

        mp2_ok = True

    except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
        # These are expected numerical failures (e.g. singular overlap,
        # MP2 amplitude divergence for near-degenerate systems).
        warnings.warn(
            f"MP2 failed with: {e}\n"
            f"Falling back to UHF density matrix for deviation computation.\n"
            f"Active space quality may be reduced. "
            f"Consider using a larger basis or manually setting the active space.",
            RuntimeWarning,
        )
        dm_raw = mf.make_rdm1()
        dm_ao  = (dm_raw[0] + dm_raw[1]) if isinstance(dm_raw, tuple) else dm_raw

    # Diagonalize in Löwdin orthogonal basis → natural orbital occupations
    dm_lo = S_invsqrt @ dm_ao @ S_invsqrt.T
    dm_lo = 0.5 * (dm_lo + dm_lo.T)                        # enforce symmetry
    no_occ    = np.clip(np.linalg.eigvalsh(dm_lo)[::-1], 0.0, 2.0)
    deviation = np.minimum(no_occ, 2.0 - no_occ)

    return deviation, no_occ, e_corr, mp2_ok


def find_gap_cutoff(values, min_n, max_n):
    """
    Find the largest gap in sorted deviation values to determine active space size.
    Returns (n_selected, gap_size, k_indices_of_selected).
    """
    values = np.asarray(values, dtype=float)
    n      = len(values)
    min_n  = max(1, min(min_n, n))
    max_n  = min(max_n, n)

    if min_n >= max_n:
        order = np.argsort(-values)
        return min_n, 0.0, list(order[:min_n])

    order    = np.argsort(-values)
    sorted_v = values[order]

    best_gap, best_n = -1.0, min_n
    for k in range(min_n, max_n + 1):
        gap = sorted_v[k - 1] - (sorted_v[k] if k < n else 0.0)
        if gap > best_gap:
            best_gap, best_n = gap, k

    return best_n, float(best_gap), list(order[:best_n])


def lowdin_population(mo_coeff, mo_list, S, ao_labels, n_atoms):
    """
    Löwdin population analysis: compute weight of each active MO on each atom.
    weight[k, atom] = sum over AOs on atom of (S^{1/2} C_k)_j^2
    """
    evals, evecs = np.linalg.eigh(S)
    mask   = evals > 1e-15
    S_sqrt = (evecs[:, mask] * np.sqrt(evals[mask])) @ evecs[:, mask].T

    weights = np.zeros((len(mo_list), n_atoms))
    for k, mo_idx in enumerate(mo_list):
        c_lo = S_sqrt @ mo_coeff[:, mo_idx]
        for ao_j, (atom_idx, *_) in enumerate(ao_labels):
            weights[k, atom_idx] += c_lo[ao_j] ** 2

    return weights


def count_active_electrons(mol, mf, final_mo_list):
    """
    Count electrons in the active space.

    Strategy:
      1. Compute per-MO occupation from UHF alpha+beta MO occupations.
         This operates purely in MO space — no basis mixing.
      2. Core MOs = occupied MOs NOT in active list with occ > CORE_OCC_THRESHOLD.
      3. nel_active = n_total_electrons - 2 × n_core.
      4. Apply bounds: 2 ≤ nel ≤ 2×n_active, enforce even count.

    Fix vs original:
      Original used Löwdin natural orbital occupations (AO basis, indexed 0..n_ao)
      but compared them against final_mo_list (MO indices).
      These index different spaces → the core count was wrong for most systems.
      This version uses mf.mo_occ (MO basis) throughout.
    """
    active_set = set(final_mo_list)

    # UHF: mf.mo_occ has shape (2, n_mo) — alpha and beta separately
    mo_occ_alpha = np.asarray(mf.mo_occ[0])
    mo_occ_beta  = np.asarray(mf.mo_occ[1])
    mo_occ_total = mo_occ_alpha + mo_occ_beta   # 0, 1, or 2 per MO

    # Identify core MOs: outside active space AND nearly doubly occupied
    core_orbs = [
        i for i, occ in enumerate(mo_occ_total)
        if i not in active_set and occ > config.CORE_OCC_THRESHOLD
    ]
    n_core = len(core_orbs)
    nel    = mol.nelectron - 2 * n_core

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print(f"  Total electrons    : {mol.nelectron}")
    print(f"  Core MOs (occ>{config.CORE_OCC_THRESHOLD}): {n_core}  →  {core_orbs}")
    print(f"  Raw active electrons: {nel}")

    # ── Bounds checks ─────────────────────────────────────────────────────────
    max_nel = 2 * len(final_mo_list)

    if nel <= 0:
        raise ValueError(
            f"Active electron count is {nel} ≤ 0.\n"
            f"  Total electrons    : {mol.nelectron}\n"
            f"  Core MOs counted   : {n_core}\n"
            f"  This usually means CORE_OCC_THRESHOLD ({config.CORE_OCC_THRESHOLD}) "
            f"is too low and all electrons are being classified as core.\n"
            f"  Fix: increase CORE_OCC_THRESHOLD in config.py (recommended: 1.95)."
        )

    if nel > max_nel:
        warnings.warn(
            f"Active electrons ({nel}) > 2 × active orbitals ({max_nel}). "
            f"Capping at {max_nel}. This may indicate CORE_OCC_THRESHOLD is too high.",
            RuntimeWarning,
        )
        nel = max_nel

    # Enforce even number (closed-shell embedding assumption)
    if nel % 2 != 0:
        nel -= 1
        print(f"  Adjusted to even   : nel={nel}")

    nel = max(2, nel)
    print(f"  Final active electrons: {nel}")

    return nel


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"[Step 1] Active Space Finder — {config.MOLECULE}")
print(f"{'='*60}")

# ── Build molecule ────────────────────────────────────────────────────────────
mol = gto.M(
    atom    = config.GEOMETRY,
    basis   = config.BASIS,
    charge  = config.CHARGE,
    spin    = config.SPIN,
    verbose = 3,
)

print(f"\n  Molecule  : {config.MOLECULE}")
print(f"  Atoms     : {config.N_ATOMS}  {config.ATOM_SYMS}")
print(f"  Basis     : {config.BASIS}")
print(f"  Charge    : {config.CHARGE}   Spin (2S): {config.SPIN}")
print(f"  Electrons : {mol.nelectron}   AOs: {mol.nao_nr()}")

# ── Phase A: UHF + Classification ────────────────────────────────────────────
print(f"\n── Phase A: UHF + Classification {'─'*20}")
mf = run_uhf(mol)
print(f"\n  UHF energy = {mf.e_tot:.8f} Ha  (converged: {mf.converged})")

if not mf.converged:
    warnings.warn(
        "UHF did not converge. All downstream results are unreliable.",
        RuntimeWarning,
    )

tier, indicators = classify(mol, mf)

print(f"\n  ┌─────────────────────────────────────┐")
print(f"  │  Tier {tier} classification              │")
print(f"  │  TM element : {str(indicators['has_tm']):5s}                  │")
print(f"  │  Spin cont. : {indicators['spin_cont']:.4f}                │")
print(f"  │  Gap (min)  : {indicators['homo_lumo_gap_eV']:.4f} eV             │")
print(f"  └─────────────────────────────────────┘")

# ── Phase B: MP2 Deviations + ASF ────────────────────────────────────────────
print(f"\n── Phase B: MP2 Deviations + ASF {'─'*20}")
deviation, no_occ, e_corr, mp2_ok = compute_mp2_deviations(mf, mol)

print(f"  MP2 used             : {mp2_ok}")
print(f"  MP2 correlation E    : {e_corr:.6f} Ha")
print(f"  Orbitals (dev>0.05)  : {int(np.sum(deviation > 0.05))}")
print(f"  Orbitals (dev>0.10)  : {int(np.sum(deviation > 0.10))}")

asf_p = config.ASF_PARAMS[tier]
print(f"\n  Running ASF (Tier {tier}):")
print(f"    entropy_threshold = {asf_p['entropy_threshold']}")
print(f"    max_norb          = {asf_p['max_norb']}")
print(f"    min_norb          = {asf_p['min_norb']}")

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
print(f"\n  ASF candidates : {n_cand} orbitals")
print(f"  MO indices     : {mo_list}")

if n_cand == 0:
    raise RuntimeError(
        "ASF returned 0 candidates.\n"
        f"Current entropy_threshold={asf_p['entropy_threshold']} (Tier {tier}).\n"
        "Try lowering entropy_threshold in config.ASF_PARAMS."
    )

# ── Phase C: Gap Detection ────────────────────────────────────────────────────
print(f"\n── Phase C: Gap Detection {'─'*27}")

# Get deviation values for the ASF candidate orbitals
cand_devs = np.array([
    deviation[i] if i < len(deviation) else 0.0
    for i in mo_list
])

print(f"\n  Candidate orbital deviations (sorted):")
print(f"  {'MO':>5}  {'dev':>8}  bar")
for mo_idx, dev in sorted(zip(mo_list, cand_devs), key=lambda x: -x[1]):
    bar = "█" * int(dev * 30)
    print(f"  {mo_idx:>5}  {dev:>8.4f}  {bar}")

n_final, gap_val, selected_k = find_gap_cutoff(
    cand_devs, config.GAP_MIN_NORB, config.GAP_MAX_NORB
)
final_mo_list = sorted(mo_list[k] for k in selected_k)

print(f"\n  Gap detected   : {gap_val:.4f}  (at position {n_final})")
print(f"  Selected orbs  : {n_final}  →  {final_mo_list}")

# Count active electrons — uses MF MO occupations, not Löwdin NO occupations
print(f"\n  Counting active electrons:")
nel = count_active_electrons(mol, mf, final_mo_list)

print(f"\n  ┌────────────────────────────────────┐")
print(f"  │  Active space: ({nel}e, {n_final}orb)         │")
print(f"  │  Orbitals    : {str(final_mo_list):<28}│")
print(f"  └────────────────────────────────────┘")

# ── Phase D: Löwdin Population ────────────────────────────────────────────────
print(f"\n── Phase D: Löwdin Population {'─'*23}")
S          = mol.intor("int1e_ovlp")
ao_labels  = mol.ao_labels(fmt=None)
weights    = lowdin_population(mo_coeff, final_mo_list, S, ao_labels, config.N_ATOMS)
dominant_atoms = np.argmax(weights, axis=1).astype(int)

print(f"\n  {'MO':>5}  {'Atom':>6}  {'Symbol':>6}  {'Weight':>8}")
print(f"  {'─'*35}")
for k, mo_idx in enumerate(final_mo_list):
    atom   = dominant_atoms[k]
    weight = weights[k, atom]
    print(f"  {mo_idx:>5}  {atom:>6}  {config.ATOM_SYMS[atom]:>6}  {weight:>8.4f}")

# Correlation strength = mean deviation of selected orbitals
final_devs   = np.array([deviation[i] for i in final_mo_list if i < len(deviation)])
corr_strength = float(np.mean(final_devs)) if len(final_devs) > 0 else 0.0

# ── Final Summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"[Step 1] Summary — {config.MOLECULE}")
print(f"{'='*60}")
print(f"  Tier                : {tier}")
print(f"  MP2 used            : {mp2_ok}")
print(f"  UHF energy          : {mf.e_tot:.8f} Ha")
print(f"  MP2 correlation     : {e_corr:.6f} Ha")
print(f"  Active space        : ({nel}e, {n_final}orb)")
print(f"  Orbitals            : {final_mo_list}")
print(f"  Correlation strength: {corr_strength:.4f}  (0=uncorrelated, 1=max)")
print(f"  Gap alpha           : {indicators.get('gap_alpha_eV', 'N/A')}")
print(f"  Gap beta            : {indicators.get('gap_beta_eV',  'N/A')}")
print(f"{'='*60}")

# ── Save ──────────────────────────────────────────────────────────────────────
results = {
    # Active space
    "nel"              : nel,
    "mo_list"          : final_mo_list,
    "mo_coeff"         : mo_coeff,
    "n_active_orbs"    : n_final,
    # Orbital analysis
    "no_occ"           : no_occ,
    "deviation"        : deviation,
    "lowdin_weights"   : weights,
    "dominant_atoms"   : dominant_atoms,
    # Classification
    "tier"             : tier,
    "indicators"       : indicators,
    "corr_strength"    : corr_strength,
    # Molecule info
    "mol_info": {
        "molecule"     : config.MOLECULE,
        "basis"        : config.BASIS,
        "n_atoms"      : config.N_ATOMS,
        "atom_syms"    : config.ATOM_SYMS,
        "n_electrons"  : mol.nelectron,
        "n_ao"         : mol.nao_nr(),
    },
    # UHF solution (needed by step2 to avoid recomputation)
    "uhf_energy"       : float(mf.e_tot),
    "mp2_energy"       : float(mf.e_tot + e_corr),
    "mp2_ok"           : mp2_ok,
    "mo_coeff_uhf"     : np.asarray(mf.mo_coeff),
    "mo_energy"        : np.asarray(mf.mo_energy),
    "mo_occ"           : np.asarray(mf.mo_occ),
    "converged"        : mf.converged,
}

with open(config.STEP1_FILE, "wb") as f:
    pickle.dump(results, f)

print(f"\n[Step 1] ✓ Saved → {config.STEP1_FILE}")