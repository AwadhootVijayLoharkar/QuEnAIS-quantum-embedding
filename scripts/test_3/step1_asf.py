"""
Step 1 — Flexible Tiered Active Space Finder
============================================

What this script does:
  Phase A: Run UHF (with Newton fallback for open-shell systems)
           Classify the molecule into a complexity tier (1/2/3)

  Phase B: Run ASF via find_from_scf with tier-appropriate parameters
           ASF internally uses block2 DMRG for accuracy
           We request a BROAD candidate pool (low threshold) so that
           Phase C can decide the cutoff adaptively

  Phase C: Adaptive gap detection
           Compute MP2 natural orbital deviation proxy: dev_i = min(n_i, 2-n_i)
             dev = 0 → orbital always full or always empty → boring
             dev = 1 → orbital half-filled → maximally correlated
           Find the LARGEST natural gap in the dev spectrum within [MIN, MAX]
           → No fixed threshold; works for any molecule automatically

  Phase D: Löwdin population analysis
           Maps each active orbital to its dominant atom
           More reliable than Mulliken (always non-negative, basis-stable)

  Phase E: Rich molecular scoring
           Computes a score vector useful for the graph transformer pipeline

Requires: config.py
Saves:    results/step1_asf.pkl
Runtime:  30 sec (Tier 1) to 5 min (Tier 3)
"""

import os
import sys
import pickle
import numpy as np
import pyscf.scf.hf as _pyscf_hf_base

import config

# ── Cache check ───────────────────────────────────────────────────────────────
FORCE_RERUN = True   # set True to redo this step

os.makedirs(config.RESULTS_DIR, exist_ok=True)

if os.path.exists(config.STEP1_FILE) and not FORCE_RERUN:
    print(f"[Step 1] Cached results at {config.STEP1_FILE}")
    print("  Set FORCE_RERUN = True to rerun.")
    sys.exit(0)

# ── Environment (must be set before any PySCF / ASF import) ──────────────────
os.environ["BLOCKEXE"]            = config.BLOCKEXE_WRAPPER
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_DEBUG_CPU_TYPE"]  = "5"

from pyscf.dmrgscf import dmrgci
dmrgci.settings.BLOCKEXE = config.BLOCKEXE_WRAPPER

from pyscf import gto, scf as pyscf_scf, mp as pyscf_mp
from asf.wrapper import find_from_scf


# ═════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═════════════════════════════════════════════════════════════════════════════

def classify_molecule(mol, mf):
    """
    Determine molecule complexity class and computational tier.

    Uses three fast indicators (all free from the UHF calculation):
      1. Presence of d/f-block elements (TM, lanthanide, actinide)
      2. UHF spin contamination ratio
      3. Alpha HOMO-LUMO gap in eV

    Returns
    -------
    complexity_class : int   1 = simple, 2 = moderate, 3 = strongly correlated
    tier             : int   1, 2, or 3
    indicators       : dict  all computed values for rich score output
    """
    # ── Indicator 1: TM / f-element ───────────────────────────────────────────
    has_tm = any(mol.atom_symbol(i) in config.TM_ELEMENTS
                 for i in range(mol.natm))

    # ── Indicator 2: Spin contamination ───────────────────────────────────────
    if hasattr(mf, 'spin_square'):
        s2_actual, _ = mf.spin_square()
        n_unpaired   = mol.spin                      # 2S (integer)
        s_val        = n_unpaired / 2.0
        s2_expected  = s_val * (s_val + 1.0)
        # For singlet (s_val=0): use 0.75 denominator to avoid divide-by-zero
        spin_cont    = s2_actual / max(s2_expected, 0.75)
    else:
        s2_actual = 0.0
        spin_cont = 1.0

    # ── Indicator 3: HOMO-LUMO gap (eV) ───────────────────────────────────────
    # Use alpha channel for both UHF and RHF
    if isinstance(mf.mo_energy, (tuple, list)):
        mo_e   = np.asarray(mf.mo_energy[0])   # alpha
        mo_occ = np.asarray(mf.mo_occ[0])
    else:
        mo_e   = np.asarray(mf.mo_energy)
        mo_occ = np.asarray(mf.mo_occ)

    occ_mask  = mo_occ > 0.5
    virt_mask = mo_occ < 0.5

    if occ_mask.any() and virt_mask.any():
        homo_e    = mo_e[occ_mask][-1]
        lumo_e    = mo_e[virt_mask][0]
        hl_gap_ev = (lumo_e - homo_e) * 27.2114   # Hartree → eV
    else:
        hl_gap_ev = 10.0   # essentially closed-shell with no virtual

    # ── Indicator 4: Singly occupied MOs (UHF) ────────────────────────────────
    if isinstance(mf.mo_occ, (tuple, list)):
        occ_a = np.asarray(mf.mo_occ[0])
        occ_b = np.asarray(mf.mo_occ[1])
        n_somo = int(np.sum(np.abs(occ_a - occ_b) > 0.5))
    else:
        n_somo = 0

    indicators = {
        'has_tm'             : bool(has_tm),
        'spin_contamination' : float(spin_cont),
        's2_actual'          : float(s2_actual),
        'homo_lumo_gap_eV'   : float(hl_gap_ev),
        'n_somo'             : int(n_somo),
    }

    # ── Classify ──────────────────────────────────────────────────────────────
    # Any TM/f-element → always Tier 3 (needs full DMRG-based treatment)
    if has_tm:
        return 3, 3, indicators

    # Open-shell or small gap → Tier 2
    if (spin_cont > config.SPIN_CONTAMINATION_TIER2_THRESHOLD or
            hl_gap_ev < config.HOMO_LUMO_TIER2_THRESHOLD_EV or
            n_somo    > 2):
        return 2, 2, indicators

    # Otherwise simple closed-shell
    return 1, 1, indicators


def compute_mp2_deviation(mf, mol):
    """
    Compute MP2 natural orbital occupation DEVIATIONS as an entropy proxy.

    Deviation:  dev_i = min(n_i, 2 - n_i)
      dev = 0.0  →  orbital always full or always empty  →  uncorrelated
      dev = 1.0  →  orbital half-filled                  →  maximally correlated

    Works for both RHF and UHF. Falls back to UHF density if MP2 fails
    (UHF SOMOs will still have fractional-looking occupations in the
    spin-averaged DM, so the proxy remains useful).

    The NOs are computed in the Löwdin orthogonal basis for numerical
    stability (avoids issues with near-linear-dependent AO basis sets).

    Returns
    -------
    deviation  : (n_MO,) array  sorted descending (highest deviation first)
    no_occ     : (n_MO,) array  natural orbital occupations (descending)
    e_corr_mp2 : float          MP2 correlation energy (0 if MP2 failed)
    """
    S = mol.intor('int1e_ovlp')

    # S^{-1/2} for Löwdin basis
    evals_S, evecs_S = np.linalg.eigh(S)
    mask      = evals_S > 1e-15
    S_invsqrt = (evecs_S[:, mask] / np.sqrt(evals_S[mask])) @ evecs_S[:, mask].T

    e_corr_mp2 = 0.0

    try:
        mymp = pyscf_mp.MP2(mf)
        mymp.verbose = 0
        e_corr_mp2, _ = mymp.kernel()
        dm1            = mymp.make_rdm1()   # MO basis; tuple for UHF

        # Build total 1-RDM in AO basis
        if isinstance(dm1, (tuple, list)):
            # UHF: alpha and beta have separate MO spaces
            dm1a_mo, dm1b_mo = dm1
            C_a     = np.asarray(mf.mo_coeff[0])
            C_b     = np.asarray(mf.mo_coeff[1])
            dm1_ao  = C_a @ dm1a_mo @ C_a.T + C_b @ dm1b_mo @ C_b.T
        else:
            # RHF: single MO space
            dm1_ao  = mf.mo_coeff @ dm1 @ mf.mo_coeff.T

        print(f"    MP2 correlation energy : {e_corr_mp2:.6f} Ha")

    except Exception as exc:
        print(f"    [MP2] Failed ({exc}) — using UHF density matrix as fallback")
        dm_raw = mf.make_rdm1()
        dm1_ao = (dm_raw[0] + dm_raw[1]) if isinstance(dm_raw, tuple) else dm_raw

    # Diagonalize in Löwdin basis → natural orbital occupations
    dm1_lo = S_invsqrt @ dm1_ao @ S_invsqrt.T
    dm1_lo = 0.5 * (dm1_lo + dm1_lo.T)   # symmetrize (numerical noise)

    no_occ_raw, _ = np.linalg.eigh(dm1_lo)

    # Sort descending, clamp to [0, 2]
    no_occ   = np.clip(no_occ_raw[::-1].copy(), 0.0, 2.0)
    deviation = np.minimum(no_occ, 2.0 - no_occ)

    return deviation, no_occ, e_corr_mp2


def find_gap_cutoff(values, min_n, max_n):
    """
    Find the LARGEST natural gap in a set of values (treated as sorted
    descending) to decide how many to keep.

    The algorithm:
      Sort values descending → v[0] >= v[1] >= ... >= v[N-1]
      For each n in [min_n, max_n]:
        gap_n = v[n-1] - v[n]   (difference at the n/n+1 boundary)
      Select n* = argmax(gap_n)   → keep the top n* values

    Physical interpretation:
      The gap marks the natural boundary between the strongly-correlated
      group and the weakly-correlated group. No fixed threshold needed.

    Parameters
    ----------
    values : array-like  one value per candidate orbital (higher = more correlated)
    min_n  : int         minimum number to select
    max_n  : int         maximum number to select

    Returns
    -------
    n_select         : int    number of orbitals selected
    gap_value        : float  size of the largest gap found
    selected_indices : list   indices into `values` of the selected orbitals
    """
    values  = np.asarray(values, dtype=float)
    n_avail = len(values)

    min_n = max(1, min(min_n, n_avail))
    max_n = min(max_n, n_avail)

    # Degenerate cases
    if min_n >= max_n or n_avail <= min_n:
        order = np.argsort(-values)
        return min_n, 0.0, list(order[:min_n])

    # Sort descending
    order       = np.argsort(-values)
    sorted_vals = values[order]

    best_gap = -1.0
    best_n   = min_n

    for n in range(min_n, max_n + 1):
        # Gap AFTER including n orbitals (between position n-1 and n)
        if n < n_avail:
            gap = float(sorted_vals[n - 1] - sorted_vals[n])
        else:
            # Past end of array: gap to zero
            gap = float(sorted_vals[n - 1])

        if gap > best_gap:
            best_gap = gap
            best_n   = n

    return best_n, float(best_gap), list(order[:best_n])


def lowdin_population(mo_coeff, mo_list, S, ao_labels, n_atoms):
    """
    Löwdin population analysis: compute atom-resolved weight of each active MO.

    Method:
      c_lo = S^{1/2} @ c_AO          (Löwdin-orthogonalized coefficients)
      weight[orbital, atom] = Σ_{μ ∈ atom} c_lo[μ]²

    Advantages over Mulliken:
      - Always non-negative (Mulliken can give negative populations)
      - More stable across different basis sets
      - Critical for molecules with unusual geometry from graph transformer

    Returns
    -------
    weights : (n_active, n_atoms)  weights[i,j] = fraction of orbital i on atom j
    """
    evals_S, evecs_S = np.linalg.eigh(S)
    mask   = evals_S > 1e-15
    S_sqrt = (evecs_S[:, mask] * np.sqrt(evals_S[mask])) @ evecs_S[:, mask].T

    n_active = len(mo_list)
    weights  = np.zeros((n_active, n_atoms))

    for k, mo_idx in enumerate(mo_list):
        c    = mo_coeff[:, mo_idx]   # AO expansion of this MO
        c_lo = S_sqrt @ c            # Löwdin transformation
        for ao_j, (atom_idx, *_) in enumerate(ao_labels):
            weights[k, atom_idx] += c_lo[ao_j] ** 2

    return weights


def count_active_electrons(mol, no_occ, final_mo_list):
    """
    Count electrons in the active space.

    Method: total electrons minus electrons in doubly-occupied core MOs.
    Core MOs are those NOT in the active list with occupation > CORE_OCC_THRESHOLD.

    This is more reliable than summing NO occupations directly because
    near-core orbitals can have occupation slightly below 2.0 but are
    physically still doubly occupied.

    Returns
    -------
    nel : int  active electron count (always even; singlet target)
    """
    active_set = set(final_mo_list)

    n_core = sum(
        1 for i, occ in enumerate(no_occ)
        if (i not in active_set) and (occ > config.CORE_OCC_THRESHOLD)
    )

    nel = mol.nelectron - 2 * n_core

    # Cross-check with NO occupation sum for sanity
    nel_occ = int(round(sum(no_occ[i] for i in final_mo_list
                            if i < len(no_occ))))
    if abs(nel - nel_occ) > 3:
        print(f"  WARNING: nel mismatch — from_core={nel}, from_occ={nel_occ}")
        print(f"           Using from_occ={nel_occ}")
        nel = nel_occ

    # Clamp to valid range and ensure even (singlet target for SQD)
    nel = max(2, min(nel, 2 * len(final_mo_list)))
    if nel % 2 != 0:
        nel -= 1

    return nel


# ═════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 62)
print(f"[Step 1] Flexible Active Space Finder — {config.MOLECULE}")
print("=" * 62)

# ── Build PySCF molecule ──────────────────────────────────────────────────────
mol = gto.M(
    atom    = config.GEOMETRY,
    basis   = config.BASIS,
    charge  = 0,
    spin    = 0,
    verbose = 3,
)

# ═════════════════════════════════════════════════════════════════════════════
# Phase A: UHF + Molecule Classification
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Phase A: UHF + Molecule Classification ──")

# SCF patch: DIIS with level-shift first; Newton solver if DIIS fails.
# Needed for open-shell Fe (and any strongly correlated TM complex).
_orig_kernel = _pyscf_hf_base.SCF.kernel

def _newton_fallback_kernel(self, dm0=None, **kwargs):
    """DIIS + level-shift first; Newton if DIIS fails."""
    self.max_cycle   = max(getattr(self, "max_cycle",   50), 400)
    self.level_shift = max(getattr(self, "level_shift", 0.0), 0.5)
    result = _orig_kernel(self, dm0=dm0, **kwargs)
    if not self.converged:
        print("    [SCF] DIIS not converged → Newton solver...")
        try:
            nw           = self.newton()
            nw.max_cycle = 400
            nw.kernel(self.mo_coeff)
            if nw.converged:
                print(f"    [SCF] ✓ Newton: E = {nw.e_tot:.8f} Ha")
                for attr in ("e_tot","mo_coeff","mo_energy","mo_occ","converged"):
                    setattr(self, attr, getattr(nw, attr))
            else:
                print("    [SCF] Newton also did not converge.")
        except Exception as exc:
            print(f"    [SCF] Newton error: {exc}")
    return result

_pyscf_hf_base.SCF.kernel = _newton_fallback_kernel

print("\n  Running UHF...")
mf = pyscf_scf.UHF(mol)
mf.kernel()

_pyscf_hf_base.SCF.kernel = _orig_kernel   # restore for now

if not mf.converged:
    print("  WARNING: UHF did not converge. Proceeding — results may be unreliable.")

print(f"  UHF energy  = {mf.e_tot:.8f} Ha")
print(f"  Converged   = {mf.converged}")

# Classify
complexity_class, tier, indicators = classify_molecule(mol, mf)

print(f"\n  Classification result:")
print(f"    Complexity class    : {complexity_class}  "
      f"(1=simple organic, 2=moderate, 3=strongly correlated)")
print(f"    Computational tier  : {tier}")
print(f"    Has TM/f-element   : {indicators['has_tm']}")
print(f"    ⟨S²⟩               : {indicators['s2_actual']:.3f}  "
      f"(contamination ratio = {indicators['spin_contamination']:.2f})")
print(f"    HOMO-LUMO gap      : {indicators['homo_lumo_gap_eV']:.2f} eV")
print(f"    n_SOMO             : {indicators['n_somo']}")

asf_p = config.ASF_PARAMS[tier]
print(f"\n  → Using ASF parameters for Tier {tier}: {asf_p}")

# ═════════════════════════════════════════════════════════════════════════════
# Phase B: MP2 deviation proxy + ASF candidate pool
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Phase B: MP2 Deviations + ASF Candidate Pool ──")

# MP2 NOs and deviation proxy
# These are computed from the SAME mf that we pass to ASF (find_from_scf).
# ASF also computes MP2 NOs from the same mf, so the indexing is consistent:
#   deviation[k] corresponds to column k of active_space.mo_coeff
print("\n  Computing MP2 natural orbital deviations...")
deviation, no_occ, e_corr_mp2 = compute_mp2_deviation(mf, mol)

n_frac_occ = int(np.sum(deviation > 0.05))
print(f"  Orbitals with deviation > 0.05 : {n_frac_occ}")

# Run ASF with tier-appropriate parameters
# We use entropy_threshold = 0.01 (very low) to get a BROAD candidate pool.
# Phase C will apply gap detection to select the final subset.
# ASF uses block2 DMRG internally (block2 is already configured via BLOCKEXE).
# Keep the Newton patch active during ASF's internal SCF stability analysis.
print(f"\n  Running ASF (Tier {tier}, entropy_threshold={asf_p['entropy_threshold']}, "
      f"max_norb={asf_p['max_norb']})...")

_pyscf_hf_base.SCF.kernel = _newton_fallback_kernel   # re-apply for ASF internals

active_space = find_from_scf(
    mf,
    entropy_threshold = asf_p["entropy_threshold"],
    max_norb          = asf_p["max_norb"],
    min_norb          = asf_p["min_norb"],
    verbose           = True,
)

_pyscf_hf_base.SCF.kernel = _orig_kernel   # restore after ASF

nel_asf    = active_space.nel
mo_list    = list(active_space.mo_list)
mo_coeff   = active_space.mo_coeff
n_cand     = len(mo_list)

print(f"\n  ASF candidate pool: {n_cand} orbitals → {mo_list}")
print(f"  ASF nel            : {nel_asf}")

if n_cand == 0:
    raise RuntimeError(
        "ASF returned 0 candidate orbitals.\n"
        f"Try lowering entropy_threshold in config.ASF_PARAMS[{tier}] to 0.001"
    )

# ═════════════════════════════════════════════════════════════════════════════
# Phase C: Adaptive Gap Detection
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Phase C: Adaptive Gap Detection ──")

# Get deviation values for the candidate orbitals.
# mo_list[k] is the column index of the k-th candidate in mo_coeff.
# Our deviation array (from compute_mp2_deviation) uses the SAME indexing
# because both ASF and compute_mp2_deviation use the same mf for MP2.
cand_deviations = np.array([
    deviation[i] if i < len(deviation) else 0.0
    for i in mo_list
])

print("\n  Candidate orbital deviation spectrum:")
print(f"  {'MO':>5}  {'deviation':>10}  {'(higher = more correlated)'}")
print(f"  {'─'*45}")
for mo_idx, dev in sorted(zip(mo_list, cand_deviations), key=lambda x: -x[1]):
    bar = "█" * int(dev * 20)
    print(f"  {mo_idx:5d}  {dev:10.4f}  {bar}")

# Find the natural gap
n_final, gap_value, selected_k = find_gap_cutoff(
    cand_deviations,
    config.GAP_MIN_NORB,
    config.GAP_MAX_NORB
)

# Build final orbital list (sorted for reproducibility)
final_mo_list = sorted(mo_list[k] for k in selected_k)

# Compute active electron count
nel_final = count_active_electrons(mol, no_occ, final_mo_list)

print(f"\n  Gap detection result:")
print(f"    Largest gap in deviation spectrum : {gap_value:.4f}")
print(f"    Final active space                : {n_final} orbitals")
print(f"    Final mo_list                     : {final_mo_list}")
print(f"    Active electrons (nel)            : {nel_final}")

# Compare with ASF's original threshold-based selection
# (informational — we always use gap detection result)
asf_selected  = set(mo_list)
gap_selected  = set(final_mo_list)
added_orbs    = gap_selected - asf_selected
removed_orbs  = asf_selected - gap_selected

print(f"\n  vs ASF threshold selection ({n_cand} orbs):")
if added_orbs:
    print(f"    Added by gap detection   : {sorted(added_orbs)}")
if removed_orbs:
    print(f"    Removed by gap detection : {sorted(removed_orbs)}")
if not added_orbs and not removed_orbs:
    print(f"    Perfect agreement ✓")

# ═════════════════════════════════════════════════════════════════════════════
# Phase D: Löwdin Population Analysis
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Phase D: Löwdin Population Analysis ──")

S         = mol.intor("int1e_ovlp")
ao_labels = mol.ao_labels(fmt=None)

lowdin_weights = lowdin_population(
    mo_coeff, final_mo_list, S, ao_labels, config.N_ATOMS
)

dominant_atoms  = np.argmax(lowdin_weights, axis=1).astype(int)
active_per_atom = np.bincount(dominant_atoms, minlength=config.N_ATOMS)
most_active_atom = int(np.argmax(active_per_atom))

# Fraction of active orbitals on transition metal atoms
tm_atom_idx    = [i for i in range(config.N_ATOMS)
                  if config.ATOM_SYMS[i] in config.TM_ELEMENTS]
metal_fraction = (float(sum(active_per_atom[i] for i in tm_atom_idx))
                  / max(1, n_final))

print(f"\nOrbital → Atom Löwdin mapping:")
print(f"  {'MO':>5}  {'→ atom':>7}  {'dev':>6}  {'Löwdin weights per atom'}")
print(f"  {'─'*70}")

for k, mo_idx in enumerate(final_mo_list):
    da  = int(dominant_atoms[k])
    dev = deviation[mo_idx] if mo_idx < len(deviation) else 0.0
    wts = "  ".join(
        f"{config.ATOM_SYMS[j]}:{lowdin_weights[k, j]:+.3f}"
        for j in range(config.N_ATOMS)
    )
    print(f"  {mo_idx:5d}  → {da} ({config.ATOM_SYMS[da]:2s})  "
          f"{dev:6.3f}  |  {wts}")

print(f"\nActive orbital count per atom:")
for i, (sym, cnt) in enumerate(zip(config.ATOM_SYMS, active_per_atom)):
    bar = "█" * int(cnt * 4)
    print(f"  Atom {i} ({sym:2s}): {cnt:2d}  {bar}")

print(f"\n  Most correlated atom : {most_active_atom} "
      f"({config.ATOM_SYMS[most_active_atom]})")
print(f"  Metal fraction       : {metal_fraction:.2f}  "
      f"(fraction of active orbs on TM atom(s))")

# ═════════════════════════════════════════════════════════════════════════════
# Phase E: Rich Molecular Scoring
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Phase E: Molecular Score Vector ──")

final_devs = np.array([deviation[i] for i in final_mo_list
                       if i < len(deviation)])

scores = {
    # ── Complexity classification ──────────────────────────────────────────
    "complexity_class"        : int(complexity_class),
    "tier_used"               : int(tier),

    # ── Electronic structure indicators (from UHF, free) ──────────────────
    "uhf_energy_Ha"           : float(mf.e_tot),
    "homo_lumo_gap_eV"        : float(indicators["homo_lumo_gap_eV"]),
    "spin_contamination"      : float(indicators["spin_contamination"]),
    "s2_actual"               : float(indicators["s2_actual"]),
    "has_tm"                  : bool(indicators["has_tm"]),
    "n_somo"                  : int(indicators["n_somo"]),

    # ── MP2 correlation indicators ─────────────────────────────────────────
    "mp2_correlation_energy"  : float(e_corr_mp2),
    "n_frac_occ"              : int(n_frac_occ),   # orbitals with dev > 0.05

    # ── Active space quality ───────────────────────────────────────────────
    "n_active_electrons"      : int(nel_final),
    "n_active_orbitals"       : int(n_final),
    "entropy_gap"             : float(gap_value),

    # Correlation strength (via deviation proxy):
    #   mean ≈ 0    → weakly correlated molecule (Class 1)
    #   mean ≈ 0.5  → strongly correlated (Class 3, e.g. Fe complexes)
    "correlation_strength"    : float(np.mean(final_devs)) if len(final_devs) > 0 else 0.0,
    "max_correlation"         : float(np.max(final_devs))  if len(final_devs) > 0 else 0.0,
    "std_correlation"         : float(np.std(final_devs))  if len(final_devs) > 0 else 0.0,

    # Orbitals with dev > 0.3 → "strongly" correlated (not just weakly fractional)
    "n_strongly_correlated"   : int(np.sum(final_devs > 0.3)),

    # ── Atom mapping ──────────────────────────────────────────────────────
    "most_active_atom"        : int(most_active_atom),
    "metal_fraction"          : float(metal_fraction),
}

print(f"\n  {'Metric':<30} {'Value'}")
print(f"  {'─'*50}")
for key, val in scores.items():
    if isinstance(val, float):
        print(f"  {key:<30} {val:.4f}")
    else:
        print(f"  {key:<30} {val}")

# ═════════════════════════════════════════════════════════════════════════════
# Final Summary
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*62}")
print(f"[Step 1] Summary for {config.MOLECULE}")
print(f"  Complexity / Tier    : {complexity_class} / {tier}")
print(f"  Active space         : {nel_final}e in {n_final} orbitals")
print(f"  Active mo_list       : {final_mo_list}")
print(f"  Most correlated atom : {most_active_atom} ({config.ATOM_SYMS[most_active_atom]})")
print(f"  Entropy gap          : {gap_value:.4f}  "
      f"({'clear' if gap_value > 0.05 else 'ambiguous'} separation)")
print(f"  Correlation strength : {scores['correlation_strength']:.3f}")
print(f"{'='*62}")

# ═════════════════════════════════════════════════════════════════════════════
# Save
# ═════════════════════════════════════════════════════════════════════════════
results = {
    # ── Core output (consumed by step2_hamiltonian.py) ────────────────────
    "nel"              : nel_final,
    "mo_list"          : final_mo_list,
    "mo_coeff"         : mo_coeff,          # full (n_AO × n_MO) MP2 NO matrix
    "n_active_orbs"    : n_final,

    # ── MP2 NOs (for diagnostics and potential reuse) ─────────────────────
    "no_occ"           : no_occ,            # all NO occupations, sorted desc
    "deviation"        : deviation,         # all deviations, same ordering

    # ── Löwdin population results ─────────────────────────────────────────
    "lowdin_weights"   : lowdin_weights,    # (n_active, n_atoms)
    "dominant_atoms"   : dominant_atoms,
    "active_per_atom"  : active_per_atom,
    "most_active_atom" : most_active_atom,

    # ── Molecular score vector (pipeline scoring) ─────────────────────────
    "scores"           : scores,

    # ── Molecule metadata ─────────────────────────────────────────────────
    "mol_info"         : {
        "molecule"     : config.MOLECULE,
        "basis"        : config.BASIS,
        "n_atoms"      : config.N_ATOMS,
        "atom_syms"    : config.ATOM_SYMS,
        "n_electrons"  : mol.nelectron,
        "n_ao"         : mol.nao_nr(),
    },
}

with open(config.STEP1_FILE, "wb") as fh:
    pickle.dump(results, fh)

print(f"\n[Step 1] ✓ Saved → {config.STEP1_FILE}")