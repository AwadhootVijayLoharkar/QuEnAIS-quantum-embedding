# ASF.py — test7 — Active Space Finder
"""
Same five-phase logic as test5's ASF.py (UHF + tier classification -> MP2
deviation proxy -> adaptive gap detection -> Loewdin population), with one
functional change:

CHANGE FROM test5: the full spin-separated MP2 1-RDM (alpha, beta, and
their sum, in AO basis) is now saved into step1_asf.pkl. test5 computed
this DM here, threw away everything except the diagonal (`no_occ`), and
then DMET.py recomputed the SAME MP2 calculation from scratch just to get
the DM back. Saving it here removes that redundant O(N^5) MP2 solve from
Step 2 entirely -- DMET.py's config.DMET_REFERENCE="mp2" path now just
reads it out of this pickle.

Rename to ASF.py when you drop this into your own test7/ folder.
"""

import os
import sys
import pickle
import argparse
import warnings
import numpy as np

import config

parser = argparse.ArgumentParser(description="Step 1: Active Space Finder")
parser.add_argument("--force", action="store_true")
args = parser.parse_args()
FORCE_RERUN = args.force

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


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def run_uhf(mol):
    mf = scf.UHF(mol)
    mf.max_cycle   = 400
    mf.level_shift = 0.5
    mf.kernel()

    if not mf.converged:
        print("  DIIS did not converge -> trying Newton solver...")
        nw = mf.newton()
        nw.max_cycle = 400
        nw.kernel(mf.mo_coeff)
        if nw.converged:
            for attr in ("e_tot", "mo_coeff", "mo_energy", "mo_occ", "converged"):
                setattr(mf, attr, getattr(nw, attr))
        else:
            warnings.warn(
                "UHF did not converge with DIIS or Newton. Results may be "
                "unreliable.", RuntimeWarning,
            )
    return mf


def classify(mol, mf):
    has_tm = any(mol.atom_symbol(i) in config.TM_ELEMENTS for i in range(mol.natm))

    s2, _ = mf.spin_square()
    S_val = mol.spin / 2.0
    s_expected = S_val * (S_val + 1.0)
    is_singlet = (mol.spin == 0)

    if is_singlet:
        spin_cont = float(s2)
        spin_contaminated = spin_cont > config.SPIN_CONTAMINATION_SINGLET_THRESHOLD
    else:
        spin_cont = float(s2 / s_expected)
        spin_contaminated = spin_cont > config.SPIN_CONTAMINATION_TIER2_THRESHOLD

    gaps, gap_min = {}, 10.0
    for label, ch in [("alpha", 0), ("beta", 1)]:
        mo_e   = np.asarray(mf.mo_energy[ch])
        mo_occ = np.asarray(mf.mo_occ[ch])
        occ_e  = mo_e[mo_occ > 0.5]
        vir_e  = mo_e[mo_occ < 0.5]
        if len(occ_e) > 0 and len(vir_e) > 0:
            gaps[label] = float((vir_e[0] - occ_e[-1]) * config.HARTREE_TO_EV)
    if gaps:
        gap_min = min(gaps.values())

    indicators = {
        "has_tm": has_tm, "is_singlet": is_singlet, "s2": float(s2),
        "s_expected": float(s_expected), "spin_cont": float(spin_cont),
        "spin_contaminated": spin_contaminated, "homo_lumo_gap_eV": float(gap_min),
        "gap_alpha_eV": gaps.get("alpha"), "gap_beta_eV": gaps.get("beta"),
    }

    if has_tm:
        tier = 3
    elif spin_contaminated or gap_min < config.HOMO_LUMO_TIER2_THRESHOLD_EV:
        tier = 2
    else:
        tier = 1

    print(f"  Spin  : {spin_cont:.4f}  contaminated={spin_contaminated}")
    print(f"  Gap   : min={gap_min:.4f} eV")
    print(f"  TM    : {has_tm}  ->  Tier {tier}")
    return tier, indicators


def compute_mp2_deviations(mf, mol):
    """
    Returns (deviation, no_occ, e_corr, mp2_ok, dm_ao_alpha, dm_ao_beta).

    NEW vs test5: returns the spin-separated AO-basis DMs (not just the
    combined total) so DMET.py can use them directly with no recompute.
    """
    S = mol.intor("int1e_ovlp")
    evals, evecs = np.linalg.eigh(S)
    mask = evals > 1e-15
    S_invsqrt = (evecs[:, mask] / np.sqrt(evals[mask])) @ evecs[:, mask].T

    mp2_ok, e_corr = False, 0.0
    Ca, Cb = np.asarray(mf.mo_coeff[0]), np.asarray(mf.mo_coeff[1])

    try:
        mymp = pyscf_mp.MP2(mf)
        mymp.verbose = 0
        e_corr, _ = mymp.kernel()
        dm1 = mymp.make_rdm1()

        if isinstance(dm1, (tuple, list)):
            dm_ao_alpha = Ca @ dm1[0] @ Ca.T
            dm_ao_beta  = Cb @ dm1[1] @ Cb.T
        else:
            dm_ao_alpha = 0.5 * (Ca @ dm1 @ Ca.T)
            dm_ao_beta  = 0.5 * (Cb @ dm1 @ Cb.T)
        mp2_ok = True

    except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
        warnings.warn(
            f"MP2 failed with: {e}\nFalling back to UHF density matrix.",
            RuntimeWarning,
        )
        dm_raw = mf.make_rdm1()
        if isinstance(dm_raw, (tuple, list)):
            dm_ao_alpha, dm_ao_beta = np.asarray(dm_raw[0]), np.asarray(dm_raw[1])
        else:
            dm_ao_alpha = dm_ao_beta = 0.5 * np.asarray(dm_raw)

    dm_ao = dm_ao_alpha + dm_ao_beta
    dm_lo = S_invsqrt @ dm_ao @ S_invsqrt.T
    dm_lo = 0.5 * (dm_lo + dm_lo.T)

    C = Ca
    dm_mo = C.T @ S @ dm_ao @ S @ C
    no_occ = np.clip(np.diag(dm_mo), 0.0, 2.0)
    deviation = np.minimum(no_occ, 2.0 - no_occ)

    return deviation, no_occ, e_corr, mp2_ok, dm_ao_alpha, dm_ao_beta


def find_gap_cutoff(values, min_n, max_n):
    values = np.asarray(values, dtype=float)
    n = len(values)
    min_n, max_n = max(1, min(min_n, n)), min(max_n, n)
    if min_n >= max_n:
        order = np.argsort(-values)
        return min_n, 0.0, list(order[:min_n])
    order = np.argsort(-values)
    sorted_v = values[order]
    best_gap, best_n = -1.0, min_n
    for k in range(min_n, max_n + 1):
        gap = sorted_v[k - 1] - (sorted_v[k] if k < n else 0.0)
        if gap > best_gap:
            best_gap, best_n = gap, k
    return best_n, float(best_gap), list(order[:best_n])


def lowdin_population(mo_coeff, mo_list, S, ao_labels, n_atoms):
    evals, evecs = np.linalg.eigh(S)
    mask = evals > 1e-15
    S_sqrt = (evecs[:, mask] * np.sqrt(evals[mask])) @ evecs[:, mask].T
    weights = np.zeros((len(mo_list), n_atoms))
    for k, mo_idx in enumerate(mo_list):
        c_lo = S_sqrt @ mo_coeff[:, mo_idx]
        for ao_j, (atom_idx, *_) in enumerate(ao_labels):
            weights[k, atom_idx] += c_lo[ao_j] ** 2
    return weights


def count_active_electrons(mol, mf, final_mo_list):
    active_set = set(final_mo_list)
    mo_occ_total = np.asarray(mf.mo_occ[0]) + np.asarray(mf.mo_occ[1])
    core_orbs = [i for i, occ in enumerate(mo_occ_total)
                 if i not in active_set and occ > config.CORE_OCC_THRESHOLD]
    n_core = len(core_orbs)
    nel = mol.nelectron - 2 * n_core

    print(f"  Total electrons: {mol.nelectron}  Core MOs: {n_core}  Raw active: {nel}")

    max_nel = 2 * len(final_mo_list)
    if nel <= 0:
        raise ValueError(
            f"Active electron count is {nel} <= 0. Increase "
            f"CORE_OCC_THRESHOLD in config.py."
        )
    if nel > max_nel:
        warnings.warn(f"Active electrons ({nel}) > 2 x active orbitals "
                      f"({max_nel}). Capping.", RuntimeWarning)
        nel = max_nel
    if nel % 2 != 0:
        nel -= 1
    nel = max(2, nel)
    print(f"  Final active electrons: {nel}")
    return nel


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}\n[Step 1] Active Space Finder — {config.MOLECULE}\n{'='*60}")

mol = gto.M(atom=config.GEOMETRY, basis=config.BASIS,
            charge=config.CHARGE, spin=config.SPIN, verbose=3)
print(f"  Atoms: {config.N_ATOMS} {config.ATOM_SYMS}  Basis: {config.BASIS}")
print(f"  Electrons: {mol.nelectron}  AOs: {mol.nao_nr()}")

print(f"\n-- Phase A: UHF + Classification --")
mf = run_uhf(mol)
print(f"  UHF energy = {mf.e_tot:.8f} Ha (converged: {mf.converged})")
if not mf.converged:
    warnings.warn("UHF did not converge. Downstream results unreliable.", RuntimeWarning)
tier, indicators = classify(mol, mf)

print(f"\n-- Phase B: MP2 Deviations + ASF --")
deviation, no_occ, e_corr, mp2_ok, dm_ao_alpha_mp2, dm_ao_beta_mp2 = \
    compute_mp2_deviations(mf, mol)
print(f"  MP2 used: {mp2_ok}  E_corr: {e_corr:.6f} Ha")

asf_p = config.ASF_PARAMS[tier]
print(f"  ASF (Tier {tier}): entropy_threshold={asf_p['entropy_threshold']}, "
      f"max_norb={asf_p['max_norb']}, min_norb={asf_p['min_norb']}")

active_space = find_from_scf(
    mf, entropy_threshold=asf_p["entropy_threshold"],
    max_norb=asf_p["max_norb"], min_norb=asf_p["min_norb"], verbose=True,
)
mo_list, mo_coeff = list(active_space.mo_list), active_space.mo_coeff
print(f"  ASF candidates: {len(mo_list)} orbitals -> {mo_list}")
if len(mo_list) == 0:
    raise RuntimeError("ASF returned 0 candidates. Lower entropy_threshold.")

print(f"\n-- Phase C: Gap Detection --")
cand_devs = np.array([deviation[i] if i < len(deviation) else 0.0 for i in mo_list])
n_final, gap_val, selected_k = find_gap_cutoff(cand_devs, config.GAP_MIN_NORB, config.GAP_MAX_NORB)
final_mo_list = sorted(mo_list[k] for k in selected_k)
print(f"  Gap detected: {gap_val:.4f} at position {n_final} -> orbitals {final_mo_list}")

nel = count_active_electrons(mol, mf, final_mo_list)

print(f"\n-- Phase D: Loewdin Population --")
S = mol.intor("int1e_ovlp")
ao_labels = mol.ao_labels(fmt=None)
weights = lowdin_population(mo_coeff, final_mo_list, S, ao_labels, config.N_ATOMS)
dominant_atoms = np.argmax(weights, axis=1).astype(int)

final_devs = np.array([deviation[i] for i in final_mo_list if i < len(deviation)])
corr_strength = float(np.mean(final_devs)) if len(final_devs) > 0 else 0.0

print(f"\n{'='*60}\n[Step 1] Summary — {config.MOLECULE}")
print(f"  Tier: {tier}  Active space: ({nel}e, {n_final}orb)  Orbitals: {final_mo_list}")
print(f"  Correlation strength: {corr_strength:.4f}")
print(f"{'='*60}")

results = {
    "nel": nel, "mo_list": final_mo_list, "mo_coeff": mo_coeff,
    "n_active_orbs": n_final, "no_occ": no_occ, "deviation": deviation,
    "lowdin_weights": weights, "dominant_atoms": dominant_atoms,
    "tier": tier, "indicators": indicators, "corr_strength": corr_strength,
    "mol_info": {
        "molecule": config.MOLECULE, "basis": config.BASIS,
        "n_atoms": config.N_ATOMS, "atom_syms": config.ATOM_SYMS,
        "n_electrons": mol.nelectron, "n_ao": mol.nao_nr(),
    },
    "uhf_energy": float(mf.e_tot), "mp2_energy": float(mf.e_tot + e_corr),
    "mp2_ok": mp2_ok,
    "mo_coeff_uhf": np.asarray(mf.mo_coeff), "mo_energy": np.asarray(mf.mo_energy),
    "mo_occ": np.asarray(mf.mo_occ), "converged": mf.converged,
    # NEW: full spin-separated MP2 1-RDM in AO basis, so DMET.py never
    # needs to recompute MP2 when config.DMET_REFERENCE == "mp2".
    "dm_ao_alpha_mp2": dm_ao_alpha_mp2,
    "dm_ao_beta_mp2":  dm_ao_beta_mp2,
    "dm_ao_total_mp2": dm_ao_alpha_mp2 + dm_ao_beta_mp2,
}

with open(config.STEP1_FILE, "wb") as f:
    pickle.dump(results, f)
print(f"\n[Step 1] Saved -> {config.STEP1_FILE}")