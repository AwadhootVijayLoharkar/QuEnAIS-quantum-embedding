# DMET.py — test_8 — DMET Embedding Hamiltonian
"""
get_reference_density / chemical_potential_correction /
embedding_consistency_score live in dmet_lib.py (no module-level side
effects) -- that's what fixed the bug where `import DMET` from
gqe_for_qsci.py silently killed the whole process via this script's own
cache-check sys.exit(0).

Saves the FULL Schmidt singular-value spectrum (`sv_all`) in addition to
the kept bath SVs, so visualization.py can plot the gap adaptive_bath()
used to pick n_bath, not just the orbitals it kept.

CHANGE vs test7: `import config` + the dmet_lib import now happen FIRST,
before numpy -- see config.py's docstring (OpenBLAS/OpenMP env vars must
be set before numpy's first import to take effect; test7's DMET.py
imported numpy before config, so the fix silently didn't apply here).

Usage: python DMET.py [--force]
"""

import config
from dmet_lib import (
    get_reference_density,
    chemical_potential_correction,
    embedding_consistency_score,  # re-exported for convenience; not called here
)

import os
import sys
import time
import pickle
import argparse
import warnings
import numpy as np

parser = argparse.ArgumentParser(description="Step 2: DMET Embedding Hamiltonian")
parser.add_argument("--force", action="store_true")
args = parser.parse_args()
FORCE_RERUN = args.force

os.makedirs(config.RESULTS_DIR, exist_ok=True)
if os.path.exists(config.STEP2_FILE) and not FORCE_RERUN:
    print(f"[Step 2] Using cached result: {config.STEP2_FILE}")
    sys.exit(0)
if not os.path.exists(config.STEP1_FILE):
    raise FileNotFoundError(f"Run ASF.py first: {config.STEP1_FILE} not found.")

with open(config.STEP1_FILE, "rb") as f:
    step1 = pickle.load(f)

nel, mo_list, mo_coeff = step1["nel"], step1["mo_list"], step1["mo_coeff"]
n_imp, mol_info = step1["n_active_orbs"], step1["mol_info"]

from pyscf import gto, scf, ao2mo
from pyscf.scf import hf as pyscf_hf


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def lowdin_matrices(S):
    evals, evecs = np.linalg.eigh(S)
    mask = evals > 1e-15
    sq = np.sqrt(evals[mask])
    S_sqrt    = (evecs[:, mask] * sq) @ evecs[:, mask].T
    S_invsqrt = (evecs[:, mask] / sq) @ evecs[:, mask].T
    return S_sqrt, S_invsqrt


def adaptive_bath(sv, n_imp, max_embed, bath_tol):
    max_bath = min(n_imp, max(0, max_embed - n_imp))
    if max_bath == 0 or len(sv) == 0:
        return 0, 0.0, 0.0
    sv_arr = np.asarray(sv, dtype=float)
    sv_above = sv_arr[sv_arr > bath_tol]
    sv_filtered = sv_above[:max_bath] if len(sv_above) > 0 else sv_arr[:max_bath]
    n_avail = len(sv_filtered)
    if n_avail == 0:
        return 0, 0.0, 0.0

    best_gap, best_n = -1.0, 1
    for n in range(1, n_avail + 1):
        gap = sv_filtered[n - 1] - (sv_filtered[n] if n < n_avail else 0.0)
        if gap > best_gap:
            best_gap, best_n = gap, n

    sv2_total = float(np.sum(sv_filtered ** 2))
    if sv2_total < 1e-30:
        return 0, 0.0, 0.0
    cumsum, n_cov = 0.0, 0
    for i, s in enumerate(sv_filtered):
        cumsum += s * s
        n_cov = i + 1
        if cumsum / sv2_total >= 0.999:
            break

    n_bath = min(max(best_n, n_cov), max_bath)
    sv2_cov = float(np.sum(sv_filtered[:n_bath] ** 2) / sv2_total)
    return n_bath, float(best_gap), sv2_cov


def _symmetrize_h2e(h2e):
    return (
        h2e + h2e.transpose(1, 0, 2, 3) + h2e.transpose(0, 1, 3, 2)
        + h2e.transpose(1, 0, 3, 2) + h2e.transpose(2, 3, 0, 1)
        + h2e.transpose(3, 2, 0, 1) + h2e.transpose(2, 3, 1, 0)
        + h2e.transpose(3, 2, 1, 0)
    ) / 8.0


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}\n[Step 2] DMET Embedding — {mol_info['molecule']}\n{'='*60}")
print(f"  Active space (Step 1): ({nel}e, {n_imp}orb) MOs={mo_list}")
print(f"  DMET_REFERENCE: {config.DMET_REFERENCE}   MU_CORRECTION: {config.MU_CORRECTION}")

mol = gto.M(atom=config.GEOMETRY, basis=config.BASIS,
            charge=config.CHARGE, spin=config.SPIN, verbose=0)
n_ao = mol.nao_nr()

print(f"\n-- Phase A: Restore UHF from Step 1 --")
mf = scf.UHF(mol)
mf.mo_coeff, mf.mo_energy, mf.mo_occ = (step1["mo_coeff_uhf"], step1["mo_energy"], step1["mo_occ"])
mf.e_tot, mf.converged = step1["uhf_energy"], step1["converged"]
print(f"  UHF energy = {mf.e_tot:.8f} Ha (restored, no recomputation)")
if not mf.converged:
    warnings.warn("Restored UHF was not converged in Step 1.", RuntimeWarning)

print(f"\n-- Phase B: Reference Density ({config.DMET_REFERENCE}) --")
dm_ao_total, dm_ao_alpha, dm_ao_beta, ref_info = get_reference_density(
    mf, mol, step1, mo_list, mo_coeff, config.DMET_REFERENCE
)
print(f"  {ref_info}")

print(f"\n-- Phase C: Schmidt Decomposition --")
S = mol.intor("int1e_ovlp")
S_sqrt, S_invsqrt = lowdin_matrices(S)

C_imp = mo_coeff[:, mo_list].copy()
Q_imp = S_sqrt @ C_imp
dm_lo = S_sqrt @ dm_ao_total @ S_sqrt
P_env = np.eye(n_ao) - Q_imp @ Q_imp.T
F = P_env @ dm_lo @ Q_imp
U_env, sv, _ = np.linalg.svd(F, full_matrices=True)

n_bath, sv_gap, sv2_cov = adaptive_bath(sv, n_imp, config.MAX_EMBED_ORBS, config.BATH_TOLERANCE)
if n_bath < config.MIN_BATH_ORBS:
    warnings.warn(f"Only {n_bath} bath orbital(s) found.", RuntimeWarning)

if n_bath > 0:
    Q_bath = U_env[:, :n_bath]
    Q_emb = np.hstack([Q_imp, Q_bath])
else:
    Q_emb = Q_imp.copy()

n_emb = n_imp + n_bath
C_emb = S_invsqrt @ Q_emb
print(f"  Impurity: {n_imp}  Bath: {n_bath}  Total emb: {n_emb} -> {2*n_emb} qubits")
print(f"  sv2 coverage: {sv2_cov:.4f}")

print(f"\n-- Phase D: Core Mean-Field Potential --")
h1e_bare = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
P_emb_lo = Q_emb @ Q_emb.T
P_core_lo = np.eye(n_ao) - P_emb_lo
dm_core_lo_alpha = P_core_lo @ (S_sqrt @ dm_ao_alpha @ S_sqrt) @ P_core_lo
dm_core_lo_beta  = P_core_lo @ (S_sqrt @ dm_ao_beta  @ S_sqrt) @ P_core_lo
dm_core_alpha = S_invsqrt @ dm_core_lo_alpha @ S_invsqrt
dm_core_beta  = S_invsqrt @ dm_core_lo_beta  @ S_invsqrt
dm_core_alpha = 0.5 * (dm_core_alpha + dm_core_alpha.T)
dm_core_beta  = 0.5 * (dm_core_beta  + dm_core_beta.T)

vj_a, vk_a = pyscf_hf.get_jk(mol, dm_core_alpha, hermi=1)
vj_b, vk_b = pyscf_hf.get_jk(mol, dm_core_beta,  hermi=1)
h1e_eff = h1e_bare + (vj_a + vj_b) - 0.5 * (vk_a + vk_b)

print(f"\n-- Phase E: Integral Transformation --")
t0 = time.time()
h1e_emb = C_emb.T @ h1e_eff @ C_emb
h1e_emb = 0.5 * (h1e_emb + h1e_emb.T)
h2e_raw = ao2mo.kernel(mol, C_emb, compact=False).reshape(n_emb, n_emb, n_emb, n_emb)
h2e_emb = _symmetrize_h2e(h2e_raw)

n_alpha = nel // 2 + nel % 2
n_beta  = nel // 2

dm_a_hf = np.diag([1.0] * n_alpha + [0.0] * (n_emb - n_alpha))
dm_b_hf = np.diag([1.0] * n_beta  + [0.0] * (n_emb - n_beta))
dm_t_hf = dm_a_hf + dm_b_hf

e1_hf = float(np.einsum("ij,ji->", h1e_emb, dm_t_hf))
J_hf  = np.einsum("pqrs,rs->pq", h2e_emb, dm_t_hf)
Ka_hf = np.einsum("prqs,rs->pq", h2e_emb, dm_a_hf)
Kb_hf = np.einsum("prqs,rs->pq", h2e_emb, dm_b_hf)
e2_hf = 0.5 * (float(np.einsum("pq,qp->", J_hf, dm_t_hf))
               - float(np.einsum("pq,qp->", Ka_hf, dm_a_hf))
               - float(np.einsum("pq,qp->", Kb_hf, dm_b_hf)))
e_hf_emb = e1_hf + e2_hf
ecore = float(mf.e_tot) - e_hf_emb

delta_hf = (e_hf_emb + ecore) - float(mf.e_tot)
assert abs(delta_hf) < 1e-6, f"ecore self-consistency failed: delta={delta_hf:.6e} Ha"
print(f"  ecore (self-consistent) = {ecore:.8f} Ha   [check OK, delta={delta_hf:.2e}]")

mu = 0.0
if config.MU_CORRECTION:
    print(f"\n-- Phase F: Chemical Potential Correction --")
    h1e_emb, mu = chemical_potential_correction(
        h1e_emb, n_emb, n_alpha, n_beta,
        config.MU_SEARCH_RANGE, config.MU_MAX_ITER, config.MU_TOL,
    )
    ecore += mu * (n_alpha + n_beta)
    print(f"  mu = {mu:.6f} Ha   ecore corrected -> {ecore:.8f} Ha")
else:
    print(f"\n-- Phase F: Chemical Potential Correction -- SKIPPED (MU_CORRECTION=False)")

dm_emb_alpha_mo = C_emb.T @ S @ dm_ao_alpha @ S @ C_emb
dm_emb_beta_mo  = C_emb.T @ S @ dm_ao_beta  @ S @ C_emb
ref_occ_alpha = np.clip(np.diag(dm_emb_alpha_mo), 0.0, 1.0)
ref_occ_beta  = np.clip(np.diag(dm_emb_beta_mo),  0.0, 1.0)

elapsed = time.time() - t0
print(f"\n  h1e shape: {h1e_emb.shape}  h2e shape: {h2e_emb.shape}  Time: {elapsed:.1f}s")

results = {
    "h1e": h1e_emb, "h2e": h2e_emb, "ecore": ecore, "mu": mu,
    "n_emb": n_emb, "n_imp": n_imp, "n_bath": n_bath,
    "n_alpha": n_alpha, "n_beta": n_beta,
    "sv": sv[:n_bath],
    "sv_all": sv,
    "sv_gap": sv_gap, "sv2_cov": sv2_cov,
    "uhf_energy": float(mf.e_tot), "reference_density_info": ref_info,
    "ref_occ_alpha": ref_occ_alpha, "ref_occ_beta": ref_occ_beta,
    "mol_info": mol_info,
}

with open(config.STEP2_FILE, "wb") as f:
    pickle.dump(results, f)
print(f"\n[Step 2] Saved -> {config.STEP2_FILE}")