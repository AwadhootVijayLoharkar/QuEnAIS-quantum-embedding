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
if config.cached_result_is_current(config.STEP2_FILE) and not FORCE_RERUN:
    print(f"[Step 2] Using cached result: {config.STEP2_FILE}")
    sys.exit(0)
if not os.path.exists(config.STEP1_FILE):
    raise FileNotFoundError(f"Run ASF.py first: {config.STEP1_FILE} not found.")
if not config.cached_result_is_current(config.STEP1_FILE):
    raise RuntimeError(
        f"{config.STEP1_FILE} was generated for a DIFFERENT molecule/basis "
        f"than config says ({config.MOLECULE}/{config.BASIS}). Re-run "
        f"ASF.py (it will recompute automatically) before running DMET.py "
        f"-- silently building an embedding on another molecule's active "
        f"space is exactly the class of bug this check exists to stop."
    )

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
    """
    FIX: this used to fall back to `sv_arr[:max_bath]` -- i.e. grab the
    top max_bath singular values REGARDLESS of whether any of them
    actually cleared bath_tol -- whenever none did. That's not a safe
    fallback, it's silently manufacturing a bath out of numerically
    meaningless near-zero singular values. On N2's (4e,4o) active space,
    the Schmidt singular values came back EXACTLY [0,0,0,0] (confirmed:
    the active-space orbitals are close to eigenvectors of the reference
    density, so there's no real impurity-environment entanglement left
    to extract), yet this fallback still produced "4 bath orbitals" --
    which is where the badly non-orthonormal C_emb and ~20 Ha unphysical
    embedding energies came from. Correct behavior: if nothing clears
    bath_tol, there IS no bath -- return 0 cleanly, and let the embedding
    fall back to being just the active space by itself (n_emb = n_imp,
    handled by the `if n_bath > 0` branch below).
    """
    max_bath = min(n_imp, max(0, max_embed - n_imp))
    if max_bath == 0 or len(sv) == 0:
        return 0, 0.0, 0.0
    sv_arr = np.asarray(sv, dtype=float)
    sv_above = sv_arr[sv_arr > bath_tol]
    if len(sv_above) == 0:
        return 0, 0.0, 0.0
    sv_filtered = sv_above[:max_bath]
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
S_check = mol.intor("int1e_ovlp")
n_elec_ref = float(np.trace(dm_ao_alpha @ S_check) + np.trace(dm_ao_beta @ S_check))
print(f"  [diag] reference density electron count: {n_elec_ref:.6f}  "
      f"(should equal mol.nelectron = {mol.nelectron})")

print(f"\n-- Phase C: Schmidt Decomposition --")
S = mol.intor("int1e_ovlp")
S_sqrt, S_invsqrt = lowdin_matrices(S)

_n_mo_chk = mo_coeff.shape[1]
_err_raw_ao   = float(np.max(np.abs(mo_coeff.T @ S @ mo_coeff - np.eye(_n_mo_chk))))
_err_already_orthonorm = float(np.max(np.abs(mo_coeff.T @ mo_coeff - np.eye(_n_mo_chk))))
print(f"  [diag] mo_coeff.T @ S @ mo_coeff vs I (raw-AO-basis convention, "
      f"expected by the rest of this code): {_err_raw_ao:.2e}")
print(f"  [diag] mo_coeff.T @ mo_coeff vs I (already-orthonormal/Lowdin-basis "
      f"convention): {_err_already_orthonorm:.2e}")
print(f"  [diag] whichever of the two lines above is near zero (~1e-10) tells "
      f"us which basis convention mo_coeff is actually in.")

C_imp = mo_coeff[:, mo_list].copy()
Q_imp = S_sqrt @ C_imp
dm_lo = S_sqrt @ dm_ao_total @ S_sqrt
P_env = np.eye(n_ao) - Q_imp @ Q_imp.T
F = P_env @ dm_lo @ Q_imp
U_env, sv, _ = np.linalg.svd(F, full_matrices=True)
print(f"  [diag] Schmidt singular values (all {len(sv)}, before bath-count "
      f"selection): {np.array2string(sv, precision=6, suppress_small=True)}")
_Uimp_overlap = float(np.max(np.abs(Q_imp.T @ U_env[:, :n_imp])))
print(f"  [diag] max |Q_imp.T @ U_env[:, :n_imp]| (should be ~1e-10 if the "
      f"first n_imp SVD vectors are genuinely orthogonal to Q_imp): "
      f"{_Uimp_overlap:.2e}")

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
orthonorm_err = float(np.max(np.abs(C_emb.T @ S @ C_emb - np.eye(n_emb))))
print(f"  [diag] C_emb orthonormality error (C_emb.T @ S @ C_emb vs I): "
      f"{orthonorm_err:.2e}  (should be ~1e-10 or smaller)")

# FIX (found via LiH n_bath=2 run: DMET+CASCI came out ~2x HF): n_alpha/
# n_beta used to be taken from Step 1's ACTIVE-SPACE-ONLY electron count
# (nel) further down. That's only correct when n_bath=0 (embedding space
# == active space). Once bath orbitals are added, n_emb > n_imp, and the
# embedding space's true electron count is whatever the reference density
# actually puts there -- which the code already computed (ref_occ_alpha/
# beta) but only used for a downstream consistency check, never to
# correct n_alpha/n_beta itself. Confirmed on your LiH run: nel-based
# gave (1,1) while the reference-density trace gave (2.0000076,
# 2.0000076) -- rounds to (2,2). Moved here (right after C_emb exists,
# before Phase D/E need it) and used as the actual FCI electron target.
dm_emb_alpha_mo = C_emb.T @ S @ dm_ao_alpha @ S @ C_emb
dm_emb_beta_mo  = C_emb.T @ S @ dm_ao_beta  @ S @ C_emb
ref_occ_alpha = np.clip(np.diag(dm_emb_alpha_mo), 0.0, 1.0)
ref_occ_beta  = np.clip(np.diag(dm_emb_beta_mo),  0.0, 1.0)
n_alpha = int(round(float(np.sum(ref_occ_alpha))))
n_beta  = int(round(float(np.sum(ref_occ_beta))))
print(f"  [diag] embedding-space electron count from reference density: "
      f"alpha={np.sum(ref_occ_alpha):.6f} -> {n_alpha}, "
      f"beta={np.sum(ref_occ_beta):.6f} -> {n_beta}  "
      f"(old nel-based value was alpha={nel//2 + nel%2}, beta={nel//2})")

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

n_core_elec = float(np.trace(dm_core_alpha @ S) + np.trace(dm_core_beta @ S))
n_emb_elec_expected = nel  # active-space electron count from Step 1
print(f"  [diag] core density electron count: {n_core_elec:.6f}  "
      f"(expect ~= mol.nelectron - active nel = {mol.nelectron - n_emb_elec_expected})")

vj_a, vk_a = pyscf_hf.get_jk(mol, dm_core_alpha, hermi=1)
vj_b, vk_b = pyscf_hf.get_jk(mol, dm_core_beta,  hermi=1)
h1e_eff = h1e_bare + (vj_a + vj_b) - 0.5 * (vk_a + vk_b)

print(f"\n-- Phase E: Integral Transformation --")
t0 = time.time()
h1e_emb = C_emb.T @ h1e_eff @ C_emb
h1e_emb = 0.5 * (h1e_emb + h1e_emb.T)
h2e_raw = ao2mo.kernel(mol, C_emb, compact=False).reshape(n_emb, n_emb, n_emb, n_emb)
h2e_emb = _symmetrize_h2e(h2e_raw)

h1e_evals = np.linalg.eigvalsh(h1e_emb)
print(f"  [diag] h1e_emb eigenvalues (Ha, before mu): "
      f"{np.array2string(h1e_evals, precision=3, suppress_small=True)}")
print(f"  [diag] for comparison, full UHF valence orbital energies (Ha): "
      f"{np.array2string(np.asarray(step1['mo_energy'])[0][:n_ao], precision=3, suppress_small=True)}")

# n_alpha/n_beta now come from the reference-density trace, computed
# right after Phase C -- see the FIX comment there.

# FIX #2 (found via LiH re-run: n_bath=2 fix alone only moved
# DMET+CASCI(active) from -15.92 -> -12.47 Ha, still ~4.6 Ha off HF/
# CASSCF's -7.86/-7.88 Ha). This used to build dm_a_hf/dm_b_hf as a
# naive "aufbau" density -- fill the FIRST n_alpha/n_beta COLUMNS of
# C_emb (which are, in order, Q_imp's impurity orbitals then Q_bath's
# bath orbitals) as "occupied", regardless of h1e_emb's actual
# eigenvalue ordering (printed just above as h1e_evals, but never used
# here). Since ecore is DEFINED as mf.e_tot - e_hf_emb, that made-up
# density directly determines how much of the TRUE HF energy gets
# folded into ecore vs. the embedding space -- an arbitrary filling
# assumption, not the real HF solution's projection into this basis.
#
# Fixed the same way ref_occ already fixes the CASCI electron count:
# project the ACTUAL full-molecule UHF density (mf, restored in Phase
# A) into the embedding basis, instead of guessing an occupation
# pattern.
dm_hf_ao_alpha, dm_hf_ao_beta = mf.make_rdm1()
dm_a_hf = C_emb.T @ S @ dm_hf_ao_alpha @ S @ C_emb
dm_b_hf = C_emb.T @ S @ dm_hf_ao_beta  @ S @ C_emb
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

# ref_occ_alpha/ref_occ_beta already computed right after Phase C.

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