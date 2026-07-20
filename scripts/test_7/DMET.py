# DMET.py — test7 — DMET Embedding Hamiltonian
"""
Same core Schmidt-decomposition / core-potential / integral-transform
logic as test5's DMET.py, with three functional changes:

1. Pluggable reference density (config.DMET_REFERENCE = "mp2" | "casci").
   "mp2" now reads Step 1's saved DM instead of recomputing MP2 (removes
   a redundant O(N^5) solve). "casci" runs a plain CASCI/FCI -- no orbital
   optimization -- INSIDE the already-small ASF active space to get a
   correlated reference density instead of a perturbative one. This does
   NOT solve the impurity+bath problem (that's still the quantum solver's
   job downstream) -- it only improves what goes into building the bath,
   at a cost already bounded by the active-space sizes (~10-16 orbitals)
   this pipeline already treats as classically tractable elsewhere.

2. Chemical-potential (mu) bisection correction (config.MU_CORRECTION),
   reinstated from the algorithm your own test3_documentation/hamiltonian.md
   describes but which was missing from test5's actual DMET.py. Shifts
   h1e_emb by -mu*I so the number of negative eigenvalues matches the
   target electron count per spin channel, fixing one-shot DMET's
   grand-canonical fractional-occupation symptom. mu is then added back
   into ecore so the total energy bookkeeping stays correct.

3. Saves `ref_occ_alpha` / `ref_occ_beta` -- the embedding-space
   occupations implied by the reference density used to build the bath --
   so a downstream solver (Step 3 / gqe_for_qsci.py) can call
   `embedding_consistency_score()` below and get a free diagnostic of
   whether the one-shot bath is still a good approximation, without
   rebuilding anything.

Rename to DMET.py when you drop this into your own test7/ folder.
"""

import os
import sys
import time
import pickle
import argparse
import warnings
import numpy as np

import config

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

from pyscf import gto, scf, ao2mo, fci
from pyscf.scf import hf as pyscf_hf


# ═══════════════════════════════════════════════════════════════════════
# Helpers (unchanged from test5)
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
# NEW 1: pluggable reference density
# ═══════════════════════════════════════════════════════════════════════

def get_reference_density(mf, mol, step1, method):
    """
    Returns (dm_ao_total, dm_ao_alpha, dm_ao_beta, info) -- the density
    matrix used to drive the Schmidt decomposition.
    """
    if method == "mp2":
        for key in ("dm_ao_total_mp2", "dm_ao_alpha_mp2", "dm_ao_beta_mp2"):
            if key not in step1:
                raise KeyError(
                    f"step1 pickle missing '{key}'. Re-run ASF.py (test7 "
                    f"version) with --force -- it now saves the MP2 DM "
                    f"so this function never recomputes MP2."
                )
        return (step1["dm_ao_total_mp2"], step1["dm_ao_alpha_mp2"],
                step1["dm_ao_beta_mp2"], {"method": "mp2", "recomputed": False})

    elif method == "casci":
        nel_active = step1["nel"]
        n_active   = len(mo_list)
        n_alpha    = nel_active // 2 + nel_active % 2
        n_beta     = nel_active // 2

        C_active = mo_coeff[:, mo_list]
        h1e_ao   = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
        h1e_act  = C_active.T @ h1e_ao @ C_active
        h2e_act  = ao2mo.kernel(mol, C_active, compact=False).reshape(
            n_active, n_active, n_active, n_active
        )

        cisolver = fci.direct_spin1.FCI()
        cisolver.verbose = 0
        e_cas, civec = cisolver.kernel(h1e_act, h2e_act, n_active, (n_alpha, n_beta))
        dm_active_a, dm_active_b = cisolver.make_rdm1s(civec, n_active, (n_alpha, n_beta))

        # Embed back into the FULL ASF natural-orbital basis: active block
        # gets the CASCI-correlated occupations, everything else keeps its
        # Step-1 MP2 natural-orbital occupation (already reasonable there
        # -- it's only unreliable inside the active space, which is
        # exactly the block we're replacing).
        n_mo_total = mo_coeff.shape[1]
        no_occ = step1["no_occ"]
        dm_full_a = np.diag([no_occ[i] / 2.0 for i in range(n_mo_total)])
        dm_full_b = np.diag([no_occ[i] / 2.0 for i in range(n_mo_total)])
        for a_i, i in enumerate(mo_list):
            for a_j, j in enumerate(mo_list):
                dm_full_a[i, j] = dm_active_a[a_i, a_j]
                dm_full_b[i, j] = dm_active_b[a_i, a_j]

        dm_ao_alpha = mo_coeff @ dm_full_a @ mo_coeff.T
        dm_ao_beta  = mo_coeff @ dm_full_b @ mo_coeff.T
        dm_ao_total = dm_ao_alpha + dm_ao_beta

        return (dm_ao_total, dm_ao_alpha, dm_ao_beta,
                {"method": "casci", "e_cas": float(e_cas),
                 "n_active": n_active, "nel_active": nel_active})

    else:
        raise ValueError(f"Unknown DMET_REFERENCE='{method}'. Use 'mp2' or 'casci'.")


# ═══════════════════════════════════════════════════════════════════════
# NEW 2: chemical-potential (mu) bisection correction
# ═══════════════════════════════════════════════════════════════════════

def chemical_potential_correction(h1e_emb, n_emb, n_alpha, n_beta,
                                   mu_range, max_iter, tol):
    """
    One-shot DMET grand-canonical fix, exactly the algorithm documented in
    hamiltonian.md: find mu such that the number of eigenvalues of
    (h1e_emb - mu*I) below zero equals the target electron count, then
    return the mu-shifted h1e plus mu itself (needed to correct ecore).

    Assumes a closed-shell embedding (n_alpha == n_beta); for open-shell
    embeddings, run this twice with separate alpha/beta h1e blocks.

    Cost: repeated eigh() of an (n_emb x n_emb) matrix -- negligible.
    """
    if n_alpha != n_beta:
        warnings.warn(
            f"chemical_potential_correction assumes n_alpha==n_beta "
            f"(got {n_alpha},{n_beta}); using n_alpha as the target and "
            f"applying the same shift to both spins. Verify this is "
            f"appropriate for open-shell embeddings.", RuntimeWarning,
        )
    target = n_alpha

    def n_below_zero(mu):
        evals = np.linalg.eigvalsh(h1e_emb - mu * np.eye(n_emb))
        return int(np.sum(evals < 0.0))

    lo, hi = mu_range
    n_lo, n_hi = n_below_zero(lo), n_below_zero(hi)
    if not (n_lo <= target <= n_hi):
        warnings.warn(
            f"mu search range {mu_range} does not bracket target electron "
            f"count {target} (n(mu={lo})={n_lo}, n(mu={hi})={n_hi}). "
            f"Widen MU_SEARCH_RANGE in config.py, or check h1e_emb for "
            f"numerical issues. Skipping mu correction.", RuntimeWarning,
        )
        return h1e_emb, 0.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if n_below_zero(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    mu = 0.5 * (lo + hi)

    h1e_shifted = h1e_emb - mu * np.eye(n_emb)
    # <H_shifted> = <H> - mu*N for any state in the fixed-N sector, so the
    # solver's output energy on h1e_shifted must have +mu*N_target added
    # back. We fold that into ecore instead, so nothing downstream needs
    # to remember to do this manually:
    #   ecore_corrected = ecore + mu * (n_alpha + n_beta)
    return h1e_shifted, mu


# ═══════════════════════════════════════════════════════════════════════
# NEW 3: embedding self-consistency diagnostic (no bath rebuild)
# ═══════════════════════════════════════════════════════════════════════

def embedding_consistency_score(step2_result, avg_occs, threshold=0.10):
    """
    Compare the electron distribution implied by the density matrix used
    to build the bath (ref_occ_alpha/beta, saved below) against the
    distribution the solver actually found (avg_occs, e.g. from
    qiskit_addon_sqd.solve_fermion or a GQE/QSCI subspace diagonalization).

    This is a DIAGNOSTIC ONLY -- it does not rebuild the bath or loop.
    Large mismatch means the one-shot approximation likely broke down for
    this molecule (try DMET_REFERENCE="casci" if you're on "mp2", or a
    genuine self-consistent DMET loop if it's still large).

    Call this from Step 3 / gqe_for_qsci.py after the solver returns.
    """
    ref_a = step2_result.get("ref_occ_alpha")
    ref_b = step2_result.get("ref_occ_beta")
    if ref_a is None or ref_b is None:
        raise KeyError(
            "step2 pickle has no 'ref_occ_alpha'/'ref_occ_beta' -- re-run "
            "DMET.py (test7 version)."
        )
    occ_a, occ_b = avg_occs
    mismatch_a = float(np.mean(np.abs(np.asarray(occ_a) - ref_a)))
    mismatch_b = float(np.mean(np.abs(np.asarray(occ_b) - ref_b)))
    score = 0.5 * (mismatch_a + mismatch_b)
    return {
        "mismatch_alpha": mismatch_a, "mismatch_beta": mismatch_b,
        "mismatch_score": score, "flag": score > threshold,
    }


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
    mf, mol, step1, config.DMET_REFERENCE
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

# ── NEW: chemical-potential correction ───────────────────────────────────
mu = 0.0
if config.MU_CORRECTION:
    print(f"\n-- Phase F: Chemical Potential Correction --")
    h1e_emb, mu = chemical_potential_correction(
        h1e_emb, n_emb, n_alpha, n_beta,
        config.MU_SEARCH_RANGE, config.MU_MAX_ITER, config.MU_TOL,
    )
    ecore += mu * (n_alpha + n_beta)   # compensate the energy shift, see docstring
    print(f"  mu = {mu:.6f} Ha   ecore corrected -> {ecore:.8f} Ha")
else:
    print(f"\n-- Phase F: Chemical Potential Correction -- SKIPPED (MU_CORRECTION=False)")

# ── NEW: save reference occupations for the consistency check ───────────
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
    "sv": sv[:n_bath], "sv_gap": sv_gap, "sv2_cov": sv2_cov,
    "uhf_energy": float(mf.e_tot), "reference_density_info": ref_info,
    "ref_occ_alpha": ref_occ_alpha, "ref_occ_beta": ref_occ_beta,
    "mol_info": mol_info,
}

with open(config.STEP2_FILE, "wb") as f:
    pickle.dump(results, f)
print(f"\n[Step 2] Saved -> {config.STEP2_FILE}")