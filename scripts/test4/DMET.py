# step2_hamiltonian.py — DMET Embedding Hamiltonian
"""
Builds an effective Hamiltonian in a compact embedding space (impurity + bath).

Phases:
  A: Rebuild MF from Step 1 (no recomputation)
  B: MP2 density matrix for Schmidt decomposition
  C: Schmidt decomposition → bath orbitals + adaptive truncation
  D: Core mean-field potential
  E: Integral transformation → h1e_emb, h2e_emb

Requires: results/step1_asf.pkl
Saves:    results/step2_hamiltonian.pkl
"""

import os
import sys
import time
import pickle
import numpy as np

import config

# ── Setup ─────────────────────────────────────────────────────────────────────
FORCE_RERUN = True

os.makedirs(config.RESULTS_DIR, exist_ok=True)
if os.path.exists(config.STEP2_FILE) and not FORCE_RERUN:
    print(f"[Step 2] Cached: {config.STEP2_FILE}")
    sys.exit(0)

if not os.path.exists(config.STEP1_FILE):
    raise FileNotFoundError(f"Run step1_asf.py first. Missing: {config.STEP1_FILE}")

with open(config.STEP1_FILE, "rb") as f:
    step1 = pickle.load(f)

nel           = step1["nel"]
mo_list       = step1["mo_list"]
mo_coeff      = step1["mo_coeff"]
n_imp         = step1["n_active_orbs"]
mol_info      = step1["mol_info"]

from pyscf import gto, scf, mp as pyscf_mp, ao2mo
from pyscf.scf import hf as pyscf_hf


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def lowdin_matrices(S):
    """Compute S^{1/2} and S^{-1/2} from overlap matrix."""
    evals, evecs = np.linalg.eigh(S)
    mask = evals > 1e-15
    sq = np.sqrt(evals[mask])
    S_sqrt    = (evecs[:, mask] * sq) @ evecs[:, mask].T
    S_invsqrt = (evecs[:, mask] / sq) @ evecs[:, mask].T
    return S_sqrt, S_invsqrt


def adaptive_bath(sv, n_imp, max_embed):
    """
    Select bath orbitals using two criteria (take the larger):
      1. Largest gap in singular value spectrum
      2. Cumulative sv² coverage ≥ 99.9%
    """
    max_bath = min(n_imp, max(0, max_embed - n_imp))
    if max_bath == 0 or len(sv) == 0:
        return 0, 0.0, 0.0

    sv_arr = np.asarray(sv[:max_bath], dtype=float)
    n_avail = len(sv_arr)

    # Gap detection
    best_gap, best_n = -1.0, 1
    for n in range(1, n_avail + 1):
        gap = sv_arr[n-1] - (sv_arr[n] if n < n_avail else 0.0)
        if gap > best_gap:
            best_gap, best_n = gap, n

    # Coverage criterion
    sv2_total = float(np.sum(sv_arr**2))
    if sv2_total < 1e-20:
        return 0, 0.0, 0.0

    cumsum, n_cov = 0.0, 0
    for i, s in enumerate(sv_arr):
        cumsum += s * s
        n_cov = i + 1
        if cumsum / sv2_total >= 0.999:
            break

    n_bath = min(max(best_n, n_cov), max_bath)
    sv2_cov = float(np.sum(sv_arr[:n_bath]**2) / sv2_total)
    return n_bath, best_gap, sv2_cov


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"[Step 2] DMET Embedding — {mol_info['molecule']}")
print(f"{'='*60}")
print(f"  Active space from Step 1: ({nel}e, {n_imp}orb)  MOs={mo_list}")

# Build molecule
mol = gto.M(
    atom    = config.GEOMETRY,
    basis   = config.BASIS,
    charge  = config.CHARGE,
    spin    = config.SPIN,
    verbose = 0,
)
n_ao = mol.nao_nr()

# ── Phase A: Restore MF from Step 1 ──────────────────────────────────────────
print("\n── Phase A: Restore UHF from Step 1 ──")
mf = scf.UHF(mol)
mf.mo_coeff  = step1["mo_coeff_uhf"]
mf.mo_energy = step1["mo_energy"]
mf.mo_occ    = step1["mo_occ"]
mf.e_tot     = step1["uhf_energy"]
mf.converged = step1["converged"]
print(f"  UHF energy = {mf.e_tot:.8f} Ha (from Step 1, no recomputation)")

# ── Phase B: MP2 Density Matrix ──────────────────────────────────────────────
print("\n── Phase B: MP2 Density Matrix ──")
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
    mp2_ok = True
except Exception:
    e_corr = 0.0
    dm_raw = mf.make_rdm1()
    dm_ao = (dm_raw[0] + dm_raw[1]) if isinstance(dm_raw, tuple) else dm_raw
    mp2_ok = False

print(f"  MP2 used: {mp2_ok}  |  E_corr = {e_corr:.6f} Ha")

# ── Phase C: Schmidt Decomposition ───────────────────────────────────────────
print("\n── Phase C: Schmidt Decomposition ──")
S = mol.intor("int1e_ovlp")
S_sqrt, S_invsqrt = lowdin_matrices(S)

C_imp = mo_coeff[:, mo_list].copy()
Q_imp = S_sqrt @ C_imp

dm_lo = S_sqrt @ dm_ao @ S_sqrt

P_env = np.eye(n_ao) - Q_imp @ Q_imp.T
F = P_env @ dm_lo @ Q_imp
U_env, sv, _ = np.linalg.svd(F, full_matrices=True)

n_bath, sv_gap, sv2_cov = adaptive_bath(sv, n_imp, config.MAX_EMBED_ORBS)
Q_bath = U_env[:, :n_bath]

n_emb = n_imp + n_bath
Q_emb = np.hstack([Q_imp, Q_bath])
C_emb = S_invsqrt @ Q_emb

print(f"  Impurity: {n_imp}  |  Bath: {n_bath}  |  Total: {n_emb} orbs = {2*n_emb} qubits")
print(f"  SV gap: {sv_gap:.4f}  |  sv² coverage: {sv2_cov:.4f}")

# ── Phase D: Core Mean-Field Potential ────────────────────────────────────────
print("\n── Phase D: Core Potential ──")
P_emb_lo   = Q_emb @ Q_emb.T
P_core_lo  = np.eye(n_ao) - P_emb_lo
dm_core_lo = P_core_lo @ dm_lo @ P_core_lo
dm_core    = S_invsqrt @ dm_core_lo @ S_invsqrt
dm_core    = 0.5 * (dm_core + dm_core.T)

h1e_bare = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
vj, vk   = pyscf_hf.get_jk(mol, dm_core, hermi=1)
h1e_eff  = h1e_bare + vj - 0.5 * vk

ecore = mol.energy_nuc() + 0.5 * float(np.einsum("ij,ji->", dm_core, h1e_bare + h1e_eff))
print(f"  E_core = {ecore:.6f} Ha")

# ── Phase E: Integral Transformation ─────────────────────────────────────────
print("\n── Phase E: Integral Transformation ──")
t0 = time.time()

h1e_emb = C_emb.T @ h1e_eff @ C_emb
h1e_emb = 0.5 * (h1e_emb + h1e_emb.T)

h2e_emb = ao2mo.kernel(mol, C_emb, compact=False).reshape(n_emb, n_emb, n_emb, n_emb)
h2e_emb = 0.5 * (h2e_emb + h2e_emb.transpose(1, 0, 2, 3))
h2e_emb = 0.5 * (h2e_emb + h2e_emb.transpose(0, 1, 3, 2))
h2e_emb = 0.5 * (h2e_emb + h2e_emb.transpose(2, 3, 0, 1))

n_alpha = nel // 2 + nel % 2
n_beta  = nel // 2

print(f"  h1e: {h1e_emb.shape}  h2e: {h2e_emb.shape}  ({time.time()-t0:.1f}s)")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"[Step 2] Summary: {mol_info['molecule']}")
print(f"  Embedding: {n_imp}(imp) + {n_bath}(bath) = {n_emb} orbs = {2*n_emb} qubits")
print(f"  Electrons: {nel} ({n_alpha}α + {n_beta}β)")
print(f"  sv² coverage: {sv2_cov:.4f}")
print(f"{'='*60}")

# ── Save ──────────────────────────────────────────────────────────────────────
results = {
    "h1e"        : h1e_emb,
    "h2e"        : h2e_emb,
    "ecore"      : ecore,
    "n_emb"      : n_emb,
    "n_imp"      : n_imp,
    "n_bath"     : n_bath,
    "n_alpha"    : n_alpha,
    "n_beta"     : n_beta,
    "sv"         : sv[:n_bath],
    "sv2_cov"    : sv2_cov,
    "uhf_energy" : float(mf.e_tot),
    "mp2_used"   : mp2_ok,
    "mp2_corr"   : float(e_corr),
    "mol_info"   : mol_info,
}

with open(config.STEP2_FILE, "wb") as f:
    pickle.dump(results, f)

print(f"\n[Step 2] ✓ Saved → {config.STEP2_FILE}")