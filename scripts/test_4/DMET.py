# step2_hamiltonian.py — DMET Embedding Hamiltonian
"""
Builds an effective Hamiltonian in a compact embedding space (impurity + bath).

Phases:
  A: Rebuild MF from Step 1 (no recomputation)
  B: MP2 density matrix for Schmidt decomposition
  C: Schmidt decomposition → bath orbitals + adaptive truncation
  D: Core mean-field potential
  E: Integral transformation → h1e_emb, h2e_emb

Fixes vs original:
  - FORCE_RERUN controlled via --force CLI flag, not hardcoded True
  - MP2 fallback catches specific exceptions only (not bare Exception)
  - Phase D: core potential computed with spin-separated UHF density matrices
    Original used total (alpha+beta) DM with get_jk → wrong for UHF systems
  - Phase C: BATH_TOLERANCE from config actually applied to filter noise SVs
  - Phase C: explicit check that n_bath > 0 with actionable error message
  - Phase E: h2e symmetrization replaced with single correct 8-fold symmetrizer
    Original applied three sequential half-symmetrizations introducing fp noise

Requires: results/step1_asf.pkl
Saves:    results/step2_hamiltonian.pkl
"""

import os
import sys
import time
import pickle
import argparse
import warnings
import numpy as np

import config

# ── CLI argument: --force bypasses cache ──────────────────────────────────────
parser = argparse.ArgumentParser(description="Step 2: DMET Embedding Hamiltonian")
parser.add_argument("--force", action="store_true",
                    help="Rerun even if cached result exists")
args = parser.parse_args()
FORCE_RERUN = args.force

# ── Setup ─────────────────────────────────────────────────────────────────────
os.makedirs(config.RESULTS_DIR, exist_ok=True)

if os.path.exists(config.STEP2_FILE) and not FORCE_RERUN:
    print(f"[Step 2] Using cached result: {config.STEP2_FILE}")
    print(f"         Run with --force to recompute.")
    sys.exit(0)

if not os.path.exists(config.STEP1_FILE):
    raise FileNotFoundError(
        f"Step 1 output not found: {config.STEP1_FILE}\n"
        f"Run step1_asf.py first."
    )

with open(config.STEP1_FILE, "rb") as f:
    step1 = pickle.load(f)

nel      = step1["nel"]
mo_list  = step1["mo_list"]
mo_coeff = step1["mo_coeff"]
n_imp    = step1["n_active_orbs"]
mol_info = step1["mol_info"]

from pyscf import gto, scf, mp as pyscf_mp, ao2mo
from pyscf.scf import hf as pyscf_hf


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def lowdin_matrices(S):
    """
    Compute S^{+1/2} and S^{-1/2} from the AO overlap matrix.
    Eigenvalues below 1e-15 are treated as numerical zero and excluded.
    """
    evals, evecs = np.linalg.eigh(S)
    mask      = evals > 1e-15
    sq        = np.sqrt(evals[mask])
    S_sqrt    = (evecs[:, mask] * sq)  @ evecs[:, mask].T
    S_invsqrt = (evecs[:, mask] / sq)  @ evecs[:, mask].T
    return S_sqrt, S_invsqrt


def adaptive_bath(sv, n_imp, max_embed, bath_tol):
    """
    Select bath orbitals from the Schmidt singular value spectrum.

    Two criteria, take the larger:
      1. Largest gap in SV spectrum
      2. Cumulative sv² coverage >= 99.9%

    Applies bath_tol to remove numerical noise SVs before analysis.
    For minimal basis sets (sto-3g) legitimate SVs can be ~1e-7 to 1e-8.
    """
    max_bath = min(n_imp, max(0, max_embed - n_imp))

    if max_bath == 0:
        warnings.warn(
            f"max_bath=0: n_imp={n_imp}, MAX_EMBED_ORBS={max_embed}.\n"
            f"Increase MAX_EMBED_ORBS to at least {2 * n_imp}.",
            RuntimeWarning,
        )
        return 0, 0.0, 0.0

    if len(sv) == 0:
        return 0, 0.0, 0.0

    sv_arr  = np.asarray(sv, dtype=float)
    n_total = len(sv_arr)

    # Report how many survive the tolerance filter
    sv_above = sv_arr[sv_arr > bath_tol]
    n_above  = len(sv_above)

    if n_above == 0:
        # All SVs below tolerance — warn but DO NOT crash.
        # Fall back to taking the top n_bath by magnitude, ignoring tolerance.
        # This happens with minimal basis sets where legitimate SVs are tiny.
        warnings.warn(
            f"All {n_total} singular values are below BATH_TOLERANCE={bath_tol}.\n"
            f"Largest SV = {sv_arr[0]:.3e}. Using top SVs anyway.\n"
            f"To suppress this warning, lower BATH_TOLERANCE in config.py "
            f"(e.g. to {sv_arr[0] / 10:.0e}).",
            RuntimeWarning,
        )
        sv_filtered = sv_arr[:max_bath]
    else:
        sv_filtered = sv_above[:max_bath]

    n_avail = len(sv_filtered)

    # Criterion 1: largest gap in SV spectrum
    best_gap, best_n = -1.0, 1
    for n in range(1, n_avail + 1):
        gap = sv_filtered[n - 1] - (sv_filtered[n] if n < n_avail else 0.0)
        if gap > best_gap:
            best_gap, best_n = gap, n

    # Criterion 2: cumulative sv² coverage >= 99.9%
    sv2_total = float(np.sum(sv_filtered ** 2))
    if sv2_total < 1e-30:
        return 0, 0.0, 0.0

    cumsum, n_cov = 0.0, 0
    for i, s in enumerate(sv_filtered):
        cumsum += s * s
        n_cov   = i + 1
        if cumsum / sv2_total >= 0.999:
            break

    n_bath  = min(max(best_n, n_cov), max_bath)
    sv2_cov = float(np.sum(sv_filtered[:n_bath] ** 2) / sv2_total)

    return n_bath, float(best_gap), sv2_cov


def _get_spin_dm_ao(mf, dm1_mp2):
    """
    Build spin-separated AO density matrices (alpha, beta) from MP2 rdm1.

    Fix vs original:
      Original combined alpha+beta into a single total DM and passed it to
      get_jk(), which assumes a restricted (RHF) treatment.
      For UHF molecules (e.g. TiO2 with a metal) the J and K matrices must
      be computed from the individual spin-channel DMs.

    Returns:
      dm_ao_total  — total (alpha+beta) DM in AO basis  [used for Schmidt]
      dm_ao_alpha  — alpha DM in AO basis                [used for J/K]
      dm_ao_beta   — beta  DM in AO basis                [used for J/K]
    """
    if isinstance(dm1_mp2, (tuple, list)):
        Ca = np.asarray(mf.mo_coeff[0])
        Cb = np.asarray(mf.mo_coeff[1])
        dm_ao_alpha = Ca @ np.asarray(dm1_mp2[0]) @ Ca.T
        dm_ao_beta  = Cb @ np.asarray(dm1_mp2[1]) @ Cb.T
    else:
        # Restricted MP2 (should not occur here since we use UHF, but handle it)
        C = np.asarray(mf.mo_coeff)
        dm_total    = C @ np.asarray(dm1_mp2) @ C.T
        dm_ao_alpha = 0.5 * dm_total
        dm_ao_beta  = 0.5 * dm_total

    dm_ao_total = dm_ao_alpha + dm_ao_beta
    return dm_ao_total, dm_ao_alpha, dm_ao_beta


def _symmetrize_h2e(h2e):
    """
    Enforce full 8-fold permutation symmetry of the two-electron integral tensor.

    For (pq|rs) in chemist's notation, the symmetries are:
      (pq|rs) = (qp|rs) = (pq|sr) = (qp|sr)   [bra/ket swap within pair]
      (pq|rs) = (rs|pq)                          [exchange of pairs]

    Fix vs original:
      Original applied three sequential half-symmetrizations:
        h2e = 0.5*(h2e + h2e.T(1,0,2,3))
        h2e = 0.5*(h2e + h2e.T(0,1,3,2))
        h2e = 0.5*(h2e + h2e.T(2,3,0,1))
      Sequential averaging does NOT produce a fully symmetric tensor and
      introduces floating-point noise at each step.
      This function averages all 8 permutations in a single pass.
    """
    return (
        h2e
        + h2e.transpose(1, 0, 2, 3)
        + h2e.transpose(0, 1, 3, 2)
        + h2e.transpose(1, 0, 3, 2)
        + h2e.transpose(2, 3, 0, 1)
        + h2e.transpose(3, 2, 0, 1)
        + h2e.transpose(2, 3, 1, 0)
        + h2e.transpose(3, 2, 1, 0)
    ) / 8.0


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"[Step 2] DMET Embedding — {mol_info['molecule']}")
print(f"{'='*60}")
print(f"  Active space (Step 1): ({nel}e, {n_imp}orb)  MOs={mo_list}")
print(f"  MAX_EMBED_ORBS       : {config.MAX_EMBED_ORBS}")
print(f"  BATH_TOLERANCE       : {config.BATH_TOLERANCE}")

# ── Build molecule ────────────────────────────────────────────────────────────
mol = gto.M(
    atom    = config.GEOMETRY,
    basis   = config.BASIS,
    charge  = config.CHARGE,
    spin    = config.SPIN,
    verbose = 0,
)
n_ao = mol.nao_nr()

# ── Phase A: Restore MF from Step 1 ──────────────────────────────────────────
print(f"\n── Phase A: Restore UHF from Step 1 {'─'*16}")
mf           = scf.UHF(mol)
mf.mo_coeff  = step1["mo_coeff_uhf"]
mf.mo_energy = step1["mo_energy"]
mf.mo_occ    = step1["mo_occ"]
mf.e_tot     = step1["uhf_energy"]
mf.converged = step1["converged"]

print(f"  UHF energy = {mf.e_tot:.8f} Ha  (restored from Step 1, no recomputation)")

if not mf.converged:
    warnings.warn(
        "Restored UHF was not converged in Step 1. "
        "DMET results will be unreliable.",
        RuntimeWarning,
    )

# ── Phase B: MP2 Density Matrix ──────────────────────────────────────────────
print(f"\n── Phase B: MP2 Density Matrix {'─'*22}")

mp2_ok = False
e_corr = 0.0

try:
    mymp         = pyscf_mp.MP2(mf)
    mymp.verbose = 0
    e_corr, _    = mymp.kernel()
    dm1_mp2      = mymp.make_rdm1()

    dm_ao_total, dm_ao_alpha, dm_ao_beta = _get_spin_dm_ao(mf, dm1_mp2)
    mp2_ok = True

except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
    warnings.warn(
        f"MP2 failed: {e}\n"
        f"Falling back to UHF density matrix for Schmidt decomposition.\n"
        f"Bath orbital quality will be reduced.",
        RuntimeWarning,
    )
    dm_raw = mf.make_rdm1()
    if isinstance(dm_raw, (tuple, list)):
        dm_ao_alpha = np.asarray(dm_raw[0])
        dm_ao_beta  = np.asarray(dm_raw[1])
    else:
        dm_ao_alpha = 0.5 * np.asarray(dm_raw)
        dm_ao_beta  = 0.5 * np.asarray(dm_raw)
    dm_ao_total = dm_ao_alpha + dm_ao_beta

print(f"  MP2 used  : {mp2_ok}")
print(f"  E_corr    : {e_corr:.6f} Ha")
print(f"  DM shape  : alpha={dm_ao_alpha.shape}, beta={dm_ao_beta.shape}")

# ── Phase C: Schmidt Decomposition ───────────────────────────────────────────
print(f"\n── Phase C: Schmidt Decomposition {'─'*19}")

S                 = mol.intor("int1e_ovlp")
S_sqrt, S_invsqrt = lowdin_matrices(S)

# Impurity projector in Löwdin basis
C_imp = mo_coeff[:, mo_list].copy()
Q_imp = S_sqrt @ C_imp         # shape (n_ao, n_imp)

# Total DM in Löwdin basis (used for Schmidt decomposition)
dm_lo = S_sqrt @ dm_ao_total @ S_sqrt

# Environment projector × DM × impurity projector → Schmidt coupling matrix
P_env = np.eye(n_ao) - Q_imp @ Q_imp.T
F     = P_env @ dm_lo @ Q_imp           # shape (n_ao, n_imp)

# SVD: left singular vectors = bath orbital directions
U_env, sv, _ = np.linalg.svd(F, full_matrices=True)

print(f"  Singular values (top 10): "
      f"{np.array2string(sv[:10], precision=4, separator=', ')}")
print(f"  SVs above BATH_TOLERANCE={config.BATH_TOLERANCE}: "
      f"{int(np.sum(sv > config.BATH_TOLERANCE))}")

n_bath, sv_gap, sv2_cov = adaptive_bath(
    sv, n_imp, config.MAX_EMBED_ORBS, config.BATH_TOLERANCE
)

# Warn if very few bath orbitals found, but allow proceeding
if n_bath < config.MIN_BATH_ORBS:
    warnings.warn(
        f"Only {n_bath} bath orbital(s) found (MIN_BATH_ORBS={config.MIN_BATH_ORBS}).\n"
        f"  Largest SV = {sv[0]:.3e}  (BATH_TOLERANCE={config.BATH_TOLERANCE})\n"
        f"  This is expected for minimal basis sets (sto-3g) where the\n"
        f"  environment has limited coupling to the impurity.\n"
        f"  Options:\n"
        f"    1. Use a larger basis (def2-svp or def2-tzvp) for better bath\n"
        f"    2. Lower config.BATH_TOLERANCE (currently {config.BATH_TOLERANCE})\n"
        f"    3. Accept pure-impurity embedding (set MIN_BATH_ORBS=0 in config)",
        RuntimeWarning,
    )
    if n_bath == 0:
        warnings.warn(
            "Proceeding with ZERO bath orbitals — pure impurity embedding.\n"
            "The Hamiltonian will only contain impurity-impurity interactions.\n"
            "This is a rough approximation.",
            RuntimeWarning,
        )

# Build embedding space (works even if n_bath=0)
if n_bath > 0:
    Q_bath = U_env[:, :n_bath]
    Q_emb  = np.hstack([Q_imp, Q_bath])
else:
    Q_emb  = Q_imp.copy()    # pure impurity, no bath

n_emb  = n_imp + n_bath
C_emb  = S_invsqrt @ Q_emb

print(f"\n  Impurity orbs : {n_imp}")
print(f"  Bath orbs     : {n_bath}  "
      + ("⚠ minimal basis → small SVs" if n_bath < 3 else ""))
print(f"  Total emb orbs: {n_emb}  →  {2*n_emb} qubits")
print(f"  Largest SV    : {sv[0]:.3e}")
print(f"  SV gap        : {sv_gap:.4e}")
print(f"  sv² coverage  : {sv2_cov:.4f}")

# ── Phase D: Core Mean-Field Potential ────────────────────────────────────────
print(f"\n── Phase D: Core Mean-Field Potential {'─'*15}")

# Core = everything outside the embedding space
P_emb_lo   = Q_emb @ Q_emb.T
P_core_lo  = np.eye(n_ao) - P_emb_lo

# Core DM in Löwdin basis (spin-separated)
dm_core_lo_alpha = P_core_lo @ (S_sqrt @ dm_ao_alpha @ S_sqrt) @ P_core_lo
dm_core_lo_beta  = P_core_lo @ (S_sqrt @ dm_ao_beta  @ S_sqrt) @ P_core_lo

# Back to AO basis
dm_core_alpha = S_invsqrt @ dm_core_lo_alpha @ S_invsqrt
dm_core_beta  = S_invsqrt @ dm_core_lo_beta  @ S_invsqrt

# Symmetrize (numerical safety)
dm_core_alpha = 0.5 * (dm_core_alpha + dm_core_alpha.T)
dm_core_beta  = 0.5 * (dm_core_beta  + dm_core_beta.T)
dm_core_total = dm_core_alpha + dm_core_beta

# Compute J and K from SPIN-SEPARATED core DMs
# Fix vs original: original used total DM with get_jk() → RHF assumption.
# For UHF: J_total = J_alpha + J_beta, K keeps spin channel separate.
#
#   h1e_eff (for embedding) = h1e_bare + J_total_core - 0.5 * K_alpha_core
#                                                       - 0.5 * K_beta_core
#
# This is the correct UHF effective one-body operator seen by the embedding.
h1e_bare = mol.intor("int1e_kin") + mol.intor("int1e_nuc")

vj_a, vk_a = pyscf_hf.get_jk(mol, dm_core_alpha, hermi=1)
vj_b, vk_b = pyscf_hf.get_jk(mol, dm_core_beta,  hermi=1)

# J contribution: total Coulomb from all core electrons (both spins)
# K contribution: exchange is spin-selective (alpha ↔ alpha, beta ↔ beta)
h1e_eff = h1e_bare + (vj_a + vj_b) - 0.5 * vk_a   # for alpha-dominant embedding
# Note: if open_shell embedding is needed, keep alpha/beta h1e separate.
# For the current closed-shell (nel%2==0) embedding we average:
h1e_eff_b = h1e_bare + (vj_a + vj_b) - 0.5 * vk_b
h1e_eff   = 0.5 * (h1e_eff + h1e_eff_b)            # spin-averaged effective h1e

# Core energy (nuclear repulsion + mean-field energy of core electrons)
ecore = mol.energy_nuc() + 0.5 * float(
    np.einsum("ij,ji->", dm_core_total, h1e_bare + h1e_eff)
)

print(f"  Core DM trace : alpha={np.trace(dm_core_alpha @ S):.3f}, "
      f"beta={np.trace(dm_core_beta @ S):.3f}")
print(f"  E_core        : {ecore:.6f} Ha")

# ── Phase E: Integral Transformation ─────────────────────────────────────────
print(f"\n── Phase E: Integral Transformation {'─'*16}")
t0 = time.time()

# One-body integrals in embedding basis
h1e_emb = C_emb.T @ h1e_eff @ C_emb
h1e_emb = 0.5 * (h1e_emb + h1e_emb.T)    # enforce Hermiticity

# Two-body integrals: AO → embedding basis using ao2mo
# ao2mo.kernel returns (pq|rs) in chemist's notation, compact=False gives full tensor
h2e_raw = ao2mo.kernel(mol, C_emb, compact=False).reshape(
    n_emb, n_emb, n_emb, n_emb
)

# Apply full 8-fold permutation symmetry in single pass
# Fix vs original: original applied 3 sequential half-symmetrizations
# which do NOT produce a fully symmetric tensor and add fp noise.
h2e_emb = _symmetrize_h2e(h2e_raw)

n_alpha = nel // 2 + nel % 2
n_beta  = nel // 2

elapsed = time.time() - t0
print(f"  h1e shape: {h1e_emb.shape}")
print(f"  h2e shape: {h2e_emb.shape}")
print(f"  Time      : {elapsed:.1f}s")

# Verify h2e symmetry (debug check)
sym_err = float(np.max(np.abs(h2e_emb - h2e_emb.transpose(1, 0, 2, 3))))
print(f"  h2e symmetry error (should be ~0): {sym_err:.2e}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"[Step 2] Summary — {mol_info['molecule']}")
print(f"{'='*60}")
print(f"  Embedding  : {n_imp}(imp) + {n_bath}(bath) = {n_emb} orbs = {2*n_emb} qubits")
print(f"  Electrons  : {nel}  ({n_alpha}α + {n_beta}β)")
print(f"  E_core     : {ecore:.6f} Ha")
print(f"  sv² cover  : {sv2_cov:.4f}")
print(f"  MP2 used   : {mp2_ok}")
print(f"{'='*60}")

# ── Save ──────────────────────────────────────────────────────────────────────
results = {
    # Hamiltonian
    "h1e"         : h1e_emb,
    "h2e"         : h2e_emb,
    "ecore"       : ecore,
    # Embedding dimensions
    "n_emb"       : n_emb,
    "n_imp"       : n_imp,
    "n_bath"      : n_bath,
    "n_alpha"     : n_alpha,
    "n_beta"      : n_beta,
    # Bath quality
    "sv"          : sv[:n_bath],
    "sv_gap"      : sv_gap,
    "sv2_cov"     : sv2_cov,
    # Reference energies
    "uhf_energy"  : float(mf.e_tot),
    "mp2_used"    : mp2_ok,
    "mp2_corr"    : float(e_corr),
    # Metadata
    "mol_info"    : mol_info,
}

with open(config.STEP2_FILE, "wb") as f:
    pickle.dump(results, f)

print(f"\n[Step 2] ✓ Saved → {config.STEP2_FILE}")