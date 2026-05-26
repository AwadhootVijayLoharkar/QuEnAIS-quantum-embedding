"""
Step 2 — One-shot DMET Embedding
=================================

Philosophy
──────────
ASF (Step 1) identified the n_imp most correlated orbitals → impurity.
These orbitals are NOT isolated: Fe 3d is bonded to N 2p.
We cannot just take h1e/h2e for those orbitals in vacuum → wrong physics.

DMET solves this by finding the minimal set of environment orbitals
that are quantum-mechanically entangled with the impurity → bath.
Everything else (core) is traced out and contributes only as a
mean-field background electric field to h1e_eff.

Result: an exact-in-principle Hamiltonian in (n_imp + n_bath) orbitals.
        This is what we feed to SQD — no self-consistent loop needed.

Phases
──────
  A  UHF with Newton fallback (need converged MF for everything below)
  B  MP2 density matrix (better than UHF for Schmidt decomp on TM complexes)
  C  Schmidt decomposition + adaptive bath truncation
  D  Core mean-field potential → h1e_eff
  E  Embedding validation (orthonormality, electron count, hermiticity)
  F  Integral transformation → h1e_emb, h2e_emb
  G  Chemical potential correction (if electron count deviates)
  H  FCI sanity check + rich embedding quality scores

Requires : results/step1_asf.pkl
Saves    : results/step2_hamiltonian.pkl
Runtime  : ~1-3 min
"""

import os
import sys
import time
import math
import pickle
import numpy as np
import pyscf.scf.hf as _pyscf_hf_base

import config

# ── Cache check ───────────────────────────────────────────────────────────────
FORCE_RERUN = True   # set True to redo this step

os.makedirs(config.RESULTS_DIR, exist_ok=True)

if os.path.exists(config.STEP2_FILE) and not FORCE_RERUN:
    print(f"[Step 2] Cached results at {config.STEP2_FILE}")
    print("  Set FORCE_RERUN = True to rerun.")
    sys.exit(0)

if not os.path.exists(config.STEP1_FILE):
    raise FileNotFoundError(
        f"[Step 2] Step 1 results not found: {config.STEP1_FILE}\n"
        "Run step1_asf.py first."
    )

# ── Load Step 1 ───────────────────────────────────────────────────────────────
with open(config.STEP1_FILE, "rb") as fh:
    step1 = pickle.load(fh)

nel           = step1["nel"]
mo_list       = step1["mo_list"]
mo_coeff      = step1["mo_coeff"]   # full (n_AO × n_MO) MP2 NO matrix
n_active_orbs = step1["n_active_orbs"]
scores_s1     = step1["scores"]
mol_info      = step1["mol_info"]

print(f"\n{'='*62}")
print(f"[Step 2] One-shot DMET Embedding — {mol_info['molecule']}")
print(f"{'='*62}")
print(f"\n  Loaded from Step 1:")
print(f"    Active space        : {nel}e in {n_active_orbs} orbs")
print(f"    Impurity mo_list    : {mo_list}")
print(f"    Complexity tier     : {scores_s1['tier_used']}")
print(f"    UHF energy          : {scores_s1['uhf_energy_Ha']:.8f} Ha")
print(f"    Max embedding orbs  : {config.MAX_EMBED_ORBS} → "
      f"{2*config.MAX_EMBED_ORBS} qubits")

# ── Environment ───────────────────────────────────────────────────────────────
os.environ["BLOCKEXE"]            = config.BLOCKEXE_WRAPPER
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_DEBUG_CPU_TYPE"]  = "5"

from pyscf.dmrgscf import dmrgci
dmrgci.settings.BLOCKEXE = config.BLOCKEXE_WRAPPER

from pyscf import gto, scf as pyscf_scf, mp as pyscf_mp
from pyscf import ao2mo, fci as pyscf_fci
from pyscf.scf import hf as _scf_hf
from scipy.optimize import brentq


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def build_lowdin(S):
    """
    Compute S^{+1/2} and S^{-1/2} via eigendecomposition of the AO
    overlap matrix S.

    Löwdin symmetric orthogonalisation minimises the change from the
    original AO basis while producing orthonormal orbitals — each
    Löwdin orbital looks as much as possible like its parent AO.

    Used throughout Step 2:
      Q_lo = S^{+1/2} @ C_AO     (AO → Löwdin basis, orthonormal)
      C_AO = S^{-1/2} @ Q_lo     (Löwdin → AO basis)
    """
    evals, evecs = np.linalg.eigh(S)
    mask    = evals > 1e-15            # drop numerically zero eigenvalues
    sq      = np.sqrt(evals[mask])
    S_sq    = (evecs[:, mask] * sq)    @ evecs[:, mask].T
    S_isq   = (evecs[:, mask] / sq)    @ evecs[:, mask].T
    return S_sq, S_isq


def get_dm_ao(mf, mol):
    """
    Return the best available one-particle density matrix in the AO basis.

    Tries MP2 first. MP2 accounts for pair correlations between electrons
    which UHF ignores. For TM complexes this materially changes which N
    orbitals appear in the Schmidt bath:
      UHF DM: reflects average occupations only
      MP2 DM: includes charge-transfer fluctuations → better bath

    Falls back to UHF DM if MP2 fails (rare; usually a convergence issue
    with very diffuse basis sets).

    Returns
    -------
    dm_ao    : (n_AO, n_AO) ndarray  total (α+β) density matrix
    mp2_ok   : bool                  True if MP2 succeeded
    e_corr   : float                 MP2 correlation energy (0.0 if failed)
    """
    try:
        mymp        = pyscf_mp.MP2(mf)
        mymp.verbose = 0
        e_corr, _   = mymp.kernel()
        dm1_mo      = mymp.make_rdm1()

        if isinstance(dm1_mo, (tuple, list)):
            # UHF: separate α/β MO spaces
            Ca, Cb = np.asarray(mf.mo_coeff[0]), np.asarray(mf.mo_coeff[1])
            dm_ao  = Ca @ dm1_mo[0] @ Ca.T + Cb @ dm1_mo[1] @ Cb.T
        else:
            dm_ao = mf.mo_coeff @ dm1_mo @ mf.mo_coeff.T

        print(f"    MP2 correlation energy : {e_corr:+.6f} Ha")
        print(f"    MP2 DM trace (= N_elec): {np.trace(dm_ao):.4f}  "
              f"(expected {mol.nelectron})")
        return dm_ao, True, float(e_corr)

    except Exception as exc:
        print(f"    [MP2] Failed ({exc}) — using UHF density matrix.")
        dm_raw = mf.make_rdm1()
        dm_ao  = (dm_raw[0] + dm_raw[1]) if isinstance(dm_raw, tuple) else dm_raw
        return dm_ao, False, 0.0


def adaptive_bath(sv, n_imp, max_embed):
    """
    Decide how many bath orbitals to keep from the Schmidt singular values.

    Two criteria are computed; the more generous (larger n_bath) wins so
    that we never discard an important bath orbital.

    Criterion 1 — SV gap detection
      Same principle as Step 1 entropy gap detection.
      Sort SVs descending; find the largest gap; cut there.
      Physical meaning: the gap separates 'strongly entangled' bath
      orbitals (above gap) from 'weakly entangled' ones (below gap).

    Criterion 2 — Cumulative sv² coverage
      sv²_i = fraction of total impurity-environment entanglement
      captured by bath orbital i.
      Keep bath orbitals until Σ sv²_i / Σ_all sv²_i ≥ 0.999.
      Ensures 99.9 % of the entanglement is represented.

    Hard constraints
      n_bath ≤ n_imp      (DMET theorem: bath rank ≤ impurity rank)
      n_bath ≤ max_embed - n_imp  (qubit budget)

    Returns
    -------
    n_bath     : int    number of bath orbitals to keep
    sv_gap     : float  value of largest gap used in criterion 1
    sv2_cov    : float  fraction of entanglement captured (0–1)
    """
    SV2_TARGET = 0.999

    max_bath = min(n_imp, max(0, max_embed - n_imp))
    if max_bath == 0 or len(sv) == 0:
        return 0, 0.0, 0.0

    sv_arr  = np.asarray(sv[:max_bath], dtype=float)
    n_avail = len(sv_arr)

    # ── Criterion 1: gap detection ─────────────────────────────────────────
    best_gap = -1.0
    best_n   = 1
    for n in range(1, n_avail + 1):
        gap = float(sv_arr[n-1] - sv_arr[n]) if n < n_avail else float(sv_arr[n-1])
        if gap > best_gap:
            best_gap, best_n = gap, n

    # ── Criterion 2: sv² coverage ──────────────────────────────────────────
    sv2_total = float(np.sum(sv_arr ** 2))
    if sv2_total < 1e-20:
        return 0, 0.0, 0.0

    cumsum, n_cov = 0.0, 0
    for i, s in enumerate(sv_arr):
        cumsum += s * s
        n_cov   = i + 1
        if cumsum / sv2_total >= SV2_TARGET:
            break

    # Take the larger → never under-represent the bath
    n_bath  = min(max(best_n, n_cov), max_bath)
    sv2_cov = float(np.sum(sv_arr[:n_bath] ** 2) / sv2_total)

    return int(n_bath), float(best_gap), float(sv2_cov)


def validate_embedding(C_emb, S, dm_lo, Q_emb):
    """
    Three quick numerical health checks on the embedding:

    Check 1 — Orthonormality of embedding orbitals
      C_emb.T @ S @ C_emb should equal identity.
      Failure → Löwdin transform accumulated numerical error.

    Check 2 — Electron count in embedding
      n_elec = Tr[Q_emb.T @ ρ_lo @ Q_emb]
      Should be close to an integer.
      Failure → one-shot DMET approximation is poor for this system.

    Check 3 — Hermiticity of DM in embedding
      ρ_emb = Q_emb.T @ ρ_lo @ Q_emb  should be Hermitian.

    Returns a dict with all values and boolean pass/fail flags.
    """
    res = {}

    # Check 1
    orth_err = float(np.max(np.abs(C_emb.T @ S @ C_emb
                                   - np.eye(C_emb.shape[1]))))
    res["orth_err"]    = orth_err
    res["orth_ok"]     = orth_err < 1e-6

    # Check 2
    dm_emb     = Q_emb.T @ dm_lo @ Q_emb
    n_elec_emb = float(np.real(np.trace(dm_emb)))
    elec_dev   = abs(n_elec_emb - round(n_elec_emb))
    res["n_elec_emb"] = n_elec_emb
    res["elec_dev"]   = elec_dev
    res["elec_ok"]    = elec_dev < 0.15

    # Check 3
    herm_err = float(np.max(np.abs(dm_emb - dm_emb.T)))
    res["herm_err"] = herm_err
    res["herm_ok"]  = herm_err < 1e-8

    return res


def chemical_potential_correction(h1e, n_emb, n_target):
    """
    Shift h1e by -μ·I so that the mean-field electron count equals n_target.

    Without this, the one-shot embedding is in the grand-canonical ensemble
    (fractional electrons). The correction enforces the canonical ensemble
    (fixed N), which is what FCI and SQD assume.

    Method: bisection on μ using the aufbau filling of h1e eigenvalues.
    Fast (no CI needed); converges in < 50 iterations.

    Only called when |n_elec_emb - round(n_elec_emb)| > 0.15 (from validation).

    Returns
    -------
    h1e_corrected : ndarray   h1e with −μ·I applied
    mu            : float     chemical potential used
    """
    def n_of_mu(mu):
        evals = np.linalg.eigvalsh(h1e - mu * np.eye(n_emb))
        return float(np.sum(evals < 0.0) * 2)   # factor 2 for spin

    # Search bracket: ± 10 Ha around mean diagonal
    mu_0 = float(np.mean(np.diag(h1e)))
    try:
        mu = brentq(lambda m: n_of_mu(m) - n_target,
                    mu_0 - 10.0, mu_0 + 10.0, xtol=1e-6, maxiter=100)
        print(f"    μ = {mu:+.6f} Ha  "
              f"(N: {n_of_mu(0.0):.2f} → {n_of_mu(mu):.2f} ≈ {n_target})")
        return h1e - mu * np.eye(n_emb), float(mu)
    except Exception as exc:
        print(f"    Bisection failed ({exc}) — no correction applied.")
        return h1e, 0.0


# ═════════════════════════════════════════════════════════════════════════════
# Phase A: UHF with Newton fallback
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Phase A: UHF ──")

mol = gto.M(
    atom    = config.GEOMETRY,
    basis   = config.BASIS,
    charge  = 0,
    spin    = 0,
    verbose = 0,
)

_orig_kernel = _pyscf_hf_base.SCF.kernel


def _newton_fallback(self, dm0=None, **kwargs):
    """
    DIIS + level-shift first. If not converged, retry with the Newton
    (second-order) solver warm-started from the DIIS MO coefficients.
    Needed for FeN6 where the open Fe 3d shell causes DIIS oscillations.
    """
    self.max_cycle   = max(getattr(self, "max_cycle",   50), 400)
    self.level_shift = max(getattr(self, "level_shift", 0.0), 0.5)
    result = _orig_kernel(self, dm0=dm0, **kwargs)
    if not self.converged:
        print("    [UHF] DIIS not converged → Newton solver...")
        try:
            nw           = self.newton()
            nw.max_cycle = 400
            nw.kernel(self.mo_coeff)
            if nw.converged:
                print(f"    [UHF] ✓ Newton: E = {nw.e_tot:.8f} Ha")
                for a in ("e_tot", "mo_coeff", "mo_energy", "mo_occ", "converged"):
                    setattr(self, a, getattr(nw, a))
            else:
                print("    [UHF] Newton also did not converge.")
        except Exception as exc:
            print(f"    [UHF] Newton error: {exc}")
    return result


_pyscf_hf_base.SCF.kernel = _newton_fallback

t_uhf = time.time()
print("\n  Running UHF...")
mf = pyscf_scf.UHF(mol)
mf.kernel()
_pyscf_hf_base.SCF.kernel = _orig_kernel   # restore immediately

print(f"  UHF : E = {mf.e_tot:.8f} Ha | converged = {mf.converged} | "
      f"{time.time()-t_uhf:.1f}s")

if not mf.converged:
    print("  WARNING: UHF did not converge. Results may be unreliable.")

n_AO  = mol.nao_nr()
n_imp = n_active_orbs   # number of impurity orbitals from Step 1

# ═════════════════════════════════════════════════════════════════════════════
# Phase B: MP2 density matrix for Schmidt decomposition
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Phase B: MP2 Density Matrix ──")
print("  (captures correlated Fe-N entanglement better than UHF alone)")

t_mp2 = time.time()
dm_ao, mp2_ok, e_corr_mp2 = get_dm_ao(mf, mol)
print(f"  Done in {time.time()-t_mp2:.1f}s | MP2 used = {mp2_ok}")

# ═════════════════════════════════════════════════════════════════════════════
# Phase C: Schmidt decomposition + adaptive bath truncation
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Phase C: Schmidt Decomposition + Adaptive Bath Truncation ──")

# ── Löwdin basis ──────────────────────────────────────────────────────────────
S             = mol.intor('int1e_ovlp')
S_sq, S_isq   = build_lowdin(S)

# ── Impurity in Löwdin basis ──────────────────────────────────────────────────
# mo_coeff[:, mo_list[k]] = k-th impurity orbital in AO basis (MP2 NO)
# Q_imp  = same in Löwdin basis → orthonormal columns by construction
C_imp = mo_coeff[:, mo_list].copy()   # (n_AO, n_imp)

# Safety: verify S-orthonormality; re-orthogonalise if needed
orth_imp = float(np.max(np.abs(C_imp.T @ S @ C_imp - np.eye(n_imp))))
if orth_imp > 1e-6:
    print(f"  Re-orthogonalising impurity (S-orth error = {orth_imp:.1e})...")
    Q_tmp, _ = np.linalg.qr(S_sq @ C_imp)
    C_imp    = S_isq @ Q_tmp[:, :n_imp]

Q_imp = S_sq @ C_imp   # (n_AO, n_imp) Löwdin basis

# ── DM in Löwdin basis ────────────────────────────────────────────────────────
# dm_lo is symmetric, Tr = N_electrons
dm_lo = S_sq @ dm_ao @ S_sq
print(f"  DM trace (Löwdin) : {np.trace(dm_lo):.4f}  "
      f"(expected {mol.nelectron})")

# ── Off-diagonal DM block (environment × impurity) ───────────────────────────
#
# F[i,j] measures how much environment orbital i shares density with
# impurity orbital j.
#
# SVD of F:
#   Large singular value sv_k → environment orbital k is strongly
#   entangled with the impurity → must be included as a bath orbital.
#   Small sv_k → barely entangled → can be discarded into core.
#
P_env        = np.eye(n_AO) - Q_imp @ Q_imp.T
F            = P_env @ dm_lo @ Q_imp       # (n_AO, n_imp)
U_env, sv, _ = np.linalg.svd(F, full_matrices=True)

print(f"\n  Schmidt singular values (first {min(16, len(sv))}):")
print(f"  {np.round(sv[:min(16, len(sv))], 5)}")

# ── Adaptive bath selection ───────────────────────────────────────────────────
n_bath, sv_gap, sv2_cov = adaptive_bath(sv, n_imp, config.MAX_EMBED_ORBS)

Q_bath = U_env[:, :n_bath]   # (n_AO, n_bath) Löwdin basis

n_emb  = n_imp + n_bath
Q_emb  = np.hstack([Q_imp, Q_bath])   # (n_AO, n_emb) Löwdin, orthonormal
C_emb  = S_isq @ Q_emb                # (n_AO, n_emb) AO basis

print(f"\n  Bath selection (adaptive gap + sv² coverage):")
print(f"    n_imp          = {n_imp}")
print(f"    n_bath         = {n_bath}  "
      f"(SV gap = {sv_gap:.4f},  sv² coverage = {sv2_cov:.4f})")
print(f"    n_emb          = {n_emb}  →  {2*n_emb} qubits")

# ═════════════════════════════════════════════════════════════════════════════
# Phase D: Core mean-field potential
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Phase D: Core Mean-Field Potential ──")

# Core = everything outside the embedding space.
# Core electrons are NOT treated quantum-mechanically.
# They contribute a Coulomb + Exchange (J − ½K) field to h1e_eff.
# This is exactly what you would get if you "froze" those electrons
# at the mean-field level.
P_emb      = Q_emb @ Q_emb.T
P_core     = np.eye(n_AO) - P_emb
dm_core_lo = P_core @ dm_lo @ P_core
dm_core    = S_isq @ dm_core_lo @ S_isq
dm_core    = 0.5 * (dm_core + dm_core.T)   # symmetrize numerical noise

n_elec_core  = float(np.real(np.trace(dm_core @ S)))
n_elec_embed = float(np.real(np.trace(dm_lo)) - np.trace(P_core @ dm_lo))
print(f"  Core electrons (approx)  : {n_elec_core:.2f}")
print(f"  Embedding electrons      : {n_elec_embed:.2f}  "
      f"(target: {nel})")

# Bare 1e integrals
h1e_AO = mol.intor('int1e_kin') + mol.intor('int1e_nuc')

# Mean-field potential from core electrons
# h1e_eff[p,q] = h1e_bare[p,q] + J_core[p,q] - 0.5 * K_core[p,q]
#   J_core[p,q] = Σ_{μν} ρ_core[μν] (pq|μν)   Coulomb
#   K_core[p,q] = Σ_{μν} ρ_core[μν] (pμ|νq)   Exchange
vj, vk       = _scf_hf.get_jk(mol, dm_core, hermi=1)
h1e_eff_AO   = h1e_AO + vj - 0.5 * vk

# Core energy constant
# E_core = E_nuc + ½ Tr[ρ_core (h1e_bare + h1e_eff)]
# Not needed for SQD vs FCI comparison (both use the same h1e_emb)
# but stored for reference/debugging.
ecore  = mol.energy_nuc()
ecore += 0.5 * float(np.einsum('ij,ji->', dm_core, h1e_AO + h1e_eff_AO))
print(f"  ecore                    : {ecore:.6f} Ha  (constant, not used by SQD)")

# ═════════════════════════════════════════════════════════════════════════════
# Phase E: Validation
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Phase E: Embedding Validation ──")

val = validate_embedding(C_emb, S, dm_lo, Q_emb)

print(f"\n  Orthonormality error : {val['orth_err']:.2e}  "
      f"{'✓' if val['orth_ok'] else '⚠  re-orthogonalising...'}")
print(f"  Electron count       : {val['n_elec_emb']:.4f}  "
      f"(dev from integer = {val['elec_dev']:.4f}  "
      f"{'✓' if val['elec_ok'] else '⚠  will apply μ correction'})")
print(f"  DM hermiticity error : {val['herm_err']:.2e}  "
      f"{'✓' if val['herm_ok'] else '⚠  WARNING'}")

# Re-orthogonalise embedding basis if needed
if not val["orth_ok"]:
    print("  Re-orthogonalising C_emb via QR...")
    Q_tmp, _  = np.linalg.qr(S_sq @ C_emb)
    C_emb     = S_isq @ Q_tmp[:, :n_emb]

# Decide active electron count
# Use nel from Step 1 as the canonical target — it uses a validated
# electron counting method (core subtraction) that is robust for TM.
n_alpha          = nel // 2 + nel % 2
n_beta           = nel // 2
apply_mu_corr    = not val["elec_ok"]

print(f"\n  Target electron count    : {nel} ({n_alpha}α + {n_beta}β)  "
      f"[from Step 1]")

# ═════════════════════════════════════════════════════════════════════════════
# Phase F: Integral transformation AO → embedding MO basis
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Phase F: Integral Transformation ──")
print(f"  AO basis: {n_AO} functions  →  embedding: {n_emb} orbitals")

t_int = time.time()

# h1e: O(N²) — trivial
# Transform the effective 1e Hamiltonian (including core field) into
# the embedding MO basis.
h1e_emb = C_emb.T @ h1e_eff_AO @ C_emb   # (n_emb, n_emb)
h1e_emb = 0.5 * (h1e_emb + h1e_emb.T)    # enforce Hermiticity

# h2e: O(N_AO⁴) — main cost for this step
# ao2mo.kernel transforms all 4 integral indices simultaneously.
# For STO-3G (48 AOs), 16 embedding orbitals: < 5 seconds.
# For cc-pVDZ (150 AOs): use density fitting (future improvement).
h2e_emb = ao2mo.kernel(mol, C_emb, compact=False)
h2e_emb = h2e_emb.reshape(n_emb, n_emb, n_emb, n_emb)

# Restore 8-fold permutation symmetry
# h2e[p,q,r,s] = h2e[q,p,r,s] = h2e[p,q,s,r] = h2e[r,s,p,q]
# Numerical noise from the transformation can slightly break this.
h2e_emb = 0.5 * (h2e_emb + h2e_emb.transpose(1, 0, 2, 3))
h2e_emb = 0.5 * (h2e_emb + h2e_emb.transpose(0, 1, 3, 2))
h2e_emb = 0.5 * (h2e_emb + h2e_emb.transpose(2, 3, 0, 1))

print(f"  h1e : {h1e_emb.shape}  h2e : {h2e_emb.shape}")
print(f"  Done in {time.time()-t_int:.1f}s")

# ═════════════════════════════════════════════════════════════════════════════
# Phase G: Chemical potential correction
# ═════════════════════════════════════════════════════════════════════════════
mu_correction = 0.0

if apply_mu_corr:
    print("\n── Phase G: Chemical Potential Correction ──")
    print(f"  Electron deviation {val['elec_dev']:.4f} > 0.15 → applying μ correction")
    h1e_emb, mu_correction = chemical_potential_correction(
        h1e_emb, n_emb, nel
    )
else:
    print("\n── Phase G: Chemical Potential Correction ── (skipped, not needed)")

# ═════════════════════════════════════════════════════════════════════════════
# Phase H: FCI sanity check + embedding quality scores
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Phase H: FCI Sanity Check + Quality Scores ──")

max_dets = math.comb(n_emb, n_alpha) ** 2
print(f"  {n_emb} orbs | {n_alpha}α + {n_beta}β | {2*n_emb} qubits")
print(f"  Max FCI determinants : C({n_emb},{n_alpha})² = {max_dets:,}")

fci_ref_e = None

if max_dets <= 5_000_000:
    print(f"  Running FCI ({max_dets:,} determinants)...")
    t_fci      = time.time()
    cisolver   = pyscf_fci.direct_spin1.FCI()
    fci_ref_e, _ = cisolver.kernel(
        h1e_emb, h2e_emb, n_emb, (n_alpha, n_beta)
    )
    print(f"  FCI done in {time.time()-t_fci:.1f}s")
    print(f"  FCI embedded energy  : {fci_ref_e:.8f} Ha  ← SQD target")
    print(f"  FCI total energy     : {fci_ref_e + ecore:.8f} Ha")
    print(f"  UHF total energy     : {mf.e_tot:.8f} Ha")
    print(f"  Embedding corr. E    : {fci_ref_e + ecore - mf.e_tot:+.8f} Ha")
else:
    print(f"  FCI skipped ({max_dets:,} dets > 5M threshold) — SQD will solve this")

# ── Embedding quality score vector ───────────────────────────────────────────
# These are stored alongside the Hamiltonian and fed to the scoring pipeline.
# They describe HOW GOOD the embedding is, independent of the quantum solver.
scores_emb = {
    # ── Embedding structure ────────────────────────────────────────────────
    "n_imp"                 : int(n_imp),
    "n_bath"                : int(n_bath),
    "n_emb"                 : int(n_emb),
    "n_qubits"              : int(2 * n_emb),
    "n_alpha"               : int(n_alpha),
    "n_beta"                : int(n_beta),

    # ── Bath quality ───────────────────────────────────────────────────────
    # sv2_coverage: 1.0 = complete bath, < 0.9 = significant truncation
    # Higher = more of the impurity-environment entanglement is captured
    "sv2_coverage"          : float(sv2_cov),
    "sv_gap"                : float(sv_gap),
    "bath_fraction"         : float(n_bath / max(1, n_imp)),
    "mp2_dm_used"           : bool(mp2_ok),

    # ── Electron count quality ─────────────────────────────────────────────
    # electron_deviation: how far n_elec_emb is from an integer
    # Close to 0 = embedding is well-defined; large = one-shot DMET poor
    "electron_deviation"    : float(val["elec_dev"]),
    "mu_correction"         : float(mu_correction),
    "electron_count_ok"     : bool(val["elec_ok"]),

    # ── Numerical quality ─────────────────────────────────────────────────
    "orth_error"            : float(val["orth_err"]),
    "herm_error"            : float(val["herm_err"]),

    # ── Energy scores ─────────────────────────────────────────────────────
    # embedding_corr_energy: how much correlation energy is in the embedding
    # More negative = more strongly correlated = more interesting for QC
    "uhf_energy"            : float(mf.e_tot),
    "ecore"                 : float(ecore),
    "mp2_correlation"       : float(e_corr_mp2),
    "fci_embedding_energy"  : float(fci_ref_e) if fci_ref_e is not None else None,
    "embedding_corr_energy" : float(fci_ref_e + ecore - mf.e_tot)
                              if fci_ref_e is not None else None,
}

print(f"\n  Embedding Quality Scores:")
print(f"  {'─'*50}")
for k, v in scores_emb.items():
    if isinstance(v, float):
        print(f"  {k:<30} {v:.6f}")
    else:
        print(f"  {k:<30} {v}")

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print(f"[Step 2] Summary for {mol_info['molecule']}")
print(f"  Impurity (ASF)     : {n_imp} orbitals  {mo_list}")
print(f"  Bath (Schmidt)     : {n_bath} orbitals  "
      f"(sv² coverage = {sv2_cov:.3f})")
print(f"  Total embedding    : {n_emb} orbitals = {2*n_emb} qubits")
print(f"  Electrons          : {nel} ({n_alpha}α + {n_beta}β)")
print(f"  MP2 DM used        : {mp2_ok}")
print(f"  μ correction       : {mu_correction:+.6f} Ha")
if fci_ref_e is not None:
    print(f"  FCI reference      : {fci_ref_e:.8f} Ha  ← SQD target")
    print(f"  Correlation energy : {fci_ref_e + ecore - mf.e_tot:+.8f} Ha")
else:
    print(f"  FCI reference      : not computed ({max_dets:,} dets > 5M)")
print(f"{'='*62}")

# ═════════════════════════════════════════════════════════════════════════════
# Save
# ═════════════════════════════════════════════════════════════════════════════
results = {
    # ── Core output (consumed by step3_sqd.py) ────────────────────────────
    "h1e"             : h1e_emb,       # (n_emb, n_emb) effective 1e integrals
    "h2e"             : h2e_emb,       # (n_emb,)*4 two-electron integrals
    "ecore"           : ecore,         # constant energy offset
    "n_emb"           : n_emb,
    "n_imp"           : n_imp,
    "n_bath"          : n_bath,
    "n_alpha"         : n_alpha,
    "n_beta"          : n_beta,
    "fci_ref_e"       : fci_ref_e,     # FCI on same H → SQD target

    # ── Schmidt singular values (diagnostics) ─────────────────────────────
    "sv"              : sv[:n_bath],

    # ── Quality scores (scoring pipeline) ────────────────────────────────
    "scores"          : scores_emb,

    # ── Metadata ─────────────────────────────────────────────────────────
    "uhf_energy"      : float(mf.e_tot),
    "mp2_correlation" : float(e_corr_mp2),
    "mu_correction"   : float(mu_correction),
    "mp2_dm_used"     : bool(mp2_ok),
}

with open(config.STEP2_FILE, "wb") as fh:
    pickle.dump(results, fh)

print(f"\n[Step 2] ✓ Saved → {config.STEP2_FILE}")