"""
Step 2 — One-shot DMET: build embedded Hamiltonian from ASF orbitals

Philosophy
──────────
ASF gives us the n_imp most correlated orbitals  →  impurity
Schmidt decomposition of the UHF density matrix  →  bath orbitals
  (the part of the environment entangled with the impurity)
Core electrons (orthogonal to both)              →  traced out
  (contribute a mean-field potential to h1e_eff)

Result: h1e_emb, h2e_emb  in a space of n_imp + n_bath ≤ 2*n_imp orbitals.
This is the Hamiltonian we feed directly to SQD.

NO self-consistent loop. NO fragment FCI/CCSD. ONE UHF calculation.

Requires: results/step1_asf.pkl
Saves:    results/step2_hamiltonian.pkl
Runtime:  ~30-60 seconds
"""
import os, sys, time, pickle, math
import numpy as np
import pyscf.scf.hf as _pyscf_hf_base

import config

FORCE_RERUN = True

os.makedirs(config.RESULTS_DIR, exist_ok=True)

if os.path.exists(config.STEP2_FILE) and not FORCE_RERUN:
    print(f"[Step 2] Already done → {config.STEP2_FILE}")
    print("  Set FORCE_RERUN = True to redo.")
    sys.exit(0)

if not os.path.exists(config.STEP1_FILE):
    raise FileNotFoundError(f"Run step1_asf.py first. Not found: {config.STEP1_FILE}")

with open(config.STEP1_FILE, "rb") as f:
    step1 = pickle.load(f)

nel              = step1["nel"]
mo_list          = step1["mo_list"]
mo_coeff         = step1["mo_coeff"]      # (n_AO, n_MO) MP2 NOs, S-orthonormal
n_active_orbs    = step1["n_active_orbs"]
most_active_atom = step1["most_active_atom"]

print(f"\n[Step 2] One-shot DMET for {config.MOLECULE}")
print(f"  Impurity : {n_active_orbs} ASF active orbitals  {mo_list}")
print(f"  Bath     : Schmidt decomposition of UHF density matrix")
print(f"  Max emb  : {config.MAX_EMBED_ORBS} orbitals ({2*config.MAX_EMBED_ORBS} qubits)")

# ── Environment ───────────────────────────────────────────────────────────────
os.environ["BLOCKEXE"]            = config.BLOCKEXE_WRAPPER
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_DEBUG_CPU_TYPE"]  = "5"

from pyscf.dmrgscf import dmrgci
dmrgci.settings.BLOCKEXE = config.BLOCKEXE_WRAPPER

from pyscf import gto, scf, ao2mo, fci as pyscf_fci
from pyscf.scf import hf as _scf_hf

mol = gto.M(
    atom    = config.GEOMETRY,
    basis   = config.BASIS,
    charge  = 0,
    spin    = 0,
    verbose = True,
)

# ── UHF with Newton fallback ──────────────────────────────────────────────────
_orig_kernel = _pyscf_hf_base.SCF.kernel

def _newton_fallback(self, dm0=None, **kwargs):
    self.max_cycle   = max(getattr(self, "max_cycle",   50), 400)
    self.level_shift = max(getattr(self, "level_shift", 0.0), 0.5)
    result = _orig_kernel(self, dm0=dm0, **kwargs)
    if not self.converged:
        print("  [UHF] DIIS failed → Newton solver...")
        try:
            nw = self.newton()
            nw.max_cycle = 400
            nw.kernel(self.mo_coeff)
            if nw.converged:
                print(f"  [UHF] ✓ Newton: E = {nw.e_tot:.8f} Ha")
                for a in ("e_tot","mo_coeff","mo_energy","mo_occ","converged"):
                    setattr(self, a, getattr(nw, a))
        except Exception as e:
            print(f"  [UHF] Newton failed: {e}")
    return result

_pyscf_hf_base.SCF.kernel = _newton_fallback

print("\n[Step 2] Running UHF...")
t0 = time.time()
mf = scf.UHF(mol)
mf.kernel()
_pyscf_hf_base.SCF.kernel = _orig_kernel
print(f"  UHF: E={mf.e_tot:.8f} Ha | converged={mf.converged} | {time.time()-t0:.1f}s")

if not mf.converged:
    raise RuntimeError("UHF did not converge.")

# ═══════════════════════════════════════════════════════════════════════════════
# ONE-SHOT DMET EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Step 2] Schmidt decomposition...")

n_AO  = mol.nao_nr()
n_imp = n_active_orbs

# ── Impurity: ASF active orbitals in AO basis ──────────────────────────────
# mo_coeff from ASF is an orthonormal matrix (NOs diagonalize the 1-RDM)
# → C_imp columns are S-orthonormal by construction
C_imp = mo_coeff[:, mo_list]   # (n_AO, n_imp)

# ── Löwdin transformation (orthogonal AO basis) ───────────────────────────
S = mol.intor('int1e_ovlp')
evals, evecs = np.linalg.eigh(S)
mask      = evals > 1e-15
S_sqrt    = (evecs[:, mask] * np.sqrt(evals[mask]))   @ evecs[:, mask].T
S_invsqrt = (evecs[:, mask] / np.sqrt(evals[mask]))   @ evecs[:, mask].T

# Safety: verify S-orthonormality of C_imp
orth_err = np.max(np.abs(C_imp.T @ S @ C_imp - np.eye(n_imp)))
if orth_err > 1e-6:
    print(f"  Warning: C_imp not S-orthonormal (err={orth_err:.1e}), re-orthogonalizing...")
    Q_tmp, _ = np.linalg.qr(S_sqrt @ C_imp)
    C_imp = S_invsqrt @ Q_tmp[:, :n_imp]

# ── Total density matrix in Löwdin basis ─────────────────────────────────
# UHF: sum alpha + beta density matrices
dm_raw = mf.make_rdm1()
dm_tot = dm_raw[0] + dm_raw[1]          # (n_AO, n_AO) total DM
dm_lo  = S_sqrt @ dm_tot @ S_sqrt       # symmetric, trace = N_electrons
print(f"  Total electrons (trace check): {np.trace(dm_lo):.2f}")

# ── Impurity in Löwdin basis ──────────────────────────────────────────────
# S-orthonormal C_imp → orthonormal Q_imp in Löwdin basis
Q_imp = S_sqrt @ C_imp                  # (n_AO, n_imp), columns are orthonormal
P_env = np.eye(n_AO) - Q_imp @ Q_imp.T  # environment projector

# ── Off-diagonal DM block: how much environment couples to impurity ────────
#
#   F[i,j] = <env_i | ρ | imp_j>
#
# The singular values of F measure the entanglement between each environment
# orbital and the impurity block. Large sv → strongly entangled → bath orbital.
F = P_env @ dm_lo @ Q_imp               # (n_AO, n_imp)
U_env, sv, _ = np.linalg.svd(F, full_matrices=True)

# Keep bath orbitals above threshold, capped at n_imp (DMET theorem) and MAX
n_bath = int(np.sum(sv > config.BATH_TOLERANCE))
n_bath = min(n_bath, n_imp)
n_bath = min(n_bath, config.MAX_EMBED_ORBS - n_imp)  # enforce total cap

Q_bath = U_env[:, :n_bath]              # (n_AO, n_bath) in Löwdin basis

print(f"  n_imp  = {n_imp}")
print(f"  n_bath = {n_bath}  (sv > {config.BATH_TOLERANCE})")
print(f"  Schmidt values: {sv[:min(n_bath+2, len(sv))].round(6)}")

# ── Embedding basis ────────────────────────────────────────────────────────
n_emb = n_imp + n_bath
Q_emb = np.hstack([Q_imp, Q_bath])      # (n_AO, n_emb) Löwdin basis
C_emb = S_invsqrt @ Q_emb              # (n_AO, n_emb) AO basis

print(f"\n  Embedding: {n_imp} impurity + {n_bath} bath = {n_emb} orbitals total")
print(f"  → {2*n_emb} qubits for SQD")

# ── Core density matrix ───────────────────────────────────────────────────
# Electrons in the complement of the embedding space
P_emb      = Q_emb @ Q_emb.T
P_core     = np.eye(n_AO) - P_emb
dm_core_lo = P_core @ dm_lo @ P_core
dm_core    = S_invsqrt @ dm_core_lo @ S_invsqrt
dm_core    = 0.5 * (dm_core + dm_core.T)   # symmetrize numerically

# ── AO 1e integrals ───────────────────────────────────────────────────────
h1e_AO = mol.intor('int1e_kin') + mol.intor('int1e_nuc')

# ── Core embedding potential ──────────────────────────────────────────────
# Core electrons are traced out but their Coulomb + exchange field acts
# on the embedding space → add to h1e
#
# h1e_eff[p,q] = h1e[p,q]
#              + sum_{i∈core} (2*J[p,q,i,i] - K[p,i,i,q])   ← mean-field from core
#              = h1e[p,q] + vj[p,q] - 0.5*vk[p,q]
#
# where vj, vk are Coulomb and exchange matrices built from dm_core
vj, vk     = _scf_hf.get_jk(mol, dm_core, hermi=1)
h1e_eff_AO = h1e_AO + vj - 0.5 * vk

# ── Transform to embedding basis ──────────────────────────────────────────
print("\n[Step 2] Transforming integrals to embedding basis...")
t1 = time.time()

h1e_emb = C_emb.T @ h1e_eff_AO @ C_emb                           # (n_emb, n_emb)
h2e_emb = ao2mo.kernel(mol, C_emb, compact=False).reshape(
              n_emb, n_emb, n_emb, n_emb)                         # (n_emb,)*4
print(f"  Done in {time.time()-t1:.1f}s")

# ── Core energy constant ──────────────────────────────────────────────────
# E_core = E_nuc + Tr[dm_core @ h1e] + 0.5*Tr[dm_core @ (J_core - 0.5*K_core)]
#        = E_nuc + 0.5 * Tr[dm_core @ (h1e + h1e_eff)]
ecore  = mol.energy_nuc()
ecore += 0.5 * np.einsum('ij,ji->', dm_core, h1e_AO + h1e_eff_AO)
print(f"  ecore = {ecore:.6f} Ha  (constant; not used in SQD vs FCI comparison)")

# ── Electron count in embedding space ─────────────────────────────────────
n_elec_emb = int(round(np.trace(Q_emb.T @ dm_lo @ Q_emb)))
if n_elec_emb % 2 != 0:
    print(f"  Warning: n_elec_emb={n_elec_emb} is odd → adjusting to {n_elec_emb-1} (singlet target)")
    n_elec_emb -= 1
n_alpha = n_elec_emb // 2
n_beta  = n_elec_emb // 2

print(f"\n  Summary:")
print(f"    h1e_emb : {h1e_emb.shape}")
print(f"    h2e_emb : {h2e_emb.shape}")
print(f"    electrons: {n_elec_emb} ({n_alpha}α + {n_beta}β)")
print(f"    n_qubits : {2*n_emb}")
print(f"    max_configs: C({n_emb},{n_alpha})² = {math.comb(n_emb, n_alpha)**2:,}")

# ── FCI sanity check ──────────────────────────────────────────────────────
n_dets = math.comb(n_emb, n_alpha) ** 2
fci_e  = None

if n_dets <= 5_000_000:
    print(f"\n[Step 2] FCI sanity check ({n_dets:,} determinants)...")
    cisolver = pyscf_fci.direct_spin1.FCI()
    fci_e, _ = cisolver.kernel(h1e_emb, h2e_emb, n_emb, (n_alpha, n_beta))
    print(f"  FCI embedded energy : {fci_e:.8f} Ha  ← SQD target")
    print(f"  FCI total energy    : {fci_e + ecore:.8f} Ha")
    print(f"  UHF total energy    : {mf.e_tot:.8f} Ha")
    print(f"  Correlation energy  : {fci_e + ecore - mf.e_tot:.8f} Ha")
else:
    print(f"\n  Skipping FCI ({n_dets:,} dets > 5M) — SQD will solve this")

# ── Save ──────────────────────────────────────────────────────────────────
results = {
    # Embedded Hamiltonian (the main output — goes into SQD)
    "h1e"        : h1e_emb,
    "h2e"        : h2e_emb,
    "ecore"      : ecore,
    # Dimensions
    "n_emb"      : n_emb,
    "n_imp"      : n_imp,
    "n_bath"     : n_bath,
    "n_alpha"    : n_alpha,
    "n_beta"     : n_beta,
    # Reference energies
    "fci_ref_e"  : fci_e,
    "uhf_energy" : mf.e_tot,
    # Schmidt values (for diagnostics)
    "sv"         : sv[:n_bath],
}

with open(config.STEP2_FILE, "wb") as f:
    pickle.dump(results, f)

print(f"\n[Step 2] ✓ Saved → {config.STEP2_FILE}")