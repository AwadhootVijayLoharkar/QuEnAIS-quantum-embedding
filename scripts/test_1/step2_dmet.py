"""
Step 2 — DMET build + simulate, then truncate Fe embedding to SQD_MAX_ORBS
         frontier orbitals for the quantum step.

Requires: results/step1_asf.pkl
Saves:    results/step2_dmet.pkl

Runtime: ~5-15 min
"""
import os
import sys
import time
import pickle
import numpy as np
import pyscf.scf.hf as _pyscf_hf_base
from pyscf import ao2mo as _pyscf_ao2mo

import config

# ── Skip if already done ──────────────────────────────────────────────────────
FORCE_RERUN = False   # set True to redo this step

os.makedirs(config.RESULTS_DIR, exist_ok=True)

if os.path.exists(config.STEP2_FILE) and not FORCE_RERUN:
    print(f"[Step 2] Results already exist at {config.STEP2_FILE}")
    print("  Set FORCE_RERUN = True to rerun.")
    sys.exit(0)

# ── Check Step 1 ──────────────────────────────────────────────────────────────
if not os.path.exists(config.STEP1_FILE):
    raise FileNotFoundError(
        f"[Step 2] Step 1 results not found: {config.STEP1_FILE}\n"
        "Run step1_asf.py first."
    )

with open(config.STEP1_FILE, "rb") as f:
    step1 = pickle.load(f)

most_active_atom = step1["most_active_atom"]
print(f"\n[Step 2] Loaded Step 1 from {config.STEP1_FILE}")
print(f"  most active atom    : {most_active_atom} ({config.ATOM_SYMS[most_active_atom]})")
print(f"  SQD target fragment : {config.MOST_ACTIVE_FRAG} (Fe)")
print(f"  Fragment definition : {config.FRAGMENT_ATOMS}  ([Fe] | [6×N])")
print(f"  Fragment solvers    : {config.FRAGMENT_SOLVERS}  (CCSD; FCI impossible on ~30-orb Fe embedding)")

# ── Environment ───────────────────────────────────────────────────────────────
os.environ["BLOCKEXE"]            = config.BLOCKEXE_WRAPPER
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_DEBUG_CPU_TYPE"]  = "5"

from pyscf.dmrgscf import dmrgci
dmrgci.settings.BLOCKEXE = config.BLOCKEXE_WRAPPER

from tangelo import SecondQuantizedMolecule
from tangelo.problem_decomposition import DMETProblemDecomposition

# ── Orbital truncation helper ─────────────────────────────────────────────────
def truncate_to_active_space(h1e, h2e, n_alpha, n_beta, n_orb, max_orbs):
    """
    Reduce a large embedding Hamiltonian to a frontier active space.

    Strategy
    --------
    1. Diagonalise h1e (acts as a proxy Fock matrix) → orbital energies.
    2. Select `max_orbs` orbitals centred on the HOMO/LUMO gap.
    3. Transform h1e and h2e into that subspace via 4-step einsum.
    4. Count active electrons by aufbau in the new ordering.

    Why h1e as Fock proxy?
      DMET's embedded h1e already contains the chemical-potential shift and
      bath contributions, so its eigenvalues faithfully rank the frontier
      orbitals by correlation importance.

    Returns
    -------
    h1e_act, h2e_act : ndarray  — integrals in active space
    n_alpha_act, n_beta_act : int — active electron counts
    C_act : ndarray (n_orb, max_orbs) — MO coefficients (embedding → active)
    sel   : list[int]  — selected column indices in Fock eigenspace
    """
    # Step 1 — orbital energies from h1e
    e_orb, C_orb = np.linalg.eigh(h1e)      # ascending order

    # Step 2 — frontier window centred on HOMO
    n_act   = min(max_orbs, n_orb)
    homo_i  = n_alpha - 1                   # last alpha-occupied (aufbau)
    n_below = n_act // 2
    n_above = n_act - n_below
    start   = max(0, homo_i - n_below + 1)
    end     = min(n_orb, start + n_act)
    start   = max(0, end - n_act)           # clamp if near top of spectrum
    sel     = list(range(start, end))
    C_act   = C_orb[:, sel]

    # Step 3a — transform h1e
    h1e_act = C_act.T @ h1e @ C_act

    # Step 3b — restore h2e to full (n,n,n,n) if stored compressed
    h2e_np = np.asarray(h2e)
    if h2e_np.ndim in (1, 2):
        h2e_np = _pyscf_ao2mo.restore(1, h2e_np, n_orb)
    # 4-step contraction  O(n_orb^4 * n_act)  — ~24 M ops for n_orb=30, n_act=8
    tmp      = np.einsum('pi,pqrs->iqrs', C_act, h2e_np)
    tmp      = np.einsum('qj,iqrs->ijrs', C_act, tmp)
    tmp      = np.einsum('rk,ijrs->ijks', C_act, tmp)
    h2e_act  = np.einsum('sl,ijks->ijkl', C_act, tmp)

    # Step 4 — active electron counts (aufbau in Fock ordering)
    n_alpha_act = max(1, sum(1 for i in sel if i < n_alpha))
    n_beta_act  = max(1, sum(1 for i in sel if i < n_beta))

    return h1e_act, h2e_act, n_alpha_act, n_beta_act, C_act, sel


# ── SCF Patch 1: Newton fallback for SecondQuantizedMolecule init ─────────────
_orig_scf_kernel = _pyscf_hf_base.SCF.kernel

def _kernel_with_newton_fallback(self, dm0=None, **kwargs):
    """Full Newton fallback — used only for the initial full-system SCF."""
    self.max_cycle   = max(getattr(self, "max_cycle",   50),  400)
    self.level_shift = max(getattr(self, "level_shift", 0.0), 0.5)
    result = _orig_scf_kernel(self, dm0=dm0, **kwargs)
    if not self.converged:
        print("  [SCF patch] DIIS failed → switching to Newton solver...")
        try:
            newton           = self.newton()
            newton.max_cycle = 400
            newton.kernel(self.mo_coeff)
            if newton.converged:
                print(f"  [SCF patch] ✓ Newton converged: E={newton.e_tot:.10f} Ha")
                self.e_tot     = newton.e_tot
                self.mo_coeff  = newton.mo_coeff
                self.mo_energy = newton.mo_energy
                self.mo_occ    = newton.mo_occ
                self.converged = True
            else:
                print("  [SCF patch] Newton also did not converge.")
        except AttributeError:
            print("  [SCF patch] .newton() not available for this solver.")
        except Exception as exc:
            print(f"  [SCF patch] Newton error: {exc}")
    return result

_pyscf_hf_base.SCF.kernel = _kernel_with_newton_fallback

# ── Build SecondQuantizedMolecule ─────────────────────────────────────────────
mol_tangelo = None
for try_spin in [4, 2, 6, 0]:
    try:
        mol_tangelo = SecondQuantizedMolecule(
            config.GEOMETRY, q=0, spin=try_spin, basis=config.BASIS
        )
        print(f"✓ SecondQuantizedMolecule converged (spin={try_spin}, basis={config.BASIS})")
        break
    except (ValueError, RuntimeError) as e:
        print(f"  {config.BASIS} spin={try_spin} failed: {e}")

if mol_tangelo is None:
    print("\n  STO-3G exhausted → trying lanl2dz (ECP basis)...")
    for try_spin in [4, 2, 6, 0]:
        try:
            mol_tangelo = SecondQuantizedMolecule(
                config.GEOMETRY, q=0, spin=try_spin, basis="lanl2dz"
            )
            print(f"✓ SecondQuantizedMolecule converged (spin={try_spin}, basis=lanl2dz)")
            break
        except (ValueError, RuntimeError) as e:
            print(f"  lanl2dz spin={try_spin} failed: {e}")

if mol_tangelo is None:
    _pyscf_hf_base.SCF.kernel = _orig_scf_kernel
    raise RuntimeError("SCF failed for all spin states and bases.")

# ── SCF Patch 2: level-shift only for DMET fragment sub-problems ──────────────
# Newton is too expensive for the many small fragment SCFs inside DMET build.
def _kernel_level_shift_only(self, dm0=None, **kwargs):
    """Lightweight patch — level-shift only, no Newton."""
    self.level_shift = max(getattr(self, "level_shift", 0.0), 0.4)
    self.max_cycle   = max(getattr(self, "max_cycle",   50),  300)
    return _orig_scf_kernel(self, dm0=dm0, **kwargs)

_pyscf_hf_base.SCF.kernel = _kernel_level_shift_only

# ── DMET ──────────────────────────────────────────────────────────────────────
dmet = DMETProblemDecomposition({
    "molecule"        : mol_tangelo,
    "fragment_atoms"  : config.FRAGMENT_ATOMS,
    "fragment_solvers": config.FRAGMENT_SOLVERS,
    "verbose"         : True,
})

t0 = time.time()
print("\n[Step 2] Building DMET (Schmidt decomposition + integral transforms)...")
dmet.build()
print(f"  dmet.build()    done in {time.time()-t0:.1f}s")

t1 = time.time()
print("[Step 2] Simulating DMET (CCSD on both fragments)...")
dmet_energy = dmet.simulate()
print(f"  dmet.simulate() done in {time.time()-t1:.1f}s")
print(f"  DMET total energy = {dmet_energy:.8f} Ha")

# Restore original SCF kernel
_pyscf_hf_base.SCF.kernel = _orig_scf_kernel

# ── Extract full Fe fragment integrals ────────────────────────────────────────
# scf_fragments[i] = [mf, h1e, mol, [n_alpha, n_beta], fock, h2e, fock_copy]
frag_data = dmet.scf_fragments[config.MOST_ACTIVE_FRAG]
h1e_full  = frag_data[1]
h2e_full  = frag_data[5]
n_alpha   = int(frag_data[3][0])
n_beta    = int(frag_data[3][1])
n_orb     = h1e_full.shape[0]

print(f"\n[Step 2] Full Fe embedding: {n_orb} orbs | {n_alpha}α + {n_beta}β")
print(f"  FCI on {n_orb} orbs would need C({n_orb},{n_alpha})² determinants — impossible")
print(f"  Truncating to {config.SQD_MAX_ORBS} frontier orbitals for SQD...")

# ── Truncate to SQD active space ──────────────────────────────────────────────
h1e_act, h2e_act, n_alpha_act, n_beta_act, C_act, sel = truncate_to_active_space(
    h1e_full, h2e_full, n_alpha, n_beta, n_orb, config.SQD_MAX_ORBS
)
n_act = len(sel)

print(f"\n[Step 2] Active space for SQD:")
print(f"  Selected orbital indices (Fock order) : {sel}")
print(f"  n_orb_active   : {n_act}")
print(f"  n_alpha_active : {n_alpha_act}")
print(f"  n_beta_active  : {n_beta_act}")
print(f"  n_qubits       : {2 * n_act}")

# ── Save ──────────────────────────────────────────────────────────────────────
results = {
    # DMET
    "dmet_energy"  : dmet_energy,
    # Full Fe embedding (30 orbs) — kept for reference
    "h1e_full"     : h1e_full,
    "h2e_full"     : h2e_full,
    "n_alpha_full" : n_alpha,
    "n_beta_full"  : n_beta,
    "n_orb_full"   : n_orb,
    # Truncated active space (SQD_MAX_ORBS orbs)
    "h1e_act"      : h1e_act,
    "h2e_act"      : h2e_act,
    "n_alpha_act"  : n_alpha_act,
    "n_beta_act"   : n_beta_act,
    "n_orb_act"    : n_act,
    "C_act"        : C_act,
    "sel"          : sel,
}

with open(config.STEP2_FILE, "wb") as f:
    pickle.dump(results, f)

print(f"\n[Step 2] ✓ Saved → {config.STEP2_FILE}")