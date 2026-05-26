"""
Step 1 — ASF + Mulliken orbital-to-atom mapping
Saves: results/step1_asf.pkl
Runtime: ~2-5 min (UHF + Newton + CASCI)
"""
import os
import sys
import pickle
import numpy as np

import config

# ── Skip if already done ──────────────────────────────────────────────────────
FORCE_RERUN = False   # ← set True to redo this step

os.makedirs(config.RESULTS_DIR, exist_ok=True)

if os.path.exists(config.STEP1_FILE) and not FORCE_RERUN:
    print(f"[Step 1] Results already exist at {config.STEP1_FILE}")
    print("  Set FORCE_RERUN = True to rerun.")
    sys.exit(0)

# ── Environment ───────────────────────────────────────────────────────────────
os.environ["BLOCKEXE"]            = config.BLOCKEXE_WRAPPER
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_DEBUG_CPU_TYPE"]  = "5"

from pyscf.dmrgscf import dmrgci
dmrgci.settings.BLOCKEXE = config.BLOCKEXE_WRAPPER

from pyscf import gto
from asf.wrapper import find_from_mol

# ── PySCF Mole ────────────────────────────────────────────────────────────────
mol_pyscf = gto.M(
    atom    = config.GEOMETRY,
    basis   = config.BASIS,
    charge  = 0,
    spin    = 0,
    verbose = 3,
)

# ── Run ASF ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print(f"[Step 1] Running ASF for {config.MOLECULE}...")
print("="*55)

active_space = find_from_mol(
    mol_pyscf,
    entropy_threshold = config.ENTROPY_THRESHOLD,
    max_norb          = config.MAX_NORB,
    min_norb          = config.MIN_NORB,
    verbose           = True,
)

nel           = active_space.nel
mo_list       = active_space.mo_list
mo_coeff      = active_space.mo_coeff
n_active_orbs = len(mo_list)

if n_active_orbs == 0:
    raise RuntimeError(
        "ASF found no active orbitals.\n"
        "Lower entropy_threshold in config.py (e.g. 0.05)."
    )

print(f"\n{'='*55}")
print(f"ASF Results for {config.MOLECULE}:")
print(f"  n_active_electrons : {nel}")
print(f"  n_active_orbitals  : {n_active_orbs}")
print(f"  active mo_list     : {mo_list}")
print(f"  mo_coeff shape     : {mo_coeff.shape}")
print(f"{'='*55}")

# ── Mulliken: map active orbitals → atoms ─────────────────────────────────────
S         = mol_pyscf.intor("int1e_ovlp")
ao_labels = mol_pyscf.ao_labels(fmt=None)

active_coeffs       = mo_coeff[:, mo_list]
orbital_atom_weight = np.zeros((n_active_orbs, config.N_ATOMS))

for orb_i in range(n_active_orbs):
    c  = active_coeffs[:, orb_i]
    CS = c * (S @ c)
    for ao_j, (atom_idx, *_) in enumerate(ao_labels):
        orbital_atom_weight[orb_i, atom_idx] += CS[ao_j]

dominant_atoms   = np.argmax(orbital_atom_weight, axis=1)
active_per_atom  = np.bincount(dominant_atoms, minlength=config.N_ATOMS)
most_active_atom = int(np.argmax(active_per_atom))

print("\nOrbital → Atom Mulliken mapping:")
print(f"  {'MO':>4}  {'→ atom':>7}  {'Weights per atom'}")
print(f"  {'─'*52}")
for i, (orb_idx, da) in enumerate(zip(mo_list, dominant_atoms)):
    weights = "  ".join(
        f"{config.ATOM_SYMS[j]}:{orbital_atom_weight[i,j]:+.3f}"
        for j in range(config.N_ATOMS)
    )
    print(f"  {orb_idx:4d}  → {da} ({config.ATOM_SYMS[da]:2s})  |  {weights}")

print(f"\nActive orbital count per atom:")
for i, (sym, cnt) in enumerate(zip(config.ATOM_SYMS, active_per_atom)):
    bar = "█" * int(cnt * 4)
    print(f"  Atom {i} ({sym:2s}): {cnt:2d}  {bar}")

print(f"\n→ ASF-guided most active atom : {most_active_atom} "
      f"({config.ATOM_SYMS[most_active_atom]})")
print(f"→ SQD target DMET fragment    : {config.MOST_ACTIVE_FRAG} (Fe)  ← fixed in config")

# ── Save ──────────────────────────────────────────────────────────────────────
results = {
    "nel"                 : nel,
    "mo_list"             : mo_list,
    "mo_coeff"            : mo_coeff,
    "n_active_orbs"       : n_active_orbs,
    "orbital_atom_weight" : orbital_atom_weight,
    "dominant_atoms"      : dominant_atoms,
    "active_per_atom"     : active_per_atom,
    "most_active_atom"    : most_active_atom,
}

with open(config.STEP1_FILE, "wb") as f:
    pickle.dump(results, f)

print(f"\n[Step 1] ✓ Saved → {config.STEP1_FILE}")