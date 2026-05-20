import os
import time
import math
import numpy as np
import pyscf.scf.hf as _pyscf_hf_base

# ── MKL / block2 fix ──────────────────────────────────────────────────────────
_WRAPPER = os.path.expanduser("~/block2main_wrapper.sh")
os.environ["BLOCKEXE"]            = _WRAPPER
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_DEBUG_CPU_TYPE"]  = "5"

from pyscf.dmrgscf import dmrgci
dmrgci.settings.BLOCKEXE = _WRAPPER

from pyscf import gto, scf, fci as pyscf_fci, lo

from asf.wrapper import (
    find_from_mol,
    find_from_scf,
    sized_space_from_mol,
    sized_space_from_scf,
    reorder_mos,
)

# ── Molecule ──────────────────────────────────────────────────────────────────
MOLECULE = "FeN6"

geometries = {
    "LiH" : [("Li", (0., 0., 0.00)), ("H",  (0., 0., 1.60))],
    "H2O" : [("O",  (0., 0., 0.00)), ("H",  (0.,  0.757,  0.586)),
                                      ("H",  (0., -0.757,  0.586))],
    "N2"  : [("N",  (0., 0., 0.00)), ("N",  (0., 0., 1.098))],
    "H6"  : [("H",  (0., 0., i*0.74)) for i in range(6)],
}

geometries["FeN6"] = [
    ("Fe", (0.000,  0.000,  0.000)),
    ("N",  (0.000,  0.000,  2.000)),
    ("N",  (0.000,  0.000, -2.000)),
    ("N",  (0.000,  2.000,  0.000)),
    ("N",  (0.000, -2.000,  0.000)),
    ("N",  (2.000,  0.000,  0.000)),
    ("N",  (-2.000, 0.000,  0.000)),
]

geometry  = geometries[MOLECULE]
atom_syms = [a[0] for a in geometry]
n_atoms   = len(geometry)

# ── PySCF Mole ────────────────────────────────────────────────────────────────
mol_pyscf = gto.M(
    atom    = geometry,
    basis   = "sto-3g",
    charge  = 0,
    spin    = 0,
    verbose = 3,
)

# ── ASF ───────────────────────────────────────────────────────────────────────
# Option A: find_from_mol (ASF handles UHF + stability internally)
active_space = find_from_mol(
    mol_pyscf,
    entropy_threshold = 0.15,
    max_norb          = 8,
    min_norb          = 2,
    verbose           = True,
)

# Option B: find_from_scf (bring your own SCF object)
# mf = scf.RHF(mol_pyscf).run()
# active_space = find_from_scf(mf, entropy_threshold=0.15, verbose=True)

# Option C: sized_space_from_mol (exact active space size)
# active_space = sized_space_from_mol(mol_pyscf, size=(2,2), verbose=True)

# ── Unpack ActiveSpace ────────────────────────────────────────────────────────
nel           = active_space.nel
mo_list       = active_space.mo_list
mo_coeff      = active_space.mo_coeff
n_active_orbs = len(mo_list)

print(f"\n{'='*55}")
print(f"ASF Results for {MOLECULE}:")
print(f"  n_active_electrons : {nel}")
print(f"  n_active_orbitals  : {n_active_orbs}")
print(f"  active mo_list     : {mo_list}  (0-indexed, MP2 NO basis)")
print(f"  mo_coeff shape     : {mo_coeff.shape}")
print(f"{'='*55}")

if n_active_orbs == 0:
    raise RuntimeError(
        "ASF found no active orbitals.\n"
        "Try: lower entropy_threshold (e.g. 0.05) "
        "or use N2/H6 for stronger correlation"
    )

# ── Mulliken population: map active orbitals → atoms ─────────────────────────
S         = mol_pyscf.intor("int1e_ovlp")
ao_labels = mol_pyscf.ao_labels(fmt=None)

active_coeffs       = mo_coeff[:, mo_list]
orbital_atom_weight = np.zeros((n_active_orbs, n_atoms))

for orb_i in range(n_active_orbs):
    c  = active_coeffs[:, orb_i]
    CS = c * (S @ c)
    for ao_j, (atom_idx, *_) in enumerate(ao_labels):
        orbital_atom_weight[orb_i, atom_idx] += CS[ao_j]

dominant_atoms  = np.argmax(orbital_atom_weight, axis=1)
active_per_atom = np.bincount(dominant_atoms, minlength=n_atoms)

print("\nOrbital → Atom Mulliken mapping:")
print(f"  {'MO':>4}  {'→ atom':>7}  {'Weights per atom'}")
print(f"  {'─'*52}")
for i, (orb_idx, da) in enumerate(zip(mo_list, dominant_atoms)):
    weights = "  ".join(
        f"{atom_syms[j]}:{orbital_atom_weight[i,j]:+.3f}" for j in range(n_atoms)
    )
    print(f"  {orb_idx:4d}  → {da} ({atom_syms[da]:2s})  |  {weights}")

print(f"\nActive orbital count per atom:")
for i, (sym, cnt) in enumerate(zip(atom_syms, active_per_atom)):
    bar = "█" * int(cnt * 4)
    print(f"  Atom {i} ({sym:2s}): {cnt:2d}  {bar}")

# most_active_atom = atom index (0-6) from ASF Mulliken analysis (informational)
# most_active_frag = DMET fragment index (always 0 = Fe in our [1,6] scheme)
most_active_atom = int(np.argmax(active_per_atom))
most_active_frag = 0    # Fe is always DMET fragment 0 in the [1,6] scheme
print(f"\n→ ASF-guided most active atom : {most_active_atom} ({atom_syms[most_active_atom]})")
print(f"→ SQD target DMET fragment    : {most_active_frag} (Fe)")

# ── Tangelo / DMET / Qiskit imports ──────────────────────────────────────────
from tangelo import SecondQuantizedMolecule
from tangelo.problem_decomposition import DMETProblemDecomposition
from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorSampler
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.fermion import solve_fermion
from qiskit_addon_sqd.configuration_recovery import recover_configurations

# ── Fragment definition ───────────────────────────────────────────────────────
# [1, 6]  →  Fragment 0 = Fe (1 atom)
#            Fragment 1 = all 6 N atoms grouped together
#
# Why 2 fragments instead of 7?
#   - 7 fragments: O(N⁴) integral transform × 7  +  FCI × 7  → hours
#   - 2 fragments: O(N⁴) integral transform × 2  +  FCI × 1  → minutes
#
# Fragment solvers:
#   - FCI  on Fe fragment  (small embedding, exact)
#   - CCSD on N6 fragment  (larger, but polynomial scaling)

fragment_atoms   = [1, 6]
fragment_solvers = ["fci", "ccsd"]

print(f"\nFragment definition : {fragment_atoms}  ([Fe] | [6×N])")
print(f"Fragment solvers    : {fragment_solvers}")
print(f"SQD target fragment : {most_active_frag} (Fe)  ← ASF-guided")

# ── SCF Patch 1: Newton fallback (for SecondQuantizedMolecule init only) ──────
# Needed because FeN6 with open-shell Fe doesn't converge with plain DIIS
_orig_scf_kernel = _pyscf_hf_base.SCF.kernel

def _kernel_with_newton_fallback(self, dm0=None, **kwargs):
    """DIIS + level-shift first; auto-retry with Newton if not converged."""
    self.max_cycle   = max(getattr(self, "max_cycle",   50),  400)
    self.level_shift = max(getattr(self, "level_shift", 0.0), 0.5)
    result = _orig_scf_kernel(self, dm0=dm0, **kwargs)
    if not self.converged:
        print("  [SCF patch] DIIS failed → switching to Newton solver...")
        try:
            newton           = self.newton()
            newton.max_cycle = 400
            newton.kernel(self.mo_coeff)        # warm-start from DIIS MOs
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

# Apply full Newton patch for SecondQuantizedMolecule initialisation
_pyscf_hf_base.SCF.kernel = _kernel_with_newton_fallback

mol_tangelo = None
for try_spin in [4, 2, 6, 0]:
    try:
        mol_tangelo = SecondQuantizedMolecule(
            geometry, q=0, spin=try_spin, basis="sto-3g"
        )
        print(f"✓ SecondQuantizedMolecule converged (spin={try_spin}, basis=sto-3g)")
        break
    except (ValueError, RuntimeError) as e:
        print(f"  sto-3g  spin={try_spin} failed: {e}")

# Fallback: lanl2dz includes a proper ECP for Fe → more stable SCF
if mol_tangelo is None:
    print("\n  STO-3G exhausted → trying lanl2dz (ECP basis, better for Fe)...")
    for try_spin in [4, 2, 6, 0]:
        try:
            mol_tangelo = SecondQuantizedMolecule(
                geometry, q=0, spin=try_spin, basis="lanl2dz"
            )
            print(f"✓ SecondQuantizedMolecule converged (spin={try_spin}, basis=lanl2dz)")
            break
        except (ValueError, RuntimeError) as e:
            print(f"  lanl2dz spin={try_spin} failed: {e}")

if mol_tangelo is None:
    _pyscf_hf_base.SCF.kernel = _orig_scf_kernel
    raise RuntimeError(
        "SCF failed for all spin states and bases (sto-3g + lanl2dz).\n"
        "Verify the geometry is physically reasonable for Fe(II/III)."
    )

# ── SCF Patch 2: Level-shift only (for DMET fragment SCFs) ───────────────────
# Newton is too expensive for the many small internal fragment SCF calls in DMET.
# Level-shift alone is sufficient to stabilise those simpler sub-problems.
def _kernel_level_shift_only(self, dm0=None, **kwargs):
    """Lighter SCF patch — level-shift only, no Newton."""
    self.level_shift = max(getattr(self, "level_shift", 0.0), 0.4)
    self.max_cycle   = max(getattr(self, "max_cycle",   50),  300)
    return _orig_scf_kernel(self, dm0=dm0, **kwargs)

_pyscf_hf_base.SCF.kernel = _kernel_level_shift_only

# ── DMET build + simulate ─────────────────────────────────────────────────────
dmet = DMETProblemDecomposition({
    "molecule"        : mol_tangelo,
    "fragment_atoms"  : fragment_atoms,
    "fragment_solvers": fragment_solvers,
    "verbose"         : True,
})

t0 = time.time()
print("\nBuilding DMET (Schmidt decomposition + integral transforms)...")
dmet.build()
print(f"dmet.build()    done in {time.time()-t0:.1f}s")

t1 = time.time()
print("Simulating DMET (FCI on Fe + CCSD on N6)...")
dmet_energy = dmet.simulate()
print(f"dmet.simulate() done in {time.time()-t1:.1f}s")
print(f"\nDMET total energy = {dmet_energy:.8f} Ha")

# Restore original SCF kernel — all DMET work is complete
_pyscf_hf_base.SCF.kernel = _orig_scf_kernel

# ── Extract Fe fragment integrals ─────────────────────────────────────────────
# scf_fragments[i] = [mf, h1e, mol, [n_a, n_b], fock, h2e, fock_copy]
#                     [0]  [1]  [2]  [3]         [4]   [5]  [6]
frag_data = dmet.scf_fragments[most_active_frag]
h1e       = frag_data[1]
h2e       = frag_data[5]
n_alpha   = int(frag_data[3][0])
n_beta    = int(frag_data[3][1])
n_orb     = h1e.shape[0]
n_qubits  = 2 * n_orb

print(f"\nFragment {most_active_frag} (Fe): "
      f"{n_orb} orbs | {n_alpha}α + {n_beta}β | {n_qubits} qubits")
print(f"Max valid configs = C({n_orb},{n_alpha})² = {math.comb(n_orb, n_alpha)**2}")

# ── FCI reference on the same embedded Hamiltonian ────────────────────────────
cisolver  = pyscf_fci.direct_spin1.FCI()
fci_energy, _ = cisolver.kernel(h1e, h2e, n_orb, (n_alpha, n_beta))
print(f"FCI fragment energy = {fci_energy:.8f} Ha  ← SQD target")

# ── Bitstring filter helper ───────────────────────────────────────────────────
def filter_bitstrings(bsm, probs, n_alpha, n_beta, n_orb):
    """Keep only bitstrings with correct α and β electron counts."""
    valid = (
        (bsm[:, :n_orb].sum(axis=1) == n_alpha) &
        (bsm[:, n_orb:].sum(axis=1) == n_beta)
    )
    return bsm[valid], probs[valid]

# ── Quantum circuit ───────────────────────────────────────────────────────────
hf_circ = QuantumCircuit(n_qubits)
for i in range(n_alpha): hf_circ.x(i)           # occupy α spin-orbitals
for i in range(n_beta):  hf_circ.x(n_orb + i)   # occupy β spin-orbitals

ansatz  = efficient_su2(
    n_qubits,
    reps                      = 3,
    entanglement              = "full",
    skip_final_rotation_layer = True,
)
rng     = np.random.default_rng(42)
params  = rng.uniform(0, 2*np.pi, ansatz.num_parameters)
circuit = hf_circ.compose(ansatz.assign_parameters(params))
circuit.measure_all()
print(f"\nCircuit: {n_qubits} qubits | depth={circuit.depth()} | "
      f"{ansatz.num_parameters} params")

# ── Sample bitstrings ─────────────────────────────────────────────────────────
n_shots = 500_000
counts  = (
    StatevectorSampler()
    .run([circuit], shots=n_shots)
    .result()[0]
    .data.meas
    .get_counts()
)
bsm, probs = counts_to_arrays(counts)
bsm, probs = filter_bitstrings(bsm, probs, n_alpha, n_beta, n_orb)
print(f"Valid bitstrings after filtering: "
      f"{bsm.shape[0]} / {math.comb(n_orb, n_alpha)**2} max")

if bsm.shape[0] == 0:
    raise RuntimeError("No valid bitstrings after filtering — increase n_shots")

# ── SQD iterative loop ────────────────────────────────────────────────────────
avg_occs = (
    np.array([1.0 if i < n_alpha else 0.0 for i in range(n_orb)]),
    np.array([1.0 if i < n_beta  else 0.0 for i in range(n_orb)]),
)
sqd_energy  = None
spin_sq_val = None

print(f"\nSQD iterations (FCI target = {fci_energy:.8f} Ha):")
print("─" * 68)

for it in range(10):
    bsm, probs = recover_configurations(
        bsm, probs, avg_occs,
        num_elec_a = n_alpha,
        num_elec_b = n_beta,
        rand_seed  = 42,
    )
    if bsm.shape[0] == 0:
        print(f"  [iter {it+1}] No valid configs after recovery — stopping.")
        break

    sqd_energy, _, avg_occs, spin_sq_val = solve_fermion(
        bsm,
        hcore      = h1e,
        eri        = h2e,
        open_shell = False,
        spin_sq    = 0.0,
    )
    print(f"  Iter {it+1:02d} | E={sqd_energy:.8f} Ha | "
          f"configs={bsm.shape[0]:4d} | "
          f"<S²>={spin_sq_val:.4f} | "
          f"ΔE={abs(sqd_energy-fci_energy):.2e} Ha")

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'═'*68}")
print(f"  Molecule                          : {MOLECULE}")
print(f"  ASF: {nel}e in {n_active_orbs} orbs → orbitals {mo_list}")
print(f"  ASF-guided most active atom       : {most_active_atom} ({atom_syms[most_active_atom]})")
print(f"  DMET fragments                    : {fragment_atoms}  ([Fe] | [6×N])")
print(f"  DMET fragment solvers             : {fragment_solvers}")
print(f"  SQD target fragment               : {most_active_frag} (Fe)")
print(f"  FCI fragment {most_active_frag} energy         : {fci_energy:.8f} Ha")
print(f"  SQD fragment {most_active_frag} energy         : {sqd_energy:.8f} Ha")
print(f"  Δ (SQD vs FCI)                    : {abs(sqd_energy-fci_energy):.2e} Ha")
print(f"  Final <S²>                        : {spin_sq_val:.6f}  (0=singlet ✓)")
print(f"  DMET total energy                 : {dmet_energy:.8f} Ha")
print(f"{'═'*68}")