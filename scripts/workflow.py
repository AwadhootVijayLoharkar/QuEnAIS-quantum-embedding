import os

# ── MKL / block2 fix ─────────────────────────────────────────────────────────
_WRAPPER = os.path.expanduser("~/block2main_wrapper.sh")
os.environ["BLOCKEXE"]          = _WRAPPER
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_DEBUG_CPU_TYPE"]  = "5"

# Tell PySCF's DMRG interface to use the wrapper
from pyscf.dmrgscf import dmrgci
dmrgci.settings.BLOCKEXE = _WRAPPER

import math
import numpy as np
from pyscf import gto, scf, fci as pyscf_fci, lo


from asf.wrapper import (
    find_from_mol,          # Mole → ActiveSpace (entropy threshold)
    find_from_scf,          # SCF  → ActiveSpace (entropy threshold)
    sized_space_from_mol,   # Mole → ActiveSpace (exact size)
    sized_space_from_scf,   # SCF  → ActiveSpace (exact size)
    reorder_mos,            # utility: sort cols as occ/active/virt
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
    verbose = 3,            # INFO level → prints entropy table from ASF
)



# ─────────────────────────────────────────────────────────────────────────────
# Option A: find_from_mol — ASF does UHF + stability analysis internally
#           Best for general use, follows exact same procedure as in docs
# ─────────────────────────────────────────────────────────────────────────────
active_space = find_from_mol(
    mol_pyscf,
    entropy_threshold = 0.15,   # orbitals with S > threshold are selected
                                # lower  → more orbitals selected
                                # higher → fewer orbitals selected
    max_norb          = 8,      # cap active space size (optional)
    min_norb          = 2,      # minimum active space size (optional)
    verbose           = True,   # prints entropy table
)

# ─────────────────────────────────────────────────────────────────────────────
# Option B: find_from_scf — bring your own RHF/UHF object
#           Use this if you want control over the SCF step
# ─────────────────────────────────────────────────────────────────────────────
# mf = scf.RHF(mol_pyscf).run()
# active_space = find_from_scf(
#     mf,
#     entropy_threshold = 0.15,
#     states            = 1,    # ground state only
#                               # states=2 → ground + first excited
#                               # states=[(0,1),(2,1)] → 1 singlet + 1 triplet
#     verbose           = True,
# )

# ─────────────────────────────────────────────────────────────────────────────
# Option C: sized_space_from_mol — request exact active space size
#           Use when you know exactly how many orbitals you want
# ─────────────────────────────────────────────────────────────────────────────
# active_space = sized_space_from_mol(
#     mol_pyscf,
#     size    = (2, 2),   # (n_electrons, n_orbitals) e.g. (2,2) = HOMO+LUMO
#                         # or just size=4 → 4 orbitals, electrons inferred
#     verbose = True,
# )
# ── Unpack ActiveSpace object ─────────────────────────────────────────────────
# Returns ActiveSpace dataclass with .nel, .mo_list, .mo_coeff
nel      = active_space.nel        # int: n_active_electrons
mo_list  = active_space.mo_list    # list[int]: active orbital indices (0-based, MP2 NO basis)
mo_coeff = active_space.mo_coeff   # ndarray (n_AO, n_MO): MP2 natural orbital coefficients

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

    # ── Map active orbitals → atoms via Mulliken population ───────────────────────
# mo_coeff[:, mo_list] → coefficients of active NOs only
# mo_list indices refer to columns of mo_coeff (MP2 NO basis)

S         = mol_pyscf.intor("int1e_ovlp")      # AO overlap (n_AO, n_AO)
ao_labels = mol_pyscf.ao_labels(fmt=None)       # [(atom_idx, sym, ao_name, ...), ...]

active_coeffs       = mo_coeff[:, mo_list]      # (n_AO, n_active_orbs)
orbital_atom_weight = np.zeros((n_active_orbs, n_atoms))

for orb_i in range(n_active_orbs):
    c  = active_coeffs[:, orb_i]
    CS = c * (S @ c)                            # Mulliken product per AO
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

most_active_frag = int(np.argmax(active_per_atom))
print(f"\n→ ASF-guided fragment: atom {most_active_frag} ({atom_syms[most_active_frag]})")



from tangelo import SecondQuantizedMolecule
from tangelo.problem_decomposition import DMETProblemDecomposition
from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorSampler
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.fermion import solve_fermion
from qiskit_addon_sqd.configuration_recovery import recover_configurations


# ── DMET fragment definition ──────────────────────────────────────────────────
# One fragment per atom (safest, most general)
# ASF tells us WHICH fragment is most correlated → send that one to SQD
fragment_atoms = [1] * n_atoms

print(f"Fragment definition : {fragment_atoms}  (1 per atom)")
print(f"SQD target fragment : {most_active_frag} ({atom_syms[most_active_frag]})"
      f"  ← ASF-guided")

# ── DMET ──────────────────────────────────────────────────────────────────────
#mol_tangelo = SecondQuantizedMolecule(geometry, q=0, spin=4, basis="sto-3g")


import pyscf.scf.hf as _pyscf_hf_base

_orig_scf_kernel = _pyscf_hf_base.SCF.kernel   # correct base class

def _kernel_with_newton_fallback(self, dm0=None, **kwargs):
    """DIIS first; auto-retry with Newton if not converged."""
    self.max_cycle   = max(getattr(self, "max_cycle",   50),  400)
    self.level_shift = max(getattr(self, "level_shift", 0.0), 0.5)

    result = _orig_scf_kernel(self, dm0=dm0, **kwargs)

    if not self.converged:
        print("  [SCF patch] DIIS failed → switching to Newton solver...")
        try:
            newton           = self.newton()
            newton.max_cycle = 400
            newton.kernel(self.mo_coeff)         # warm-start from DIIS MOs

            if newton.converged:
                print(f"  [SCF patch] ✓ Newton converged: E={newton.e_tot:.10f} Ha")
                # Write results back so Tangelo's `if not mf.converged` check passes
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

# ── Apply patch before ANY Tangelo/DMET SCF calls ─────────────────────────────
_pyscf_hf_base.SCF.kernel = _kernel_with_newton_fallback

mol_tangelo = None
for try_spin in [4, 2, 6, 0]:
    try:
        mol_tangelo = SecondQuantizedMolecule(
            geometry, q=0, spin=try_spin, basis="sto-3g"
        )
        print(f"✓ SecondQuantizedMolecule converged (spin={try_spin})")
        break
    except (ValueError, RuntimeError) as e:
        print(f"  sto-3g spin={try_spin} failed: {e}")

# ── Fallback: lanl2dz has a proper ECP for Fe → more stable SCF ───────────────
if mol_tangelo is None:
    print("\n  STO-3G exhausted → trying lanl2dz (ECP basis, better for Fe)...")
    for try_spin in [4, 2, 6, 0]:
        try:
            mol_tangelo = SecondQuantizedMolecule(
                geometry, q=0, spin=try_spin, basis="lanl2dz"
            )
            print(f"✓ SecondQuantizedMolecule (lanl2dz) converged (spin={try_spin})")
            break
        except (ValueError, RuntimeError) as e:
            print(f"  lanl2dz spin={try_spin} failed: {e}")

if mol_tangelo is None:
    _pyscf_hf_base.SCF.kernel = _orig_scf_kernel   # restore before raising
    raise RuntimeError(
        "SCF failed for all spin states and bases (sto-3g + lanl2dz).\n"
        "Verify the geometry is physically reasonable for Fe(II/III)."
    )

# ── DMET (keep patch active — fragment SCFs also need Newton) ─────────────────
dmet = DMETProblemDecomposition({
    "molecule"        : mol_tangelo,
    "fragment_atoms"  : fragment_atoms,
    "fragment_solvers": "ccsd",
    "verbose"         : True,
})
dmet.build()
dmet_energy = dmet.simulate()

# ── Restore kernel after all SCF work is done ─────────────────────────────────
_pyscf_hf_base.SCF.kernel = _orig_scf_kernel
print(f"\nDMET total energy = {dmet_energy:.8f} Ha")


##### added later on till here




# ── Extract ASF-guided fragment integrals ─────────────────────────────────────
# scf_fragments[i] = [RHF, h1e, Mole, [n_a,n_b], fock, h2e, fock_copy]
#                     [0]   [1]  [2]   [3]        [4]   [5]  [6]

frag_data = dmet.scf_fragments[most_active_frag]
h1e       = frag_data[1]               # embedded one-electron integrals
h2e       = frag_data[5]               # embedded two-electron integrals
n_alpha   = int(frag_data[3][0])
n_beta    = int(frag_data[3][1])
n_orb     = h1e.shape[0]
n_qubits  = 2 * n_orb

print(f"\nFragment {most_active_frag} ({atom_syms[most_active_frag]}): "
      f"{n_orb} orbs | {n_alpha}α+{n_beta}β | {n_qubits} qubits")
print(f"Max valid configs = C({n_orb},{n_alpha})² = {math.comb(n_orb, n_alpha)**2}")

# ── FCI reference on the same fragment Hamiltonian ────────────────────────────
cisolver  = pyscf_fci.direct_spin1.FCI()
fci_energy, _ = cisolver.kernel(h1e, h2e, n_orb, (n_alpha, n_beta))
print(f"FCI fragment energy = {fci_energy:.8f} Ha  ← correct SQD target")

# ── Filter helper ─────────────────────────────────────────────────────────────
def filter_bitstrings(bsm, probs, n_alpha, n_beta, n_orb):
    valid = (
        (bsm[:, :n_orb].sum(axis=1) == n_alpha) &
        (bsm[:, n_orb:].sum(axis=1) == n_beta)
    )
    return bsm[valid], probs[valid]

    



# ── Circuit ───────────────────────────────────────────────────────────────────
hf_circ = QuantumCircuit(n_qubits)
for i in range(n_alpha): hf_circ.x(i)
for i in range(n_beta):  hf_circ.x(n_orb + i)

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
print(f"\nCircuit: {n_qubits} qubits | {circuit.depth()} depth | {ansatz.num_parameters} params")

# ── Sample ────────────────────────────────────────────────────────────────────
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
print(f"Valid bitstrings: {bsm.shape[0]} / {math.comb(n_orb, n_alpha)**2} max")

if bsm.shape[0] == 0:
    raise RuntimeError("No valid bitstrings — increase n_shots")

# ── SQD loop ──────────────────────────────────────────────────────────────────
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
        print(f"  [iter {it+1}] No valid configs after recovery"); break

    sqd_energy, _, avg_occs, spin_sq_val = solve_fermion(
        bsm,
        hcore      = h1e,
        eri        = h2e,
        open_shell = False,
        spin_sq    = 0.0,
    )
    print(f"  Iter {it+1:02d} | E={sqd_energy:.8f} Ha | "
          f"configs={bsm.shape[0]:3d} | "
          f"<S²>={spin_sq_val:.4f} | "
          f"ΔE={abs(sqd_energy-fci_energy):.2e} Ha")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'═'*68}")
print(f"  Molecule                          : {MOLECULE}")
print(f"  ASF: {nel}e in {n_active_orbs} orbs → orbitals {mo_list}")
print(f"  ASF-guided fragment               : {most_active_frag} ({atom_syms[most_active_frag]})")
print(f"  DMET fragments                    : {fragment_atoms}")
print(f"  FCI fragment {most_active_frag} energy         : {fci_energy:.8f} Ha")
print(f"  SQD fragment {most_active_frag} energy         : {sqd_energy:.8f} Ha")
print(f"  Δ (SQD vs FCI)                    : {abs(sqd_energy-fci_energy):.2e} Ha")
print(f"  Final <S²>                        : {spin_sq_val:.6f}  (0=singlet ✓)")
print(f"  DMET total energy                 : {dmet_energy:.8f} Ha")
print(f"{'═'*68}")