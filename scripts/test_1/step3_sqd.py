"""
Step 3 — Quantum circuit sampling + SQD iterative solver
         Uses the truncated (SQD_MAX_ORBS) active space from Step 2.

Requires: results/step1_asf.pkl
          results/step2_dmet.pkl
Runtime:  ~5-20 min
"""
import os
import sys
import math
import pickle
import numpy as np

import config

# ── Check dependencies ────────────────────────────────────────────────────────
for fpath, label in [(config.STEP1_FILE, "Step 1 (ASF)"),
                     (config.STEP2_FILE, "Step 2 (DMET)")]:
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"[Step 3] {label} results not found: {fpath}\n"
            "Run the corresponding script first."
        )

# ── Load results ──────────────────────────────────────────────────────────────
with open(config.STEP1_FILE, "rb") as f:
    step1 = pickle.load(f)
with open(config.STEP2_FILE, "rb") as f:
    step2 = pickle.load(f)

# Step 1
nel              = step1["nel"]
mo_list          = step1["mo_list"]
n_active_orbs    = step1["n_active_orbs"]
most_active_atom = step1["most_active_atom"]

# Step 2 — use truncated active space integrals
h1e         = step2["h1e_act"]
h2e         = step2["h2e_act"]
n_alpha     = step2["n_alpha_act"]
n_beta      = step2["n_beta_act"]
n_orb       = step2["n_orb_act"]
dmet_energy = step2["dmet_energy"]

# Reference info
n_orb_full  = step2["n_orb_full"]
sel         = step2["sel"]

n_qubits = 2 * n_orb

print(f"\n[Step 3] Loaded results from Steps 1 & 2")
print(f"  Full Fe embedding     : {n_orb_full} orbs (CCSD used in DMET)")
print(f"  SQD active space      : {n_orb} orbs (indices {sel} in Fock order)")
print(f"  n_alpha / n_beta      : {n_alpha} / {n_beta}")
print(f"  n_qubits              : {n_qubits}")
print(f"  Max valid configs     : C({n_orb},{n_alpha})² = {math.comb(n_orb, n_alpha)**2}")

# ── Imports ───────────────────────────────────────────────────────────────────
from pyscf import fci as pyscf_fci
from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorSampler
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.fermion import solve_fermion
from qiskit_addon_sqd.configuration_recovery import recover_configurations

# ── FCI reference on truncated Hamiltonian ────────────────────────────────────
# With n_orb = SQD_MAX_ORBS (e.g. 8), this is fast: C(8,4)² = 4900 determinants
cisolver  = pyscf_fci.direct_spin1.FCI()
fci_energy, _ = cisolver.kernel(h1e, h2e, n_orb, (n_alpha, n_beta))
print(f"  FCI active-space energy = {fci_energy:.8f} Ha  ← SQD target")

# ── Bitstring filter ──────────────────────────────────────────────────────────
def filter_bitstrings(bsm, probs, n_alpha, n_beta, n_orb):
    """Keep only bitstrings with correct α and β electron counts."""
    valid = (
        (bsm[:, :n_orb].sum(axis=1) == n_alpha) &
        (bsm[:, n_orb:].sum(axis=1) == n_beta)
    )
    return bsm[valid], probs[valid]

# ── Quantum circuit ───────────────────────────────────────────────────────────
hf_circ = QuantumCircuit(n_qubits)
for i in range(n_alpha): hf_circ.x(i)           # fill α spin-orbitals
for i in range(n_beta):  hf_circ.x(n_orb + i)   # fill β spin-orbitals

ansatz = efficient_su2(
    n_qubits,
    reps                      = config.ANSATZ_REPS,
    entanglement              = "full",
    skip_final_rotation_layer = True,
)
rng    = np.random.default_rng(42)
params = rng.uniform(0, 2*np.pi, ansatz.num_parameters)
circuit = hf_circ.compose(ansatz.assign_parameters(params))
circuit.measure_all()

print(f"\n  Circuit: {n_qubits} qubits | depth={circuit.depth()} "
      f"| {ansatz.num_parameters} params")

# ── Sample ────────────────────────────────────────────────────────────────────
print(f"\n[Step 3] Sampling {config.N_SHOTS:,} shots...")
counts = (
    StatevectorSampler()
    .run([circuit], shots=config.N_SHOTS)
    .result()[0]
    .data.meas
    .get_counts()
)
bsm, probs = counts_to_arrays(counts)
bsm, probs = filter_bitstrings(bsm, probs, n_alpha, n_beta, n_orb)
print(f"  Valid bitstrings: {bsm.shape[0]} / {math.comb(n_orb, n_alpha)**2} max")

if bsm.shape[0] == 0:
    raise RuntimeError(
        "No valid bitstrings — increase N_SHOTS in config.py"
    )

# ── SQD iterative loop ────────────────────────────────────────────────────────
avg_occs = (
    np.array([1.0 if i < n_alpha else 0.0 for i in range(n_orb)]),
    np.array([1.0 if i < n_beta  else 0.0 for i in range(n_orb)]),
)
sqd_energy  = None
spin_sq_val = None

print(f"\nSQD iterations (FCI target = {fci_energy:.8f} Ha):")
print("─" * 68)

for it in range(config.SQD_ITERS):
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
print(f"  Molecule                          : {config.MOLECULE}")
print(f"  ASF: {nel}e in {n_active_orbs} orbs → orbitals {mo_list}")
print(f"  ASF-guided most active atom       : {most_active_atom} "
      f"({config.ATOM_SYMS[most_active_atom]})")
print(f"  DMET fragments                    : {config.FRAGMENT_ATOMS}  ([Fe] | [6×N])")
print(f"  DMET fragment solvers             : {config.FRAGMENT_SOLVERS}")
print(f"  DMET total energy                 : {dmet_energy:.8f} Ha")
print(f"  Fe embedding size (full)          : {n_orb_full} orbs")
print(f"  SQD active space                  : {n_orb} orbs, {n_qubits} qubits")
print(f"  FCI active-space energy           : {fci_energy:.8f} Ha")
print(f"  SQD active-space energy           : {sqd_energy:.8f} Ha")
print(f"  Δ (SQD vs FCI)                    : {abs(sqd_energy-fci_energy):.2e} Ha")
print(f"  Final <S²>                        : {spin_sq_val:.6f}  (0=singlet ✓)")
print(f"{'═'*68}")