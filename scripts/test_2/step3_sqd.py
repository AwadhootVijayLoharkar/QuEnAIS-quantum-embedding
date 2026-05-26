"""
Step 3 — SQD: sample quantum circuit, iteratively solve embedded Hamiltonian

Requires: results/step1_asf.pkl
          results/step2_hamiltonian.pkl
Runtime:  ~5-20 min
"""
import os, sys, math, pickle
import numpy as np

import config

# ── Check dependencies ────────────────────────────────────────────────────────
for fpath, label in [(config.STEP1_FILE, "Step 1 (ASF)"),
                     (config.STEP2_FILE, "Step 2 (Hamiltonian)")]:
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"[Step 3] {label} not found: {fpath}\n"
            "Run the corresponding script first."
        )

with open(config.STEP1_FILE, "rb") as f:
    step1 = pickle.load(f)
with open(config.STEP2_FILE, "rb") as f:
    step2 = pickle.load(f)

# Step 1
nel              = step1["nel"]
mo_list          = step1["mo_list"]
n_active_orbs    = step1["n_active_orbs"]
most_active_atom = step1["most_active_atom"]

# Step 2
h1e        = step2["h1e"]
h2e        = step2["h2e"]
ecore      = step2["ecore"]
n_alpha    = step2["n_alpha"]
n_beta     = step2["n_beta"]
n_emb      = step2["n_emb"]
n_imp      = step2["n_imp"]
n_bath     = step2["n_bath"]
fci_ref_e  = step2["fci_ref_e"]
uhf_energy = step2["uhf_energy"]
sv         = step2["sv"]

n_qubits = 2 * n_emb
max_cfg  = math.comb(n_emb, n_alpha) ** 2

print(f"\n[Step 3] SQD on DMET-embedded Hamiltonian")
print(f"  Impurity          : {n_imp} ASF orbitals {mo_list}")
print(f"  Bath              : {n_bath} orbitals  (Schmidt values: {sv.round(4)})")
print(f"  Embedding size    : {n_emb} orbitals = {n_qubits} qubits")
print(f"  Electrons         : {n_alpha + n_beta} ({n_alpha}α + {n_beta}β)")
print(f"  Max configurations: {max_cfg:,}")
if fci_ref_e is not None:
    print(f"  FCI reference     : {fci_ref_e:.8f} Ha  ← target")
else:
    print(f"  FCI reference     : not computed (too large)")

if n_qubits > 30:
    print(f"\n  ⚠  {n_qubits} qubits is large for statevector simulation.")
    print(f"     Consider increasing BATH_TOLERANCE or decreasing MAX_EMBED_ORBS in config.py")

# ── Imports ───────────────────────────────────────────────────────────────────
from pyscf import fci as pyscf_fci
from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorSampler
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.fermion import solve_fermion
from qiskit_addon_sqd.configuration_recovery import recover_configurations

# ── FCI reference (if not computed in step 2) ─────────────────────────────────
if fci_ref_e is None and max_cfg <= 5_000_000:
    print("\n[Step 3] Computing FCI reference...")
    cisolver  = pyscf_fci.direct_spin1.FCI()
    fci_ref_e, _ = cisolver.kernel(h1e, h2e, n_emb, (n_alpha, n_beta))
    print(f"  FCI reference: {fci_ref_e:.8f} Ha")

# ── Bitstring filter ──────────────────────────────────────────────────────────
def filter_bitstrings(bsm, probs, n_alpha, n_beta, n_orb):
    """Keep only bitstrings with the correct number of α and β electrons."""
    valid = (
        (bsm[:, :n_orb].sum(axis=1) == n_alpha) &
        (bsm[:, n_orb:].sum(axis=1) == n_beta)
    )
    return bsm[valid], probs[valid]

# ── Build quantum circuit ─────────────────────────────────────────────────────
# HF reference state: fill lowest α and β spin-orbitals
hf_circ = QuantumCircuit(n_qubits)
for i in range(n_alpha): hf_circ.x(i)
for i in range(n_beta):  hf_circ.x(n_emb + i)

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

print(f"\n[Step 3] Circuit: {n_qubits} qubits | depth={circuit.depth()} "
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
bsm, probs = filter_bitstrings(bsm, probs, n_alpha, n_beta, n_emb)
print(f"  Valid bitstrings: {bsm.shape[0]:,} / {max_cfg:,} max")

if bsm.shape[0] == 0:
    raise RuntimeError("No valid bitstrings — increase N_SHOTS in config.py")

# ── SQD iterative loop ────────────────────────────────────────────────────────
avg_occs = (
    np.array([1.0 if i < n_alpha else 0.0 for i in range(n_emb)]),
    np.array([1.0 if i < n_beta  else 0.0 for i in range(n_emb)]),
)
sqd_energy  = None
spin_sq_val = None

target_str = f"{fci_ref_e:.8f}" if fci_ref_e is not None else "N/A"
print(f"\nSQD iterations (FCI target = {target_str} Ha):")
print("─" * 70)

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

    delta_str = (f"{abs(sqd_energy - fci_ref_e):.2e}"
                 if fci_ref_e is not None else "N/A")
    print(f"  Iter {it+1:02d} | E={sqd_energy:.8f} Ha | "
          f"configs={bsm.shape[0]:4d} | "
          f"<S²>={spin_sq_val:.4f} | "
          f"ΔE={delta_str} Ha")

# ── Final summary ─────────────────────────────────────────────────────────────
delta_final = (f"{abs(sqd_energy - fci_ref_e):.2e} Ha"
               if fci_ref_e is not None else "N/A")
print(f"\n{'═'*70}")
print(f"  Molecule                  : {config.MOLECULE}")
print(f"  ASF active space          : {nel}e in {n_active_orbs} orbs → {mo_list}")
print(f"  DMET impurity             : {n_imp} orbitals (ASF)")
print(f"  DMET bath                 : {n_bath} orbitals (Schmidt decomp)")
print(f"  DMET core                 : traced out → mean-field h1e_eff")
print(f"  Total embedding           : {n_emb} orbitals = {n_qubits} qubits")
print(f"  UHF energy                : {uhf_energy:.8f} Ha")
print(f"  FCI embedded energy       : {target_str} Ha")
print(f"  SQD embedded energy       : {sqd_energy:.8f} Ha")
print(f"  Δ (SQD vs FCI)            : {delta_final}")
print(f"  Final <S²>                : {spin_sq_val:.6f}  (0 = singlet ✓)")
print(f"{'═'*70}")