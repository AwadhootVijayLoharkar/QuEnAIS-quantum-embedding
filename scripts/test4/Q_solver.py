# step3_solver.py — Quantum Solver (SQD / SKQD / SqDRIFT)
"""
Solves the embedded Hamiltonian from Step 2 using a quantum(-inspired) solver.

Solvers:
  SQD     — Random ansatz sampling + iterative subspace diagonalization
  SKQD    — Krylov time-evolution sampling + cumulative diagonalization
  SqDRIFT — qDRIFT ensemble sampling + iterative diagonalization

Requires: results/step1_asf.pkl, results/step2_hamiltonian.pkl
Saves:    results/step3_results.pkl
"""

import os
import sys
import math
import pickle
import numpy as np
from collections import Counter

import config

# ── Setup ─────────────────────────────────────────────────────────────────────
FORCE_RERUN = True
STEP3_FILE  = os.path.join(config.RESULTS_DIR, "step3_results.pkl")

os.makedirs(config.RESULTS_DIR, exist_ok=True)
if os.path.exists(STEP3_FILE) and not FORCE_RERUN:
    print(f"[Step 3] Cached: {STEP3_FILE}")
    sys.exit(0)

for path, name in [(config.STEP1_FILE, "Step 1"), (config.STEP2_FILE, "Step 2")]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run {name} first. Missing: {path}")

with open(config.STEP1_FILE, "rb") as f:
    step1 = pickle.load(f)
with open(config.STEP2_FILE, "rb") as f:
    step2 = pickle.load(f)

h1e        = step2["h1e"]
h2e        = step2["h2e"]
n_emb      = step2["n_emb"]
n_alpha    = step2["n_alpha"]
n_beta     = step2["n_beta"]
uhf_energy = step2["uhf_energy"]
mp2_corr   = step2["mp2_corr"]
mp2_energy = uhf_energy + mp2_corr
mol_info   = step1["mol_info"]

n_qubits = 2 * n_emb

print(f"\n{'='*60}")
print(f"[Step 3] Quantum Solver — {mol_info['molecule']}")
print(f"{'='*60}")
print(f"  Solver   : {config.QUANTUM_SOLVER.upper()}")
print(f"  Embedding: {n_emb} orbs = {n_qubits} qubits")
print(f"  Electrons: {n_alpha}α + {n_beta}β")
print(f"  UHF ref  : {uhf_energy:.8f} Ha")
print(f"  MP2 ref  : {mp2_energy:.8f} Ha")

from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.fermion import solve_fermion
from qiskit_addon_sqd.configuration_recovery import recover_configurations


# ═══════════════════════════════════════════════════════════════════════════════
# Backend dispatch
# ═══════════════════════════════════════════════════════════════════════════════

def sample_circuits(circuits, shots):
    """Run circuits on configured backend. Returns list of count dicts."""
    backend = config.BACKEND.lower()

    if backend == "local":
        from qiskit.primitives import StatevectorSampler
        res = StatevectorSampler().run(circuits, shots=shots).result()
        return [res[i].data.meas.get_counts() for i in range(len(circuits))]

    elif backend == "mps":
        from qiskit_aer import AerSimulator
        from qiskit import transpile
        sim = AerSimulator(
            method="matrix_product_state",
            matrix_product_state_max_bond_dimension=config.MPS_MAX_BOND_DIM,
            matrix_product_state_truncation_threshold=config.MPS_TRUNC_THRESH,
        )
        tc = transpile(circuits, backend=sim, optimization_level=1)
        result = sim.run(tc, shots=shots).result()
        return [result.get_counts(i) for i in range(len(circuits))]

    elif backend == "ibm":
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        service = QiskitRuntimeService()
        hw = (service.backend(config.IBM_BACKEND_NAME) if config.IBM_BACKEND_NAME
              else service.least_busy(operational=True, simulator=False,
                                      min_num_qubits=circuits[0].num_qubits))
        pm = generate_preset_pass_manager(config.IBM_OPTIMIZATION_LEVEL, backend=hw)
        tc = pm.run(circuits)
        job = SamplerV2(mode=hw).run([(c,) for c in tc], shots=shots)
        result = job.result()
        return [result[i].data.meas.get_counts() for i in range(len(circuits))]

    else:
        raise ValueError(f"Unknown BACKEND: '{backend}'")


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def filter_bitstrings(bsm, probs):
    """Keep only bitstrings with correct (n_alpha, n_beta) particle number."""
    valid = ((bsm[:, :n_emb].sum(axis=1) == n_alpha) &
             (bsm[:, n_emb:].sum(axis=1) == n_beta))
    return bsm[valid], probs[valid]


def hf_bitstring():
    """Hartree-Fock reference as a boolean array."""
    row = np.zeros(2 * n_emb, dtype=bool)
    for i in range(n_alpha): row[i] = True
    for i in range(n_beta):  row[n_emb + i] = True
    return row


def print_iteration_header():
    """Print table header for iteration tracking."""
    print(f"\n  {'─'*80}")
    print(f"  {'Iter':>4} | {'Energy (Ha)':>14} | {'configs':>7} | "
          f"{'vs UHF':>12} | {'vs MP2':>12} | {'ΔE(prev)':>12}")
    print(f"  {'─'*80}")


def print_iteration(label, energy, n_configs, prev_energy=None):
    """Print one iteration with energy gains."""
    vs_uhf = energy - uhf_energy
    vs_mp2 = energy - mp2_energy
    delta  = f"{energy - prev_energy:+.6f}" if prev_energy is not None else "    ---"

    # Color indicators
    uhf_mark = "↓" if vs_uhf < 0 else "↑"
    mp2_mark = "↓" if vs_mp2 < 0 else "↑"

    print(f"  {label:>4} | {energy:>14.8f} | {n_configs:>7d} | "
          f"{vs_uhf:+.6f} {uhf_mark} | {vs_mp2:+.6f} {mp2_mark} | {delta}")


def iterative_solve(bsm, probs, n_iters):
    """Shared iterative recover + solve loop used by SQD and SqDRIFT."""
    avg_occs = (
        np.array([1.0 if i < n_alpha else 0.0 for i in range(n_emb)]),
        np.array([1.0 if i < n_beta  else 0.0 for i in range(n_emb)]),
    )

    iterations = []
    energy, spin_sq, prev_energy = None, None, None

    print_iteration_header()

    for it in range(n_iters):
        bsm, probs = recover_configurations(
            bsm, probs, avg_occs,
            num_elec_a=n_alpha, num_elec_b=n_beta, rand_seed=42 + it,
        )
        if bsm.shape[0] == 0:
            break

        energy, _, avg_occs, spin_sq = solve_fermion(
            bsm, hcore=h1e, eri=h2e, open_shell=False, spin_sq=0.0,
        )

        print_iteration(f"{it+1:02d}", energy, bsm.shape[0], prev_energy)

        iterations.append({"iter": it+1, "energy": float(energy),
                           "n_configs": int(bsm.shape[0]),
                           "vs_uhf": float(energy - uhf_energy),
                           "vs_mp2": float(energy - mp2_energy)})
        prev_energy = energy

    print(f"  {'─'*80}")
    return energy, spin_sq, iterations


# ═══════════════════════════════════════════════════════════════════════════════
# SQD
# ═══════════════════════════════════════════════════════════════════════════════

def run_sqd():
    """Random ansatz sampling + iterative subspace diagonalization."""
    print("\n── SQD ──")

    hf_circ = QuantumCircuit(n_qubits)
    for i in range(n_alpha): hf_circ.x(i)
    for i in range(n_beta):  hf_circ.x(n_emb + i)

    ansatz = efficient_su2(n_qubits, reps=config.ANSATZ_REPS,
                           entanglement="full", skip_final_rotation_layer=True)
    params = np.random.default_rng(42).uniform(0, 2*np.pi, ansatz.num_parameters)
    circ = hf_circ.compose(ansatz.assign_parameters(params))
    circ.measure_all()

    print(f"  Circuit: {n_qubits}q, depth={circ.depth()}")
    print(f"  Sampling {config.N_SHOTS} shots...")

    raw = sample_circuits([circ], config.N_SHOTS)[0]
    bsm, probs = counts_to_arrays(raw)
    bsm, probs = filter_bitstrings(bsm, probs)
    print(f"  Valid configs: {bsm.shape[0]}")

    energy, spin_sq, iterations = iterative_solve(bsm, probs, config.SQD_ITERS)
    return energy, spin_sq, iterations


# ═══════════════════════════════════════════════════════════════════════════════
# SKQD
# ═══════════════════════════════════════════════════════════════════════════════

def run_skqd():
    """Krylov time-evolution sampling."""
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import LieTrotter

    print("\n── SKQD ──")
    print(f"  Krylov dim={config.SKQD_KRYLOV_DIM}, dt={config.SKQD_DT}")

    H_qubit = _build_jw_hamiltonian()
    print(f"  JW Hamiltonian: {len(H_qubit)} Pauli terms")

    ref = QuantumCircuit(n_qubits)
    for i in range(n_alpha): ref.x(i)
    for i in range(n_beta):  ref.x(n_emb + i)

    evol = PauliEvolutionGate(
        H_qubit, time=config.SKQD_DT / config.SKQD_TROTTER_REPS,
        synthesis=LieTrotter(reps=config.SKQD_TROTTER_REPS),
    )

    circs = []
    for k in range(config.SKQD_KRYLOV_DIM):
        qc = ref.copy()
        for _ in range(k):
            qc.append(evol, range(n_qubits))
        qc.measure_all()
        circs.append(qc)

    print(f"  Sampling {len(circs)} Krylov circuits...")
    all_counts = sample_circuits(circs, config.SKQD_SHOTS)

    iterations = []
    energy, spin_sq, prev_energy = None, None, None
    cumulative = Counter()

    print_iteration_header()

    for k, raw in enumerate(all_counts):
        cumulative.update(raw)
        bsm, probs = counts_to_arrays(dict(cumulative))
        bsm, probs = filter_bitstrings(bsm, probs)

        if bsm.shape[0] < 2:
            continue

        try:
            energy, _, _, spin_sq = solve_fermion(
                bsm, hcore=h1e, eri=h2e, open_shell=False, spin_sq=0.0,
            )

            print_iteration(f"k={k}", energy, bsm.shape[0], prev_energy)

            iterations.append({"k": k, "energy": float(energy),
                               "n_configs": int(bsm.shape[0]),
                               "vs_uhf": float(energy - uhf_energy),
                               "vs_mp2": float(energy - mp2_energy)})
            prev_energy = energy
        except Exception as e:
            print(f"  k={k:2d} | Error: {e}")

    print(f"  {'─'*80}")
    return energy, spin_sq, iterations


# ═══════════════════════════════════════════════════════════════════════════════
# SqDRIFT
# ═══════════════════════════════════════════════════════════════════════════════

def run_sqdrift():
    """qDRIFT ensemble sampling + iterative diagonalization."""
    try:
        from qiskit_fermions.operators.library import FCIDump
        from qiskit_fermions.operators import FermionOperator
        from qiskit_fermions.operators.grouping import group_terms_by_electronic_structure
        from qiskit_fermions.circuit import FermionicCircuit
        from qiskit_fermions.circuit.library import Evolution
        from qiskit_fermions.transpiler.presets import generate_preset_jw_pass_manager
        from qiskit_fermions.transpiler.passes import QDriftTrotterization
        from qiskit_fermions.transpiler import FermionicPassManager
    except ImportError:
        raise ImportError("qiskit-fermions required for SqDRIFT. pip install qiskit-fermions")

    import tempfile
    from pyscf.tools import fcidump as pyscf_fcidump

    print("\n── SqDRIFT ──")
    print(f"  circuits={config.SQDRIFT_NUM_CIRCUITS}, groups={config.SQDRIFT_NUM_GROUPS}")

    fd, tmp = tempfile.mkstemp(suffix=".fcidump")
    os.close(fd)
    try:
        pyscf_fcidump.from_integrals(tmp, h1e, h2e, n_emb, n_alpha + n_beta,
                                     ms=abs(n_alpha - n_beta))
        hamil = FermionOperator.from_fcidump(FCIDump.from_file(tmp))
    finally:
        os.unlink(tmp)

    group_terms_by_electronic_structure(hamil, n_qubits)

    evo = Evolution(n_qubits, hamil, config.SQDRIFT_TIME)
    template = FermionicCircuit(n_qubits)
    template.append(evo, template.modes)

    pm = generate_preset_jw_pass_manager()
    circuits = []
    for i in range(config.SQDRIFT_NUM_CIRCUITS):
        pm.optimization = FermionicPassManager(
            [QDriftTrotterization(config.SQDRIFT_NUM_GROUPS, rng=42 + i)]
        )
        transpiled = pm.run(template)

        hf_qc = QuantumCircuit(n_qubits)
        for j in range(n_alpha): hf_qc.x(j)
        for j in range(n_beta):  hf_qc.x(n_emb + j)

        full = hf_qc.compose(transpiled)
        full.measure_all()
        circuits.append(full)

    print(f"  Sampling {len(circuits)} circuits...")
    all_counts = sample_circuits(circuits, config.SQDRIFT_SHOTS)

    cumulative = Counter()
    for counts in all_counts:
        cumulative.update(counts)

    bsm, probs = counts_to_arrays(dict(cumulative))
    bsm, probs = filter_bitstrings(bsm, probs)

    # Inject HF reference
    hf_row = hf_bitstring()
    if bsm.shape[0] == 0 or not any(np.array_equal(bsm[i], hf_row) for i in range(bsm.shape[0])):
        bsm = np.vstack([bsm, hf_row[np.newaxis, :]]) if bsm.shape[0] > 0 else hf_row[np.newaxis, :]
        probs = np.append(probs, 1.0 / max(bsm.shape[0], 1))
        probs /= probs.sum()

    print(f"  Valid configs: {bsm.shape[0]}")

    energy, spin_sq, iterations = iterative_solve(bsm, probs, config.SQDRIFT_ITERS)
    return energy, spin_sq, iterations


# ═══════════════════════════════════════════════════════════════════════════════
# JW Hamiltonian builder (for SKQD)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_jw_hamiltonian():
    """Build Jordan-Wigner qubit Hamiltonian from h1e, h2e."""
    from openfermion import FermionOperator as OF_FermionOp, jordan_wigner
    from qiskit.quantum_info import SparsePauliOp

    n_so = 2 * n_emb
    fop = OF_FermionOp()

    for p in range(n_emb):
        for q in range(n_emb):
            h = complex(h1e[p, q])
            if abs(h) < 1e-10:
                continue
            fop += OF_FermionOp(f"{p}^ {q}", h)
            fop += OF_FermionOp(f"{n_emb+p}^ {n_emb+q}", h)

    for p in range(n_emb):
        for q in range(n_emb):
            for r in range(n_emb):
                for s in range(n_emb):
                    h = 0.5 * complex(h2e[p, q, r, s])
                    if abs(h) < 1e-10:
                        continue
                    np_, nq, nr, ns = n_emb+p, n_emb+q, n_emb+r, n_emb+s
                    fop += OF_FermionOp(((p,1),(r,1),(s,0),(q,0)), h)
                    fop += OF_FermionOp(((np_,1),(nr,1),(ns,0),(nq,0)), h)
                    fop += OF_FermionOp(((p,1),(nr,1),(ns,0),(q,0)), h)
                    fop += OF_FermionOp(((np_,1),(r,1),(s,0),(nq,0)), h)

    jw = jordan_wigner(fop)

    labels, coeffs = [], []
    for term, coeff in jw.terms.items():
        arr = ['I'] * n_so
        for idx, pauli in term:
            arr[idx] = pauli
        labels.append(''.join(reversed(arr)))
        coeffs.append(complex(coeff))

    if not labels:
        return SparsePauliOp('I' * n_so, coeffs=[0.0])

    op = SparsePauliOp(labels, coeffs=coeffs).simplify()
    # Drop imaginary noise (Hamiltonian is Hermitian → real coefficients)
    op = SparsePauliOp(op.paulis, coeffs=np.real(op.coeffs))
    return op


# ═══════════════════════════════════════════════════════════════════════════════
# Dispatch and run
# ═══════════════════════════════════════════════════════════════════════════════

solvers = {"sqd": run_sqd, "skqd": run_skqd, "sqdrift": run_sqdrift}

if config.QUANTUM_SOLVER not in solvers:
    raise ValueError(f"Unknown solver: '{config.QUANTUM_SOLVER}'. Use: {list(solvers.keys())}")

energy, spin_sq, iterations = solvers[config.QUANTUM_SOLVER]()

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"[Step 3] Final Summary: {mol_info['molecule']}")
print(f"{'='*60}")
if energy is not None:
    print(f"  Solver energy  : {energy:.8f} Ha")
    print(f"  UHF energy     : {uhf_energy:.8f} Ha")
    print(f"  MP2 energy     : {mp2_energy:.8f} Ha")
    print(f"  ────────────────────────────────────")
    print(f"  Gain vs UHF    : {energy - uhf_energy:+.8f} Ha  "
          f"({'lower ✓' if energy < uhf_energy else 'higher ✗'})")
    print(f"  Gain vs MP2    : {energy - mp2_energy:+.8f} Ha  "
          f"({'lower ✓' if energy < mp2_energy else 'higher ✗'})")
    if len(iterations) > 1:
        first_e = iterations[0]["energy"]
        print(f"  Improvement    : {energy - first_e:+.8f} Ha over {len(iterations)} iterations")
else:
    print(f"  Energy: N/A")
if spin_sq is not None:
    print(f"  <S²>           : {spin_sq:.6f}  (0 = singlet)")
print(f"{'='*60}")

# ── Save ──────────────────────────────────────────────────────────────────────
output = {
    "solver"      : config.QUANTUM_SOLVER,
    "energy"      : float(energy) if energy is not None else None,
    "uhf_energy"  : uhf_energy,
    "mp2_energy"  : mp2_energy,
    "spin_sq"     : float(spin_sq) if spin_sq is not None else None,
    "iterations"  : iterations,
    "mol_info"    : mol_info,
}

with open(STEP3_FILE, "wb") as f:
    pickle.dump(output, f)

print(f"\n[Step 3] ✓ Saved → {STEP3_FILE}")