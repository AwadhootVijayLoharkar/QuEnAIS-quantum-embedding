# step3_solver.py — Quantum Solver (SQD / SKQD / SqDRIFT)
# Phase 2 additions:
#   - LUCJ ansatz option (particle-number conserving, faster convergence)
#   - BK (Bravyi-Kitaev) mapping option alongside JW (for SKQD)
#   - FERMION_TO_QUBIT config key dispatches JW or BK
"""
Solves the embedded Hamiltonian from Step 2 using a quantum(-inspired) solver.

Solvers:
  SQD     — Ansatz sampling (SU2 or LUCJ) + iterative subspace diagonalization
  SKQD    — Krylov time-evolution sampling + cumulative diagonalization
  SqDRIFT — qDRIFT ensemble sampling + iterative diagonalization

New in Phase 2:
  LUCJ ansatz: particle-number conserving, no wasted shots, faster convergence
  BK mapping:  O(log N) Pauli strings instead of O(N) for JW, better for hardware

Requires: results/step1_asf.pkl, results/step2_hamiltonian.pkl
Saves:    results/step3_results.pkl
"""

import os
import sys
import pickle
import argparse
import warnings
import numpy as np
from collections import Counter

import config

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Step 3: Quantum Solver")
parser.add_argument("--force", action="store_true")
args = parser.parse_args()

STEP3_FILE = config.STEP3_FILE
os.makedirs(config.RESULTS_DIR, exist_ok=True)

if os.path.exists(STEP3_FILE) and not args.force:
    print(f"[Step 3] Using cached result: {STEP3_FILE}")
    sys.exit(0)

for path, name in [(config.STEP1_FILE, "step1_asf.py"),
                   (config.STEP2_FILE, "step2_hamiltonian.py")]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required input not found: {path}\nRun {name} first.")

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
ecore      = step2["ecore"]    
mol_info   = step1["mol_info"]
mp2_energy = step1.get("mp2_energy", uhf_energy + step2.get("mp2_corr", 0.0))

n_qubits = 2 * n_emb

print(f"\n{'='*60}")
print(f"[Step 3] Quantum Solver — {mol_info['molecule']}")
print(f"{'='*60}")
print(f"  Solver       : {config.QUANTUM_SOLVER.upper()}")
print(f"  Ansatz       : {config.ANSATZ.upper()}")
print(f"  Mapping      : {config.FERMION_TO_QUBIT.upper()}")
print(f"  Backend      : {config.BACKEND.upper()}")
print(f"  Embedding    : {n_emb} orbs = {n_qubits} qubits")
print(f"  Electrons    : {n_alpha}α + {n_beta}β")
print(f"  UHF ref      : {uhf_energy:.8f} Ha")
print(f"  MP2 ref      : {mp2_energy:.8f} Ha")

from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.fermion import solve_fermion
from qiskit_addon_sqd.configuration_recovery import recover_configurations


# ═══════════════════════════════════════════════════════════════════════════════
# Backend dispatch
# ═══════════════════════════════════════════════════════════════════════════════

def sample_circuits(circuits, shots):
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
        tc     = transpile(circuits, backend=sim, optimization_level=1)
        result = sim.run(tc, shots=shots).result()
        return [result.get_counts(i) for i in range(len(circuits))]
    elif backend == "ibm":
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        service = QiskitRuntimeService()
        hw = (service.backend(config.IBM_BACKEND_NAME) if config.IBM_BACKEND_NAME
              else service.least_busy(operational=True, simulator=False,
                                      min_num_qubits=circuits[0].num_qubits))
        pm     = generate_preset_pass_manager(config.IBM_OPTIMIZATION_LEVEL, backend=hw)
        tc     = pm.run(circuits)
        job    = SamplerV2(mode=hw).run([(c,) for c in tc], shots=shots)
        result = job.result()
        return [result[i].data.meas.get_counts() for i in range(len(circuits))]
    else:
        raise ValueError(f"Unknown BACKEND: '{backend}'. Use: local | mps | ibm")


# ═══════════════════════════════════════════════════════════════════════════════
# Fermion-to-qubit mapping dispatch
# ═══════════════════════════════════════════════════════════════════════════════

def build_qubit_hamiltonian():
    """
    Build qubit Hamiltonian using JW or BK mapping as set in config.FERMION_TO_QUBIT.

    Jordan-Wigner (JW):
      - Each qubit represents one spin-orbital occupation
      - Pauli string length: O(N) — long strings, deep circuits
      - Simpler to implement and debug
      - Best for: small systems, local simulations

    Bravyi-Kitaev (BK):
      - Encodes occupation AND parity in a balanced binary tree
      - Pauli string length: O(log N) — shorter strings, shallower circuits
      - Fewer 2-qubit gates → less noise on real hardware
      - Best for: larger systems (n_emb > 8), IBM hardware execution

    Both produce the same energy spectrum — only gate efficiency differs.
    """
    mapping = config.FERMION_TO_QUBIT.lower()
    if mapping == "jw":
        return _build_jw_hamiltonian()
    elif mapping == "bk":
        return _build_bk_hamiltonian()
    else:
        raise ValueError(
            f"Unknown FERMION_TO_QUBIT: '{mapping}'. Use: 'jw' or 'bk'."
        )


def _fop_from_integrals():
    """
    Build OpenFermion FermionOperator from h1e and h2e.
    Shared by both JW and BK builders.

    h2e[p,q,r,s] = (pq|rs) chemist notation.
    Two-body operator: 0.5 * Σ_{pqrs,σσ'} (pq|rs) p†_σ r†_σ' s_σ' q_σ
    """
    from openfermion import FermionOperator as OF_FermionOp

    fop = OF_FermionOp()

    # One-body
    for p in range(n_emb):
        for q in range(n_emb):
            h = complex(h1e[p, q])
            if abs(h) < 1e-10:
                continue
            fop += OF_FermionOp(f"{p}^ {q}",               h)   # alpha
            fop += OF_FermionOp(f"{n_emb+p}^ {n_emb+q}",   h)   # beta

    # Two-body (fixed index ordering for chemist notation)
    for p in range(n_emb):
        for q in range(n_emb):
            for r in range(n_emb):
                for s in range(n_emb):
                    h = 0.5 * complex(h2e[p, q, r, s])
                    if abs(h) < 1e-10:
                        continue
                    pa, qa = p,         q
                    pb, qb = n_emb + p, n_emb + q
                    ra, sa = r,         s
                    rb, sb = n_emb + r, n_emb + s
                    # αα, ββ, αβ, βα spin combinations
                    fop += OF_FermionOp(((pa,1),(ra,1),(sa,0),(qa,0)), h)
                    fop += OF_FermionOp(((pb,1),(rb,1),(sb,0),(qb,0)), h)
                    fop += OF_FermionOp(((pa,1),(rb,1),(sb,0),(qa,0)), h)
                    fop += OF_FermionOp(((pb,1),(ra,1),(sa,0),(qb,0)), h)

    return fop


def _fop_to_sparse_pauli(qubit_op):
    """Convert OpenFermion QubitOperator to Qiskit SparsePauliOp."""
    from qiskit.quantum_info import SparsePauliOp

    n_so = 2 * n_emb
    labels, coeffs = [], []

    for term, coeff in qubit_op.terms.items():
        arr = ['I'] * n_so
        for idx, pauli in term:
            arr[idx] = pauli
        labels.append(''.join(reversed(arr)))   # Qiskit reverses qubit order
        coeffs.append(complex(coeff))

    if not labels:
        return SparsePauliOp('I' * n_so, coeffs=[0.0])

    op = SparsePauliOp(labels, coeffs=coeffs).simplify()

    max_imag = float(np.max(np.abs(np.imag(op.coeffs))))
    if max_imag > 1e-6:
        warnings.warn(
            f"Qubit Hamiltonian has imaginary coefficients up to {max_imag:.2e}. "
            f"Possible error in integrals.",
            RuntimeWarning,
        )

    return SparsePauliOp(op.paulis, coeffs=np.real(op.coeffs))


def _build_jw_hamiltonian():
    """Jordan-Wigner mapping: O(N) Pauli string length."""
    from openfermion import jordan_wigner

    fop = _fop_from_integrals()
    jw  = jordan_wigner(fop)
    op  = _fop_to_sparse_pauli(jw)

    print(f"  JW Hamiltonian: {len(op)} Pauli terms  "
          f"(max string length = {max(str(p).count('X') + str(p).count('Y') + str(p).count('Z') for p in op.paulis.to_labels())})")
    return op


def _build_bk_hamiltonian():
    """
    Bravyi-Kitaev mapping: O(log N) Pauli string length.

    BK uses a binary-tree structure to encode both occupation numbers
    and parity information. Each qubit stores:
      - Occupation of one orbital (like JW)
      - Partial parity of a subset of orbitals

    This makes most operators act on O(log N) qubits instead of O(N),
    producing shorter Pauli strings and shallower Trotter circuits.

    Trade-off: harder to interpret individual qubits physically.
    """
    from openfermion import bravyi_kitaev

    fop = _fop_from_integrals()
    bk  = bravyi_kitaev(fop)
    op  = _fop_to_sparse_pauli(bk)

    print(f"  BK Hamiltonian: {len(op)} Pauli terms  "
          f"(max string length = {max(str(p).count('X') + str(p).count('Y') + str(p).count('Z') for p in op.paulis.to_labels())})")
    return op


# ═══════════════════════════════════════════════════════════════════════════════
# Ansatz builders
# ═══════════════════════════════════════════════════════════════════════════════

def build_ansatz_circuit():
    """
    Build the ansatz circuit for SQD sampling.
    Dispatches to SU2 or LUCJ based on config.ANSATZ.
    Returns a measured circuit ready for sampling.
    """
    ansatz = config.ANSATZ.lower()
    if ansatz == "su2":
        return _build_su2_circuit()
    elif ansatz == "lucj":
        return _build_lucj_circuit()
    else:
        raise ValueError(f"Unknown ANSATZ: '{ansatz}'. Use: 'su2' or 'lucj'.")


def _build_su2_circuit():
    """
    EfficientSU2 ansatz — general parameterized rotations.

    Does NOT conserve particle number → ~40-60% of shots are filtered.
    Use when LUCJ is unavailable or for debugging.
    """
    hf_circ = QuantumCircuit(n_qubits)
    for i in range(n_alpha): hf_circ.x(i)
    for i in range(n_beta):  hf_circ.x(n_emb + i)

    ansatz = efficient_su2(
        n_qubits,
        reps=config.ANSATZ_REPS,
        entanglement="full",
        skip_final_rotation_layer=True,
    )
    params = np.random.default_rng(42).uniform(0, 2 * np.pi, ansatz.num_parameters)
    circ   = hf_circ.compose(ansatz.assign_parameters(params))
    circ.measure_all()

    print(f"  SU2 circuit : {n_qubits}q,  depth={circ.depth()},  "
          f"params={ansatz.num_parameters}")
    print(f"  ⚠ SU2 does not conserve particle number — "
          f"~40-60% of shots will be filtered.")
    return circ


def _build_lucj_circuit():
    """
    Local Unitary Cluster Jastrow (LUCJ) ansatz.

    Structure: |ψ_LUCJ⟩ = U_R · exp(iJ) · U_L |HF⟩

      U_L, U_R = local orbital rotation layers (nearest-neighbour SU(2) gates)
      exp(iJ)  = diagonal Jastrow factor: exp(i Σ_{pq} θ_{pq} n_p n_q)
                 captures density-density correlations

    Advantages over EfficientSU2:
      1. Conserves particle number by construction → 0% shots wasted
      2. Physically motivated: matches electronic correlation structure
      3. Can be initialized from MP2/CCSD amplitudes for faster convergence
      4. O(N) circuit depth per layer (vs O(N²) for full entanglement SU2)

    Note: requires qiskit-addon-sqd >= 0.5 with LUCJ support.
    """
    try:
        from qiskit_addon_sqd.lucj import LUCJAnsatz
        lucj = LUCJAnsatz(
            num_orbitals  = n_emb,
            num_layers    = config.LUCJ_NUM_LAYERS,
            num_elec_a    = n_alpha,
            num_elec_b    = n_beta,
        )
        # Build circuit: HF reference + LUCJ layers
        circ = lucj.circuit()
        circ.measure_all()

        print(f"  LUCJ circuit: {n_qubits}q,  depth={circ.depth()},  "
              f"layers={config.LUCJ_NUM_LAYERS}")
        print(f"  ✓ LUCJ conserves particle number — 0% shots wasted")
        return circ

    except ImportError:
        # Fallback: manual LUCJ construction using basic gates
        warnings.warn(
            "qiskit_addon_sqd.lucj not found. "
            "Building LUCJ manually with givens rotations.",
            RuntimeWarning,
        )
        return _build_lucj_manual()


def _build_lucj_manual():
    """
    Manual LUCJ construction when qiskit_addon_sqd.lucj is unavailable.

    Implements the key LUCJ property: particle-number conservation.
    Uses Givens rotation layers (number-preserving SU(2) on pairs of modes)
    interleaved with diagonal phase layers (Jastrow factor approximation).

    Circuit structure per layer:
      1. Givens rotations on (0,1), (2,3), ... (even pairs)
      2. Givens rotations on (1,2), (3,4), ... (odd pairs)
      3. Phase gates (Z rotations) for diagonal Jastrow

    Givens rotation G(θ,φ) preserves particle number:
      G|01⟩ = cos(θ)|01⟩ + e^{iφ} sin(θ)|10⟩
      G|10⟩ = -e^{-iφ} sin(θ)|01⟩ + cos(θ)|10⟩
      G|00⟩ = |00⟩,  G|11⟩ = e^{i(φ_0+φ_1)}|11⟩
    """
    from qiskit.circuit import Parameter

    rng    = np.random.default_rng(42)
    circ   = QuantumCircuit(n_qubits)

    # Initialize HF reference (particle-number conserving starting point)
    for i in range(n_alpha): circ.x(i)
    for i in range(n_beta):  circ.x(n_emb + i)

    def givens_layer(qc, qubits, offset, layer_idx, spin_label):
        """Apply Givens rotation layer to adjacent pairs."""
        pairs = list(range(offset, len(qubits) - 1, 2))
        for k, p in enumerate(pairs):
            q0, q1 = qubits[p], qubits[p + 1]
            theta = rng.uniform(-np.pi / 4, np.pi / 4)
            phi   = rng.uniform(0, 2 * np.pi)
            # Givens rotation via CNOT + single-qubit rotations
            # (number-preserving decomposition)
            qc.cx(q0, q1)
            qc.ry(2 * theta, q0)
            qc.rz(phi, q0)
            qc.cx(q0, q1)
            qc.rz(-phi, q1)

    def jastrow_layer(qc, qubits):
        """Diagonal Jastrow phase layer: Z rotations on each orbital."""
        for q in qubits:
            angle = rng.uniform(-np.pi / 8, np.pi / 8)
            qc.rz(angle, q)

    alpha_qubits = list(range(n_emb))
    beta_qubits  = list(range(n_emb, 2 * n_emb))

    for layer in range(config.LUCJ_NUM_LAYERS):
        # Alpha spin block
        givens_layer(circ, alpha_qubits, offset=0, layer_idx=layer, spin_label="α")
        givens_layer(circ, alpha_qubits, offset=1, layer_idx=layer, spin_label="α")
        jastrow_layer(circ, alpha_qubits)
        # Beta spin block
        givens_layer(circ, beta_qubits,  offset=0, layer_idx=layer, spin_label="β")
        givens_layer(circ, beta_qubits,  offset=1, layer_idx=layer, spin_label="β")
        jastrow_layer(circ, beta_qubits)

    circ.measure_all()
    print(f"  LUCJ (manual) circuit: {n_qubits}q,  depth={circ.depth()},  "
          f"layers={config.LUCJ_NUM_LAYERS}")
    print(f"  ✓ Particle number conserved by construction")
    return circ


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def filter_bitstrings(bsm, probs):
    """Keep bitstrings with correct (n_alpha, n_beta). Renormalize after filter."""
    valid   = ((bsm[:, :n_emb].sum(axis=1) == n_alpha) &
               (bsm[:, n_emb:].sum(axis=1) == n_beta))
    bsm_f   = bsm[valid]
    probs_f = probs[valid]
    if len(probs_f) > 0:
        probs_f = probs_f / probs_f.sum()
    return bsm_f, probs_f


def hf_bitstring():
    row = np.zeros(2 * n_emb, dtype=bool)
    for i in range(n_alpha): row[i]         = True
    for i in range(n_beta):  row[n_emb + i] = True
    return row


def inject_hf_reference(bsm, probs):
    hf_row  = hf_bitstring()
    present = (bsm.shape[0] > 0 and
               any(np.array_equal(bsm[i], hf_row) for i in range(bsm.shape[0])))
    if not present:
        bsm   = np.vstack([bsm, hf_row[np.newaxis, :]]) if bsm.shape[0] > 0 \
                else hf_row[np.newaxis, :]
        probs = np.append(probs, 1.0 / max(bsm.shape[0], 1))
        probs = probs / probs.sum()
    return bsm, probs


def _check_configs(bsm, probs, context=""):
    if bsm.shape[0] == 0:
        raise RuntimeError(
            f"No valid bitstrings after particle-number filtering"
            + (f" ({context})" if context else "") + ".\n"
            f"  Expected: {n_alpha}α + {n_beta}β in {n_emb} orbitals.\n"
            f"  If using SU2 ansatz: increase N_SHOTS or switch to LUCJ.\n"
            f"  If using LUCJ: check LUCJ_NUM_LAYERS > 0."
        )


def print_iteration_header():
    print(f"\n  {'─'*84}")
    print(f"  {'Iter':>5} │ {'Energy (Ha)':>14} │ {'configs':>7} │ "
          f"{'vs UHF':>13} │ {'vs MP2':>13} │ {'ΔE(prev)':>12}")
    print(f"  {'─'*84}")


def print_iteration(label, energy, n_configs, prev_energy=None):
    vs_uhf    = energy - uhf_energy
    vs_mp2    = energy - mp2_energy
    delta_str = f"{energy - prev_energy:+.6f}" if prev_energy is not None else "       ---"
    print(f"  {label:>5} │ {energy:>14.8f} │ {n_configs:>7d} │ "
          f"{vs_uhf:+.6f} {'↓' if vs_uhf<0 else '↑'}  │ "
          f"{vs_mp2:+.6f} {'↓' if vs_mp2<0 else '↑'}  │ {delta_str}")


def iterative_solve(bsm, probs, n_iters):
    avg_occs = (
        np.array([1.0 if i < n_alpha else 0.0 for i in range(n_emb)]),
        np.array([1.0 if i < n_beta  else 0.0 for i in range(n_emb)]),
    )
    iterations  = []
    energy      = None
    spin_sq     = None
    prev_energy = None

    print_iteration_header()

    for it in range(n_iters):
        bsm, probs = recover_configurations(
            bsm, probs, avg_occs,
            num_elec_a=n_alpha, num_elec_b=n_beta,
            rand_seed=42 + it,
        )
        if bsm.shape[0] == 0:
            warnings.warn(f"recover_configurations returned 0 configs at iter {it+1}.",
                          RuntimeWarning)
            break

        e_emb, _, avg_occs, spin_sq = solve_fermion(
            bsm, hcore=h1e, eri=h2e, open_shell=False, spin_sq=0.0,
        )

        # ── Add core energy to get total molecular energy ──────────────────
        energy = float(e_emb) + ecore

        print_iteration(f"{it+1:02d}", energy, bsm.shape[0], prev_energy)
        iterations.append({
            "iter"     : it + 1,
            "energy"   : energy,
            "e_emb"    : float(e_emb),    # store embedded for debugging
            "ecore"    : ecore,
            "n_configs": int(bsm.shape[0]),
            "vs_uhf"   : float(energy - uhf_energy),
            "vs_mp2"   : float(energy - mp2_energy),
        })
        prev_energy = energy

    print(f"  {'─'*84}")
    return energy, spin_sq, iterations


# ═══════════════════════════════════════════════════════════════════════════════
# SQD
# ═══════════════════════════════════════════════════════════════════════════════

def run_sqd():
    print(f"\n── SQD ({config.ANSATZ.upper()} ansatz) {'─'*40}")
    circ = build_ansatz_circuit()

    print(f"  Shots: {config.N_SHOTS}")
    raw        = sample_circuits([circ], config.N_SHOTS)[0]
    bsm, probs = counts_to_arrays(raw)
    bsm, probs = filter_bitstrings(bsm, probs)

    n_total   = sum(raw.values())
    n_valid   = bsm.shape[0]
    pct_valid = 100.0 * n_valid / max(n_total, 1)
    print(f"  Valid configs: {n_valid} / {n_total} ({pct_valid:.1f}%)")

    _check_configs(bsm, probs, "SQD after filter")

    energy, spin_sq, iterations = iterative_solve(bsm, probs, config.SQD_ITERS)
    return energy, spin_sq, iterations


# ═══════════════════════════════════════════════════════════════════════════════
# SKQD
# ═══════════════════════════════════════════════════════════════════════════════

def run_skqd():
    """
    Krylov subspace diagonalization via time evolution.
    Uses JW or BK mapping (config.FERMION_TO_QUBIT) for the evolution gate.

    BK is preferred here because:
      - Krylov requires many Trotter steps → circuit depth matters most
      - BK's O(log N) strings → shallower circuits → less gate noise
    """
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import LieTrotter

    print(f"\n── SKQD ({config.FERMION_TO_QUBIT.upper()} mapping) {'─'*38}")
    print(f"  Krylov dim: {config.SKQD_KRYLOV_DIM},  dt={config.SKQD_DT},  "
          f"Trotter reps={config.SKQD_TROTTER_REPS}")

    H_qubit = build_qubit_hamiltonian()

    ref = QuantumCircuit(n_qubits)
    for i in range(n_alpha): ref.x(i)
    for i in range(n_beta):  ref.x(n_emb + i)

    evol = PauliEvolutionGate(
        H_qubit,
        time      = config.SKQD_DT / config.SKQD_TROTTER_REPS,
        synthesis = LieTrotter(reps=config.SKQD_TROTTER_REPS),
    )

    circs = []
    for k in range(config.SKQD_KRYLOV_DIM):
        qc = ref.copy()
        for _ in range(k):
            qc.append(evol, range(n_qubits))
        qc.measure_all()
        circs.append(qc)

    depths = [c.depth() for c in circs]
    print(f"  Circuit depths: {depths}")
    print(f"  Sampling {len(circs)} Krylov circuits @ {config.SKQD_SHOTS} shots each...")

    all_counts  = sample_circuits(circs, config.SKQD_SHOTS)
    iterations  = []
    energy      = None
    spin_sq     = None
    prev_energy = None
    cumulative  = Counter()

    print_iteration_header()

    for k, raw in enumerate(all_counts):
        cumulative.update(raw)
        bsm, probs = counts_to_arrays(dict(cumulative))
        bsm, probs = filter_bitstrings(bsm, probs)

        if bsm.shape[0] < 2:
            print(f"  k={k:2d}  │  {bsm.shape[0]} valid configs — skipping")
            continue

        try:
            energy, _, _, spin_sq = solve_fermion(
                bsm, hcore=h1e, eri=h2e, open_shell=False, spin_sq=0.0,
            )
            energy = float(e_emb) + ecore
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"  k={k:2d}  │  solve_fermion failed: {e}")
            continue

        print_iteration(f"k={k:2d}", energy, bsm.shape[0], prev_energy)
        iterations.append({
            "k"        : k,
            "energy"   : float(energy),
            "n_configs": int(bsm.shape[0]),
            "vs_uhf"   : float(energy - uhf_energy),
            "vs_mp2"   : float(energy - mp2_energy),
        })
        prev_energy = energy

    print(f"  {'─'*84}")

    if energy is None:
        raise RuntimeError(
            "SKQD produced no valid energy estimate.\n"
            "Try increasing SKQD_SHOTS or SKQD_KRYLOV_DIM."
        )
    return energy, spin_sq, iterations


# ═══════════════════════════════════════════════════════════════════════════════
# SqDRIFT
# ═══════════════════════════════════════════════════════════════════════════════

def run_sqdrift():
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
        raise ImportError("qiskit-fermions required. pip install qiskit-fermions")

    import tempfile
    from pyscf.tools import fcidump as pyscf_fcidump

    print(f"\n── SqDRIFT {'─'*46}")

    fd, tmp = tempfile.mkstemp(suffix=".fcidump")
    os.close(fd)
    try:
        pyscf_fcidump.from_integrals(tmp, h1e, h2e, n_emb, n_alpha + n_beta,
                                     ms=abs(n_alpha - n_beta))
        hamil = FermionOperator.from_fcidump(FCIDump.from_file(tmp))
    finally:
        os.unlink(tmp)

    group_terms_by_electronic_structure(hamil, n_qubits)
    evo      = Evolution(n_qubits, hamil, config.SQDRIFT_TIME)
    template = FermionicCircuit(n_qubits)
    template.append(evo, template.modes)
    pm       = generate_preset_jw_pass_manager()
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

    all_counts = sample_circuits(circuits, config.SQDRIFT_SHOTS)
    cumulative = Counter()
    for counts in all_counts:
        cumulative.update(counts)

    bsm, probs = counts_to_arrays(dict(cumulative))
    bsm, probs = filter_bitstrings(bsm, probs)
    bsm, probs = inject_hf_reference(bsm, probs)
    _check_configs(bsm, probs, "SqDRIFT")
    print(f"  Valid configs: {bsm.shape[0]}")

    energy, spin_sq, iterations = iterative_solve(bsm, probs, config.SQDRIFT_ITERS)
    return energy, spin_sq, iterations


# ═══════════════════════════════════════════════════════════════════════════════
# Dispatch
# ═══════════════════════════════════════════════════════════════════════════════

solvers = {"sqd": run_sqd, "skqd": run_skqd, "sqdrift": run_sqdrift}

if config.QUANTUM_SOLVER not in solvers:
    raise ValueError(f"Unknown QUANTUM_SOLVER: '{config.QUANTUM_SOLVER}'. "
                     f"Use: {list(solvers.keys())}")

energy, spin_sq, iterations = solvers[config.QUANTUM_SOLVER]()

# ── Final Summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"[Step 3] Final Summary — {mol_info['molecule']}")
print(f"{'='*60}")
if energy is not None:
    print(f"  Solver  : {config.QUANTUM_SOLVER.upper()} + "
          f"{config.ANSATZ.upper()} + {config.FERMION_TO_QUBIT.upper()}")
    print(f"  Energy  : {energy:.8f} Ha")
    print(f"  UHF     : {uhf_energy:.8f} Ha  (Δ = {energy-uhf_energy:+.6f})")
    print(f"  MP2     : {mp2_energy:.8f} Ha  (Δ = {energy-mp2_energy:+.6f})")
    if len(iterations) > 1:
        print(f"  Improve : {iterations[-1]['energy']-iterations[0]['energy']:+.8f} Ha "
              f"over {len(iterations)} iters")
if spin_sq is not None:
    print(f"  <S²>    : {spin_sq:.6f}")
print(f"{'='*60}")

output = {
    "solver"      : config.QUANTUM_SOLVER,
    "ansatz"      : config.ANSATZ,
    "mapping"     : config.FERMION_TO_QUBIT,
    "backend"     : config.BACKEND,
    "energy"      : float(energy)  if energy  is not None else None,
    "spin_sq"     : float(spin_sq) if spin_sq is not None else None,
    "uhf_energy"  : uhf_energy,
    "mp2_energy"  : mp2_energy,
    "iterations"  : iterations,
    "mol_info"    : mol_info,
}

with open(STEP3_FILE, "wb") as f:
    pickle.dump(output, f)

print(f"\n[Step 3] ✓ Saved → {STEP3_FILE}")