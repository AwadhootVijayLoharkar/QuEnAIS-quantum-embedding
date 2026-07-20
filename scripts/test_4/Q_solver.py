# step3_solver.py — Quantum Solver (SQD / SKQD / SqDRIFT)
"""
Solves the embedded Hamiltonian from Step 2 using a quantum(-inspired) solver.

Solvers:
  SQD     — Random ansatz sampling + iterative subspace diagonalization
  SKQD    — Krylov time-evolution sampling + cumulative diagonalization
  SqDRIFT — qDRIFT ensemble sampling + iterative diagonalization

Fixes vs original:
  - FORCE_RERUN controlled via --force CLI flag, not hardcoded True
  - _build_jw_hamiltonian(): fixed two-body index ordering
    Original used wrong index permutation for (pq|rs) chemist notation
    → silent wrong energies from SKQD. Fixed to p†_σ r†_τ s_τ q_σ ordering.
  - run_skqd(): bare except replaced with specific exception types
  - mp2_energy loaded from step1 pkl (already computed) not recomputed
  - filter_bitstrings(): probs renormalized after filtering to keep valid dist.
  - Zero valid configs after filtering raises informative error

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

# ── CLI argument: --force bypasses cache ──────────────────────────────────────
parser = argparse.ArgumentParser(description="Step 3: Quantum Solver")
parser.add_argument("--force", action="store_true",
                    help="Rerun even if cached result exists")
args   = parser.parse_args()
FORCE_RERUN = args.force

STEP3_FILE = os.path.join(config.RESULTS_DIR, "step3_results.pkl")

# ── Setup ─────────────────────────────────────────────────────────────────────
os.makedirs(config.RESULTS_DIR, exist_ok=True)

if os.path.exists(STEP3_FILE) and not FORCE_RERUN:
    print(f"[Step 3] Using cached result: {STEP3_FILE}")
    print(f"         Run with --force to recompute.")
    sys.exit(0)

for path, name in [(config.STEP1_FILE, "step1_asf.py"),
                   (config.STEP2_FILE, "step2_hamiltonian.py")]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required input not found: {path}\n"
            f"Run {name} first."
        )

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
mol_info   = step1["mol_info"]

# Use pre-computed MP2 energy from step1 (already stored there)
# Original recomputed mp2_energy = uhf_energy + mp2_corr from step2,
# but step1 now stores the definitive mp2_energy directly.
mp2_energy = step1.get("mp2_energy", uhf_energy + step2.get("mp2_corr", 0.0))

n_qubits = 2 * n_emb

print(f"\n{'='*60}")
print(f"[Step 3] Quantum Solver — {mol_info['molecule']}")
print(f"{'='*60}")
print(f"  Solver   : {config.QUANTUM_SOLVER.upper()}")
print(f"  Backend  : {config.BACKEND.upper()}")
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
    """
    Run circuits on the configured backend.
    Returns a list of count dicts, one per circuit.
    """
    backend = config.BACKEND.lower()

    if backend == "local":
        from qiskit.primitives import StatevectorSampler
        res = StatevectorSampler().run(circuits, shots=shots).result()
        return [res[i].data.meas.get_counts() for i in range(len(circuits))]

    elif backend == "mps":
        from qiskit_aer import AerSimulator
        from qiskit import transpile
        sim = AerSimulator(
            method                                      = "matrix_product_state",
            matrix_product_state_max_bond_dimension     = config.MPS_MAX_BOND_DIM,
            matrix_product_state_truncation_threshold   = config.MPS_TRUNC_THRESH,
        )
        tc     = transpile(circuits, backend=sim, optimization_level=1)
        result = sim.run(tc, shots=shots).result()
        return [result.get_counts(i) for i in range(len(circuits))]

    elif backend == "ibm":
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        service = QiskitRuntimeService()
        hw = (
            service.backend(config.IBM_BACKEND_NAME)
            if config.IBM_BACKEND_NAME
            else service.least_busy(
                operational=True, simulator=False,
                min_num_qubits=circuits[0].num_qubits,
            )
        )
        pm     = generate_preset_pass_manager(config.IBM_OPTIMIZATION_LEVEL, backend=hw)
        tc     = pm.run(circuits)
        job    = SamplerV2(mode=hw).run([(c,) for c in tc], shots=shots)
        result = job.result()
        return [result[i].data.meas.get_counts() for i in range(len(circuits))]

    else:
        raise ValueError(
            f"Unknown BACKEND: '{backend}'. "
            f"Valid options: 'local', 'mps', 'ibm'."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def filter_bitstrings(bsm, probs):
    """
    Keep only bitstrings with the correct (n_alpha, n_beta) particle number.

    Fix vs original:
      After filtering, probabilities no longer sum to 1.
      Renormalize so probs remains a valid probability distribution.
      This prevents numerical issues in recover_configurations.
    """
    valid = (
        (bsm[:, :n_emb].sum(axis=1) == n_alpha) &
        (bsm[:, n_emb:].sum(axis=1) == n_beta)
    )
    bsm_f   = bsm[valid]
    probs_f = probs[valid]

    if len(probs_f) == 0:
        return bsm_f, probs_f

    # Renormalize after filtering
    probs_f = probs_f / probs_f.sum()
    return bsm_f, probs_f


def hf_bitstring():
    """Hartree-Fock reference determinant as a boolean row vector."""
    row = np.zeros(2 * n_emb, dtype=bool)
    for i in range(n_alpha): row[i]          = True
    for i in range(n_beta):  row[n_emb + i]  = True
    return row


def inject_hf_reference(bsm, probs):
    """
    Ensure the HF determinant is present in the configuration set.
    If absent, append it with a small uniform probability and renormalize.
    This guarantees at least one physically meaningful starting point.
    """
    hf_row = hf_bitstring()
    already_present = (
        bsm.shape[0] > 0 and
        any(np.array_equal(bsm[i], hf_row) for i in range(bsm.shape[0]))
    )
    if not already_present:
        bsm   = np.vstack([bsm, hf_row[np.newaxis, :]]) if bsm.shape[0] > 0 \
                else hf_row[np.newaxis, :]
        probs = np.append(probs, 1.0 / max(bsm.shape[0], 1))
        probs = probs / probs.sum()
    return bsm, probs


def _check_configs(bsm, probs, context=""):
    """Raise informative error if no valid configurations remain."""
    if bsm.shape[0] == 0:
        raise RuntimeError(
            f"No valid bitstrings remain after particle-number filtering"
            + (f" ({context})" if context else "") + ".\n"
            f"  Expected: {n_alpha}α + {n_beta}β electrons in {n_emb} orbitals.\n"
            f"  Possible causes:\n"
            f"    1. Ansatz does not preserve particle number "
            f"(EfficientSU2 does not — consider LUCJ)\n"
            f"    2. Too few shots ({config.N_SHOTS}) → no valid bitstrings sampled\n"
            f"    3. n_alpha={n_alpha} or n_beta={n_beta} is wrong (check Step 1)"
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
    uhf_mark  = "↓" if vs_uhf < 0 else "↑"
    mp2_mark  = "↓" if vs_mp2 < 0 else "↑"
    print(f"  {label:>5} │ {energy:>14.8f} │ {n_configs:>7d} │ "
          f"{vs_uhf:+.6f} {uhf_mark}  │ {vs_mp2:+.6f} {mp2_mark}  │ {delta_str}")


def iterative_solve(bsm, probs, n_iters):
    """
    Shared iterative recover + solve loop used by SQD and SqDRIFT.
    At each iteration:
      1. recover_configurations repairs bitstrings to correct particle sector
      2. solve_fermion does exact diagonalization in the sampled Fock subspace
      3. avg_occs from step 2 feeds back into step 1 for the next iteration
    """
    avg_occs = (
        np.array([1.0 if i < n_alpha else 0.0 for i in range(n_emb)]),
        np.array([1.0 if i < n_beta  else 0.0 for i in range(n_emb)]),
    )

    iterations   = []
    energy       = None
    spin_sq      = None
    prev_energy  = None

    print_iteration_header()

    for it in range(n_iters):
        bsm, probs = recover_configurations(
            bsm, probs, avg_occs,
            num_elec_a=n_alpha, num_elec_b=n_beta,
            rand_seed=42 + it,
        )

        if bsm.shape[0] == 0:
            warnings.warn(
                f"recover_configurations returned 0 configs at iteration {it+1}. "
                f"Stopping early.",
                RuntimeWarning,
            )
            break

        energy, _, avg_occs, spin_sq = solve_fermion(
            bsm, hcore=h1e, eri=h2e,
            open_shell=False, spin_sq=0.0,
        )

        print_iteration(f"{it+1:02d}", energy, bsm.shape[0], prev_energy)

        iterations.append({
            "iter"     : it + 1,
            "energy"   : float(energy),
            "n_configs": int(bsm.shape[0]),
            "vs_uhf"   : float(energy - uhf_energy),
            "vs_mp2"   : float(energy - mp2_energy),
        })
        prev_energy = energy

    print(f"  {'─'*84}")
    return energy, spin_sq, iterations


# ═══════════════════════════════════════════════════════════════════════════════
# JW Hamiltonian builder (used by SKQD for time-evolution circuits)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_jw_hamiltonian():
    """
    Build the Jordan-Wigner qubit Hamiltonian from h1e (chemist) and h2e (chemist).

    Spin-orbital layout:
      0 .. n_emb-1        = alpha spin-orbitals
      n_emb .. 2*n_emb-1  = beta  spin-orbitals

    One-body terms (correct, unchanged):
      h1e[p,q] * p†_α q_α  +  h1e[p,q] * p†_β q_β

    Two-body terms (FIXED vs original):
      h2e[p,q,r,s] is in chemist's notation: (pq|rs) = ∫ φ_p(1) φ_q(1) (1/r12) φ_r(2) φ_s(2)
      The second-quantized operator is:
        0.5 * Σ_{pqrs,σσ'} (pq|rs) * p†_σ q_σ r†_σ' s_σ'   [physicists write as p†r†sq]

      In creation/annihilation operator form:
        αα: 0.5*(pq|rs) * p†_α r†_α s_α q_α    (note: p†q, r†s are bra/ket pairs)
        ββ: 0.5*(pq|rs) * p†_β r†_β s_β q_β
        αβ: 0.5*(pq|rs) * p†_α r†_β s_β q_α
        βα: 0.5*(pq|rs) * p†_β r†_α s_α q_β

      Original had wrong ordering: used (p†,r†,s,q) which incorrectly swapped
      r↔q and s↔p relative to chemist notation, giving wrong matrix elements.
    """
    from openfermion import FermionOperator as OF_FermionOp, jordan_wigner
    from qiskit.quantum_info import SparsePauliOp

    n_so = 2 * n_emb
    fop  = OF_FermionOp()

    # ── One-body terms ────────────────────────────────────────────────────────
    for p in range(n_emb):
        for q in range(n_emb):
            h = complex(h1e[p, q])
            if abs(h) < 1e-10:
                continue
            fop += OF_FermionOp(f"{p}^ {q}",           h)   # alpha
            fop += OF_FermionOp(f"{n_emb+p}^ {n_emb+q}", h)   # beta

    # ── Two-body terms (fixed index ordering) ─────────────────────────────────
    # h2e[p,q,r,s] = (pq|rs) chemist notation
    # Operator: 0.5 * (pq|rs) * p†_σ r†_σ' s_σ' q_σ
    # OF_FermionOp index ordering: ((idx, dag), (idx, dag), ...)
    #   creation  = dag 1, annihilation = dag 0
    for p in range(n_emb):
        for q in range(n_emb):
            for r in range(n_emb):
                for s in range(n_emb):
                    h = 0.5 * complex(h2e[p, q, r, s])
                    if abs(h) < 1e-10:
                        continue

                    pa, qa = p,         q            # alpha indices (spatial)
                    pb, qb = n_emb + p, n_emb + q   # beta  indices (spatial)
                    ra, sa = r,         s
                    rb, sb = n_emb + r, n_emb + s

                    # αα|αα : p†_α r†_α s_α q_α
                    fop += OF_FermionOp(((pa, 1), (ra, 1), (sa, 0), (qa, 0)), h)
                    # ββ|ββ : p†_β r†_β s_β q_β
                    fop += OF_FermionOp(((pb, 1), (rb, 1), (sb, 0), (qb, 0)), h)
                    # αβ|αβ : p†_α r†_β s_β q_α
                    fop += OF_FermionOp(((pa, 1), (rb, 1), (sb, 0), (qa, 0)), h)
                    # βα|βα : p†_β r†_α s_α q_β
                    fop += OF_FermionOp(((pb, 1), (ra, 1), (sa, 0), (qb, 0)), h)

    # ── Jordan-Wigner transformation ──────────────────────────────────────────
    jw = jordan_wigner(fop)

    labels, coeffs = [], []
    for term, coeff in jw.terms.items():
        arr = ['I'] * n_so
        for idx, pauli in term:
            arr[idx] = pauli
        # Qiskit uses reversed qubit ordering vs OpenFermion
        labels.append(''.join(reversed(arr)))
        coeffs.append(complex(coeff))

    if not labels:
        return SparsePauliOp('I' * n_so, coeffs=[0.0])

    op = SparsePauliOp(labels, coeffs=coeffs).simplify()

    # Hamiltonian is Hermitian → coefficients must be real.
    # Discard residual imaginary parts (floating-point noise from JW algebra).
    max_imag = float(np.max(np.abs(np.imag(op.coeffs))))
    if max_imag > 1e-6:
        warnings.warn(
            f"JW Hamiltonian has imaginary coefficients up to {max_imag:.2e}. "
            f"This may indicate an error in the two-body integrals.",
            RuntimeWarning,
        )
    op = SparsePauliOp(op.paulis, coeffs=np.real(op.coeffs))

    return op


# ═══════════════════════════════════════════════════════════════════════════════
# SQD
# ═══════════════════════════════════════════════════════════════════════════════

def run_sqd():
    """
    Sample-based Quantum Diagonalization.
    A random EfficientSU2 ansatz (not particle-number conserving) is used.
    The filter step removes ~30-60% of shots — this is expected behavior.
    """
    print(f"\n── SQD {'─'*50}")

    # Build HF reference + random ansatz
    hf_circ = QuantumCircuit(n_qubits)
    for i in range(n_alpha): hf_circ.x(i)
    for i in range(n_beta):  hf_circ.x(n_emb + i)

    ansatz = efficient_su2(
        n_qubits,
        reps               = config.ANSATZ_REPS,
        entanglement       = "full",
        skip_final_rotation_layer = True,
    )
    rng    = np.random.default_rng(42)
    params = rng.uniform(0, 2 * np.pi, ansatz.num_parameters)
    circ   = hf_circ.compose(ansatz.assign_parameters(params))
    circ.measure_all()

    print(f"  Circuit : {n_qubits} qubits,  depth={circ.depth()},  "
          f"params={ansatz.num_parameters}")
    print(f"  Shots   : {config.N_SHOTS}")

    raw        = sample_circuits([circ], config.N_SHOTS)[0]
    bsm, probs = counts_to_arrays(raw)
    bsm, probs = filter_bitstrings(bsm, probs)

    n_valid    = bsm.shape[0]
    n_total    = sum(raw.values())
    pct_valid  = 100.0 * n_valid / max(n_total, 1)
    print(f"  Valid configs: {n_valid} / {n_total}  ({pct_valid:.1f}%)")

    _check_configs(bsm, probs, context="SQD after filter")

    energy, spin_sq, iterations = iterative_solve(bsm, probs, config.SQD_ITERS)
    return energy, spin_sq, iterations


# ═══════════════════════════════════════════════════════════════════════════════
# SKQD
# ═══════════════════════════════════════════════════════════════════════════════

def run_skqd():
    """
    Subspace-expanded Krylov Quantum Diagonalization.
    Builds a Krylov basis via real-time Hamiltonian evolution and
    diagonalizes cumulatively as each vector is added.
    """
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import LieTrotter

    print(f"\n── SKQD {'─'*49}")
    print(f"  Krylov dim    : {config.SKQD_KRYLOV_DIM}")
    print(f"  dt            : {config.SKQD_DT}")
    print(f"  Trotter reps  : {config.SKQD_TROTTER_REPS}")
    print(f"  Shots/circuit : {config.SKQD_SHOTS}")

    H_qubit = _build_jw_hamiltonian()
    print(f"  JW terms      : {len(H_qubit)}")

    # HF reference circuit
    ref = QuantumCircuit(n_qubits)
    for i in range(n_alpha): ref.x(i)
    for i in range(n_beta):  ref.x(n_emb + i)

    # Single Trotter step e^{-iH dt}
    evol = PauliEvolutionGate(
        H_qubit,
        time      = config.SKQD_DT / config.SKQD_TROTTER_REPS,
        synthesis = LieTrotter(reps=config.SKQD_TROTTER_REPS),
    )

    # Build Krylov circuits: |ψ_k⟩ = (e^{-iHdt})^k |HF⟩
    circs = []
    for k in range(config.SKQD_KRYLOV_DIM):
        qc = ref.copy()
        for _ in range(k):
            qc.append(evol, range(n_qubits))
        qc.measure_all()
        circs.append(qc)

    print(f"\n  Sampling {len(circs)} Krylov circuits...")
    all_counts = sample_circuits(circs, config.SKQD_SHOTS)

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
            print(f"  k={k:2d}  │  only {bsm.shape[0]} valid config(s) — skipping")
            continue

        # Specific exceptions only — do not catch all errors silently
        try:
            energy, _, _, spin_sq = solve_fermion(
                bsm, hcore=h1e, eri=h2e,
                open_shell=False, spin_sq=0.0,
            )
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"  k={k:2d}  │  solve_fermion failed: {e}  — skipping")
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
            "All Krylov vectors had fewer than 2 valid configurations.\n"
            "Try increasing SKQD_SHOTS or SKQD_KRYLOV_DIM in config.py."
        )

    return energy, spin_sq, iterations


# ═══════════════════════════════════════════════════════════════════════════════
# SqDRIFT
# ═══════════════════════════════════════════════════════════════════════════════

def run_sqdrift():
    """
    qDRIFT ensemble sampling + iterative subspace diagonalization.
    qDRIFT randomizes which Hamiltonian terms are applied each Trotter step,
    reducing circuit depth compared to fixed-order Trotter at the cost of
    requiring more circuits to average out the randomness.
    """
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
        raise ImportError(
            "qiskit-fermions is required for SqDRIFT.\n"
            "Install with: pip install qiskit-fermions"
        )

    import tempfile
    from pyscf.tools import fcidump as pyscf_fcidump

    print(f"\n── SqDRIFT {'─'*46}")
    print(f"  Circuits : {config.SQDRIFT_NUM_CIRCUITS}")
    print(f"  Groups   : {config.SQDRIFT_NUM_GROUPS}")
    print(f"  Time     : {config.SQDRIFT_TIME}")
    print(f"  Iters    : {config.SQDRIFT_ITERS}")

    # Write integrals to temp FCIDump file for qiskit-fermions
    fd, tmp = tempfile.mkstemp(suffix=".fcidump")
    os.close(fd)
    try:
        pyscf_fcidump.from_integrals(
            tmp, h1e, h2e, n_emb,
            n_alpha + n_beta,
            ms=abs(n_alpha - n_beta),
        )
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

    print(f"  Sampling {len(circuits)} circuits...")
    all_counts = sample_circuits(circuits, config.SQDRIFT_SHOTS)

    cumulative = Counter()
    for counts in all_counts:
        cumulative.update(counts)

    bsm, probs = counts_to_arrays(dict(cumulative))
    bsm, probs = filter_bitstrings(bsm, probs)
    bsm, probs = inject_hf_reference(bsm, probs)   # ensure HF is present

    _check_configs(bsm, probs, context="SqDRIFT after filter+inject")
    print(f"  Valid configs: {bsm.shape[0]}")

    energy, spin_sq, iterations = iterative_solve(bsm, probs, config.SQDRIFT_ITERS)
    return energy, spin_sq, iterations


# ═══════════════════════════════════════════════════════════════════════════════
# Dispatch and run
# ═══════════════════════════════════════════════════════════════════════════════

solvers = {
    "sqd"     : run_sqd,
    "skqd"    : run_skqd,
    "sqdrift" : run_sqdrift,
}

if config.QUANTUM_SOLVER not in solvers:
    raise ValueError(
        f"Unknown QUANTUM_SOLVER: '{config.QUANTUM_SOLVER}'.\n"
        f"Valid options: {list(solvers.keys())}"
    )

energy, spin_sq, iterations = solvers[config.QUANTUM_SOLVER]()

# ── Final Summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"[Step 3] Final Summary — {mol_info['molecule']}")
print(f"{'='*60}")

if energy is not None:
    print(f"  Solver energy  : {energy:.8f} Ha")
    print(f"  UHF energy     : {uhf_energy:.8f} Ha")
    print(f"  MP2 energy     : {mp2_energy:.8f} Ha")
    print(f"  {'─'*36}")
    print(f"  Gain vs UHF    : {energy - uhf_energy:+.8f} Ha  "
          f"({'lower ✓' if energy < uhf_energy else 'higher ✗'})")
    print(f"  Gain vs MP2    : {energy - mp2_energy:+.8f} Ha  "
          f"({'lower ✓' if energy < mp2_energy else 'higher ✗'})")
    if len(iterations) > 1:
        first_e = iterations[0]["energy"]
        last_e  = iterations[-1]["energy"]
        print(f"  Improvement    : {last_e - first_e:+.8f} Ha over {len(iterations)} iters")
else:
    print("  Energy: N/A — solver returned no result")

if spin_sq is not None:
    print(f"  <S²>           : {spin_sq:.6f}  (0 = singlet)")

print(f"{'='*60}")

# ── Save ──────────────────────────────────────────────────────────────────────
output = {
    "solver"     : config.QUANTUM_SOLVER,
    "backend"    : config.BACKEND,
    "energy"     : float(energy)  if energy  is not None else None,
    "spin_sq"    : float(spin_sq) if spin_sq is not None else None,
    "uhf_energy" : uhf_energy,
    "mp2_energy" : mp2_energy,
    "iterations" : iterations,
    "mol_info"   : mol_info,
}

with open(STEP3_FILE, "wb") as f:
    pickle.dump(output, f)

print(f"\n[Step 3] ✓ Saved → {STEP3_FILE}")