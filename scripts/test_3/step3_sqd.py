"""
Step 3 — Quantum Solver: SQD and SKQD
======================================

Two solvers are implemented and selectable via config.QUANTUM_SOLVER:

SQD  (Sampling-based Quantum Diagonalization)
─────────────────────────────────────────────
  Original approach. Uses one random EfficientSU2 circuit to generate
  diverse electron configurations (bitstrings). Iterates:
    sample → filter → recover_configurations → solve_fermion → repeat.
  
  Advantage: no Hamiltonian-to-qubit conversion needed.
  Limitation: random circuit may miss important configurations.

SKQD  (Sampling-based Krylov Quantum Diagonalization)
──────────────────────────────────────────────────────
  Builds a Krylov basis by time-evolving the HF reference state:
    |ψ_0⟩ = |HF⟩
    |ψ_1⟩ = e^{-iHt}       |HF⟩
    |ψ_2⟩ = e^{-2iHt}      |HF⟩
    ...
    |ψ_K⟩ = e^{-KiHt}      |HF⟩
  
  Sample bitstrings from each Krylov vector.
  Accumulate all bitstrings as K grows.
  Solve fermion Hamiltonian in the cumulative subspace.
  
  Advantage: systematic convergence to ground state.
             Each Krylov vector has higher ground-state overlap.
  Requirement: needs JW qubit Hamiltonian for time evolution circuits.
               pip install openfermion

Future solvers (SQDrift, QITE, VQE, ...):
  Add a new function run_X() and register it in SOLVER_REGISTRY.
  The dispatch machinery handles everything else.

Requires : results/step1_asf.pkl
           results/step2_hamiltonian.pkl
           openfermion (SKQD only)
Saves    : results/step3_results.pkl
Runtime  : SQD ~5-20 min | SKQD ~10-30 min
"""

import os
import sys
import math
import pickle
import numpy as np
from collections import Counter

import config

# ── Cache / dependency checks ─────────────────────────────────────────────────
FORCE_RERUN   = True
STEP3_FILE    = os.path.join(config.RESULTS_DIR, "step3_results.pkl")

os.makedirs(config.RESULTS_DIR, exist_ok=True)

if os.path.exists(STEP3_FILE) and not FORCE_RERUN:
    print(f"[Step 3] Cached results at {STEP3_FILE}")
    print("  Set FORCE_RERUN = True to rerun.")
    sys.exit(0)

for fpath, label in [(config.STEP1_FILE, "Step 1 (ASF)"),
                     (config.STEP2_FILE, "Step 2 (Hamiltonian)")]:
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"[Step 3] {label} not found: {fpath}\n"
            "Run the corresponding script first."
        )

# ── Load Step 1 + Step 2 ──────────────────────────────────────────────────────
with open(config.STEP1_FILE, "rb") as f:
    step1 = pickle.load(f)
with open(config.STEP2_FILE, "rb") as f:
    step2 = pickle.load(f)

# Step 1 outputs
nel              = step1["nel"]
mo_list          = step1["mo_list"]
n_active_orbs    = step1["n_active_orbs"]
most_active_atom = step1.get("most_active_atom", 0)
scores_s1        = step1["scores"]
mol_info         = step1["mol_info"]

# Step 2 outputs
h1e       = step2["h1e"]
h2e       = step2["h2e"]
n_emb     = step2["n_emb"]
n_imp     = step2["n_imp"]
n_bath    = step2["n_bath"]
n_alpha   = step2["n_alpha"]
n_beta    = step2["n_beta"]
fci_ref_e = step2["fci_ref_e"]
scores_s2 = step2["scores"]

n_qubits = 2 * n_emb
max_cfg  = math.comb(n_emb, n_alpha) ** 2

print(f"\n{'='*65}")
print(f"[Step 3] Quantum Solver — {mol_info['molecule']}")
print(f"{'='*65}")
print(f"\n  Solver selected      : {config.QUANTUM_SOLVER.upper()}")
print(f"  Impurity             : {n_imp} orbitals  {mo_list}")
print(f"  Bath                 : {n_bath} orbitals")
print(f"  Embedding            : {n_emb} orbitals = {n_qubits} qubits")
print(f"  Electrons            : {n_alpha + n_beta} ({n_alpha}α + {n_beta}β)")
print(f"  Max configurations   : C({n_emb},{n_alpha})² = {max_cfg:,}")

if fci_ref_e is not None:
    print(f"  FCI reference        : {fci_ref_e:.8f} Ha  ← SQD/SKQD target")
else:
    print(f"  FCI reference        : not computed (too large)")

if n_qubits > 24:
    print(f"\n  ⚠ WARNING: {n_qubits} qubits may be slow with StatevectorSampler.")
    print(f"    Consider: reduce MAX_EMBED_ORBS in config.py (current = {config.MAX_EMBED_ORBS})")

# ── Imports ───────────────────────────────────────────────────────────────────
from pyscf import fci as pyscf_fci
from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.fermion import solve_fermion
from qiskit_addon_sqd.configuration_recovery import recover_configurations


# ═════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═════════════════════════════════════════════════════════════════════════════

def filter_bitstrings(bsm, probs, n_alpha, n_beta, n_orb):
    """
    Keep only bitstrings that have EXACTLY n_alpha alpha electrons
    and n_beta beta electrons.

    Column layout (qiskit_addon_sqd convention):
      bsm[:, :n_orb]   = alpha spin-orbitals
      bsm[:, n_orb:]   = beta  spin-orbitals

    This enforces the conservation of particle number, which is
    violated by the random/Trotter circuits that explore the full
    Hilbert space.
    """
    valid = (
        (bsm[:, :n_orb].sum(axis=1) == n_alpha) &
        (bsm[:, n_orb:].sum(axis=1) == n_beta)
    )
    return bsm[valid], probs[valid]


# ═════════════════════════════════════════════════════════════════════════════
# Backend helpers
# ═════════════════════════════════════════════════════════════════════════════

def _run_local(circuits: list, shots: int) -> list:
    """
    Exact statevector simulation (no noise, no approximation).

    When to use:
      Development, unit tests, circuits with ≤ ~20 qubits.
      Results are deterministic up to shot sampling.

    Limitation:
      Memory grows as 2^n_qubits → impractical above ~28 qubits.
    """
    from qiskit.primitives import StatevectorSampler
    res = StatevectorSampler().run(circuits, shots=shots).result()
    return [res[i].data.meas.get_counts() for i in range(len(circuits))]


def _run_mps(circuits: list, shots: int, cfg) -> list:
    """
    Matrix Product State (tensor network) simulation via Qiskit Aer.

    MPS represents the state as a 1-D chain of tensors connected by bonds
    of dimension χ (MPS_MAX_BOND_DIM):
      χ → ∞      exact (same cost as statevector)
      χ = 256    accurate when entanglement growth is moderate

    When to use:
      Circuits too large for statevector (20-50 qubits) but either:
        (a) circuit is shallow / weakly entangled (EfficientSU2 ansatz), or
        (b) you want a cheap feasibility check before IBM submission.

    Limitation:
      For deeply entangled circuits (large Trotter depth, many CNOT layers)
      the required χ grows exponentially → MPS ≈ statevector cost.
      Monitor the printed depth after transpilation as a proxy.

    Requires:  pip install qiskit-aer
    """
    try:
        from qiskit_aer import AerSimulator
    except ImportError:
        raise ImportError(
            "qiskit-aer is required for MPS simulation.\n"
            "Install: pip install qiskit-aer\n"
            "Or switch: set config.BACKEND = 'local'"
        )
    from qiskit import transpile

    max_bond = getattr(cfg, "MPS_MAX_BOND_DIM", 256)
    trunc    = getattr(cfg, "MPS_TRUNC_THRESH", 1e-6)

    print(f"  [MPS] Bond dimension = {max_bond} | trunc threshold = {trunc:.0e}")

    sim = AerSimulator(
        method                                       = "matrix_product_state",
        matrix_product_state_max_bond_dimension      = max_bond,
        matrix_product_state_truncation_threshold    = trunc,
    )

    # Decompose high-level gates (PauliEvolutionGate, etc.) into native gates
    transpiled = transpile(circuits, backend=sim, optimization_level=1)
    depths = [c.depth() for c in (transpiled if isinstance(transpiled, list)
                                  else [transpiled])]
    print(f"  [MPS] Circuit depth(s) after transpile: "
          f"min={min(depths)} max={max(depths)}")

    job    = sim.run(transpiled, shots=shots)
    result = job.result()

    return [result.get_counts(i) for i in range(len(circuits))]


def _run_ibm(circuits: list, shots: int, cfg) -> list:
    """
    Execute on IBM Quantum hardware.

    Open plan (free tier) limitations vs paid plans:
    ┌─────────────────┬──────────────┬──────────────┐
    │ Feature         │ Open (free)  │ Pay-as-go    │
    ├─────────────────┼──────────────┼──────────────┤
    │ Session mode    │ ✗ blocked    │ ✓            │
    │ Batch mode      │ ✗ blocked    │ ✓            │
    │ Direct job mode │ ✓            │ ✓            │
    │ Free minutes/mo │ 10 min       │ paid/min     │
    └─────────────────┴──────────────┴──────────────┘

    This function uses direct job mode only → works on all plan types.

    Circuit depth warning
    ─────────────────────
    Real hardware has a decoherence budget. Beyond ~3000 gate layers
    the circuit output is dominated by noise, not physics.
    SKQD circuits with many Trotter reps can exceed 1,000,000 layers.

    Before using BACKEND="ibm" reduce in config.py:
      SKQD_TROTTER_REPS = 1       (was 2)   halves depth
      SKQD_KRYLOV_DIM   = 4       (was 10)  fewer deep circuits
      IBM_MAX_CIRCUIT_DEPTH = 3000          hard rejection limit

    Requires: qiskit-ibm-runtime with saved credentials
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as IBMSampler
    except ImportError:
        raise ImportError(
            "qiskit-ibm-runtime required.\n"
            "Install: pip install qiskit-ibm-runtime"
        )
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    # ── Connect ───────────────────────────────────────────────────────────────
    service   = QiskitRuntimeService()
    name      = getattr(cfg, "IBM_BACKEND_NAME", None)
    n_q       = circuits[0].num_qubits
    opt_lvl   = getattr(cfg, "IBM_OPTIMIZATION_LEVEL", 1)
    max_depth = getattr(cfg, "IBM_MAX_CIRCUIT_DEPTH", 3000)

    backend = (
        service.backend(name) if name
        else service.least_busy(
            operational=True, simulator=False, min_num_qubits=n_q
        )
    )

    print(f"  [IBM] Backend            : {backend.name}")
    print(f"  [IBM] Backend qubits     : {backend.num_qubits}")
    print(f"  [IBM] Optimization level : {opt_lvl}")
    print(f"  [IBM] Max depth limit    : {max_depth}")

    # ── Transpile ─────────────────────────────────────────────────────────────
    print(f"  [IBM] Transpiling {len(circuits)} circuit(s)...")
    pm         = generate_preset_pass_manager(opt_lvl, backend=backend)
    transpiled = pm.run(circuits)

    if not isinstance(transpiled, list):
        transpiled = [transpiled]

    depths = [c.depth() for c in transpiled]
    print(f"  [IBM] Circuit depths after transpile: "
          f"min={min(depths)}  max={max(depths)}")

    # ── Depth guard ───────────────────────────────────────────────────────────
    # Reject before spending queue time on circuits that will return noise
    if max(depths) > max_depth:
        raise RuntimeError(
            f"\n  Circuit depth {max(depths):,} exceeds IBM_MAX_CIRCUIT_DEPTH "
            f"= {max_depth}.\n"
            f"  Circuits this deep produce noise, not physics, on real hardware.\n"
            f"\n"
            f"  Reduce depth by changing config.py:\n"
            f"    SKQD_TROTTER_REPS = 1     "
            f"# was {cfg.SKQD_TROTTER_REPS}, halves depth per Krylov step\n"
            f"    SKQD_KRYLOV_DIM   = 4     "
            f"# was {cfg.SKQD_KRYLOV_DIM}, fewer deep circuits\n"
            f"\n"
            f"  Or test with lighter backends first:\n"
            f"    BACKEND = 'local'          # exact, free, instant\n"
            f"    BACKEND = 'mps'            # handles larger circuits\n"
            f"\n"
            f"  To suppress this check (not recommended):\n"
            f"    IBM_MAX_CIRCUIT_DEPTH = {max(depths) + 1}"
        )

    # ── Submit (direct job mode — works on open/free plan) ───────────────────
    # Session and Batch modes are blocked on the open plan.
    # SamplerV2(mode=backend) submits a single job directly, no Session needed.
    print(f"  [IBM] Submitting all {len(transpiled)} circuit(s) as one job "
          f"(shots = {shots:,})...")
    print(f"  [IBM] Note: open plan = direct job mode (no Session/Batch)")

    sampler = IBMSampler(mode=backend)
    job     = sampler.run([(c,) for c in transpiled], shots=shots)

    print(f"  [IBM] Job ID : {job.job_id()}")
    print(f"  [IBM] Track at https://quantum.ibm.com")
    print(f"  [IBM] Waiting for results...")

    result = job.result()
    return [result[i].data.meas.get_counts() for i in range(len(transpiled))]

def sample_circuits(circuits: list, shots: int, cfg) -> list:
    """
    Dispatch circuit execution to the backend set in config.BACKEND.

    Returns: list of count dicts, one per circuit  →  {bitstring: count}

    Backend comparison:

    | Setting | Cost      | Noise | Max qubits | Best for            |
    |---------|-----------|-------|------------|---------------------|
    | local   | low       | none  | ~28        | testing, debugging  |
    | mps     | medium    | none  | ~50+       | mid-size circuits   |
    | ibm     | queue+$   | yes   | 127        | production runs     |
    """
    backend_type = getattr(cfg, "BACKEND", "local").lower()

    if backend_type == "local":
        return _run_local(circuits, shots)
    elif backend_type == "mps":
        return _run_mps(circuits, shots, cfg)
    elif backend_type == "ibm":
        return _run_ibm(circuits, shots, cfg)
    else:
        raise ValueError(
            f"Unknown BACKEND '{backend_type}'.\n"
            f"Valid options: 'local', 'mps', 'ibm'\n"
            f"Set config.BACKEND accordingly."
        )


def build_jw_hamiltonian(h1e, h2e, n_orb):
    """
    Build the Jordan-Wigner qubit Hamiltonian from h1e, h2e.

    The fermionic Hamiltonian in spin-orbital basis is:
      H = Σ_{pq,σ} h1e[p,q] a†_{pσ} a_{qσ}
        + 0.5 Σ_{pqrs,σσ'} h2e[p,q,r,s] a†_{pσ} a†_{rσ'} a_{sσ'} a_{qσ}

    Spin-orbital ordering:
      0, 1, ..., n_orb-1        → alpha orbitals
      n_orb, ..., 2*n_orb-1    → beta orbitals

    This matches qiskit_addon_sqd's bitstring convention exactly.
    h2e[p,q,r,s] is in PySCF chemist notation (pq|rs).

    Requires: openfermion (pip install openfermion)

    Returns
    -------
    SparsePauliOp with 2*n_orb qubits in Qiskit ordering.
    """
    try:
        from openfermion import FermionOperator, jordan_wigner
    except ImportError:
        raise ImportError(
            "openfermion is required for SKQD.\n"
            "Install: pip install openfermion\n"
            "Or switch to SQD: set config.QUANTUM_SOLVER = 'sqd'"
        )
    from qiskit.quantum_info import SparsePauliOp

    n_so     = 2 * n_orb
    ferm_op  = FermionOperator()

    # ── 1-body: Σ_{pq,σ} h1e[p,q] a†_{pσ} a_{qσ} ────────────────────────────
    for p in range(n_orb):
        for q in range(n_orb):
            h = complex(h1e[p, q])
            if abs(h) < 1e-10:
                continue
            # Alpha (spin-orbital p → p, q → q)
            ferm_op += FermionOperator(f'{p}^ {q}', h)
            # Beta  (spin-orbital p → n_orb+p, q → n_orb+q)
            ferm_op += FermionOperator(f'{n_orb+p}^ {n_orb+q}', h)

    # ── 2-body: 0.5 Σ h2e[p,q,r,s] × (αααα + ββββ + αββα + βααβ) ───────────
    # PySCF convention: H_2 = 0.5 Σ h2e[p,q,r,s] a†_p a†_r a_s a_q
    # Spin-resolved: separate αα, ββ, and mixed αβ/βα contributions.
    for p in range(n_orb):
        for q in range(n_orb):
            for r in range(n_orb):
                for s in range(n_orb):
                    h = 0.5 * complex(h2e[p, q, r, s])
                    if abs(h) < 1e-10:
                        continue

                    NP = p + n_orb   # beta orbital p
                    NQ = q + n_orb   # beta orbital q
                    NR = r + n_orb   # beta orbital r
                    NS = s + n_orb   # beta orbital s

                    # αααα: a†_{pα} a†_{rα} a_{sα} a_{qα}
                    ferm_op += FermionOperator(((p, 1),(r, 1),(s, 0),(q, 0)), h)
                    # ββββ: a†_{pβ} a†_{rβ} a_{sβ} a_{qβ}
                    ferm_op += FermionOperator(((NP,1),(NR,1),(NS,0),(NQ,0)), h)
                    # αββα: a†_{pα} a†_{rβ} a_{sβ} a_{qα}
                    ferm_op += FermionOperator(((p, 1),(NR,1),(NS,0),(q, 0)), h)
                    # βααβ: a†_{pβ} a†_{rα} a_{sα} a_{qβ}
                    ferm_op += FermionOperator(((NP,1),(r, 1),(s, 0),(NQ,0)), h)

    # ── Jordan-Wigner transform ───────────────────────────────────────────────
    jw_op = jordan_wigner(ferm_op)

    # ── Convert to Qiskit SparsePauliOp ──────────────────────────────────────
    # openfermion: qubit 0 is leftmost in string
    # Qiskit: qubit 0 is rightmost in string → reverse
    labels, coeffs = [], []
    for term, coeff in jw_op.terms.items():
        arr = ['I'] * n_so
        for qubit_idx, pauli_char in term:
            arr[qubit_idx] = pauli_char
        labels.append(''.join(reversed(arr)))   # Qiskit ordering
        coeffs.append(complex(coeff))

    if not labels:
        return SparsePauliOp('I' * n_so, coeffs=[0.0])

    return SparsePauliOp(labels, coeffs=coeffs).simplify()


# ═════════════════════════════════════════════════════════════════════════════
# Solver 1: SQD
# ═════════════════════════════════════════════════════════════════════════════

def run_sqd(h1e, h2e, n_emb, n_alpha, n_beta, fci_ref_e, cfg):
    """
    Sampling-based Quantum Diagonalization.

    Pipeline:
      1. Build HF reference + EfficientSU2 ansatz with RANDOM parameters
         (random params → broad Hilbert space coverage, not optimization)
      2. Sample cfg.N_SHOTS bitstrings
      3. Filter to (n_alpha, n_beta) particle-number sector
      4. Iterative loop:
           a. recover_configurations: randomly flip some bits to find
              configurations the circuit missed → expand subspace
           b. solve_fermion: diagonalize H in subspace of known configs
           c. update avg_occs: occupation numbers from new ground state
              → guide next recover_configurations call
      5. Track energy convergence per iteration

    Why random parameters?
      We want DIVERSITY in bitstrings, not the optimal state.
      A random ansatz samples broadly across the Hilbert space.
      solve_fermion finds the ground state within the sampled subspace.
    """
    print("\n── SQD Solver ──────────────────────────────────────────────")

    # ── Build circuit ──────────────────────────────────────────────────────────
    # Start from HF state: fill lowest alpha and beta spin-orbitals
    hf_circ = QuantumCircuit(2 * n_emb)
    for i in range(n_alpha): hf_circ.x(i)           # alpha orbitals
    for i in range(n_beta):  hf_circ.x(n_emb + i)   # beta orbitals

    ansatz = efficient_su2(
        2 * n_emb,
        reps                      = cfg.ANSATZ_REPS,
        entanglement              = "full",
        skip_final_rotation_layer = True,
    )
    rng    = np.random.default_rng(42)
    params = rng.uniform(0, 2 * np.pi, ansatz.num_parameters)
    circ   = hf_circ.compose(ansatz.assign_parameters(params))
    circ.measure_all()

    print(f"  Circuit: {2*n_emb} qubits | depth = {circ.depth()} | "
          f"{ansatz.num_parameters} parameters")

    # ── Sample ────────────────────────────────────────────────────────────────
    print(f"  Sampling {cfg.N_SHOTS:,} shots...")
    print(f"  Backend: {getattr(cfg, 'BACKEND', 'local').upper()}")
    raw_counts = sample_circuits([circ], cfg.N_SHOTS, cfg)[0]
    bsm, probs = counts_to_arrays(raw_counts)
    bsm, probs = filter_bitstrings(bsm, probs, n_alpha, n_beta, n_emb)

    print(f"  Valid bitstrings : {bsm.shape[0]:,} / {max_cfg:,} max")

    if bsm.shape[0] == 0:
        raise RuntimeError(
            "No valid bitstrings after filtering.\n"
            "Try increasing N_SHOTS in config.py"
        )

    # ── SQD iterative loop ────────────────────────────────────────────────────
    # avg_occs: initial guess = aufbau filling
    avg_occs = (
        np.array([1.0 if i < n_alpha else 0.0 for i in range(n_emb)]),
        np.array([1.0 if i < n_beta  else 0.0 for i in range(n_emb)]),
    )

    iterations  = []
    sqd_energy  = None
    spin_sq_val = None

    target_str = f"{fci_ref_e:.8f}" if fci_ref_e is not None else "N/A"
    print(f"\n  Iterating (FCI target = {target_str} Ha):")
    print(f"  {'─'*60}")

    for it in range(cfg.SQD_ITERS):
        # Recover missing configurations by guided bit-flipping
        bsm, probs = recover_configurations(
            bsm, probs, avg_occs,
            num_elec_a = n_alpha,
            num_elec_b = n_beta,
            rand_seed  = 42 + it,   # vary seed per iteration
        )

        if bsm.shape[0] == 0:
            print(f"  [iter {it+1}] No valid configs after recovery — stopping.")
            break

        # Diagonalize H in the subspace of known configurations
        sqd_energy, _, avg_occs, spin_sq_val = solve_fermion(
            bsm,
            hcore      = h1e,
            eri        = h2e,
            open_shell = False,
            spin_sq    = 0.0,
        )

        delta = abs(sqd_energy - fci_ref_e) if fci_ref_e is not None else float('nan')

        iterations.append({
            "iter"     : it + 1,
            "energy"   : float(sqd_energy),
            "n_configs": int(bsm.shape[0]),
            "spin_sq"  : float(spin_sq_val),
            "delta"    : float(delta),
        })

        print(f"  Iter {it+1:02d} | E = {sqd_energy:.8f} Ha | "
              f"configs = {bsm.shape[0]:5d} | "
              f"<S²> = {spin_sq_val:.4f} | "
              f"ΔE = {delta:.2e} Ha")

    # Convergence check: last ΔE < 1 mHa
    converged = (
        len(iterations) > 0 and
        fci_ref_e is not None and
        iterations[-1]["delta"] < 1e-3
    )

    return {
        "solver"       : "sqd",
        "energy"       : float(sqd_energy) if sqd_energy is not None else None,
        "error_vs_fci" : abs(sqd_energy - fci_ref_e)
                         if (sqd_energy is not None and fci_ref_e is not None) else None,
        "n_configs"    : int(bsm.shape[0]) if bsm.shape[0] else 0,
        "spin_sq"      : float(spin_sq_val) if spin_sq_val is not None else None,
        "iterations"   : iterations,
        "converged"    : converged,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Solver 2: SKQD
# ═════════════════════════════════════════════════════════════════════════════

def run_skqd(h1e, h2e, n_emb, n_alpha, n_beta, fci_ref_e, cfg):
    """
    Sampling-based Krylov Quantum Diagonalization.

    Key difference from SQD:
      Instead of one static random circuit, build a SYSTEMATIC Krylov basis
      by repeatedly applying the time evolution operator e^{-iHt} to the HF
      reference state:

        |ψ_k⟩ = (e^{-iH·dt})^k |HF⟩

      The Krylov vectors naturally span the low-energy subspace because
      time evolution mixes in the exact ground state eigenvector.

    For each k = 0, 1, ..., K:
      1. Sample bitstrings from |ψ_k⟩
      2. Accumulate ALL bitstrings from |ψ_0⟩ ... |ψ_k⟩
      3. Filter to (n_alpha, n_beta) sector
      4. Call solve_fermion on cumulative subspace
      5. Energy decreases monotonically as K increases

    Convergence:
      For large K, the accumulated bitstrings span the ground state
      support → E_SKQD → E_FCI monotonically.

    Implementation detail:
      Time evolution is approximated by LieTrotter product formula:
        e^{-iHt} ≈ Π_j e^{-iH_j t/r}   (r Trotter steps)
      where H = Σ H_j (sum of Pauli terms from JW transform).

    Requires: openfermion for Jordan-Wigner transform of h1e, h2e.
    """
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import LieTrotter

    print("\n── SKQD Solver ─────────────────────────────────────────────")
    print(f"  Krylov dim   : {cfg.SKQD_KRYLOV_DIM}")
    print(f"  Time step dt : {cfg.SKQD_DT} Ha⁻¹ per Krylov step")
    print(f"  Trotter reps : {cfg.SKQD_TROTTER_REPS}  "
          f"(higher = more accurate, deeper circuit)")
    print(f"  Shots/circuit: {cfg.SKQD_SHOTS:,}")

    n_qubits_local = 2 * n_emb

    # ── Step 1: Build JW qubit Hamiltonian ────────────────────────────────────
    # Needed to construct the time evolution gate e^{-iH_qubit · dt}
    # The JW encoding maps fermionic bitstrings directly to qubit states,
    # so bitstrings from the qubit circuit ARE fermionic occupation vectors.
    print("\n  Building Jordan-Wigner qubit Hamiltonian...")
    H_qubit = build_jw_hamiltonian(h1e, h2e, n_emb)
    print(f"  SparsePauliOp: {len(H_qubit)} Pauli terms | {n_qubits_local} qubits")

    # ── Step 2: Reference (HF) state ─────────────────────────────────────────
    ref_qc = QuantumCircuit(n_qubits_local)
    for i in range(n_alpha): ref_qc.x(i)          # fill α orbitals
    for i in range(n_beta):  ref_qc.x(n_emb + i)  # fill β orbitals

    print(f"  Reference: HF state with {n_alpha}α + {n_beta}β electrons")

    # ── Step 3: Build Krylov circuits ────────────────────────────────────────
    # Circuit k: apply the evolution gate k times to the HF state
    # Each application adds one time step dt → total time = k * dt
    #
    # LieTrotter: e^{-iH dt} ≈ Π_{j} e^{-iH_j dt/reps}
    # SKQD_TROTTER_REPS: higher = better approximation, deeper circuit
    evol_gate = PauliEvolutionGate(
        H_qubit,
        time      = cfg.SKQD_DT / cfg.SKQD_TROTTER_REPS,
        synthesis = LieTrotter(reps=cfg.SKQD_TROTTER_REPS),
    )

    krylov_circs = []
    for k in range(cfg.SKQD_KRYLOV_DIM):
        qc = ref_qc.copy()
        for _ in range(k):
            qc.append(evol_gate, range(n_qubits_local))
        qc.measure_all()
        krylov_circs.append(qc)

    print(f"  Built {cfg.SKQD_KRYLOV_DIM} Krylov circuits")

    # ── Step 4: Sample each Krylov circuit ────────────────────────────────────
    '''print(f"\n  Sampling Krylov circuits...")
    sampler        = StatevectorSampler()
    krylov_counts  = []   # list of count dicts, one per Krylov vector

    for k, circ in enumerate(krylov_circs):
        raw = (
            sampler.run([circ], shots=cfg.SKQD_SHOTS)
            .result()[0]
            .data.meas.get_counts()
        )
        # Quick validity check
        bsm_k, prob_k = counts_to_arrays(raw)
        bsm_k, prob_k = filter_bitstrings(bsm_k, prob_k, n_alpha, n_beta, n_emb)
        krylov_counts.append(raw)
        print(f"    k = {k:2d} | {len(raw):6d} unique bitstrings | "
              f"{bsm_k.shape[0]:5d} valid")'''


    print(f"\n  Sampling {cfg.SKQD_KRYLOV_DIM} Krylov circuits "
      f"[backend = {getattr(cfg, 'BACKEND', 'local').upper()}]...")
    all_raw = sample_circuits(krylov_circs, cfg.SKQD_SHOTS, cfg)

    krylov_counts = []
    for k, raw in enumerate(all_raw):
        bsm_k, prob_k = counts_to_arrays(raw)
        bsm_k, prob_k = filter_bitstrings(bsm_k, prob_k, n_alpha, n_beta, n_emb)
        krylov_counts.append(raw)
        print(f"    k = {k:2d} | {len(raw):6d} unique bitstrings | "
          f"{bsm_k.shape[0]:5d} valid")


    # ── Step 5: Accumulate and solve ──────────────────────────────────────────
    # For each k, use ALL bitstrings from Krylov vectors 0..k.
    # The subspace grows as k increases → energy decreases monotonically.
    iterations   = []
    skqd_energy  = None
    spin_sq_last = None

    target_str = f"{fci_ref_e:.8f}" if fci_ref_e is not None else "N/A"
    print(f"\n  Convergence (FCI target = {target_str} Ha):")
    print(f"  {'─'*60}")

    for k in range(cfg.SKQD_KRYLOV_DIM):
        # Accumulate counts from Krylov steps 0..k (inclusive)
        cumulative = Counter()
        for d in krylov_counts[:k + 1]:
            cumulative.update(d)

        # Convert and filter
        bsm, probs = counts_to_arrays(dict(cumulative))
        bsm, probs = filter_bitstrings(bsm, probs, n_alpha, n_beta, n_emb)
        n_valid    = bsm.shape[0]

        if n_valid < 2:
            print(f"  k = {k:2d} | Only {n_valid} valid config(s) — skipping")
            iterations.append({
                "k": k, "energy": None, "n_configs": n_valid,
                "delta": None, "spin_sq": None,
            })
            continue

        # Solve in the accumulated subspace
        # No recover_configurations needed: Krylov vectors naturally provide
        # systematic coverage of the low-energy Hilbert space
        try:
            energy, _, _, spin_sq = solve_fermion(
                bsm,
                hcore      = h1e,
                eri        = h2e,
                open_shell = False,
                spin_sq    = 0.0,
            )

            skqd_energy  = energy
            spin_sq_last = spin_sq
            delta = abs(energy - fci_ref_e) if fci_ref_e is not None else float('nan')

            iterations.append({
                "k"        : k,
                "energy"   : float(energy),
                "n_configs": int(n_valid),
                "spin_sq"  : float(spin_sq),
                "delta"    : float(delta),
            })

            print(f"  k = {k:2d} | E = {energy:.8f} Ha | "
                  f"configs = {n_valid:5d} | "
                  f"<S²> = {spin_sq:.4f} | "
                  f"ΔE = {delta:.2e} Ha")

        except Exception as exc:
            print(f"  k = {k:2d} | solve_fermion error: {exc}")
            iterations.append({
                "k": k, "energy": None, "n_configs": n_valid,
                "delta": None, "spin_sq": None,
            })

    valid_iters = [it for it in iterations if it["energy"] is not None]
    converged   = (
        len(valid_iters) > 0 and
        fci_ref_e is not None and
        valid_iters[-1]["delta"] < 1e-3
    )

    return {
        "solver"       : "skqd",
        "energy"       : float(skqd_energy) if skqd_energy is not None else None,
        "error_vs_fci" : abs(skqd_energy - fci_ref_e)
                         if (skqd_energy is not None and fci_ref_e is not None) else None,
        "n_configs"    : valid_iters[-1]["n_configs"] if valid_iters else 0,
        "spin_sq"      : float(spin_sq_last) if spin_sq_last is not None else None,
        "iterations"   : iterations,
        "converged"    : converged,
    }

# ═════════════════════════════════════════════════════════════════════════════
# Solver 3 : SqDRIFT
# ═════════════════════════════════════════════════════════════════════════════

def run_sqdrift(h1e, h2e, n_emb, n_alpha, n_beta, fci_ref_e, cfg):
    """
    SqDRIFT: Sampling-based Quantum Diagonalization with qDRIFT circuits.

    Difference from SQD:
      Instead of one random EfficientSU2, generates an ENSEMBLE of time
      evolution circuits by subsampling Hamiltonian terms via qDRIFT.
      Each randomization explores a different region of Hilbert space.

    Pipeline:
      1. h1e/h2e → FermionOperator (via temp FCIDUMP, PySCF → qiskit_fermions)
      2. Group symmetry-related terms  → shorter circuits per group
      3. Build Evolution circuit template
      4. Transpile N randomized qDRIFT circuits (each uses different rng seed)
      5. Prepend HF init + append measurement to each circuit
      6. Sample all circuits via sample_circuits() (backend-dispatched)
      7. Aggregate bitstrings, run iterative recover_configs + solve_fermion
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
            "Install: pip install qiskit-fermions\n"
            "Or switch solver: config.QUANTUM_SOLVER = 'sqd'"
        )

    import os, tempfile
    from pyscf.tools import fcidump as pyscf_fcidump
    from collections import Counter

    num_modes    = 2 * n_emb
    num_circuits = getattr(cfg, "SQDRIFT_NUM_CIRCUITS", 10)
    num_groups   = getattr(cfg, "SQDRIFT_NUM_GROUPS",   10)
    time_scale   = getattr(cfg, "SQDRIFT_TIME",         1.0)
    n_iters      = getattr(cfg, "SQDRIFT_ITERS",        10)
    shots        = getattr(cfg, "SQDRIFT_SHOTS",
                           getattr(cfg, "N_SHOTS", 8192))

    print("\n── SqDRIFT Solver ──────────────────────────────────────────")
    print(f"  num_circuits : {num_circuits}")
    print(f"  num_groups   : {num_groups}  (qDRIFT terms per circuit)")
    print(f"  time_scale   : {time_scale} Ha⁻¹")
    print(f"  shots/circuit: {shots:,}")
    print(f"  backend      : {getattr(cfg, 'BACKEND', 'local').upper()}")

    # ── Step 1: h1e/h2e → FermionOperator via FCIDUMP ─────────────────────────
    # qiskit_fermions expects a FCIDump object; PySCF can write one from integrals.
    print("\n  Building FermionOperator from h1e/h2e (via FCIDUMP)...")
    fd, tmp_path = tempfile.mkstemp(suffix=".fcidump")
    os.close(fd)

    try:
        pyscf_fcidump.from_integrals(
            tmp_path,
            h1e, h2e,
            n_emb,
            n_alpha + n_beta,        # total electrons
            ms=abs(n_alpha - n_beta),
        )
        fcidump_obj = FCIDump.from_file(tmp_path)
        hamil       = FermionOperator.from_fcidump(fcidump_obj)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # ── Step 2: Group symmetry-related terms ──────────────────────────────────
    # Orbitals related by spatial/spin symmetry share identical coefficients.
    # Grouping them allows their Pauli strings to cancel → shorter circuits.
    group_terms_by_electronic_structure(hamil, num_modes)
    n_unique_groups = len(set(hamil.groups))
    print(f"  Grouped into {n_unique_groups} unique excitation groups")

    # ── Step 3: Build Evolution circuit template ───────────────────────────────
    # No HF init and no measurement yet — added per-circuit below.
    evo_gate      = Evolution(num_modes, hamil, time_scale)
    circ_template = FermionicCircuit(num_modes)
    circ_template.append(evo_gate, circ_template.modes)

    # ── Step 4 + 5: Generate randomized circuits, add HF init + measure ────────
    # Each seed gives a different random subsampling of Hamiltonian groups.
    print(f"\n  Generating {num_circuits} randomized qDRIFT circuits...")
    pm = generate_preset_jw_pass_manager()

    sqdrift_circuits = []
    for i in range(num_circuits):
        # Fresh QDriftTrotterization per circuit — different rng seed = different
        # subsampling of excitation groups from the Hamiltonian
        pm.optimization = FermionicPassManager(
            [QDriftTrotterization(num_groups, rng=42 + i)]
        )
        transpiled = pm.run(circ_template)   # QuantumCircuit after JW + qDRIFT

        # Prepend HF reference state (matches qiskit_addon_sqd bitstring convention:
        #   qubits 0..n_emb-1       → alpha spin-orbitals
        #   qubits n_emb..2*n_emb-1 → beta  spin-orbitals)
        hf_qc = QuantumCircuit(num_modes)
        for j in range(n_alpha): hf_qc.x(j)
        for j in range(n_beta):  hf_qc.x(n_emb + j)

        full_qc = hf_qc.compose(transpiled)
        full_qc.measure_all()
        sqdrift_circuits.append(full_qc)

    depths = [c.depth() for c in sqdrift_circuits]
    print(f"  Depths: min={min(depths)}  max={max(depths)}  "
          f"mean={np.mean(depths):.1f}")

    '''
    # ── Step 6: Sample all circuits (backend-dispatched) ──────────────────────
    print(f"\n  Sampling {num_circuits} circuits...")
    all_counts = sample_circuits(sqdrift_circuits, shots, cfg)

    # Aggregate bitstrings from all circuits in the ensemble
    cumulative = Counter()
    for counts in all_counts:
        cumulative.update(counts)

    bsm, probs = counts_to_arrays(dict(cumulative))
    bsm, probs = filter_bitstrings(bsm, probs, n_alpha, n_beta, n_emb)

    print(f"  Total shots collected    : {sum(cumulative.values()):,}")
    print(f"  Valid (N-conserving) configs: {bsm.shape[0]:,}")

    if bsm.shape[0] == 0:
        raise RuntimeError(
            "No valid bitstrings after filtering.\n"
            "Try: increase SQDRIFT_NUM_CIRCUITS or SQDRIFT_SHOTS in config.py"
        )

    # ── Step 7: Iterative SQD post-processing (identical to run_sqd) ──────────
    # recover_configurations expands the subspace; solve_fermion diagonalizes H.
    avg_occs = (
        np.array([1.0 if i < n_alpha else 0.0 for i in range(n_emb)]),
        np.array([1.0 if i < n_beta  else 0.0 for i in range(n_emb)]),
    ) '''
      # ── Step 6: Sample all circuits (backend-dispatched) ──────────────────────
    print(f"\n  Sampling {num_circuits} circuits...")
    all_counts = sample_circuits(sqdrift_circuits, shots, cfg)

    # Aggregate bitstrings from all circuits in the ensemble
    cumulative = Counter()
    for counts in all_counts:
        cumulative.update(counts)

    bsm, probs = counts_to_arrays(dict(cumulative))
    bsm, probs = filter_bitstrings(bsm, probs, n_alpha, n_beta, n_emb)

    print(f"  Total shots collected       : {sum(cumulative.values()):,}")
    print(f"  Valid (N-conserving) configs: {bsm.shape[0]:,} / {max_cfg:,} possible")

    # ── Small-space exhaustive seeding ────────────────────────────────────────
    # When the full configuration space is ≤ 5000 determinants, enumerate ALL
    # valid bitstrings and merge them with the sampled set.
    # This guarantees recover_configurations has a meaningful subspace to
    # expand from and prevents it from being stuck on only a handful of configs.
    if max_cfg <= 5000:
        from itertools import combinations
        print(f"\n  Config space is tiny ({max_cfg} dets) → injecting all valid "
              f"bitstrings exhaustively...")

        all_rows = []
        for alpha_bits in combinations(range(n_emb), n_alpha):
            for beta_bits in combinations(range(n_emb), n_beta):
                row = np.zeros(2 * n_emb, dtype=bool)
                for b in alpha_bits:
                    row[b] = True
                for b in beta_bits:
                    row[n_emb + b] = True
                all_rows.append(row)

        bsm_exhaust = np.array(all_rows, dtype=bool)
        # Uniform probability over exhaustive set
        prob_exhaust = np.ones(len(bsm_exhaust)) / len(bsm_exhaust)

        if bsm.shape[0] > 0:
            # Merge sampled + exhaustive, remove exact duplicates
            bsm_combined  = np.vstack([bsm_exhaust, bsm])
            prob_combined = np.concatenate([prob_exhaust, probs])
            # Deduplicate by treating each row as a string key
            seen = {}
            for i, row in enumerate(bsm_combined):
                key = row.tobytes()
                if key not in seen:
                    seen[key] = (row, prob_combined[i])
            bsm   = np.array([v[0] for v in seen.values()], dtype=bool)
            probs = np.array([v[1] for v in seen.values()])
            probs = probs / probs.sum()   # renormalize
        else:
            bsm   = bsm_exhaust
            probs = prob_exhaust

        print(f"  Configs after exhaustive merge: {bsm.shape[0]:,} "
              f"(= full space)")

    # ── Always inject HF reference explicitly ────────────────────────────────
    # Ensures the Hartree-Fock determinant is always in the subspace
    # regardless of what the qDRIFT circuits happened to sample.
    hf_row = np.zeros(2 * n_emb, dtype=bool)
    for j in range(n_alpha): hf_row[j]        = True
    for j in range(n_beta):  hf_row[n_emb + j] = True
    hf_key = hf_row.tobytes()

    existing_keys = {bsm[i].tobytes() for i in range(bsm.shape[0])}
    if hf_key not in existing_keys:
        bsm   = np.vstack([bsm,   hf_row[np.newaxis, :]])
        probs = np.append(probs, 1.0 / (bsm.shape[0]))
        probs = probs / probs.sum()
        print(f"  Injected HF reference (was missing from sampled configs)")

    if bsm.shape[0] == 0:
        raise RuntimeError(
            "No valid bitstrings after filtering + seeding.\n"
            "Try: increase SQDRIFT_NUM_CIRCUITS or SQDRIFT_SHOTS in config.py"
        )

    # ── THIS BLOCK WAS MISSING — add it here ──────────────────────────────────
    # avg_occs: initial aufbau guess for recover_configurations guidance.
    # Alpha: fill lowest n_alpha orbitals. Beta: fill lowest n_beta orbitals.
    # recover_configurations uses this to decide which bits to flip.
    avg_occs = (
        np.array([1.0 if i < n_alpha else 0.0 for i in range(n_emb)]),
        np.array([1.0 if i < n_beta  else 0.0 for i in range(n_emb)]),
    )

    iterations  = []
    sqd_energy  = None
    spin_sq_val = None

    target_str = f"{fci_ref_e:.8f}" if fci_ref_e is not None else "N/A"
    print(f"\n  Iterating (FCI target = {target_str} Ha):")
    print(f"  {'─'*60}")

    for it in range(n_iters):
        bsm, probs = recover_configurations(
            bsm, probs, avg_occs,
            num_elec_a=n_alpha,
            num_elec_b=n_beta,
            rand_seed=42 + it,
        )

        if bsm.shape[0] == 0:
            print(f"  [iter {it+1}] No configs after recovery — stopping.")
            break

        sqd_energy, _, avg_occs, spin_sq_val = solve_fermion(
                bsm,
                hcore=h1e,
                eri=h2e,
                open_shell=True,    # ← alpha and beta treated independently
                    spin_sq=None,       # ← no penalty; ground state is already singlet
                )

        delta = (abs(sqd_energy - fci_ref_e)
                 if fci_ref_e is not None else float("nan"))

        iterations.append({
            "iter"     : it + 1,
            "energy"   : float(sqd_energy),
            "n_configs": int(bsm.shape[0]),
            "spin_sq"  : float(spin_sq_val),
            "delta"    : float(delta),
        })

        print(f"  Iter {it+1:02d} | E = {sqd_energy:.8f} Ha | "
              f"configs = {bsm.shape[0]:5d} | "
              f"<S²> = {spin_sq_val:.4f} | "
              f"ΔE = {delta:.2e} Ha")

    converged = (
        len(iterations) > 0
        and fci_ref_e is not None
        and iterations[-1]["delta"] < 1e-3
    )

    return {
        "solver"       : "sqdrift",
        "energy"       : float(sqd_energy) if sqd_energy is not None else None,
        "error_vs_fci" : (abs(sqd_energy - fci_ref_e)
                          if sqd_energy is not None and fci_ref_e is not None
                          else None),
        "n_configs"    : int(bsm.shape[0]),
        "spin_sq"      : float(spin_sq_val) if spin_sq_val is not None else None,
        "iterations"   : iterations,
        "converged"    : converged,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Solver registry — add future solvers here
# ═════════════════════════════════════════════════════════════════════════════

SOLVER_REGISTRY = {
    "sqd"  : run_sqd,
    # "skqd" : run_skqd,   ← uncomment when openfermion is installed
    "skqd" : run_skqd,
    "sqdrift" : run_sqdrift
    # Future:
    # "sqdrift" : run_sqdrift,
    # "vqe"     : run_vqe,
    # "qite"    : run_qite,
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
solver_name = getattr(config, "QUANTUM_SOLVER", "sqd").lower()

if solver_name not in SOLVER_REGISTRY:
    raise ValueError(
        f"Unknown solver '{solver_name}'.\n"
        f"Available: {list(SOLVER_REGISTRY.keys())}\n"
        f"Set config.QUANTUM_SOLVER accordingly."
    )

solver_fn = SOLVER_REGISTRY[solver_name]

# Optional FCI reference (computed from same h1e, h2e if not already done)
if fci_ref_e is None:
    n_dets = math.comb(n_emb, n_alpha) ** 2
    if n_dets <= 5_000_000:
        print(f"\n  Computing FCI reference ({n_dets:,} determinants)...")
        cisolver  = pyscf_fci.direct_spin1.FCI()
        fci_ref_e, _ = cisolver.kernel(h1e, h2e, n_emb, (n_alpha, n_beta))
        print(f"  FCI reference : {fci_ref_e:.8f} Ha")

# Run solver
solver_results = solver_fn(h1e, h2e, n_emb, n_alpha, n_beta, fci_ref_e, config)

# ── Final summary ─────────────────────────────────────────────────────────────
final_energy = solver_results["energy"]
error_vs_fci = solver_results["error_vs_fci"]
n_configs    = solver_results["n_configs"]
spin_sq      = solver_results["spin_sq"]
converged    = solver_results["converged"]

print(f"\n{'═'*65}")
print(f"[Step 3] Final Summary — {mol_info['molecule']}")
print(f"{'═'*65}")
print(f"  Molecule                   : {mol_info['molecule']}")
print(f"  Basis                      : {mol_info['basis']}")
print(f"  Complexity class / tier    : "
      f"{scores_s1.get('complexity_class','?')} / {scores_s1.get('tier_used','?')}")
print(f"  ASF active space           : {nel}e in {n_active_orbs} orbs → {mo_list}")
print(f"  DMET: imp + bath           : {n_imp} + {n_bath} = {n_emb} orbs "
      f"= {n_qubits} qubits")
print(f"  Sv² bath coverage          : {scores_s2.get('sv2_coverage', 'N/A'):.4f}")
print(f"  MP2 DM used for bath       : {scores_s2.get('mp2_dm_used', 'N/A')}")
print(f"  Solver                     : {solver_name.upper()}")
print(f"  Final energy ({solver_name.upper():5s})        : "
      f"{final_energy:.8f} Ha" if final_energy is not None else "  Final energy: N/A")
print(f"  FCI reference energy       : "
      f"{fci_ref_e:.8f} Ha" if fci_ref_e is not None else "  FCI reference: N/A")
print(f"  Δ ({solver_name.upper()} vs FCI)          : "
      f"{error_vs_fci:.2e} Ha" if error_vs_fci is not None else "  Δ: N/A")
print(f"  Final <S²>                 : "
      f"{spin_sq:.6f}  (0 = singlet ✓)" if spin_sq is not None else "  <S²>: N/A")
print(f"  Final subspace configs     : {n_configs:,}")
print(f"  Converged (ΔE < 1 mHa)    : {converged}")

# ── Correlation + ASF scores for pipeline ────────────────────────────────────
print(f"\n  Molecular Score Vector (for graph transformer pipeline):")
print(f"  {'─'*50}")
score_keys = [
    "complexity_class", "correlation_strength", "max_correlation",
    "homo_lumo_gap_eV", "n_active_orbitals", "n_active_electrons",
    "entropy_gap",
]
for k in score_keys:
    v = scores_s1.get(k, "N/A")
    if isinstance(v, float):
        print(f"  {k:<30} {v:.4f}")
    else:
        print(f"  {k:<30} {v}")

# Embedding quality
embed_keys = ["sv2_coverage", "bath_fraction", "embedding_corr_energy"]
for k in embed_keys:
    v = scores_s2.get(k, "N/A")
    if isinstance(v, float):
        print(f"  {k:<30} {v:.4f}")
    elif v is not None:
        print(f"  {k:<30} {v}")

print(f"{'═'*65}")

# ── Save ──────────────────────────────────────────────────────────────────────
output = {
    # ── Solver output ─────────────────────────────────────────────────────────
    "solver"        : solver_name,
    "energy"        : final_energy,
    "fci_ref_e"     : fci_ref_e,
    "error_vs_fci"  : error_vs_fci,
    "n_configs"     : n_configs,
    "spin_sq"       : spin_sq,
    "converged"     : converged,
    "iterations"    : solver_results["iterations"],

    # ── Combined score vector for scoring pipeline ────────────────────────────
    "pipeline_score": {
        **{k: scores_s1.get(k) for k in [
            "complexity_class", "tier_used",
            "correlation_strength", "max_correlation", "std_correlation",
            "n_strongly_correlated", "homo_lumo_gap_eV",
            "n_active_electrons", "n_active_orbitals", "entropy_gap",
            "mp2_correlation_energy", "metal_fraction",
        ]},
        **{k: scores_s2.get(k) for k in [
            "n_emb", "n_qubits", "sv2_coverage", "bath_fraction",
            "mp2_dm_used", "embedding_corr_energy",
        ]},
        "quantum_error_vs_fci" : float(error_vs_fci) if error_vs_fci is not None else None,
        "quantum_spin_sq"      : float(spin_sq) if spin_sq is not None else None,
    },

    # ── Metadata ──────────────────────────────────────────────────────────────
    "mol_info"      : mol_info,
}

with open(STEP3_FILE, "wb") as fh:
    pickle.dump(output, fh)

print(f"\n[Step 3] ✓ Saved → {STEP3_FILE}")