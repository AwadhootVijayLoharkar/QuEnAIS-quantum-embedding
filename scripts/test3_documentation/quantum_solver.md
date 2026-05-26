# step3_sqd.py — Quantum Solver

## Purpose
Solve the embedding Hamiltonian (h1e, h2e) from Step 2 using a
quantum-classical hybrid algorithm. Two solvers are available:
SQD and SKQD. Three execution backends: local, MPS, IBM.

---

## Solver Comparison

| Feature | SQD | SKQD |
|---------|-----|------|
| Circuit strategy | One random EfficientSU2 | K time-evolution circuits |
| Basis construction | Random sampling | Systematic Krylov basis |
| Convergence | Iterative recovery | Monotonic with K |
| JW transform needed | ✗ | ✓ (openfermion) |
| Suited for | Quick tests | Strongly correlated systems |

---

## Solver 1: SQD

### Algorithm

```

1. Build circuit:

   HF state (X gates on qubits 0..n_alpha-1 and n_emb..n_emb+n_beta-1)

   + EfficientSU2 ansatz with RANDOM parameters


2. Sample N_SHOTS bitstrings
3. Filter to (n_alpha, n_beta) particle-number sector


4. Iterate SQD_ITERS times:

   a. recover_configurations(bsm, probs, avg_occs)
      → randomly flip bits guided by occupation numbers
      → expands subspace to cover missed configurations
   b. solve_fermion(bsm, h1e, h2e)
      → diagonalise H in subspace of known bitstrings
      → returns ground energy + new occupation numbers
   c. Update avg_occs from new ground state
```

### Why Random Parameters?
The goal is DIVERSITY of bitstrings, not the optimal state.
A random ansatz samples broadly across Hilbert space.
`solve_fermion` then finds the ground state within the sampled subspace.
Optimising the circuit parameters would bias sampling toward one state.

### Convergence
Energy decreases as `recover_configurations` expands the subspace.
Converged when ΔE < 1 mHa vs FCI reference.

---

## Solver 2: SKQD

### Algorithm

```
Krylov basis construction:
  |ψ_0⟩ = |HF⟩
  |ψ_1⟩ = e^{-iH·dt}     |HF⟩
  |ψ_2⟩ = e^{-2iH·dt}    |HF⟩
  ...
  |ψ_K⟩ = e^{-KiH·dt}    |HF⟩

For each k = 0..K:

  1. Sample SKQD_SHOTS bitstrings from |ψ_k⟩
  2. Accumulate ALL bitstrings from |ψ_0⟩...|ψ_k⟩

  3. Filter to (n_alpha, n_beta) sector
  4. solve_fermion on cumulative subspace

  5. Energy decreases monotonically as k increases

```

### Why Time Evolution?
The time-evolution operator e^{-iHt} mixes in ALL eigenstates of H
according to their energy. As t increases, the Krylov vectors span
more of the low-energy subspace. Unlike random circuits, this is
physically motivated and converges systematically.

### Trotter Approximation
```
e^{-iH·dt} ≈ Π_j e^{-iH_j·dt/reps}   (LieTrotter product formula)

H = Σ_j H_j   (sum of Pauli terms from Jordan-Wigner)
reps = SKQD_TROTTER_REPS
```

Higher reps → better approximation, deeper circuit.

**Circuit depth scaling:**
```
depth ≈ n_Pauli_terms × SKQD_TROTTER_REPS × k
```

For FeN6: 1819 Pauli terms × 2 reps × k → depth ~262,000 at k=9

This is why MPS is needed for testing and IBM circuits must use
SKQD_TROTTER_REPS=1 and SKQD_KRYLOV_DIM≤4.

### Jordan-Wigner Transform (build_jw_hamiltonian)

```
Fermionic H → Qubit H via Jordan-Wigner mapping

Spin-orbital ordering:
  0..n_orb-1        alpha orbitals  (qiskit_addon_sqd convention)
  n_orb..2*n_orb-1  beta orbitals

Four spin sectors:
  αααα: a†_{pα} a†_{rα} a_{sα} a_{qα}
  ββββ: a†_{pβ} a†_{rβ} a_{sβ} a_{qβ}
  αββα: a†_{pα} a†_{rβ} a_{sβ} a_{qα}   ← mixed spin
  βααβ: a†_{pβ} a†_{rα} a_{sα} a_{qβ}   ← mixed spin

openfermion → jordan_wigner → SparsePauliOp
Qubit index reversal: openfermion q0=leftmost, Qiskit q0=rightmost
```

---

## Backend System

### filter_bitstrings
```
bsm[:, :n_orb]   = alpha spin-orbital occupations
bsm[:, n_orb:]   = beta  spin-orbital occupations

Keep only rows where:
  sum(alpha columns) == n_alpha  AND
  sum(beta columns)  == n_beta

Enforces particle number conservation violated by circuits
that explore the full Hilbert space.
```

### sample_circuits dispatcher

```python
BACKEND = "local"   → _run_local()   exact statevector
BACKEND = "mps"     → _run_mps()     tensor network via Aer
BACKEND = "ibm"     → _run_ibm()     IBM Quantum hardware
```

### _run_mps
```
AerSimulator(method="matrix_product_state")
Bond dimension χ = MPS_MAX_BOND_DIM

Accuracy: χ → ∞ recovers exact statevector
Cost:     O(χ² × circuit_depth × 2^(n/2))  approximately

Circuits decomposed into native gates via transpile() before MPS run.
Works for SKQD circuits that exceed local simulator memory.
```

### _run_ibm
```
Free plan restrictions:
  ✗ Session mode  (HTTP 400 error if attempted)
  ✗ Batch mode
  ✓ Direct job mode  → SamplerV2(mode=backend)

All circuits batched as one job → single queue wait.
Depth guard: if max(depths) > IBM_MAX_CIRCUIT_DEPTH → RuntimeError
             with instructions on how to reduce depth.

Real hardware noise budget: ~1000-3000 gate layers (T2 limited).
```

---

## Output: step3_results.pkl

| Key | Type | Description |
|-----|------|-------------|
| `solver` | str | "sqd" or "skqd" |
| `energy` | float | Final ground state energy (Ha) |
| `fci_ref_e` | float | FCI reference from Step 2 |
| `error_vs_fci` | float | \|E_solver - E_FCI\| (Ha) |
| `n_configs` | int | Final subspace size |
| `spin_sq` | float | ⟨S²⟩ (0.0 = clean singlet) |
| `converged` | bool | ΔE < 1 mHa vs FCI |
| `iterations` | list[dict] | Per-iteration energy history |
| `pipeline_score` | dict | Combined score vector for ML pipeline |

### pipeline_score contents
Merges scores from Steps 1, 2, 3 into one flat dict:
```
complexity_class, tier_used          ← from Step 1
correlation_strength, max_correlation ← from Step 1
sv2_coverage, bath_fraction           ← from Step 2
n_emb, n_qubits                       ← from Step 2
quantum_error_vs_fci                  ← from Step 3
quantum_spin_sq                       ← from Step 3
```

---

## Solver Registry

To add a new solver (e.g. VQE):
```python
def run_vqe(h1e, h2e, n_emb, n_alpha, n_beta, fci_ref_e, cfg):
    ...
    return {"solver": "vqe", "energy": ..., "error_vs_fci": ...,
            "n_configs": ..., "spin_sq": ..., "iterations": ...,
            "converged": ...}

SOLVER_REGISTRY["vqe"] = run_vqe
```

Set `config.QUANTUM_SOLVER = "vqe"` to use it. No other changes needed.

---

## Key Tunable Parameters (config.py)

```python

### SQD

N_SHOTS        = 500_000   # more shots → more configs → lower energy
SQD_ITERS      = 10        # more iters → more recovery → lower energy
ANSATZ_REPS    = 3         # deeper ansatz → broader sampling

### SKQD

SKQD_KRYLOV_DIM   = 10     # more vectors → lower energy, more IBM cost
SKQD_DT           = 0.5    # larger dt → faster Krylov convergence
SKQD_TROTTER_REPS = 2      # reduce to 1 for IBM hardware
SKQD_SHOTS        = 8192   # increase to 10000+ for IBM hardware

### Backend

BACKEND                = "local"
MPS_MAX_BOND_DIM       = 256
IBM_MAX_CIRCUIT_DEPTH  = 3000
```