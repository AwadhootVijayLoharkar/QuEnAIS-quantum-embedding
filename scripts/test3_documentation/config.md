# config.py — Configuration Reference

## Purpose
Single source of truth for all parameters used across Steps 1, 2, and 3.
Change values here only — never hardcode in the step scripts.

---

## Molecule Selection

```python
MOLECULE = "FeN6"   # key into geometries dict
```

Available molecules and their complexity:

| Key   | Description        | Tier | Notes |
|-------|--------------------|------|-------|
| `LiH` | Lithium hydride    | 1    | Simplest test case |
| `H2O` | Water              | 1    | Standard benchmark |
| `N2`  | Nitrogen dimer     | 2    | Triple bond correlation |
| `H6`  | Hydrogen chain     | 2    | 1D Hubbard-like |
| `FeN6`| Iron hexanitride   | 3    | Strongly correlated TM |

Add new molecules by inserting into the `geometries` dict and
updating `ATOM_SYMS`, `N_ATOMS` derived from the geometry.

---

## Complexity Classification

Step 1 auto-classifies every molecule before running ASF.

```
Three indicators checked (all free from UHF):

  1. d/f-block element present   → always Tier 3
  2. Spin contamination ratio    → Tier 2+ if > SPIN_CONTAMINATION_TIER2_THRESHOLD

  3. HOMO-LUMO gap               → Tier 2+ if < HOMO_LUMO_TIER2_THRESHOLD_EV

```

Tier affects only ASF parameter selection (max_norb, entropy_threshold).
Gap detection in Phase C is always adaptive regardless of tier.

---

## ASF Parameters

```python
ASF_PARAMS = {
    1: {"entropy_threshold": 0.01, "max_norb": 12, "min_norb": 2},
    2: {"entropy_threshold": 0.01, "max_norb": 14, "min_norb": 2},
    3: {"entropy_threshold": 0.01, "max_norb": 16, "min_norb": 2},
}
```

`entropy_threshold = 0.01` is intentionally low to get a BROAD candidate
pool. The adaptive gap detection in Phase C then trims to the final size.
Raising this threshold will miss correlated orbitals.

---

## Active Space Size Control

```python
GAP_MIN_NORB = 2    # minimum orbitals gap detection will select
GAP_MAX_NORB = 8    # maximum orbitals → controls qubit count upstream
```

`GAP_MAX_NORB = 8` → DMET embedding ≤ 16 orbitals → ≤ 32 qubits.
Reduce if IBM job is too deep. Increase if energy is not converging.

---

## Embedding Parameters

```python
MAX_EMBED_ORBS = 16    # n_imp + n_bath hard ceiling
BATH_TOLERANCE = 1e-8  # Schmidt SV below this → core (unused in current code)
CORE_OCC_THRESHOLD = 1.8  # NO occupation above this → doubly occupied core
```

`MAX_EMBED_ORBS` is the main qubit budget knob:
- `16` orbs → 32 qubits (manageable on MPS, borderline IBM)
- `8`  orbs → 16 qubits (safe for IBM free plan after Trotter reduction)

---

## SQD Solver Parameters

```python
N_SHOTS     = 500_000   # bitstring samples per circuit
SQD_ITERS   = 10        # recover_configurations → solve_fermion iterations
ANSATZ_REPS = 3         # EfficientSU2 repetitions (depth ∝ reps)
```

Higher `N_SHOTS` → more unique configurations found → lower energy.
Higher `SQD_ITERS` → more recovery steps → monotonically decreasing energy.

---

## SKQD Solver Parameters

```python
SKQD_KRYLOV_DIM   = 10    # number of Krylov vectors
SKQD_DT           = 0.5   # time step per Krylov step (Ha⁻¹)
SKQD_TROTTER_REPS = 2     # Trotter steps per gate (accuracy vs depth)
SKQD_SHOTS        = 8192  # shots per Krylov circuit
```

Circuit depth scales as:  `depth ≈ n_Pauli_terms × TROTTER_REPS × k`

For IBM hardware reduce to:
```python
SKQD_TROTTER_REPS = 1    # halves depth
SKQD_KRYLOV_DIM   = 4    # only 4 circuits
SKQD_SHOTS        = 10_000
```

---

## Backend Selection

```python
BACKEND = "local"   # "local" | "mps" | "ibm"
```

### local
Exact statevector simulation. No approximation. Memory = O(2^n).
Best for: testing with ≤ 20 qubits.

### mps
Matrix Product State via Aer. Controlled approximation.

```python
MPS_MAX_BOND_DIM = 256    # χ: higher = more accurate, slower
MPS_TRUNC_THRESH = 1e-6   # singular value cutoff during contraction
```

Bond dimension guide:

| χ   | Use case |
|-----|----------|
| 32  | Shallow circuits, quick tests |
| 256 | Default, good balance |
| 512 | High accuracy, slow above 30 qubits |

### ibm

```python
IBM_BACKEND_NAME       = None     # None = least_busy auto-select
IBM_OPTIMIZATION_LEVEL = 1        # transpiler depth 0-3
IBM_MAX_CIRCUIT_DEPTH  = 3000     # reject before queue if exceeded
```

`IBM_MAX_CIRCUIT_DEPTH` prevents submitting circuits that will return
noise instead of physics. Typical hardware decoherence budget:
T2 ~100-300 μs, gate time ~100 ns → ~1000-3000 usable layers.

---

## Solver Selection

```python
QUANTUM_SOLVER = "skqd"   # "sqd" | "skqd"
```

Use `sqd` if openfermion is not installed (no JW transform needed).
Use `skqd` for systematic convergence on strongly correlated systems.