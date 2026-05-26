# step2_hamiltonian.py — DMET Embedding

## Purpose
Construct an effective Hamiltonian (h1e, h2e) in a small embedding
space of (n_imp + n_bath) orbitals. This Hamiltonian is chemically
exact in the embedding space — it includes the effect of all other
electrons through a mean-field potential, not by ignoring them.

---

## Why Not Just Use the Active Space Integrals?

The impurity orbitals (Fe 3d) are NOT isolated in a molecule.
They are covalently bonded to N 2p orbitals.

| Approach | Problem |
|----------|---------|
| Use h1e/h2e for impurity only | Missing Fe-N covalency entirely |
| Use full molecule | Too many qubits |
| DMET | Exact in embedding space ✓ |

DMET finds the minimal set of environment orbitals (bath) that are
quantum-mechanically entangled with the impurity, then traces out
everything else — contributing it as a mean-field background.

---

## Eight-Phase Pipeline

### Phase A — UHF with Newton Fallback

Rebuilds the molecule and runs UHF with the same Newton fallback as
Step 1. Must converge to a valid reference state for the Schmidt
decomposition to be physically meaningful.

---

### Phase B — MP2 Density Matrix

```
DM choice:
  MP2 density matrix  →  captures charge-transfer fluctuations
  UHF density matrix  →  mean-field only, misses Fe-N correlation

For FeN6: MP2 DM shows N→Fe donation clearly in off-diagonal blocks
           UHF DM shows only the static ionic picture
```

The MP2 DM is used for Schmidt decomposition. This is the key
improvement over naive DMET implementations.

Falls back to UHF DM if MP2 fails (rare).

---

### Phase C — Schmidt Decomposition

**Löwdin orthogonalisation:**
```
S^{+1/2}   transforms AO basis → orthonormal Löwdin basis
S^{-1/2}   transforms back → AO basis

Why Löwdin: each orbital looks as much as possible like its parent AO.
            Minimises basis rotation → physically interpretable bath orbitals.
```

**Schmidt decomposition:**
```
F = P_env @ ρ_lo @ Q_imp          environment × impurity DM block
F = U_env · diag(sv) · V†         SVD

sv_k large → environment orbital k strongly entangled with impurity
sv_k ≈ 0   → environment orbital k barely entangled → discard to core
```

**Adaptive bath selection (two criteria, take max):**

Criterion 1 — SV gap detection:
```
Same algorithm as Step 1 gap detection but on Schmidt SVs.
Finds natural boundary between entangled and unentangled bath orbitals.
```

Criterion 2 — sv² coverage:
```
Keep bath orbitals until Σ sv²_i / Σ_all sv²_i ≥ 0.999
Ensures 99.9% of total impurity-environment entanglement is captured.
```

**Hard constraints:**
```
n_bath ≤ n_imp           DMET theorem: bath rank ≤ impurity rank
n_bath ≤ MAX_EMBED_ORBS - n_imp    qubit budget
```

---

### Phase D — Core Mean-Field Potential

Core electrons are frozen at the mean-field level and contribute a
Coulomb-Exchange field to the effective 1e Hamiltonian:

```
h1e_eff[p,q] = h1e_bare[p,q] + J_core[p,q] - 0.5 * K_core[p,q]

J_core: Coulomb repulsion from core electrons
K_core: Exchange interaction from core electrons (same spin only)

ecore = E_nuc + 0.5 * Tr[ρ_core (h1e_bare + h1e_eff)]
```

`ecore` is a constant energy offset. Not needed for SQD vs FCI
comparison (both use same h1e_emb) but stored for reference.

---

### Phase E — Validation

Three numerical health checks:

| Check | Formula | Pass threshold |
|-------|---------|---------------|
| Orthonormality | max\|C_emb^T S C_emb - I\| | < 1e-6 |
| Electron count | \|n_elec_emb - round(n_elec_emb)\| | < 0.15 |
| DM hermiticity | max\|ρ - ρ^T\| | < 1e-8 |

If orthonormality fails → re-orthogonalise via QR decomposition.
If electron count fails → apply chemical potential correction (Phase G).

---

### Phase F — Integral Transformation

```
h1e_emb = C_emb^T @ h1e_eff_AO @ C_emb          O(n_AO² × n_emb²)
h2e_emb = ao2mo.kernel(mol, C_emb)               O(n_AO⁴)

h2e[p,q,r,s] = (pq|rs) two-electron integrals in embedding MO basis
               = PySCF chemist notation
```

Symmetry restoration after transformation:
```python
h2e = 0.5 * (h2e + h2e.T(1,0,2,3))   # p↔q swap
h2e = 0.5 * (h2e + h2e.T(0,1,3,2))   # r↔s swap
h2e = 0.5 * (h2e + h2e.T(2,3,0,1))   # pq↔rs swap
```

Main cost step. For STO-3G FeN6 (48 AOs): ~30 seconds.

---

### Phase G — Chemical Potential Correction

If electron count deviates > 0.15 from integer:

```
Shift h1e → h1e - μ·I

Find μ such that Σ_{eigenvalues of h1e-μI < 0} = n_target / 2

Method: bisection (Brentq algorithm, converges in < 50 iterations)
```

This enforces canonical ensemble (fixed N) which FCI and SQD assume.
One-shot DMET is grand-canonical by default.

---

### Phase H — FCI + Quality Scores

FCI is run on the embedding Hamiltonian if feasible
(max determinants ≤ 5,000,000):

```
C(n_emb, n_alpha)² determinants
C(6, 1)² = 36         ← trivial, always runs
C(16, 8)² = 12,870²   ← ~165M, skipped
```

FCI result stored as the exact reference target for Step 3.

**Quality scores computed:**

| Score | Meaning |
|-------|---------|
| `sv2_coverage` | Fraction of entanglement in bath (1.0 = complete) |
| `bath_fraction` | n_bath / n_imp ratio |
| `electron_deviation` | How fractional the electron count was |
| `embedding_corr_energy` | E_FCI - E_UHF in embedding (more negative = more correlated) |
| `mp2_dm_used` | Whether MP2 or UHF DM was used for Schmidt |

---

## Output: step2_hamiltonian.pkl

| Key | Shape/Type | Description |
|-----|-----------|-------------|
| `h1e` | (n_emb, n_emb) | Effective 1e integrals in embedding basis |
| `h2e` | (n_emb,)×4 | 2e integrals, chemist notation |
| `ecore` | float | Constant energy offset from core |
| `n_emb` | int | Total embedding orbitals (imp + bath) |
| `n_imp` | int | Impurity orbital count (from Step 1) |
| `n_bath` | int | Bath orbital count (from Schmidt) |
| `n_alpha` | int | Alpha electrons in embedding |
| `n_beta` | int | Beta electrons in embedding |
| `fci_ref_e` | float\|None | FCI energy in embedding (if feasible) |
| `sv` | (n_bath,) | Schmidt singular values kept |
| `scores` | dict | Full embedding quality score vector |

---

## Key Tunable Parameters (config.py)

```python
MAX_EMBED_ORBS = 16      # n_imp + n_bath ceiling → controls qubit count
BATH_TOLERANCE = 1e-8    # SV below this → core (currently unused)
CORE_OCC_THRESHOLD = 1.8 # for electron counting
```