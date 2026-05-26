# Physics Concepts Guide — QuEnAIS Pipeline

A first-principles explanation of every physics and quantum chemistry
concept used in the pipeline. No prior quantum chemistry assumed.

---

## 1. The Core Problem: Why Molecules Are Hard to Simulate

A molecule contains N electrons. Each electron can be in one of many
orbitals. The quantum state of the molecule is a superposition of ALL
possible ways to arrange N electrons across those orbitals.

```
Number of possible arrangements = C(n_orbitals, n_alpha) × C(n_orbitals, n_beta)

Example — FeN6 in STO-3G:
  48 spatial orbitals × 2 spins = 96 spin-orbitals
  ~50 alpha electrons, ~50 beta electrons
  Configurations: C(48,25)² ≈ 10^27

This is completely intractable — more states than atoms in the universe.
```

**The fundamental challenge:** Electrons are correlated. The motion of
one electron depends on every other electron simultaneously (not just
on average). Capturing this "electron correlation" is the hard part.

---

## 2. Hartree-Fock (HF) — The Starting Point

**What it is:**
The simplest approximation. Each electron moves in the AVERAGE field
of all other electrons. The wavefunction is a single Slater determinant
(one way to arrange electrons antisymmetrically).

```
Ψ_HF = det[φ_1, φ_2, ..., φ_N]

φ_i = molecular orbital (MO) — a one-electron wavefunction
Each MO is a linear combination of atomic orbitals (AOs)
```

**Why we use it:**
- Fast to compute (minutes for large molecules)
- Gives good molecular geometry, dipole moments
- Essential starting point for all correlated methods

**What it misses:**
- Instantaneous electron-electron repulsion (correlation energy)
- Critical for: transition metals, bond breaking, excited states

**Unrestricted HF (UHF):**
Alpha (↑) and beta (↓) electrons have SEPARATE sets of MOs.
Needed for open-shell molecules like Fe with unpaired 3d electrons.

**Newton fallback:**
The standard DIIS algorithm can oscillate for open-shell Fe systems.
The Newton (second-order) solver uses curvature information to converge
even when DIIS fails.

---

## 3. Spin Contamination — Measuring UHF Quality

In exact quantum mechanics, the spin quantum number S is well-defined:
```
⟨S²⟩_exact = S(S+1)    e.g. singlet: 0,  doublet: 0.75,  triplet: 2.0
```

UHF can mix different spin states, giving a "contaminated" result:
```
contamination ratio = ⟨S²⟩_actual / ⟨S²⟩_expected

1.0 → pure spin state (ideal UHF)
1.3 → 30% contamination → significant open-shell character
2.0 → completely wrong spin state
```

**Why it matters for classification:**
High spin contamination signals that a single-determinant description
is insufficient → the molecule needs correlated treatment → Tier 2+.

---

## 4. HOMO-LUMO Gap — The Electronic Gap

```
HOMO = Highest Occupied Molecular Orbital   (last filled orbital)
LUMO = Lowest Unoccupied Molecular Orbital  (first empty orbital)

Gap = E_LUMO - E_HOMO  (in eV)
```

**Physical meaning:**
```
Large gap (> 5 eV):  electrons stay in their ground configuration
                     single-determinant HF works well
                     Example: H2O (9 eV), benzene (5 eV)

Small gap (< 1 eV):  many configurations nearly degenerate in energy
                     strong mixing between ground and excited states
                     single-determinant fails → needs multireference
                     Example: near-transition-state geometries, metals
```

**Pipeline use:** Small gap → Tier 2+ classification → larger active space.

---

## 5. MP2 — Pair Correlation Energy

**What it is:**
Second-order Møller-Plesset perturbation theory. Adds the lowest-order
correction to HF by considering how pairs of electrons avoid each other.

```
E_MP2 = E_HF + E_corr(2)

E_corr(2) = Σ_{occ i,j} Σ_{virt a,b} |⟨ij||ab⟩|² / (ε_i + ε_j - ε_a - ε_b)

where ⟨ij||ab⟩ = antisymmetrized two-electron integral
      ε_i, ε_a = orbital energies (occupied/virtual)
```

**Physical picture:**
Two electrons in orbitals i, j briefly "scatter" into virtual
orbitals a, b. The energy denominator shows this is most important
when orbitals i,j and a,b are close in energy (small gap → large MP2).

**Why MP2 density matrix is better than UHF for DMET:**
The MP2 one-particle density matrix (1-RDM) includes charge-transfer
fluctuations. For FeN6 this means N→Fe donation appears explicitly,
giving a more physically correct bath selection in Schmidt decomposition.

---

## 6. Natural Orbitals and Deviation Proxy

**Natural orbitals (NOs):**
The eigenvectors of the one-particle density matrix.
Their eigenvalues (occupation numbers n_i) tell how much each orbital
is populated:
```
n_i = 2.0  → always doubly occupied → not correlated
n_i = 0.0  → always empty           → not correlated
n_i = 1.0  → half-filled            → maximally correlated
```

**Deviation proxy (min(n_i, 2 - n_i)):**
```
dev_i = min(n_i, 2 - n_i)

dev = 0.0  →  n_i ≈ 0 or 2  →  boring, uncorrelated orbital
dev = 1.0  →  n_i ≈ 1       →  strongly correlated orbital

This is a cheap proxy for the entanglement entropy of each orbital
without needing expensive CASSCF or DMRG.
```

**Why this works as an active space selector:**
Orbitals with high deviation are those where the electron sometimes
IS there and sometimes IS NOT — precisely the orbitals responsible for
correlation energy. These are the ones we must treat quantum-mechanically.

---

## 7. Gap Detection — Adaptive Active Space Boundary

A fixed deviation threshold (e.g. "keep orbitals with dev > 0.3")
fails for different molecules. Instead we find the NATURAL gap:

```
Sort orbitals by deviation (descending):  d_1 ≥ d_2 ≥ ... ≥ d_N

gap_n = d_n - d_{n+1}    (difference at the n / n+1 boundary)

Select n* = argmax(gap_n)  over  n ∈ [GAP_MIN_NORB, GAP_MAX_NORB]
```

**Physical picture:**
Strongly correlated orbitals cluster with similar high deviation.
A gap in the spectrum marks where the "important" group ends and the
"spectator" group begins. This boundary is universal and molecule-independent.

---

## 8. Löwdin Orthogonalisation

AO basis functions (atom-centered Gaussians) are NOT orthogonal to each
other — they overlap. Many algorithms require orthonormal orbitals.

**Löwdin symmetric orthogonalisation:**
```
S         = AO overlap matrix  (S_μν = ⟨φ_μ|φ_ν⟩)
S^{+1/2}  = transforms AO → Löwdin orthonormal basis
S^{-1/2}  = transforms back to AO basis

Minimises the total rotation from the original AO basis.
Each Löwdin orbital looks as much as possible like its parent AO.
```

**Why not Gram-Schmidt?**
Gram-Schmidt is sequential and basis-order-dependent. Löwdin is
symmetric — all orbitals treated equally. Critical for unambiguous
population analysis (Step 1) and Schmidt decomposition (Step 2).

---

## 9. DMET — Density Matrix Embedding Theory

**The core idea:**
Split the molecule into:
```
Impurity  = correlated region (Fe 3d → identified by ASF)
Bath      = environment orbitals entangled with impurity
Core      = everything else → traced out at mean-field level
```

Construct an EXACT (in principle) Hamiltonian in the small
impurity + bath space that reproduces the same physics as the
full molecule.

**One-shot DMET (used here):**
No self-consistent loop. Single Schmidt decomposition of the MP2
density matrix. Faster, sufficient for ground state energy.

**Why this is not an approximation in principle:**
If the bath is complete (all entangled orbitals included), the
DMET Hamiltonian has the SAME ground state energy as the full molecule.
The adaptive bath selection captures 99.9% of entanglement (sv² coverage).

---

## 10. Schmidt Decomposition — Finding the Bath

**Setup:**
Partition the Hilbert space into impurity (A) and environment (B):
```
Ψ = Σ_k λ_k |ψ_k^A⟩ ⊗ |ψ_k^B⟩     (Schmidt decomposition)

λ_k = Schmidt coefficients (singular values of the reduced density matrix)
```

**In practice (density matrix approach):**
```
F = P_env @ ρ_lo @ Q_imp       off-diagonal DM block
                                (environment × impurity)

SVD:  F = U · diag(sv) · V†

sv_k large → environment orbital k is strongly entangled with impurity
             → MUST include as bath orbital
sv_k ≈ 0   → barely entangled → discard into core
```

**sv² coverage:**
```
Fraction of total entanglement captured = Σ_{kept} sv_k² / Σ_{all} sv_k²

The pipeline targets 99.9% coverage → quantifies bath completeness.
sv2_coverage = 1.0 means the bath perfectly represents the environment.
```

---

## 11. Core Mean-Field Potential

After removing the bath, remaining "core" electrons are not ignored —
they contribute an electrostatic field to the effective 1e Hamiltonian:

```
h1e_eff = h1e_bare + J_core - 0.5 × K_core

J_core[p,q] = Σ_{μν} ρ_core[μν] (pq|μν)   ← Coulomb repulsion
K_core[p,q] = Σ_{μν} ρ_core[μν] (pμ|νq)   ← Exchange interaction
                                               (same spin only, -½ factor)
```

**Physical picture:**
Core electrons create an electric field that shifts all orbital
energies in the embedding space. Without this correction, the
embedding Hamiltonian would be missing ~90% of the electrostatic
environment.

---

## 12. Chemical Potential Correction

One-shot DMET operates in the grand-canonical ensemble (μ not fixed).
This can give fractional electron counts in the embedding:

```
n_elec_emb = Tr[ρ_embedding]  might be 2.3 instead of 2
```

**Fix:** Shift the 1e Hamiltonian by −μI until n_elec = integer:
```
h1e → h1e - μ × I

Find μ via bisection on: N(μ) = Σ_{ε_i(μ) < 0} 2 = N_target
```

This enforces canonical ensemble (fixed N) — required by FCI and SQD.

---

## 13. Full Configuration Interaction (FCI) — The Exact Answer

**What it is:**
Diagonalise H in the COMPLETE basis of all electron configurations
(all Slater determinants). Exact within the basis set. No approximations.

```
H |Ψ_FCI⟩ = E_FCI |Ψ_FCI⟩

|Ψ_FCI⟩ = Σ_{all configs} c_I |I⟩

where |I⟩ = one way to assign N electrons to M orbitals
```

**Cost:** Scales as C(M, N/2)² determinants.
```
C(6,1)²  = 36        ← trivial, solved in milliseconds
C(16,8)² = 165M      ← borderline (~minutes)
C(20,10)² = 34B      ← impossible classically
```

**Role in the pipeline:**
FCI on the EMBEDDING Hamiltonian (not full molecule) is feasible and
serves as the exact reference target. SQD/SKQD aim to match E_FCI.

---

## 14. Jordan-Wigner Transform — Mapping Fermions to Qubits

**The problem:**
Quantum computers work with qubits (two-level systems).
Electrons are fermions — they obey the Pauli exclusion principle
and anti-commute: a_p a_q = -a_q a_p.
Qubits don't naturally anti-commute.

**Jordan-Wigner solution:**
Encode each spin-orbital as one qubit. Represent creation/annihilation
operators using Pauli matrices + string of Z operators:

```
a†_p = (Π_{j<p} Z_j) ⊗ |1⟩⟨0|_p = (Π_{j<p} Z_j) ⊗ ½(X_p - iY_p)
a_p  = (Π_{j<p} Z_j) ⊗ |0⟩⟨1|_p = (Π_{j<p} Z_j) ⊗ ½(X_p + iY_p)
```

The Z string enforces the anti-commutation relations.

**Result:**
The fermionic Hamiltonian H becomes a sum of Pauli strings:
```
H_qubit = Σ_k c_k P_k

where P_k ∈ {I, X, Y, Z}^⊗n   (tensor product of Pauli operators)
c_k = complex coefficient
```

For FeN6 embedding (12 qubits): 1819 Pauli terms.

**Bitstring interpretation:**
JW qubit state |0101...⟩ directly encodes orbital occupations:
qubit k in |1⟩ = orbital k is occupied. This is why bitstrings from
the quantum circuit are directly usable as electron configurations.

---

## 15. Time Evolution and Krylov Basis (SKQD)

**Time evolution operator:**
```
e^{-iHt} |ψ⟩ = time-evolved state after time t

In the energy eigenbasis: H|n⟩ = E_n|n⟩

e^{-iHt}|ψ⟩ = Σ_n c_n e^{-iE_n t} |n⟩

The ground state |0⟩ rotates slowly (low E_0).
Excited states rotate faster (higher E_n).
```

**Krylov basis:**
```
|ψ_0⟩ = |HF⟩                          initial state
|ψ_1⟩ = e^{-iH·dt}|HF⟩                after one time step
|ψ_2⟩ = e^{-2iH·dt}|HF⟩               after two time steps
...
|ψ_K⟩ = e^{-KiH·dt}|HF⟩

As K increases, the Krylov vectors span more of the low-energy
Hilbert space including increasingly more of the ground state.
Sampling bitstrings from many Krylov vectors gives systematic
coverage of the configurations needed to represent |ψ_0⟩.
```

**Why systematic convergence?**
Unlike random circuits, Krylov vectors are physically motivated.
The time evolution operator is the propagator of the Schrödinger
equation — it naturally explores the physically relevant subspace.

---

## 16. Trotter-Suzuki Decomposition

**Problem:**
e^{-iHt} with H = Σ_j H_j (sum of non-commuting terms) cannot be
implemented directly as a quantum circuit.

**Lie-Trotter product formula:**
```
e^{-iHt} ≈ [Π_j e^{-iH_j t/r}]^r

Each e^{-iH_j t/r} is a single Pauli rotation gate (one circuit layer).
r = SKQD_TROTTER_REPS

Error ∝ t²/r × Σ_{j<k} ||[H_j, H_k]||²

More reps → smaller error but deeper circuit.
```

**Trotter error vs circuit depth tradeoff:**
```
SKQD_TROTTER_REPS = 1   shallow circuits, Trotter error ~5%
SKQD_TROTTER_REPS = 2   2× deeper, Trotter error ~1%
SKQD_TROTTER_REPS = 4   4× deeper, Trotter error ~0.1%
```

For IBM hardware use reps=1. For MPS simulation reps=2 is fine.

---

## 17. SQD — Sampling-based Quantum Diagonalization

**Core insight:**
You don't need the FULL wavefunction to get the ground state energy.
You only need to know WHICH CONFIGURATIONS have significant weight.

```
|Ψ_ground⟩ = Σ_I c_I |I⟩    (full FCI expansion, exponentially many terms)

SQD approximation:
  Sample a set S of bitstrings from a quantum circuit
  Solve H restricted to the subspace {|I⟩ : I ∈ S}
  Energy is variational upper bound to true E_ground
  
  As |S| increases → E_SQD decreases → approaches E_FCI
```

**Configuration recovery:**
`recover_configurations` uses the occupation numbers from the current
ground state to flip bits and discover NEW configurations the circuit
missed. This expands the subspace iteratively without re-sampling.

**Why EfficientSU2 with random parameters?**
```
EfficientSU2 = alternating rotation layers + CNOT entangling layers

Random parameters → uniform distribution over accessible Hilbert space
                 → maximally DIVERSE set of bitstrings
                 → covers more configurations than any single optimal state

Goal: diversity of subspace, not quality of single state.
```

---

## 18. Particle Number Filtering

The random/time-evolution circuits explore the FULL Hilbert space
including states with wrong electron numbers:

```
|0101 0101⟩  2α + 2β electrons ← wrong if n_alpha=1, n_beta=1
|1000 0001⟩  1α + 1β electrons ← correct

Physical constraint: electron number is conserved.
Circuits don't automatically enforce this.
```

`filter_bitstrings` keeps only configurations with exactly
(n_alpha, n_beta) electrons. This is critical — without filtering,
`solve_fermion` would receive unphysical configurations.

---

## 19. Matrix Product States (MPS) — Tensor Network Simulation

**Problem with statevector simulation:**
Memory grows as 2^n_qubits. For 30 qubits → 8 GB. For 40 qubits → 8 TB.

**MPS representation:**
```
Instead of storing all 2^n amplitudes explicitly:
  Ψ[s_1, s_2, ..., s_n] = A_1[s_1] · A_2[s_2] · ... · A_n[s_n]

where s_i ∈ {0,1} is the state of qubit i
      A_i = matrix of dimension (χ × χ)  [χ = bond dimension]

Memory: O(n × χ² × 2)  instead of O(2^n)
```

**Bond dimension χ and entanglement:**
```
χ = 1      product state only (no entanglement)     exact
χ = 32     low entanglement (shallow circuits)       fast
χ = 256    moderate entanglement                     balanced
χ → ∞     any state (full statevector)              exact

Trotter circuits with many CNOT layers build up entanglement.
When entanglement exceeds what χ can represent, MPS truncates
(discards small singular values) → introduces approximation.
```

**When MPS is accurate for SKQD:**
The SKQD time evolution circuits are structured (Trotter) but the
Krylov vectors at small k are close to the HF reference (low entanglement).
MPS is accurate for early Krylov vectors (k < 5) and approximate
for later ones (k approaching K). The circuit depth output tells you how
entangled the circuit is — very deep circuits (depth > 100,000) indicate
significant entanglement growth.

---

## 20. IBM Quantum Hardware — Real vs Simulated

**What real quantum hardware gives you:**
```
Advantage: actual quantum gate operations (not classical simulation)
           can in principle access exponentially large Hilbert spaces

Reality (NISQ era):
  Gate fidelity:    99.0-99.9% per 2-qubit gate
  Decoherence:      T1 ~100 μs, T2 ~100-300 μs
  Circuit budget:   ~1000-3000 gate layers before noise dominates
  Queue time:       minutes to hours depending on load
```

**Open plan (free tier) restrictions:**
```
Session mode  ← requires dedicated QPU access, paid plans only
Batch mode    ← requires dedicated QPU access, paid plans only
Direct mode   ← available on all plans ✓ (used by this pipeline)
```

**Circuit depth problem for SKQD:**
```
FeN6 SKQD circuit (k=9, reps=2):
  1819 Pauli terms × 2 Trotter reps × 9 steps = 32,742 operator layers
  After transpilation to hardware native gates: ~1,074,658 gate layers

This is ~300× beyond the hardware decoherence budget.
Output would be completely dominated by noise.

Solution: SKQD_TROTTER_REPS=1, SKQD_KRYLOV_DIM=4
  1819 × 1 × 3 = 5,457 operator layers → ~8,000-15,000 after transpile
  Still above budget for free plan hardware.
  
For IBM: SQD (not SKQD) is currently more practical — 
  single EfficientSU2 circuit is much shallower.
```

---

## 21. Variational Principle — Why Energies Are Always Upper Bounds

All quantum solvers (SQD, SKQD, FCI) exploit this:

```
For any normalised trial state |Φ⟩:
  ⟨Φ|H|Φ⟩ ≥ E_0    (ground state energy)

with equality iff |Φ⟩ = |Ψ_ground⟩

Therefore:
  E_FCI ≤ E_SQD ≤ E_SKQD ≤ E_HF

Every method gives an upper bound.
Better method = tighter upper bound = closer to E_FCI.
```

**Practical implication:**
If E_SQD > E_FCI, the solver is fine — the subspace just needs to be
expanded (more shots, more Krylov vectors, more iterations).
E_SQD can NEVER go below E_FCI (it's exact in the embedding space).

---

## 22. ⟨S²⟩ — Spin Purity Check

After solving for the ground state:
```
⟨S²⟩ = S(S+1)

Singlet (S=0):  ⟨S²⟩ = 0.000  ← target for closed-shell molecules
Doublet (S=½): ⟨S²⟩ = 0.750
Triplet (S=1):  ⟨S²⟩ = 2.000

If ⟨S²⟩ ≈ 0 after SQD → ground state is a clean singlet ✓
If ⟨S²⟩ ≫ 0 → spin contamination → try open_shell=True in solve_fermion
```

For FeN6 with 1α + 1β electrons, ⟨S²⟩ = 0 confirms the
quantum solver found the singlet ground state correctly.

---

## 23. Pipeline Score Vector — Connecting to ML

The final `pipeline_score` dict aggregates metrics across all three
steps for a graph transformer machine learning model:

```
Input molecule → [Step 1, 2, 3] → pipeline_score → ML model

The ML model learns to predict:

  - Which molecules are hardest for classical methods
  - Which embedding size is needed

  - Which quantum solver will converge fastest

Key predictive features:

correlation_strength   mean dev of active orbitals
                       → 0 = classical OK, ~1 = needs QC

sv2_coverage           completeness of DMET bath
                       → 1.0 = physically correct embedding

quantum_error_vs_fci   how close SQD/SKQD got to exact
                       → training target for ML

embedding_corr_energy  how much correlation energy is in embedding
                       → larger magnitude = more quantum advantage
```

---

## Quick Reference — Method Hierarchy

```
Accuracy:  FCI > SKQD ≈ SQD > MP2 > HF
Cost:      FCI > SKQD > SQD > MP2 > HF

FCI:  exact within basis, exponential cost
SKQD: systematic convergence, polynomial circuit cost + classical diag
SQD:  random subspace, iterative recovery, practical for NISQ
MP2:  pair correlations only, misses higher-order effects
HF:   no correlation, fast, used as reference

Scaling:
  HF:   O(N^3)
  MP2:  O(N^5)
  FCI:  O(C(N,N/2)^2)  exponential
  SQD:  O(shots + subspace_size × classical FCI)
  SKQD: O(K × shots + subspace_size × classical FCI)
```

---

## Glossary

| Term | Definition |
|------|-----------|
| AO | Atomic orbital — basis function centred on an atom |
| MO | Molecular orbital — linear combination of AOs |
| NO | Natural orbital — eigenvector of density matrix |
| HF | Hartree-Fock — mean-field approximation |
| MP2 | Møller-Plesset 2nd order — first correlated correction |
| FCI | Full CI — exact diagonalisation in given orbital basis |
| DMET | Density Matrix Embedding Theory — fragment Hamiltonian method |
| SQD | Sampling-based Quantum Diagonalization |
| SKQD | Sampling-based Krylov Quantum Diagonalization |
| JW | Jordan-Wigner — fermion to qubit mapping |
| MPS | Matrix Product State — 1D tensor network |
| χ | Bond dimension — MPS accuracy parameter |
| ASF | Active Space Finder — identifies correlated orbitals |
| DMRG | Density Matrix Renormalisation Group — used inside ASF |
| 1-RDM | One-particle reduced density matrix |
| Trotter | Product formula approximation to time evolution |
| Krylov | Subspace spanned by repeated H application |
| NISQ | Noisy Intermediate-Scale Quantum (current hardware era) |