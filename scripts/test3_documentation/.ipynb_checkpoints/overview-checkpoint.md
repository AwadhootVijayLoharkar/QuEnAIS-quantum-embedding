# QuEnAIS — Quantum Embedding AI Solver

## Overview
QuEnAIS is a three-step quantum chemistry pipeline that combines classical
embedding theory with quantum computing to solve strongly correlated molecular
systems. It targets transition-metal complexes (e.g. FeN6) where standard
DFT/HF methods fail.

## Architecture

```
Molecule Geometry
      │
      ▼
┌─────────────┐
│   Step 1    │  Active Space Finder (ASF)
│  step1_asf  │  UHF → MP2 NOs → gap detection → impurity orbitals
└──────┬──────┘
       │  nel, mo_list, mo_coeff
       ▼
┌─────────────┐
│   Step 2    │  DMET Embedding
│  step2_ham  │  Schmidt decomp → bath → h1e, h2e in embedding space
└──────┬──────┘
       │  h1e, h2e, n_emb, n_alpha, n_beta
       ▼
┌─────────────┐
│   Step 3    │  Quantum Solver
│  step3_sqd  │  SQD or SKQD on local / MPS / IBM backend
└──────┬──────┘
       │  energy, error_vs_fci, pipeline_score
       ▼
  results/step3_results.pkl
```

## Why This Approach

| Problem | Solution |
|---------|----------|
| Full molecule too large for QC | DMET reduces to small embedding (≤16 orbitals) |
| Random circuits miss configurations | SKQD uses systematic Krylov basis |
| Local simulator limited to ~28 qubits | MPS tensor network handles larger circuits |
| IBM free plan blocks Session mode | Direct job mode used instead |

## File Structure

```
project/
├── config.py            ← all tuneable parameters
├── step1_asf.py         ← classical active space selection
├── step2_hamiltonian.py ← classical DMET embedding
├── step3_sqd.py         ← quantum solver (SQD / SKQD)
└── results/
    ├── step1_asf.pkl
    ├── step2_hamiltonian.pkl
    └── step3_results.pkl
```

## Dependencies

```
pyscf              classical quantum chemistry (UHF, MP2, FCI)
qiskit             circuit construction
qiskit-aer         MPS tensor network simulator
qiskit-ibm-runtime IBM Quantum hardware access
qiskit-addon-sqd   SQD fermion solver
openfermion        Jordan-Wigner transform (SKQD only)
asf                active space finder (block2 DMRG internally)
```

## Execution Flow

```bash
python step1_asf.py          # ~30 sec – 5 min depending on molecule tier
python step2_hamiltonian.py  # ~1-3 min
python step3_sqd.py          # ~5-30 min depending on backend + solver
```

## Backend Selection

| Backend | Use Case | Max Qubits | Cost |
|---------|----------|------------|------|
| `local` | Testing, debugging | ~28 | Free, instant |
| `mps`   | Large circuits, no queue | ~50+ | Free, local CPU |
| `ibm`   | Production on real hardware | 127 | Queue + credits |

## Solver Selection

| Solver | Circuit Strategy | Advantage |
|--------|-----------------|-----------|
| `sqd`  | One random EfficientSU2 | No Hamiltonian qubit encoding needed |
| `skqd` | Krylov basis via time evolution | Systematic convergence to ground state |

## Output Score Vector (step3_results.pkl → pipeline_score)

Used downstream by the graph transformer pipeline:

```
complexity_class       molecule difficulty tier (1/2/3)
correlation_strength   mean MP2 deviation of active orbitals
sv2_coverage           fraction of entanglement captured by bath
quantum_error_vs_fci   |E_solver - E_FCI| in Hartree
quantum_spin_sq        ⟨S²⟩ of final state (0 = clean singlet)
embedding_corr_energy  FCI - UHF correlation in embedding space
```