# QuEnAIS Pipeline — Complete Parameter Reference

---

## 1. Molecule & Geometry

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `MOLECULE` | `"FeN6"` | Key into the `geometries` dict. Switch to `"LiH"`, `"H2O"`, `"N2"`, `"H6"`, or add your own entry. |
| `geometries` | dict | Cartesian coordinates in Angstrom for each molecule. Add custom entries as `("symbol", (x, y, z))` tuples. |
| `BASIS` | `"sto-3g"` | PySCF basis set string. `"sto-3g"` is minimal/fast. `"cc-pVDZ"` is more accurate but ~10× more AOs → slower integrals. |

---

## 2. Complexity Classification Thresholds

These decide which ASF parameter tier (1/2/3) is used.

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `TM_ELEMENTS` | large set | Any atom in this set forces Tier 3 (DMRG-level treatment). Add custom elements if needed. |
| `SPIN_CONTAMINATION_TIER2_THRESHOLD` | `1.3` | UHF ⟨S²⟩_actual / ⟨S²⟩_expected ratio. Above this → Tier 2+. Lower value = more molecules classified as open-shell. Range: 1.0 (strictest) – 2.0 (lenient). |
| `HOMO_LUMO_TIER2_THRESHOLD_EV` | `1.0` | Gap in eV below which a molecule is considered near-degenerate → Tier 2+. Raise to 2.0 to catch more weakly correlated systems. |

---

## 3. ASF (Active Space Finder) Parameters per Tier

Controls the candidate orbital pool returned by `find_from_scf`.

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `entropy_threshold` | `0.01` | Lower = broader candidate pool (more orbitals considered). Raise to 0.1 to get a tighter pool and trust ASF more. Lower to 0.001 if ASF returns 0 candidates. |
| `max_norb` (Tier 1/2/3) | `12 / 14 / 16` | Hard cap on the number of candidate orbitals returned by ASF. Raise for strongly correlated systems if gap detection hits the ceiling. Directly controls qubit count upper bound. |
| `min_norb` | `2` | Minimum orbitals ASF must return. Raise to 4 if the active space is consistently too small. |

---

## 4. Adaptive Gap Detection

After ASF returns candidates, gap detection selects the final active space.

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `GAP_MIN_NORB` | `2` | Minimum orbitals to keep after gap detection, regardless of where the gap falls. Raise to 4 for TM systems where 2 orbitals is physically unrealistic. |
| `GAP_MAX_NORB` | `8` | Maximum orbitals kept. Controls qubit ceiling: qubits ≤ 2 × (GAP_MAX_NORB + n_bath). Lower to 6 to cut qubit count; raise to 12 for more accuracy. |
| `CORE_OCC_THRESHOLD` | `1.8` | Natural orbital occupation above this is treated as a doubly-occupied core (not active). Lower to 1.7 to include borderline orbitals in the active space. |

---

## 5. DMET Embedding

Controls how the environment is compressed into bath orbitals.

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `BATH_TOLERANCE` | `1e-8` | Schmidt singular value cutoff. Values below this are discarded as bath orbitals. Raise to 1e-4 to reduce n_bath (fewer qubits, less accurate). Lower to 1e-12 for near-exact bath (more qubits). |
| `MAX_EMBED_ORBS` | `16` | Hard cap: n_imp + n_bath ≤ this. Controls max qubit count = 2 × MAX_EMBED_ORBS. Reduce to 10 for faster runs; raise to 20 for higher accuracy. |

---

## 6. Execution Backend

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `BACKEND` | `"ibm"` | `"local"` = exact statevector (≤ 24 qubits, instant). `"mps"` = tensor network Aer simulator (handles 20–50 qubits). `"ibm"` = real hardware (requires credentials, queue time). |
| `MPS_MAX_BOND_DIM` | `256` | MPS accuracy vs cost tradeoff. `32` = fast/rough. `256` = balanced. `512` = near-exact but slow for >30 qubits. Double and check if energy changes to test convergence. |
| `MPS_TRUNC_THRESH` | `1e-6` | Singular values below this are dropped during MPS contraction. Lower to `1e-10` for near-exact; raise to `1e-4` for speed. |
| `IBM_BACKEND_NAME` | `None` | `None` = auto-select least busy device with enough qubits. Set to `"ibm_brisbane"` etc. to pin to a specific device. |
| `IBM_OPTIMIZATION_LEVEL` | `1` | Qiskit transpiler optimization: `0` = no optimization (fastest compile). `1` = balanced (recommended). `2/3` = aggressive gate reduction (slow compile, lowest gate count). |
| `IBM_MAX_CIRCUIT_DEPTH` | `3000` | Hard rejection limit. Circuits deeper than this are refused before submission to prevent spending queue time on noise-dominated results. |

---

## 7. Quantum Solver Selection

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `QUANTUM_SOLVER` | `"skqd"` | `"sqd"` = random EfficientSU2 circuit. `"skqd"` = systematic Krylov time evolution. `"sqdrift"` = qDRIFT ensemble from Hamiltonian directly. |

---

## 8. SQD Parameters

Used when `QUANTUM_SOLVER = "sqd"`.

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `N_SHOTS` | `8192` | Bitstrings sampled from the circuit per iteration. Double to 16384 if valid configs < 20% of max. Diminishing returns above 65536. |
| `SQD_ITERS` | `10` | Number of recover_configurations → solve_fermion cycles. Raise to 20 if energy is still decreasing at iter 10. |
| `ANSATZ_REPS` | `3` | EfficientSU2 repetitions. More reps = more expressive circuit but deeper (more hardware noise). Range: 1 (shallow) – 5 (deep). |

---

## 9. SKQD Parameters

Used when `QUANTUM_SOLVER = "skqd"`.

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `SKQD_KRYLOV_DIM` | `10` | Number of Krylov vectors (time-evolved states). More = monotonically better energy but more circuits to execute. Start at 5 for IBM hardware to save queue time. |
| `SKQD_DT` | `0.5` | Time step per Krylov evolution in Ha⁻¹. Smaller = more accurate Trotter but less exploration per step. Larger = more mixing but more Trotter error. Typical range: 0.2 – 2.0. |
| `SKQD_TROTTER_REPS` | `2` | Trotter product formula repetitions per time step. Higher = lower Trotter error but linearly deeper circuit. Set to 1 for IBM hardware to halve circuit depth. |
| `SKQD_SHOTS` | `8192` | Shots per Krylov circuit. Raise to 16384 if valid configs are low at early Krylov steps. |

---

## 10. SqDRIFT Parameters

Used when `QUANTUM_SOLVER = "sqdrift"`.

| Parameter | Default | Recommended fix | What it does |
|-----------|---------|-----------------|--------------|
| `SQDRIFT_NUM_CIRCUITS` | `10` | `100` | Size of the randomized circuit ensemble. Each circuit samples different Hamiltonian groups. More circuits = broader Hilbert space coverage. |
| `SQDRIFT_NUM_GROUPS` | `10` | `300` | Number of excitation groups subsampled per circuit by qDRIFT. **Critical**: must be a meaningful fraction of total groups. If ASF returns 2221 groups, set this to at least 200–500. Too small = circuits barely mix the state = only 3 configs found. |
| `SQDRIFT_TIME` | `1.0` | `3.0` | Time scaling factor for the Evolution gate. Larger = more state mixing per circuit = more diverse bitstrings. Too large can cause Trotter aliasing. Range: 0.5 – 5.0. |
| `SQDRIFT_ITERS` | `10` | `15` | recover_configurations → solve_fermion iterations after sampling. Raise if energy still decreasing at last iter. |
| `SQDRIFT_SHOTS` | `8192` | `16384` | Shots per circuit. Multiply by NUM_CIRCUITS to get total budget. Total shots should be ≥ 10× the number of valid configurations needed. |

---

## 11. Diagnostic Rules of Thumb