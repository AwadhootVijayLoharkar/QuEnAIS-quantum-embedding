# step1_asf.py — Active Space Finder

## Purpose
Identify the minimal set of molecular orbitals that captures the
chemically relevant electron correlation. These become the impurity
orbitals passed to DMET in Step 2.

---

## Why Active Space Selection Matters

Full-molecule quantum computation is intractable:
- FeN6 in STO-3G has 48 AOs → 96 spin-orbitals → 2^96 Hilbert space
- Active space reduces this to 6-16 orbitals → manageable for QC

The challenge: which orbitals to pick?
- Too few → miss important correlation
- Too many → exceed qubit budget
- Wrong ones → wrong chemistry

---

## Five-Phase Pipeline

### Phase A — UHF + Classification

```
mol (geometry + basis)
    │
    ▼
UHF (DIIS + level_shift=0.5)
    │ not converged?
    ▼
Newton second-order solver (warm-started from DIIS MOs)
    │
    ▼
classify_molecule()
    ├── has d/f-block element?  → Tier 3
    ├── spin contamination > 1.3? → Tier 2+
    └── HOMO-LUMO gap < 1.0 eV?  → Tier 2+
```

The Newton fallback is critical for Fe complexes where the open 3d
shell causes DIIS oscillations. Without it, UHF may not converge.

**Spin contamination ratio:**
```
ratio = ⟨S²⟩_actual / ⟨S²⟩_expected
1.0 = pure spin state (ideal)
1.3 = 30% contamination → significant open-shell character → Tier 2+
```

---

### Phase B — MP2 Deviation Proxy

**Natural orbital deviation:**
```
dev_i = min(n_i, 2 - n_i)

where n_i = MP2 natural orbital occupation

dev = 0.0  →  always full (n=2) or always empty (n=0)  →  uncorrelated
dev = 1.0  →  half-filled (n=1)                         →  maximally correlated
```

This proxy avoids expensive CASSCF or DMRG for initial screening.
MP2 captures pair correlations that UHF misses, giving a better picture
of which orbitals fluctuate in occupation.

**Fallback:** If MP2 fails (rare), uses UHF DM. SOMOs appear as
fractionally occupied in spin-averaged DM → proxy still useful.

---

### Phase C — Adaptive Gap Detection

ASF returns a broad candidate pool (entropy_threshold=0.01).
Gap detection then finds the natural cutoff:

```
Sort candidates by deviation (descending)
For n = GAP_MIN_NORB to GAP_MAX_NORB:
    gap_n = dev[n-1] - dev[n]
Select n* = argmax(gap_n)
```

**Why this works:**
Correlated orbitals cluster with high deviation. A gap in the spectrum
marks the natural boundary between the correlated group and the
essentially-uncorrelated rest.

No fixed threshold needed — works for any molecule automatically.

Example deviation spectrum:
```
MO 32:  0.9890  ████████████████████
MO 31:  0.9820  ███████████████████
MO 30:  0.9750  ███████████████████
         ← GAP 0.0289 ← selected cutoff
MO 29:  0.9461  ██████████████████
MO 28:  0.0312  ▌
MO 27:  0.0201  ▌
```

---

### Phase D — Löwdin Population Analysis

Maps each active MO to its dominant atom:

```
c_lo = S^{1/2} @ c_AO              (Löwdin-orthogonalized coefficients)
weight[orbital, atom] = Σ_{μ∈atom} c_lo[μ]²
```

**Why Löwdin not Mulliken:**

| Property | Mulliken | Löwdin |
|----------|----------|--------|
| Always non-negative | ✗ | ✓ |
| Basis-set stable | ✗ | ✓ |
| Physical interpretation | Poor | Good |

Outputs: which atom each active orbital belongs to, metal fraction.

---

### Phase E — Molecular Score Vector

Computes 15+ metrics stored for the graph transformer pipeline:

| Metric | Physical meaning |
|--------|-----------------|
| `complexity_class` | Tier 1/2/3 classification |
| `correlation_strength` | Mean deviation of active orbitals |
| `max_correlation` | Highest single-orbital deviation |
| `entropy_gap` | Clarity of the active space boundary |
| `n_strongly_correlated` | Orbitals with dev > 0.3 |
| `metal_fraction` | Fraction of active orbs on TM atom(s) |
| `homo_lumo_gap_eV` | Electronic gap in eV |
| `mp2_correlation_energy` | Total MP2 correlation |

---

## Output: step1_asf.pkl

| Key | Shape/Type | Description |
|-----|-----------|-------------|
| `nel` | int | Active electrons (always even, singlet) |
| `mo_list` | list[int] | Column indices of active MOs |
| `mo_coeff` | (n_AO, n_MO) | Full MP2 NO coefficient matrix |
| `n_active_orbs` | int | Number of active orbitals selected |
| `no_occ` | (n_MO,) | All NO occupations, sorted descending |
| `deviation` | (n_MO,) | All deviations, same ordering |
| `lowdin_weights` | (n_active, n_atoms) | Atom weights per active MO |
| `scores` | dict | Full score vector |
| `mol_info` | dict | Molecule metadata |

---

## Electron Counting Method

```python
nel = mol.nelectron - 2 * n_core

n_core = orbitals NOT in active list with NO occupation > CORE_OCC_THRESHOLD (1.8)
```

Cross-checked against sum of NO occupations in active space.
If mismatch > 3 electrons, uses occupation sum as fallback.
Always clamped to even number (singlet target for SQD).

---

## Key Tunable Parameters (config.py)

```python
GAP_MIN_NORB = 2           # never select fewer than this
GAP_MAX_NORB = 8           # never select more than this
CORE_OCC_THRESHOLD = 1.8   # above this → doubly occupied core
ASF_PARAMS[3]["max_norb"] = 16  # Tier 3 candidate pool size
```