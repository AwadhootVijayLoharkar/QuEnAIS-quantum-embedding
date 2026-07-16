# Wiring DMET into gqe-for-qsci — integration guide

Four new files (built here, need to land inside your `gqe-for-qsci`
checkout) plus one small edit to `factory.py`. Read in this order.

## 0. Files delivered

| File | Goes where |
|---|---|
| `dmet_molecule_adapter.py` | `gqe-for-qsci/dmet_molecule_adapter.py` (repo root, next to `train.py`) |
| `dmet_excitation_pool.py` | `gqe-for-qsci/dmet_excitation_pool.py` (repo root) |
| `validate_excitation_generator.py` | `gqe-for-qsci/validate_excitation_generator.py` (repo root) |
| `dmet_embedding.yaml` | `gqe-for-qsci/configs/molecule/dmet_embedding.yaml` |

Kept at repo root rather than inside `gqe_qsci/` on purpose — these are
your integration code, not upstream gqe-for-qsci code, so keeping them
separate makes future `git submodule update` pulls from upstream safe
(same reasoning as the `sampler.py` fix note: don't let your work live
somewhere upstream changes could silently clobber it — except here it's
even better, since these are *new* files upstream will never touch).

## 1. Validate first — mandatory, not optional

I could not run tequila/openfermion/cudaq myself (none available in my
sandbox) to confirm `dmet_excitation_pool.py`'s hand-built excitation
generator matches tequila's exact sign/normalization convention. Both use
standard, well-documented Jordan-Wigner conventions, so I'm fairly
confident, but "fairly confident" isn't good enough for a physics result —
a sign error here wouldn't crash anything, it would just silently seed
the operator pool with wrong-phase excitations.

```bash
cd gqe-for-qsci
python validate_excitation_generator.py
```

Read its output carefully. Three possible outcomes:
- **PASS, ratio = 1.0 for every case** — proceed, no changes needed.
- **PASS, but a consistent non-1.0 ratio** (e.g. always -1.0) — the script
  will tell you exactly what to do: multiply
  `excitation_generator_qubit_op()`'s return value by that constant in
  `dmet_excitation_pool.py`. One-line fix.
- **FAIL** (inconsistent ratios, or mismatched Pauli-string sets) — stop,
  don't run this on real DMET data yet, paste me the full output and I'll
  fix the generator construction.

## 2. `factory.py` patch

`create_operator_pool()` currently hardcodes a `match` on
`cfg.operator_pool.spec` pointing only at the original tequila-based
pools. Add two new cases:

```python
# add near the top, with the other imports
from dmet_excitation_pool import DMETPauliEvolutionPool, DMETExcitationPool

...

    def create_operator_pool(self, cfg):
        if self.operator_pool is not None:
            return self.operator_pool
        molecule = self.create_molecule(cfg)
        match cfg.operator_pool.spec:
            case "pauli_evolution":
                operator_pool = PauliEvolutionPool(
                    molecule,
                    params=cfg.operator_pool.params,
                    threshold=cfg.operator_pool.ccsd_threshold,
                    remove_z_ladder=cfg.operator_pool.remove_z_ladder,
                    only_use_first_pauli=cfg.operator_pool.only_use_first_pauli
                )
            case "excitation":
                operator_pool = ExcitationPool(molecule, params=cfg.operator_pool.params, threshold=cfg.operator_pool.ccsd_threshold)
            case "dmet_pauli_evolution":
                operator_pool = DMETPauliEvolutionPool(
                    molecule,
                    params=cfg.operator_pool.params,
                    threshold=cfg.operator_pool.ccsd_threshold,
                    remove_z_ladder=cfg.operator_pool.remove_z_ladder,
                    only_use_first_pauli=cfg.operator_pool.only_use_first_pauli
                )
            case "dmet_excitation":
                operator_pool = DMETExcitationPool(molecule, params=cfg.operator_pool.params, threshold=cfg.operator_pool.ccsd_threshold)
            case _:
                raise ValueError(f"Unknown operator pool specification: {cfg.operator_pool.spec}")
        self.operator_pool = operator_pool
        return operator_pool
```

That's the only code change needed outside the new files — `create_molecule`
already works unchanged, since it just does `instantiate(cfg.molecule)`,
and hydra's `instantiate` happily calls a plain function (`load_from_dmet_pickle`)
just as well as a class constructor.

## 3. Point a config at your real DMET output

Edit `configs/molecule/dmet_embedding.yaml`'s `step2_pickle_path` to your
actual DMET output path, or override on the command line:

```bash
python train.py molecule=dmet_embedding \
  molecule.step2_pickle_path=/path/to/results/step2_hamiltonian.pkl \
  operator_pool.spec=dmet_excitation \
  trainer.max_iters=30
```

Use `operator_pool.spec=dmet_excitation` or `operator_pool.spec=dmet_pauli_evolution`
— NOT the plain `pauli_evolution`/`excitation` values, those still point
at the tequila-based pools which will build gates for the wrong (real,
non-DMET) active space if pointed at a `DMETEmbeddingMolecule`.

## 4. One practical limit to know about before trying a big system

`reference_keys: ["R-CASCI", "R-CCSD"]` in `default.yaml` makes the
pipeline compute a full FCI (`compute_casci()`) over the *entire*
embedding space just for logging/comparison purposes. FCI cost scales
combinatorially with embedding size — fine up to roughly 12-16 embedding
orbitals, intractable well before DMET's `MAX_EMBED_ORBS` ceiling on a
larger system. If a run hangs or blows up memory at the
`create_wandb_logger` step (before any GQE training even starts), drop
`"R-CASCI"` from `reference_keys` — it's a reference/logging value only,
not used anywhere in the actual training or diagonalization.

## 5. Test plan

1. `validate_excitation_generator.py` — must pass first (step 1 above).
2. Run on a DMET output from a **small** system you already trust (N2 is
   fine, or whatever gave you the flat-SQD-energy result originally) with
   `operator_pool.spec=dmet_excitation`.
3. Compare against SQD's result on the *same* DMET output, and against
   this integration's own `compute_casci()` reference (should match DMET's
   own internal HF self-consistency check to reasonable precision, modulo
   the note in `dmet_molecule_adapter.py` about my adapter's fresh SCF
   possibly finding a different-but-equivalent HF solution than DMET's
   internal naive-aufbau reference — the *total* energy accounting stays
   correct either way, since `ecore` is a fixed additive constant
   independent of which reference state anything downstream uses).
4. Only then move to TiO2 / larger systems, watching the FCI-cost caveat
   in step 4.

## 6. Status

**Step 1 (excitation generator validation): PASSED**, confirmed on your
HPC — singles (both spins) and doubles all matched tequila's actual
output exactly, ratio 1.0 on every Pauli term. One real bug was found and
fixed along the way: the raw anti-Hermitian generator differed from
tequila's stored (Hermitian) convention by a consistent factor of `-i`,
now baked into `excitation_generator_qubit_op()` in
`dmet_excitation_pool.py`. That was the single highest-risk piece of this
integration (the one part I couldn't test myself, no cudaq/tequila/
openfermion/pyscf available in my sandbox) — it's now empirically
confirmed, not just reasoned through.

`dmet_molecule_adapter.py`'s PySCF "custom Hamiltonian" construction and
the `factory.py` wiring are still unverified end-to-end — they're built
from standard, well-documented PySCF patterns, but haven't been run
against real DMET output yet. That's the next step (section 5, test plan).