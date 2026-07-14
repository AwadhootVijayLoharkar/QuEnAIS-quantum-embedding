# GQE-for-QSCI Setup Guide (HPC / Singularity / mamba)

Reference for installing and running `gqe-for-qsci` inside an existing mamba
environment, as a git submodule, on an HPC cluster without Docker. Covers
every issue hit during setup and training, with root causes and fixes, so
none of this has to be re-derived from scratch next time.

Target layout: your own pipeline repo has a mamba env with your existing
packages already installed. `pyci` and `gqe-for-qsci` are added as **git
submodules** (not clones, not vendored copies) specifically so future
upstream updates can be pulled cleanly without merge conflicts.

---

## Prerequisites

- An existing mamba/conda environment with your own pipeline already installed.
- HPC cluster with Singularity/Apptainer (no Docker) — this guide replaces
  `gqe-for-qsci`'s `dockerfile` with a plain pip/source-build install into
  your existing env.
- Compiler toolchain + MPI available (`gcc`, `make`, `mpicc`).
- Read the **Hardware architecture considerations** section before installing
  on a new node — CPU and GPU architecture both matter here.

---

## Step 1 — `pyci` as a submodule (source build, not PyPI)

`pyci` here means **theochem/pyci**. This is *not* the same as the `pyci` or
`qc-PyCI` packages on PyPI — those are unrelated projects. `theochem/pyci`
has to be built from source.

```bash
cd /path/to/your/pipeline/repo
git submodule add https://github.com/theochem/pyci.git external/pyci
git submodule update --init --recursive
cd external/pyci
make
pip install .
```

---

## Step 2 — `gqe-for-qsci` as a submodule

```bash
cd /path/to/your/pipeline/repo
git submodule add https://github.com/<upstream-org>/gqe-for-qsci.git external/gqe-for-qsci
git submodule update --init --recursive
cd external/gqe-for-qsci
```

Since this env already has many of your own pinned packages installed,
install `gqe-for-qsci` **without letting it clobber your existing pins**:

```bash
pip install --no-deps -e .
```

Then install only the packages it needs that you don't already have
(check `pyproject.toml` for the full pinned list): `torch`, `cudaq`,
`pytorch-lightning`, `hydra-core`, `wandb`, `tequila-basic`, `pyscf`,
`mpi4py`, etc. Install these selectively rather than letting pip resolve
the whole dependency tree — that's what causes the conflicts in Step 3.

---

## Step 3 — Dependency conflict triage

Run `pip check` after installing. Two conflicts are known and **safe to
ignore** in this setup:

- **qiskit**: `gqe-for-qsci` pins `qiskit==2.0.0`, but your pipeline needs
  `qiskit==2.4.0` for `qiskit-fermions`/`qiskit-ibm-runtime`/etc. Verified via
  `grep -r "import qiskit" gqe_qsci/` that `gqe-for-qsci`'s actual code never
  imports qiskit — the pin is unused. **Keep qiskit at 2.4.0**, do not
  downgrade. (If you already downgraded: `pip install "qiskit==2.4.0"` to
  revert.)
- **scipy**: `gqe-for-qsci` wants `scipy~=1.15.3`, conda-forge has `1.15.2`.
  Patch-level gap only (bugfix releases), safe to ignore. 1.15.3 isn't
  available on conda-forge for this setup anyway.

Any *other* `pip check` conflicts should be investigated individually — don't
assume they're all this benign.

---

## Step 4 — `pkg_resources` / setuptools fix

Symptom:
```
ModuleNotFoundError: No module named 'pkg_resources'
```
Cause: setuptools ≥82.0.0 (Feb 2026) removed `pkg_resources` entirely;
`tequila`/other deps still import it.

Fix:
```bash
pip install "setuptools<82"
```
A residual `UserWarning` about `pkg_resources` being deprecated is expected
and harmless after this.

---

## Step 5 — Hydra config key

Use `trainer.max_iters`, **not** `trainer.epochs` — the latter doesn't exist
in this config schema and Hydra will refuse to override it in struct mode.

```bash
python train.py molecule=n2 trainer.max_iters=2
```

---

## Step 6 — wandb non-interactive mode

Symptom:
```
wandb.errors.errors.UsageError: No API key configured
```
Happens in any non-interactive context (piped through `tee`, run under
`gdb`, batch/sbatch jobs) where wandb's interactive login prompt can't
display.

Fix — set before invocation:
```bash
export WANDB_MODE=offline
```
Use `WANDB_MODE=disabled` instead if you want to skip wandb entirely (useful
for quick debug runs — confirmed this does **not** affect training behavior
or mask/cause any errors; it's a clean way to rule wandb in or out when
debugging).

---

## Step 7 — Custom MPI plugin activation

Symptom:
```
RuntimeError: Unable to open distributed interface library
'...libcudaq_distributed_interface_mpi.so'
```
Cause: CUDA-Q's MPI plugin isn't prebuilt — it must be compiled locally,
once, per environment.

Fix:
```bash
source $CONDA_PREFIX/lib/python3.11/site-packages/distributed_interfaces/activate_custom_mpi.sh
conda env config vars set -p <env_path> LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
conda env config vars set -p <env_path> MPI_PATH="$CONDA_PREFIX"
mamba deactivate
mamba activate <env_name>
```
`conda env config vars set` persists these vars for the environment so they
survive future activations — don't rely on a one-off `export`.

Verify mpi4py and the plugin agree on the same MPI build (worth re-checking
if you ever touch MPI/conda packages):
```bash
python -c "from mpi4py import MPI; print(MPI.__file__)"
ldd $(python -c "from mpi4py import MPI; print(MPI.__file__)") | grep -i libmpi
ldd "$CONDA_PREFIX/lib/python3.11/site-packages/distributed_interfaces/libcudaq_distributed_interface_mpi.so" | grep -i libmpi
```
Both should resolve to the identical `libmpi.so.N` file inside the conda
env. If they diverge (one from `/usr/lib`, one from the env), that's a real
ABI mismatch worth fixing before debugging anything else.

---

## Step 8 — CUDA-Q target selection

Symptom: `RuntimeError: architecture mismatch` on GPUs cuQuantum doesn't
support (see hardware table below).

Fix — force the CPU simulator target, **must be set before Python starts**
(env var read at interpreter/module init, too late to set inside the script):
```bash
export CUDAQ_DEFAULT_SIMULATOR=qpp-cpu
```

---

## Step 9 — `sampler.py` MPI/pickle bug (resolved)

Symptom:
```
TypeError: cannot pickle 'cudaq.mlir._mlir_libs._quakeDialects.cudaq_runtime.SampleResult' object
```
This one took a long diagnostic chain to pin down (checkpointing, wandb, the
DataLoader, and a real MPI ABI mismatch were all ruled out along the way —
none of those were the cause). The eventual full traceback (obtained by
manually walking `exception.__traceback__` instead of relying on
`traceback.print_exc()`, which was itself silently failing) pointed at:

```
gqe_qsci/gqe/sampler.py:70, in run
  res = MPI.COMM_WORLD.allgather(res)
```

**Root cause**: `sampler.py` imports `from mpi4py import MPI` at module load
time. mpi4py automatically calls `MPI_Init()` on import — this happens
regardless of whether you actually intend to use MPI. Because mpi4py and
CUDA-Q's custom distributed-interface plugin link against the *same*
`libmpi.so` in this environment, that auto-init flips the shared MPI
runtime's global "initialized" state to `True`, even for a plain single-
process `python train.py ...` run with no `mpirun`/`srun --mpi` involved.

`Sampler.run()` checked `cudaq.mpi.is_initialized()` on its own, without also
checking `self.mpi` (the config flag — `sampler.mpi: false` — that's
actually supposed to gate MPI-distributed sampling). So it wrongly took the
"gather results across all MPI ranks" branch and called
`MPI.COMM_WORLD.allgather(res)` on `res`, a plain list of raw
`cudaq.SampleResult` objects. Those objects have no pickle support (a known,
still-open upstream CUDA-Q issue —
[NVIDIA/cuda-quantum#1422](https://github.com/NVIDIA/cuda-quantum/issues/1422)),
and `allgather` needs to pickle whatever it's given, even in a size-1
gather. That's the crash.

**Fix** — in `gqe_qsci/gqe/sampler.py`, `Sampler.run()`, guard both
`cudaq.mpi.is_initialized()` checks with `self.mpi`:

```python
        if self.mpi and cudaq.mpi.is_initialized():
            ...
        else:
            ...

        if self.mpi and isinstance(res[0], tuple) and len(res[0]) == 2:
            ...

        if self.mpi and cudaq.mpi.is_initialized():
            res = MPI.COMM_WORLD.allgather(res)
            res = [x for xs in res for x in xs]
```

Two lines change (adding `self.mpi and` in front of both
`cudaq.mpi.is_initialized()` checks); nothing else in the file changes.

**⚠️ This is a local patch to a third-party submodule.** `git submodule
update` on `gqe-for-qsci` will silently wipe this fix and reintroduce the
crash. Either:
- commit this fix to your own fork/branch of `gqe-for-qsci` and point the
  submodule at that fork, or
- keep a copy of this diff and manually reapply it after any submodule
  update, checking `sampler.py` lines ~41 and ~69 first.

---

## Hardware architecture considerations

### GPU: compute capability

| GPU | Architecture | Compute capability | cuQuantum/cuStateVec support |
|---|---|---|---|
| TITAN V | Volta | 7.0 | **Not supported** — `RuntimeError: architecture mismatch`. Workaround: `CUDAQ_DEFAULT_SIMULATOR=qpp-cpu` (Step 8). |
| A100 (SXM4-40GB) | Ampere | 8.0 | Fully supported, no issues. |

### CPU: AVX-512

PySCF ships a compiled `libcgto.so` with a **non-dispatching** AVX-512
codepath (unlike numpy/OpenBLAS, which do runtime CPU dispatch). On a CPU
without AVX-512, this crashes with `Illegal instruction (core dumped)`
(confirmed root cause via `gdb` backtrace on `GTOint2c()`).

AVX-512 support is per-chip, not per-vendor:

| CPU | Microarchitecture | AVX-512 | Result |
|---|---|---|---|
| AMD Threadripper 1920X | Zen 1 | No | SIGILL |
| AMD EPYC 7402 | Zen 2 | No | SIGILL (confirmed) |
| Intel Xeon Silver 4108 | Skylake-SP | Yes | No crash |
| AMD Zen 4+ (Genoa etc.) | Zen 4 | Yes | Should be fine (not directly tested here) |
| Intel 12th gen+ consumer (Alder Lake+) | — | Disabled on consumer chips | Would crash — check `/proc/cpuinfo` for `avx512` flag before assuming any given Intel/AMD chip has it |

**Fix — confirmed working**, rebuild PySCF from source targeting the actual
CPU instead of using the prebuilt wheel:
```bash
pip install pyscf==2.11.0 --no-binary pyscf --force-reinstall --no-deps
```
Check `AVX-512` support on any new node before running anything:
```bash
grep -o 'avx512[a-z]*' /proc/cpuinfo | sort -u
```
If that returns nothing, expect the SIGILL and rebuild pyscf first.

**Reinstalling pyscf — dependency safety check**: if you also have
`pyscf-dmrgscf` and `openfermionpyscf` installed (from PyPI, while pyscf
itself may be from conda-forge), reinstalling pyscf via `--no-binary
--force-reinstall --no-deps` does **not** touch either of those — `--no-deps`
means pip won't reinstall/touch anything else, and neither `dmrgscf` nor
`openfermionpyscf` pin an exact pyscf version tightly enough to conflict.
Mixing conda-forge pyscf with PyPI dmrgscf/openfermionpyscf is fine as long
as you don't `pip install pyscf` normally afterward (which would replace the
conda-forge build with an unrelated wheel and reintroduce the crash on
non-AVX-512 CPUs).

---

## Running CPU-only

If you don't have (or don't want to use) a GPU on a given node:
```bash
export CUDAQ_DEFAULT_SIMULATOR=qpp-cpu   # must be set before python starts
```
Combine with the AVX-512 check above — CPU-only runs still hit the pyscf
SIGILL issue if the CPU lacks AVX-512, it's unrelated to the CUDA-Q target
choice.

---

## Quick reference — all issues hit so far

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `ModuleNotFoundError: No module named 'pkg_resources'` | setuptools ≥82 removed pkg_resources | `pip install "setuptools<82"` |
| 2 | Hydra `Could not override 'trainer.epochs'` | wrong config key | use `trainer.max_iters` |
| 3 | `wandb.errors.errors.UsageError: No API key configured` | non-interactive context, no login | `WANDB_MODE=offline` (or `disabled`) |
| 4 | qiskit/scipy `pip check` conflicts | unused pin (qiskit) / patch-level gap (scipy) | ignore both, keep qiskit 2.4.0 |
| 5 | `RuntimeError: architecture mismatch` (TITAN V) | Volta (cc 7.0) unsupported by cuQuantum | `CUDAQ_DEFAULT_SIMULATOR=qpp-cpu` |
| 6 | `Illegal instruction (core dumped)` | pyscf's `libcgto.so` non-dispatching AVX-512 codepath, CPU lacks AVX-512 | `pip install pyscf==2.11.0 --no-binary pyscf --force-reinstall --no-deps` |
| 7 | `RuntimeError: Unable to open distributed interface library ...mpi.so` | CUDA-Q MPI plugin not built for this env | `activate_custom_mpi.sh` + `conda env config vars set LD_LIBRARY_PATH`/`MPI_PATH` |
| 8 | `TypeError: cannot pickle 'cudaq...SampleResult' object` | `sampler.py` checked `cudaq.mpi.is_initialized()` without also checking `self.mpi`; mpi4py's import-time auto-init made that check spuriously true even in single-process runs | guard both checks in `sampler.py`'s `run()` with `self.mpi and ...` (Step 9) |