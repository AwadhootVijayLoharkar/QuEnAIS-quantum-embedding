# QuEnAIS — Quantum Embedding for Strongly Correlated Molecules

A Python package for quantum chemistry simulations combining classical
embedding (DMET) with quantum solvers (SQD/SKQD/SqDRIFT) for strongly
correlated molecules.

This package was developed (still in active development) as part of Awadhoot Loharkar's Master's thesis
at the Fraunhofer Institute for Industrial Mathematics (ITWM),
and forms a small part of the larger
[QuEnAIS (Quantum Enhanced AI Drug Synthesizer)](https://www.quantensysteme.info/projektatlas/projekte/q/quenais)
project, which combines generative AI with HPC and quantum computing to
accelerate drug discovery. The project is funded under the Eureka call
"Applied Quantum Technologies" and coordinated by Fraunhofer ITWM, with
partners Cortex Discovery GmbH and QunaSys ApS.

## Pipeline Overview

| Step 0 | Step 1 | Step 2 | Step 3 | Step 4 |
|---|---|---|---|---|
| CIF file → Classical (HF/MP2) | Active Space Finder | DMET Embedding | Quantum Solver | Visualization |

> **Note:** Developers are currently working on adding a generative
> quantum eigensolver for the SQD algorithm, as it significantly reduces
> resource requirements. Because of this ongoing work, you may run into
> issues when launching SQD. If you hit a problem you can't resolve,
> please contact the developer at a.loharkar@edu.rptu.de.

## Requirements

- Python 3.10 or 3.11
- Linux or macOS (see **Windows users** below)
- A C/Fortran toolchain (clang, cmake, gfortran, openblas) and Rust — all provided by `environment.yml`

## Installation

### Windows users — use WSL

Do not install this package directly on native Windows. `block2==0.5.2`
(a required dependency) only ships build wheels that conflict with pip —
both the root Windows pip and the pip inside a mamba/conda virtual env
created on Windows. There is no working native-Windows install path.

Instead, install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
and run everything from an Ubuntu (or other Linux) WSL shell, then follow the
steps below as if on native Linux.

### 1. Clone the repo

```bash
git clone https://github.com/AwadhootVijayLoharkar/quenais.git
cd quenais
```

### 2. Create and activate the conda environment

```bash
mamba env create -f environment.yml -p ./quenais-env
mamba activate ./quenais-env
```

Using `-p ./quenais-env` (a local path, not `--name`) avoids read-only
errors on shared/HPC systems where the default conda envs directory isn't
writable.

### 3. Run the install script

```bash
bash install.sh
```

This installs everything in the required order: base packages, quantum
packages, the `block2` wrapper (its paths depend on the current
environment and can't be hardcoded), ASF, qiskit-fermions, and finally
`quenais` itself

### 4. Verify

```bash
python -c "import quenais; print(quenais.__version__)"
pytest tests/ -v
```

If you switch conda environments later, regenerate the block2 wrapper:

```bash
mamba activate ./your-new-env
python quenais/utils/regenerate_wrapper.py
```

## Getting Started

For the Python API, configuration options, and a full walkthrough of the
pipeline, see [notebooks/tutorial.ipynb](notebooks/tutorial.ipynb).

### Command line

```bash
# Run full pipeline
quenais-run --molecule TiO2 --basis def2-svp

# Run specific steps only
quenais-run --molecule TiO2 --steps 0 1 2

# Choose solver and ansatz
quenais-run --molecule TiO2 --solver sqd --ansatz lucj --mapping bk

# Skip geometry scan
quenais-run --molecule TiO2 --no-scan

# Force rerun ignoring cache
quenais-run --molecule TiO2 --force
```

## Project Structure

```
your-project/
├── cif_files/
│   └── TiO2.cif        ← place your CIF files here
└── results/
    ├── step0_classical.pkl
    ├── step1_asf.pkl
    ├── step2_hamiltonian.pkl
    ├── step3_results.pkl
    └── plots/
        ├── plot1_energy_comparison.png
        ├── plot2_convergence.png
        ├── plot3_orbital_deviations.png
        ├── plot4_bath_svs.png
        ├── plot5_lowdin_heatmap.png
        ├── plot6_geometry_scan.png
        └── plot7_summary.png
```

## Running Tests

```bash
pytest tests/ -v
```

## License

Apache-2.0
