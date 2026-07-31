# QuEnAIS — Quantum Embedding for Strongly Correlated Molecules

A Python package for quantum chemistry simulations combining classical
embedding (DMET) with quantum solvers (SQD/SKQD/SqDRIFT) for strongly
correlated molecules.

## Pipeline Overview
CIF file → Classical (HF/MP2) → Active Space Finder → DMET Embedding → Quantum Solver → Visualization Step 0 Step 1 Step 2 Step 3 Step 4

## Installation

### 1. Install manual prerequisites first (see INSTALL.md)

```bash

##### ASF (Active Space Finder)

git clone https://github.com/HQSquantumsimulations/ActiveSpaceFinder.git
cd ActiveSpaceFinder
pip install .
./init_dmrgscf_settings.sh

##### qiskit-fermions (requires Rust + clang)

git clone https://github.com/Qiskit/qiskit-fermions.git
cd qiskit-fermions
pip install --group build
pip install --no-build-isolation .

##### block2

pip install block2
```

### 2. Or use the automated install script

```bash
git clone https://github.com/AwadhootVijayLoharkar/quenais.git
cd quenais
bash install.sh
```

### 3. Install quenais

```bash
pip install -e ".[quantum]"
```

## Usage

### Command line

```bash

### Run full pipeline

quenais-run --molecule TiO2 --basis def2-svp

### Run specific steps only

quenais-run --molecule TiO2 --steps 0 1 2

### Choose solver and ansatz

quenais-run --molecule TiO2 --solver sqd --ansatz lucj --mapping bk

### Skip geometry scan

quenais-run --molecule TiO2 --no-scan

### Force rerun ignoring cache

quenais-run --molecule TiO2 --force
```

### Python API

```python
from quenais.config import Config
from quenais.classical.runner import main as run_classical
from quenais.active_space.finder import main as run_asf
from quenais.embedding.hamiltonian import main as run_hamiltonian
from quenais.quantum.solver import main as run_solver
from quenais.visualization.plots import main as run_viz

### Configure

cfg = Config(
    molecule       = "TiO2",
    basis          = "def2-svp",
    quantum_solver = "sqd",
    ansatz         = "lucj",
    project_dir    = "/path/to/your/project",
)
cfg.validate()
cfg.make_dirs()
cfg.load_geometry()

### Run pipeline

run_classical(cfg)
run_asf(cfg)
run_hamiltonian(cfg)
run_solver(cfg)
run_viz(cfg)
```

## Configuration

Key parameters in `Config`:

| Parameter | Default | Description |
|---|---|---|
| molecule | "TiO2" | Molecule name (must match CIF file) |
| basis | "def2-svp" | Basis set |
| quantum_solver | "sqd" | Solver: sqd, skqd, sqdrift |
| ansatz | "lucj" | Ansatz: lucj (recommended), su2 |
| fermion_to_qubit | "bk" | Mapping: bk (recommended), jw |
| backend | "mps" | Backend: mps, local, ibm |
| n_shots | 8192 | Number of circuit shots |
| sqd_iters | 10 | SQD iterations |

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

## Requirements

- Python 3.10 or 3.11
- PySCF >= 2.4
- Qiskit >= 1.0
- qiskit-aer >= 0.14
- qiskit-addon-sqd >= 0.5
- ffsim >= 0.0.50
- openfermion >= 1.6
- ASF (manual install)
- qiskit-fermions (manual install, requires Rust)
- block2 (manual install)

## Running Tests

```bash
pytest tests/ -v
```

## License

Apache-2.0