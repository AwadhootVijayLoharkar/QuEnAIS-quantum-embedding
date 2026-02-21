#!/usr/bin/env bash
set -e

# Ensure conda/mamba is initialized
source ~/.bashrc 2>/dev/null || true



echo "Creating project structure..."

# ----------------------------
# Core directories
# ----------------------------
mkdir -p docs
mkdir -p configs/templates
mkdir -p configs/experiments
mkdir -p src/quenais/{core,dft,embedding,solvers,workflows,ml,utils}
mkdir -p experiments/formaldehyde_test
mkdir -p data/{raw,intermediate,results}
mkdir -p tests
mkdir -p notebooks
mkdir -p .github/workflows

# ----------------------------
# Documentation files
# ----------------------------
touch docs/01_project_overview.md
touch docs/02_scientific_background.md
touch docs/03_mathematical_formulation.md
touch docs/04_architecture_plan.md
touch docs/05_solver_strategy.md
touch docs/06_data_pipeline.md
touch docs/07_ml_integration.md
touch docs/roadmap.md

# ----------------------------
# Core Python package files
# ----------------------------
touch src/quenais/__init__.py

touch src/quenais/core/molecule.py
touch src/quenais/core/integrals.py
touch src/quenais/core/hamiltonian.py

touch src/quenais/dft/pyscf_driver.py
touch src/quenais/embedding/dmet.py

touch src/quenais/solvers/base_solver.py
touch src/quenais/solvers/dmrg_solver.py
touch src/quenais/solvers/skqd_solver.py

touch src/quenais/workflows/simple_pipeline.py
touch src/quenais/ml/dataset_builder.py

touch src/quenais/utils/io.py
touch src/quenais/utils/logging.py

# ----------------------------
# Experiment scaffold
# ----------------------------
touch experiments/formaldehyde_test/run_test.py
touch experiments/formaldehyde_test/analysis.ipynb

# ----------------------------
# Config template
# ----------------------------
cat <<EOF > configs/templates/default.yaml
molecule:
  geometry_file: data/raw/example.xyz
  basis: sto-3g

embedding:
  method: dmet
  fragment_atoms: []

solver:
  type: dmrg
  bond_dimension: 50
  sweeps: 5

output:
  save_rdm: true
  save_energy: true
EOF

# ----------------------------
# Environment file (Python 3.12 compatible)
# ----------------------------
cat <<EOF > environment.yml
name: quenais
channels:
  - conda-forge
dependencies:
  - python=3.12
  - numpy
  - scipy
  - pandas
  - matplotlib
  - pyyaml
  - pytest
  - jupyterlab
  - pyscf
  - pip
  - pip:
      - itensor
EOF

# ----------------------------
# Minimal pyproject.toml
# ----------------------------
cat <<EOF > pyproject.toml
[project]
name = "quenais"
version = "0.0.1"
description = "Hybrid quantum-classical embedding framework (DFT + DMET + advanced solvers)"
authors = [{name = "QuEnAIS Project"}]
requires-python = ">=3.12"

[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"
EOF

# ----------------------------
# Dockerfile
# ----------------------------
cat <<EOF > Dockerfile
FROM mambaorg/micromamba:latest

COPY environment.yml /tmp/environment.yml
RUN micromamba create -y -f /tmp/environment.yml

WORKDIR /app
COPY . /app

CMD ["bash"]
EOF

# ----------------------------
# Basic CI pipeline
# ----------------------------
cat <<EOF > .github/workflows/ci.yml
name: CI

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Conda
        uses: conda-incubator/setup-miniconda@v2
        with:
          activate-environment: quenais
          environment-file: environment.yml
      - name: Run Tests
        run: pytest tests/
EOF

# ----------------------------
# Basic test file
# ----------------------------
cat <<EOF > tests/test_imports.py
def test_import():
    import quenais
EOF

# ----------------------------
# Install required libraries into current env
# ----------------------------
echo "Installing required libraries into current active mamba environment..."

mamba install -c conda-forge numpy scipy pandas matplotlib pyyaml pytest jupyterlab pyscf -y
pip install itensor

echo "Bootstrap complete."
echo "You can now run: git add . && git commit -m 'Initial project scaffold'"