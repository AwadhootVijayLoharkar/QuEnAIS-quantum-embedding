import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tmq import *

fe_n6 = MoleculeConfig(
    name  = "FeN6",
    geometry = [
        ("Fe", (0.000,  0.000,  0.000)),
        ("N",  (0.000,  0.000,  2.000)),
        ("N",  (0.000,  0.000, -2.000)),
        ("N",  (0.000,  2.000,  0.000)),
        ("N",  (0.000, -2.000,  0.000)),
        ("N",  (2.000,  0.000,  0.000)),
        ("N", (-2.000,  0.000,  0.000)),
    ],
    basis  = "def2-SVP",
    charge = 0,
    spin   = 4,          # Fe(II) high-spin: 4 unpaired electrons
)

cfg = PipelineConfig(
    active_space = ActiveSpaceConfig(
        entropy_threshold = 0.07,
        max_norb          = 16,
        min_norb          = 4,
    ),
    embedding = EmbeddingConfig(
        fragment_atoms  = [1, 2, 2, 2],   # Fe | N-N | N-N | N-N
        fragment_solver = "fci",
    ),
    sqd = SQDConfig(
        n_shots      = 1_000_000,
        n_iterations = 20,
        entanglement = "circular",
        spin_sq      = None,          # open-shell: no spin constraint
    ),
    solver_name = "sqd",
)

result = Pipeline(fe_n6, cfg).run()