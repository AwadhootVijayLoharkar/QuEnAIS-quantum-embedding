# run_example.py  (at project_root/)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config         import MoleculeConfig, ActiveSpaceConfig, EmbeddingConfig, SQDConfig, PipelineConfig
from pipeline       import run_pipeline
from binding_energy import compute_binding_energy

fen6 = MoleculeConfig(
    name     = "FeN6",
    geometry = [
        ("Fe", ( 0.,  0.,  0.)),
        ("N",  ( 0.,  0.,  2.)), ("N",  ( 0.,  0., -2.)),
        ("N",  ( 0.,  2.,  0.)), ("N",  ( 0., -2.,  0.)),
        ("N",  ( 2.,  0.,  0.)), ("N",  (-2.,  0.,  0.)),
    ],
    basis  = "sto-3g",
    spin   = 4,
)

cfg = PipelineConfig(
    active_space = ActiveSpaceConfig(entropy_threshold=0.07, max_norb=12),
    embedding    = EmbeddingConfig(fragment_atoms=[1,2,2,2], fragment_solver="fci"),
    sqd          = SQDConfig(n_shots=500_000, n_iterations=10, spin_sq=None),
    solver_name  = "sqd",
)

result = run_pipeline(fen6, cfg)
print(result.summary())