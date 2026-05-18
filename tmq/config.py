from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union

Geometry = List[Tuple[str, Tuple[float, float, float]]]


# ── Molecule ──────────────────────────────────────────────────────────────────
@dataclass
class MoleculeConfig:
    name     : str
    geometry : Geometry
    basis    : str  = "sto-3g"
    charge   : int  = 0
    spin     : int  = 0          # 2S = number of unpaired electrons
    verbose  : int  = 0

    @property
    def is_open_shell(self) -> bool:
        return self.spin != 0


# ── Active Space ──────────────────────────────────────────────────────────────
@dataclass
class ActiveSpaceConfig:
    method            : str             = "entropy"   # "entropy" | "sized"
    entropy_threshold : float           = 0.15
    max_norb          : int             = 8
    min_norb          : int             = 2
    size              : Optional[Tuple[int, int]] = None  # (n_elec, n_orb) for "sized"


# ── Embedding ─────────────────────────────────────────────────────────────────
@dataclass
class EmbeddingConfig:
    method          : str            = "dmet"
    fragment_atoms  : Optional[List[int]] = None   # None → auto (1 per atom)
    fragment_solver : str            = "fci"
    verbose         : bool           = False


# ── SQD Solver ────────────────────────────────────────────────────────────────
@dataclass
class SQDConfig:
    n_shots      : int   = 500_000
    n_iterations : int   = 10
    reps         : int   = 3
    entanglement : str   = "full"     # "full" | "linear" | "circular"
    rand_seed    : int   = 42
    spin_sq      : Optional[float] = 0.0   # None → open-shell, 0.0 → singlet


# ── VQE Solver (placeholder) ──────────────────────────────────────────────────
@dataclass
class VQEConfig:
    ansatz        : str   = "efficient_su2"
    reps          : int   = 3
    optimizer     : str   = "COBYLA"
    max_iter      : int   = 500
    backend       : str   = "statevector"


# ── Top-level Pipeline ────────────────────────────────────────────────────────
@dataclass
class PipelineConfig:
    active_space         : ActiveSpaceConfig = field(default_factory=ActiveSpaceConfig)
    embedding            : EmbeddingConfig   = field(default_factory=EmbeddingConfig)
    sqd                  : SQDConfig         = field(default_factory=SQDConfig)
    run_fci_reference    : bool              = True
    solver_name          : str               = "sqd"   # "sqd" | "fci" | "vqe"