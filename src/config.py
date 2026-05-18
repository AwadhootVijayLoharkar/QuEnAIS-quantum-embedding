from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

Geometry = List[Tuple[str, Tuple[float, float, float]]]


@dataclass
class MoleculeConfig:
    name     : str
    geometry : Geometry
    basis    : str = "sto-3g"
    charge   : int = 0
    spin     : int = 0
    verbose  : int = 0

    @property
    def is_open_shell(self) -> bool:
        return self.spin != 0

    @property
    def n_atoms(self) -> int:
        return len(self.geometry)

    @property
    def atom_symbols(self) -> List[str]:
        return [a[0] for a in self.geometry]


@dataclass
class ActiveSpaceConfig:
    method            : str                      = "entropy"  # "entropy" | "sized"
    entropy_threshold : float                    = 0.15
    max_norb          : int                      = 8
    min_norb          : int                      = 2
    size              : Optional[Tuple[int,int]] = None       # for method="sized"


@dataclass
class EmbeddingConfig:
    method          : str                 = "dmet"
    fragment_atoms  : Optional[List[int]] = None   # None → 1 per atom
    fragment_solver : str                 = "fci"
    verbose         : bool                = False


@dataclass
class SQDConfig:
    n_shots      : int            = 500_000
    n_iterations : int            = 10
    reps         : int            = 3
    entanglement : str            = "full"
    rand_seed    : int            = 42
    spin_sq      : Optional[float] = 0.0   # None = open-shell, 0.0 = singlet


@dataclass
class PipelineConfig:
    active_space         : ActiveSpaceConfig = field(default_factory=ActiveSpaceConfig)
    embedding            : EmbeddingConfig   = field(default_factory=EmbeddingConfig)
    sqd                  : SQDConfig         = field(default_factory=SQDConfig)
    run_fci_reference    : bool              = True
    solver_name          : str               = "sqd"   # "sqd" | "fci"