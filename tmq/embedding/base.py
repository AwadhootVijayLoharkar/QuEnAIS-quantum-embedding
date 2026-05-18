# base.py
from abc import ABC, abstractmethod
from tmq.config import MoleculeConfig, EmbeddingConfig
import numpy as np


class EmbeddingMethod(ABC):
    """Abstract base — subclass for DMET, NEVPT2, DMRG-DMET, etc."""

    @abstractmethod
    def build(self, mol_cfg: MoleculeConfig, emb_cfg: EmbeddingConfig): ...

    @abstractmethod
    def run(self) -> float:
        """Returns total embedded energy in Hartree."""
        ...

    @abstractmethod
    def get_fragment_hamiltonian(self, frag_idx: int):
        """Returns (h1e, h2e, n_alpha, n_beta)."""
        ...