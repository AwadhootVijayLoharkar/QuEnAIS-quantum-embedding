from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Type
import numpy as np
from tmq.results import SolverResult


class BaseSolver(ABC):
    """
    All quantum/classical solvers implement this interface.
    New algorithms (VQE, QAOA, DMRG …) just subclass BaseSolver.
    """
    name: str = "base"

    @abstractmethod
    def solve(
        self,
        h1e     : np.ndarray,
        h2e     : np.ndarray,
        n_orb   : int,
        n_alpha : int,
        n_beta  : int,
    ) -> SolverResult:
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}()"


# ── Solver Registry ───────────────────────────────────────────────────────────
_REGISTRY: Dict[str, Type[BaseSolver]] = {}


def register_solver(cls: Type[BaseSolver]) -> Type[BaseSolver]:
    """Decorator — registers a solver by its .name attribute."""
    _REGISTRY[cls.name] = cls
    return cls


def get_solver(name: str) -> Type[BaseSolver]:
    if name not in _REGISTRY:
        raise KeyError(
            f"Solver '{name}' not found. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]