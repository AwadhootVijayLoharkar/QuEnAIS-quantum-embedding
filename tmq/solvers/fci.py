from __future__ import annotations
import numpy as np
from pyscf import fci as pyscf_fci
from tmq.results import SolverResult
from tmq.solvers.base import BaseSolver, register_solver


@register_solver
class FCISolver(BaseSolver):
    name = "fci"

    def solve(self, h1e, h2e, n_orb, n_alpha, n_beta) -> SolverResult:
        solver = pyscf_fci.direct_spin1.FCI()
        energy, civec = solver.kernel(h1e, h2e, n_orb, (n_alpha, n_beta))

        # Average occupations from RDM
        rdm1 = solver.make_rdm1(civec, n_orb, (n_alpha, n_beta))
        avg_occs = (
            np.diag(rdm1[0]) / 1.0,
            np.diag(rdm1[1]) / 1.0,
        )
        return SolverResult(
            energy    = energy,
            avg_occs  = avg_occs,
            spin_sq   = 0.0,
            converged = True,
            n_configs = 0,
        )