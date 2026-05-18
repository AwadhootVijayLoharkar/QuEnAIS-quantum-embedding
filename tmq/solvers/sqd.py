from __future__ import annotations
import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorSampler
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.fermion import solve_fermion
from qiskit_addon_sqd.configuration_recovery import recover_configurations
from tmq.config import SQDConfig
from tmq.results import SolverResult
from tmq.solvers.base import BaseSolver, register_solver


def _filter_bitstrings(bsm, probs, n_alpha, n_beta, n_orb):
    valid = (
        (bsm[:, :n_orb].sum(axis=1) == n_alpha) &
        (bsm[:, n_orb:].sum(axis=1) == n_beta)
    )
    return bsm[valid], probs[valid]


@register_solver
class SQDSolver(BaseSolver):
    name = "sqd"

    def __init__(self, cfg: SQDConfig = None):
        self.cfg = cfg or SQDConfig()

    def solve(self, h1e, h2e, n_orb, n_alpha, n_beta) -> SolverResult:
        cfg      = self.cfg
        n_qubits = 2 * n_orb

        # ── Build & sample circuit ────────────────────────────────────────────
        hf_circ = QuantumCircuit(n_qubits)
        for i in range(n_alpha): hf_circ.x(i)
        for i in range(n_beta):  hf_circ.x(n_orb + i)

        ansatz = efficient_su2(
            n_qubits,
            reps                      = cfg.reps,
            entanglement              = cfg.entanglement,
            skip_final_rotation_layer = True,
        )
        rng    = np.random.default_rng(cfg.rand_seed)
        params = rng.uniform(0, 2 * np.pi, ansatz.num_parameters)
        circuit = hf_circ.compose(ansatz.assign_parameters(params))
        circuit.measure_all()

        counts = (
            StatevectorSampler()
            .run([circuit], shots=cfg.n_shots)
            .result()[0]
            .data.meas
            .get_counts()
        )

        bsm, probs = counts_to_arrays(counts)
        bsm, probs = _filter_bitstrings(bsm, probs, n_alpha, n_beta, n_orb)

        if bsm.shape[0] == 0:
            raise RuntimeError(
                "No valid bitstrings after filtering. "
                "Increase n_shots or check n_alpha/n_beta."
            )

        # ── SQD iterative loop ────────────────────────────────────────────────
        avg_occs = (
            np.array([1.0 if i < n_alpha else 0.0 for i in range(n_orb)]),
            np.array([1.0 if i < n_beta  else 0.0 for i in range(n_orb)]),
        )
        energy = None
        spin_sq = None
        converged = False

        for it in range(cfg.n_iterations):
            bsm, probs = recover_configurations(
                bsm, probs, avg_occs,
                num_elec_a = n_alpha,
                num_elec_b = n_beta,
                rand_seed  = cfg.rand_seed,
            )
            if bsm.shape[0] == 0:
                break

            energy, _, avg_occs, spin_sq = solve_fermion(
                bsm,
                hcore      = h1e,
                eri        = h2e,
                open_shell = cfg.spin_sq is None,
                spin_sq    = cfg.spin_sq,
            )
            converged = True   # mark as converged on last successful iter

        return SolverResult(
            energy    = energy,
            avg_occs  = avg_occs,
            spin_sq   = spin_sq,
            converged = converged,
            n_configs = bsm.shape[0],
            n_iters   = cfg.n_iterations,
        )