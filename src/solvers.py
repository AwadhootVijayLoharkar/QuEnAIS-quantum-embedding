from __future__ import annotations
import math
import numpy as np
from pyscf import fci as pyscf_fci
from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorSampler
from qiskit_addon_sqd.counts import counts_to_arrays
from qiskit_addon_sqd.fermion import solve_fermion
from qiskit_addon_sqd.configuration_recovery import recover_configurations

from config import SQDConfig
from results import SolverResult


# ── Registry ──────────────────────────────────────────────────────────────────
_REGISTRY = {}

def register(name):
    """Decorator: register a solver function under a name."""
    def decorator(fn):
        _REGISTRY[name] = fn
        return fn
    return decorator

def get_solver(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"Solver '{name}' not found. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


# ── FCI ───────────────────────────────────────────────────────────────────────
@register("fci")
def solve_fci(h1e, h2e, n_orb, n_alpha, n_beta, cfg=None) -> SolverResult:
    solver       = pyscf_fci.direct_spin1.FCI()
    energy, civec = solver.kernel(h1e, h2e, n_orb, (n_alpha, n_beta))
    rdm1          = solver.make_rdm1(civec, n_orb, (n_alpha, n_beta))
    avg_occs      = (np.diag(rdm1[0]), np.diag(rdm1[1]))
    return SolverResult(
        energy    = energy,
        avg_occs  = avg_occs,
        spin_sq   = 0.0,
        converged = True,
        n_configs = 0,
    )


# ── SQD ───────────────────────────────────────────────────────────────────────
def _filter_bitstrings(bsm, probs, n_alpha, n_beta, n_orb):
    valid = (
        (bsm[:, :n_orb].sum(axis=1) == n_alpha) &
        (bsm[:, n_orb:].sum(axis=1) == n_beta)
    )
    return bsm[valid], probs[valid]


@register("sqd")
def solve_sqd(h1e, h2e, n_orb, n_alpha, n_beta, cfg: SQDConfig = None) -> SolverResult:
    cfg      = cfg or SQDConfig()
    n_qubits = 2 * n_orb

    # ── Build circuit ─────────────────────────────────────────────────────────
    hf_circ = QuantumCircuit(n_qubits)
    for i in range(n_alpha): hf_circ.x(i)
    for i in range(n_beta):  hf_circ.x(n_orb + i)

    ansatz  = efficient_su2(
        n_qubits,
        reps                      = cfg.reps,
        entanglement              = cfg.entanglement,
        skip_final_rotation_layer = True,
    )
    rng     = np.random.default_rng(cfg.rand_seed)
    params  = rng.uniform(0, 2 * np.pi, ansatz.num_parameters)
    circuit = hf_circ.compose(ansatz.assign_parameters(params))
    circuit.measure_all()

    # ── Sample ────────────────────────────────────────────────────────────────
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
        raise RuntimeError("No valid bitstrings. Increase n_shots.")

    # ── Iterative diagonalization ─────────────────────────────────────────────
    avg_occs = (
        np.array([1.0 if i < n_alpha else 0.0 for i in range(n_orb)]),
        np.array([1.0 if i < n_beta  else 0.0 for i in range(n_orb)]),
    )
    energy  = None
    spin_sq = None

    for it in range(cfg.n_iterations):
        bsm, probs = recover_configurations(
            bsm, probs, avg_occs,
            num_elec_a = n_alpha,
            num_elec_b = n_beta,
            rand_seed  = cfg.rand_seed,
        )
        if bsm.shape[0] == 0:
            print(f"  [iter {it+1}] No valid configs after recovery"); break

        energy, _, avg_occs, spin_sq = solve_fermion(
            bsm,
            hcore      = h1e,
            eri        = h2e,
            open_shell = cfg.spin_sq is None,
            spin_sq    = cfg.spin_sq,
        )
        print(f"  Iter {it+1:02d} | E={energy:.8f} | "
              f"configs={bsm.shape[0]} | <S²>={spin_sq:.4f}")

    return SolverResult(
        energy    = energy,
        avg_occs  = avg_occs,
        spin_sq   = spin_sq,
        converged = energy is not None,
        n_configs = bsm.shape[0],
        n_iters   = cfg.n_iterations,
    )