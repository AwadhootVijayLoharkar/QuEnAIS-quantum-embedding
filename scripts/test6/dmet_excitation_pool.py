# dmet_excitation_pool.py
#
# DMET-compatible replacements for gqe_qsci.gqe.operator_pool's
# PauliEvolutionPool / ExcitationPool.
#
# The original pool classes build their UCCSD excitation ansatz via
# tequila: `tq.Molecule(geometry=..., basis_set=..., active_orbitals=...)`
# then `tq_molecule.make_excitation_gate(indices, angle)`. That
# reconstructs a Hamiltonian from *real geometry*, which has nothing to
# do with DMET's embedding-space Hamiltonian (a rotated impurity+bath
# basis, not a subset of canonical MOs). Pointing that machinery at a
# DMETEmbeddingMolecule would silently build gates for the wrong active
# space -- it would run, just be physically wrong.
#
# This module replaces the *gate-generator construction* step only, using
# plain OpenFermion (the same library your own quantum_methods.py already
# uses for its qubit Hamiltonian) operating directly on abstract
# spin-orbital indices -- no geometry, no basis set, no tequila. Standard
# UCC excitation generator convention:
#
#   single, p (occ) -> q (virt):
#     G_pq = a_q^dag a_p  -  a_p^dag a_q
#
#   simultaneous multi-index excitation, pairs (p_i -> q_i):
#     G = prod_i(a_{q_i}^dag a_{p_i})  -  prod_i(a_{p_i}^dag a_{q_i})
#
# jordan_wigner(G) gives the exact Pauli decomposition, term-by-term
# equivalent to tequila's `paulistring.coeff` per term.
#
# IMPORTANT: I could not run tequila/openfermion/cudaq myself (not
# available in my sandbox) to confirm this matches tequila's internal
# sign/normalization convention exactly. Standard UCC generator
# conventions can differ by a global sign or factor between libraries.
# RUN validate_excitation_generator.py FIRST, on a real small molecule
# where tequila still works, and confirm the two agree (or find the
# fixed offset to correct for) BEFORE trusting DMET results built with
# this file.

from abc import ABC
from collections import Counter

import cudaq
import numpy
from openfermion import FermionOperator, jordan_wigner
from tequila.quantumchemistry.chemistry_tools import ClosedShellAmplitudes

from gqe_qsci.gqe.operator_pool import OperatorPool
from gqe_qsci.gqe.utils import convert_pauli_to_cudaq_spin, get_pauli_evolution_gate_count
from dmet_molecule_adapter import DMETEmbeddingMolecule


def excitation_generator_qubit_op(indices: list[tuple[int, int]]):
    """
    Build the Jordan-Wigner-mapped qubit operator for the standard
    antisymmetrized UCC excitation generator over the given list of
    (occ, virt) spin-orbital index pairs (simultaneous excitation).

    Returns an openfermion.QubitOperator. Its .terms dict maps
    ((qubit_idx, pauli_letter), ...) tuples -> complex coefficient,
    exactly analogous to tequila's per-paulistring `.coeff`.
    """
    forward = FermionOperator.identity()
    backward = FermionOperator.identity()
    for (p, q) in indices:
        forward *= FermionOperator(((q, 1), (p, 0)))   # a_q^dag a_p
        backward *= FermionOperator(((p, 1), (q, 0)))  # a_p^dag a_q
    generator = forward - backward
    return jordan_wigner(generator)


def _qubit_op_terms_to_cudaq(
    qubit_op, remove_z_ladder: bool = False
) -> list[tuple[cudaq.SpinOperatorTerm, complex]]:
    """
    Convert every term of an openfermion QubitOperator into
    (cudaq unit-Pauli term, coefficient) pairs, skipping the identity
    term (empty tuple) if present -- genuine excitation generators
    shouldn't have one, but guard anyway rather than silently mis-signing
    the pool if a numerical artifact produces a tiny one.

    If remove_z_ladder=True, Z operators are stripped from each term's
    Pauli dict *before* conversion to a cudaq term -- matching the
    original PauliEvolutionPool's `remove_z_ladder` behavior, which
    filters the JW parity-string Z's out before building the pool entry.
    """
    out = []
    for term, coeff in qubit_op.terms.items():
        if len(term) == 0:
            continue
        pauli_dict = {idx: letter for (idx, letter) in term}
        if remove_z_ladder:
            pauli_dict = {k: v for k, v in pauli_dict.items() if v.lower() != "z"}
            if not pauli_dict:
                continue
        cudaq_term = convert_pauli_to_cudaq_spin(pauli_dict)
        if cudaq_term is None:
            continue
        out.append((cudaq_term, coeff))
    return out


class DMETUCCSDBasedPool(OperatorPool, ABC):
    """
    Same interface/behavior as UCCSDBasedPool, except the excitation
    ansatz is built via OpenFermion (excitation_generator_qubit_op)
    instead of tequila. `generate_excitations()` is inherited unchanged
    from the original logic below -- it only operates on abstract
    spin-orbital indices derived from CCSD amplitudes and never touches
    geometry, so it's correct as-is for DMET's embedding space too.
    """

    def __init__(self, molecule: DMETEmbeddingMolecule, params: list[float] | None, threshold: float = 1e-8, **kwargs):
        super().__init__(molecule, params, threshold=threshold, **kwargs)

    def get_vocab_size(self):
        raise NotImplementedError("Subclasses must implement this method")

    def build_operator_pool(self):
        raise NotImplementedError("Subclasses must implement this method")

    def get_gate_count(self, seq: list[int]) -> Counter:
        raise NotImplementedError("Subclasses must implement this method")

    def generate_excitations(self, threshold: float):
        """
        Identical logic to UCCSDBasedPool.generate_excitations -- this
        part never depended on geometry, only on CCSD t1/t2 amplitudes,
        so it's reused verbatim (copied rather than imported, to keep
        this module self-contained and avoid depending on tequila-only
        codepaths elsewhere in UCCSDBasedPool).
        """
        ccsd_amplitudes = ClosedShellAmplitudes(
            tIjAb=self.molecule.ccsd_amplitude["t2"], tIA=self.molecule.ccsd_amplitude["t1"]
        )
        amplitudes_all = ccsd_amplitudes.make_parameter_dictionary(threshold=0.0, screening=False)
        amplitudes = {
            k: v for k, v in amplitudes_all.items()
            if not numpy.isclose(v, 0.0, atol=threshold)
        }
        amplitudes = dict(sorted(amplitudes.items(), key=lambda x: numpy.fabs(x[1]), reverse=True))
        indices = {}
        for key, t in amplitudes.items():
            assert (len(key) % 2 == 0)
            if not numpy.isclose(t, 0.0, atol=threshold):
                if len(key) == 2:
                    angle = 2.0 * t
                    idx_a = (2 * key[0], 2 * key[1])
                    idx_b = (2 * key[0] + 1, 2 * key[1] + 1)
                    indices[idx_a] = angle
                    indices[idx_b] = angle
                else:
                    assert len(key) == 4
                    angle = 2.0 * t
                    idx_abab = (2 * key[0] + 1, 2 * key[1] + 1, 2 * key[2], 2 * key[3])
                    indices[idx_abab] = angle
                    if key[0] != key[2] and key[1] != key[3]:
                        idx_aaaa = (2 * key[0], 2 * key[1], 2 * key[2], 2 * key[3])
                        idx_bbbb = (2 * key[0] + 1, 2 * key[1] + 1, 2 * key[2] + 1, 2 * key[3] + 1)
                        partner = tuple([key[2], key[1], key[0], key[3]])
                        partner_t = amplitudes_all.get(partner, 0.0)
                        anglex = 2.0 * (t - partner_t)
                        indices[idx_aaaa] = anglex
                        indices[idx_bbbb] = anglex
        return indices

    def generate_excitation_generators(self, threshold: float):
        """
        DMET-aware replacement for make_uccsd_ansatz(): returns a list of
        (angle, qubit_op) pairs -- one per excitation -- instead of a
        tequila QCircuit. qubit_op is the JW-mapped generator for that
        excitation (openfermion.QubitOperator).
        """
        screened_indices = self.generate_excitations(threshold=threshold)
        generators = []
        for idx, angle in screened_indices.items():
            converted = [(idx[2 * i], idx[2 * i + 1]) for i in range(len(idx) // 2)]
            qubit_op = excitation_generator_qubit_op(converted)
            generators.append((angle, qubit_op))
        return generators


class DMETPauliEvolutionPool(DMETUCCSDBasedPool):
    """DMET-aware equivalent of PauliEvolutionPool."""

    def __init__(
        self,
        molecule: DMETEmbeddingMolecule,
        params: list[float] | None,
        threshold: float = 1e-8,
        remove_z_ladder: bool = False,
        only_use_first_pauli: bool = False,
    ):
        super().__init__(molecule, params, threshold=threshold, remove_z_ladder=remove_z_ladder, only_use_first_pauli=only_use_first_pauli)

    def get_vocab_size(self):
        return len(self.pool)

    def build_operator_pool(self, threshold, remove_z_ladder=False, only_use_first_pauli=False):
        generators = self.generate_excitation_generators(threshold=threshold)
        seen = set()
        operator_pool = [self.get_identity_operator()]
        for angle, qubit_op in generators:
            for term, _coeff in _qubit_op_terms_to_cudaq(qubit_op, remove_z_ladder=remove_z_ladder):
                if str(term) in seen:
                    continue
                seen.add(str(term))
                if self.params is None:
                    operator_pool.append(angle * cudaq.SpinOperator(term))
                else:
                    for p in self.params:
                        operator_pool.append(p * cudaq.SpinOperator(term))
                if only_use_first_pauli:
                    break
        return operator_pool

    def get_gate_count(self, seq: list[int]) -> Counter:
        counts = Counter()
        for i in seq:
            operator = self.pool[i]
            for term in operator:
                pauli = term.get_pauli_word(self.n_qubits)
                count = get_pauli_evolution_gate_count(pauli)
                counts.update(count)
        return counts


class DMETExcitationPool(DMETUCCSDBasedPool):
    """DMET-aware equivalent of ExcitationPool."""

    def __init__(self, molecule: DMETEmbeddingMolecule, params: list[float] | None, threshold: float = 1e-8):
        super().__init__(molecule, params)

    def get_vocab_size(self):
        return len(self.pool)

    def build_operator_pool(self, threshold):
        generators = self.generate_excitation_generators(threshold=threshold)
        operator_pool = [self.get_identity_operator()]
        for angle, qubit_op in generators:
            operator = None
            for cudaq_term, coeff in _qubit_op_terms_to_cudaq(qubit_op):
                weighted = cudaq_term * coeff
                operator = weighted if operator is None else (operator + weighted)
            if operator is None:
                continue
            if self.params is None:
                operator_pool.append(angle * cudaq.SpinOperator(operator))
            else:
                for p in self.params:
                    operator_pool.append(p * cudaq.SpinOperator(operator))
        return operator_pool

    def get_gate_count(self, seq: list[int]) -> Counter:
        counts = Counter()
        for i in seq:
            operator = self.pool[i]
            for term in operator:
                count = get_pauli_evolution_gate_count(term.get_pauli_word(self.n_qubits))
                counts.update(count)
        return counts