# gqe_for_qsci.py — test7
"""
Consolidated DMET -> GQE-for-QSCI bridge. This merges test6's
dmet_molecule_adapter.py (DMETEmbeddingMolecule) and
dmet_excitation_pool.py (DMETExcitationPool / DMETPauliEvolutionPool)
into one file, unchanged in physics/logic from test6 (that code was
already careful and self-documented), plus one addition:

NEW: after CASCI/CCSD references are computed, this calls
DMET.embedding_consistency_score() using the CASCI 1-RDM as the "solver
output" proxy, and prints/saves the mismatch score. When you wire this
into the real GQE training loop (train.py, external gqe-for-qsci repo),
call the same function again with the GQE run's actual final avg_occs
(from its subspace diagonalization) -- that's the real diagnostic; the
CASCI-based one here is just a same-file sanity check you get for free.

REMINDER (carried over from test6, still true): the excitation-generator
sign/convention was cross-checked against tequila via
validate_excitation_generator.py (unchanged from test6, not reproduced
here -- copy it over as-is). Your 100-epoch TiO2 GQE run tracking CASCI/
CCSD references and converging toward them is good empirical evidence the
convention is right, but run the formal check once for certainty.

This file needs: pyscf, cudaq, openfermion, tequila (only the
ClosedShellAmplitudes helper), and the external gqe_qsci package
(gqe-for-qsci submodule, per GQE_README.md) on the Python path.

Rename to gqe_for_qsci.py (or split back into two files, your call) when
you drop this into your own test7/ folder.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from abc import ABC
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyscf import gto, scf, ao2mo, cc, mcscf, lib

import cudaq
from openfermion import FermionOperator, jordan_wigner
from tequila.quantumchemistry.chemistry_tools import ClosedShellAmplitudes

from gqe_qsci.gqe.operator_pool import OperatorPool
from gqe_qsci.gqe.utils import convert_pauli_to_cudaq_spin, get_pauli_evolution_gate_count

# DMET.py (test7) -- for the consistency-score diagnostic
import DMET as dmet_step2


# ═══════════════════════════════════════════════════════════════════════
# Part 1 — DMETEmbeddingMolecule (from dmet_molecule_adapter.py)
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Hamiltonian:
    h1: np.ndarray
    h2: np.ndarray
    e_core: float | np.floating


class DMETEmbeddingMolecule:
    """
    Drop-in replacement for gqe_qsci.molecule.PySCFMolecule, built directly
    from a DMET embedding Hamiltonian (h1e, h2e, ecore, n_alpha, n_beta)
    instead of a real molecular geometry + active-orbital list. DMET's
    impurity+bath space is a rotated combination of AOs, not a subset of
    canonical MOs, so it can't be expressed the way PySCFMolecule expects
    -- this uses PySCF's "custom Hamiltonian" pattern instead: hand
    h1e_emb/h2e_emb to PySCF as if they were AO integrals, run a genuine
    SCF on them, and fold DMET's `ecore` in as the "nuclear repulsion"
    constant so `mf.e_tot` comes out as the correct DMET-consistent total
    energy directly for ANY psi_emb (SCF, CCSD, FCI, GQE, ...).
    """

    def __init__(self, h1e_emb: np.ndarray, h2e_emb: np.ndarray, ecore: float,
                 n_alpha: int, n_beta: int, num_threads: int | None = 1,
                 cache_key_extra: str = ""):
        lib.num_threads(num_threads)

        n_emb = h1e_emb.shape[0]
        assert h1e_emb.shape == (n_emb, n_emb)
        assert h2e_emb.shape == (n_emb, n_emb, n_emb, n_emb)

        self.norb = n_emb
        self.nelec = (n_alpha, n_beta)
        self.spin = n_alpha - n_beta
        self._ecore = float(ecore)
        self._h1e_emb = np.asarray(h1e_emb)
        self._h2e_emb = np.asarray(h2e_emb)

        # kept only for interface compatibility -- do NOT feed to tequila
        self.geometry = None
        self.basis = None
        self.active_indices = list(range(n_emb))

        self.mol = self._build_fake_mol(n_alpha, n_beta)
        self.hf = self._run_embedding_scf()
        self.mc = mcscf.CASCI(self.hf, self.norb, self.nelec)
        self.cas_hamiltonian = Hamiltonian(h1=self._h1e_emb, h2=self._h2e_emb,
                                            e_core=self._ecore)

        self._ccsd_amplitude = None
        self._cache_key = self._build_cache_key(cache_key_extra)
        self._cache_dir = Path(".cache") / "pyscf_dmet"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _build_fake_mol(self, n_alpha, n_beta):
        mol = gto.M(verbose=0)
        mol.nelectron = n_alpha + n_beta
        mol.spin = n_alpha - n_beta
        mol.incore_anyway = True
        mol.build(dump_input=False, parse_arg=False)
        return mol

    def _run_embedding_scf(self):
        n_emb = self.norb
        h1e = self._h1e_emb
        eri8 = ao2mo.restore(8, self._h2e_emb, n_emb)  # exact: already 8-fold symmetrized

        mf = scf.RHF(self.mol) if self.spin == 0 else scf.UHF(self.mol)
        mf.get_hcore = lambda *a, **k: h1e
        mf.get_ovlp  = lambda *a, **k: np.eye(n_emb)
        mf._eri = eri8
        mf.energy_nuc = lambda *a, **k: self._ecore
        mf.max_cycle = 200
        mf.conv_tol = 1e-10
        mf.kernel()

        if not mf.converged:
            raise RuntimeError(
                "Embedding-space SCF did not converge. Check h1e_emb/h2e_emb "
                "for numerical issues before trusting downstream results."
            )
        return mf

    def _build_cache_key(self, extra):
        payload = {
            "h1_hash": hashlib.sha256(self._h1e_emb.tobytes()).hexdigest(),
            "h2_hash": hashlib.sha256(self._h2e_emb.tobytes()).hexdigest(),
            "ecore": self._ecore, "nelec": self.nelec, "extra": extra,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str)
                               .encode("utf-8")).hexdigest()

    @property
    def ccsd_amplitude(self):
        if self._ccsd_amplitude is None:
            _ = self.compute_ccsd()
        return self._ccsd_amplitude

    def compute_casci(self):
        cache_path = self._cache_dir / f"{self._cache_key}_casci.npz"
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as data:
                return data["energy"]
        e_fci, civec = self.mc.fcisolver.kernel(
            self.cas_hamiltonian.h1, self.cas_hamiltonian.h2, self.norb,
            self.nelec, ecore=self.cas_hamiltonian.e_core,
        )
        # kept for the consistency-score hook below
        self._last_casci_civec = civec
        np.savez(cache_path, energy=e_fci)
        return e_fci

    def compute_ccsd(self):
        cache_path = self._cache_dir / f"{self._cache_key}_ccsd.npz"
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as data:
                self._ccsd_amplitude = {"t1": data["t1"], "t2": data["t2"]}
                return data["energy"]

        mycc = cc.RCCSD(self.hf) if self.spin == 0 else cc.UCCSD(self.hf)
        mycc.verbose = 0
        mycc.kernel()
        e_tot = self.hf.e_tot + mycc.e_corr
        t1, t2 = mycc.t1, mycc.t2
        self._ccsd_amplitude = {"t1": t1, "t2": t2}
        np.savez(cache_path, energy=e_tot,
                 t1=np.asarray(t1, dtype=object) if self.spin != 0 else t1,
                 t2=np.asarray(t2, dtype=object) if self.spin != 0 else t2,
                 allow_pickle=True)
        return e_tot

    # ── NEW: consistency-score hook ──────────────────────────────────────
    def casci_avg_occs(self):
        """
        Diagonal occupations implied by the CASCI ground state, in the
        SAME embedding-orbital ordering as DMET.py's ref_occ_alpha/beta.
        Useful as a free, same-file consistency check (compute_casci()
        must be called first); the real diagnostic should use the GQE
        run's own converged avg_occs instead -- see module docstring.
        """
        if not hasattr(self, "_last_casci_civec"):
            self.compute_casci()
        dm_a, dm_b = self.mc.fcisolver.make_rdm1s(
            self._last_casci_civec, self.norb, self.nelec
        )
        return np.clip(np.diag(dm_a), 0.0, 1.0), np.clip(np.diag(dm_b), 0.0, 1.0)


def load_from_dmet_pickle(step2_pickle_path: str, **kwargs) -> DMETEmbeddingMolecule:
    """Build a DMETEmbeddingMolecule directly from DMET.py's step2 pickle."""
    with open(step2_pickle_path, "rb") as f:
        step2 = pickle.load(f)
    mol = DMETEmbeddingMolecule(
        h1e_emb=step2["h1e"], h2e_emb=step2["h2e"], ecore=step2["ecore"],
        n_alpha=step2["n_alpha"], n_beta=step2["n_beta"],
        cache_key_extra=step2.get("mol_info", {}).get("molecule", ""),
        **kwargs,
    )
    mol._step2_result = step2   # kept for the consistency check below
    return mol


def run_consistency_check(mol: DMETEmbeddingMolecule, threshold=None):
    """
    Free sanity check: does the CASCI ground state (computed on the same
    embedding Hamiltonian GQE will train against) look like the density
    that was used to BUILD that Hamiltonian's bath in the first place?

    Large mismatch = one-shot DMET's bath likely doesn't represent the
    correlated solution well for this molecule -- rerun DMET.py with
    DMET_REFERENCE="casci" if currently on "mp2", or treat GQE energies
    on this embedding with extra skepticism.
    """
    import config as cfg
    threshold = cfg.CONSISTENCY_MISMATCH_THRESHOLD if threshold is None else threshold
    occ_a, occ_b = mol.casci_avg_occs()
    result = dmet_step2.embedding_consistency_score(
        mol._step2_result, (occ_a, occ_b), threshold=threshold
    )
    print(f"[Consistency check] mismatch_score={result['mismatch_score']:.4f} "
          f"(threshold={threshold})  flagged={result['flag']}")
    return result


# ═══════════════════════════════════════════════════════════════════════
# Part 2 — DMET-aware operator pools (from dmet_excitation_pool.py)
# ═══════════════════════════════════════════════════════════════════════
#
# Unchanged from test6: builds the UCC excitation generator directly from
# OpenFermion on abstract spin-orbital indices (no geometry, no tequila
# Hamiltonian reconstruction -- pointing tequila's own machinery at a
# DMETEmbeddingMolecule would silently build gates for the wrong active
# space). Still cross-check with validate_excitation_generator.py once.

def excitation_generator_qubit_op(indices: list[tuple[int, int]]):
    forward = FermionOperator.identity()
    backward = FermionOperator.identity()
    for (p, q) in indices:
        forward *= FermionOperator(((q, 1), (p, 0)))
        backward *= FermionOperator(((p, 1), (q, 0)))
    generator = -1j * (forward - backward)
    return jordan_wigner(generator)


def _qubit_op_terms_to_cudaq(qubit_op, remove_z_ladder: bool = False):
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
    def __init__(self, molecule: DMETEmbeddingMolecule, params, threshold: float = 1e-8, **kwargs):
        super().__init__(molecule, params, threshold=threshold, **kwargs)

    def get_vocab_size(self):
        raise NotImplementedError

    def build_operator_pool(self):
        raise NotImplementedError

    def get_gate_count(self, seq):
        raise NotImplementedError

    def generate_excitations(self, threshold: float):
        ccsd_amplitudes = ClosedShellAmplitudes(
            tIjAb=self.molecule.ccsd_amplitude["t2"], tIA=self.molecule.ccsd_amplitude["t1"]
        )
        amplitudes_all = ccsd_amplitudes.make_parameter_dictionary(threshold=0.0, screening=False)
        amplitudes = {k: v for k, v in amplitudes_all.items()
                      if not np.isclose(v, 0.0, atol=threshold)}
        amplitudes = dict(sorted(amplitudes.items(), key=lambda x: np.fabs(x[1]), reverse=True))
        indices = {}
        for key, t in amplitudes.items():
            assert len(key) % 2 == 0
            if not np.isclose(t, 0.0, atol=threshold):
                if len(key) == 2:
                    angle = 2.0 * t
                    indices[(2 * key[0], 2 * key[1])] = angle
                    indices[(2 * key[0] + 1, 2 * key[1] + 1)] = angle
                else:
                    assert len(key) == 4
                    angle = 2.0 * t
                    idx_abab = (2 * key[0] + 1, 2 * key[1] + 1, 2 * key[2], 2 * key[3])
                    indices[idx_abab] = angle
                    if key[0] != key[2] and key[1] != key[3]:
                        idx_aaaa = (2 * key[0], 2 * key[1], 2 * key[2], 2 * key[3])
                        idx_bbbb = (2 * key[0] + 1, 2 * key[1] + 1, 2 * key[2] + 1, 2 * key[3] + 1)
                        partner = (key[2], key[1], key[0], key[3])
                        partner_t = amplitudes_all.get(partner, 0.0)
                        anglex = 2.0 * (t - partner_t)
                        indices[idx_aaaa] = anglex
                        indices[idx_bbbb] = anglex
        return indices

    def generate_excitation_generators(self, threshold: float):
        screened_indices = self.generate_excitations(threshold=threshold)
        generators = []
        for idx, angle in screened_indices.items():
            converted = [(idx[2 * i], idx[2 * i + 1]) for i in range(len(idx) // 2)]
            generators.append((angle, excitation_generator_qubit_op(converted)))
        return generators


class DMETPauliEvolutionPool(DMETUCCSDBasedPool):
    def __init__(self, molecule, params, threshold: float = 1e-8,
                 remove_z_ladder: bool = False, only_use_first_pauli: bool = False):
        super().__init__(molecule, params, threshold=threshold,
                          remove_z_ladder=remove_z_ladder,
                          only_use_first_pauli=only_use_first_pauli)

    def get_vocab_size(self):
        return len(self.pool)

    def build_operator_pool(self, threshold, remove_z_ladder=False, only_use_first_pauli=False):
        generators = self.generate_excitation_generators(threshold=threshold)
        seen = set()
        operator_pool = [self.get_identity_operator()]
        for angle, qubit_op in generators:
            for term, _ in _qubit_op_terms_to_cudaq(qubit_op, remove_z_ladder=remove_z_ladder):
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

    def get_gate_count(self, seq):
        counts = Counter()
        for i in seq:
            for term in self.pool[i]:
                counts.update(get_pauli_evolution_gate_count(term.get_pauli_word(self.n_qubits)))
        return counts


class DMETExcitationPool(DMETUCCSDBasedPool):
    def __init__(self, molecule, params, threshold: float = 1e-8):
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

    def get_gate_count(self, seq):
        counts = Counter()
        for i in seq:
            for term in self.pool[i]:
                counts.update(get_pauli_evolution_gate_count(term.get_pauli_word(self.n_qubits)))
        return counts


# ═══════════════════════════════════════════════════════════════════════
# Smoke test / entry point
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import config as cfg

    mol = load_from_dmet_pickle(cfg.STEP2_FILE)
    print(f"Embedding: {mol.norb} orbitals, nelec={mol.nelec}, spin={mol.spin}")
    e_casci = mol.compute_casci()
    print(f"CASCI (this embedding) = {e_casci:.8f} Ha")
    run_consistency_check(mol)

    # Hand this `mol` to gqe_qsci's training entrypoint the same way
    # test6's dmet_embedding.yaml did (_target_: gqe_for_qsci.load_from_dmet_pickle),
    # with operator_pool.spec pointed at DMETExcitationPool / DMETPauliEvolutionPool.