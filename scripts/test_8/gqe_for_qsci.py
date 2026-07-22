# gqe_for_qsci.py — test_8
"""
Consolidated DMET -> GQE-for-QSCI bridge (dmet_molecule_adapter.py +
dmet_excitation_pool.py merged into one file).

FIX #1 (carried over from test7): `import DMET as dmet_step2` is gone.
DMET.py is a script -- importing it ran its argparse + cache-check code as
a side effect, and on a cache hit it called sys.exit(0) at module level,
silently killing this whole process before any of this file's own code
ran. This imports `dmet_lib` instead, which has zero module-level side
effects.

FIX #2 (new in test_8): this file used to require the external gqe_qsci
package to already be importable -- in practice that meant cd-ing into
that repo's own folder before running Python so it landed on sys.path (or
relying on a stale PYTHONPATH). That's exactly the "go to the
gqe-for-qsci folder to run this" annoyance. Fixed by explicitly adding
config.GQE_QSCI_REPO_PATH to sys.path at import time, below -- so this
script works from test_8/ (or anywhere else) regardless of cwd, as long
as config.GQE_QSCI_REPO_PATH points at the right place.

FIX #3 (new in test_8): `import config` now happens FIRST, before numpy /
pyscf -- needed both for the OpenBLAS/OpenMP env-var fix (see config.py)
and so GQE_QSCI_REPO_PATH is available before the gqe_qsci imports below.

Needs: pyscf, cudaq, openfermion, tequila (ClosedShellAmplitudes only),
and the external gqe_qsci package (path taken from config.py).
"""

from __future__ import annotations

import config

import sys
if config.GQE_QSCI_REPO_PATH not in sys.path:
    sys.path.insert(0, config.GQE_QSCI_REPO_PATH)

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

import dmet_lib


# ═══════════════════════════════════════════════════════════════════════
# Part 1 — DMETEmbeddingMolecule
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Hamiltonian:
    h1: np.ndarray
    h2: np.ndarray
    e_core: float | np.floating


class DMETEmbeddingMolecule:
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
        eri8 = ao2mo.restore(8, self._h2e_emb, n_emb)

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
        """
        FIX: the on-disk cache used to store only the energy, not the CI
        vector. On a cache MISS this method sets self._last_casci_civec
        fine -- but on a cache HIT (i.e. re-running after a previous
        successful run already wrote the .npz file), it returned early
        without ever setting that attribute. casci_avg_occs()'s
        `if not hasattr(...)` guard didn't help either, because calling
        compute_casci() again just hits the same disk cache again --
        it's a dead end every time the cache file already exists, which
        is exactly what happened on your second run:
        AttributeError: 'DMETEmbeddingMolecule' object has no attribute
        '_last_casci_civec'. Now the civec is cached too, so a cache hit
        restores it instead of skipping it.
        """
        cache_path = self._cache_dir / f"{self._cache_key}_casci.npz"
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as data:
                self._last_casci_civec = data["civec"]
                return data["energy"]
        e_fci, civec = self.mc.fcisolver.kernel(
            self.cas_hamiltonian.h1, self.cas_hamiltonian.h2, self.norb,
            self.nelec, ecore=self.cas_hamiltonian.e_core,
        )
        self._last_casci_civec = civec
        np.savez(cache_path, energy=e_fci, civec=np.asarray(civec))
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

    def casci_avg_occs(self):
        if not hasattr(self, "_last_casci_civec"):
            self.compute_casci()
        dm_a, dm_b = self.mc.fcisolver.make_rdm1s(
            self._last_casci_civec, self.norb, self.nelec
        )
        return np.clip(np.diag(dm_a), 0.0, 1.0), np.clip(np.diag(dm_b), 0.0, 1.0)


def load_from_dmet_pickle(step2_pickle_path: str, **kwargs) -> DMETEmbeddingMolecule:
    with open(step2_pickle_path, "rb") as f:
        step2 = pickle.load(f)
    mol = DMETEmbeddingMolecule(
        h1e_emb=step2["h1e"], h2e_emb=step2["h2e"], ecore=step2["ecore"],
        n_alpha=step2["n_alpha"], n_beta=step2["n_beta"],
        cache_key_extra=step2.get("mol_info", {}).get("molecule", ""),
        **kwargs,
    )
    mol._step2_result = step2
    return mol


def run_consistency_check(mol: DMETEmbeddingMolecule, threshold=None):
    """Uses dmet_lib now, not DMET -- see module docstring for why that matters."""
    threshold = config.CONSISTENCY_MISMATCH_THRESHOLD if threshold is None else threshold
    occ_a, occ_b = mol.casci_avg_occs()
    result = dmet_lib.embedding_consistency_score(
        mol._step2_result, (occ_a, occ_b), threshold=threshold
    )
    print(f"[Consistency check] mismatch_score={result['mismatch_score']:.4f} "
          f"(threshold={threshold})  flagged={result['flag']}")
    return result


# ═══════════════════════════════════════════════════════════════════════
# Part 2 — DMET-aware operator pools
# ═══════════════════════════════════════════════════════════════════════

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
    mol = load_from_dmet_pickle(config.STEP2_FILE)
    print(f"Embedding: {mol.norb} orbitals, nelec={mol.nelec}, spin={mol.spin}")

    # Independent sanity check: DMET.py's own "ecore self-consistent"
    # assertion is tautological (ecore is DEFINED as mf.e_tot - e_hf_emb,
    # so that check can never fail regardless of whether h1e_emb/h2e_emb
    # are actually right). This is a REAL check instead: mol.hf is an
    # actual converged self-consistent HF calculation run on h1e_emb/
    # h2e_emb/ecore (see _run_embedding_scf). If this doesn't land close
    # to the full molecule's UHF energy, the embedding Hamiltonian itself
    # is wrong, independent of mu or the reference-density method.
    uhf_ref = mol._step2_result.get("uhf_energy")
    print(f"Embedding-space HF (real SCF, independent check) = {mol.hf.e_tot:.8f} Ha")
    if uhf_ref is not None:
        print(f"  vs full-molecule UHF energy = {uhf_ref:.8f} Ha   "
              f"(diff = {mol.hf.e_tot - uhf_ref:+.4f} Ha)")

    e_casci = mol.compute_casci()
    print(f"CASCI (this embedding) = {e_casci:.8f} Ha")
    run_consistency_check(mol)