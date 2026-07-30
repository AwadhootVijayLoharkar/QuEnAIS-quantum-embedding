# dmet_molecule_adapter.py
#
# Drop-in replacement for gqe_qsci.molecule.PySCFMolecule that is built
# directly from a DMET embedding Hamiltonian (h1e, h2e, ecore, n_alpha,
# n_beta) instead of from a real molecular geometry + active-orbital list.
#
# Why this is needed (not just a config change):
#   PySCFMolecule builds mol/hf/mc from real atoms via `gto.M(atom=...)`
#   and picks an active space as a *subset of canonical MOs*. DMET's
#   embedding space (impurity + bath, from a Schmidt decomposition) is a
#   genuinely different orbital basis -- a rotated combination, not a
#   subset of canonical MOs -- so it can't be expressed that way.
#
#   PySCF fully supports building a mean-field object directly from
#   arbitrary one/two-body integrals (the "custom Hamiltonian" pattern,
#   used e.g. for FCIDUMP-based or model-Hamiltonian calculations). This
#   adapter uses exactly that pattern: it hands h1e_emb/h2e_emb to PySCF
#   as if they were real AO integrals, runs a genuine SCF on them (finding
#   the true HF-optimal orbitals *within the embedding space*), and folds
#   DMET's `ecore` in as the "nuclear repulsion" constant -- so `mf.e_tot`
#   comes out as the correct DMET-consistent total energy directly
#   (E_total = ecore + <psi_emb|H_emb|psi_emb> holds for ANY psi_emb,
#   including the true SCF/CCSD/FCI solution -- not just DMET's own
#   naive aufbau reference used internally for its ecore bookkeeping).
#
# Requires: pyscf. Run only where pyscf/openfermion/tequila/cudaq are
# actually installed (your HPC env) -- this file is untested by me since
# none of those packages are available in my sandbox. Validate with
# validate_excitation_generator.py before trusting results from this.

from dataclasses import dataclass
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from pyscf import gto, scf, ao2mo, cc, mcscf, lib


@dataclass(frozen=True, slots=True)
class Hamiltonian:
    h1: np.ndarray
    h2: np.ndarray
    e_core: float | np.floating


class DMETEmbeddingMolecule:
    """
    Same public interface as gqe_qsci.molecule.PySCFMolecule:
      .cas_hamiltonian  -> Hamiltonian(h1, h2, e_core)
      .norb, .nelec, .spin
      .hf                (real, converged PySCF mean-field object)
      .compute_casci()
      .compute_ccsd()
      .ccsd_amplitude    (property -> {"t1":..., "t2":...})
      .geometry, .basis, .active_indices   (kept for compatibility -- see
                                             note below, NOT used by the
                                             DMET-aware operator pool)

    `.geometry`/`.basis`/`.active_indices` are kept only so code that
    merely *checks for their presence* doesn't break. Do NOT use them to
    build tequila-based ansätze (tequila would reconstruct a Hamiltonian
    from real geometry that has nothing to do with h1e_emb/h2e_emb). Use
    DMETExcitationPool / DMETPauliEvolutionPool instead of
    ExcitationPool / PauliEvolutionPool for this molecule type.
    """

    def __init__(
        self,
        h1e_emb: np.ndarray,
        h2e_emb: np.ndarray,
        ecore: float,
        n_alpha: int,
        n_beta: int,
        num_threads: int | None = 1,
        cache_key_extra: str = "",
    ):
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

        # geometry/basis/active_indices kept only for interface
        # compatibility -- see class docstring. Do not feed to tequila.
        self.geometry = None
        self.basis = None
        self.active_indices = list(range(n_emb))

        self.mol = self._build_fake_mol(n_alpha, n_beta)
        self.hf = self._run_embedding_scf()
        self.mc = mcscf.CASCI(self.hf, self.norb, self.nelec)

        self.cas_hamiltonian = Hamiltonian(
            h1=self._h1e_emb, h2=self._h2e_emb, e_core=self._ecore
        )

        self._ccsd_amplitude = None
        self._cache_key = self._build_cache_key(cache_key_extra)
        self._cache_dir = Path(".cache") / "pyscf_dmet"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Fake-Hamiltonian PySCF setup
    # ------------------------------------------------------------------
    def _build_fake_mol(self, n_alpha: int, n_beta: int) -> gto.Mole:
        mol = gto.M(verbose=0)
        mol.nelectron = n_alpha + n_beta
        mol.spin = n_alpha - n_beta
        mol.incore_anyway = True
        mol.build(dump_input=False, parse_arg=False)
        return mol

    def _run_embedding_scf(self):
        n_emb = self.norb
        h1e = self._h1e_emb
        # 8-fold-symmetric compact storage for efficiency; h2e_emb was
        # already 8-fold symmetrized in DMET Phase E, so this is exact,
        # not an approximation.
        eri8 = ao2mo.restore(8, self._h2e_emb, n_emb)

        if self.spin == 0:
            mf = scf.RHF(self.mol)
        else:
            mf = scf.UHF(self.mol)

        mf.get_hcore = lambda *args, **kwargs: h1e
        mf.get_ovlp = lambda *args, **kwargs: np.eye(n_emb)
        mf._eri = eri8
        # Fold DMET's ecore in as the "nuclear repulsion" constant, so
        # mf.e_tot = ecore + <HF|H_emb|HF> = DMET total energy directly.
        mf.energy_nuc = lambda *args, **kwargs: self._ecore
        mf.max_cycle = 200
        mf.conv_tol = 1e-10
        mf.kernel()

        if not mf.converged:
            raise RuntimeError(
                "Embedding-space SCF did not converge. Check h1e_emb/h2e_emb "
                "for numerical issues (e.g. insufficient Hermiticity/symmetry) "
                "before trusting downstream CCSD/CASCI results."
            )
        return mf

    def _build_cache_key(self, extra: str) -> str:
        payload = {
            "h1_hash": hashlib.sha256(self._h1e_emb.tobytes()).hexdigest(),
            "h2_hash": hashlib.sha256(self._h2e_emb.tobytes()).hexdigest(),
            "ecore": self._ecore,
            "nelec": self.nelec,
            "extra": extra,
        }
        payload_json = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Reference calculations (mirrors PySCFMolecule's interface exactly)
    # ------------------------------------------------------------------
    @property
    def ccsd_amplitude(self):
        if self._ccsd_amplitude is None:
            _ = self.compute_ccsd()
        return self._ccsd_amplitude

    def compute_casci(self):
        cache_path = self._cache_dir / f"{self._cache_key}_casci.npz"
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as data:
                self._last_casci_civec = data["civec"]
                return data["energy"]
        e_fci, civec = self.mc.fcisolver.kernel(
            self.cas_hamiltonian.h1,
            self.cas_hamiltonian.h2,
            self.norb,
            self.nelec,
            ecore=self.cas_hamiltonian.e_core,
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

        if self.spin == 0:
            mycc = cc.RCCSD(self.hf)
        else:
            mycc = cc.UCCSD(self.hf)
        mycc.verbose = 0
        mycc.kernel()
        e_tot = self.hf.e_tot + mycc.e_corr

        # UCCSD returns (t1a, t1b) / (t2aa, t2ab, t2bb); the excitation
        # pool code expects closed-shell-style t1/t2 (single array each).
        # For a closed-shell embedding (n_alpha == n_beta) RCCSD already
        # gives that directly. For open-shell embeddings, DMETExcitationPool
        # must be used in its open-shell mode -- see dmet_excitation_pool.py.
        if self.spin == 0:
            t1, t2 = mycc.t1, mycc.t2
        else:
            t1, t2 = mycc.t1, mycc.t2  # tuples; open-shell pool handles this

        self._ccsd_amplitude = {"t1": t1, "t2": t2}
        np.savez(
            cache_path,
            energy=e_tot,
            t1=np.asarray(t1, dtype=object) if self.spin != 0 else t1,
            t2=np.asarray(t2, dtype=object) if self.spin != 0 else t2,
            allow_pickle=True,
        )
        return e_tot


def load_from_dmet_pickle(step2_pickle_path: str, **kwargs) -> DMETEmbeddingMolecule:
    """
    Convenience constructor: load directly from a DMET.py (step2_hamiltonian.py)
    output pickle, using its exact key names.
    """
    with open(step2_pickle_path, "rb") as f:
        step2 = pickle.load(f)

    return DMETEmbeddingMolecule(
        h1e_emb=step2["h1e"],
        h2e_emb=step2["h2e"],
        ecore=step2["ecore"],
        n_alpha=step2["n_alpha"],
        n_beta=step2["n_beta"],
        cache_key_extra=step2.get("mol_info", {}).get("molecule", ""),
        **kwargs,
    )