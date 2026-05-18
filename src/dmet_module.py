from __future__ import annotations
from tangelo.problem_decomposition import DMETProblemDecomposition

from config import MoleculeConfig, EmbeddingConfig
from molecule_builder import build_tangelo_mol


class DMETEmbedding:
    """Thin wrapper around Tangelo DMET."""

    def __init__(self):
        self._dmet       = None
        self._n_atoms    = None
        self._frag_atoms = None

    def build(self, mol_cfg: MoleculeConfig, emb_cfg: EmbeddingConfig):
        self._n_atoms    = mol_cfg.n_atoms
        self._frag_atoms = emb_cfg.fragment_atoms or [1] * self._n_atoms

        if sum(self._frag_atoms) != self._n_atoms:
            raise ValueError(
                f"fragment_atoms sums to {sum(self._frag_atoms)} "
                f"but molecule has {self._n_atoms} atoms."
            )

        tangelo_mol = build_tangelo_mol(mol_cfg)
        self._dmet  = DMETProblemDecomposition({
            "molecule"        : tangelo_mol,
            "fragment_atoms"  : self._frag_atoms,
            "fragment_solvers": emb_cfg.fragment_solver,
            "verbose"         : emb_cfg.verbose,
        })
        self._dmet.build()

    def run(self) -> float:
        if self._dmet is None:
            raise RuntimeError("Call build() before run()")
        return self._dmet.simulate()

    def get_fragment_hamiltonian(self, frag_idx: int):
        """Returns (h1e, h2e, n_alpha, n_beta) for the given fragment."""
        data = self._dmet.scf_fragments[frag_idx]
        return data[1], data[5], int(data[3][0]), int(data[3][1])