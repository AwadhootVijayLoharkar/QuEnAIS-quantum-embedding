from __future__ import annotations
import numpy as np
from pyscf import gto
from asf.wrapper import find_from_mol, find_from_scf, sized_space_from_mol
from tmq.config import ActiveSpaceConfig, MoleculeConfig
from tmq.results import ActiveSpaceResult


class ActiveSpaceFinder:
    """Wraps ASF and performs Mulliken orbital→atom mapping."""

    def __init__(self, cfg: ActiveSpaceConfig):
        self.cfg = cfg

    def run(self, mol_pyscf: gto.Mole) -> ActiveSpaceResult:
        cfg = self.cfg

        # ── Run ASF ──────────────────────────────────────────────────────────
        if cfg.method == "entropy":
            active_space = find_from_mol(
                mol_pyscf,
                entropy_threshold = cfg.entropy_threshold,
                max_norb          = cfg.max_norb,
                min_norb          = cfg.min_norb,
                verbose           = (mol_pyscf.verbose >= 3),
            )
        elif cfg.method == "sized":
            if cfg.size is None:
                raise ValueError("ActiveSpaceConfig.size must be set for method='sized'")
            active_space = sized_space_from_mol(
                mol_pyscf,
                size    = cfg.size,
                verbose = (mol_pyscf.verbose >= 3),
            )
        else:
            raise ValueError(f"Unknown active space method: {cfg.method}")

        nel      = active_space.nel
        mo_list  = active_space.mo_list
        mo_coeff = active_space.mo_coeff

        if len(mo_list) == 0:
            raise RuntimeError(
                "ASF found 0 active orbitals. "
                "Lower entropy_threshold or use method='sized'."
            )

        # ── Mulliken orbital → atom mapping ───────────────────────────────────
        n_atoms   = mol_pyscf.natm
        n_active  = len(mo_list)
        S         = mol_pyscf.intor("int1e_ovlp")
        ao_labels = mol_pyscf.ao_labels(fmt=None)

        active_coeffs       = mo_coeff[:, mo_list]
        orbital_atom_weight = np.zeros((n_active, n_atoms))

        for orb_i in range(n_active):
            c  = active_coeffs[:, orb_i]
            CS = c * (S @ c)
            for ao_j, (atom_idx, *_) in enumerate(ao_labels):
                orbital_atom_weight[orb_i, atom_idx] += CS[ao_j]

        dominant_atoms  = np.argmax(orbital_atom_weight, axis=1)
        active_per_atom = np.bincount(dominant_atoms, minlength=n_atoms)
        most_active_frag = int(np.argmax(active_per_atom))

        return ActiveSpaceResult(
            nel                 = nel,
            n_active_orbs       = n_active,
            mo_list             = mo_list,
            mo_coeff            = mo_coeff,
            orbital_atom_weight = orbital_atom_weight,
            dominant_atoms      = dominant_atoms,
            active_per_atom     = active_per_atom,
            most_active_frag    = most_active_frag,
        )