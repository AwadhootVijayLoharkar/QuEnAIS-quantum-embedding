from __future__ import annotations
from pyscf import gto
from tangelo import SecondQuantizedMolecule

# no cross-module import needed here — only config
from config import MoleculeConfig


def build_pyscf_mol(cfg: MoleculeConfig) -> gto.Mole:
    return gto.M(
        atom    = cfg.geometry,
        basis   = cfg.basis,
        charge  = cfg.charge,
        spin    = cfg.spin,
        verbose = cfg.verbose,
    )


def build_tangelo_mol(cfg: MoleculeConfig) -> SecondQuantizedMolecule:
    return SecondQuantizedMolecule(
        cfg.geometry,
        q     = cfg.charge,
        spin  = cfg.spin,
        basis = cfg.basis,
    )