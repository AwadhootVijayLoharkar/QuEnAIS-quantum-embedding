from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np


@dataclass
class ActiveSpaceResult:
    nel           : int
    n_active_orbs : int
    mo_list       : List[int]
    mo_coeff      : np.ndarray
    orbital_atom_weight : np.ndarray   # (n_active_orbs, n_atoms)
    dominant_atoms      : np.ndarray   # (n_active_orbs,)
    active_per_atom     : np.ndarray   # (n_atoms,)
    most_active_frag    : int

    def summary(self) -> str:
        lines = [
            f"  n_active_electrons : {self.nel}",
            f"  n_active_orbitals  : {self.n_active_orbs}",
            f"  active mo_list     : {self.mo_list}",
            f"  most_active_frag   : {self.most_active_frag}",
        ]
        return "\n".join(lines)


@dataclass
class SolverResult:
    energy    : float
    avg_occs  : tuple
    spin_sq   : float
    converged : bool
    n_configs : int
    n_iters   : int = 0
    extras    : Dict = field(default_factory=dict)


@dataclass
class FragmentResult:
    fragment_idx  : int
    n_orb         : int
    n_alpha       : int
    n_beta        : int
    fci_energy    : Optional[float]
    solver_result : SolverResult

    @property
    def energy(self) -> float:
        return self.solver_result.energy

    @property
    def delta_vs_fci(self) -> Optional[float]:
        if self.fci_energy is None:
            return None
        return abs(self.energy - self.fci_energy)


@dataclass
class PipelineResult:
    molecule_name    : str
    dmet_energy      : float
    active_space     : ActiveSpaceResult
    fragment_result  : FragmentResult
    binding_energy   : Optional[float] = None

    def summary(self) -> str:
        fr = self.fragment_result
        lines = [
            f"\n{'═'*65}",
            f"  Molecule           : {self.molecule_name}",
            f"  DMET total energy  : {self.dmet_energy:.8f} Ha",
            f"  Fragment {fr.fragment_idx}          : {fr.n_orb} orbs | "
            f"{fr.n_alpha}α+{fr.n_beta}β",
        ]
        if fr.fci_energy:
            lines.append(f"  FCI  energy        : {fr.fci_energy:.8f} Ha")
        lines += [
            f"  Solver energy      : {fr.energy:.8f} Ha",
            f"  Δ (solver vs FCI)  : {fr.delta_vs_fci:.2e} Ha" if fr.delta_vs_fci else "",
            f"  <S²>               : {fr.solver_result.spin_sq:.6f}",
        ]
        if self.binding_energy is not None:
            lines.append(f"  Binding energy     : {self.binding_energy:.8f} Ha")
        lines.append(f"{'═'*65}")
        return "\n".join(l for l in lines if l)


@dataclass
class BindingEnergyResult:
    complex_result  : PipelineResult
    fragment_results: List[PipelineResult]
    binding_energy  : float             # E_complex - Σ E_fragments (Ha)
    binding_energy_eV : float           # same in eV

    def summary(self) -> str:
        lines = [
            f"\n{'═'*65}",
            f"  Binding Energy Breakdown",
            f"  Complex energy     : {self.complex_result.dmet_energy:.8f} Ha",
        ]
        for r in self.fragment_results:
            lines.append(f"  {r.molecule_name:20s} : {r.dmet_energy:.8f} Ha")
        lines += [
            f"  {'─'*50}",
            f"  ΔE (binding)       : {self.binding_energy:.8f} Ha",
            f"                     : {self.binding_energy_eV:.4f} eV",
            f"{'═'*65}",
        ]
        return "\n".join(lines)