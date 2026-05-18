from __future__ import annotations
from typing import List

from config  import MoleculeConfig, PipelineConfig
from results import BindingEnergyResult, PipelineResult
from pipeline import run_pipeline

HA_TO_EV = 27.2114


def compute_binding_energy(
    complex_mol   : MoleculeConfig,
    fragment_mols : List[MoleculeConfig],
    pipe_cfg      : PipelineConfig,
) -> BindingEnergyResult:
    """
    ΔE_bind = E(complex) - Σ_i E(fragment_i)
    All energies taken from DMET total energies.
    """
    print(f"\n{'─'*60}")
    print(f"  Binding energy: {complex_mol.name}")
    print(f"{'─'*60}")

    complex_result = run_pipeline(complex_mol, pipe_cfg)

    frag_results: List[PipelineResult] = []
    for fmol in fragment_mols:
        frag_results.append(run_pipeline(fmol, pipe_cfg))

    E_complex = complex_result.dmet_energy
    E_frags   = sum(r.dmet_energy for r in frag_results)
    dE_ha     = E_complex - E_frags

    return BindingEnergyResult(
        complex_result    = complex_result,
        fragment_results  = frag_results,
        binding_energy    = dE_ha,
        binding_energy_eV = dE_ha * HA_TO_EV,
    )