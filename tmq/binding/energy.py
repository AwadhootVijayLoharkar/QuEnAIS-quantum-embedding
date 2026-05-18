from __future__ import annotations
from typing import List
from tmq.config import MoleculeConfig, PipelineConfig
from tmq.results import BindingEnergyResult, PipelineResult

HA_TO_EV = 27.2114


class BindingEnergyCalculator:
    """
    Computes:  ΔE_bind = E(complex) - Σ_i E(fragment_i)

    All energies come from DMET total energies for consistency.
    The quantum solver refines the most-correlated fragment of each system.

    Usage
    -----
    calc = BindingEnergyCalculator(
        complex_mol   = fe_n6_cfg,
        fragment_mols = [fe_cfg, n2_cfg, n2_cfg, n2_cfg],
        pipeline_cfg  = pipeline_cfg,
    )
    result = calc.compute()
    print(result.summary())
    """

    def __init__(
        self,
        complex_mol   : MoleculeConfig,
        fragment_mols : List[MoleculeConfig],
        pipeline_cfg  : PipelineConfig,
    ):
        self.complex_mol   = complex_mol
        self.fragment_mols = fragment_mols
        self.pipeline_cfg  = pipeline_cfg

    def compute(self) -> BindingEnergyResult:
        from tmq.pipeline import Pipeline   # local import avoids circular

        print(f"\n{'─'*60}")
        print(f"  Computing binding energy for {self.complex_mol.name}")
        print(f"{'─'*60}")

        # ── Run complex ───────────────────────────────────────────────────────
        complex_result = Pipeline(self.complex_mol, self.pipeline_cfg).run()

        # ── Run each fragment ─────────────────────────────────────────────────
        frag_results: List[PipelineResult] = []
        for fmol in self.fragment_mols:
            result = Pipeline(fmol, self.pipeline_cfg).run()
            frag_results.append(result)

        # ── ΔE = E_complex - Σ E_frags ────────────────────────────────────────
        E_complex = complex_result.dmet_energy
        E_frags   = sum(r.dmet_energy for r in frag_results)
        dE_ha     = E_complex - E_frags
        dE_eV     = dE_ha * HA_TO_EV

        return BindingEnergyResult(
            complex_result   = complex_result,
            fragment_results = frag_results,
            binding_energy   = dE_ha,
            binding_energy_eV= dE_eV,
        )