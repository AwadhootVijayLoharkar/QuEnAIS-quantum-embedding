from __future__ import annotations
import logging
from tmq.config import MoleculeConfig, PipelineConfig
from tmq.results import PipelineResult, FragmentResult
from tmq.molecule.builder import build_pyscf_mol
from tmq.active_space.asf import ActiveSpaceFinder
from tmq.embedding.dmet import DMETEmbedding
from tmq.solvers.base import get_solver
from tmq.solvers.fci import FCISolver

log = logging.getLogger(__name__)


class Pipeline:
    """
    Full pipeline:
      MoleculeConfig + PipelineConfig
          │
          ├─ 1. Build PySCF molecule
          ├─ 2. ASF  → ActiveSpaceResult
          ├─ 3. DMET → total energy + fragment Hamiltonians
          ├─ 4. Quantum solver on ASF-guided fragment
          └─ 5. Return PipelineResult
    """

    def __init__(self, mol_cfg: MoleculeConfig, pipe_cfg: PipelineConfig):
        self.mol_cfg  = mol_cfg
        self.pipe_cfg = pipe_cfg

    def run(self) -> PipelineResult:
        mol_cfg  = self.mol_cfg
        pipe_cfg = self.pipe_cfg

        print(f"\n{'='*60}")
        print(f"  Pipeline: {mol_cfg.name}")
        print(f"{'='*60}")

        # ── Step 1: Build PySCF molecule ──────────────────────────────────────
        mol_pyscf = build_pyscf_mol(mol_cfg)

        # ── Step 2: Active Space ──────────────────────────────────────────────
        print("\n[1/4] Running ASF...")
        asf    = ActiveSpaceFinder(pipe_cfg.active_space)
        as_res = asf.run(mol_pyscf)
        print(as_res.summary())

        # ── Step 3: DMET ──────────────────────────────────────────────────────
        print("\n[2/4] Building & running DMET...")
        dmet = DMETEmbedding()
        dmet.build(mol_cfg, pipe_cfg.embedding)
        dmet_energy = dmet.run()
        print(f"  DMET total energy = {dmet_energy:.8f} Ha")

        # ── Step 4: Extract fragment Hamiltonian ──────────────────────────────
        frag_idx = as_res.most_active_frag
        h1e, h2e, n_alpha, n_beta = dmet.get_fragment_hamiltonian(frag_idx)
        n_orb = h1e.shape[0]
        print(f"\n[3/4] Fragment {frag_idx}: "
              f"{n_orb} orbs | {n_alpha}α+{n_beta}β | {2*n_orb} qubits")

        # ── Step 4a: FCI reference (optional) ────────────────────────────────
        fci_energy = None
        if pipe_cfg.run_fci_reference:
            fci_res   = FCISolver().solve(h1e, h2e, n_orb, n_alpha, n_beta)
            fci_energy = fci_res.energy
            print(f"  FCI  reference    = {fci_energy:.8f} Ha")

        # ── Step 5: Quantum solver ─────────────────────────────────────────────
        print(f"\n[4/4] Running solver: {pipe_cfg.solver_name}...")
        SolverCls  = get_solver(pipe_cfg.solver_name)
        solver     = SolverCls(pipe_cfg.sqd) if pipe_cfg.solver_name == "sqd" \
                     else SolverCls()
        solver_res = solver.solve(h1e, h2e, n_orb, n_alpha, n_beta)

        frag_result = FragmentResult(
            fragment_idx  = frag_idx,
            n_orb         = n_orb,
            n_alpha       = n_alpha,
            n_beta        = n_beta,
            fci_energy    = fci_energy,
            solver_result = solver_res,
        )

        result = PipelineResult(
            molecule_name   = mol_cfg.name,
            dmet_energy     = dmet_energy,
            active_space    = as_res,
            fragment_result = frag_result,
        )

        print(result.summary())
        return result