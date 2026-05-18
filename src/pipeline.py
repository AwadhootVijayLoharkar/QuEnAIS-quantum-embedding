from __future__ import annotations

# all siblings — no package prefix needed
from config  import MoleculeConfig, PipelineConfig
from results import PipelineResult, FragmentResult
from molecule_builder import build_pyscf_mol
from asf_module  import run_asf
from dmet_module import DMETEmbedding
from solvers     import get_solver, solve_fci


def run_pipeline(mol_cfg: MoleculeConfig,
                 pipe_cfg: PipelineConfig) -> PipelineResult:
    """
    Full pipeline:
      1. Build PySCF molecule
      2. ASF  → active space + most-correlated fragment
      3. DMET → total energy + embedded Hamiltonians
      4. FCI reference  (optional)
      5. Quantum solver on ASF-guided fragment
    """
    print(f"\n{'='*60}")
    print(f"  Pipeline: {mol_cfg.name}")
    print(f"{'='*60}")

    # ── 1. Build PySCF molecule ───────────────────────────────────────────────
    mol_pyscf = build_pyscf_mol(mol_cfg)

    # ── 2. Active space ───────────────────────────────────────────────────────
    print("\n[1/4] Running ASF...")
    as_res = run_asf(mol_pyscf, pipe_cfg.active_space)
    print(as_res.summary())

    # ── 3. DMET ───────────────────────────────────────────────────────────────
    print("\n[2/4] Running DMET...")
    dmet = DMETEmbedding()
    dmet.build(mol_cfg, pipe_cfg.embedding)
    dmet_energy = dmet.run()
    print(f"  DMET total energy = {dmet_energy:.8f} Ha")

    # ── 4. Fragment Hamiltonian ───────────────────────────────────────────────
    frag_idx = as_res.most_active_frag
    h1e, h2e, n_alpha, n_beta = dmet.get_fragment_hamiltonian(frag_idx)
    n_orb = h1e.shape[0]
    print(f"\n[3/4] Fragment {frag_idx}: "
          f"{n_orb} orbs | {n_alpha}α+{n_beta}β | {2*n_orb} qubits")

    # ── 5. FCI reference ─────────────────────────────────────────────────────
    fci_energy = None
    if pipe_cfg.run_fci_reference:
        fci_res    = solve_fci(h1e, h2e, n_orb, n_alpha, n_beta)
        fci_energy = fci_res.energy
        print(f"  FCI reference = {fci_energy:.8f} Ha")

    # ── 6. Solver ─────────────────────────────────────────────────────────────
    print(f"\n[4/4] Running solver: {pipe_cfg.solver_name!r}...")
    solver_fn  = get_solver(pipe_cfg.solver_name)
    solver_res = solver_fn(
        h1e, h2e, n_orb, n_alpha, n_beta,
        cfg = pipe_cfg.sqd,
    )

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