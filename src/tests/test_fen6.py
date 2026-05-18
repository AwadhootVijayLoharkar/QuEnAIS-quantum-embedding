# conftest.py already added src/ to sys.path before this runs
import pytest
import numpy as np

from config          import MoleculeConfig, ActiveSpaceConfig, EmbeddingConfig, SQDConfig, PipelineConfig
from pipeline        import run_pipeline
from binding_energy  import compute_binding_energy
from solvers         import solve_fci, solve_sqd, get_solver
from molecule_builder import build_pyscf_mol
from asf_module      import run_asf
from dmet_module     import DMETEmbedding


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fen6_mol():
    return MoleculeConfig(
        name     = "FeN6",
        geometry = [
            ("Fe", ( 0.000,  0.000,  0.000)),
            ("N",  ( 0.000,  0.000,  2.000)),
            ("N",  ( 0.000,  0.000, -2.000)),
            ("N",  ( 0.000,  2.000,  0.000)),
            ("N",  ( 0.000, -2.000,  0.000)),
            ("N",  ( 2.000,  0.000,  0.000)),
            ("N",  (-2.000,  0.000,  0.000)),
        ],
        basis   = "sto-3g",
        charge  = 0,
        spin    = 4,
        verbose = 0,
    )


@pytest.fixture(scope="module")
def fast_cfg():
    """Minimal config for quick test runs."""
    return PipelineConfig(
        active_space = ActiveSpaceConfig(
            entropy_threshold = 0.07,
            max_norb          = 12,
            min_norb          = 2,
        ),
        embedding = EmbeddingConfig(
            fragment_atoms  = [1, 2, 2, 2],
            fragment_solver = "fci",
            verbose         = False,
        ),
        sqd = SQDConfig(
            n_shots      = 20_000,
            n_iterations = 3,
            entanglement = "linear",
            spin_sq      = None,     # open-shell
            rand_seed    = 42,
        ),
        run_fci_reference = True,
        solver_name       = "sqd",
    )


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestImports:
    def test_all_modules_importable(self):
        import config, results, molecule_builder
        import asf_module, dmet_module, solvers
        import pipeline, binding_energy

    def test_solver_registry(self):
        assert get_solver("fci")  is not None
        assert get_solver("sqd")  is not None
        with pytest.raises(KeyError):
            get_solver("nonexistent")


class TestActiveSpace:
    def test_asf_finds_orbitals(self, fen6_mol, fast_cfg):
        mol   = build_pyscf_mol(fen6_mol)
        res   = run_asf(mol, fast_cfg.active_space)
        assert res.nel > 0
        assert res.n_active_orbs >= 2
        assert 0 <= res.most_active_frag < fen6_mol.n_atoms

    def test_mulliken_weights_sum_to_one(self, fen6_mol, fast_cfg):
        mol = build_pyscf_mol(fen6_mol)
        res = run_asf(mol, fast_cfg.active_space)
        # Each orbital's weights across atoms should sum to ~1.0
        row_sums = res.orbital_atom_weight.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.05)


class TestDMET:
    def test_dmet_energy_is_negative(self, fen6_mol, fast_cfg):
        dmet = DMETEmbedding()
        dmet.build(fen6_mol, fast_cfg.embedding)
        energy = dmet.run()
        assert energy < 0.0, f"Expected negative energy, got {energy}"

    def test_fragment_hamiltonian_shapes(self, fen6_mol, fast_cfg):
        dmet = DMETEmbedding()
        dmet.build(fen6_mol, fast_cfg.embedding)
        dmet.run()
        h1e, h2e, n_alpha, n_beta = dmet.get_fragment_hamiltonian(0)
        n = h1e.shape[0]
        assert h1e.shape == (n, n)
        assert h2e.shape == (n, n, n, n)
        assert n_alpha > 0
        assert n_beta  > 0


class TestSolvers:
    def test_fci_solver(self, fen6_mol, fast_cfg):
        dmet = DMETEmbedding()
        dmet.build(fen6_mol, fast_cfg.embedding)
        dmet.run()
        h1e, h2e, n_alpha, n_beta = dmet.get_fragment_hamiltonian(0)
        res = solve_fci(h1e, h2e, h1e.shape[0], n_alpha, n_beta)
        assert res.converged
        assert res.energy < 0.0

    def test_sqd_converges_near_fci(self, fen6_mol, fast_cfg):
        dmet = DMETEmbedding()
        dmet.build(fen6_mol, fast_cfg.embedding)
        dmet.run()
        h1e, h2e, n_alpha, n_beta = dmet.get_fragment_hamiltonian(0)
        n_orb = h1e.shape[0]

        fci_res = solve_fci(h1e, h2e, n_orb, n_alpha, n_beta)
        sqd_res = solve_sqd(h1e, h2e, n_orb, n_alpha, n_beta, cfg=fast_cfg.sqd)

        assert sqd_res.converged
        delta = abs(sqd_res.energy - fci_res.energy)
        assert delta < 0.1, f"SQD-FCI gap = {delta:.4f} Ha (> 0.1 Ha threshold)"


class TestFullPipeline:
    def test_pipeline_returns_result(self, fen6_mol, fast_cfg):
        result = run_pipeline(fen6_mol, fast_cfg)
        assert result.dmet_energy < 0.0
        assert result.fragment_result.energy is not None
        assert result.fragment_result.solver_result.converged

    def test_pipeline_summary_prints(self, fen6_mol, fast_cfg, capsys):
        result = run_pipeline(fen6_mol, fast_cfg)
        out = capsys.readouterr().out
        assert "DMET" in out
        assert fen6_mol.name in out


class TestBindingEnergy:
    def test_binding_energy_computes(self):
        fe = MoleculeConfig(
            name="Fe", geometry=[("Fe",(0.,0.,0.))],
            basis="sto-3g", spin=4,
        )
        n2 = MoleculeConfig(
            name="N2",
            geometry=[("N",(0.,0.,0.)), ("N",(0.,0.,1.098))],
            basis="sto-3g",
        )
        fen6 = MoleculeConfig(
            name="FeN6",
            geometry=[
                ("Fe",(0.,0.,0.)),
                ("N",(0.,0.,2.)), ("N",(0.,0.,-2.)),
                ("N",(0.,2.,0.)), ("N",(0.,-2.,0.)),
                ("N",(2.,0.,0.)), ("N",(-2.,0.,0.)),
            ],
            basis="sto-3g", spin=4,
        )
        cfg = PipelineConfig(
            active_space = ActiveSpaceConfig(entropy_threshold=0.15, max_norb=8),
            embedding    = EmbeddingConfig(fragment_solver="fci"),
            sqd          = SQDConfig(n_shots=10_000, n_iterations=2),
            solver_name  = "sqd",
        )
        result = compute_binding_energy(fen6, [fe, n2, n2, n2], cfg)
        assert isinstance(result.binding_energy, float)
        assert isinstance(result.binding_energy_eV, float)
        print(result.summary())