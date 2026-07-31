"""
QuEnAIS command-line interface.
Usage: quenais-run --molecule TiO2 --basis def2-svp
"""

import argparse
import os


def run_pipeline():
    parser = argparse.ArgumentParser(description="QuEnAIS Quantum Embedding Pipeline")
    parser.add_argument("--molecule",    default="TiO2")
    parser.add_argument("--basis",       default="def2-svp")
    parser.add_argument("--solver",      default="sqd",
                        choices=["sqd", "skqd", "sqdrift"])
    parser.add_argument("--ansatz",      default="lucj",
                        choices=["su2", "lucj"])
    parser.add_argument("--mapping",     default="bk",
                        choices=["jw", "bk"])
    parser.add_argument("--backend",     default="mps",
                        choices=["local", "mps", "ibm"])
    parser.add_argument("--shots",       type=int, default=8192)
    parser.add_argument("--project-dir", default=".")
    parser.add_argument("--steps",       nargs="+", type=int,
                        default=[0, 1, 2, 3, 4],
                        help="Steps to run: 0=classical 1=asf 2=hamiltonian "
                             "3=solver 4=visualize")
    parser.add_argument("--force",       action="store_true",
                        help="Rerun all steps ignoring cache")
    parser.add_argument("--no-scan",          action="store_true")
    parser.add_argument("--no-quantum-scan",  action="store_true")
    args = parser.parse_args()

    from quenais.config import Config
    cfg = Config(
        molecule         = args.molecule,
        basis            = args.basis,
        quantum_solver   = args.solver,
        ansatz           = args.ansatz,
        fermion_to_qubit = args.mapping,
        backend          = args.backend,
        n_shots          = args.shots,
        project_dir      = os.path.abspath(args.project_dir),
    )
    cfg.validate()
    cfg.make_dirs()
    cfg.load_geometry()

    step_map = {
        0: ("Classical",   "quenais.classical.runner",     "main"),
        1: ("ASF",         "quenais.active_space.finder",  "main"),
        2: ("Hamiltonian", "quenais.embedding.hamiltonian","main"),
        3: ("Solver",      "quenais.quantum.solver",       "main"),
        4: ("Visualize",   "quenais.visualization.plots",  "main"),
    }

    for step in sorted(args.steps):
        name, module_path, func = step_map[step]
        print(f"\n{'='*60}")
        print(f"  Step {step}: {name}")
        print(f"{'='*60}")
        import importlib
        mod = importlib.import_module(module_path)
        fn  = getattr(mod, func)
        if step == 4:
            fn(cfg, force=args.force,
               no_scan=args.no_scan,
               no_quantum_scan=args.no_quantum_scan)
        else:
            fn(cfg, force=args.force)

    print("\n[QuEnAIS] Pipeline complete.")