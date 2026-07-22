# run_all.py — test_8
"""
Runs the whole pipeline in the correct order, all from this folder:

    ASF.py -> classical_methods.py -> DMET.py -> gqe_for_qsci.py (smoke
    test / consistency check) -> visualization.py

NOTE the order: ASF.py runs BEFORE classical_methods.py on purpose --
classical_methods.py's CASSCF/NEVPT2 only get a fair, matching active
space if step1_asf.pkl already exists. Running classical_methods.py first
silently falls back to a generic active-space guess, which for a case
like N2 can produce a CASSCF active space with zero correlating degrees
of freedom (CASSCF energy == HF energy -- not a real comparison number).

Pass --force to re-run every cacheable step from scratch (forwarded to
ASF.py / classical_methods.py / DMET.py / gqe_for_qsci.py).

run_gqe_training.py (the real, long-running external GQE training job) is
NOT included here -- run that separately once you're happy with the
Step 1/2 diagnostics printed by this script.

Usage: python run_all.py [--force]
"""

import subprocess
import sys

STEPS = ["ASF.py", "classical_methods.py", "DMET.py", "gqe_for_qsci.py", "visualization.py"]


def main():
    extra_args = sys.argv[1:]
    for step in STEPS:
        args = extra_args if step != "visualization.py" else []
        print(f"\n{'#'*60}\n# Running {step} {' '.join(args)}\n{'#'*60}")
        result = subprocess.run([sys.executable, step] + args)
        if result.returncode != 0:
            print(f"\n[run_all] {step} exited with code {result.returncode} -- stopping.")
            sys.exit(result.returncode)
    print(f"\n[run_all] All steps finished. See results/ and results/plots/.")


if __name__ == "__main__":
    main()