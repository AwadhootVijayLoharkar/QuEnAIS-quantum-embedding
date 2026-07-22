# run_gqe_training.py — test_8
"""
Runs the EXTERNAL gqe_qsci repo's real training entrypoint (train.py) from
inside test_8/, without you having to cd into that repo's own folder.

That external repo's train.py likely needs its OWN directory as the
working directory (Hydra config discovery, relative paths -- see its
GQE_README.md), so this script sets that up for you via subprocess, using
config.GQE_QSCI_REPO_PATH as cwd, and tees combined stdout/stderr to BOTH
your terminal and config.GQE_LOG_FILE -- the exact file visualization.py
already knows how to parse for "[epoch N] {...}" lines.

EDIT config.GQE_QSCI_REPO_PATH (in config.py) to wherever you actually
cloned that repo before running this.

Any extra CLI/Hydra overrides you pass to this script are forwarded as-is
to train.py, e.g.:

    python run_gqe_training.py num_epochs=100 batch_size=8

Usage: python run_gqe_training.py [extra args forwarded to train.py]
"""

import config

import os
import sys
import subprocess


def main():
    repo = config.GQE_QSCI_REPO_PATH
    entry = os.path.join(repo, config.GQE_TRAIN_ENTRYPOINT)

    if not os.path.isdir(repo):
        raise FileNotFoundError(
            f"config.GQE_QSCI_REPO_PATH does not exist: {repo}\n"
            f"Edit that path in config.py to point at your gqe_qsci checkout "
            f"(or set the GQE_QSCI_REPO_PATH environment variable)."
        )
    if not os.path.exists(entry):
        raise FileNotFoundError(
            f"{config.GQE_TRAIN_ENTRYPOINT} not found in {repo}.\n"
            f"Check config.GQE_TRAIN_ENTRYPOINT."
        )

    cmd = [sys.executable, entry] + list(config.GQE_TRAIN_ARGS) + sys.argv[1:]
    print(f"[run_gqe_training] cwd={repo}")
    print(f"[run_gqe_training] cmd={' '.join(cmd)}")
    print(f"[run_gqe_training] logging to {config.GQE_LOG_FILE}\n")

    os.makedirs(os.path.dirname(config.GQE_LOG_FILE) or ".", exist_ok=True)
    with open(config.GQE_LOG_FILE, "w") as logf:
        proc = subprocess.Popen(
            cmd, cwd=repo, env=os.environ.copy(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
        proc.wait()

    if proc.returncode != 0:
        print(f"\n[run_gqe_training] train.py exited with code {proc.returncode}")
    else:
        print(f"\n[run_gqe_training] Done. Run visualization.py to plot the results.")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()