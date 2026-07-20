# visualization.py — test7
"""
Turns everything the pipeline has produced so far into PNG plots and CSV
tables, using only what's actually saved to disk:

  step0_classical.pkl  -- HF/MP2/CCSD/CASSCF/NEVPT2 on the full molecule
  step1_asf.pkl         -- active space, NO deviation spectrum, tier
  step2_hamiltonian.pkl -- Schmidt SV spectrum, bath quality, ecore, mu
  <config.GQE_LOG_FILE> -- raw stdout/log from `python train.py ... > log`
                           in the external gqe-for-qsci repo (the same
                           "[epoch N] {...}" format you already showed me)

Each output is generated independently and skipped with a printed reason
if its input is missing -- this never hard-fails just because you haven't
run every step yet.

Outputs (config.PLOTS_DIR / config.RESULTS_DIR):
  fig1_asf_deviation_spectrum.png
  fig2_dmet_schmidt_spectrum.png
  fig3_gqe_energy_convergence.png
  fig4_gqe_circuit_resources.png
  fig5_method_comparison.png
  results_summary.csv
  gqe_epoch_log.csv

Usage: python visualization.py
"""

import os
import re
import ast
import csv
import pickle
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

os.makedirs(config.PLOTS_DIR, exist_ok=True)


def _load(path, label):
    if not os.path.exists(path):
        print(f"  [skip] {label}: {path} not found")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


step0 = _load(config.STEP0_FILE, "Step 0 (classical)")
step1 = _load(config.STEP1_FILE, "Step 1 (ASF)")
step2 = _load(config.STEP2_FILE, "Step 2 (DMET)")


# ═══════════════════════════════════════════════════════════════════════
# GQE training-log parser
# ═══════════════════════════════════════════════════════════════════════

EPOCH_RE   = re.compile(r"\[epoch (\d+)\]\s*(\{.*?\})", re.DOTALL)
FLOAT64_RE = re.compile(r"np\.float64\(([^)]*)\)")


def parse_gqe_log(log_path):
    """
    Parses lines like:
      [epoch 0] {'GQE-optimized/energy/min': -997.78, ...}
    (the exact format `grep -oE '\\[epoch [0-9]+\\] \\{[^}]*\\}'` on your
    training log produces). Strips np.float64(...) wrappers before using
    ast.literal_eval (never eval()) to parse each dict safely.
    """
    if not log_path or not os.path.exists(log_path):
        print(f"  [skip] GQE log: {log_path} not found (set config.GQE_LOG_FILE)")
        return []

    rows = []
    with open(log_path) as f:
        text = f.read()

    for m in EPOCH_RE.finditer(text):
        epoch = int(m.group(1))
        dict_str = FLOAT64_RE.sub(r"\1", m.group(2))
        try:
            d = ast.literal_eval(dict_str)
        except (ValueError, SyntaxError) as e:
            warnings.warn(f"Could not parse epoch {epoch}: {e}")
            continue
        d["epoch"] = epoch
        rows.append(d)

    rows.sort(key=lambda r: r["epoch"])
    return rows


gqe_rows = parse_gqe_log(config.GQE_LOG_FILE)
if gqe_rows:
    print(f"  Parsed {len(gqe_rows)} epochs from {config.GQE_LOG_FILE}")


def _col(rows, key):
    """Extract one column as a numpy array, skipping rows where it's absent."""
    xs, ys = [], []
    for r in rows:
        if key in r and r[key] is not None:
            xs.append(r["epoch"])
            ys.append(float(r[key]))
    return np.array(xs), np.array(ys)


# ═══════════════════════════════════════════════════════════════════════
# Fig 1 — ASF deviation spectrum
# ═══════════════════════════════════════════════════════════════════════

def plot_asf_spectrum():
    if step1 is None:
        return
    dev = np.asarray(step1["deviation"])
    active = set(step1["mo_list"])
    order = np.argsort(-dev)
    colors = ["#C44E52" if i in active else "#4C72B0" for i in order]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(order)), dev[order], color=colors)
    ax.set_xlabel("Natural orbital (sorted by deviation)")
    ax.set_ylabel("deviation = min(n, 2-n)")
    ax.set_title(f"ASF deviation spectrum — {step1['mol_info']['molecule']} "
                 f"(Tier {step1['tier']}, red = selected active space)")
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "fig1_asf_deviation_spectrum.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════
# Fig 2 — DMET Schmidt singular-value spectrum (the gap adaptive_bath used)
# ═══════════════════════════════════════════════════════════════════════

def plot_dmet_spectrum():
    if step2 is None:
        return
    sv_all = np.asarray(step2.get("sv_all"))
    if sv_all is None or sv_all.size == 0:
        print("  [skip] fig2: step2 pickle has no 'sv_all' -- re-run DMET.py "
              "(this test7 revision) to save the full spectrum.")
        return
    n_bath = step2["n_bath"]

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#C44E52" if i < n_bath else "#4C72B0" for i in range(len(sv_all))]
    ax.bar(range(len(sv_all)), sv_all, color=colors)
    ax.set_yscale("log")
    ax.set_xlabel("Schmidt singular value index (sorted descending)")
    ax.set_ylabel("singular value (log scale)")
    ax.set_title(f"DMET Schmidt spectrum — {step2['mol_info']['molecule']}  "
                 f"(red = kept as bath, n_bath={n_bath}, "
                 f"sv2_cov={step2['sv2_cov']:.4f})")
    fig.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "fig2_dmet_schmidt_spectrum.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════
# Fig 3 — GQE energy convergence vs CASCI / CCSD references
# ═══════════════════════════════════════════════════════════════════════

def plot_gqe_convergence():
    if not gqe_rows:
        return
    fig, ax = plt.subplots(figsize=(9, 5))

    series = [
        ("GQE-optimized(best_so_far)/energy - R-CASCI", "GQE-optimized vs CASCI", "#DD8452"),
        ("Local-refined(best_so_far)/energy - R-CASCI", "Local-refined vs CASCI", "#55A868"),
        ("Global-refined(best_so_far)/energy - R-CASCI", "Global-refined vs CASCI", "#4C72B0"),
        ("Global-refined(best_so_far)/energy - R-CCSD", "Global-refined vs CCSD", "#8172B2"),
    ]
    plotted = False
    for key, label, color in series:
        x, y = _col(gqe_rows, key)
        if len(x) > 0:
            ax.plot(x, y, label=label, color=color, linewidth=1.8)
            plotted = True
    if not plotted:
        print("  [skip] fig3: none of the expected energy-error columns found in the log")
        plt.close(fig)
        return

    ax.axhline(1.6e-3, color="gray", linestyle="--", linewidth=1,
               label="chemical accuracy (1.6 mHa)")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("|energy error| (Ha, log scale)")
    ax.set_title("GQE-for-QSCI convergence vs classical references")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "fig3_gqe_energy_convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════
# Fig 4 — circuit resource growth over training
# ═══════════════════════════════════════════════════════════════════════

def plot_gqe_resources():
    if not gqe_rows:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for key, label in [("GQE-optimized/cx_count/max", "cx_count (max)"),
                        ("GQE-optimized/total_gates/max", "total_gates (max)")]:
        x, y = _col(gqe_rows, key)
        if len(x) > 0:
            ax1.plot(x, y, label=label, linewidth=1.5)
    ax1.set_xlabel("epoch"); ax1.set_ylabel("gate count")
    ax1.set_title("Circuit resources per epoch")
    ax1.legend(fontsize=8)

    for key, label in [("Global-refined(best_so_far)/subspace_dim", "Global-refined subspace_dim"),
                        ("Local-refined(best_so_far)/subspace_dim", "Local-refined subspace_dim")]:
        x, y = _col(gqe_rows, key)
        if len(x) > 0:
            ax2.plot(x, y, label=label, linewidth=1.5)
    ax2.set_xlabel("epoch"); ax2.set_ylabel("number of configurations")
    ax2.set_title("Subspace size growth")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "fig4_gqe_circuit_resources.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════
# Fig 5 — method comparison bar chart (classical vs DMET+GQE, same molecule)
# ═══════════════════════════════════════════════════════════════════════

def plot_method_comparison():
    labels, energies, colors = [], [], []

    if step0 is not None:
        for name, data in step0["methods"].items():
            e = data.get("energy")
            if e is not None:
                labels.append(name); energies.append(e); colors.append("#4C72B0")

    if step2 is not None:
        ref_info = step2.get("reference_density_info", {})
        if ref_info.get("method") == "casci":
            # ecore + CASCI(active space only) -- an approximate DMET total
            # energy at the CASCI-in-active-space level, comparable to the
            # classical numbers above because ecore already carries the
            # rest of the molecule's mean-field contribution.
            labels.append("DMET+CASCI(active)")
            energies.append(step2["ecore"] + ref_info["e_cas"])
            colors.append("#C44E52")

    if gqe_rows:
        # Reconstruct absolute energy from the error-vs-CASCI column plus
        # the CASCI(active-space) reference, if both are available.
        if step2 is not None and step2.get("reference_density_info", {}).get("method") == "casci":
            e_cas_active = step2["reference_density_info"]["e_cas"]
            _, err = _col(gqe_rows, "Global-refined(best_so_far)/energy - R-CASCI")
            if len(err) > 0:
                labels.append("DMET+GQE (Global-refined, final)")
                energies.append(step2["ecore"] + e_cas_active + err[-1])
                colors.append("#55A868")

    if not labels:
        print("  [skip] fig5: no energies available from any stage yet")
        return

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels)), 5))
    ax.bar(labels, energies, color=colors)
    ax.set_ylabel("Total energy (Ha)")
    ax.set_title(f"Method comparison — {config.MOLECULE}")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "fig5_method_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

    return labels, energies


# ═══════════════════════════════════════════════════════════════════════
# CSV outputs
# ═══════════════════════════════════════════════════════════════════════

def write_results_summary_csv(comparison):
    path = os.path.join(config.RESULTS_DIR, "results_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["molecule", config.MOLECULE])
        w.writerow(["basis", config.BASIS])
        w.writerow([])

        if step1 is not None:
            w.writerow(["-- Step 1: ASF --"])
            w.writerow(["tier", step1["tier"]])
            w.writerow(["active_space_nel", step1["nel"]])
            w.writerow(["active_space_norb", step1["n_active_orbs"]])
            w.writerow(["mo_list", step1["mo_list"]])
            w.writerow(["correlation_strength", step1["corr_strength"]])
            w.writerow([])

        if step2 is not None:
            w.writerow(["-- Step 2: DMET --"])
            w.writerow(["n_imp", step2["n_imp"]])
            w.writerow(["n_bath", step2["n_bath"]])
            w.writerow(["n_emb", step2["n_emb"]])
            w.writerow(["sv2_coverage", step2["sv2_cov"]])
            w.writerow(["ecore_Ha", step2["ecore"]])
            w.writerow(["mu_Ha", step2["mu"]])
            w.writerow(["reference_density_method",
                        step2.get("reference_density_info", {}).get("method")])
            w.writerow([])

        if comparison:
            labels, energies = comparison
            w.writerow(["-- Method comparison (Ha) --"])
            for lbl, e in zip(labels, energies):
                w.writerow([lbl, e])

    print(f"  Saved {path}")


def write_gqe_epoch_csv():
    if not gqe_rows:
        return
    all_keys = sorted({k for r in gqe_rows for k in r.keys()})
    path = os.path.join(config.RESULTS_DIR, "gqe_epoch_log.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in gqe_rows:
            w.writerow(r)
    print(f"  Saved {path}  ({len(gqe_rows)} epochs, {len(all_keys)} columns)")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}\n[Visualization] {config.MOLECULE}\n{'='*60}")

plot_asf_spectrum()
plot_dmet_spectrum()
plot_gqe_convergence()
plot_gqe_resources()
comparison = plot_method_comparison()

write_results_summary_csv(comparison)
write_gqe_epoch_csv()

print(f"\n[Visualization] Done. See {config.PLOTS_DIR} and {config.RESULTS_DIR}")