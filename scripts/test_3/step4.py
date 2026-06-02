"""
Step 4 — Visualization & Analysis Dashboard
=============================================

Generates a comprehensive set of plots and data exports from Steps 1-3.

Panels produced
───────────────
  Fig 1 — Active Space Analysis
    1a. MP2 natural orbital deviation spectrum (all orbitals)
    1b. Natural orbital occupations with core/active/virtual zones
    1c. Gap detection landscape (gaps between consecutive deviations)

  Fig 2 — Löwdin Population Analysis
    2a. Heatmap: active orbital weight per atom
    2b. Bar chart: active orbital count per atom
    2c. Dominant atom assignment per active orbital

  Fig 3 — DMET Embedding Quality
    3a. Schmidt singular values with adaptive cutoff
    3b. Cumulative sv² coverage curve
    3c. Electron count decomposition (core / active / total)

  Fig 4 — Solver Convergence
    4a. Energy vs iteration (absolute)
    4b. ΔE vs FCI per iteration (log scale)
    4c. Subspace configuration count vs iteration

  Fig 5 — Molecular Score Dashboard
    5a. Radar chart of normalized key metrics
    5b. Horizontal bar chart: all pipeline scores
    5c. Tier classification summary card

  Fig 6 — Summary Table
    6a. Printable summary table image (all key numbers in one view)

Requires : results/step1_asf.pkl
           results/step2_hamiltonian.pkl
           results/step3_results.pkl
Saves    : results/figures/fig1_active_space.png
           results/figures/fig2_lowdin.png
           results/figures/fig3_embedding.png
           results/figures/fig4_convergence.png
           results/figures/fig5_scores.png
           results/figures/fig6_summary.png
           results/step4_data.csv
           results/step4_summary.txt
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.table import Table
import csv

import config

# ═════════════════════════════════════════════════════════════════════════════
# Paths
# ═════════════════════════════════════════════════════════════════════════════
RESULTS_DIR = config.RESULTS_DIR
FIG_DIR     = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

STEP1_FILE = config.STEP1_FILE
STEP2_FILE = config.STEP2_FILE
STEP3_FILE = os.path.join(RESULTS_DIR, "step3_results.pkl")
CSV_FILE   = os.path.join(RESULTS_DIR, "step4_data.csv")
TXT_FILE   = os.path.join(RESULTS_DIR, "step4_summary.txt")

for fpath, label in [
    (STEP1_FILE, "Step 1 (ASF)"),
    (STEP2_FILE, "Step 2 (Hamiltonian)"),
    (STEP3_FILE, "Step 3 (Solver)"),
]:
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"[Step 4] {label} not found: {fpath}\n"
            "Run the corresponding script first."
        )

# ═════════════════════════════════════════════════════════════════════════════
# Load all results
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("[Step 4] Loading results...")
print("=" * 65)

with open(STEP1_FILE, "rb") as f: step1 = pickle.load(f)
with open(STEP2_FILE, "rb") as f: step2 = pickle.load(f)
with open(STEP3_FILE, "rb") as f: step3 = pickle.load(f)

# Step 1
deviation        = step1["deviation"]          # all MO deviations
no_occ           = step1["no_occ"]             # all NO occupations
mo_list          = step1["mo_list"]            # final active MO indices
n_active_orbs    = step1["n_active_orbs"]
nel              = step1["nel"]
lowdin_weights   = step1["lowdin_weights"]     # (n_active, n_atoms)
dominant_atoms   = step1["dominant_atoms"]
active_per_atom  = step1["active_per_atom"]
most_active_atom = step1["most_active_atom"]
scores_s1        = step1["scores"]
mol_info         = step1["mol_info"]

# Step 2
h1e       = step2["h1e"]
h2e       = step2["h2e"]
n_emb     = step2["n_emb"]
n_imp     = step2["n_imp"]
n_bath    = step2["n_bath"]
n_alpha   = step2["n_alpha"]
n_beta    = step2["n_beta"]
fci_ref_e = step2["fci_ref_e"]
sv        = step2["sv"]                        # bath singular values
scores_s2 = step2["scores"]

# Step 3
solver_name  = step3["solver"]
final_energy = step3["energy"]
error_vs_fci = step3["error_vs_fci"]
iterations   = step3["iterations"]
spin_sq      = step3["spin_sq"]
converged    = step3["converged"]
pipeline_score = step3.get("pipeline_score", {})

molecule  = mol_info["molecule"]
atom_syms = mol_info["atom_syms"]
n_atoms   = mol_info["n_atoms"]
n_qubits  = 2 * n_emb

print(f"  Molecule  : {molecule}")
print(f"  Solver    : {solver_name.upper()}")
print(f"  Active    : {nel}e in {n_active_orbs} orbs")
print(f"  Embedding : {n_emb} orbs = {n_qubits} qubits")
print(f"  FCI ref   : {fci_ref_e:.8f} Ha" if fci_ref_e else "  FCI ref: N/A")
print(f"  Final E   : {final_energy:.8f} Ha" if final_energy else "  Final E: N/A")
print(f"  ΔE        : {error_vs_fci:.2e} Ha" if error_vs_fci else "  ΔE: N/A")

# ═════════════════════════════════════════════════════════════════════════════
# Shared style
# ═════════════════════════════════════════════════════════════════════════════
STYLE = {
    "active_color"   : "#2196F3",   # blue
    "core_color"     : "#4CAF50",   # green
    "virtual_color"  : "#9E9E9E",   # grey
    "bath_color"     : "#FF9800",   # orange
    "fci_color"      : "#F44336",   # red
    "solver_color"   : "#2196F3",   # blue
    "gap_color"      : "#E91E63",   # pink
    "bg_color"       : "#FAFAFA",
    "grid_color"     : "#E0E0E0",
    "title_size"     : 13,
    "label_size"     : 10,
    "tick_size"      : 8,
    "dpi"            : 150,
}

plt.rcParams.update({
    "font.family"        : "DejaVu Sans",
    "axes.facecolor"     : STYLE["bg_color"],
    "figure.facecolor"   : "white",
    "axes.grid"          : True,
    "grid.color"         : STYLE["grid_color"],
    "grid.linewidth"     : 0.6,
    "axes.spines.top"    : False,
    "axes.spines.right"  : False,
})


def save_fig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved  {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 1 — Active Space Analysis
# ═════════════════════════════════════════════════════════════════════════════
print("\n[Fig 1] Active Space Analysis...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f"{molecule}  |  Active Space Analysis  "
    f"({nel}e in {n_active_orbs} orbs)",
    fontsize=STYLE["title_size"] + 1, fontweight="bold", y=1.02
)

active_set = set(mo_list)
n_mo       = len(deviation)
x_all      = np.arange(n_mo)

# ── 1a: Deviation spectrum ────────────────────────────────────────────────────
ax = axes[0]
colors_dev = [
    STYLE["active_color"]  if i in active_set else
    STYLE["core_color"]    if no_occ[i] > config.CORE_OCC_THRESHOLD else
    STYLE["virtual_color"]
    for i in range(n_mo)
]
bars = ax.bar(x_all, deviation, color=colors_dev, edgecolor="none", alpha=0.85)

# Mark active orbitals with a star
for mo_idx in mo_list:
    ax.annotate(
        "★", (mo_idx, deviation[mo_idx] + 0.01),
        ha="center", va="bottom", fontsize=8,
        color=STYLE["active_color"]
    )

ax.set_xlabel("MO index",        fontsize=STYLE["label_size"])
ax.set_ylabel("Deviation min(n, 2-n)", fontsize=STYLE["label_size"])
ax.set_title("MP2 NO Deviation Spectrum", fontsize=STYLE["title_size"])
ax.set_ylim(0, 1.05)

legend_patches = [
    mpatches.Patch(color=STYLE["active_color"],  label="Active (★)"),
    mpatches.Patch(color=STYLE["core_color"],    label="Core (occ>1.8)"),
    mpatches.Patch(color=STYLE["virtual_color"], label="Virtual"),
]
ax.legend(handles=legend_patches, fontsize=STYLE["tick_size"])
ax.tick_params(labelsize=STYLE["tick_size"])

# ── 1b: NO Occupations ───────────────────────────────────────────────────────
ax = axes[1]
colors_occ = [
    STYLE["active_color"]  if i in active_set else
    STYLE["core_color"]    if no_occ[i] > config.CORE_OCC_THRESHOLD else
    STYLE["virtual_color"]
    for i in range(n_mo)
]
ax.bar(x_all, no_occ, color=colors_occ, edgecolor="none", alpha=0.85)
ax.axhline(config.CORE_OCC_THRESHOLD, color=STYLE["core_color"],
           linestyle="--", linewidth=1.5, label=f"Core threshold ({config.CORE_OCC_THRESHOLD})")
ax.axhline(1.0, color=STYLE["gap_color"],
           linestyle=":", linewidth=1.2, label="Half-filled (1.0)")

ax.set_xlabel("MO index",      fontsize=STYLE["label_size"])
ax.set_ylabel("NO occupation", fontsize=STYLE["label_size"])
ax.set_title("Natural Orbital Occupations", fontsize=STYLE["title_size"])
ax.set_ylim(0, 2.2)
ax.legend(fontsize=STYLE["tick_size"])
ax.tick_params(labelsize=STYLE["tick_size"])

# Annotate active orbitals
for mo_idx in mo_list:
    ax.annotate(
        f"{mo_idx}", (mo_idx, no_occ[mo_idx] + 0.05),
        ha="center", va="bottom", fontsize=7,
        color=STYLE["active_color"], fontweight="bold"
    )

# ── 1c: Gap landscape ────────────────────────────────────────────────────────
ax = axes[2]
# Show deviation values for candidate orbitals (from ASF pool)
cand_devs = sorted(
    [(i, deviation[i]) for i in range(n_mo) if deviation[i] > 0.001],
    key=lambda x: -x[1]
)
if len(cand_devs) > 0:
    cand_indices  = [c[0] for c in cand_devs]
    cand_dev_vals = [c[1] for c in cand_devs]
    x_cand        = np.arange(len(cand_devs))

    # Compute gaps between consecutive deviation values
    gaps = []
    for k in range(len(cand_dev_vals) - 1):
        gaps.append(cand_dev_vals[k] - cand_dev_vals[k+1])
    gaps.append(cand_dev_vals[-1])   # gap to zero at end

    bar_colors = [STYLE["active_color"]] * len(gaps)
    if len(gaps) > 0:
        best_gap_pos = int(np.argmax(gaps))
        bar_colors[best_gap_pos] = STYLE["gap_color"]

    ax.bar(x_cand, gaps, color=bar_colors, edgecolor="none", alpha=0.85)
    ax.set_xticks(x_cand)
    ax.set_xticklabels(
        [str(c[0]) for c in cand_devs],
        rotation=45, ha="right", fontsize=7
    )
    ax.set_xlabel("MO (sorted by deviation)", fontsize=STYLE["label_size"])
    ax.set_ylabel("Gap to next",              fontsize=STYLE["label_size"])
    ax.set_title("Gap Detection Landscape\n(pink = largest gap = cutoff)",
                 fontsize=STYLE["title_size"])
    ax.tick_params(labelsize=STYLE["tick_size"])

plt.tight_layout()
save_fig(fig, "fig1_active_space.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 2 — Löwdin Population Analysis
# ═════════════════════════════════════════════════════════════════════════════
print("[Fig 2] Löwdin Population Analysis...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f"{molecule}  |  Löwdin Population Analysis  "
    f"({n_active_orbs} active orbitals)",
    fontsize=STYLE["title_size"] + 1, fontweight="bold", y=1.02
)

# ── 2a: Heatmap ───────────────────────────────────────────────────────────────
ax = axes[0]
if lowdin_weights.shape[0] > 0 and lowdin_weights.shape[1] > 0:
    cmap = cm.Blues
    im   = ax.imshow(lowdin_weights, cmap=cmap, aspect="auto",
                     vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Löwdin weight", shrink=0.8)

    ax.set_xticks(range(n_atoms))
    ax.set_xticklabels(
        [f"{i}\n{atom_syms[i]}" for i in range(n_atoms)],
        fontsize=STYLE["tick_size"]
    )
    ax.set_yticks(range(n_active_orbs))
    ax.set_yticklabels(
        [f"MO {mo_list[k]}" for k in range(n_active_orbs)],
        fontsize=STYLE["tick_size"]
    )
    ax.set_xlabel("Atom index",    fontsize=STYLE["label_size"])
    ax.set_ylabel("Active orbital", fontsize=STYLE["label_size"])
    ax.set_title("Orbital–Atom Weight Heatmap", fontsize=STYLE["title_size"])

    # Annotate cells with value if > 0.1
    for k in range(n_active_orbs):
        for j in range(n_atoms):
            w = lowdin_weights[k, j]
            if w > 0.1:
                ax.text(j, k, f"{w:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if w > 0.6 else "black")

# ── 2b: Active orbital count per atom ────────────────────────────────────────
ax = axes[1]
x_atoms = np.arange(n_atoms)
atom_labels = [f"{atom_syms[i]}\n({i})" for i in range(n_atoms)]
bar_colors  = [
    STYLE["active_color"] if i == most_active_atom else
    "#90CAF9"
    for i in range(n_atoms)
]
bars = ax.bar(x_atoms, active_per_atom, color=bar_colors, edgecolor="none", alpha=0.9)

for bar, cnt in zip(bars, active_per_atom):
    if cnt > 0:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            str(cnt),
            ha="center", va="bottom", fontsize=STYLE["tick_size"],
            fontweight="bold"
        )

ax.set_xticks(x_atoms)
ax.set_xticklabels(atom_labels, fontsize=STYLE["tick_size"])
ax.set_ylabel("Active orbital count", fontsize=STYLE["label_size"])
ax.set_title(
    "Active Orbitals per Atom\n(blue = most correlated)",
    fontsize=STYLE["title_size"]
)
ax.tick_params(labelsize=STYLE["tick_size"])

# ── 2c: Per-orbital dominant atom + deviation ─────────────────────────────────
ax = axes[2]
x_orbs  = np.arange(n_active_orbs)
orb_devs = [deviation[mo_list[k]] if mo_list[k] < len(deviation) else 0.0
            for k in range(n_active_orbs)]

scatter_colors = [
    plt.cm.tab10(dominant_atoms[k] % 10)
    for k in range(n_active_orbs)
]
sc = ax.scatter(x_orbs, orb_devs, c=scatter_colors, s=120, zorder=5,
                edgecolors="grey", linewidths=0.5)
ax.bar(x_orbs, orb_devs, color=scatter_colors, alpha=0.3, edgecolor="none")

for k in range(n_active_orbs):
    da  = dominant_atoms[k]
    ax.annotate(
        f"{atom_syms[da]}",
        (x_orbs[k], orb_devs[k] + 0.02),
        ha="center", va="bottom", fontsize=8, fontweight="bold"
    )

ax.set_xticks(x_orbs)
ax.set_xticklabels([f"MO {mo_list[k]}" for k in range(n_active_orbs)],
                   rotation=45, ha="right", fontsize=STYLE["tick_size"])
ax.set_ylabel("Deviation proxy", fontsize=STYLE["label_size"])
ax.set_ylim(0, 1.1)
ax.set_title("Per-Orbital Deviation + Dominant Atom",
             fontsize=STYLE["title_size"])

# Legend for atoms
seen_atoms = sorted(set(dominant_atoms))
legend_elements = [
    mpatches.Patch(
        color=plt.cm.tab10(a % 10),
        label=f"Atom {a} ({atom_syms[a]})"
    )
    for a in seen_atoms
]
ax.legend(handles=legend_elements, fontsize=STYLE["tick_size"],
          loc="upper right")

plt.tight_layout()
save_fig(fig, "fig2_lowdin.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 3 — DMET Embedding Quality
# ═════════════════════════════════════════════════════════════════════════════
print("[Fig 3] DMET Embedding Quality...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f"{molecule}  |  DMET Embedding  "
    f"(imp={n_imp}  bath={n_bath}  total={n_emb} orbs = {n_qubits} qubits)",
    fontsize=STYLE["title_size"] + 1, fontweight="bold", y=1.02
)

# ── 3a: Schmidt singular values ───────────────────────────────────────────────
ax = axes[0]
sv_vals = np.asarray(sv)
x_sv    = np.arange(len(sv_vals))

sv_colors = [STYLE["bath_color"]] * len(sv_vals)
ax.bar(x_sv, sv_vals, color=sv_colors, edgecolor="none", alpha=0.85)
ax.axhline(float(config.BATH_TOLERANCE),
           color=STYLE["fci_color"], linestyle="--", linewidth=1.5,
           label=f"BATH_TOLERANCE = {config.BATH_TOLERANCE:.0e}")

ax.set_xlabel("Bath orbital index", fontsize=STYLE["label_size"])
ax.set_ylabel("Schmidt singular value", fontsize=STYLE["label_size"])
ax.set_title("Schmidt Singular Values\n(kept bath orbitals)", fontsize=STYLE["title_size"])
ax.legend(fontsize=STYLE["tick_size"])
ax.tick_params(labelsize=STYLE["tick_size"])

for i, v in enumerate(sv_vals):
    ax.text(i, v + max(sv_vals) * 0.01, f"{v:.3f}",
            ha="center", va="bottom", fontsize=7)

# ── 3b: Cumulative sv² coverage ───────────────────────────────────────────────
ax = axes[1]
sv2_total = float(np.sum(sv_vals ** 2))
sv2_cumul = np.cumsum(sv_vals ** 2) / sv2_total if sv2_total > 0 else np.zeros(len(sv_vals))

ax.plot(x_sv, sv2_cumul * 100, "o-",
        color=STYLE["bath_color"], linewidth=2, markersize=6)
ax.fill_between(x_sv, sv2_cumul * 100, alpha=0.2, color=STYLE["bath_color"])
ax.axhline(99.9, color=STYLE["fci_color"], linestyle="--", linewidth=1.5,
           label="99.9% threshold")
ax.axhline(float(scores_s2.get("sv2_coverage", 1.0)) * 100,
           color=STYLE["active_color"], linestyle=":", linewidth=1.5,
           label=f"Actual coverage = {scores_s2.get('sv2_coverage',1.0)*100:.1f}%")

ax.set_xlabel("Bath orbitals included", fontsize=STYLE["label_size"])
ax.set_ylabel("Cumulative sv² coverage (%)", fontsize=STYLE["label_size"])
ax.set_title("Cumulative Entanglement Coverage", fontsize=STYLE["title_size"])
ax.set_ylim(0, 105)
ax.legend(fontsize=STYLE["tick_size"])
ax.tick_params(labelsize=STYLE["tick_size"])

# ── 3c: Electron count decomposition ─────────────────────────────────────────
ax = axes[2]
total_e    = mol_info["n_electrons"]
active_e   = nel
core_e     = total_e - active_e
virtual_e  = 0   # virtual orbitals are empty by definition

categories = ["Core\n(frozen)", "Active\n(quantum)", "Total"]
values     = [core_e, active_e, total_e]
bar_cols   = [STYLE["core_color"], STYLE["active_color"], "#607D8B"]
bars = ax.bar(categories, values, color=bar_cols, edgecolor="none",
              alpha=0.85, width=0.5)

for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.2,
        str(val),
        ha="center", va="bottom", fontsize=11, fontweight="bold"
    )

ax.set_ylabel("Electrons", fontsize=STYLE["label_size"])
ax.set_title(
    f"Electron Count Decomposition\n"
    f"Total={total_e}  Core={core_e}  Active={active_e}",
    fontsize=STYLE["title_size"]
)
ax.set_ylim(0, total_e * 1.25)
ax.tick_params(labelsize=STYLE["tick_size"])

# Annotate qubit count
ax.text(
    1, active_e / 2,
    f"{n_qubits} qubits\n({n_emb} orbs × 2)",
    ha="center", va="center", fontsize=9,
    color="white", fontweight="bold"
)

plt.tight_layout()
save_fig(fig, "fig3_embedding.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 4 — Solver Convergence
# ═════════════════════════════════════════════════════════════════════════════
print("[Fig 4] Solver Convergence...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f"{molecule}  |  {solver_name.upper()} Convergence  "
    f"(converged = {converged})",
    fontsize=STYLE["title_size"] + 1, fontweight="bold", y=1.02
)

# Parse iterations — handle both SQD (iter key) and SKQD (k key)
valid_iters = [it for it in iterations if it.get("energy") is not None]
x_key       = "k" if "k" in (valid_iters[0] if valid_iters else {}) else "iter"
x_vals      = [it[x_key] for it in valid_iters]
energies    = [it["energy"] for it in valid_iters]
n_configs   = [it["n_configs"] for it in valid_iters]
deltas      = [it.get("delta") for it in valid_iters]
spin_sqs    = [it.get("spin_sq", 0.0) for it in valid_iters]

x_label = "Krylov dimension k" if x_key == "k" else "Iteration"

# ── 4a: Absolute energy ───────────────────────────────────────────────────────
ax = axes[0]
if energies:
    ax.plot(x_vals, energies, "o-",
            color=STYLE["solver_color"], linewidth=2, markersize=6,
            label=solver_name.upper(), zorder=5)

    if fci_ref_e is not None:
        ax.axhline(fci_ref_e, color=STYLE["fci_color"],
                   linestyle="--", linewidth=2, label=f"FCI = {fci_ref_e:.6f} Ha")

    # Shade region between solver and FCI
    if fci_ref_e is not None:
        ax.fill_between(x_vals, energies, fci_ref_e,
                        alpha=0.12, color=STYLE["solver_color"])

    ax.set_xlabel(x_label,         fontsize=STYLE["label_size"])
    ax.set_ylabel("Energy (Ha)",   fontsize=STYLE["label_size"])
    ax.set_title("Energy Convergence", fontsize=STYLE["title_size"])
    ax.legend(fontsize=STYLE["tick_size"])
    ax.tick_params(labelsize=STYLE["tick_size"])

    # Annotate final value
    ax.annotate(
        f"Final: {energies[-1]:.6f} Ha",
        xy=(x_vals[-1], energies[-1]),
        xytext=(-60, -25), textcoords="offset points",
        fontsize=8, color=STYLE["solver_color"],
        arrowprops=dict(arrowstyle="->", color=STYLE["solver_color"])
    )

# ── 4b: ΔE vs FCI (log scale) ────────────────────────────────────────────────
ax = axes[1]
if deltas and any(d is not None and d > 0 for d in deltas):
    valid_deltas = [(x, d) for x, d in zip(x_vals, deltas)
                   if d is not None and d > 0]
    xd = [v[0] for v in valid_deltas]
    yd = [v[1] for v in valid_deltas]

    ax.semilogy(xd, yd, "s-",
                color=STYLE["gap_color"], linewidth=2, markersize=6,
                label="|E - E_FCI|")
    ax.axhline(1e-3, color=STYLE["fci_color"], linestyle="--", linewidth=1.5,
               label="1 mHa threshold")
    ax.axhline(1e-4, color="purple", linestyle=":", linewidth=1.2,
               label="0.1 mHa (chemical accuracy)")

    ax.set_xlabel(x_label,       fontsize=STYLE["label_size"])
    ax.set_ylabel("|ΔE| (Ha)",   fontsize=STYLE["label_size"])
    ax.set_title("Error vs FCI (log scale)", fontsize=STYLE["title_size"])
    ax.legend(fontsize=STYLE["tick_size"])
    ax.tick_params(labelsize=STYLE["tick_size"])

    # Color background: green if below 1 mHa
    if yd[-1] < 1e-3:
        ax.set_facecolor("#E8F5E9")

# ── 4c: Subspace configuration count ─────────────────────────────────────────
ax = axes[2]
if n_configs:
    import math as _math
    max_possible = _math.comb(n_emb, n_alpha) ** 2

    ax.plot(x_vals, n_configs, "D-",
            color=STYLE["bath_color"], linewidth=2, markersize=6,
            label="Configs in subspace")
    ax.axhline(max_possible, color=STYLE["fci_color"],
               linestyle="--", linewidth=1.5,
               label=f"Full space = {max_possible:,}")

    ax.fill_between(x_vals, n_configs, alpha=0.15, color=STYLE["bath_color"])

    coverage_pct = n_configs[-1] / max_possible * 100 if max_possible > 0 else 0
    ax.set_xlabel(x_label,             fontsize=STYLE["label_size"])
    ax.set_ylabel("Configurations",    fontsize=STYLE["label_size"])
    ax.set_title(
        f"Subspace Growth\n(final = {n_configs[-1]:,} / {max_possible:,} = "
        f"{coverage_pct:.1f}%)",
        fontsize=STYLE["title_size"]
    )
    ax.legend(fontsize=STYLE["tick_size"])
    ax.tick_params(labelsize=STYLE["tick_size"])

plt.tight_layout()
save_fig(fig, "fig4_convergence.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 5 — Molecular Score Dashboard
# ═════════════════════════════════════════════════════════════════════════════
print("[Fig 5] Molecular Score Dashboard...")

fig = plt.figure(figsize=(20, 7))
fig.suptitle(
    f"{molecule}  |  Molecular Score Dashboard",
    fontsize=STYLE["title_size"] + 2, fontweight="bold", y=1.01
)

gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# ── 5a: Radar chart ───────────────────────────────────────────────────────────
ax_radar = fig.add_subplot(gs[0], polar=True)

radar_metrics = {
    "Correlation\nstrength"  : float(scores_s1.get("correlation_strength", 0)),
    "Max\ncorrelation"       : float(scores_s1.get("max_correlation", 0)),
    "Entropy\ngap"           : min(float(scores_s1.get("entropy_gap", 0)) * 5, 1.0),
    "Bath\ncoverage"         : float(scores_s2.get("sv2_coverage", 0)),
    "Metal\nfraction"        : float(scores_s1.get("metal_fraction", 0)),
    "MP2\nused"              : 1.0 if scores_s2.get("mp2_dm_used") else 0.0,
}

categories   = list(radar_metrics.keys())
values_radar = list(radar_metrics.values())
N            = len(categories)
angles       = [n / float(N) * 2 * np.pi for n in range(N)]
angles      += angles[:1]
values_radar += values_radar[:1]

ax_radar.set_theta_offset(np.pi / 2)
ax_radar.set_theta_direction(-1)
ax_radar.set_rlabel_position(0)

ax_radar.plot(angles, values_radar, "o-",
              linewidth=2, color=STYLE["active_color"])
ax_radar.fill(angles, values_radar, alpha=0.25, color=STYLE["active_color"])
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(categories, size=STYLE["tick_size"])
ax_radar.set_ylim(0, 1.0)
ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
ax_radar.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], size=6)
ax_radar.set_title("Normalised Score Radar", fontsize=STYLE["title_size"],
                   pad=15)

# ── 5b: All pipeline scores horizontal bar ────────────────────────────────────
ax_bar = fig.add_subplot(gs[1])

bar_metrics = {
    "UHF energy (Ha)"          : abs(float(scores_s1.get("uhf_energy_Ha", 0))),
    "HOMO-LUMO gap (eV)"       : float(scores_s1.get("homo_lumo_gap_eV", 0)),
    "Spin contamination"        : float(scores_s1.get("spin_contamination", 1)),
    "MP2 correlation (Ha)"     : abs(float(scores_s1.get("mp2_correlation_energy", 0))),
    "Corr strength"            : float(scores_s1.get("correlation_strength", 0)),
    "Entropy gap"              : float(scores_s1.get("entropy_gap", 0)),
    "Bath fraction"            : float(scores_s2.get("bath_fraction", 0)),
    "SV² coverage"             : float(scores_s2.get("sv2_coverage", 0)),
    "ΔE vs FCI (mHa)"          : (error_vs_fci * 1000 if error_vs_fci else 0),
}

keys   = list(bar_metrics.keys())
vals   = list(bar_metrics.values())
y_pos  = np.arange(len(keys))

norm_vals = []
for v in vals:
    if max(vals) > 0:
        norm_vals.append(v / max(vals))
    else:
        norm_vals.append(0.0)

bar_h = ax_bar.barh(y_pos, norm_vals, color=STYLE["active_color"],
                    alpha=0.75, edgecolor="none")
ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(keys, fontsize=STYLE["tick_size"])
ax_bar.set_xlabel("Normalised value", fontsize=STYLE["label_size"])
ax_bar.set_title("Pipeline Score Vector\n(normalised to max)", fontsize=STYLE["title_size"])
ax_bar.set_xlim(0, 1.15)
ax_bar.tick_params(labelsize=STYLE["tick_size"])

# Annotate raw values
for i, (bar, raw) in enumerate(zip(bar_h, vals)):
    ax_bar.text(
        bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
        f"{raw:.3g}", va="center", fontsize=7, color="#333333"
    )

# ── 5c: Tier classification card ─────────────────────────────────────────────
ax_card = fig.add_subplot(gs[2])
ax_card.set_xlim(0, 10)
ax_card.set_ylim(0, 10)
ax_card.axis("off")

tier       = int(scores_s1.get("tier_used", 1))
tier_color = ["#4CAF50", "#FF9800", "#F44336"][tier - 1]
tier_label = ["Simple organic", "Moderate correlation", "Strongly correlated"][tier - 1]

# Draw tier badge
badge = mpatches.FancyBboxPatch(
    (1, 7), 8, 2.2,
    boxstyle="round,pad=0.1",
    linewidth=2, edgecolor=tier_color, facecolor=tier_color + "33"
)
ax_card.add_patch(badge)
ax_card.text(5, 8.4, f"TIER  {tier}",
             ha="center", va="center", fontsize=22, fontweight="bold",
             color=tier_color)
ax_card.text(5, 7.4, tier_label,
             ha="center", va="center", fontsize=10, color=tier_color)

# Summary items
items = [
    ("Molecule",        molecule),
            ("Basis",           mol_info.get("basis", "?")),
    ("Electrons",       f"{mol_info['n_electrons']}  (active: {nel})"),
    ("AOs / MOs",       f"{mol_info['n_ao']}"),
    ("Qubits",          str(n_qubits)),
    ("Solver",          solver_name.upper()),
    ("FCI energy",      f"{fci_ref_e:.6f} Ha" if fci_ref_e else "N/A"),
    ("Final energy",    f"{final_energy:.6f} Ha" if final_energy else "N/A"),
    ("ΔE vs FCI",       f"{error_vs_fci*1000:.2f} mHa" if error_vs_fci else "N/A"),
    ("Converged",       "✓ Yes" if converged else "✗ No"),
]

y_start = 6.8
for label, val in items:
    y_start -= 0.62
    ax_card.text(1.2, y_start, f"{label}:", fontsize=8.5,
                 color="#555555", va="center")
    ax_card.text(5.5, y_start, val, fontsize=8.5,
                 color="#111111", fontweight="bold", va="center")

plt.tight_layout()
save_fig(fig, "fig5_scores.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 6 — Summary Table
# ═════════════════════════════════════════════════════════════════════════════
print("[Fig 6] Summary Table...")

fig, ax = plt.subplots(figsize=(14, 10))
ax.axis("off")
fig.suptitle(
    f"{molecule}  |  Pipeline Summary Table",
    fontsize=STYLE["title_size"] + 2, fontweight="bold"
)

table_data = [
    # Section header rows are marked with empty second column → styled differently
    ["── MOLECULE ──", ""],
    ["Molecule",             molecule],
    ["Basis",                mol_info.get("basis", "?")],
    ["Total electrons",      str(mol_info["n_electrons"])],
    ["AO basis functions",   str(mol_info["n_ao"])],
    ["Atoms",                "  ".join(f"{s}({i})" for i, s in enumerate(atom_syms))],

    ["── COMPLEXITY ──", ""],
    ["Complexity class",     f"{scores_s1.get('complexity_class','?')}  "
                              f"({'simple organic' if scores_s1.get('complexity_class')==1 else 'moderate' if scores_s1.get('complexity_class')==2 else 'strongly correlated'})"],
    ["Tier used",            str(scores_s1.get("tier_used","?"))],
    ["Has TM element",       str(scores_s1.get("has_tm","?"))],
    ["UHF energy",           f"{scores_s1.get('uhf_energy_Ha',0):.8f} Ha"],
    ["HOMO-LUMO gap",        f"{scores_s1.get('homo_lumo_gap_eV',0):.4f} eV"],
    ["Spin contamination",   f"{scores_s1.get('spin_contamination',0):.4f}"],
    ["⟨S²⟩ actual",         f"{scores_s1.get('s2_actual',0):.4f}"],
    ["n_SOMO",               str(scores_s1.get("n_somo","?"))],

    ["── ACTIVE SPACE (Step 1) ──", ""],
    ["Active electrons",     str(nel)],
    ["Active orbitals",      f"{n_active_orbs}  → MO list: {mo_list}"],
    ["Entropy gap",          f"{scores_s1.get('entropy_gap',0):.6f}"],
    ["Correlation strength", f"{scores_s1.get('correlation_strength',0):.4f}  "
                              f"(0=uncorrelated, 1=max)"],
    ["Max correlation",      f"{scores_s1.get('max_correlation',0):.4f}"],
    ["MP2 correlation E",    f"{scores_s1.get('mp2_correlation_energy',0):.6f} Ha"],
    ["Most active atom",     f"{most_active_atom} ({atom_syms[most_active_atom]})"],
    ["Metal fraction",       f"{scores_s1.get('metal_fraction',0):.4f}"],

    ["── EMBEDDING (Step 2) ──", ""],
    ["Impurity orbitals",    str(n_imp)],
    ["Bath orbitals",        str(n_bath)],
    ["Embedding orbitals",   str(n_emb)],
    ["Qubits",               str(n_qubits)],
    ["n_alpha / n_beta",     f"{n_alpha} / {n_beta}"],
    ["SV² bath coverage",    f"{scores_s2.get('sv2_coverage',0):.6f}"],
    ["Bath fraction",        f"{scores_s2.get('bath_fraction',0):.4f}"],
    ["MP2 DM used",          str(scores_s2.get("mp2_dm_used","?"))],
    ["Electron deviation",   f"{scores_s2.get('electron_deviation',0):.6f}"],
    ["μ correction",         f"{scores_s2.get('mu_correction',0):+.6f} Ha"],
    ["ecore",                f"{scores_s2.get('ecore',0):.6f} Ha"],
    ["Embed corr energy",    f"{scores_s2.get('embedding_corr_energy','N/A')}"],

    ["── SOLVER (Step 3) ──", ""],
    ["Solver",               solver_name.upper()],
    ["FCI reference",        f"{fci_ref_e:.8f} Ha" if fci_ref_e else "N/A"],
    ["Final energy",         f"{final_energy:.8f} Ha" if final_energy else "N/A"],
    ["ΔE vs FCI",            f"{error_vs_fci:.2e} Ha  ({error_vs_fci*1000:.3f} mHa)"
                              if error_vs_fci else "N/A"],
    ["Final ⟨S²⟩",          f"{spin_sq:.6f}" if spin_sq is not None else "N/A"],
    ["Iterations run",       str(len(valid_iters))],
    ["Configs (final)",      f"{n_configs[-1]:,}" if n_configs else "N/A"],
    ["Converged (< 1 mHa)", "✓ Yes" if converged else "✗ No"],
]

col_labels = ["Parameter", "Value"]
cell_text  = [row for row in table_data]

col_widths = [0.35, 0.65]
y_pos      = 0.98
x_param    = 0.02
x_value    = 0.38

HEADER_BG  = "#1565C0"
SECTION_BG = "#E3F2FD"
ROW_BG_A   = "#FFFFFF"
ROW_BG_B   = "#F5F5F5"

ax.text(x_param, y_pos + 0.01, "Parameter",
        transform=ax.transAxes, fontsize=9, fontweight="bold",
        color="white", va="bottom",
        bbox=dict(boxstyle="square,pad=0.3", facecolor=HEADER_BG, linewidth=0))
ax.text(x_value, y_pos + 0.01, "Value",
        transform=ax.transAxes, fontsize=9, fontweight="bold",
        color="white", va="bottom",
        bbox=dict(boxstyle="square,pad=0.3", facecolor=HEADER_BG, linewidth=0))

row_height = 0.026
y_cur      = y_pos - row_height
row_count  = 0

for param, value in cell_text:
    is_section = value == ""
    bg_color   = (SECTION_BG if is_section else
                  ROW_BG_A   if row_count % 2 == 0 else
                  ROW_BG_B)

    rect = mpatches.FancyBboxPatch(
        (0.0, y_cur - 0.002), 1.0, row_height,
        transform=ax.transAxes,
        boxstyle="square,pad=0", linewidth=0,
        facecolor=bg_color
    )
    ax.add_patch(rect)

    if is_section:
        ax.text(0.02, y_cur + row_height * 0.35, param,
                transform=ax.transAxes, fontsize=8.5, fontweight="bold",
                color=HEADER_BG, va="center")
    else:
        ax.text(x_param, y_cur + row_height * 0.35, param,
                transform=ax.transAxes, fontsize=8, color="#333333", va="center")
        ax.text(x_value, y_cur + row_height * 0.35, str(value),
                transform=ax.transAxes, fontsize=8, color="#111111",
                fontweight="bold", va="center")
        row_count += 1

    y_cur -= row_height

plt.tight_layout()
save_fig(fig, "fig6_summary.png")


# ═════════════════════════════════════════════════════════════════════════════
# CSV Export
# ═════════════════════════════════════════════════════════════════════════════
print("\n[CSV] Exporting data...")

csv_rows = []

# Iteration data
for it in iterations:
    row = {
        "molecule"    : molecule,
        "solver"      : solver_name,
        "step"        : it.get("iter", it.get("k", "?")),
        "energy_Ha"   : it.get("energy"),
        "delta_Ha"    : it.get("delta"),
        "n_configs"   : it.get("n_configs"),
        "spin_sq"     : it.get("spin_sq"),
    }
    csv_rows.append(row)

if csv_rows:
    fieldnames = list(csv_rows[0].keys())
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  ✓ Saved  {CSV_FILE}")

# ═════════════════════════════════════════════════════════════════════════════
# Text Summary
# ═════════════════════════════════════════════════════════════════════════════
print("[TXT] Writing text summary...")

lines = [
    "=" * 65,
    f"QuEnAIS Pipeline Summary — {molecule}",
    "=" * 65,
    "",
    f"  Molecule          : {molecule}",
    f"  Basis             : {mol_info.get('basis','?')}",
    f"  Total electrons   : {mol_info['n_electrons']}",
    f"  AO functions      : {mol_info['n_ao']}",
    "",
    f"  Complexity tier   : {scores_s1.get('tier_used','?')}",
    f"  UHF energy        : {scores_s1.get('uhf_energy_Ha',0):.8f} Ha",
    f"  HOMO-LUMO gap     : {scores_s1.get('homo_lumo_gap_eV',0):.4f} eV",
    f"  Spin contamination: {scores_s1.get('spin_contamination',0):.4f}",
    "",
    f"  Active space      : {nel}e in {n_active_orbs} orbs  {mo_list}",
    f"  Corr strength     : {scores_s1.get('correlation_strength',0):.4f}",
    f"  Most active atom  : {most_active_atom} ({atom_syms[most_active_atom]})",
    f"  Metal fraction    : {scores_s1.get('metal_fraction',0):.4f}",
    "",
    f"  DMET imp + bath   : {n_imp} + {n_bath} = {n_emb} orbs = {n_qubits} qubits",
    f"  SV² coverage      : {scores_s2.get('sv2_coverage',0):.6f}",
    f"  MP2 DM used       : {scores_s2.get('mp2_dm_used','?')}",
    "",
    f"  Solver            : {solver_name.upper()}",
    f"  FCI reference     : {f'{fci_ref_e:.8f} Ha' if fci_ref_e else 'N/A'}",
    f"  Final energy      : {f'{final_energy:.8f} Ha' if final_energy else 'N/A'}",
    f"  Delta E vs FCI    : {f'{error_vs_fci:.2e} Ha  ({error_vs_fci*1000:.3f} mHa)' if error_vs_fci else 'N/A'}",
    f"  Final <S^2>       : {f'{spin_sq:.6f}' if spin_sq is not None else 'N/A'}",
    f"  Converged         : {'Yes' if converged else 'No'}",
    "",
    "  Iteration log:",
    "  " + "-" * 60,
]

for it in iterations:
    step   = it.get("iter", it.get("k", "?"))
    e      = it.get("energy")
    d      = it.get("delta")
    ncfg   = it.get("n_configs")
    sq     = it.get("spin_sq")
    e_str  = f"{e:.8f}" if e is not None else "N/A"
    d_str  = f"{d:.2e}"  if d is not None else "N/A"
    sq_str = f"{sq:.4f}" if sq is not None else "N/A"
    lines.append(
        f"  step={step:>3}  E={e_str} Ha  "
        f"ΔE={d_str} Ha  configs={ncfg}  <S²>={sq_str}"
    )

lines += [
    "",
    "=" * 65,
    "Generated by step4_visualize.py",
    "=" * 65,
]

with open(TXT_FILE, "w") as f:
    f.write("\n".join(lines))

print(f"  ✓ Saved  {TXT_FILE}")

# ═════════════════════════════════════════════════════════════════════════════
# Final summary
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"[Step 4] Complete — {molecule}")
print(f"{'='*65}")
print(f"\n  Figures saved to: {FIG_DIR}/")
print(f"    fig1_active_space.png  — MO deviations, occupations, gap landscape")
print(f"    fig2_lowdin.png        — Orbital-atom weight heatmap + bar charts")
print(f"    fig3_embedding.png     — Schmidt SVs, bath coverage, electron count")
print(f"    fig4_convergence.png   — Energy convergence, ΔE, config growth")
print(f"    fig5_scores.png        — Radar chart, score bars, tier summary card")
print(f"    fig6_summary.png       — Printable full-pipeline summary table")
print(f"\n  Data exports:")
print(f"    {CSV_FILE}")
print(f"    {TXT_FILE}")
print(f"\n  Key result:")
print(f"    {solver_name.upper()} energy : "
      f"{final_energy:.8f} Ha" if final_energy else "    N/A")
print(f"    FCI reference  : "
      f"{fci_ref_e:.8f} Ha" if fci_ref_e else "    N/A")
print(f"    ΔE             : "
      f"{error_vs_fci:.2e} Ha  ({error_vs_fci*1000:.3f} mHa)" if error_vs_fci else "    N/A")
print(f"    Converged      : {'✓ Yes' if converged else '✗ No'}")
print(f"{'='*65}")