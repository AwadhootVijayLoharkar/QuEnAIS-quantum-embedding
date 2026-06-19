# step4_visualize.py — Visualization
"""
Produces all diagnostic and results plots for the pipeline.

Plots generated:
  1. Energy comparison bar chart   — all methods side by side
  2. SQD convergence curve         — energy vs iteration
  3. Active orbital deviations     — MP2 deviation per orbital (Phase 1 Step 1)
  4. Bath singular values          — Schmidt SV spectrum (Phase C Step 2)
  5. Löwdin population heatmap     — MO-to-atom weights
  6. (optional) Geometry scan PES  — potential energy surface vs bond length

Usage:
  python step4_visualize.py [--no-scan]
  All plots saved to results/plots/
"""

import os
import sys
import pickle
import argparse
import warnings
import numpy as np

import config

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Step 4: Visualization")
parser.add_argument("--no-scan", action="store_true",
                    help="Skip geometry scan even if GEOMETRY_SCAN=True in config")
args = parser.parse_args()

os.makedirs(config.PLOTS_DIR, exist_ok=True)

# Matplotlib backend — non-interactive for HPC/server environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# Style
plt.rcParams.update({
    "figure.dpi"      : 150,
    "font.size"       : 11,
    "axes.titlesize"  : 12,
    "axes.labelsize"  : 11,
    "legend.fontsize" : 10,
    "axes.spines.top" : False,
    "axes.spines.right": False,
})

# ── Load available results ────────────────────────────────────────────────────
def _load(path, name):
    if not os.path.exists(path):
        print(f"  [WARN] {name} not found: {path} — skipping related plots.")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

step0 = _load(config.STEP0_FILE, "Step 0 (classical)")
step1 = _load(config.STEP1_FILE, "Step 1 (ASF)")
step2 = _load(config.STEP2_FILE, "Step 2 (DMET)")
step3 = _load(config.STEP3_FILE, "Step 3 (quantum solver)")

molecule = config.MOLECULE
print(f"\n[Step 4] Visualization — {molecule}")
print(f"  Output dir: {config.PLOTS_DIR}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Energy Comparison Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════

def plot_energy_comparison():
    """
    Bar chart comparing all available method energies.
    Y-axis: energy relative to HF (correlation energy), in kcal/mol.
    This makes differences between methods clearly visible.
    """
    if step0 is None and step3 is None:
        print("  [Skip] Plot 1: no data available.")
        return

    methods = {}

    # Collect classical methods from step0
    if step0 is not None:
        e_hf = step0["methods"].get("HF", {}).get("energy")
        for name, data in step0["methods"].items():
            e = data.get("energy")
            if e is not None and e_hf is not None:
                methods[name] = e
    else:
        e_hf = step3["uhf_energy"] if step3 else None
        if e_hf:
            methods["HF"] = e_hf

    # Add MP2 from step1 if available
    if step1 is not None and "mp2_energy" in step1:
        methods.setdefault("MP2", step1["mp2_energy"])

    # Add quantum solver result from step3
    if step3 is not None and step3.get("energy") is not None:
        solver_label = (f"{step3['solver'].upper()}\n"
                        f"({step3.get('ansatz','?').upper()}+"
                        f"{step3.get('mapping','?').upper()})")
        methods[solver_label] = step3["energy"]

    if not methods or e_hf is None:
        print("  [Skip] Plot 1: insufficient data.")
        return

    # Convert to correlation energy (vs HF) in kcal/mol
    labels = list(methods.keys())
    corr_e = [(methods[m] - e_hf) * config.HARTREE_TO_KCAL_MOL for m in labels]

    # Color scheme: classical = blue shades, quantum = orange
    colors = []
    classical_set = {"HF", "MP2", "CCSD", "CCSD(T)", "CCSD_T", "CASSCF", "NEVPT2"}
    for lbl in labels:
        base = lbl.split("\n")[0].upper().replace("(", "").replace(")", "")
        if any(c in base for c in classical_set):
            colors.append("#4C72B0")
        else:
            colors.append("#DD8452")

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))

    bars = ax.bar(range(len(labels)), corr_e, color=colors,
                  edgecolor="white", linewidth=0.8, width=0.65)

    # Value labels on bars
    for bar, val in zip(bars, corr_e):
        ypos = bar.get_height() + (0.5 if val >= 0 else -1.5)
        ax.text(bar.get_x() + bar.get_width() / 2., ypos,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Correlation Energy vs HF  (kcal/mol)")
    ax.set_title(f"Energy Comparison — {molecule} / {config.BASIS}")

    legend_elements = [
        Patch(facecolor="#4C72B0", label="Classical methods"),
        Patch(facecolor="#DD8452", label="Quantum solver (DMET)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "plot1_energy_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Plot 1: Energy comparison → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2 — SQD Convergence Curve
# ═══════════════════════════════════════════════════════════════════════════════

def plot_convergence():
    """
    Energy convergence across SQD/SKQD iterations.
    Shows: solver energy per iteration + UHF and MP2 reference lines.
    """
    if step3 is None or not step3.get("iterations"):
        print("  [Skip] Plot 2: no iteration data in Step 3.")
        return

    iters = step3["iterations"]
    if len(iters) < 2:
        print("  [Skip] Plot 2: only 1 iteration — no convergence to plot.")
        return

    # x-axis label depends on solver
    solver = step3["solver"]
    if solver == "skqd":
        x_vals = [it.get("k", it.get("iter", i)) for i, it in enumerate(iters)]
        x_label = "Krylov vector index k"
    else:
        x_vals  = [it["iter"] for it in iters]
        x_label = "SQD iteration"

    energies    = [it["energy"] for it in iters]
    n_configs   = [it["n_configs"] for it in iters]
    uhf_ref     = step3["uhf_energy"]
    mp2_ref     = step3["mp2_energy"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)

    # Energy panel
    ax1.plot(x_vals, energies, "o-", color="#DD8452",
             linewidth=2, markersize=7, label="Quantum solver", zorder=3)
    ax1.axhline(uhf_ref, color="#4C72B0", linewidth=1.5,
                linestyle="--", alpha=0.8, label=f"UHF  ({uhf_ref:.6f} Ha)")
    ax1.axhline(mp2_ref, color="#55A868", linewidth=1.5,
                linestyle="-.", alpha=0.8, label=f"MP2  ({mp2_ref:.6f} Ha)")

    # Mark final energy
    ax1.axhline(energies[-1], color="#DD8452", linewidth=0.8,
                linestyle=":", alpha=0.6)
    ax1.annotate(f"Final: {energies[-1]:.6f} Ha",
                 xy=(x_vals[-1], energies[-1]),
                 xytext=(-60, 15), textcoords="offset points",
                 fontsize=9, color="#DD8452",
                 arrowprops=dict(arrowstyle="->", color="#DD8452", lw=1.0))

    ax1.set_ylabel("Energy (Ha)")
    ax1.set_title(f"SQD Convergence — {molecule}  "
                  f"[{step3['solver'].upper()} + "
                  f"{step3.get('ansatz','?').upper()} + "
                  f"{step3.get('mapping','?').upper()}]")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Config count panel
    ax2.bar(x_vals, n_configs, color="#9B59B6", alpha=0.7, width=0.6)
    ax2.set_ylabel("# configs")
    ax2.set_xlabel(x_label)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "plot2_convergence.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Plot 2: Convergence → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Active Orbital Deviations (Step 1 ASF)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_orbital_deviations():
    """
    Bar chart of MP2 natural orbital deviations.
    dev = min(n_i, 2-n_i): 0 = uncorrelated, 1 = maximally correlated.
    Active orbitals (from gap detection) are highlighted.
    Color = dominant atom for each orbital.
    """
    if step1 is None:
        print("  [Skip] Plot 3: Step 1 data not available.")
        return

    deviation    = step1["deviation"]
    final_mo_list= step1["mo_list"]
    atom_syms    = step1["mol_info"]["atom_syms"]
    dominant     = step1["dominant_atoms"]
    lowdin_w     = step1["lowdin_weights"]
    n_orbs       = len(deviation)

    # Use only orbitals with dev > 0.005 to keep plot readable
    show_mask = deviation > 0.005
    show_orbs = np.where(show_mask)[0]

    if len(show_orbs) == 0:
        print("  [Skip] Plot 3: all deviations < 0.005.")
        return

    active_set = set(final_mo_list)

    # Color by dominant atom
    unique_atoms = list(set(atom_syms))
    cmap         = plt.cm.get_cmap("tab10", len(unique_atoms))
    atom_color   = {sym: cmap(i) for i, sym in enumerate(unique_atoms)}

    fig, ax = plt.subplots(figsize=(max(10, len(show_orbs) * 0.5), 5))

    for k, mo_idx in enumerate(show_orbs):
        dev       = deviation[mo_idx]
        is_active = mo_idx in active_set

        # Dominant atom for this orbital
        if mo_idx < len(dominant):
            atom_idx = dominant[np.searchsorted(final_mo_list, mo_idx)
                                if mo_idx in active_set else 0]
            sym = atom_syms[min(atom_idx, len(atom_syms)-1)]
        else:
            sym = "?"

        color     = atom_color.get(sym, "gray")
        edgecolor = "black" if is_active else "none"
        lw        = 2.0    if is_active else 0.0

        ax.bar(k, dev, color=color, edgecolor=edgecolor,
               linewidth=lw, width=0.8, alpha=0.85 if is_active else 0.5)

        if is_active:
            ax.text(k, dev + 0.01, str(mo_idx), ha="center",
                    va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(show_orbs)))
    ax.set_xticklabels([str(i) for i in show_orbs], rotation=90, fontsize=8)
    ax.set_xlabel("MO index")
    ax.set_ylabel("Deviation  min(n, 2-n)")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"MP2 Natural Orbital Deviations — {molecule}\n"
                 f"(bold border = active space, color = dominant atom)")

    # Legend: atoms
    legend_patches = [Patch(color=atom_color[s], label=s) for s in unique_atoms]
    legend_patches.append(
        Patch(facecolor="white", edgecolor="black", linewidth=2, label="active space")
    )
    ax.legend(handles=legend_patches, loc="upper right", framealpha=0.9)

    # Gap line
    if len(final_mo_list) > 0:
        last_active_k = max(k for k, mo in enumerate(show_orbs) if mo in active_set)
        ax.axvline(last_active_k + 0.5, color="red", linewidth=1.5,
                   linestyle="--", alpha=0.7, label="gap cutoff")

    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "plot3_orbital_deviations.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Plot 3: Orbital deviations → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 4 — Bath Singular Values
# ═══════════════════════════════════════════════════════════════════════════════

def plot_bath_singular_values():
    """
    Scatter + bar plot of Schmidt singular values from DMET embedding.
    Shows which SVs were selected as bath orbitals and the selection threshold.
    """
    if step2 is None:
        print("  [Skip] Plot 4: Step 2 data not available.")
        return

    sv_bath = step2.get("sv", np.array([]))
    n_bath  = step2["n_bath"]
    n_imp   = step2["n_imp"]

    if len(sv_bath) == 0:
        print("  [Skip] Plot 4: no singular values stored.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(sv_bath))
    ax.bar(x, sv_bath, color="#4C72B0", alpha=0.8, width=0.7)
    ax.axhline(config.BATH_TOLERANCE, color="red", linewidth=1.5,
               linestyle="--", label=f"BATH_TOLERANCE = {config.BATH_TOLERANCE:.0e}")

    ax.set_yscale("log")
    ax.set_xlabel("Bath orbital index")
    ax.set_ylabel("Singular value (log scale)")
    ax.set_title(f"Schmidt Singular Values — {molecule}\n"
                 f"Embedding: {n_imp} imp + {n_bath} bath = {n_imp+n_bath} orbs "
                 f"= {2*(n_imp+n_bath)} qubits\n"
                 f"sv² coverage: {step2.get('sv2_cov', 0.0):.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "plot4_bath_svs.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Plot 4: Bath singular values → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 5 — Löwdin Population Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def plot_lowdin_heatmap():
    """
    Heatmap of Löwdin population weights: active MOs × atoms.
    weight[k, atom] = how much MO k is localized on that atom.
    Helps identify which atoms drive the correlation.
    """
    if step1 is None:
        print("  [Skip] Plot 5: Step 1 data not available.")
        return

    weights   = step1["lowdin_weights"]     # shape (n_active, n_atoms)
    mo_list   = step1["mo_list"]
    atom_syms = step1["mol_info"]["atom_syms"]

    if weights.shape[0] == 0:
        print("  [Skip] Plot 5: empty Löwdin weights.")
        return

    fig, ax = plt.subplots(figsize=(max(4, len(atom_syms)), max(4, len(mo_list) * 0.5)))

    im = ax.imshow(weights, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Löwdin weight")

    ax.set_xticks(range(len(atom_syms)))
    ax.set_xticklabels(
        [f"{sym}({i})" for i, sym in enumerate(atom_syms)],
        rotation=30, ha="right"
    )
    ax.set_yticks(range(len(mo_list)))
    ax.set_yticklabels([f"MO {m}" for m in mo_list])
    ax.set_xlabel("Atom")
    ax.set_ylabel("Active MO")
    ax.set_title(f"Löwdin Population — {molecule}\n"
                 f"Active space: ({step1['nel']}e, {step1['n_active_orbs']}orb)")

    # Annotate cells with weight values
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            val = weights[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if val < 0.6 else "white")

    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "plot5_lowdin_heatmap.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Plot 5: Löwdin heatmap → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 6 — Geometry Scan (Potential Energy Surface)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_geometry_scan():
    """
    Potential energy surface: energy vs bond length for the selected atom pair.
    Runs the configured SCAN_METHOD at each geometry.
    Marks equilibrium geometry from the CIF file.

    This plot requires multiple single-point calculations — can be slow.
    Disable with --no-scan flag or GEOMETRY_SCAN=False in config.
    """
    if not config.GEOMETRY_SCAN or args.no_scan:
        print("  [Skip] Plot 6: geometry scan disabled "
              "(GEOMETRY_SCAN=False or --no-scan).")
        return

    from pyscf import gto, scf, mp, cc

    print(f"\n  Running geometry scan: {config.SCAN_METHOD} "
          f"on atoms {config.SCAN_ATOM_PAIR}...")
    print(f"  Distances: {config.SCAN_DISTANCES} Å")

    a_idx, b_idx = config.SCAN_ATOM_PAIR
    base_geom    = list(config.GEOMETRY)

    energies  = []
    distances = []
    failed    = []

    for d in config.SCAN_DISTANCES:
        # Move atom b along x-axis relative to atom a
        sym_a, coord_a = base_geom[a_idx]
        sym_b, _       = base_geom[b_idx]

        new_geom       = list(base_geom)
        new_coord_b    = (coord_a[0] + d, coord_a[1], coord_a[2])
        new_geom[b_idx]= (sym_b, new_coord_b)

        try:
            mol_scan = gto.M(
                atom    = new_geom,
                basis   = config.BASIS,
                charge  = config.CHARGE,
                spin    = config.SPIN,
                verbose = 0,
            )

            mf = scf.RHF(mol_scan) if config.SPIN == 0 else scf.UHF(mol_scan)
            mf.verbose = 0
            mf.kernel()

            method = config.SCAN_METHOD.upper()
            if method == "HF":
                e = float(mf.e_tot)
            elif method == "MP2":
                mymp = mp.MP2(mf)
                mymp.verbose = 0
                mymp.kernel()
                e = float(mf.e_tot + mymp.e_corr)
            elif method == "CCSD":
                mycc = cc.CCSD(mf)
                mycc.verbose = 0
                mycc.kernel()
                e = float(mf.e_tot + mycc.e_corr)
            else:
                e = float(mf.e_tot)

            energies.append(e)
            distances.append(d)
            print(f"    d={d:.3f} Å → {e:.6f} Ha")

        except Exception as ex:
            warnings.warn(f"Scan failed at d={d:.3f} Å: {ex}", RuntimeWarning)
            failed.append(d)

    if len(energies) < 3:
        print("  [Skip] Plot 6: fewer than 3 successful scan points.")
        return

    energies  = np.array(energies)
    distances = np.array(distances)

    # Relative energy in kcal/mol (vs minimum)
    e_min  = energies.min()
    e_rel  = (energies - e_min) * config.HARTREE_TO_KCAL_MOL
    d_eq   = distances[np.argmin(energies)]

    # Equilibrium distance from CIF
    sym_a, c_a = base_geom[a_idx]
    sym_b, c_b = base_geom[b_idx]
    d_cif = float(np.linalg.norm(np.array(c_a) - np.array(c_b)))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(distances, e_rel, "o-", color="#4C72B0",
            linewidth=2, markersize=6, label=f"{config.SCAN_METHOD}")
    ax.axvline(d_eq, color="#DD8452", linewidth=1.5,
               linestyle="--", alpha=0.8,
               label=f"Scan minimum: {d_eq:.3f} Å")
    ax.axvline(d_cif, color="#55A868", linewidth=1.5,
               linestyle="-.", alpha=0.8,
               label=f"CIF distance : {d_cif:.3f} Å")

    ax.set_xlabel(f"{sym_a}–{sym_b} distance (Å)")
    ax.set_ylabel("Relative energy (kcal/mol)")
    ax.set_title(f"Potential Energy Surface — {molecule} / {config.BASIS}\n"
                 f"Scan: {sym_a}({a_idx})–{sym_b}({b_idx}) bond,  "
                 f"method: {config.SCAN_METHOD}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-1)

    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, "plot6_geometry_scan.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Plot 6: Geometry scan (PES) → {path}")

    # Save scan data
    scan_pkl = os.path.join(config.RESULTS_DIR, "geometry_scan.pkl")
    with open(scan_pkl, "wb") as f:
        pickle.dump({
            "distances": distances,
            "energies" : energies,
            "method"   : config.SCAN_METHOD,
            "atom_pair": config.SCAN_ATOM_PAIR,
            "d_eq"     : d_eq,
            "d_cif"    : d_cif,
        }, f)
    print(f"  ✓ Scan data → {scan_pkl}")


# ═══════════════════════════════════════════════════════════════════════════════
# Run all plots
# ═══════════════════════════════════════════════════════════════════════════════

print()
plot_energy_comparison()
plot_convergence()
plot_orbital_deviations()
plot_bath_singular_values()
plot_lowdin_heatmap()
plot_geometry_scan()

print(f"\n[Step 4] All plots saved to: {config.PLOTS_DIR}/")
print(f"  plot1_energy_comparison.png  — method comparison bar chart")
print(f"  plot2_convergence.png        — SQD iteration convergence")
print(f"  plot3_orbital_deviations.png — MP2 NO deviations (active space)")
print(f"  plot4_bath_svs.png           — Schmidt singular values (DMET bath)")
print(f"  plot5_lowdin_heatmap.png     — MO-to-atom population weights")
print(f"  plot6_geometry_scan.png      — PES scan (if enabled)")