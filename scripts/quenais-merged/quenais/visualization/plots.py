"""
Visualization for QuEnAIS pipeline.
"""

import os
import csv
import time
import pickle
import warnings
import numpy as np


def main(cfg, force=False, no_scan=False, no_quantum_scan=False):
    """
    Generate all plots and summary CSV.
    cfg             : quenais.config.Config
    force           : rerun even if plots exist
    no_scan         : skip geometry scan
    no_quantum_scan : skip quantum curve in scan
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({
        "figure.dpi"       : 150,
        "font.size"        : 11,
        "axes.titlesize"   : 12,
        "axes.labelsize"   : 11,
        "legend.fontsize"  : 10,
        "axes.spines.top"  : False,
        "axes.spines.right": False,
    })

    os.makedirs(cfg.plots_dir, exist_ok=True)

    # ── Load results ──────────────────────────────────────────────────────────
    def _load(path, name):
        if not os.path.exists(path):
            print(f"  [WARN] {name} not found: {path} — skipping related plots.")
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    step0 = _load(cfg.step0_file, "Step 0 (classical)")
    step1 = _load(cfg.step1_file, "Step 1 (ASF)")
    step2 = _load(cfg.step2_file, "Step 2 (DMET)")
    step3 = _load(cfg.step3_file, "Step 3 (quantum solver)")

    molecule = cfg.molecule
    print(f"\n[Step 4] Visualization — {molecule}")
    print(f"  Output dir: {cfg.plots_dir}")

    # ── Plot 1: Energy Comparison ─────────────────────────────────────────────
    def plot_energy_comparison():
        if step0 is None and step3 is None:
            print("  [Skip] Plot 1: no data available.")
            return

        methods = {}
        e_hf    = None

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

        if step1 is not None and "mp2_energy" in step1:
            methods.setdefault("MP2", step1["mp2_energy"])

        if step3 is not None and step3.get("energy") is not None:
            solver_label = (f"{step3['solver'].upper()}\n"
                            f"({step3.get('ansatz','?').upper()}+"
                            f"{step3.get('mapping','?').upper()})")
            methods[solver_label] = step3["energy"]

        if not methods or e_hf is None:
            print("  [Skip] Plot 1: insufficient data.")
            return

        labels = list(methods.keys())
        corr_e = [(methods[m] - e_hf) * cfg.hartree_to_kcal_mol for m in labels]

        classical_set = {"HF","MP2","CCSD","CCSD(T)","CCSD_T","CASSCF","NEVPT2"}
        colors = []
        for lbl in labels:
            base = lbl.split("\n")[0].upper().replace("(","").replace(")","")
            colors.append("#4C72B0" if any(c in base for c in classical_set)
                          else "#DD8452")

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
        bars = ax.bar(range(len(labels)), corr_e, color=colors,
                      edgecolor="white", linewidth=0.8, width=0.65)

        for bar, val in zip(bars, corr_e):
            ypos = bar.get_height() + (0.5 if val >= 0 else -1.5)
            ax.text(bar.get_x() + bar.get_width() / 2., ypos,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_ylabel("Correlation Energy vs HF  (kcal/mol)")
        ax.set_title(f"Energy Comparison — {molecule} / {cfg.basis}")
        ax.legend(handles=[
            Patch(facecolor="#4C72B0", label="Classical methods"),
            Patch(facecolor="#DD8452", label="Quantum solver (DMET)"),
        ], loc="lower right")

        plt.tight_layout()
        path = os.path.join(cfg.plots_dir, "plot1_energy_comparison.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Plot 1: Energy comparison → {path}")

    # ── Plot 2: SQD Convergence ───────────────────────────────────────────────
    def plot_convergence():
        if step3 is None or not step3.get("iterations"):
            print("  [Skip] Plot 2: no iteration data.")
            return
        iters = step3["iterations"]
        if len(iters) < 2:
            print("  [Skip] Plot 2: only 1 iteration.")
            return

        solver   = step3["solver"]
        x_vals   = ([it.get("k", it.get("iter", i)) for i, it in enumerate(iters)]
                    if solver == "skqd" else [it["iter"] for it in iters])
        x_label  = "Krylov vector index k" if solver == "skqd" else "SQD iteration"
        energies = [it["energy"]   for it in iters]
        n_configs= [it["n_configs"] for it in iters]
        uhf_ref  = step3["uhf_energy"]
        mp2_ref  = step3["mp2_energy"]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7),
                                       gridspec_kw={"height_ratios": [3, 1]},
                                       sharex=True)

        ax1.plot(x_vals, energies, "o-", color="#DD8452",
                 linewidth=2, markersize=7, label="Quantum solver", zorder=3)
        ax1.axhline(uhf_ref, color="#4C72B0", linewidth=1.5,
                    linestyle="--", alpha=0.8, label=f"UHF  ({uhf_ref:.6f} Ha)")
        ax1.axhline(mp2_ref, color="#55A868", linewidth=1.5,
                    linestyle="-.", alpha=0.8, label=f"MP2  ({mp2_ref:.6f} Ha)")
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

        ax2.bar(x_vals, n_configs, color="#9B59B6", alpha=0.7, width=0.6)
        ax2.set_ylabel("# configs")
        ax2.set_xlabel(x_label)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(cfg.plots_dir, "plot2_convergence.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Plot 2: Convergence → {path}")

    # ── Plot 3: Orbital Deviations ────────────────────────────────────────────
    def plot_orbital_deviations():
        if step1 is None:
            print("  [Skip] Plot 3: Step 1 data not available.")
            return

        deviation     = step1["deviation"]
        final_mo_list = step1["mo_list"]
        atom_syms     = step1["mol_info"]["atom_syms"]
        dominant      = step1["dominant_atoms"]

        show_orbs = np.where(deviation > 0.005)[0]
        if len(show_orbs) == 0:
            print("  [Skip] Plot 3: all deviations < 0.005.")
            return

        active_set   = set(final_mo_list)
        unique_atoms = list(set(atom_syms))
        cmap         = matplotlib.colormaps.get_cmap("tab10").resampled(len(unique_atoms))
        atom_color   = {sym: cmap(i) for i, sym in enumerate(unique_atoms)}

        fig, ax = plt.subplots(figsize=(max(10, len(show_orbs) * 0.5), 5))

        for k, mo_idx in enumerate(show_orbs):
            dev       = deviation[mo_idx]
            is_active = mo_idx in active_set
            sym       = atom_syms[0]
            if mo_idx in active_set and mo_idx < len(final_mo_list):
                pos = list(final_mo_list).index(mo_idx)
                if pos < len(dominant):
                    sym = atom_syms[min(dominant[pos], len(atom_syms)-1)]

            ax.bar(k, dev, color=atom_color.get(sym, "gray"),
                   edgecolor="black" if is_active else "none",
                   linewidth=2.0 if is_active else 0.0,
                   width=0.8, alpha=0.85 if is_active else 0.5)
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

        legend_patches = [Patch(color=atom_color[s], label=s) for s in unique_atoms]
        legend_patches.append(
            Patch(facecolor="white", edgecolor="black", linewidth=2,
                  label="active space")
        )
        ax.legend(handles=legend_patches, loc="upper right", framealpha=0.9)

        if final_mo_list:
            last_active_k = max(k for k, mo in enumerate(show_orbs)
                                if mo in active_set)
            ax.axvline(last_active_k + 0.5, color="red", linewidth=1.5,
                       linestyle="--", alpha=0.7)

        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(cfg.plots_dir, "plot3_orbital_deviations.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Plot 3: Orbital deviations → {path}")

    # ── Plot 4: Bath Singular Values ──────────────────────────────────────────
    def plot_bath_singular_values():
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
        ax.bar(np.arange(len(sv_bath)), sv_bath,
               color="#4C72B0", alpha=0.8, width=0.7)
        ax.axhline(cfg.bath_tolerance, color="red", linewidth=1.5,
                   linestyle="--",
                   label=f"bath_tolerance = {cfg.bath_tolerance:.0e}")
        ax.set_yscale("log")
        ax.set_xlabel("Bath orbital index")
        ax.set_ylabel("Singular value (log scale)")
        ax.set_title(f"Schmidt Singular Values — {molecule}\n"
                     f"Embedding: {n_imp} imp + {n_bath} bath = "
                     f"{n_imp+n_bath} orbs = {2*(n_imp+n_bath)} qubits\n"
                     f"sv² coverage: {step2.get('sv2_cov', 0.0):.4f}")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        path = os.path.join(cfg.plots_dir, "plot4_bath_svs.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Plot 4: Bath singular values → {path}")

    # ── Plot 5: Löwdin Heatmap ────────────────────────────────────────────────
    def plot_lowdin_heatmap():
        if step1 is None:
            print("  [Skip] Plot 5: Step 1 data not available.")
            return

        weights   = step1["lowdin_weights"]
        mo_list   = step1["mo_list"]
        atom_syms = step1["mol_info"]["atom_syms"]

        if weights.shape[0] == 0:
            print("  [Skip] Plot 5: empty Löwdin weights.")
            return

        fig, ax = plt.subplots(
            figsize=(max(4, len(atom_syms)), max(4, len(mo_list) * 0.5))
        )
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
                     f"Active space: ({step1['nel']}e, "
                     f"{step1['n_active_orbs']}orb)")

        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                val = weights[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="black" if val < 0.6 else "white")

        plt.tight_layout()
        path = os.path.join(cfg.plots_dir, "plot5_lowdin_heatmap.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Plot 5: Löwdin heatmap → {path}")

    # ── Quantum scan helper ───────────────────────────────────────────────────
    def _quantum_energy_at_geometry(new_geom):
        from pyscf import gto, scf, ao2mo
        from pyscf.scf import hf as pyscf_hf
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import efficient_su2
        from qiskit_addon_sqd.counts import counts_to_arrays
        from qiskit_addon_sqd.fermion import solve_fermion
        from qiskit_addon_sqd.configuration_recovery import recover_configurations

        try:
            mol_s = gto.M(
                atom    = new_geom,
                basis   = cfg.basis,
                charge  = cfg.charge,
                spin    = cfg.spin,
                verbose = 0,
            )
        except Exception as e:
            warnings.warn(f"  [QScan] mol build failed: {e}", RuntimeWarning)
            return None

        try:
            is_restricted = (cfg.spin == 0)
            mf_s = scf.RHF(mol_s) if is_restricted else scf.UHF(mol_s)
            mf_s.max_cycle   = 150
            mf_s.level_shift = 0.3
            mf_s.verbose     = 0
            mf_s.kernel()
            if not mf_s.converged:
                nw = mf_s.newton()
                nw.verbose = 0; nw.max_cycle = 100
                nw.kernel(mf_s.mo_coeff)
                if nw.converged:
                    for attr in ("e_tot","mo_coeff","mo_energy","mo_occ","converged"):
                        setattr(mf_s, attr, getattr(nw, attr))
        except Exception as e:
            warnings.warn(f"  [QScan] MF failed: {e}", RuntimeWarning)
            return None

        if step1 is None:
            return None

        mo_list = step1["mo_list"]
        n_imp   = step1["n_active_orbs"]
        nel     = step1["nel"]
        n_ao    = mol_s.nao_nr()

        mc_raw = np.asarray(mf_s.mo_coeff)
        mo_coeff_s = mc_raw[0] if mc_raw.ndim == 3 else mc_raw

        try:
            S_s          = mol_s.intor("int1e_ovlp")
            evals, evecs = np.linalg.eigh(S_s)
            mask         = evals > 1e-15
            sq           = np.sqrt(evals[mask])
            S_sqrt_s     = (evecs[:, mask] * sq)  @ evecs[:, mask].T
            S_invsqrt_s  = (evecs[:, mask] / sq)  @ evecs[:, mask].T

            valid_mo = [i for i in mo_list if i < mo_coeff_s.shape[1]]
            if len(valid_mo) < 2:
                warnings.warn("  [QScan] fewer than 2 valid MOs.", RuntimeWarning)
                return None

            C_imp_s = mo_coeff_s[:, valid_mo]
            Q_imp_s = S_sqrt_s @ C_imp_s

            dm_raw = mf_s.make_rdm1()
            if isinstance(dm_raw, (tuple, list)):
                dm_ao_s = np.asarray(dm_raw[0]) + np.asarray(dm_raw[1])
            else:
                dm_raw = np.asarray(dm_raw)
                dm_ao_s = (dm_raw[0] + dm_raw[1] if dm_raw.ndim == 3
                           else dm_raw)

            dm_lo_s          = S_sqrt_s @ dm_ao_s @ S_sqrt_s
            P_env_s          = np.eye(n_ao) - Q_imp_s @ Q_imp_s.T
            F_s              = P_env_s @ dm_lo_s @ Q_imp_s
            U_env_s, sv_s, _ = np.linalg.svd(F_s, full_matrices=True)

            n_bath_max = min(n_imp, max(0, cfg.max_embed_orbs - n_imp))
            n_above    = int(np.sum(sv_s > cfg.bath_tolerance))
            n_bath_s   = min(n_bath_max, max(0, n_above))

            Q_emb_s = (np.hstack([Q_imp_s, U_env_s[:, :n_bath_s]])
                       if n_bath_s > 0 else Q_imp_s.copy())
            n_emb_s = len(valid_mo) + n_bath_s
            C_emb_s = S_invsqrt_s @ Q_emb_s

            h1e_bare_s = mol_s.intor("int1e_kin") + mol_s.intor("int1e_nuc")
            P_emb_lo   = Q_emb_s @ Q_emb_s.T
            P_core_lo  = np.eye(n_ao) - P_emb_lo
            dm_core_ao = S_invsqrt_s @ (P_core_lo @ dm_lo_s @ P_core_lo) @ S_invsqrt_s
            dm_core_ao = 0.5 * (dm_core_ao + dm_core_ao.T)
            vj_s, vk_s = pyscf_hf.get_jk(mol_s, 0.5 * dm_core_ao, hermi=1)
            h1e_eff_s  = h1e_bare_s + 2 * vj_s - vk_s
            ecore_s    = mol_s.energy_nuc() + 0.5 * float(
                np.einsum("ij,ji->", dm_core_ao, h1e_bare_s + h1e_eff_s)
            )
            h1e_s = C_emb_s.T @ h1e_eff_s @ C_emb_s
            h1e_s = 0.5 * (h1e_s + h1e_s.T)
            h2e_s = ao2mo.kernel(mol_s, C_emb_s, compact=False).reshape(
                n_emb_s, n_emb_s, n_emb_s, n_emb_s
            )
        except Exception as e:
            warnings.warn(f"  [QScan] DMET build failed: {e}", RuntimeWarning)
            return None

        try:
            n_alpha_s  = nel // 2 + nel % 2
            n_beta_s   = nel // 2
            n_qubits_s = 2 * n_emb_s

            hf_c = QuantumCircuit(n_qubits_s)
            for i in range(n_alpha_s): hf_c.x(i)
            for i in range(n_beta_s):  hf_c.x(n_emb_s + i)

            ansatz_c = efficient_su2(
                n_qubits_s, reps=2, entanglement="linear",
                skip_final_rotation_layer=True,
            )
            params = np.random.default_rng(0).uniform(0, 2*np.pi,
                                                       ansatz_c.num_parameters)
            circ_s = hf_c.compose(ansatz_c.assign_parameters(params))
            circ_s.measure_all()

            shots_s = (cfg.quantum_scan_shots if cfg.quantum_scan_fast
                       else cfg.n_shots)
            iters_s = (cfg.quantum_scan_iters if cfg.quantum_scan_fast
                       else cfg.sqd_iters)

            from qiskit_aer import AerSimulator
            from qiskit import transpile as qk_transpile
            sim_s = AerSimulator(
                method="matrix_product_state",
                matrix_product_state_max_bond_dimension=cfg.mps_max_bond_dim,
                matrix_product_state_truncation_threshold=cfg.mps_trunc_thresh,
            )
            tc_s  = qk_transpile(circ_s, backend=sim_s, optimization_level=1)
            raw_s = sim_s.run(tc_s, shots=shots_s).result().get_counts()

            bsm_s, probs_s = counts_to_arrays(raw_s)

            if bsm_s.shape[0] > 0:
                valid = ((bsm_s[:, :n_emb_s].sum(axis=1) == n_alpha_s) &
                         (bsm_s[:, n_emb_s:].sum(axis=1) == n_beta_s))
                bsm_s   = bsm_s[valid]
                probs_s = probs_s[valid]
                if probs_s.sum() > 0:
                    probs_s = probs_s / probs_s.sum()

            if bsm_s.shape[0] == 0:
                warnings.warn("  [QScan] no valid configs after filter.",
                              RuntimeWarning)
                return None

            hf_row_s = np.zeros(n_qubits_s, dtype=bool)
            for i in range(n_alpha_s): hf_row_s[i]           = True
            for i in range(n_beta_s):  hf_row_s[n_emb_s + i] = True
            if not any(np.array_equal(bsm_s[i], hf_row_s)
                       for i in range(bsm_s.shape[0])):
                bsm_s   = np.vstack([bsm_s, hf_row_s[np.newaxis, :]])
                probs_s = np.append(probs_s, 1.0 / bsm_s.shape[0])
                probs_s = probs_s / probs_s.sum()

            avg_occs_s = (
                np.array([1.0 if i < n_alpha_s else 0.0 for i in range(n_emb_s)]),
                np.array([1.0 if i < n_beta_s  else 0.0 for i in range(n_emb_s)]),
            )
            energy_s = None
            for it_s in range(iters_s):
                bsm_s, probs_s = recover_configurations(
                    bsm_s, probs_s, avg_occs_s,
                    num_elec_a=n_alpha_s, num_elec_b=n_beta_s,
                    rand_seed=10 + it_s,
                )
                if bsm_s.shape[0] == 0:
                    break
                e_emb_s, _, avg_occs_s, _ = solve_fermion(
                    bsm_s, hcore=h1e_s, eri=h2e_s,
                    open_shell=False, spin_sq=0.0,
                )
                energy_s = float(e_emb_s) + ecore_s

            return energy_s

        except Exception as e:
            warnings.warn(f"  [QScan] SQD failed: {e}", RuntimeWarning)
            return None

    # ── Plot 6: Geometry Scan ─────────────────────────────────────────────────
    def plot_geometry_scan():
        if not cfg.geometry_scan or no_scan:
            print("  [Skip] Plot 6: geometry scan disabled.")
            return None

        from pyscf import gto, scf, mp, cc

        a_idx, b_idx = cfg.scan_atom_pair
        base_geom    = list(cfg.geometry)
        sym_a, c_a   = base_geom[a_idx]
        sym_b, c_b   = base_geom[b_idx]
        d_cif        = float(np.linalg.norm(np.array(c_a) - np.array(c_b)))

        print(f"\n  ── Classical scan ({cfg.scan_method}) ─────────────────────────")

        cl_energies  = []
        cl_distances = []

        for d in cfg.scan_distances:
            new_geom        = list(base_geom)
            new_geom[b_idx] = (sym_b, (c_a[0] + d, c_a[1], c_a[2]))
            try:
                mol_s = gto.M(atom=new_geom, basis=cfg.basis,
                              charge=cfg.charge, spin=cfg.spin, verbose=0)
                mf_s  = (scf.RHF(mol_s) if cfg.spin == 0 else scf.UHF(mol_s))
                mf_s.verbose = 0
                mf_s.kernel()
                method = cfg.scan_method.upper()
                if method == "HF":
                    e = float(mf_s.e_tot)
                elif method == "MP2":
                    mymp = mp.MP2(mf_s); mymp.verbose = 0; mymp.kernel()
                    e = float(mf_s.e_tot + mymp.e_corr)
                elif method == "CCSD":
                    mycc = cc.CCSD(mf_s); mycc.verbose = 0; mycc.kernel()
                    e = float(mf_s.e_tot + mycc.e_corr)
                else:
                    e = float(mf_s.e_tot)
                cl_energies.append(e)
                cl_distances.append(d)
                print(f"    d={d:.3f} Å → {e:.6f} Ha")
            except Exception as ex:
                warnings.warn(f"Classical scan failed at d={d:.3f} Å: {ex}",
                              RuntimeWarning)

        if len(cl_energies) < 3:
            print("  [Skip] Plot 6: fewer than 3 classical scan points.")
            return None

        cl_energies  = np.array(cl_energies)
        cl_distances = np.array(cl_distances)

        run_quantum = (cfg.quantum_scan and not no_quantum_scan
                       and step1 is not None and step2 is not None)

        qm_energies  = []
        qm_distances = []

        if run_quantum:
            mode_str = "fast" if cfg.quantum_scan_fast else "full"
            print(f"\n  ── Quantum scan (DMET+SQD, {mode_str} mode) ──────────────────")
            shots_q = (cfg.quantum_scan_shots if cfg.quantum_scan_fast
                       else cfg.n_shots)
            iters_q = (cfg.quantum_scan_iters if cfg.quantum_scan_fast
                       else cfg.sqd_iters)
            print(f"     shots={shots_q}, iters={iters_q}")
            print(f"     Estimated time: "
                  f"~{len(cfg.scan_distances) * (2 if cfg.quantum_scan_fast else 10)} min")

            for d in cfg.scan_distances:
                new_geom        = list(base_geom)
                new_geom[b_idx] = (sym_b, (c_a[0] + d, c_a[1], c_a[2]))
                t0  = time.time()
                e_q = _quantum_energy_at_geometry(new_geom)
                elapsed = time.time() - t0
                if e_q is not None:
                    qm_energies.append(e_q)
                    qm_distances.append(d)
                    print(f"    d={d:.3f} Å → {e_q:.6f} Ha  ({elapsed:.0f}s)")
                else:
                    print(f"    d={d:.3f} Å → FAILED")

        all_energies = list(cl_energies) + qm_energies
        e_ref        = min(all_energies)

        cl_rel = (cl_energies - e_ref) * cfg.hartree_to_kcal_mol
        qm_rel = (np.array(qm_energies) - e_ref) * cfg.hartree_to_kcal_mol \
                  if qm_energies else np.array([])

        d_eq_cl = cl_distances[np.argmin(cl_energies)]
        d_eq_qm = (qm_distances[np.argmin(qm_energies)]
                   if qm_energies else None)

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(cl_distances, cl_rel, "o-", color="#4C72B0",
                linewidth=2, markersize=6,
                label=f"Classical: {cfg.scan_method}  (eq={d_eq_cl:.3f} Å)")

        if len(qm_rel) > 0:
            ax.plot(qm_distances, qm_rel, "s--", color="#DD8452",
                    linewidth=2, markersize=7,
                    label=(f"Quantum: DMET+{cfg.quantum_solver.upper()} "
                           f"({'fast' if cfg.quantum_scan_fast else 'full'})  "
                           f"(eq={d_eq_qm:.3f} Å)"))
            ax.axvline(d_eq_qm, color="#DD8452", linewidth=1,
                       linestyle=":", alpha=0.6)

        ax.axvline(d_eq_cl, color="#4C72B0", linewidth=1.2,
                   linestyle="--", alpha=0.7)
        ax.axvline(d_cif, color="#55A868", linewidth=1.5,
                   linestyle="-.",
                   label=f"CIF bond length: {d_cif:.3f} Å")

        ax.set_xlabel(f"{sym_a}({a_idx})–{sym_b}({b_idx}) distance (Å)")
        ax.set_ylabel("Relative energy (kcal/mol)")
        ax.set_title(f"Potential Energy Surface — {molecule} / {cfg.basis}\n"
                     f"Classical vs Quantum comparison")
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=-1)

        for d_eq, color, label in [
            (d_eq_cl, "#4C72B0", cfg.scan_method),
            *([(d_eq_qm, "#DD8452", "Quantum")] if d_eq_qm else []),
        ]:
            ax.annotate(f"{label} eq: {d_eq:.3f} Å",
                        xy=(d_eq, 0), xytext=(d_eq + 0.15, 3),
                        fontsize=9, color=color,
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.0))

        plt.tight_layout()
        path = os.path.join(cfg.plots_dir, "plot6_geometry_scan.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Plot 6: Geometry scan (PES) → {path}")

        scan_pkl = os.path.join(cfg.results_dir, "geometry_scan.pkl")
        with open(scan_pkl, "wb") as f:
            pickle.dump({
                "cl_distances": cl_distances,
                "cl_energies" : cl_energies,
                "qm_distances": np.array(qm_distances),
                "qm_energies" : np.array(qm_energies) if qm_energies else np.array([]),
                "method"      : cfg.scan_method,
                "atom_pair"   : cfg.scan_atom_pair,
                "d_eq_cl"     : d_eq_cl,
                "d_eq_qm"     : d_eq_qm,
                "d_cif"       : d_cif,
            }, f)
        print(f"  ✓ Scan data → {scan_pkl}")

        return {
            "cl_distances": cl_distances, "cl_energies": cl_energies,
            "qm_distances": np.array(qm_distances),
            "qm_energies" : np.array(qm_energies) if qm_energies else np.array([]),
            "d_eq_cl": d_eq_cl, "d_eq_qm": d_eq_qm, "d_cif": d_cif,
        }

    # ── Plot 7 + CSV: Summary ─────────────────────────────────────────────────
    def plot_and_save_summary(scan_data=None):
        data = {}

        data["molecule"]     = molecule
        data["basis"]        = cfg.basis
        data["charge"]       = cfg.charge
        data["spin_2s"]      = cfg.spin
        data["n_atoms"]      = cfg.n_atoms
        data["atom_symbols"] = " ".join(cfg.atom_syms)

        e_hf = None
        if step0 is not None:
            e_hf = step0["methods"].get("HF", {}).get("energy")
            data["E_HF_Ha"] = e_hf
            data["total_time_classical_s"] = step0.get("total_time", "N/A")
            for method in ["MP2","CCSD","CCSD_T","CASSCF","NEVPT2"]:
                entry = step0["methods"].get(method, {})
                e     = entry.get("energy")
                data[f"E_{method}_Ha"] = e
                if e is not None and e_hf is not None:
                    data[f"E_{method}_vs_HF_kcal"] = (
                        (e - e_hf) * cfg.hartree_to_kcal_mol
                    )

        if step1 is not None:
            data["tier"]             = step1.get("tier")
            data["corr_strength"]    = step1.get("corr_strength")
            data["nel_active"]       = step1.get("nel")
            data["n_active_orbs"]    = step1.get("n_active_orbs")
            data["active_mo_indices"]= str(step1.get("mo_list", []))
            data["uhf_energy_Ha"]    = step1.get("uhf_energy")
            data["mp2_energy_Ha"]    = step1.get("mp2_energy")
            data["mp2_corr_Ha"]      = (step1.get("mp2_energy", 0)
                                        - step1.get("uhf_energy", 0))
            data["homo_lumo_gap_eV"] = step1.get("indicators",{}).get("homo_lumo_gap_eV")
            data["s2"]               = step1.get("indicators",{}).get("s2")
            data["has_tm"]           = step1.get("indicators",{}).get("has_tm")

        if step2 is not None:
            data["n_imp_orbs"]     = step2.get("n_imp")
            data["n_bath_orbs"]    = step2.get("n_bath")
            data["n_emb_orbs"]     = step2.get("n_emb")
            data["n_qubits"]       = 2 * step2.get("n_emb", 0)
            data["n_alpha"]        = step2.get("n_alpha")
            data["n_beta"]         = step2.get("n_beta")
            data["sv2_coverage"]   = step2.get("sv2_cov")
            data["sv_gap"]         = step2.get("sv_gap")
            data["ecore_Ha"]       = step2.get("ecore")
            data["mp2_used_in_dmet"]= step2.get("mp2_used")

        if step3 is not None:
            data["quantum_solver"] = step3.get("solver")
            data["quantum_ansatz"] = step3.get("ansatz")
            data["quantum_mapping"]= step3.get("mapping")
            data["quantum_backend"]= step3.get("backend")
            data["E_quantum_Ha"]   = step3.get("energy")
            data["spin_sq_quantum"]= step3.get("spin_sq")
            data["n_sqd_iters"]    = len(step3.get("iterations", []))
            data["n_shots"]        = cfg.n_shots
            e_q = step3.get("energy")
            if e_q is not None and e_hf is not None:
                data["E_quantum_vs_HF_kcal"] = (
                    (e_q - e_hf) * cfg.hartree_to_kcal_mol
                )
            if e_q is not None and step1 is not None:
                mp2_e = step1.get("mp2_energy")
                if mp2_e is not None:
                    data["E_quantum_vs_MP2_kcal"] = (
                        (e_q - mp2_e) * cfg.hartree_to_kcal_mol
                    )
            iters = step3.get("iterations", [])
            if len(iters) >= 2:
                data["sqd_improvement_Ha"] = (
                    iters[-1]["energy"] - iters[0]["energy"]
                )

        if scan_data is not None:
            data["scan_method"]           = cfg.scan_method
            data["scan_atom_pair"]        = str(cfg.scan_atom_pair)
            data["scan_d_eq_classical_A"] = scan_data.get("d_eq_cl")
            data["scan_d_eq_quantum_A"]   = scan_data.get("d_eq_qm")
            data["scan_d_cif_A"]          = scan_data.get("d_cif")

        data["config_quantum_solver"] = cfg.quantum_solver
        data["config_ansatz"]         = cfg.ansatz
        data["config_mapping"]        = cfg.fermion_to_qubit
        data["config_backend"]        = cfg.backend
        data["config_basis"]          = cfg.basis
        data["config_n_shots"]        = cfg.n_shots
        data["config_sqd_iters"]      = cfg.sqd_iters

        csv_path = os.path.join(cfg.results_dir, "simulation_summary.csv")
        with open(csv_path, "w", newline="") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(["Parameter", "Value"])
            for k, v in data.items():
                writer.writerow([k, v if v is not None else "N/A"])
        print(f"  ✓ Summary CSV → {csv_path}")

        # 4-panel summary dashboard
        fig = plt.figure(figsize=(16, 11))
        fig.suptitle(f"Simulation Summary — {molecule} / {cfg.basis}",
                     fontsize=14, fontweight="bold", y=0.98)

        gs   = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.35)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[1, 0])
        ax_d = fig.add_subplot(gs[1, 1])

        palette = {
            "HF":"#4C72B0","MP2":"#55A868","CCSD":"#C44E52",
            "CCSD_T":"#8172B2","CASSCF":"#937860","NEVPT2":"#DA8BC3",
            "Quantum":"#DD8452",
        }

        energy_methods = {}
        method_colors  = {}
        if step0 is not None and e_hf is not None:
            for name, d in step0["methods"].items():
                e = d.get("energy")
                if e is not None:
                    energy_methods[name] = e
                    method_colors[name]  = palette.get(name, "#7F7F7F")
        if step3 is not None and step3.get("energy") is not None:
            lbl = f"Quantum\n({step3.get('solver','').upper()})"
            energy_methods[lbl] = step3["energy"]
            method_colors[lbl]  = palette["Quantum"]

        if energy_methods and e_hf is not None:
            labels_a = list(energy_methods.keys())
            vals_a   = [(energy_methods[m] - e_hf) * cfg.hartree_to_kcal_mol
                        for m in labels_a]
            colors_a = [method_colors.get(m.split("\n")[0], "#7F7F7F")
                        for m in labels_a]
            bars_a   = ax_a.barh(labels_a, vals_a, color=colors_a,
                                 edgecolor="white", height=0.6)
            ax_a.axvline(0, color="black", linewidth=0.8,
                         linestyle="--", alpha=0.5)
            for bar, val in zip(bars_a, vals_a):
                ax_a.text(val + (0.3 if val < 0 else -0.3),
                          bar.get_y() + bar.get_height()/2.,
                          f"{val:.1f}", va="center",
                          ha="right" if val < 0 else "left", fontsize=8)
            ax_a.set_xlabel("Correlation vs HF (kcal/mol)")
        else:
            ax_a.text(0.5, 0.5, "No energy data", ha="center",
                      va="center", transform=ax_a.transAxes)
        ax_a.set_title("A. Energy Ladder", fontweight="bold")

        ax_b.axis("off")
        ax_b.set_title("B. Correlation Indicators", fontweight="bold")
        rows_b = []
        if step1 is not None:
            rows_b = [
                ("Tier",           step1.get("tier","N/A")),
                ("Has TM element", step1.get("indicators",{}).get("has_tm","N/A")),
                ("Corr. strength", f"{step1.get('corr_strength',0):.4f}"),
                ("Active space",   f"({step1.get('nel','?')}e, "
                                   f"{step1.get('n_active_orbs','?')}orb)"),
                ("HOMO-LUMO gap",
                 f"{step1.get('indicators',{}).get('homo_lumo_gap_eV','N/A'):.3f} eV"
                 if isinstance(step1.get("indicators",{}).get("homo_lumo_gap_eV"),float)
                 else "N/A"),
                ("⟨S²⟩",
                 f"{step1.get('indicators',{}).get('s2','N/A'):.4f}"
                 if isinstance(step1.get("indicators",{}).get("s2"),float)
                 else "N/A"),
                ("MP2 corr.",        f"{data.get('mp2_corr_Ha',0):.4f} Ha"),
                ("MP2 used in DMET", step2.get("mp2_used","N/A") if step2 else "N/A"),
            ]
        if rows_b:
            tbl = ax_b.table(cellText=[[k,str(v)] for k,v in rows_b],
                             colLabels=["Quantity","Value"],
                             loc="center", cellLoc="left")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1, 1.5)
            for (r,c), cell in tbl.get_celld().items():
                if r == 0:
                    cell.set_facecolor("#4C72B0")
                    cell.set_text_props(color="white", fontweight="bold")
                elif r % 2 == 0:
                    cell.set_facecolor("#EEF2FF")
        else:
            ax_b.text(0.5, 0.5, "No Step 1 data", ha="center",
                      va="center", transform=ax_b.transAxes)

        ax_c.axis("off")
        ax_c.set_title("C. Embedding (DMET)", fontweight="bold")
        rows_c = []
        if step2 is not None:
            rows_c = [
                ("Impurity orbs",  step2.get("n_imp","N/A")),
                ("Bath orbs",      step2.get("n_bath","N/A")),
                ("Total emb orbs", step2.get("n_emb","N/A")),
                ("Qubits",         2 * step2.get("n_emb",0)),
                ("Electrons",      f"{step2.get('n_alpha','?')}α + "
                                   f"{step2.get('n_beta','?')}β"),
                ("sv² coverage",   f"{step2.get('sv2_cov',0):.4f}"),
                ("E_core",         f"{step2.get('ecore',0):.6f} Ha"),
                ("Fermion mapping", cfg.fermion_to_qubit.upper()),
                ("Ansatz",          cfg.ansatz.upper()),
                ("Backend",         cfg.backend.upper()),
                ("Shots",           cfg.n_shots),
                ("SQD iters",       cfg.sqd_iters),
            ]
            if step3 is not None and step3.get("energy") is not None:
                rows_c.append(("Final E (quantum)",
                               f"{step3['energy']:.8f} Ha"))
        if rows_c:
            tbl_c = ax_c.table(cellText=[[k,str(v)] for k,v in rows_c],
                               colLabels=["Parameter","Value"],
                               loc="center", cellLoc="left")
            tbl_c.auto_set_font_size(False)
            tbl_c.set_fontsize(10)
            tbl_c.scale(1, 1.35)
            for (r,c), cell in tbl_c.get_celld().items():
                if r == 0:
                    cell.set_facecolor("#DD8452")
                    cell.set_text_props(color="white", fontweight="bold")
                elif r % 2 == 0:
                    cell.set_facecolor("#FFF4EE")
        else:
            ax_c.text(0.5, 0.5, "No Step 2 data", ha="center",
                      va="center", transform=ax_c.transAxes)

        ax_d.set_title("D. SQD Energy Convergence", fontweight="bold")
        if step3 is not None and len(step3.get("iterations",[])) >= 2:
            iters_d = step3["iterations"]
            x_d     = [it.get("iter", it.get("k", i))
                       for i, it in enumerate(iters_d)]
            y_d     = [it["energy"] for it in iters_d]
            uhf_d   = step3["uhf_energy"]
            mp2_d   = step3["mp2_energy"]
            ax_d.plot(x_d, y_d, "o-", color="#DD8452",
                      linewidth=2, markersize=5, label="Quantum", zorder=3)
            ax_d.axhline(uhf_d, color="#4C72B0", linewidth=1.5,
                         linestyle="--", label="UHF", alpha=0.8)
            ax_d.axhline(mp2_d, color="#55A868", linewidth=1.5,
                         linestyle="-.", label="MP2", alpha=0.8)
            ax_d.set_xlabel("Iteration")
            ax_d.set_ylabel("Energy (Ha)")
            ax_d.legend(fontsize=9, framealpha=0.9)
            ax_d.grid(True, alpha=0.3)
            if len(y_d) >= 2:
                delta = y_d[-1] - y_d[0]
                ax_d.annotate(
                    f"ΔE = {delta*cfg.hartree_to_kcal_mol:+.2f} kcal/mol\n"
                    f"over {len(y_d)} iters",
                    xy=(x_d[-1], y_d[-1]),
                    xytext=(-50, 20), textcoords="offset points",
                    fontsize=9, color="#DD8452",
                    arrowprops=dict(arrowstyle="->", color="#DD8452", lw=1.0),
                )
        else:
            ax_d.text(0.5, 0.5, "No SQD iteration data", ha="center",
                      va="center", transform=ax_d.transAxes)
            ax_d.set_xlabel("Iteration")
            ax_d.set_ylabel("Energy (Ha)")

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        path7 = os.path.join(cfg.plots_dir, "plot7_summary.png")
        plt.savefig(path7, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Plot 7: Summary dashboard → {path7}")

    # ── Run all ───────────────────────────────────────────────────────────────
    print()
    plot_energy_comparison()
    plot_convergence()
    plot_orbital_deviations()
    plot_bath_singular_values()
    plot_lowdin_heatmap()
    scan_result = plot_geometry_scan()
    plot_and_save_summary(scan_data=scan_result)

    print(f"\n[Step 4] All outputs saved to: {cfg.plots_dir}/")
    print(f"  plot1_energy_comparison.png  — method comparison bar chart")
    print(f"  plot2_convergence.png        — SQD iteration convergence")
    print(f"  plot3_orbital_deviations.png — MP2 NO deviations")
    print(f"  plot4_bath_svs.png           — Schmidt singular values")
    print(f"  plot5_lowdin_heatmap.png     — MO-to-atom population weights")
    print(f"  plot6_geometry_scan.png      — PES: classical + quantum")
    print(f"  plot7_summary.png            — Full simulation summary")
    print(f"  ../simulation_summary.csv    — All quantities (CSV)")