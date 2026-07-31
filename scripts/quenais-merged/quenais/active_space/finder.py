"""
Active Space Finder for QuEnAIS pipeline.
"""

import os
import pickle
import warnings
import numpy as np


# ── Helper functions (module level, called from main) ─────────────────────────

def run_uhf(mol):
    from pyscf import scf
    mf = scf.UHF(mol)
    mf.max_cycle   = 400
    mf.level_shift = 0.5
    mf.kernel()
    if not mf.converged:
        print("  DIIS did not converge → trying Newton solver...")
        nw = mf.newton()
        nw.max_cycle = 400
        nw.kernel(mf.mo_coeff)
        if nw.converged:
            for attr in ("e_tot", "mo_coeff", "mo_energy", "mo_occ", "converged"):
                setattr(mf, attr, getattr(nw, attr))
        else:
            warnings.warn(
                "UHF did not converge with DIIS or Newton.",
                RuntimeWarning,
            )
    return mf


def classify(mol, mf, cfg):
    has_tm = any(mol.atom_symbol(i) in cfg.tm_elements
                 for i in range(mol.natm))

    s2, _      = mf.spin_square()
    S_val      = mol.spin / 2.0
    s_expected = S_val * (S_val + 1.0)
    is_singlet = (mol.spin == 0)

    if is_singlet:
        spin_cont         = float(s2)
        spin_contaminated = spin_cont > cfg.spin_contamination_singlet_threshold
        spin_metric_label = f"⟨S²⟩={s2:.4f} (singlet, expected 0)"
    else:
        spin_cont         = float(s2 / s_expected)
        spin_contaminated = spin_cont > cfg.spin_contamination_tier2_threshold
        spin_metric_label = (f"⟨S²⟩={s2:.4f} / expected {s_expected:.4f} "
                             f"= ratio {spin_cont:.3f}")

    gaps    = {}
    gap_min = 10.0

    for label, ch in [("alpha", 0), ("beta", 1)]:
        mo_e   = np.asarray(mf.mo_energy[ch])
        mo_occ = np.asarray(mf.mo_occ[ch])
        occ_e  = mo_e[mo_occ > 0.5]
        vir_e  = mo_e[mo_occ < 0.5]
        if len(occ_e) > 0 and len(vir_e) > 0:
            gaps[label] = float((vir_e[0] - occ_e[-1]) * cfg.hartree_to_ev)

    if gaps:
        gap_min = min(gaps.values())

    gap_label  = "  ".join(f"{k}={v:.3f} eV" for k, v in gaps.items())
    gap_label += f"  →  min={gap_min:.3f} eV"

    indicators = {
        "has_tm"           : has_tm,
        "is_singlet"       : is_singlet,
        "s2"               : float(s2),
        "s_expected"       : float(s_expected),
        "spin_cont"        : float(spin_cont),
        "spin_contaminated": spin_contaminated,
        "homo_lumo_gap_eV" : float(gap_min),
        "gap_alpha_eV"     : gaps.get("alpha"),
        "gap_beta_eV"      : gaps.get("beta"),
    }

    if has_tm:
        tier = 3
    elif spin_contaminated or gap_min < cfg.homo_lumo_tier2_threshold_ev:
        tier = 2
    else:
        tier = 1

    print(f"  Spin  : {spin_metric_label}")
    print(f"         contaminated = {'YES → Tier≥2' if spin_contaminated else 'no'}")
    print(f"  Gap   : {gap_label}")
    print(f"         below threshold = "
          f"{'YES → Tier≥2' if gap_min < cfg.homo_lumo_tier2_threshold_ev else 'no'}")
    print(f"  TM    : {'YES → Tier 3' if has_tm else 'no'}")

    return tier, indicators


def compute_mp2_deviations(mf, mol):
    from pyscf import mp as pyscf_mp

    S             = mol.intor("int1e_ovlp")
    evals, evecs  = np.linalg.eigh(S)
    mask          = evals > 1e-15
    S_invsqrt     = (evecs[:, mask] / np.sqrt(evals[mask])) @ evecs[:, mask].T

    mp2_ok = False
    e_corr = 0.0

    try:
        mymp         = pyscf_mp.MP2(mf)
        mymp.verbose = 0
        e_corr, _    = mymp.kernel()
        dm1          = mymp.make_rdm1()

        if isinstance(dm1, (tuple, list)):
            Ca   = np.asarray(mf.mo_coeff[0])
            Cb   = np.asarray(mf.mo_coeff[1])
            dm_ao= Ca @ dm1[0] @ Ca.T + Cb @ dm1[1] @ Cb.T
        else:
            dm_ao = mf.mo_coeff @ dm1 @ mf.mo_coeff.T

        mp2_ok = True

    except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
        warnings.warn(
            f"MP2 failed with: {e}\n"
            f"Falling back to UHF density matrix.",
            RuntimeWarning,
        )
        dm_raw = mf.make_rdm1()
        dm_ao  = (dm_raw[0] + dm_raw[1]) if isinstance(dm_raw, tuple) else dm_raw

    dm_lo     = S_invsqrt @ dm_ao @ S_invsqrt.T
    dm_lo     = 0.5 * (dm_lo + dm_lo.T)
    no_occ    = np.clip(np.linalg.eigvalsh(dm_lo)[::-1], 0.0, 2.0)
    deviation = np.minimum(no_occ, 2.0 - no_occ)

    return deviation, no_occ, e_corr, mp2_ok


def find_gap_cutoff(values, min_n, max_n):
    values = np.asarray(values, dtype=float)
    n      = len(values)
    min_n  = max(1, min(min_n, n))
    max_n  = min(max_n, n)

    if min_n >= max_n:
        order = np.argsort(-values)
        return min_n, 0.0, list(order[:min_n])

    order    = np.argsort(-values)
    sorted_v = values[order]

    best_gap, best_n = -1.0, min_n
    for k in range(min_n, max_n + 1):
        gap = sorted_v[k - 1] - (sorted_v[k] if k < n else 0.0)
        if gap > best_gap:
            best_gap, best_n = gap, k

    return best_n, float(best_gap), list(order[:best_n])


def lowdin_population(mo_coeff, mo_list, S, ao_labels, n_atoms):
    evals, evecs = np.linalg.eigh(S)
    mask   = evals > 1e-15
    S_sqrt = (evecs[:, mask] * np.sqrt(evals[mask])) @ evecs[:, mask].T

    weights = np.zeros((len(mo_list), n_atoms))
    for k, mo_idx in enumerate(mo_list):
        c_lo = S_sqrt @ mo_coeff[:, mo_idx]
        for ao_j, (atom_idx, *_) in enumerate(ao_labels):
            weights[k, atom_idx] += c_lo[ao_j] ** 2

    return weights


def count_active_electrons(mol, mf, final_mo_list, cfg):
    active_set   = set(final_mo_list)
    mo_occ_alpha = np.asarray(mf.mo_occ[0])
    mo_occ_beta  = np.asarray(mf.mo_occ[1])
    mo_occ_total = mo_occ_alpha + mo_occ_beta

    core_orbs = [
        i for i, occ in enumerate(mo_occ_total)
        if i not in active_set and occ > cfg.core_occ_threshold
    ]
    n_core = len(core_orbs)
    nel    = mol.nelectron - 2 * n_core

    print(f"  Total electrons    : {mol.nelectron}")
    print(f"  Core MOs (occ>{cfg.core_occ_threshold}): {n_core}  →  {core_orbs}")
    print(f"  Raw active electrons: {nel}")

    max_nel = 2 * len(final_mo_list)

    if nel <= 0:
        raise ValueError(
            f"Active electron count is {nel} ≤ 0.\n"
            f"  Total electrons : {mol.nelectron}\n"
            f"  Core MOs counted: {n_core}\n"
            f"  Fix: increase core_occ_threshold in Config (recommended: 1.95)."
        )

    if nel > max_nel:
        warnings.warn(
            f"Active electrons ({nel}) > 2 × active orbitals ({max_nel}). "
            f"Capping at {max_nel}.",
            RuntimeWarning,
        )
        nel = max_nel

    if nel % 2 != 0:
        nel -= 1
        print(f"  Adjusted to even   : nel={nel}")

    nel = max(2, nel)
    print(f"  Final active electrons: {nel}")

    return nel


# ── Entry point ───────────────────────────────────────────────────────────────

def _validate_block2_wrapper(cfg):
    """
    Check that the block2 wrapper points to the current environment.
    Warns if it points to a different env — a common mistake when
    switching environments without regenerating the wrapper.
    """
    import sys
    wrapper = cfg.blockexe_wrapper
    if not os.path.exists(wrapper):
        warnings.warn(
            f"block2 wrapper not found: {wrapper}\n"
            f"Run: bash install.sh  OR  python quenais/utils/regenerate_wrapper.py",
            RuntimeWarning,
        )
        return

    current_env = os.path.dirname(sys.executable)
    with open(wrapper) as f:
        content = f.read()

    if current_env not in content and "block2main" in content:
        warnings.warn(
            f"block2 wrapper may point to wrong environment.\n"
            f"  Current env : {current_env}\n"
            f"  Wrapper     : {wrapper}\n"
            f"  Fix: run bash install.sh to regenerate the wrapper\n"
            f"  OR:  python -c \"from quenais.utils.cif_parser import *\" "
            f"(see INSTALL.md)",
            RuntimeWarning,
        )


def main(cfg, force=False):
    """
    Run Active Space Finder.
    cfg   : quenais.config.Config
    force : rerun even if cached result exists
    """
    _validate_block2_wrapper(cfg)
    from pyscf import gto
    from pyscf.dmrgscf import dmrgci
    from asf.wrapper import find_from_scf
    os.makedirs(cfg.results_dir, exist_ok=True)

    if os.path.exists(cfg.step1_file) and not force:
        print(f"[Step 1] Using cached result: {cfg.step1_file}")
        return

    os.environ["BLOCKEXE"]            = cfg.blockexe_wrapper
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["MKL_DEBUG_CPU_TYPE"]  = "5"
    dmrgci.settings.BLOCKEXE          = cfg.blockexe_wrapper

    print(f"\n{'='*60}")
    print(f"[Step 1] Active Space Finder — {cfg.molecule}")
    print(f"{'='*60}")

    mol = gto.M(
        atom    = cfg.geometry,
        basis   = cfg.basis,
        charge  = cfg.charge,
        spin    = cfg.spin,
        verbose = 3,
    )

    print(f"\n  Molecule  : {cfg.molecule}")
    print(f"  Atoms     : {cfg.n_atoms}  {cfg.atom_syms}")
    print(f"  Basis     : {cfg.basis}")
    print(f"  Charge    : {cfg.charge}   Spin (2S): {cfg.spin}")
    print(f"  Electrons : {mol.nelectron}   AOs: {mol.nao_nr()}")

    # ── Phase A: UHF + Classification ─────────────────────────────────────────
    print(f"\n── Phase A: UHF + Classification {'─'*20}")
    mf = run_uhf(mol)
    print(f"\n  UHF energy = {mf.e_tot:.8f} Ha  (converged: {mf.converged})")

    if not mf.converged:
        warnings.warn("UHF did not converge. All downstream results unreliable.",
                      RuntimeWarning)

    tier, indicators = classify(mol, mf, cfg)

    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  Tier {tier} classification              │")
    print(f"  │  TM element : {str(indicators['has_tm']):5s}                  │")
    print(f"  │  Spin cont. : {indicators['spin_cont']:.4f}                │")
    print(f"  │  Gap (min)  : {indicators['homo_lumo_gap_eV']:.4f} eV             │")
    print(f"  └─────────────────────────────────────┘")

    # ── Phase B: MP2 Deviations + ASF ─────────────────────────────────────────
    print(f"\n── Phase B: MP2 Deviations + ASF {'─'*20}")
    deviation, no_occ, e_corr, mp2_ok = compute_mp2_deviations(mf, mol)

    print(f"  MP2 used             : {mp2_ok}")
    print(f"  MP2 correlation E    : {e_corr:.6f} Ha")
    print(f"  Orbitals (dev>0.05)  : {int(np.sum(deviation > 0.05))}")
    print(f"  Orbitals (dev>0.10)  : {int(np.sum(deviation > 0.10))}")

    asf_p = cfg.asf_params[tier]
    print(f"\n  Running ASF (Tier {tier}):")
    print(f"    entropy_threshold = {asf_p['entropy_threshold']}")
    print(f"    max_norb          = {asf_p['max_norb']}")
    print(f"    min_norb          = {asf_p['min_norb']}")

    active_space = find_from_scf(
        mf,
        entropy_threshold = asf_p["entropy_threshold"],
        max_norb          = asf_p["max_norb"],
        min_norb          = asf_p["min_norb"],
        verbose           = True,
    )

    mo_list  = list(active_space.mo_list)
    mo_coeff = active_space.mo_coeff
    n_cand   = len(mo_list)
    print(f"\n  ASF candidates : {n_cand} orbitals")
    print(f"  MO indices     : {mo_list}")

    if n_cand == 0:
        raise RuntimeError(
            "ASF returned 0 candidates.\n"
            f"entropy_threshold={asf_p['entropy_threshold']} (Tier {tier}).\n"
            "Try lowering entropy_threshold in cfg.asf_params."
        )

    # ── Phase C: Gap Detection ─────────────────────────────────────────────────
    print(f"\n── Phase C: Gap Detection {'─'*27}")

    cand_devs = np.array([
        deviation[i] if i < len(deviation) else 0.0
        for i in mo_list
    ])

    print(f"\n  Candidate orbital deviations (sorted):")
    print(f"  {'MO':>5}  {'dev':>8}  bar")
    for mo_idx, dev in sorted(zip(mo_list, cand_devs), key=lambda x: -x[1]):
        bar = "█" * int(dev * 30)
        print(f"  {mo_idx:>5}  {dev:>8.4f}  {bar}")

    n_final, gap_val, selected_k = find_gap_cutoff(
        cand_devs, cfg.gap_min_norb, cfg.gap_max_norb
    )
    final_mo_list = sorted(mo_list[k] for k in selected_k)

    print(f"\n  Gap detected   : {gap_val:.4f}  (at position {n_final})")
    print(f"  Selected orbs  : {n_final}  →  {final_mo_list}")

    print(f"\n  Counting active electrons:")
    nel = count_active_electrons(mol, mf, final_mo_list, cfg)

    print(f"\n  ┌────────────────────────────────────┐")
    print(f"  │  Active space: ({nel}e, {n_final}orb)         │")
    print(f"  │  Orbitals    : {str(final_mo_list):<28}│")
    print(f"  └────────────────────────────────────┘")

    # ── Phase D: Löwdin Population ────────────────────────────────────────────
    print(f"\n── Phase D: Löwdin Population {'─'*23}")
    S              = mol.intor("int1e_ovlp")
    ao_labels      = mol.ao_labels(fmt=None)
    weights        = lowdin_population(mo_coeff, final_mo_list, S,
                                       ao_labels, cfg.n_atoms)
    dominant_atoms = np.argmax(weights, axis=1).astype(int)

    print(f"\n  {'MO':>5}  {'Atom':>6}  {'Symbol':>6}  {'Weight':>8}")
    print(f"  {'─'*35}")
    for k, mo_idx in enumerate(final_mo_list):
        atom   = dominant_atoms[k]
        weight = weights[k, atom]
        print(f"  {mo_idx:>5}  {atom:>6}  {cfg.atom_syms[atom]:>6}  {weight:>8.4f}")

    final_devs    = np.array([deviation[i] for i in final_mo_list
                               if i < len(deviation)])
    corr_strength = float(np.mean(final_devs)) if len(final_devs) > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"[Step 1] Summary — {cfg.molecule}")
    print(f"{'='*60}")
    print(f"  Tier                : {tier}")
    print(f"  Active space        : ({nel}e, {n_final}orb)")
    print(f"  Correlation strength: {corr_strength:.4f}")
    print(f"{'='*60}")

    results = {
        "nel"            : nel,
        "mo_list"        : final_mo_list,
        "mo_coeff"       : mo_coeff,
        "n_active_orbs"  : n_final,
        "no_occ"         : no_occ,
        "deviation"      : deviation,
        "lowdin_weights" : weights,
        "dominant_atoms" : dominant_atoms,
        "tier"           : tier,
        "indicators"     : indicators,
        "corr_strength"  : corr_strength,
        "mol_info": {
            "molecule"   : cfg.molecule,
            "basis"      : cfg.basis,
            "n_atoms"    : cfg.n_atoms,
            "atom_syms"  : cfg.atom_syms,
            "n_electrons": mol.nelectron,
            "n_ao"       : mol.nao_nr(),
        },
        "uhf_energy"     : float(mf.e_tot),
        "mp2_energy"     : float(mf.e_tot + e_corr),
        "mp2_ok"         : mp2_ok,
        "mo_coeff_uhf"   : np.asarray(mf.mo_coeff),
        "mo_energy"      : np.asarray(mf.mo_energy),
        "mo_occ"         : np.asarray(mf.mo_occ),
        "converged"      : mf.converged,
    }

    with open(cfg.step1_file, "wb") as f:
        pickle.dump(results, f)

    print(f"\n[Step 1] ✓ Saved → {cfg.step1_file}")
    return results

