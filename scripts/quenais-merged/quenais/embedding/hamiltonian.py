"""
DMET Embedding Hamiltonian for QuEnAIS pipeline.
"""

import os
import time
import pickle
import warnings
import numpy as np


# ── Helper functions ──────────────────────────────────────────────────────────

def lowdin_matrices(S):
    evals, evecs = np.linalg.eigh(S)
    mask      = evals > 1e-15
    sq        = np.sqrt(evals[mask])
    S_sqrt    = (evecs[:, mask] * sq)  @ evecs[:, mask].T
    S_invsqrt = (evecs[:, mask] / sq)  @ evecs[:, mask].T
    return S_sqrt, S_invsqrt


def adaptive_bath(sv, n_imp, max_embed, bath_tol):
    max_bath = min(n_imp, max(0, max_embed - n_imp))

    if max_bath == 0:
        warnings.warn(
            f"max_bath=0: n_imp={n_imp}, max_embed_orbs={max_embed}.\n"
            f"Increase max_embed_orbs to at least {2 * n_imp}.",
            RuntimeWarning,
        )
        return 0, 0.0, 0.0

    if len(sv) == 0:
        return 0, 0.0, 0.0

    sv_arr   = np.asarray(sv, dtype=float)
    n_total  = len(sv_arr)
    sv_above = sv_arr[sv_arr > bath_tol]
    n_above  = len(sv_above)

    if n_above == 0:
        warnings.warn(
            f"All {n_total} singular values are below bath_tolerance={bath_tol}.\n"
            f"Largest SV = {sv_arr[0]:.3e}. Using top SVs anyway.\n"
            f"Lower bath_tolerance in Config to suppress this warning.",
            RuntimeWarning,
        )
        sv_filtered = sv_arr[:max_bath]
    else:
        sv_filtered = sv_above[:max_bath]

    n_avail = len(sv_filtered)

    best_gap, best_n = -1.0, 1
    for n in range(1, n_avail + 1):
        gap = sv_filtered[n - 1] - (sv_filtered[n] if n < n_avail else 0.0)
        if gap > best_gap:
            best_gap, best_n = gap, n

    sv2_total = float(np.sum(sv_filtered ** 2))
    if sv2_total < 1e-30:
        return 0, 0.0, 0.0

    cumsum, n_cov = 0.0, 0
    for i, s in enumerate(sv_filtered):
        cumsum += s * s
        n_cov   = i + 1
        if cumsum / sv2_total >= 0.999:
            break

    n_bath  = min(max(best_n, n_cov), max_bath)
    sv2_cov = float(np.sum(sv_filtered[:n_bath] ** 2) / sv2_total)

    return n_bath, float(best_gap), sv2_cov


def _get_spin_dm_ao(mf, dm1_mp2):
    if isinstance(dm1_mp2, (tuple, list)):
        Ca = np.asarray(mf.mo_coeff[0])
        Cb = np.asarray(mf.mo_coeff[1])
        dm_ao_alpha = Ca @ np.asarray(dm1_mp2[0]) @ Ca.T
        dm_ao_beta  = Cb @ np.asarray(dm1_mp2[1]) @ Cb.T
    else:
        C           = np.asarray(mf.mo_coeff)
        dm_total    = C @ np.asarray(dm1_mp2) @ C.T
        dm_ao_alpha = 0.5 * dm_total
        dm_ao_beta  = 0.5 * dm_total

    dm_ao_total = dm_ao_alpha + dm_ao_beta
    return dm_ao_total, dm_ao_alpha, dm_ao_beta


def _symmetrize_h2e(h2e):
    return (
        h2e
        + h2e.transpose(1, 0, 2, 3)
        + h2e.transpose(0, 1, 3, 2)
        + h2e.transpose(1, 0, 3, 2)
        + h2e.transpose(2, 3, 0, 1)
        + h2e.transpose(3, 2, 0, 1)
        + h2e.transpose(2, 3, 1, 0)
        + h2e.transpose(3, 2, 1, 0)
    ) / 8.0


# ── Entry point ───────────────────────────────────────────────────────────────

def main(cfg, force=False):
    """
    Build DMET embedding Hamiltonian.
    cfg   : quenais.config.Config
    force : rerun even if cached result exists
    """
    from pyscf import gto, scf, mp as pyscf_mp, ao2mo
    from pyscf.scf import hf as pyscf_hf

    os.makedirs(cfg.results_dir, exist_ok=True)

    if os.path.exists(cfg.step2_file) and not force:
        print(f"[Step 2] Using cached result: {cfg.step2_file}")
        return

    if not os.path.exists(cfg.step1_file):
        raise FileNotFoundError(
            f"Step 1 output not found: {cfg.step1_file}\n"
            f"Run finder.main(cfg) first."
        )

    with open(cfg.step1_file, "rb") as f:
        step1 = pickle.load(f)

    nel      = step1["nel"]
    mo_list  = step1["mo_list"]
    mo_coeff = step1["mo_coeff"]
    n_imp    = step1["n_active_orbs"]
    mol_info = step1["mol_info"]

    print(f"\n{'='*60}")
    print(f"[Step 2] DMET Embedding — {mol_info['molecule']}")
    print(f"{'='*60}")
    print(f"  Active space (Step 1): ({nel}e, {n_imp}orb)  MOs={mo_list}")
    print(f"  max_embed_orbs       : {cfg.max_embed_orbs}")
    print(f"  bath_tolerance       : {cfg.bath_tolerance}")

    mol = gto.M(
        atom    = cfg.geometry,
        basis   = cfg.basis,
        charge  = cfg.charge,
        spin    = cfg.spin,
        verbose = 0,
    )
    n_ao = mol.nao_nr()

    # ── Phase A: Restore MF from Step 1 ──────────────────────────────────────
    print(f"\n── Phase A: Restore UHF from Step 1 {'─'*16}")
    mf           = scf.UHF(mol)
    mf.mo_coeff  = step1["mo_coeff_uhf"]
    mf.mo_energy = step1["mo_energy"]
    mf.mo_occ    = step1["mo_occ"]
    mf.e_tot     = step1["uhf_energy"]
    mf.converged = step1["converged"]

    print(f"  UHF energy = {mf.e_tot:.8f} Ha  (restored from Step 1)")

    if not mf.converged:
        warnings.warn(
            "Restored UHF was not converged in Step 1.",
            RuntimeWarning,
        )

    # ── Phase B: MP2 Density Matrix ──────────────────────────────────────────
    print(f"\n── Phase B: MP2 Density Matrix {'─'*22}")

    mp2_ok = False
    e_corr = 0.0

    try:
        mymp         = pyscf_mp.MP2(mf)
        mymp.verbose = 0
        e_corr, _    = mymp.kernel()
        dm1_mp2      = mymp.make_rdm1()
        dm_ao_total, dm_ao_alpha, dm_ao_beta = _get_spin_dm_ao(mf, dm1_mp2)
        mp2_ok = True

    except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
        warnings.warn(
            f"MP2 failed: {e}\nFalling back to UHF density matrix.",
            RuntimeWarning,
        )
        dm_raw = mf.make_rdm1()
        if isinstance(dm_raw, (tuple, list)):
            dm_ao_alpha = np.asarray(dm_raw[0])
            dm_ao_beta  = np.asarray(dm_raw[1])
        else:
            dm_ao_alpha = 0.5 * np.asarray(dm_raw)
            dm_ao_beta  = 0.5 * np.asarray(dm_raw)
        dm_ao_total = dm_ao_alpha + dm_ao_beta

    print(f"  MP2 used  : {mp2_ok}")
    print(f"  E_corr    : {e_corr:.6f} Ha")
    print(f"  DM shape  : alpha={dm_ao_alpha.shape}, beta={dm_ao_beta.shape}")

    # ── Phase C: Schmidt Decomposition ───────────────────────────────────────
    print(f"\n── Phase C: Schmidt Decomposition {'─'*19}")

    S                 = mol.intor("int1e_ovlp")
    S_sqrt, S_invsqrt = lowdin_matrices(S)

    C_imp = mo_coeff[:, mo_list].copy()
    Q_imp = S_sqrt @ C_imp

    dm_lo = S_sqrt @ dm_ao_total @ S_sqrt
    P_env = np.eye(n_ao) - Q_imp @ Q_imp.T
    F     = P_env @ dm_lo @ Q_imp

    U_env, sv, _ = np.linalg.svd(F, full_matrices=True)

    print(f"  Singular values (top 10): "
          f"{np.array2string(sv[:10], precision=4, separator=', ')}")
    print(f"  SVs above bath_tolerance={cfg.bath_tolerance}: "
          f"{int(np.sum(sv > cfg.bath_tolerance))}")

    n_bath, sv_gap, sv2_cov = adaptive_bath(
        sv, n_imp, cfg.max_embed_orbs, cfg.bath_tolerance
    )

    if n_bath < cfg.min_bath_orbs:
        warnings.warn(
            f"Only {n_bath} bath orbital(s) found "
            f"(min_bath_orbs={cfg.min_bath_orbs}).\n"
            f"  Largest SV = {sv[0]:.3e}  "
            f"(bath_tolerance={cfg.bath_tolerance})\n"
            f"  Options:\n"
            f"    1. Use a larger basis (def2-svp or def2-tzvp)\n"
            f"    2. Lower cfg.bath_tolerance\n"
            f"    3. Set cfg.min_bath_orbs=0 to accept pure-impurity embedding",
            RuntimeWarning,
        )
        if n_bath == 0:
            warnings.warn(
                "Proceeding with ZERO bath orbitals — pure impurity embedding.",
                RuntimeWarning,
            )

    if n_bath > 0:
        Q_bath = U_env[:, :n_bath]
        Q_emb  = np.hstack([Q_imp, Q_bath])
    else:
        Q_emb  = Q_imp.copy()

    n_emb = n_imp + n_bath
    C_emb = S_invsqrt @ Q_emb

    print(f"\n  Impurity orbs : {n_imp}")
    print(f"  Bath orbs     : {n_bath}"
          + ("  ⚠ minimal basis → small SVs" if n_bath < 3 else ""))
    print(f"  Total emb orbs: {n_emb}  →  {2*n_emb} qubits")
    print(f"  Largest SV    : {sv[0]:.3e}")
    print(f"  SV gap        : {sv_gap:.4e}")
    print(f"  sv² coverage  : {sv2_cov:.4f}")

    # ── Phase D: Core Mean-Field Potential ────────────────────────────────────
    print(f"\n── Phase D: Core Mean-Field Potential {'─'*15}")

    P_emb_lo         = Q_emb @ Q_emb.T
    P_core_lo        = np.eye(n_ao) - P_emb_lo

    dm_core_lo_alpha = P_core_lo @ (S_sqrt @ dm_ao_alpha @ S_sqrt) @ P_core_lo
    dm_core_lo_beta  = P_core_lo @ (S_sqrt @ dm_ao_beta  @ S_sqrt) @ P_core_lo

    dm_core_alpha    = S_invsqrt @ dm_core_lo_alpha @ S_invsqrt
    dm_core_beta     = S_invsqrt @ dm_core_lo_beta  @ S_invsqrt
    dm_core_alpha    = 0.5 * (dm_core_alpha + dm_core_alpha.T)
    dm_core_beta     = 0.5 * (dm_core_beta  + dm_core_beta.T)
    dm_core_total    = dm_core_alpha + dm_core_beta

    h1e_bare      = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    vj_a, vk_a    = pyscf_hf.get_jk(mol, dm_core_alpha, hermi=1)
    vj_b, vk_b    = pyscf_hf.get_jk(mol, dm_core_beta,  hermi=1)

    h1e_eff       = h1e_bare + (vj_a + vj_b) - 0.5 * vk_a
    h1e_eff_b     = h1e_bare + (vj_a + vj_b) - 0.5 * vk_b
    h1e_eff       = 0.5 * (h1e_eff + h1e_eff_b)

    ecore = mol.energy_nuc() + 0.5 * float(
        np.einsum("ij,ji->", dm_core_total, h1e_bare + h1e_eff)
    )

    print(f"  Core DM trace : alpha={np.trace(dm_core_alpha @ S):.3f}, "
          f"beta={np.trace(dm_core_beta @ S):.3f}")
    print(f"  E_core        : {ecore:.6f} Ha")

    # ── Phase E: Integral Transformation ─────────────────────────────────────
    print(f"\n── Phase E: Integral Transformation {'─'*16}")
    t0 = time.time()

    h1e_emb = C_emb.T @ h1e_eff @ C_emb
    h1e_emb = 0.5 * (h1e_emb + h1e_emb.T)

    h2e_raw = ao2mo.kernel(mol, C_emb, compact=False).reshape(
        n_emb, n_emb, n_emb, n_emb
    )
    h2e_emb = _symmetrize_h2e(h2e_raw)

    n_alpha = nel // 2 + nel % 2
    n_beta  = nel // 2

    elapsed = time.time() - t0
    print(f"  h1e shape: {h1e_emb.shape}")
    print(f"  h2e shape: {h2e_emb.shape}")
    print(f"  Time      : {elapsed:.1f}s")

    sym_err = float(np.max(np.abs(h2e_emb - h2e_emb.transpose(1, 0, 2, 3))))
    print(f"  h2e symmetry error (should be ~0): {sym_err:.2e}")

    print(f"\n{'='*60}")
    print(f"[Step 2] Summary — {mol_info['molecule']}")
    print(f"{'='*60}")
    print(f"  Embedding  : {n_imp}(imp) + {n_bath}(bath) = {n_emb} orbs = {2*n_emb} qubits")
    print(f"  Electrons  : {nel}  ({n_alpha}α + {n_beta}β)")
    print(f"  E_core     : {ecore:.6f} Ha")
    print(f"  sv² cover  : {sv2_cov:.4f}")
    print(f"  MP2 used   : {mp2_ok}")
    print(f"{'='*60}")

    results = {
        "h1e"        : h1e_emb,
        "h2e"        : h2e_emb,
        "ecore"      : ecore,
        "n_emb"      : n_emb,
        "n_imp"      : n_imp,
        "n_bath"     : n_bath,
        "n_alpha"    : n_alpha,
        "n_beta"     : n_beta,
        "sv"         : sv[:n_bath],
        "sv_gap"     : sv_gap,
        "sv2_cov"    : sv2_cov,
        "uhf_energy" : float(mf.e_tot),
        "mp2_used"   : mp2_ok,
        "mp2_corr"   : float(e_corr),
        "mol_info"   : mol_info,
    }

    with open(cfg.step2_file, "wb") as f:
        pickle.dump(results, f)

    print(f"\n[Step 2] ✓ Saved → {cfg.step2_file}")
    return results