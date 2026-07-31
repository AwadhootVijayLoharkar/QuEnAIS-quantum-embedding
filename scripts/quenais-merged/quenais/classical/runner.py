"""
Classical reference methods for QuEnAIS pipeline.
"""

import os
import time
import pickle
import warnings
import numpy as np


def main(cfg, force=False):
    """
    Run classical quantum chemistry methods.
    cfg   : quenais.config.Config
    force : rerun even if cached result exists
    """
    from pyscf import gto, scf, mp, cc, mcscf
    from pyscf.mrpt import nevpt2 as pyscf_nevpt2

    os.makedirs(cfg.results_dir, exist_ok=True)

    if os.path.exists(cfg.step0_file) and not force:
        print(f"[Step 0] Using cached result: {cfg.step0_file}")
        return

    print(f"\n{'='*60}")
    print(f"[Step 0] Classical Methods — {cfg.molecule}")
    print(f"{'='*60}")
    print(f"  Basis     : {cfg.basis}")
    print(f"  Charge    : {cfg.charge}   Spin (2S): {cfg.spin}")
    print(f"  Methods   : {cfg.classical_methods}")

    mol = gto.M(
        atom    = cfg.geometry,
        basis   = cfg.basis,
        charge  = cfg.charge,
        spin    = cfg.spin,
        verbose = 0,
    )
    print(f"  Electrons : {mol.nelectron}   AOs: {mol.nao_nr()}")

    step1 = None
    if os.path.exists(cfg.step1_file):
        with open(cfg.step1_file, "rb") as f:
            step1 = pickle.load(f)
        print(f"  Step 1 loaded: ({step1['nel']}e, {step1['n_active_orbs']}orb)")
    else:
        print(f"  Step 1 not found — CASSCF/NEVPT2 will use MP2 natural orbital guess")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _timer(method_name):
    """Simple context manager for timing."""
    class Timer:
        def __enter__(self):
            self.t0 = time.time()
            return self
        def __exit__(self, *a):
            self.elapsed = time.time() - self.t0
            print(f"  [{method_name}] done in {self.elapsed:.1f}s")
    return Timer()


def _run_hf(mol):
    """
    Run RHF for singlet closed-shell, UHF otherwise.
    Returns (mf, energy).
    """
    print(f"\n── HF {'─'*52}")
    is_restricted = (mol.spin == 0)
    mf = scf.RHF(mol) if is_restricted else scf.UHF(mol)
    mf.max_cycle   = 400
    mf.level_shift = 0.3
    mf.verbose     = 0

    with _timer("HF"):
        mf.kernel()

    if not mf.converged:
        nw = mf.newton()
        nw.verbose = 0
        nw.kernel(mf.mo_coeff)
        if nw.converged:
            mf.mo_coeff  = nw.mo_coeff
            mf.mo_energy = nw.mo_energy
            mf.mo_occ    = nw.mo_occ
            mf.e_tot     = nw.e_tot
            mf.converged = True

    label = "RHF" if is_restricted else "UHF"
    print(f"  {label} energy   : {mf.e_tot:.8f} Ha  (converged: {mf.converged})")

    if not mf.converged:
        warnings.warn(f"HF did not converge. Results unreliable.", RuntimeWarning)

    return mf, float(mf.e_tot)


def _run_mp2(mf):
    """Run MP2 on top of the provided mean-field."""
    print(f"\n── MP2 {'─'*51}")
    try:
        mymp = mp.MP2(mf)
        mymp.verbose = 0
        with _timer("MP2"):
            e_corr, _ = mymp.kernel()
        e_mp2 = float(mf.e_tot + e_corr)
        print(f"  E_corr     : {e_corr:.8f} Ha")
        print(f"  MP2 energy : {e_mp2:.8f} Ha")
        return e_mp2, float(e_corr), mymp
    except Exception as e:
        warnings.warn(f"MP2 failed: {e}", RuntimeWarning)
        return None, None, None


def _run_ccsd(mf):
    """Run CCSD on top of the provided mean-field."""
    print(f"\n── CCSD {'─'*50}")
    try:
        mycc = cc.CCSD(mf)
        mycc.verbose  = 0
        mycc.max_cycle = 200
        with _timer("CCSD"):
            mycc.kernel()
        e_ccsd = float(mf.e_tot + mycc.e_corr)
        print(f"  E_corr      : {mycc.e_corr:.8f} Ha")
        print(f"  CCSD energy : {e_ccsd:.8f} Ha")
        print(f"  Converged   : {mycc.converged}")
        if not mycc.converged:
            warnings.warn("CCSD did not converge.", RuntimeWarning)
        return e_ccsd, float(mycc.e_corr), mycc
    except Exception as e:
        warnings.warn(f"CCSD failed: {e}", RuntimeWarning)
        return None, None, None


def _run_ccsd_t(mf, mycc):
    """Run CCSD(T) perturbative triples correction."""
    print(f"\n── CCSD(T) {'─'*47}")
    if mycc is None:
        print("  Skipped — CCSD not available.")
        return None, None
    try:
        with _timer("CCSD(T)"):
            e_t = mycc.ccsd_t()
        e_ccsdt = float(mf.e_tot + mycc.e_corr + e_t)
        print(f"  (T) correction : {e_t:.8f} Ha")
        print(f"  CCSD(T) energy : {e_ccsdt:.8f} Ha")
        return e_ccsdt, float(e_t)
    except Exception as e:
        warnings.warn(f"CCSD(T) failed: {e}", RuntimeWarning)
        return None, None


def _run_casscf(mol, mf, nel, norb, mo_coeff_guess=None):
    """
    Run CASSCF with the given active space.
    Uses mo_coeff_guess if provided (from Step 1 ASF), otherwise MP2 NOs.
    """
    print(f"\n── CASSCF({nel}e, {norb}o) {'─'*43}")
    try:
        mc = mcscf.CASSCF(mf, norb, nel)
        mc.verbose    = 0
        mc.max_cycle  = 500
        mc.conv_tol   = 1e-8

        if mo_coeff_guess is not None:
            # Use ASF-selected orbitals as starting point
            mo = mcscf.addons.sort_mo(mc, mf.mo_coeff, mo_coeff_guess, base=0)
        else:
            mo = mf.mo_coeff

        with _timer(f"CASSCF({nel}e,{norb}o)"):
            mc.kernel(mo)

        print(f"  CASSCF energy  : {mc.e_tot:.8f} Ha")
        print(f"  CI energy      : {mc.e_cas:.8f} Ha")
        print(f"  Converged      : {mc.converged}")
        if not mc.converged:
            warnings.warn("CASSCF did not converge.", RuntimeWarning)
        return float(mc.e_tot), mc
    except Exception as e:
        warnings.warn(f"CASSCF failed: {e}", RuntimeWarning)
        return None, None


def _run_nevpt2(mc):
    """
    Run SC-NEVPT2 (strongly-contracted) on top of CASSCF.
    More numerically stable than CASPT2 for transition metals.
    """
    print(f"\n── NEVPT2 {'─'*48}")
    if mc is None:
        print("  Skipped — CASSCF not available.")
        return None
    try:
        with _timer("NEVPT2"):
            e_nevpt2 = pyscf_nevpt2.NEVPT2(mc).kernel()
        e_total = float(mc.e_tot + e_nevpt2)
        print(f"  E_corr (NEVPT2) : {e_nevpt2:.8f} Ha")
        print(f"  NEVPT2 energy   : {e_total:.8f} Ha")
        return e_total
    except Exception as e:
        warnings.warn(f"NEVPT2 failed: {e}", RuntimeWarning)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

    # ── Run methods ───────────────────────────────────────────────────────────
    results = {
        "molecule" : cfg.molecule,
        "basis"    : cfg.basis,
        "methods"  : {},
    }

    t_total = time.time()

    mf, e_hf = _run_hf(mol)
    results["methods"]["HF"] = {"energy": e_hf, "converged": mf.converged}

    mymp = None
    if "MP2" in cfg.classical_methods:
        e_mp2, e_corr_mp2, mymp = _run_mp2(mf)
        results["methods"]["MP2"] = {
            "energy" : e_mp2,
            "e_corr" : e_corr_mp2,
            "success": e_mp2 is not None,
        }

    mycc = None
    if "CCSD" in cfg.classical_methods:
        e_ccsd, e_corr_cc, mycc = _run_ccsd(mf)
        results["methods"]["CCSD"] = {
            "energy"   : e_ccsd,
            "e_corr"   : e_corr_cc,
            "success"  : e_ccsd is not None,
            "converged": mycc.converged if mycc else False,
        }

    if "CCSD_T" in cfg.classical_methods:
        e_ccsdt, e_t = _run_ccsd_t(mf, mycc)
        results["methods"]["CCSD_T"] = {
            "energy"        : e_ccsdt,
            "e_t_correction": e_t,
            "success"       : e_ccsdt is not None,
        }

    mc = None
    if "CASSCF" in cfg.classical_methods:
        if step1 is not None:
            nel_cas  = step1["nel"]
            norb_cas = step1["n_active_orbs"]
            mo_guess = step1["mo_list"]
        else:
            nel_cas  = min(mol.nelectron, 10)
            norb_cas = min(mol.nao_nr() // 2, 8)
            mo_guess = None
        e_casscf, mc = _run_casscf(mol, mf, nel_cas, norb_cas, mo_guess)
        results["methods"]["CASSCF"] = {
            "energy"   : e_casscf,
            "nel"      : nel_cas,
            "norb"     : norb_cas,
            "success"  : e_casscf is not None,
            "converged": mc.converged if mc else False,
        }

    if "NEVPT2" in cfg.classical_methods:
        e_nevpt2 = _run_nevpt2(mc)
        results["methods"]["NEVPT2"] = {
            "energy" : e_nevpt2,
            "success": e_nevpt2 is not None,
        }

    results["total_time"] = time.time() - t_total

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Step 0] Results — {cfg.molecule} / {cfg.basis}")
    print(f"{'='*60}")
    print(f"\n  {'Method':<12} {'Energy (Ha)':>16} {'vs HF (Ha)':>14} "
          f"{'vs HF (kcal/mol)':>18}")
    print(f"  {'─'*62}")

    for method, data in results["methods"].items():
        e = data.get("energy")
        if e is None:
            print(f"  {method:<12} {'FAILED':>16}")
            continue
        vs_hf    = e - e_hf
        vs_hf_kc = vs_hf * cfg.hartree_to_kcal_mol
        print(f"  {method:<12} {e:>16.8f} {vs_hf:>+14.6f} {vs_hf_kc:>+18.2f}")

    print(f"\n  Total time: {results['total_time']:.1f}s")
    print(f"{'='*60}")

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(cfg.step0_file, "wb") as f:
        pickle.dump(results, f)

    print(f"\n[Step 0] ✓ Saved → {cfg.step0_file}")
    return results