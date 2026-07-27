# classical_methods.py — test_8 — Classical Reference Methods
"""
Runs classical quantum chemistry methods on the full molecule for
comparison against the DMET+GQE result.

Methods (config.CLASSICAL_METHODS): HF, MP2, CCSD, CCSD_T, CASSCF, NEVPT2.
CASSCF/NEVPT2 reuse the Step 1 (ASF) active space as their starting guess
if step1_asf.pkl exists -- run ASF.py FIRST for a fair active-space match
with the DMET/GQE pipeline. If you skip that, this falls back to a
generic active-space guess, which can be a meaningless comparison (e.g.
cramming N electrons into N/2 orbitals leaves zero correlating degrees of
freedom, so CASSCF trivially returns exactly the HF energy). run_all.py
runs ASF.py before this script for exactly that reason.

Usage: python classical_methods.py [--force]
"""

import config

import os
import sys
import time
import pickle
import argparse
import warnings

parser = argparse.ArgumentParser(description="Step 0: Classical Reference Methods")
parser.add_argument("--force", action="store_true")
args = parser.parse_args()

os.makedirs(config.RESULTS_DIR, exist_ok=True)
if config.cached_result_is_current(config.STEP0_FILE) and not args.force:
    print(f"[Step 0] Using cached result: {config.STEP0_FILE}")
    sys.exit(0)

from pyscf import gto, scf, mp, cc, mcscf
from pyscf.mrpt import nevpt2 as pyscf_nevpt2

print(f"\n{'='*60}\n[Step 0] Classical Methods — {config.MOLECULE}\n{'='*60}")
print(f"  Basis: {config.BASIS}  Charge: {config.CHARGE}  Spin(2S): {config.SPIN}")
print(f"  Methods: {config.CLASSICAL_METHODS}")

mol = gto.M(atom=config.GEOMETRY, basis=config.BASIS,
            charge=config.CHARGE, spin=config.SPIN, verbose=0)
print(f"  Electrons: {mol.nelectron}   AOs: {mol.nao_nr()}")

step1 = None
if os.path.exists(config.STEP1_FILE):
    with open(config.STEP1_FILE, "rb") as f:
        step1 = pickle.load(f)
    print(f"  Step 1 loaded: ({step1['nel']}e, {step1['n_active_orbs']}orb) active space")
else:
    print(f"  Step 1 not found -- CASSCF/NEVPT2 will use a fallback active space.")
    print(f"  (Run ASF.py first for a meaningful CASSCF comparison number.)")


def _timer(name):
    class T:
        def __enter__(self):
            self.t0 = time.time(); return self
        def __exit__(self, *a):
            print(f"  [{name}] done in {time.time()-self.t0:.1f}s")
    return T()


def _run_hf(mol):
    print(f"\n-- HF --")
    is_restricted = (mol.spin == 0)
    mf = scf.RHF(mol) if is_restricted else scf.UHF(mol)
    mf.max_cycle, mf.level_shift, mf.verbose = 400, 0.3, 0
    with _timer("HF"):
        mf.kernel()
    if not mf.converged:
        nw = mf.newton(); nw.verbose = 0
        nw.kernel(mf.mo_coeff)
        if nw.converged:
            mf.mo_coeff, mf.mo_energy = nw.mo_coeff, nw.mo_energy
            mf.mo_occ, mf.e_tot, mf.converged = nw.mo_occ, nw.e_tot, True
    print(f"  {'RHF' if is_restricted else 'UHF'} energy: {mf.e_tot:.8f} Ha "
          f"(converged: {mf.converged})")
    if not mf.converged:
        warnings.warn("HF did not converge.", RuntimeWarning)
    return mf, float(mf.e_tot)


def _run_mp2(mf):
    print(f"\n-- MP2 --")
    try:
        mymp = mp.MP2(mf); mymp.verbose = 0
        with _timer("MP2"):
            e_corr, _ = mymp.kernel()
        e_mp2 = float(mf.e_tot + e_corr)
        print(f"  E_corr: {e_corr:.8f} Ha   MP2 energy: {e_mp2:.8f} Ha")
        return e_mp2, float(e_corr), mymp
    except Exception as e:
        warnings.warn(f"MP2 failed: {e}", RuntimeWarning)
        return None, None, None


def _run_ccsd(mf):
    print(f"\n-- CCSD --")
    try:
        mycc = cc.CCSD(mf); mycc.verbose, mycc.max_cycle = 0, 200
        with _timer("CCSD"):
            mycc.kernel()
        e_ccsd = float(mf.e_tot + mycc.e_corr)
        print(f"  E_corr: {mycc.e_corr:.8f} Ha   CCSD energy: {e_ccsd:.8f} Ha   "
              f"Converged: {mycc.converged}")
        if not mycc.converged:
            warnings.warn("CCSD did not converge.", RuntimeWarning)
        return e_ccsd, float(mycc.e_corr), mycc
    except Exception as e:
        warnings.warn(f"CCSD failed: {e}", RuntimeWarning)
        return None, None, None


def _run_ccsd_t(mf, mycc):
    print(f"\n-- CCSD(T) --")
    if mycc is None:
        print("  Skipped -- CCSD not available.")
        return None, None
    try:
        with _timer("CCSD(T)"):
            e_t = mycc.ccsd_t()
        e_ccsdt = float(mf.e_tot + mycc.e_corr + e_t)
        print(f"  (T) correction: {e_t:.8f} Ha   CCSD(T) energy: {e_ccsdt:.8f} Ha")
        return e_ccsdt, float(e_t)
    except Exception as e:
        warnings.warn(f"CCSD(T) failed: {e}", RuntimeWarning)
        return None, None


def _run_casscf(mol, mf, nel, norb, mo_guess=None):
    print(f"\n-- CASSCF({nel}e,{norb}o) --")
    try:
        mc = mcscf.CASSCF(mf, norb, nel)
        mc.verbose, mc.max_cycle, mc.conv_tol = 0, 500, 1e-8
        mo = mcscf.addons.sort_mo(mc, mf.mo_coeff, mo_guess, base=0) \
            if mo_guess is not None else mf.mo_coeff
        with _timer(f"CASSCF({nel}e,{norb}o)"):
            mc.kernel(mo)
        print(f"  CASSCF energy: {mc.e_tot:.8f} Ha   CI energy: {mc.e_cas:.8f} Ha   "
              f"Converged: {mc.converged}")
        if not mc.converged:
            warnings.warn("CASSCF did not converge.", RuntimeWarning)
        return float(mc.e_tot), mc
    except Exception as e:
        warnings.warn(f"CASSCF failed: {e}", RuntimeWarning)
        return None, None


def _run_nevpt2(mc):
    print(f"\n-- NEVPT2 --")
    if mc is None:
        print("  Skipped -- CASSCF not available.")
        return None
    try:
        with _timer("NEVPT2"):
            e_nevpt2 = pyscf_nevpt2.NEVPT2(mc).kernel()
        e_total = float(mc.e_tot + e_nevpt2)
        print(f"  E_corr(NEVPT2): {e_nevpt2:.8f} Ha   NEVPT2 energy: {e_total:.8f} Ha")
        return e_total
    except Exception as e:
        warnings.warn(f"NEVPT2 failed: {e}", RuntimeWarning)
        return None


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
results = {"molecule": config.MOLECULE, "basis": config.BASIS, "methods": {}}
t_total = time.time()

mf, e_hf = _run_hf(mol)
results["methods"]["HF"] = {"energy": e_hf, "converged": mf.converged}

mymp = None
if "MP2" in config.CLASSICAL_METHODS:
    e_mp2, e_corr_mp2, mymp = _run_mp2(mf)
    results["methods"]["MP2"] = {"energy": e_mp2, "e_corr": e_corr_mp2, "success": e_mp2 is not None}

mycc = None
if "CCSD" in config.CLASSICAL_METHODS:
    e_ccsd, e_corr_cc, mycc = _run_ccsd(mf)
    results["methods"]["CCSD"] = {
        "energy": e_ccsd, "e_corr": e_corr_cc, "success": e_ccsd is not None,
        "converged": mycc.converged if mycc else False,
    }

if "CCSD_T" in config.CLASSICAL_METHODS:
    e_ccsdt, e_t = _run_ccsd_t(mf, mycc)
    results["methods"]["CCSD_T"] = {
        "energy": e_ccsdt, "e_t_correction": e_t, "success": e_ccsdt is not None,
    }

mc = None
if "CASSCF" in config.CLASSICAL_METHODS:
    if step1 is not None:
        nel_cas, norb_cas, mo_guess = step1["nel"], step1["n_active_orbs"], step1["mo_list"]
        print(f"\n  Using Step 1 active space: ({nel_cas}e, {norb_cas}o)")
    else:
        nel_cas  = min(mol.nelectron, 10)
        norb_cas = min(mol.nao_nr() // 2, 8)
        mo_guess = None
        print(f"\n  No Step 1 found. Fallback: ({nel_cas}e, {norb_cas}o)")
        if nel_cas >= 2 * norb_cas:
            warnings.warn(
                f"Fallback active space ({nel_cas}e, {norb_cas}o) leaves no "
                f"correlating degrees of freedom -- CASSCF will trivially "
                f"return the HF energy. Run ASF.py first.", RuntimeWarning,
            )

    e_casscf, mc = _run_casscf(mol, mf, nel_cas, norb_cas, mo_guess)
    results["methods"]["CASSCF"] = {
        "energy": e_casscf, "nel": nel_cas, "norb": norb_cas,
        "success": e_casscf is not None, "converged": mc.converged if mc else False,
    }

if "NEVPT2" in config.CLASSICAL_METHODS:
    e_nevpt2 = _run_nevpt2(mc)
    results["methods"]["NEVPT2"] = {"energy": e_nevpt2, "success": e_nevpt2 is not None}

results["total_time"] = time.time() - t_total

print(f"\n{'='*60}\n[Step 0] Results — {config.MOLECULE} / {config.BASIS}\n{'='*60}")
print(f"\n  {'Method':<12} {'Energy (Ha)':>16} {'vs HF (Ha)':>14} {'vs HF (kcal/mol)':>18}")
print(f"  {'-'*62}")
for method, data in results["methods"].items():
    e = data.get("energy")
    if e is None:
        print(f"  {method:<12} {'FAILED':>16}")
        continue
    vs_hf = e - e_hf
    print(f"  {method:<12} {e:>16.8f} {vs_hf:>+14.6f} "
          f"{vs_hf * config.HARTREE_TO_KCAL_MOL:>+18.2f}")
print(f"\n  Total time: {results['total_time']:.1f}s\n{'='*60}")

with open(config.STEP0_FILE, "wb") as f:
    pickle.dump(results, f)
print(f"\n[Step 0] Saved -> {config.STEP0_FILE}")