# dmet_lib.py — test_8
"""
Reusable DMET helper functions with NO module-level side effects -- safe
to import from anywhere (DMET.py, gqe_for_qsci.py, a notebook, etc.).

This exists to fix a real bug: gqe_for_qsci.py used to do
`import DMET as dmet_step2` to reuse embedding_consistency_score(). DMET.py
is a SCRIPT -- it parses argv and, on a cache hit, calls sys.exit(0) at
module level. Importing it therefore executed DMET.py's whole
cache-check-and-possibly-recompute logic as a side effect of import, and
on a cache hit, sys.exit(0) killed the entire importing process before any
of ITS OWN code ran.

Fix: the reusable functions live here, with zero side effects. Both
DMET.py and gqe_for_qsci.py import from this module; DMET.py no longer
defines these functions inline.
"""

import warnings
import numpy as np


def get_reference_density(mf, mol, step1, mo_list, mo_coeff, method):
    """
    Returns (dm_ao_total, dm_ao_alpha, dm_ao_beta, info).

    mo_list and mo_coeff are explicit parameters rather than read from a
    calling script's module-level globals -- that implicit-global
    dependency is exactly the kind of thing that breaks silently when a
    function gets reused outside its original script's context.
    """
    if method == "mp2":
        for key in ("dm_ao_total_mp2", "dm_ao_alpha_mp2", "dm_ao_beta_mp2"):
            if key not in step1:
                raise KeyError(
                    f"step1 pickle missing '{key}'. Re-run ASF.py with --force."
                )
        return (step1["dm_ao_total_mp2"], step1["dm_ao_alpha_mp2"],
                step1["dm_ao_beta_mp2"], {"method": "mp2", "recomputed": False})

    elif method == "casci":
        from pyscf import ao2mo, fci
        from pyscf.scf import hf as pyscf_hf
        nel_active = step1["nel"]
        n_active   = len(mo_list)
        n_alpha    = nel_active // 2 + nel_active % 2
        n_beta     = nel_active // 2

        # FIX: this used to build h1e_act from bare kinetic+nuclear
        # attraction only -- completely ignoring the Coulomb/exchange
        # screening from the ~10 "core" (non-active) electrons sitting
        # around the active space. DMET.py's own Phase D DOES add that
        # core mean-field potential when building h1e_emb for the
        # embedding solver -- so the reference density here and the
        # embedding CASCI in gqe_for_qsci.py were being computed from two
        # DIFFERENT effective Hamiltonians (one screened, one not). That
        # mismatch feeds directly into a badly-chosen Schmidt bath, which
        # plausibly explains why mismatch_score/mu/ecore stayed large (or
        # got worse) even after the active-space basis fix. Fixed by
        # applying the same core mean-field correction here, using this
        # basis's own occupation numbers (no_occ, now guaranteed
        # consistent with mo_coeff -- see ASF.py's project_occupations).
        n_mo_total = mo_coeff.shape[1]
        no_occ = step1["no_occ"]
        active_set = set(mo_list)
        dm_core_mo = np.diag([
            no_occ[i] / 2.0 if i not in active_set else 0.0
            for i in range(n_mo_total)
        ])
        dm_core_ao = mo_coeff @ dm_core_mo @ mo_coeff.T  # alpha == beta (closed-shell core)

        h1e_bare = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
        vj_a, vk_a = pyscf_hf.get_jk(mol, dm_core_ao, hermi=1)
        vj_b, vk_b = vj_a, vk_a
        h1e_screened = h1e_bare + (vj_a + vj_b) - 0.5 * (vk_a + vk_b)

        C_active = mo_coeff[:, mo_list]
        h1e_act  = C_active.T @ h1e_screened @ C_active
        h2e_act  = ao2mo.kernel(mol, C_active, compact=False).reshape(
            n_active, n_active, n_active, n_active
        )

        cisolver = fci.direct_spin1.FCI()
        cisolver.verbose = 0
        e_cas, civec = cisolver.kernel(h1e_act, h2e_act, n_active, (n_alpha, n_beta))
        dm_active_a, dm_active_b = cisolver.make_rdm1s(civec, n_active, (n_alpha, n_beta))

        # REAL FIX (found via the [0. 0. 0. 0.] Schmidt-singular-value
        # diagnostic): the old version built dm_full as a bare diagonal
        # (no_occ) for every non-active orbital, then only filled in the
        # active-active block from CASCI -- leaving every active/
        # non-active CROSS TERM at exactly zero. That's a block-diagonal
        # density matrix with NO impurity-environment coupling at all.
        # DMET's Schmidt decomposition exists specifically to extract
        # that coupling from the reference density -- hand it a density
        # with none, and the SVD has nothing real to find (confirmed:
        # all 4 singular values came back exactly 0.0, and the resulting
        # "bath" was numerically arbitrary noise, not real physics,
        # which is exactly why C_emb came out badly non-orthonormal).
        #
        # Fixed by starting from the FULL MP2 density (which has genuine
        # active-core coupling, since real correlated densities aren't
        # block-diagonal) and only overwriting the active-active block
        # with the more accurate CASCI values -- preserves the
        # entanglement structure Schmidt decomposition needs, while
        # still getting a better active-space density where it matters.
        for _key in ("dm_ao_alpha_mp2", "dm_ao_beta_mp2"):
            if _key not in step1:
                raise KeyError(
                    f"step1 pickle missing '{_key}' -- needed even for "
                    f"DMET_REFERENCE='casci' now, to preserve "
                    f"active-core coupling in the reference density. "
                    f"Re-run ASF.py with --force."
                )
        S = mol.intor("int1e_ovlp")
        dm_full_a = mo_coeff.T @ S @ step1["dm_ao_alpha_mp2"] @ S @ mo_coeff
        dm_full_b = mo_coeff.T @ S @ step1["dm_ao_beta_mp2"]  @ S @ mo_coeff

        # [diag] direct look at the actual off-diagonal coupling BEFORE
        # the CASCI block overwrite, to test whether mo_coeff nearly
        # diagonalizes this density on its own (which would mean the
        # zero Schmidt singular values are inherent to the basis choice,
        # not an artifact of the block-diagonal construction we just fixed).
        non_active = [i for i in range(mo_coeff.shape[1]) if i not in set(mo_list)]
        cross_block_a = dm_full_a[np.ix_(mo_list, non_active)]
        cross_block_b = dm_full_b[np.ix_(mo_list, non_active)]
        print(f"  [diag] MP2 density active<->non-active cross-coupling "
              f"(alpha): max|.|={np.max(np.abs(cross_block_a)):.2e}, "
              f"(beta): max|.|={np.max(np.abs(cross_block_b)):.2e}  "
              f"(if these are ~0, mo_coeff is diagonalizing this density "
              f"on its own -- an architectural mismatch, not a bug)")

        for a_i, i in enumerate(mo_list):
            for a_j, j in enumerate(mo_list):
                dm_full_a[i, j] = dm_active_a[a_i, a_j]
                dm_full_b[i, j] = dm_active_b[a_i, a_j]

        dm_ao_alpha = mo_coeff @ dm_full_a @ mo_coeff.T
        dm_ao_beta  = mo_coeff @ dm_full_b @ mo_coeff.T
        dm_ao_total = dm_ao_alpha + dm_ao_beta

        return (dm_ao_total, dm_ao_alpha, dm_ao_beta,
                {"method": "casci", "e_cas": float(e_cas),
                 "n_active": n_active, "nel_active": nel_active})
    else:
        raise ValueError(f"Unknown DMET_REFERENCE='{method}'. Use 'mp2' or 'casci'.")


def chemical_potential_correction(h1e_emb, n_emb, n_alpha, n_beta,
                                   mu_range="auto", max_iter=60, tol=1e-10):
    """
    mu_range="auto" derives the search bracket from h1e_emb's own
    eigenvalue spectrum instead of a fixed guess -- a fixed (-5, 5) Ha
    guess doesn't bracket the true chemical potential once Phase D's core
    mean-field potential shifts h1e_emb's eigenvalues (confirmed on your
    N2 run: 4 of 8 embedding eigenvalues already sat below -5 Ha while the
    target electron count was 3). If bisection still can't bracket the
    target even with the auto-derived range, it warns and returns the
    Hamiltonian unshifted (mu=0.0) rather than silently returning a wrong
    answer -- treat that warning as a real diagnostic, not noise.
    """
    evals_now = np.linalg.eigvalsh(h1e_emb)
    if mu_range == "auto":
        margin = max(1.0, 0.1 * (evals_now.max() - evals_now.min()))
        mu_range = (float(evals_now.min()) - margin, float(evals_now.max()) + margin)

    if n_alpha != n_beta:
        warnings.warn(
            f"chemical_potential_correction assumes n_alpha==n_beta "
            f"(got {n_alpha},{n_beta}); using n_alpha as the target and "
            f"applying the same shift to both spins.", RuntimeWarning,
        )
    target = n_alpha

    def n_below_zero(mu):
        return int(np.sum(np.linalg.eigvalsh(h1e_emb - mu * np.eye(n_emb)) < 0.0))

    lo, hi = mu_range
    n_lo, n_hi = n_below_zero(lo), n_below_zero(hi)
    if not (n_lo <= target <= n_hi):
        warnings.warn(
            f"mu search range {mu_range} does not bracket target electron "
            f"count {target} (n(mu={lo:.3f})={n_lo}, n(mu={hi:.3f})={n_hi}) "
            f"even with the auto-derived bracket. This usually means "
            f"n_alpha != n_beta for this embedding, or a genuine "
            f"degeneracy at the target count. Skipping mu correction.",
            RuntimeWarning,
        )
        return h1e_emb, 0.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if n_below_zero(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    mu = 0.5 * (lo + hi)

    h1e_shifted = h1e_emb - mu * np.eye(n_emb)
    return h1e_shifted, mu


def embedding_consistency_score(step2_result, avg_occs, threshold=0.10):
    """
    Diagnostic only -- compares the occupations implied by whatever
    density built the bath (ref_occ_alpha/beta, saved by DMET.py) against
    whatever a solver later reports (avg_occs). No loop, no rebuild.

    A large, flagged mismatch means the reference density used to build
    the impurity+bath split doesn't match the embedding's own correlated
    solution -- treat it as "don't trust this embedding yet," not as
    something to silently ignore.
    """
    ref_a = step2_result.get("ref_occ_alpha")
    ref_b = step2_result.get("ref_occ_beta")
    if ref_a is None or ref_b is None:
        raise KeyError(
            "step2 pickle has no 'ref_occ_alpha'/'ref_occ_beta' -- re-run DMET.py."
        )
    occ_a, occ_b = avg_occs
    mismatch_a = float(np.mean(np.abs(np.asarray(occ_a) - ref_a)))
    mismatch_b = float(np.mean(np.abs(np.asarray(occ_b) - ref_b)))
    score = 0.5 * (mismatch_a + mismatch_b)
    return {
        "mismatch_alpha": mismatch_a, "mismatch_beta": mismatch_b,
        "mismatch_score": score, "flag": score > threshold,
    }