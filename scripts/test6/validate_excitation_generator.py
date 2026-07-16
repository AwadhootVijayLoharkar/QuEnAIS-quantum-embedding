# validate_excitation_generator.py
#
# RUN THIS BEFORE TRUSTING ANY RESULT FROM dmet_excitation_pool.py.
#
# Cross-checks excitation_generator_qubit_op() (the pure-OpenFermion
# excitation generator used by DMETExcitationPool / DMETPauliEvolutionPool)
# against tequila's actual make_excitation_gate() output, for hand-picked
# index pairs, on a real molecule (H2/STO-3G) -- completely independent of
# any CCSD amplitude computation or the DMET adapter, so this isolates
# exactly the one thing I couldn't verify myself (no tequila/openfermion/
# cudaq available in my sandbox): does my generator match tequila's sign
# and normalization convention?
#
# Expected outcome if the conventions agree: every Pauli string that
# appears in tequila's generator also appears in mine with the SAME
# coefficient ratio (should print "1.0" or a single consistent constant
# for every term). If the ratio is consistent but not 1.0 (e.g. always
# -1.0, or always some factor), that's a trivial one-line fix -- negate
# (or scale) the generator in excitation_generator_qubit_op(). If the
# ratio is INCONSISTENT across terms, or the Pauli-string sets don't
# match, stop and don't trust dmet_excitation_pool.py -- come back and
# tell me exactly what printed, and I'll fix the generator construction.
#
# Run this on your HPC where tequila + openfermion + cudaq are installed:
#   python validate_excitation_generator.py

import numpy as np
import tequila as tq
from openfermion import jordan_wigner

from dmet_excitation_pool import excitation_generator_qubit_op


def tequila_generator_terms(tq_molecule, indices, angle=1.0):
    """
    Build the tequila excitation gate and pull out its generator's
    Pauli decomposition as {pauli_word_string: coeff}.
    """
    gate = tq_molecule.make_excitation_gate(indices=indices, angle=angle)
    terms = {}
    for g in [gate] if not hasattr(gate, "gates") else gate.gates:
        for p in g.generator.paulistrings:
            # p behaves like a dict {qubit_idx: pauli_letter}, plus p._coeff
            word = ",".join(f"{k}{v}" for k, v in sorted(p.items()))
            coeff = p._coeff
            terms[word] = terms.get(word, 0.0) + coeff
    return terms


def openfermion_generator_terms(indices):
    """
    Build my OpenFermion-based generator and pull out its Pauli
    decomposition in the SAME {pauli_word_string: coeff} format, so the
    two dicts are directly comparable key-for-key.
    """
    qubit_op = excitation_generator_qubit_op(indices)
    terms = {}
    for term, coeff in qubit_op.terms.items():
        if len(term) == 0:
            continue
        word = ",".join(f"{idx}{letter}" for idx, letter in sorted(term))
        terms[word] = terms.get(word, 0.0) + coeff
    return terms


def compare(name, indices, tq_molecule):
    print(f"\n{'='*70}")
    print(f"Test case: {name}   indices={indices}")
    print(f"{'='*70}")

    ref = tequila_generator_terms(tq_molecule, indices, angle=1.0)
    mine = openfermion_generator_terms(indices)

    all_words = sorted(set(ref) | set(mine))
    ratios = []
    ok = True
    for w in all_words:
        rv = ref.get(w, None)
        mv = mine.get(w, None)
        if rv is None or mv is None:
            print(f"  MISMATCH  {w!r:20s}  tequila={rv}   mine={mv}   <-- present in only one!")
            ok = False
            continue
        ratio = mv / rv if abs(rv) > 1e-14 else float("nan")
        ratios.append(ratio)
        print(f"  {w!r:20s}  tequila={rv:.6f}   mine={mv:.6f}   ratio(mine/tequila)={ratio:.6f}")

    if ok and ratios:
        spread = max(ratios) - min(ratios)
        if spread < 1e-8:
            print(f"\n  PASS: identical Pauli-string set, consistent ratio = {ratios[0]:.6f}")
            if abs(ratios[0] - 1.0) > 1e-8:
                print(f"  NOTE: ratio is not 1.0 -- multiply excitation_generator_qubit_op()'s")
                print(f"        result by {ratios[0]:.6f} to match tequila's convention exactly.")
        else:
            print(f"\n  FAIL: ratios are inconsistent across terms (spread={spread:.6f}).")
            print(f"        Do not trust dmet_excitation_pool.py yet -- report this output.")
    elif not ok:
        print(f"\n  FAIL: Pauli-string sets don't match. Do not trust dmet_excitation_pool.py")
        print(f"        yet -- report this output.")


if __name__ == "__main__":
    # H2 / STO-3G, 2 electrons in 2 spatial orbitals (4 spin-orbitals).
    # Spin-orbital indexing here matches the interleaved convention used
    # throughout gqe-for-qsci's operator_pool.py: 2*i = alpha_i, 2*i+1 = beta_i.
    tq_molecule = tq.Molecule(
        geometry="H 0.0 0.0 0.0\nH 0.0 0.0 0.74",
        basis_set="sto-3g",
        active_orbitals=[0, 1],
        transformation="jordan-wigner",
    )

    # Single excitation: spatial orbital 0 -> 1, alpha spin
    # (matches how generate_excitations() builds idx_a = (2*key[0], 2*key[1]))
    compare("single excitation (alpha, 0->1)", [(0, 2)], tq_molecule)

    # Single excitation: spatial orbital 0 -> 1, beta spin
    compare("single excitation (beta, 0->1)", [(1, 3)], tq_molecule)

    # Double excitation, alpha-beta (matches idx_abab construction)
    compare("double excitation (alpha-beta)", [(1, 3), (0, 2)], tq_molecule)

    print(f"\n{'='*70}")
    print("If all cases above say PASS with a consistent ratio (ideally 1.0,")
    print("or note the fixed scale factor to apply), the generator")
    print("construction in dmet_excitation_pool.py is confirmed correct and")
    print("safe to use on DMET's embedding Hamiltonian.")
    print(f"{'='*70}")