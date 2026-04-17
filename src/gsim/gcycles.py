"""
Utilities for translation-related Pauli closures on 1D chains.

This module implements Lie-closure and preprocessing routines for Pauli-string
generators with translation structure on a chain of length `n`. It supports two
boundary conventions:

1. periodic boundaries (`boundaries='periodic'`)
   Generators are represented by canonical translation classes ("cycles").
   Each Pauli pattern is reduced to a canonical representative under cyclic
   shifts.

2. open boundaries (`boundaries='open'`)
   Generators are represented in a mixed "abstract" form consisting of
   - bulk terms, which stand for a translational family of shifts, and
   - fixed terms, which encode boundary corrections.
   Internally, these are flattened to dense sparse vectors for Gaussian
   elimination and then lifted back to the abstract bulk/fixed form.

The main public entry points are

- `lie_closure_cycles(...)`
    Compute only the closed Lie basis.
- `pauli_cycle_preprocessing(...)`
    Compute the closed Lie basis together with sparse commutator data.

Representation conventions
--------------------------
Pauli strings are internally encoded by compact immutable signatures

    ((q0, p0), (q1, p1), ...)

where each `qi` is a site index and each `pi` is an integer code:
`X -> 1`, `Y -> 2`, `Z -> 3`.

For open boundaries, abstract vectors use keys of the form

    ('B', sig)   bulk/translational family
    ('F', sig)   fixed boundary correction

where `sig` is again an integer Pauli signature.

Notes
-----
- This module builds on `NormalizedBasis` from `gstrings.py` for sparse basis
  management and Gaussian elimination.
- The periodic and open-boundary workflows share the same high-level closure
  logic but use different commutator and canonicalization routines.
- Numerical coefficients smaller than `THRESHOLD` are discarded.
"""

import warnings
import PauliEngine as pe
import numpy as np
from typing import Dict, Tuple, List, Literal
from collections import Counter

from .gstrings import NormalizedBasis

# -- globals --
THRESHOLD = 1e-12
_P2I = {'X': 1, 'Y': 2, 'Z': 3}
_I2P = {1: 'X', 2: 'Y', 3: 'Z'}

# -- utils --

def get_canonical_cycle(sig: Tuple[Tuple[int, int], ...], n: int) -> Tuple:
    """
    Return the canonical cyclic representative of a Pauli signature.

    Parameters
    ----------
    sig : Tuple[Tuple[int, int], ...]
        Sparse Pauli signature encoded as `(site, pauli_code)` pairs.
    n : int
        Chain length.

    Returns
    -------
    Tuple
        Lexicographically smallest representative obtained from `sig` under all
        cyclic shifts on a length-`n` ring.

    Notes
    -----
    This is the basic canonicalization routine for the periodic-boundary
    workflow. The empty signature is returned unchanged.
    """
    if not sig: return ()
    best_sig = None
    for k, _ in sig:
        shifted_sig = tuple(sorted(((idx - k) % n, pauli) for idx, pauli in sig))
        if best_sig is None or shifted_sig < best_sig:
            best_sig = shifted_sig
    return best_sig


def _to_canonical_sparse_vector(pauli_list: List[pe.PauliString], n: int) -> Dict[Tuple, complex]:
    """
    Return the canonical cyclic representative of a Pauli signature.

    Parameters
    ----------
    sig : Tuple[Tuple[int, int], ...]
        Sparse Pauli signature encoded as `(site, pauli_code)` pairs.
    n : int
        Chain length.

    Returns
    -------
    Tuple
        Lexicographically smallest representative obtained from `sig` under all
        cyclic shifts on a length-`n` ring.

    Notes
    -----
    This is the basic canonicalization routine for the periodic-boundary
    workflow. The empty signature is returned unchanged.
    """
    vec = {}
    for ps in pauli_list:
        c_expr, p_dict = ps.to_dictionary()
        c = pe.PauliString.to_complex(c_expr)
        if abs(c) < THRESHOLD: continue
        sig = tuple(sorted((k, _P2I[v]) for k, v in p_dict.items()))
        canonical_sig = get_canonical_cycle(sig, n)
        vec[canonical_sig] = vec.get(canonical_sig, 0.0) + c
    return {k: v for k, v in vec.items() if abs(v) > THRESHOLD}


def cycle_commutator(vec_P: Dict[Tuple, complex], vec_Q: Dict[Tuple, complex],
                     n: int, ps_cache: dict) -> Dict[Tuple, complex]:
    """
    Compute the commutator of two periodic translation-reduced Pauli sums.

    Parameters
    ----------
    vec_P, vec_Q : Dict[Tuple, complex]
        Sparse vectors in canonical cyclic-signature form.
    n : int
        Chain length.
    ps_cache : dict
        Cache mapping signatures to unit-coefficient `pe.PauliString` objects.

    Returns
    -------
    Dict[Tuple, complex]
        Sparse commutator vector in canonical cyclic-signature form.

    Notes
    -----
    The commutator is evaluated by summing over the active relative shifts that
    can produce overlapping support between the two Pauli representatives.
    """
    total_commutator = {}
    for (sig_p, coeff_p) in vec_P.items():
        for (sig_q, coeff_q) in vec_Q.items():
            combined_coeff = coeff_p * coeff_q
            if sig_p not in ps_cache:
                ps_cache[sig_p] = pe.PauliString({k: _I2P[v] for k, v in sig_p}, 1.0)
            ps_p = ps_cache[sig_p]

            active_shifts = set()
            for i, _ in sig_p:
                for j, _ in sig_q:
                    active_shifts.add((i - j) % n)

            for m in active_shifts:
                shifted_sig_q = tuple(sorted(((idx + m) % n, p) for idx, p in sig_q))
                if shifted_sig_q not in ps_cache:
                    ps_cache[shifted_sig_q] = pe.PauliString({k: _I2P[v] for k, v in shifted_sig_q}, 1.0)
                ps_TmQ = ps_cache[shifted_sig_q]

                comm = ps_p.commutator(ps_TmQ)
                c_expr, res_dict = comm.to_dictionary()
                val = pe.PauliString.to_complex(c_expr) * combined_coeff

                if abs(val) > THRESHOLD:
                    res_sig_int = tuple(sorted((k, _P2I[v]) for k, v in res_dict.items()))
                    canonical_sig = get_canonical_cycle(res_sig_int, n)
                    total_commutator[canonical_sig] = total_commutator.get(canonical_sig, 0.0) + val

    return {k: v for k, v in total_commutator.items() if abs(v) > THRESHOLD}


def _shift_sig_obc(sig: Tuple, offset: int) -> Tuple:
    """
    Shift an open-boundary Pauli signature by a fixed offset.

    Parameters
    ----------
    sig : Tuple
        Sparse Pauli signature.
    offset : int
        Integer site shift.

    Returns
    -------
    Tuple
        Shifted signature with site indices increased by `offset`.
    """
    return tuple(sorted((idx + offset, p) for idx, p in sig))


def _get_obc_canonical_and_offset(sig: Tuple) -> Tuple[Tuple, int]:
    """
    Canonicalize an open-boundary signature by shifting its leftmost support to site 0.

    Parameters
    ----------
    sig : Tuple
        Sparse Pauli signature.

    Returns
    -------
    Tuple[Tuple, int]
        `(canonical_sig, offset)` where `canonical_sig` is the left-aligned
        representative and `offset` is the original minimum site index.

    Notes
    -----
    This is the basic normalization used to separate bulk translational content
    from boundary corrections in the open-boundary workflow.
    """
    if not sig: return (), 0
    min_idx = min(idx for idx, _ in sig)
    canonical = tuple(sorted((idx - min_idx, p) for idx, p in sig))
    return canonical, min_idx


def _to_obc_abstract_vector(pauli_list: List[pe.PauliString], n: int) -> Dict[Tuple, complex]:
    """
    Convert Pauli strings into the abstract open-boundary bulk/fixed representation.

    Parameters
    ----------
    pauli_list : List[pe.PauliString]
        PauliEngine Pauli strings, possibly given as abstract representatives.
    n : int
        Chain length.

    Returns
    -------
    Dict[Tuple, complex]
        Sparse abstract vector with keys of the form
        `('B', canonical_sig)` for bulk terms and `('F', sig)` for fixed terms.

    Notes
    -----
    This routine detects repeated translated occurrences of the same canonical
    pattern and factors out the dominant translationally invariant bulk
    contribution, leaving only boundary-sensitive corrections as fixed terms.
    """
    tally = {}
    for ps in pauli_list:
        c_expr, p_dict = ps.to_dictionary()
        c = pe.PauliString.to_complex(c_expr)
        if abs(c) < THRESHOLD: continue

        sig = tuple(sorted((k, _P2I[v]) for k, v in p_dict.items()))
        can_sig, offset = _get_obc_canonical_and_offset(sig)

        if can_sig not in tally: tally[can_sig] = {}
        tally[can_sig][offset] = tally[can_sig].get(offset, 0.0) + c

    vec = {}
    for can_sig, offsets in tally.items():
        span = max(idx for idx, _ in can_sig) + 1 if can_sig else 0
        expected_positions = n - span + 1

        if len(offsets) == 1 and expected_positions > 1:
            offset_val = list(offsets.keys())[0]
            if offset_val == 0:
                bulk_c = list(offsets.values())[0]
                vec[('B', can_sig)] = bulk_c
                continue

        if expected_positions > 0:
            counts = Counter(complex(round(c.real, 8), round(c.imag, 8)) for c in offsets.values())
            missing = expected_positions - len(offsets)
            if missing > 0: counts[0.0j] += missing
            bulk_c = complex(counts.most_common(1)[0][0])
        else:
            bulk_c = 0.0

        if abs(bulk_c) > THRESHOLD:
            vec[('B', can_sig)] = bulk_c

        for i in range(expected_positions):
            c = offsets.get(i, 0.0)
            diff = c - bulk_c
            if abs(diff) > THRESHOLD:
                sig_i = _shift_sig_obc(can_sig, i)
                vec[('F', sig_i)] = vec.get(('F', sig_i), 0.0) + diff

        for i, c in offsets.items():
            if i < 0 or i >= expected_positions:
                sig_i = _shift_sig_obc(can_sig, i)
                vec[('F', sig_i)] = vec.get(('F', sig_i), 0.0) + c

    return {k: v for k, v in vec.items() if abs(v) > THRESHOLD}


def _lift_dense_to_abstract(dense_vec: Dict[Tuple, complex], n: int) -> Dict[Tuple, complex]:
    """
    Lift a dense open-boundary sparse vector back to the abstract bulk/fixed form.

    Parameters
    ----------
    dense_vec : Dict[Tuple, complex]
        Dense sparse vector keyed directly by site-resolved signatures.
    n : int
        Chain length.

    Returns
    -------
    Dict[Tuple, complex]
        Abstract open-boundary vector with bulk and fixed terms.

    Notes
    -----
    This is the inverse parsing step used after Gaussian elimination in the
    dense representation. It removes redundant translational "null-space bloat"
    by reconstructing the compact bulk/fixed description.
    """
    tally = {}
    for sig, c in dense_vec.items():
        can_sig, offset = _get_obc_canonical_and_offset(sig)
        if can_sig not in tally: tally[can_sig] = {}
        tally[can_sig][offset] = tally[can_sig].get(offset, 0.0) + c

    vec = {}
    for can_sig, offsets in tally.items():
        span = max(idx for idx, _ in can_sig) + 1 if can_sig else 0
        expected_positions = n - span + 1

        if expected_positions > 0:
            counts = Counter(complex(round(c.real, 8), round(c.imag, 8)) for c in offsets.values())
            missing = expected_positions - len(offsets)
            if missing > 0: counts[0.0j] += missing
            bulk_c = complex(counts.most_common(1)[0][0])
        else:
            bulk_c = 0.0

        if abs(bulk_c) > THRESHOLD:
            vec[('B', can_sig)] = bulk_c

        for i in range(expected_positions):
            c = offsets.get(i, 0.0)
            diff = c - bulk_c
            if abs(diff) > THRESHOLD:
                sig_i = _shift_sig_obc(can_sig, i)
                vec[('F', sig_i)] = vec.get(('F', sig_i), 0.0) + diff

        for i, c in offsets.items():
            if i < 0 or i >= expected_positions:
                sig_i = _shift_sig_obc(can_sig, i)
                vec[('F', sig_i)] = vec.get(('F', sig_i), 0.0) + c

    return {k: v for k, v in vec.items() if abs(v) > THRESHOLD}


def _flatten_obc_vector(abs_vec: Dict[Tuple, complex], n: int) -> Dict[Tuple, complex]:
    """
    Expand an abstract open-boundary bulk/fixed vector into a dense sparse vector.

    Parameters
    ----------
    abs_vec : Dict[Tuple, complex]
        Abstract open-boundary vector with bulk (`'B'`) and fixed (`'F'`) keys.
    n : int
        Chain length.

    Returns
    -------
    Dict[Tuple, complex]
        Dense sparse vector indexed directly by concrete site-resolved signatures.

    Notes
    -----
    Bulk terms are expanded over all allowed shifts on the chain, while fixed
    terms are inserted directly.
    """
    dense = {}
    for (type_flag, sig), c in abs_vec.items():
        if type_flag == 'B':
            span = max(idx for idx, _ in sig) + 1 if sig else 0
            for i in range(n - span + 1):
                sig_i = _shift_sig_obc(sig, i)
                dense[sig_i] = dense.get(sig_i, 0.0) + c
        elif type_flag == 'F':
            dense[sig] = dense.get(sig, 0.0) + c
    return {k: v for k, v in dense.items() if abs(v) > THRESHOLD}


def obc_commutator(vec_A: Dict[Tuple, complex], vec_B: Dict[Tuple, complex],
                   n: int, ps_cache: dict) -> Dict[Tuple, complex]:
    """
    Compute the commutator of two open-boundary abstract Pauli vectors.

    Parameters
    ----------
    vec_A, vec_B : Dict[Tuple, complex]
        Abstract open-boundary vectors with bulk/fixed keys.
    n : int
        Chain length.
    ps_cache : dict
        Cache mapping signatures to unit-coefficient `pe.PauliString` objects.

    Returns
    -------
    Dict[Tuple, complex]
        Abstract open-boundary commutator vector in bulk/fixed form.

    Notes
    -----
    This routine handles all combinations of bulk and fixed terms:
    - fixed-fixed,
    - bulk-fixed,
    - bulk-bulk.

    Internally it uses a safe commutator routine that temporarily shifts
    signatures upward if negative indices would appear during intermediate
    calculations.
    """
    total_comm = {}

    def add_to_comm(k, v):
        if abs(v) > THRESHOLD:
            total_comm[k] = total_comm.get(k, 0.0) + v

    def get_ps(sig):
        if sig not in ps_cache:
            ps_cache[sig] = pe.PauliString({k: _I2P[v] for k, v in sig}, 1.0)
        return ps_cache[sig]

    def safe_commute(sig1, sig2):
        min1 = min((idx for idx, _ in sig1), default=0)
        min2 = min((idx for idx, _ in sig2), default=0)
        global_min = min(min1, min2)

        if global_min < 0:
            shift_up = -global_min
            safe_sig1 = _shift_sig_obc(sig1, shift_up)
            safe_sig2 = _shift_sig_obc(sig2, shift_up)
            comm = get_ps(safe_sig1).commutator(get_ps(safe_sig2))
            val = pe.PauliString.to_complex(comm.get_coeff())
            if abs(val) < THRESHOLD: return (), 0.0

            res_dict = comm.to_dictionary()[1]
            safe_res_sig = tuple(sorted((k, _P2I[v]) for k, v in res_dict.items()))
            return _shift_sig_obc(safe_res_sig, -shift_up), val
        else:
            comm = get_ps(sig1).commutator(get_ps(sig2))
            val = pe.PauliString.to_complex(comm.get_coeff())
            if abs(val) < THRESHOLD: return (), 0.0

            res_dict = comm.to_dictionary()[1]
            final_res_sig = tuple(sorted((k, _P2I[v]) for k, v in res_dict.items()))
            return final_res_sig, val

    for (type_a, sig_a), c_a in vec_A.items():
        for (type_b, sig_b), c_b in vec_B.items():
            c_ab = c_a * c_b
            span_a = max(idx for idx, _ in sig_a) + 1 if sig_a else 0
            span_b = max(idx for idx, _ in sig_b) + 1 if sig_b else 0

            if type_a == 'F' and type_b == 'F':
                res_sig, val = safe_commute(sig_a, sig_b)
                if abs(val) > THRESHOLD:
                    add_to_comm(('F', res_sig), c_ab * val)

            elif (type_a == 'B' and type_b == 'F') or (type_a == 'F' and type_b == 'B'):
                is_swapped = (type_a == 'F')
                bulk_sig, bulk_span = (sig_b, span_b) if is_swapped else (sig_a, span_a)
                fixed_sig = sig_a if is_swapped else sig_b

                min_f = min(idx for idx, _ in fixed_sig) if fixed_sig else 0
                max_f = max(idx for idx, _ in fixed_sig) if fixed_sig else -1

                for i in range(max(0, min_f - bulk_span + 1), min(n - bulk_span + 1, max_f + 1)):
                    shifted_bulk_sig = _shift_sig_obc(bulk_sig, i)
                    sig1 = fixed_sig if is_swapped else shifted_bulk_sig
                    sig2 = shifted_bulk_sig if is_swapped else fixed_sig

                    res_sig, val = safe_commute(sig1, sig2)
                    if abs(val) > THRESHOLD:
                        add_to_comm(('F', res_sig), c_ab * val)

            elif type_a == 'B' and type_b == 'B':
                for m in range(1 - span_b, span_a):
                    shifted_sig_bm = _shift_sig_obc(sig_b, m)
                    res_sig, val = safe_commute(sig_a, shifted_sig_bm)
                    if abs(val) < THRESHOLD: continue

                    can_res, offset = _get_obc_canonical_and_offset(res_sig)
                    span_res = max(idx for idx, _ in can_res) + 1 if can_res else 0

                    i_min = max(0, -m)
                    i_max = min(n - span_a, n - span_b - m)
                    if i_max < i_min: continue

                    add_to_comm(('B', can_res), c_ab * val)

                    for k in range(0, i_min + offset):
                        if k <= n - span_res:
                            add_to_comm(('F', _shift_sig_obc(can_res, k)), -c_ab * val)

                    for k in range(i_max + offset + 1, n - span_res + 1):
                        if k >= 0:
                            add_to_comm(('F', _shift_sig_obc(can_res, k)), -c_ab * val)

    return {k: v for k, v in total_comm.items() if abs(v) > THRESHOLD}


def _reconstruct_output_basis(space: NormalizedBasis) -> List[List[pe.PauliString]]:
    """
    Reconstruct the final Lie basis in user-facing PauliEngine format.

    Parameters
    ----------
    space : NormalizedBasis
        Internal normalized sparse basis.
    n : int
        Chain length.

    Returns
    -------
    List[List[pe.PauliString]]
        Final Lie basis as lists of PauliEngine Pauli strings.

    Notes
    -----
    The returned format is the same for both boundary conventions. This routine
    only converts the internal sparse signature representation back to
    `pe.PauliString` objects.
    """
    final_basis = []
    for vec in space.matrix:
        op_list = []
        for sig_key, coeff in vec.items():
            c_clean = complex(round(coeff.real, 10), round(coeff.imag, 10))
            sig_dict = {k: _I2P[v] for k, v in sig_key}
            op_list.append(pe.PauliString(sig_dict, c_clean))
        final_basis.append(op_list)
    return final_basis


# -- main --

def lie_closure_cycles(generators: List[List[pe.PauliString]], n: int,
                       boundaries: Literal['periodic', 'open'] = 'periodic',
                       max_iterations: int = 10000, verbose: bool = False) -> List[List[pe.PauliString]]:
    """
    Compute the Lie closure of translation-structured Pauli generators.

    Parameters
    ----------
    generators : List[List[pe.PauliString]]
        Initial generating set, where each generator is a linear combination of
        Pauli strings.
    n : int
        Chain length.
    boundaries : {'periodic', 'open'}, optional
        Boundary convention used for canonicalization and commutator handling.
    max_iterations : int, optional
        Maximum number of closure epochs.
    verbose : bool, optional
        If `True`, print progress information.

    Returns
    -------
    List[List[pe.PauliString]]
        Closed Lie basis in user-facing PauliEngine format.

    Notes
    -----
    For `boundaries='periodic'`, the computation is performed directly on
    canonical cyclic representatives.
    For `boundaries='open'`, generators are first parsed into an abstract
    bulk/fixed form, flattened for elimination, and lifted back as needed.
    """
    space = NormalizedBasis()
    ps_cache = {}
    abstract_basis = []

    for gen in generators:
        if boundaries == 'periodic':
            vec = _to_canonical_sparse_vector(gen, n)
            for sig in vec:
                if sig not in ps_cache: ps_cache[sig] = pe.PauliString({k: _I2P[v] for k, v in sig}, 1.0)

            _, remainder = space.decompose(vec)
            if remainder:
                space.insert_remainder(remainder)
        else:
            abs_vec = _to_obc_abstract_vector(gen, n)
            dense_vec = _flatten_obc_vector(abs_vec, n)

            for (t, sig) in abs_vec:
                if sig not in ps_cache: ps_cache[sig] = pe.PauliString({k: _I2P[v] for k, v in sig}, 1.0)

            coeffs, remainder = space.decompose(dense_vec)
            if remainder:
                new_idx, _ = space.insert_remainder(remainder)

                abs_rem = _lift_dense_to_abstract(space.matrix[new_idx], n)
                for (t, sig) in abs_rem:
                    if sig not in ps_cache: ps_cache[sig] = pe.PauliString({k: _I2P[v] for k, v in sig}, 1.0)
                abstract_basis.append(abs_rem)

    if verbose: print(f"Preprocessing done. Initial basis size: {len(space.matrix)}", flush=True)

    epoch, old_len, new_len = 0, 0, len(space.matrix)

    while new_len > old_len and epoch < max_iterations:
        current_basis = space.matrix

        for alpha_idx in range(old_len, new_len):
            for beta_idx in range(alpha_idx):
                vec_A, vec_B = current_basis[alpha_idx], current_basis[beta_idx]

                if boundaries == 'periodic':
                    commutator_vec = cycle_commutator(vec_A, vec_B, n, ps_cache)
                    _, remainder = space.decompose(commutator_vec)
                    if remainder:
                        space.insert_remainder(remainder)
                else:
                    abs_A = abstract_basis[alpha_idx]
                    abs_B = abstract_basis[beta_idx]
                    abs_comm = obc_commutator(abs_A, abs_B, n, ps_cache)
                    dense_comm = _flatten_obc_vector(abs_comm, n)

                    coeffs, remainder = space.decompose(dense_comm)
                    if remainder:
                        new_idx, _ = space.insert_remainder(remainder)

                        abs_rem = _lift_dense_to_abstract(space.matrix[new_idx], n)
                        for (t, sig) in abs_rem:
                            if sig not in ps_cache: ps_cache[sig] = pe.PauliString({k: _I2P[v] for k, v in sig}, 1.0)
                        abstract_basis.append(abs_rem)

        old_len = new_len
        new_len = len(space.matrix)
        epoch += 1
        if verbose: print(f"Epoch {epoch} complete. Basis size: {new_len}", flush=True)

    if epoch == max_iterations:
        warnings.warn(f"Reached max iterations {max_iterations} in lie_closure_cycles", UserWarning)

    return _reconstruct_output_basis(space)


def pauli_cycle_preprocessing(generators: List[List[pe.PauliString]], n: int,
                              boundaries: Literal['periodic', 'open'] = 'periodic',
                              max_iterations: int = 10000, verbose: bool = False):
    """
    Compute Lie closure and sparse commutator data for translation-structured Pauli generators.

    Parameters
    ----------
    generators : List[List[pe.PauliString]]
        Initial generating set, where each generator is a linear combination of
        Pauli strings.
    n : int
        Chain length.
    boundaries : {'periodic', 'open'}, optional
        Boundary convention used for canonicalization and commutator handling.
    max_iterations : int, optional
        Maximum number of closure epochs.
    verbose : bool, optional
        If `True`, print progress information.

    Returns
    -------
    final_basis : List[List[pe.PauliString]]
        Closed Lie basis in user-facing PauliEngine format.
    adj_blocks : List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        Sparse commutator data recorded during closure construction.

    Notes
    -----
    This routine mirrors `lie_closure_cycles(...)` but additionally stores
    sparse commutator information in the format expected by downstream
    simulator routines.
    """
    space = NormalizedBasis()
    ps_cache = {}
    abstract_basis = []

    for gen in generators:
        if boundaries == 'periodic':
            vec = _to_canonical_sparse_vector(gen, n)
            for sig in vec:
                if sig not in ps_cache: ps_cache[sig] = pe.PauliString({k: _I2P[v] for k, v in sig}, 1.0)

            _, remainder = space.decompose(vec)
            if remainder:
                space.insert_remainder(remainder)
        else:
            abs_vec = _to_obc_abstract_vector(gen, n)
            dense_vec = _flatten_obc_vector(abs_vec, n)

            for (t, sig) in abs_vec:
                if sig not in ps_cache: ps_cache[sig] = pe.PauliString({k: _I2P[v] for k, v in sig}, 1.0)

            coeffs, remainder = space.decompose(dense_vec)
            if remainder:
                new_idx, norm_factor = space.insert_remainder(remainder)

                # Directly lift the perfectly normalized dense basis. No history attached!
                abs_rem = _lift_dense_to_abstract(space.matrix[new_idx], n)
                for (t, sig) in abs_rem:
                    if sig not in ps_cache: ps_cache[sig] = pe.PauliString({k: _I2P[v] for k, v in sig}, 1.0)
                abstract_basis.append(abs_rem)

    initial_length = len(space.matrix)
    if verbose: print(f"Preprocessing done. Initial basis size: {initial_length}", flush=True)

    adj_store = [{'u': [], 'v': [], 'f': []} for _ in range(initial_length)]
    epoch, old_len, new_len = 0, 0, initial_length

    while new_len > old_len and epoch < max_iterations:
        while len(adj_store) < new_len:
            adj_store.append({'u': [], 'v': [], 'f': []})

        current_basis = space.matrix

        for alpha_idx in range(old_len, new_len):
            for beta_idx in range(alpha_idx):
                vec_A, vec_B = current_basis[alpha_idx], current_basis[beta_idx]

                if boundaries == 'periodic':
                    commutator_vec = cycle_commutator(vec_A, vec_B, n, ps_cache)
                    coeffs, remainder = space.decompose(commutator_vec)
                    if remainder:
                        new_idx, norm_factor = space.insert_remainder(remainder)
                        coeffs[new_idx] = norm_factor.imag
                        while len(adj_store) < len(space.matrix):
                            adj_store.append({'u': [], 'v': [], 'f': []})
                else:
                    abs_A = abstract_basis[alpha_idx]
                    abs_B = abstract_basis[beta_idx]
                    abs_comm = obc_commutator(abs_A, abs_B, n, ps_cache)
                    dense_comm = _flatten_obc_vector(abs_comm, n)

                    coeffs, remainder = space.decompose(dense_comm)
                    if remainder:
                        new_idx, norm_factor = space.insert_remainder(remainder)
                        coeffs[new_idx] = norm_factor.imag

                        abs_rem = _lift_dense_to_abstract(space.matrix[new_idx], n)
                        for (t, sig) in abs_rem:
                            if sig not in ps_cache: ps_cache[sig] = pe.PauliString({k: _I2P[v] for k, v in sig}, 1.0)
                        abstract_basis.append(abs_rem)

                        while len(adj_store) < len(space.matrix):
                            adj_store.append({'u': [], 'v': [], 'f': []})

                for gamma_idx, f_val in coeffs.items():
                    if abs(f_val) > THRESHOLD:
                        adj_store[gamma_idx]['u'].append(beta_idx)
                        adj_store[gamma_idx]['v'].append(alpha_idx)
                        adj_store[gamma_idx]['f'].append(f_val)

        old_len = new_len
        new_len = len(space.matrix)
        epoch += 1
        if verbose: print(f"Epoch {epoch} complete. Basis size: {new_len}", flush=True)

    if epoch == max_iterations:
        warnings.warn(f"Reached max iterations {max_iterations} in pauli_cycle_preprocessing", UserWarning)

    final_basis = _reconstruct_output_basis(space)

    adj_blocks = []
    for store in adj_store:
        adj_blocks.append((
            np.array(store['u'], dtype=int),
            np.array(store['v'], dtype=int),
            np.array(store['f'], dtype=float)
        ))

    return final_basis, adj_blocks