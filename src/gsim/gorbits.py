"""
Utilities for Pauli-orbit Lie preprocessing in permutation-invariant systems.

This module provides preprocessing routines for Lie algebras expressed in the
Pauli-orbit basis on `n` qubits. The basis elements are labeled by triples
`(p, q, r)`, counting the number of `X`, `Y`, and `Z` factors in an orbit,
with the remaining `n - p - q - r` sites carrying the identity.

Two related, but conceptually distinct, preprocessing tasks are implemented:

1. Full structure constants
   `structure_constants_orbits(...)` computes the complete sparse commutator
   data for a user-supplied Lie basis expressed in the orbit basis.

2. Targeted expectation-space generators
   `adjoint_orbit_generators_targeted(...)` computes the actual sparse matrices
   that act on expectation-value vectors for a selected set of target
   generators. This is typically much cheaper than computing the full
   structure-constant tensor when only a few circuit generators are needed.

Representation conventions
--------------------------
- Orbit basis elements are labeled by triples `(p, q, r)` with
  `0 < p + q + r <= n`.
- The helper `build_orbit_maps(...)` provides a reversible mapping between
  orbit triples and dense linear indices.
- Input Lie bases are represented as lists of sparse dictionaries
  `{(p, q, r): coeff}`, allowing both pure orbit generators and linear
  combinations thereof.

Mathematical conventions
------------------------
The routines in this file work with the same commutator convention used
throughout the codebase: raw structure constants are stored first, and
expectation-space evolution matrices are constructed separately from them.

In particular:
- `structure_constants_orbits(...)` returns grouped raw commutator data.
- `adjoint_orbit_generators_targeted(...)` returns or reconstructs the
  matrices `M_mu` acting directly on expectation vectors `e`, i.e.
      d/dtheta e = M_mu e.

Notes
-----
- The heavy numerical loops are implemented in Numba.
- Factorials and orbit sizes are precomputed to avoid repeated combinatorial
  overhead.
- This module is intended as a low-level preprocessing backend for
  permutation-invariant g-sim workflows.
"""

import numpy as np
from numba import njit
from typing import Dict, Tuple, List
from math import factorial

# -- utils --
def _precompute_factorials(n_max):
    """
    Precompute factorials from 0 to `n_max`.

    Parameters
    ----------
    n_max : int
        Largest factorial argument required.

    Returns
    -------
    np.ndarray
        Array `fact_cache` with `fact_cache[k] = k!` as floating-point values.

    Notes
    -----
    The values are stored as `float64` because they are used inside Numba-based
    arithmetic kernels.
    """
    return np.array([factorial(i) for i in range(n_max + 1)], dtype=np.float64)


def _precompute_orbit_sizes(n: int, idx_to_tuple: np.ndarray, fact_cache: np.ndarray) -> np.ndarray:
    """
    Precompute the size of each Pauli orbit on `n` qubits.

    Parameters
    ----------
    n : int
        Number of qubits.
    idx_to_tuple : np.ndarray
        Array mapping dense orbit indices to triples `(p, q, r)`.
    fact_cache : np.ndarray
        Precomputed factorials.

    Returns
    -------
    np.ndarray
        Array of orbit sizes, where entry `idx` equals the number of Pauli
        strings in the orbit labeled by `idx_to_tuple[idx]`.

    Notes
    -----
    For an orbit `(p, q, r)`, the size is
        n! / (p! q! r! (n-p-q-r)!).
    """
    sizes = np.zeros(len(idx_to_tuple), dtype=np.float64)
    fn = fact_cache[n]
    for idx, (p, q, r) in enumerate(idx_to_tuple):
        s = n - p - q - r
        sizes[idx] = fn / (fact_cache[p] * fact_cache[q] * fact_cache[r] * fact_cache[s])
    return sizes

@njit(fastmath=True)
def orbit_commutator(n: int,
                      p1: int, q1: int, r1: int,
                      p2: int, q2: int, r2: int,
                      fact_cache: np.ndarray):
    """
    Compute the commutator of two pure Pauli orbits.

    Parameters
    ----------
    n : int
        Number of qubits.
    p1, q1, r1 : int
        Orbit counts for the first operand.
    p2, q2, r2 : int
        Orbit counts for the second operand.
    fact_cache : np.ndarray
        Precomputed factorials.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Parallel arrays `(keys_p, keys_q, keys_r, vals)` describing the sparse
        commutator output in orbit coordinates.

    Notes
    -----
    The output is not grouped as a Python dictionary in order to avoid object
    overhead inside the Numba kernel. The entries correspond to orbit labels
    `(tp, tq, tr)` with associated raw combinatorial coefficients.

    This routine is the core low-level commutator engine used by both the full
    and targeted preprocessing workflows.
    """
    s1 = n - p1 - q1 - r1
    s2 = n - p2 - q2 - r2

    # Heuristic size estimation for output buffer
    max_terms = 5000
    keys_p = np.zeros(max_terms, dtype=np.int32)
    keys_q = np.zeros(max_terms, dtype=np.int32)
    keys_r = np.zeros(max_terms, dtype=np.int32)
    vals = np.zeros(max_terms, dtype=np.float64)
    count = 0

    fn = fact_cache[n]

    # 1. Row 1 (X)
    for nXX in range(max(0, p1 - (n - p2)), min(p1, p2) + 1):
        rem_r1 = p1 - nXX

        for nXY in range(max(0, rem_r1 - (n - q2)), min(rem_r1, q2) + 1):
            rem_r1_2 = rem_r1 - nXY

            for nXZ in range(max(0, rem_r1_2 - (n - r2)), min(rem_r1_2, r2) + 1):
                nXI = p1 - nXX - nXY - nXZ
                if nXI > s2: continue

                rem_p2 = p2 - nXX
                rem_q2 = q2 - nXY
                rem_r2 = r2 - nXZ
                rem_s2 = s2 - nXI

                # 2. Row 2 (Y)
                for nYX in range(max(0, q1 - (rem_q2 + rem_r2 + rem_s2)), min(q1, rem_p2) + 1):
                    rem_r2_1 = q1 - nYX
                    for nYY in range(max(0, rem_r2_1 - (rem_r2 + rem_s2)), min(rem_r2_1, rem_q2) + 1):
                        rem_r2_2 = rem_r2_1 - nYY
                        for nYZ in range(max(0, rem_r2_2 - rem_s2), min(rem_r2_2, rem_r2) + 1):
                            nYI = q1 - nYX - nYY - nYZ
                            if nYI > rem_s2: continue

                            rem_p2_2 = rem_p2 - nYX
                            rem_q2_2 = rem_q2 - nYY
                            rem_r2_2 = rem_r2 - nYZ
                            rem_s2_2 = rem_s2 - nYI

                            # 3. Row 3 (Z)
                            for nZX in range(max(0, r1 - (rem_q2_2 + rem_r2_2 + rem_s2_2)), min(r1, rem_p2_2) + 1):
                                rem_r3_1 = r1 - nZX
                                for nZY in range(max(0, rem_r3_1 - (rem_r2_2 + rem_s2_2)), min(rem_r3_1, rem_q2_2) + 1):
                                    rem_r3_2 = rem_r3_1 - nZY

                                    min_zz = max(0, rem_r3_2 - rem_s2_2)
                                    max_zz = min(rem_r3_2, rem_r2_2)

                                    for nZZ in range(min_zz, max_zz + 1):
                                        # Deterministic remainder
                                        nZI = rem_r3_2 - nZZ
                                        nIX = rem_p2_2 - nZX
                                        nIY = rem_q2_2 - nZY
                                        nIZ = rem_r2_2 - nZZ
                                        nII = rem_s2_2 - nZI

                                        # Parity Check (Fast bitwise)
                                        D = nXY + nXZ + nYX + nYZ + nZX + nZY
                                        if D % 2 == 0: continue

                                        # Sign
                                        E = nYX + nZY + nXZ
                                        exponent = E + (D + 1) // 2
                                        sign = 1.0 if (exponent % 2 != 0) else -1.0

                                        # Orbit Label
                                        tp = nXI + nIX + nYZ + nZY
                                        tq = nYI + nIY + nZX + nXZ
                                        tr = nZI + nIZ + nXY + nYX

                                        denom = (fact_cache[nXX] * fact_cache[nXY] * fact_cache[nXZ] * fact_cache[nXI] *
                                                 fact_cache[nYX] * fact_cache[nYY] * fact_cache[nYZ] * fact_cache[nYI] *
                                                 fact_cache[nZX] * fact_cache[nZY] * fact_cache[nZZ] * fact_cache[nZI] *
                                                 fact_cache[nIX] * fact_cache[nIY] * fact_cache[nIZ] * fact_cache[nII])

                                        weight = sign * (fn / denom)

                                        found = False
                                        for k in range(count):
                                            if keys_p[k] == tp and keys_q[k] == tq and keys_r[k] == tr:
                                                vals[k] += weight
                                                found = True
                                                break

                                        if not found:
                                            if count < max_terms:
                                                keys_p[count] = tp
                                                keys_q[count] = tq
                                                keys_r[count] = tr
                                                vals[count] = weight
                                                count += 1

    return keys_p[:count], keys_q[:count], keys_r[:count], vals[:count]


def build_orbit_maps(n: int):
    """
    Construct dense index maps for Pauli orbits on `n` qubits.

    Parameters
    ----------
    n : int
        Number of qubits.

    Returns
    -------
    tuple_to_idx : np.ndarray
        Three-dimensional array such that `tuple_to_idx[p, q, r]` gives the
        dense index of the orbit `(p, q, r)`, or `-1` if invalid.
    idx_to_tuple : np.ndarray
        Array such that `idx_to_tuple[idx] = (p, q, r)`.

    Notes
    -----
    Only nontrivial orbits with `0 < p + q + r <= n` are enumerated.
    """

    tuple_to_idx = np.full((n + 1, n + 1, n + 1), -1, dtype=np.int32)
    idx_to_tuple = []

    counter = 0
    # Enumerate all valid orbits s.t. p+q+r <= n
    for s_val in range(1, n + 1):
        for p in range(s_val + 1):
            for q in range(s_val - p + 1):
                r = s_val - p - q
                tuple_to_idx[p, q, r] = counter
                idx_to_tuple.append((p, q, r))
                counter += 1

    return tuple_to_idx, np.array(idx_to_tuple, dtype=np.int32)


def basis_to_dense(dla_basis: List[Dict[Tuple[int, int, int], int]],
                   tuple_to_idx: np.ndarray):
    """
    Convert a sparse orbit-basis Lie basis into a dense matrix form.

    Parameters
    ----------
    dla_basis : List[Dict[Tuple[int, int, int], int]]
        Lie basis expressed as sparse dictionaries over orbit triples.
    tuple_to_idx : np.ndarray
        Dense lookup mapping `(p, q, r)` to orbit indices.

    Returns
    -------
    basis_matrix : np.ndarray
        Dense array of shape `(dim, total_orbits)` containing the basis vectors
        in orbit coordinates.
    pivots : np.ndarray
        Pivot index for each basis vector, chosen as the first occupied orbit
        under sorted key order.

    Notes
    -----
    This dense representation is used inside the Numba kernels for fast
    elimination and projection.
    """
    dim = len(dla_basis)
    total_orbits = np.max(tuple_to_idx) + 1

    basis_matrix = np.zeros((dim, total_orbits), dtype=np.float64)
    pivots = np.zeros(dim, dtype=np.int32)

    for i, vec in enumerate(dla_basis):
        first = True
        sorted_keys = sorted(vec.keys())

        for key in sorted_keys:
            p, q, r = key
            val = vec[key]
            idx = tuple_to_idx[p, q, r]
            basis_matrix[i, idx] = float(val)

            if first:
                pivots[i] = idx
                first = False

    return basis_matrix, pivots

@njit(fastmath=True)
def structure_constants_numba(n: int,
                               basis_matrix: np.ndarray,
                               pivots: np.ndarray,
                               idx_to_tuple: np.ndarray,
                               tuple_to_idx: np.ndarray,
                               fact_cache: np.ndarray):
    """
    Compute full sparse commutator data for an orbit-basis Lie algebra.

    Parameters
    ----------
    n : int
        Number of qubits.
    basis_matrix : np.ndarray
        Dense orbit-coordinate basis matrix.
    pivots : np.ndarray
        Pivot index of each basis vector.
    idx_to_tuple : np.ndarray
        Map from dense orbit index to `(p, q, r)`.
    tuple_to_idx : np.ndarray
        Map from `(p, q, r)` to dense orbit index.
    fact_cache : np.ndarray
        Precomputed factorials.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Parallel arrays `(out_u, out_v, out_g, out_f)` representing the sparse
        raw commutator data.

    Notes
    -----
    The output encodes grouped structure-constant information for later
    reconstruction in Python. This is the full preprocessing route and should
    be used when the entire algebra tensor is actually needed.
    """
    dim = basis_matrix.shape[0]
    total_orbits = basis_matrix.shape[1]

    max_entries = dim * dim * 5  # Pre-allocate output buffers (estimate size)

    out_u = np.zeros(max_entries, dtype=np.int32)  # beta
    out_v = np.zeros(max_entries, dtype=np.int32)  # alpha
    out_g = np.zeros(max_entries, dtype=np.int32)
    out_f = np.zeros(max_entries, dtype=np.float64)  # value
    out_ptr = 0

    comm_vec = np.zeros(total_orbits, dtype=np.float64)

    for alpha in range(dim):
        for beta in range(alpha):

            comm_vec[:] = 0.0  # Clear buffer

            row_A = basis_matrix[alpha]
            row_B = basis_matrix[beta]

            for idx_A in range(pivots[alpha], total_orbits):
                val_A = row_A[idx_A]
                if val_A == 0: continue

                pa, qa, ra = idx_to_tuple[idx_A]

                for idx_B in range(pivots[beta], total_orbits):
                    val_B = row_B[idx_B]
                    if val_B == 0: continue

                    pb, qb, rb = idx_to_tuple[idx_B]
                    kp, kq, kr, kv = orbit_commutator(n, pa, qa, ra, pb, qb, rb, fact_cache)
                    coeff_prod = val_A * val_B

                    for k in range(len(kv)):

                        tp, tq, tr = kp[k], kq[k], kr[k]
                        t_idx = tuple_to_idx[tp, tq, tr]

                        ts = n - tp - tq - tr
                        if ts < 0: continue  # should not happen

                        norm = fact_cache[n] / (fact_cache[tp] * fact_cache[tq] * fact_cache[tr] * fact_cache[ts])

                        term_val = coeff_prod * (2.0 * kv[k]) / norm
                        comm_vec[t_idx] += term_val

            for gamma in range(dim):
                piv = pivots[gamma]

                scalar = comm_vec[piv]

                if abs(scalar) > 1e-12:
                    if out_ptr < max_entries:
                        out_u[out_ptr] = beta
                        out_v[out_ptr] = alpha
                        out_g[out_ptr] = gamma
                        out_f[out_ptr] = scalar
                        out_ptr += 1

                    row_G = basis_matrix[gamma]

                    for k in range(piv, total_orbits):
                        comm_vec[k] -= scalar * row_G[k]

    return out_u[:out_ptr], out_v[:out_ptr], out_g[:out_ptr], out_f[:out_ptr]


def structure_constants_orbits(dla_basis: List[Dict],
                               n: int,
                               fact_cache: List[int] = None) -> List[
    Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute the full sparse structure constants for an orbit-basis Lie algebra.

    Parameters
    ----------
    dla_basis : List[Dict]
        Lie basis expressed as sparse dictionaries over orbit triples.
    n : int
        Number of qubits.
    fact_cache : np.ndarray, optional
        Precomputed factorials. If omitted, they are generated internally.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        Grouped sparse commutator data. Entry `adj_blocks[gamma]` contains
        arrays `(u, v, f)` associated with basis element `gamma`.

    Notes
    -----
    This routine computes the full commutator tensor, grouped by output basis
    element. It matches the same raw-structure-constant convention used by the
    other primitive modules.
    """
    tuple_to_idx, idx_to_tuple = build_orbit_maps(n)

    basis_mat, pivots = basis_to_dense(dla_basis, tuple_to_idx)

    if fact_cache is None:
        fact_cache = _precompute_factorials(n)

    u, v, g, f = structure_constants_numba(n, basis_mat, pivots, idx_to_tuple, tuple_to_idx, fact_cache)

    dim = len(dla_basis)

    sort_idx = np.argsort(g, kind='stable')
    g_sorted = g[sort_idx]
    u_sorted = u[sort_idx]
    v_sorted = v[sort_idx]
    f_sorted = f[sort_idx]

    unique_g, split_indices = np.unique(g_sorted, return_index=True)
    split_indices = split_indices[1:]

    grouped_u = np.split(u_sorted, split_indices)
    grouped_v = np.split(v_sorted, split_indices)
    grouped_f = np.split(f_sorted, split_indices)

    adj_blocks = [(np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float))
                  for _ in range(dim)]

    for gamma_idx, u_chunk, v_chunk, f_chunk in zip(unique_g, grouped_u, grouped_v, grouped_f):
        adj_blocks[gamma_idx] = (u_chunk, v_chunk, f_chunk)

    return adj_blocks


@njit(fastmath=True)
def adjoint_generators_targeted_numba(n: int,
                                      basis_matrix: np.ndarray,
                                      pivots: np.ndarray,
                                      pivot_coeffs: np.ndarray,
                                      idx_to_tuple: np.ndarray,
                                      tuple_to_idx: np.ndarray,
                                      target_orbits: np.ndarray,
                                      fact_cache: np.ndarray,
                                      orbit_sizes: np.ndarray):
    """
    Compute selected expectation-space generator matrices in Numba.

    Parameters
    ----------
    n : int
        Number of qubits.
    basis_matrix : np.ndarray
        Dense orbit-coordinate basis matrix.
    pivots : np.ndarray
        Pivot indices of the basis vectors.
    pivot_coeffs : np.ndarray
        Pivot coefficients of the basis vectors.
    idx_to_tuple : np.ndarray
        Map from dense orbit index to `(p, q, r)`.
    tuple_to_idx : np.ndarray
        Map from `(p, q, r)` to dense orbit index.
    target_orbits : np.ndarray
        Array of selected pure orbit generators to be targeted.
    fact_cache : np.ndarray
        Precomputed factorials.
    orbit_sizes : np.ndarray
        Precomputed sizes of all orbit basis elements.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Parallel arrays `(out_t, out_r, out_c, out_v)` describing the sparse
        entries of the targeted expectation-space matrices.

    Notes
    -----
    The returned entries correspond directly to the matrices `M_mu` acting on
    expectation vectors, not to raw structure-constant slices.
    """
    dim = basis_matrix.shape[0]
    total_orbits = basis_matrix.shape[1]
    num_targets = target_orbits.shape[0]

    max_entries = max(num_targets * dim * 8, 1)
    out_t = np.zeros(max_entries, dtype=np.int32)
    out_r = np.zeros(max_entries, dtype=np.int32)
    out_c = np.zeros(max_entries, dtype=np.int32)
    out_v = np.zeros(max_entries, dtype=np.float64)
    out_ptr = 0

    comm_vec = np.zeros(total_orbits, dtype=np.float64)

    for t in range(num_targets):
        pm, qm, rm = target_orbits[t]
        size_mu = orbit_sizes[tuple_to_idx[pm, qm, rm]]

        for beta in range(dim):
            comm_vec[:] = 0.0
            row_B = basis_matrix[beta]

            for idx_B in range(pivots[beta], total_orbits):
                val_B = row_B[idx_B]
                if val_B == 0.0:
                    continue

                pb, qb, rb = idx_to_tuple[idx_B]
                size_B = orbit_sizes[idx_B]

                kp, kq, kr, kv = orbit_commutator(n, pm, qm, rm, pb, qb, rb, fact_cache)
                prefactor = (2.0 * val_B) / (size_mu * size_B)

                for k in range(len(kv)):
                    tp, tq, tr = kp[k], kq[k], kr[k]
                    t_idx = tuple_to_idx[tp, tq, tr]
                    comm_vec[t_idx] += prefactor * kv[k]

            for gamma in range(dim):
                piv = pivots[gamma]
                piv_coeff = pivot_coeffs[gamma]
                if piv_coeff == 0.0:
                    continue

                scalar = comm_vec[piv] / piv_coeff
                if abs(scalar) > 1e-12:
                    if out_ptr >= max_entries:
                        new_max = max_entries * 2

                        new_t = np.zeros(new_max, dtype=np.int32)
                        new_r = np.zeros(new_max, dtype=np.int32)
                        new_c = np.zeros(new_max, dtype=np.int32)
                        new_v = np.zeros(new_max, dtype=np.float64)
                        new_t[:max_entries] = out_t
                        new_r[:max_entries] = out_r
                        new_c[:max_entries] = out_c
                        new_v[:max_entries] = out_v
                        out_t = new_t
                        out_r = new_r
                        out_c = new_c
                        out_v = new_v
                        max_entries = new_max

                    out_t[out_ptr] = t
                    out_r[out_ptr] = beta
                    out_c[out_ptr] = gamma
                    out_v[out_ptr] = -scalar
                    out_ptr += 1

                    row_G = basis_matrix[gamma]
                    for k in range(piv, total_orbits):
                        comm_vec[k] -= scalar * row_G[k]

    return out_t[:out_ptr], out_r[:out_ptr], out_c[:out_ptr], out_v[:out_ptr]


def adjoint_orbit_generators_targeted(dla_basis: List[Dict[Tuple[int, int, int], float]],
                                       target_indices: List[int],
                                       n: int,
                                       verbose: bool = False,
                                       fact_cache: np.ndarray = None,
                                       return_matrices: bool = True):
    """
    Compute expectation-space evolution generators for selected orbit-basis targets.

    Parameters
    ----------
    dla_basis : List[Dict[Tuple[int, int, int], float]]
        Lie basis expressed as sparse dictionaries over orbit triples.
    target_indices : List[int]
        Indices of the target basis elements whose expectation-space generators
        should be computed.
    n : int
        Number of qubits.
    verbose : bool, optional
        If `True`, print progress information.
    fact_cache : np.ndarray, optional
        Precomputed factorials. If omitted, they are generated internally.
    return_matrices : bool, optional
        If `True`, return CSR matrices directly. Otherwise return raw sparse
        coordinate triples.

    Returns
    -------
    list
        Either
        - a list of `(rows, cols, vals)` triples if `return_matrices=False`, or
        - a list of `scipy.sparse.csr_matrix` objects if `return_matrices=True`.

    Raises
    ------
    ValueError
        If a selected target basis element is not a single pure orbit with
        coefficient exactly `1`.

    Notes
    -----
    This routine computes the actual expectation-space evolution matrices
    associated with the selected generators. It is therefore the appropriate
    large-scale preprocessing routine when only a small number of circuit
    generators is needed.

    In contrast, `structure_constants_orbits(...)` computes the full grouped raw
    commutator tensor for the entire algebra.
    """

    tuple_to_idx, idx_to_tuple = build_orbit_maps(n)
    basis_mat, pivots = basis_to_dense(dla_basis, tuple_to_idx)
    pivot_coeffs = basis_mat[np.arange(len(dla_basis)), pivots]

    if fact_cache is None:
        fact_cache = _precompute_factorials(n)
    orbit_sizes = _precompute_orbit_sizes(n, idx_to_tuple, fact_cache)

    target_orbits_list = []
    for global_idx in target_indices:
        vec = dla_basis[global_idx]
        if len(vec) != 1:
            raise ValueError(
                "Targeted generator matrices are exact only when each target basis element "
                "is a single averaged orbit basis element."
            )
        pivot_key = next(iter(vec.keys()))
        pivot_val = float(vec[pivot_key])
        if abs(pivot_val - 1.0) > 1e-12:
            raise ValueError(
                "Targeted generator matrices expect each target to be normalized as a "
                "single averaged orbit with coefficient 1."
            )
        target_orbits_list.append(pivot_key)

    target_orbits = np.array(target_orbits_list, dtype=np.int32)

    if verbose:
        print(f"Accelerated Targeted Generator Matrices: {len(target_indices)} targets, Basis Size {len(dla_basis)}")

    t_idx, rows, cols, vals = adjoint_generators_targeted_numba(
        n, basis_mat, pivots, pivot_coeffs, idx_to_tuple, tuple_to_idx,
        target_orbits, fact_cache, orbit_sizes
    )

    if not return_matrices:

        raw_blocks = []
        for t in range(len(target_indices)):
            mask = (t_idx == t)
            raw_blocks.append((rows[mask], cols[mask], vals[mask]))

        return raw_blocks

    else:
        import scipy.sparse as sp

        mats = []
        dim = len(dla_basis)
        for t in range(len(target_indices)):
            mask = (t_idx == t)
            mats.append(sp.csr_matrix((vals[mask], (rows[mask], cols[mask])), shape=(dim, dim)))

        return mats
