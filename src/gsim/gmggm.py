"""
Utilities for the MGGM basis on fixed-Hamming-weight subspaces.

This module implements preprocessing routines for the MGGM basis used to
represent operators acting on the fixed-Hamming-weight subspace
    H_k subset (C^2)^{⊗ n},
whose dimension is
    d_k = binom(n, k).

The MGGM basis is split into three sectors:

- TYPE_A : antisymmetric off-diagonal generators A_ab
- TYPE_S : symmetric off-diagonal generators S_ab
- TYPE_P : diagonal/projector-like generators P_a

Two complementary workflows are provided.

1. Full preprocessing (`FastMGGMBasis`)
   Computes the complete sparse commutator data of the MGGM basis in grouped
   COO-like form. This is the analogue of the full structure-constant
   preprocessing used by the other primitive modules.

2. Targeted preprocessing (`FastMGGMBasisTargeted`)
   Avoids constructing the full tensor and instead builds only the sparse
   adjoint / expectation-space matrix associated with a user-supplied linear
   combination of TYPE_A generators. This is the efficient route used in
   experiments where all circuit generators lie in the antisymmetric sector.

Representation conventions
--------------------------
Basis elements are indexed by logical labels a,b in {0, ..., d_k-1}. The array

    basis_map_arr[type, a, b]

maps MGGM labels to dense basis indices. For off-diagonal sectors only the
canonical ordering a < b is stored.

Mathematical conventions
------------------------
- The full preprocessing stores raw grouped commutator data.
- The targeted preprocessing returns sparse matrices that act directly on
  coefficient / expectation vectors.

These are different objects and should not be confused.

Notes
-----
- The heavy combinatorial loops are implemented in Numba.
- The targeted implementation currently assumes support only on TYPE_A
  generators.
- This module is intended as a low-level backend for HW-preserving g-sim
  workflows.
"""

import numpy as np
import scipy.sparse as sp
from numba import njit
from scipy.special import binom
from typing import List, Tuple
import time

# -- globals --
TYPE_A = 0
TYPE_S = 1
TYPE_P = 2

# -- utils --

@njit
def _resolve_njit(t, x, y):
    """
    Resolve MGGM index symmetries in Numba.

    Parameters
    ----------
    t : int
        Sector label (`TYPE_A`, `TYPE_S`, or `TYPE_P`).
    x, y : int
        Logical basis indices.

    Returns
    -------
    Tuple[float, int, int, int]
        `(symmetry_factor, t_out, x_out, y_out)` after enforcing the canonical
        MGGM indexing convention.

    Notes
    -----
    This helper enforces:
    - `P_x` stays diagonal,
    - `S_xy = S_yx`,
    - `A_yx = -A_xy`,
    - `A_xx = 0`,
    - `S_xx = 2 P_x`.
    """
    if t == TYPE_P:
        return 1.0, TYPE_P, x, x
    if x == y:
        if t == TYPE_S:
            return 2.0, TYPE_P, x, x
        else:  # A_xx = 0
            return 0.0, -1, -1, -1
    elif x > y:
        if t == TYPE_S:
            return 1.0, TYPE_S, y, x
        else:  # A_yx = -A_xy
            return -1.0, TYPE_A, y, x
    else:
        return 1.0, t, x, y

# -- main --

@njit
def compute_structure_constants_njit(dk, a_idx, b_idx, basis_map, threshold):
    """
    Compute the full sparse MGGM commutator data in Numba.

    Parameters
    ----------
    dk : int
        Dimension of the fixed-HW sector, i.e. `binom(n, k)`.
    a_idx, b_idx : np.ndarray
        Parallel arrays enumerating all off-diagonal index pairs `a < b`.
    basis_map : np.ndarray
        Dense lookup table mapping MGGM labels `(type, a, b)` to basis indices.
    threshold : float
        Numerical cutoff below which coefficients are discarded.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Parallel arrays `(lam_out, u_out, v_out, f_out)` encoding the sparse
        grouped commutator data.

    Notes
    -----
    The output is a raw sparse commutator representation. It is not yet a list
    of sparse matrices and should be interpreted as grouped structure-constant
    data.
    """
    N_off = len(a_idx)

    # Pre-allocate large flat buffers
    max_terms = 4 * dk**3
    lam_out = np.empty(max_terms, dtype=np.int32)
    u_out = np.empty(max_terms, dtype=np.int32)
    v_out = np.empty(max_terms, dtype=np.int32)
    f_out = np.empty(max_terms, dtype=np.float64)
    count = 0

    for i in range(N_off):
        a, b = a_idx[i], b_idx[i]
        mu_A = basis_map[TYPE_A, a, b]
        mu_S = basis_map[TYPE_S, a, b]

        # 1. Commutators with P
        for c in (a, b):
            nu_P = basis_map[TYPE_P, c, c]

            # [A_ab, P_c]
            coeff_A1, t_A1, x_A1, y_A1 = (1.0, TYPE_S, a, c) if b == c else (-1.0, TYPE_S, b, c)
            c_sym, t_final, x_f, y_f = _resolve_njit(t_A1, x_A1, y_A1)
            lam = basis_map[t_final, x_f, y_f]
            f_val = coeff_A1 * c_sym

            if abs(f_val) >= threshold:
                alpha, beta, f_final = (mu_A, nu_P, f_val) if mu_A > nu_P else (nu_P, mu_A, -f_val)
                lam_out[count] = lam;
                u_out[count] = beta;
                v_out[count] = alpha;
                f_out[count] = f_final
                count += 1

            # [S_ab, P_c]
            coeff_S1, t_S1, x_S1, y_S1 = (-1.0, TYPE_A, a, c) if b == c else (-1.0, TYPE_A, b, c)
            c_sym, t_final, x_f, y_f = _resolve_njit(t_S1, x_S1, y_S1)
            lam = basis_map[t_final, x_f, y_f]
            f_val = coeff_S1 * c_sym

            if abs(f_val) >= threshold:
                alpha, beta, f_final = (mu_S, nu_P, f_val) if mu_S > nu_P else (nu_P, mu_S, -f_val)
                lam_out[count] = lam;
                u_out[count] = beta;
                v_out[count] = alpha;
                f_out[count] = f_final
                count += 1

        # 2. [A_ab, S_ab] -> fully overlapping pair
        for (coeff, t_out, x_out, y_out) in [(1.0, TYPE_S, a, a), (-1.0, TYPE_S, b, b)]:
            c_sym, t_final, x_f, y_f = _resolve_njit(t_out, x_out, y_out)
            lam = basis_map[t_final, x_f, y_f]
            f_val = coeff * c_sym
            if abs(f_val) >= threshold:
                alpha, beta, f_final = (mu_A, mu_S, f_val) if mu_A > mu_S else (mu_S, mu_A, -f_val)
                lam_out[count] = lam;
                u_out[count] = beta;
                v_out[count] = alpha;
                f_out[count] = f_final
                count += 1

        # 3. Off-diagonal sharing exactly ONE index 'k'
        for k in range(dk):
            if k == a or k == b: continue

            for c, d in [(min(a, k), max(a, k)), (min(b, k), max(b, k))]:
                nu_S = basis_map[TYPE_S, c, d]
                nu_A = basis_map[TYPE_A, c, d]

                # [A_ab, S_cd]
                terms_AS = [
                    (1.0, TYPE_S, a, d) if b == c else (0.0, -1, 0, 0),
                    (1.0, TYPE_S, a, c) if b == d else (0.0, -1, 0, 0),
                    (-1.0, TYPE_S, b, d) if a == c else (0.0, -1, 0, 0),
                    (-1.0, TYPE_S, b, c) if a == d else (0.0, -1, 0, 0)
                ]
                for (coeff, t_out, x_out, y_out) in terms_AS:
                    if coeff == 0.0: continue
                    c_sym, t_final, x_f, y_f = _resolve_njit(t_out, x_out, y_out)
                    lam = basis_map[t_final, x_f, y_f]
                    f_val = coeff * c_sym
                    if abs(f_val) >= threshold:
                        alpha, beta, f_final = (mu_A, nu_S, f_val) if mu_A > nu_S else (nu_S, mu_A, -f_val)
                        lam_out[count] = lam;
                        u_out[count] = beta;
                        v_out[count] = alpha;
                        f_out[count] = f_final
                        count += 1

                # [A_ab, A_cd] (enforce mu > nu via nu_A > mu_A to prevent computing twice)
                if nu_A > mu_A:
                    terms_AA = [
                        (1.0, TYPE_A, a, d) if b == c else (0.0, -1, 0, 0),
                        (-1.0, TYPE_A, a, c) if b == d else (0.0, -1, 0, 0),
                        (-1.0, TYPE_A, b, d) if a == c else (0.0, -1, 0, 0),
                        (1.0, TYPE_A, b, c) if a == d else (0.0, -1, 0, 0)
                    ]
                    for (coeff, t_out, x_out, y_out) in terms_AA:
                        if coeff == 0.0: continue
                        c_sym, t_final, x_f, y_f = _resolve_njit(t_out, x_out, y_out)
                        if t_final == -1: continue
                        lam = basis_map[t_final, x_f, y_f]
                        f_val = coeff * c_sym
                        if abs(f_val) >= threshold:
                            alpha, beta, f_final = (nu_A, mu_A, f_val)  # already know nu_A > mu_A
                            lam_out[count] = lam;
                            u_out[count] = beta;
                            v_out[count] = alpha;
                            f_out[count] = f_final
                            count += 1

                # [S_ab, S_cd] (enforce mu > nu via nu_S > mu_S)
                if nu_S > mu_S:
                    terms_SS = [
                        (-1.0, TYPE_A, a, d) if b == c else (0.0, -1, 0, 0),
                        (-1.0, TYPE_A, a, c) if b == d else (0.0, -1, 0, 0),
                        (-1.0, TYPE_A, b, d) if a == c else (0.0, -1, 0, 0),
                        (-1.0, TYPE_A, b, c) if a == d else (0.0, -1, 0, 0)
                    ]
                    for (coeff, t_out, x_out, y_out) in terms_SS:
                        if coeff == 0.0: continue
                        c_sym, t_final, x_f, y_f = _resolve_njit(t_out, x_out, y_out)
                        if t_final == -1: continue
                        lam = basis_map[t_final, x_f, y_f]
                        f_val = coeff * c_sym
                        if abs(f_val) >= threshold:
                            alpha, beta, f_final = (nu_S, mu_S, f_val)  # already know nu_S > mu_S
                            lam_out[count] = lam;
                            u_out[count] = beta;
                            v_out[count] = alpha;
                            f_out[count] = f_final
                            count += 1

    return lam_out[:count], u_out[:count], v_out[:count], f_out[:count]

class FastMGGMBasis:
    """
    Full MGGM preprocessing backend.

    This class builds the complete MGGM basis map and computes the full grouped
    sparse commutator data for the fixed-HW sector `(n, k)`.

    Parameters
    ----------
    n : int
        Number of qubits.
    k : int
        Fixed Hamming weight.
    threshold : float, optional
        Numerical cutoff below which coefficients are discarded.

    Attributes
    ----------
    n, k : int
        Problem parameters.
    d_k : int
        Dimension of the fixed-HW sector, `binom(n, k)`.
    d : int
        Dimension of the full MGGM operator basis, `d_k**2`.
    basis_map_arr : np.ndarray
        Dense lookup table mapping `(type, a, b)` to basis indices.
    lam_arr, u_arr, v_arr, f_arr : np.ndarray
        Parallel arrays storing the raw grouped commutator data.
    time_sc : float
        Wall-clock time spent in the full structure-constant construction.
    """

    def __init__(self, n, k, threshold=1e-12):
        self.n = n
        self.k = k
        self.d_k = int(binom(n, k))
        self.d = self.d_k ** 2
        self.threshold = threshold

        self.basis_map_arr = np.full((3, self.d_k, self.d_k), -1, dtype=np.int32)
        self._build_basis_map()

        t0 = time.time()
        self._build_structure_constants()
        t1 = time.time()
        self.time_sc =t1-t0

    def _build_basis_map(self):
        """
        Construct the dense MGGM basis-index lookup table.

        Notes
        -----
        The basis is ordered by sectors:
        - all antisymmetric off-diagonal generators,
        - all symmetric off-diagonal generators,
        - all diagonal generators.
        Only canonical pairs `a < b` are stored explicitly for the off-diagonal
        sectors.
        """
        a_idx, b_idx = np.triu_indices(self.d_k, k=1)
        self.a_idx = np.array(a_idx, dtype=np.int32)
        self.b_idx = np.array(b_idx, dtype=np.int32)
        N_off = len(a_idx)

        for i, (a, b) in enumerate(zip(a_idx, b_idx)):
            self.basis_map_arr[TYPE_A, a, b] = i  # Map A
            self.basis_map_arr[TYPE_S, a, b] = N_off + i  # Map S

        for a in range(self.d_k):
            self.basis_map_arr[TYPE_P, a, a] = 2 * N_off + a  # Map P

    def _build_structure_constants(self):
        """
        Compute and store the full sparse grouped commutator data.

        Notes
        -----
        This method fills the arrays `lam_arr`, `u_arr`, `v_arr`, and `f_arr`
        using the Numba kernel `compute_structure_constants_njit(...)`.
        """
        self.lam_arr, self.u_arr, self.v_arr, self.f_arr = compute_structure_constants_njit(
            self.d_k, self.a_idx, self.b_idx, self.basis_map_arr, self.threshold
        )

    def get_adj_blocks(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Return the grouped sparse commutator data in block form.

        Returns
        -------
        List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
            List `adj_blocks` such that `adj_blocks[gamma] = (u, v, f)`, where
            `gamma` labels the output basis element and `(u, v, f)` are the
            sparse commutator entries associated with it.

        Notes
        -----
        This is the full-preprocessing analogue of the grouped sparse format
        used elsewhere in the codebase. It stores raw commutator data rather
        than ready-to-use evolution matrices.
        """
        adj_blocks = [(np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float)) for _ in
                      range(self.d)]

        if len(self.lam_arr) == 0:
            return adj_blocks

        # 2. Vectorized grouping using numpy (blazing fast)
        sort_idx = np.argsort(self.lam_arr)
        lam_s = self.lam_arr[sort_idx]
        u_s = self.u_arr[sort_idx]
        v_s = self.v_arr[sort_idx]
        f_s = self.f_arr[sort_idx]

        # Find the boundaries where the gamma index changes
        unique_lams, start_indices = np.unique(lam_s, return_index=True)
        end_indices = np.append(start_indices[1:], len(lam_s))

        # Slice the contiguous memory blocks directly
        for lam, start, end in zip(unique_lams, start_indices, end_indices):
            adj_blocks[lam] = (u_s[start:end], v_s[start:end], f_s[start:end])

        return adj_blocks

@njit
def _targeted_A_action_njit(dk, active_a, active_b, active_coeffs, basis_map_arr, threshold):
    """
    Build the sparse adjoint-action matrix for a TYPE_A-supported generator.

    Parameters
    ----------
    dk : int
        Dimension of the fixed-HW sector.
    active_a, active_b : np.ndarray
        Parallel arrays encoding the active antisymmetric generators `A_ab`
        with `a < b`.
    active_coeffs : np.ndarray
        Coefficients of the active generators.
    basis_map_arr : np.ndarray
        Dense MGGM basis-index lookup table.
    threshold : float
        Numerical cutoff below which coefficients are discarded.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Sparse matrix data `(rows, cols, vals)` representing the linear map
        induced by
            G = sum_t active_coeffs[t] * A_{active_a[t], active_b[t]}.

    Notes
    -----
    This kernel constructs the actual adjoint / coefficient-space action of
    `G`; it does not return grouped structure-constant slices.
    """
    n_active = len(active_a)
    max_terms = max(1, n_active * (8 * dk + 8))

    rows = np.empty(max_terms, dtype=np.int32)
    cols = np.empty(max_terms, dtype=np.int32)
    vals = np.empty(max_terms, dtype=np.float64)
    count = 0

    for t in range(n_active):
        a = active_a[t]
        b = active_b[t]
        c_mu = active_coeffs[t]

        if abs(c_mu) < threshold:
            continue

        mu_S = basis_map_arr[TYPE_S, a, b]

        # 1) [A_ab, P_a] = -S_ab, [A_ab, P_b] = +S_ab
        nu = basis_map_arr[TYPE_P, a, a]
        rows[count] = mu_S
        cols[count] = nu
        vals[count] = -c_mu
        count += 1

        nu = basis_map_arr[TYPE_P, b, b]
        rows[count] = mu_S
        cols[count] = nu
        vals[count] = +c_mu
        count += 1

        # 2) [A_ab, S_ab] = 2 P_a - 2 P_b
        nu = mu_S

        lam = basis_map_arr[TYPE_P, a, a]
        rows[count] = lam
        cols[count] = nu
        vals[count] = +2.0 * c_mu
        count += 1

        lam = basis_map_arr[TYPE_P, b, b]
        rows[count] = lam
        cols[count] = nu
        vals[count] = -2.0 * c_mu
        count += 1

        for k in range(dk):
            if k == a or k == b:
                continue

            # pair (a, k)
            c = a if a < k else k
            d = k if a < k else a
            nu_S = basis_map_arr[TYPE_S, c, d]
            nu_A = basis_map_arr[TYPE_A, c, d]

            # [A_ab, S_cd]
            terms_AS = (
                (1.0, TYPE_S, a, d) if b == c else (0.0, -1, 0, 0),
                (1.0, TYPE_S, a, c) if b == d else (0.0, -1, 0, 0),
                (-1.0, TYPE_S, b, d) if a == c else (0.0, -1, 0, 0),
                (-1.0, TYPE_S, b, c) if a == d else (0.0, -1, 0, 0),
            )
            for coeff, t_out, x_out, y_out in terms_AS:
                if coeff == 0.0:
                    continue
                c_sym, t_fin, x_fin, y_fin = _resolve_njit(t_out, x_out, y_out)
                if t_fin == -1:
                    continue
                lam = basis_map_arr[t_fin, x_fin, y_fin]
                f_val = coeff * c_sym * c_mu
                if abs(f_val) >= threshold:
                    rows[count] = lam
                    cols[count] = nu_S
                    vals[count] = f_val
                    count += 1

            # [A_ab, A_cd]
            terms_AA = (
                (1.0, TYPE_A, a, d) if b == c else (0.0, -1, 0, 0),
                (-1.0, TYPE_A, a, c) if b == d else (0.0, -1, 0, 0),
                (-1.0, TYPE_A, b, d) if a == c else (0.0, -1, 0, 0),
                (1.0, TYPE_A, b, c) if a == d else (0.0, -1, 0, 0),
            )
            for coeff, t_out, x_out, y_out in terms_AA:
                if coeff == 0.0:
                    continue
                c_sym, t_fin, x_fin, y_fin = _resolve_njit(t_out, x_out, y_out)
                if t_fin == -1:
                    continue
                lam = basis_map_arr[t_fin, x_fin, y_fin]
                f_val = coeff * c_sym * c_mu
                if abs(f_val) >= threshold:
                    rows[count] = lam
                    cols[count] = nu_A
                    vals[count] = f_val
                    count += 1

            # pair (b, k)
            c = b if b < k else k
            d = k if b < k else b
            nu_S = basis_map_arr[TYPE_S, c, d]
            nu_A = basis_map_arr[TYPE_A, c, d]

            terms_AS = (
                (1.0, TYPE_S, a, d) if b == c else (0.0, -1, 0, 0),
                (1.0, TYPE_S, a, c) if b == d else (0.0, -1, 0, 0),
                (-1.0, TYPE_S, b, d) if a == c else (0.0, -1, 0, 0),
                (-1.0, TYPE_S, b, c) if a == d else (0.0, -1, 0, 0),
            )
            for coeff, t_out, x_out, y_out in terms_AS:
                if coeff == 0.0:
                    continue
                c_sym, t_fin, x_fin, y_fin = _resolve_njit(t_out, x_out, y_out)
                if t_fin == -1:
                    continue
                lam = basis_map_arr[t_fin, x_fin, y_fin]
                f_val = coeff * c_sym * c_mu
                if abs(f_val) >= threshold:
                    rows[count] = lam
                    cols[count] = nu_S
                    vals[count] = f_val
                    count += 1

            terms_AA = (
                (1.0, TYPE_A, a, d) if b == c else (0.0, -1, 0, 0),
                (-1.0, TYPE_A, a, c) if b == d else (0.0, -1, 0, 0),
                (-1.0, TYPE_A, b, d) if a == c else (0.0, -1, 0, 0),
                (1.0, TYPE_A, b, c) if a == d else (0.0, -1, 0, 0),
            )
            for coeff, t_out, x_out, y_out in terms_AA:
                if coeff == 0.0:
                    continue
                c_sym, t_fin, x_fin, y_fin = _resolve_njit(t_out, x_out, y_out)
                if t_fin == -1:
                    continue
                lam = basis_map_arr[t_fin, x_fin, y_fin]
                f_val = coeff * c_sym * c_mu
                if abs(f_val) >= threshold:
                    rows[count] = lam
                    cols[count] = nu_A
                    vals[count] = f_val
                    count += 1

    return rows[:count], cols[:count], vals[:count]

class FastMGGMBasisTargeted:
    """
    Targeted MGGM backend for TYPE_A-supported generators.

    This class avoids constructing the full commutator tensor and instead
    builds only the sparse adjoint-action matrix associated with a supplied
    linear combination of antisymmetric MGGM generators.

    Parameters
    ----------
    n : int
        Number of qubits.
    k : int
        Fixed Hamming weight.
    threshold : float, optional
        Numerical cutoff below which coefficients are discarded.

    Attributes
    ----------
    n, k : int
        Problem parameters.
    d_k : int
        Dimension of the fixed-HW sector, `binom(n, k)`.
    d : int
        Dimension of the full MGGM operator basis, `d_k**2`.
    basis_map_arr : np.ndarray
        Dense lookup table mapping `(type, a, b)` to basis indices.
    A_to_pair : Dict[int, Tuple[int, int]]
        Reverse map from antisymmetric basis index to canonical pair `(a, b)`.
    """
    def __init__(self, n, k, threshold=1e-12):
        self.n = n
        self.k = k
        self.d_k = int(binom(n, k))
        self.d = self.d_k ** 2
        self.threshold = threshold

        self.basis_map_arr = np.full((3, self.d_k, self.d_k), -1, dtype=np.int32)
        self._build_basis_map()
        self._build_reverse_map()

    def _build_basis_map(self):
        """
        Construct the dense MGGM basis-index lookup table.

        Notes
        -----
        This uses the same sector ordering convention as `FastMGGMBasis`.
        """
        a_idx, b_idx = np.triu_indices(self.d_k, k=1)
        N_off = len(a_idx)

        for i, (a, b) in enumerate(zip(a_idx, b_idx)):
            self.basis_map_arr[TYPE_A, a, b] = i
            self.basis_map_arr[TYPE_S, a, b] = N_off + i

        for a in range(self.d_k):
            self.basis_map_arr[TYPE_P, a, a] = 2 * N_off + a

    def _build_reverse_map(self):
        """
        Build a reverse lookup for antisymmetric basis generators.

        Notes
        -----
        The map `A_to_pair` is used to translate sparse coefficient vectors in
        basis-index form into the `(a, b)` pair representation required by the
        targeted Numba kernel.
        """
        self.A_to_pair = {}
        a_idx, b_idx = np.triu_indices(self.d_k, k=1)
        for a, b in zip(a_idx, b_idx):
            mu = self.basis_map_arr[TYPE_A, a, b]
            self.A_to_pair[int(mu)] = (int(a), int(b))

    def build_targeted_adjoint_generator(self, vec):
        """
        Build the sparse adjoint-action generator for a TYPE_A-supported generator.

        Parameters
        ----------
        vec : Dict[int, float]
            Sparse coefficient dictionary describing
                G = sum_mu vec[mu] B_mu.
            Only support on TYPE_A generators is currently allowed.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse matrix representing the adjoint / coefficient-space action of
            `G` on the full MGGM basis.

        Raises
        ------
        ValueError
            If `vec` contains support outside the TYPE_A sector.

        Notes
        -----
        This is the intended fast path for HW-preserving amplitude-encoding
        experiments, where all circuit generators lie in the antisymmetric MGGM
        sector.
        """
        active_a = []
        active_b = []
        active_coeffs = []

        for mu, c_mu in vec.items():
            if abs(c_mu) < self.threshold:
                continue
            if mu not in self.A_to_pair:
                raise ValueError("Targeted builder currently supports TYPE_A-only vectors.")
            a, b = self.A_to_pair[mu]
            active_a.append(a)
            active_b.append(b)
            active_coeffs.append(float(c_mu))

        if not active_a:
            return sp.csr_matrix((self.d, self.d), dtype=np.float64)

        active_a = np.array(active_a, dtype=np.int32)
        active_b = np.array(active_b, dtype=np.int32)
        active_coeffs = np.array(active_coeffs, dtype=np.float64)

        rows, cols, vals = _targeted_A_action_njit(
            self.d_k, active_a, active_b, active_coeffs,
            self.basis_map_arr, self.threshold
        )

        return sp.coo_matrix((vals, (rows, cols)), shape=(self.d, self.d)).tocsr()