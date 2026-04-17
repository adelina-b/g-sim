"""
Helper routines for the permutation-equivariant QNN graph-state experiment.

This module contains the low-level utilities used to construct the input
expectation vector e_in for graph states in the Pauli-orbit basis, restricted
to disconnected graphs whose connected components are small enough to permit
exact stabilizer enumeration.

Main functionality
------------------
1. Generate random disconnected graphs as lists of connected-component
   adjacency matrices.
2. Compute the exact Pauli-orbit expectation data for each connected component.
3. Combine component-wise orbit data by convolution.
4. Assemble the full input expectation vector e_in in a user-supplied
   Pauli-orbit Lie basis.

The implementation is designed for the numerical experiment in the
permutation-equivariant QNN subsection of the paper, where graph states are
used as data inputs and evolved under permutation-invariant orbit generators.

Notes
-----
- The exact stabilizer enumeration scales exponentially in the component size,
  so `max_comp_size` should remain small.
- The returned `e_in` vector uses the same orbit-basis convention as the
  `gsim.gorbits` backend.
- This file is intended as experiment support code rather than a general
  graph-state library.
"""

import numpy as np
import networkx as nx
from numba import njit

import gsim

def generate_random_disconnected_components(
    n: int,
    max_comp_size: int = 10,
    p: float = 0.5,
    seed: int = 0,
) -> list[np.ndarray]:
    """
    Generate a disconnected graph on `n` vertices as a list of adjacency
    matrices for its connected components.

    Parameters
    ----------
    n : int
        Total number of vertices.
    max_comp_size : int, optional
        Maximum allowed size of any connected component.
    p : float, optional
        Edge probability used for Erdős-Rényi sampling of components of size
        at least 3.
    seed : int, optional
        Seed controlling the full random generation process.

    Returns
    -------
    list[np.ndarray]
        List of adjacency matrices, one per connected component.

    Notes
    -----
    For a fixed seed, this function is deterministic. Components of size 1
    and 2 are generated as isolated vertices and dimers, respectively.
    Larger components are sampled until connected.
    """
    rng = np.random.default_rng(seed)

    components_matrices: list[np.ndarray] = []
    nodes_left = n

    while nodes_left > 0:
        comp_size = int(rng.integers(1, min(nodes_left, max_comp_size) + 1))
        nodes_left -= comp_size

        if comp_size == 1:
            A = np.array([[0]], dtype=np.int32)

        elif comp_size == 2:
            A = np.array([[0, 1], [1, 0]], dtype=np.int32)

        else:
            while True:
                graph_seed = int(rng.integers(0, 2**32 - 1))
                G = nx.erdos_renyi_graph(comp_size, p, seed=graph_seed)
                if nx.is_connected(G):
                    break
            A = nx.to_numpy_array(G, dtype=np.int32)

        components_matrices.append(A)

    return components_matrices

def get_global_edge_list(components_adj_matrices: list) -> list:
    """
    Convert component-local adjacency matrices into a global edge list.

    Parameters
    ----------
    components_adj_matrices : list[np.ndarray]
        List of adjacency matrices, one for each connected component.

    Returns
    -------
    list[tuple[int, int]]
        List of edges `(i, j)` in global vertex indexing, obtained by placing
        the components consecutively along the full vertex set.

    Notes
    -----
    The function assumes that the components are ordered and indexed
    consecutively. For each component, only the upper-triangular part of the
    adjacency matrix is read, so each undirected edge appears exactly once.
    """
    edges = []
    current_node_offset = 0

    for A in components_adj_matrices:
        n_c = A.shape[0]
        rows, cols = np.where(np.triu(A, 1) == 1)

        for i, j in zip(rows, cols):
            global_i = int(i + current_node_offset)
            global_j = int(j + current_node_offset)
            edges.append((global_i, global_j))
        current_node_offset += n_c

    return edges

@njit
def _exact_component_orbits(A_comp):
    """
    Compute exact Pauli-orbit expectation data for a connected graph component.

    Parameters
    ----------
    A_comp : np.ndarray
        Adjacency matrix of a connected component.

    Returns
    -------
    np.ndarray
        Array `counts[p, q, r]` containing the exact signed stabilizer counts
        contributing to the Pauli orbit labeled by `(p, q, r)`.

    Notes
    -----
    This routine enumerates all stabilizer elements of the graph state on the
    component and groups them by orbit type `(n_X, n_Y, n_Z)`.
    """
    n_c = A_comp.shape[0]
    counts = np.zeros((n_c + 1, n_c + 1, n_c + 1), dtype=np.float64)
    total = 1 << n_c

    for k in range(total):
        U = np.zeros(n_c, dtype=np.int32)
        temp = k
        for i in range(n_c):
            U[i] = temp % 2
            temp //= 2

        V_Z = np.zeros(n_c, dtype=np.int32)
        for i in range(n_c):
            for j in range(n_c):
                V_Z[i] ^= (A_comp[i, j] & U[j])

        n_X = n_Y = n_Z = 0
        for i in range(n_c):
            if U[i] == 1 and V_Z[i] == 0:
                n_X += 1
            elif U[i] == 1 and V_Z[i] == 1:
                n_Y += 1
            elif U[i] == 0 and V_Z[i] == 1:
                n_Z += 1

        E_U = 0
        for i in range(n_c):
            if U[i] == 1:
                for j in range(i + 1, n_c):
                    if U[j] == 1 and A_comp[i, j] == 1:
                        E_U += 1

        sign = 1.0 if (E_U + n_Y // 2) % 2 == 0 else -1.0
        counts[n_X, n_Y, n_Z] += sign

    return counts


@njit
def _convolve_distributions(dp_current, comp_counts, n_current, n_comp, n_total):
    """
    Convolve accumulated orbit counts with those of a new connected component.

    Parameters
    ----------
    dp_current : np.ndarray
        Current accumulated orbit-count tensor.
    comp_counts : np.ndarray
        Orbit-count tensor for one connected component.
    n_current : int
        Number of qubits represented by `dp_current`.
    n_comp : int
        Size of the new component.
    n_total : int
        Total system size.

    Returns
    -------
    np.ndarray
        Updated orbit-count tensor after adding the new component.

    Notes
    -----
    Because disconnected graph states factorize over connected components,
    their stabilizer-orbit data combine by convolution.
    """
    new_dp = np.zeros((n_total + 1, n_total + 1, n_total + 1), dtype=np.float64)

    for p1 in range(n_current + 1):
        for q1 in range(n_current + 1 - p1):
            for r1 in range(n_current + 1 - p1 - q1):
                val1 = dp_current[p1, q1, r1]
                if val1 == 0:
                    continue

                for p2 in range(n_comp + 1):
                    for q2 in range(n_comp + 1 - p2):
                        for r2 in range(n_comp + 1 - p2 - q2):
                            val2 = comp_counts[p2, q2, r2]
                            if val2 == 0:
                                continue

                            new_dp[p1 + p2, q1 + q2, r1 + r2] += val1 * val2

    return new_dp

def _orbit_size(n: int, p: int, q: int, r: int, fact_cache:np.ndarray = None) -> int:
    """
    Compute the size of the Pauli orbit labeled by `(p, q, r)`.

    Parameters
    ----------
    n : int
        Total number of qubits.
    p, q, r : int
        Numbers of X, Y, and Z letters in the orbit label.
    fact_cache : np.ndarray, optional
        Precomputed factorials.

    Returns
    -------
    int
        Number of distinct Pauli strings in the orbit.

    Notes
    -----
    The orbit size is
        n! / (p! q! r! (n-p-q-r)!).
    """
    if fact_cache is None:
        fact_cache = gsim.gorbits._precompute_factorials(n)
    fn = fact_cache[n]
    return fn / (fact_cache[p] * fact_cache[q] * fact_cache[r] * fact_cache[n-p-q-r])


def generate_ein_disconnected_graph(n: int, components_adj_matrices: list, g_basis: list[dict]) -> np.ndarray:
    """
    Construct the input expectation vector e_in for a disconnected graph state.

    Parameters
    ----------
    n : int
        Total number of qubits.
    components_adj_matrices : list[np.ndarray]
        Adjacency matrices of the connected components.
    g_basis : list[dict]
        Lie basis expressed in the Pauli-orbit representation.

    Returns
    -------
    np.ndarray
        Expectation-value vector e_in in the supplied orbit basis.

    Notes
    -----
    This routine computes exact orbit expectations component-wise, combines
    them by convolution, divides by orbit sizes to obtain normalized orbit
    expectation values, and finally evaluates the supplied basis vectors
    against those orbit expectations.
    """
    dp = np.zeros((n + 1, n + 1, n + 1), dtype=np.float64)
    dp[0, 0, 0] = 1.0

    n_current = 0
    for A_comp in components_adj_matrices:
        n_comp = A_comp.shape[0]
        comp_counts = _exact_component_orbits(A_comp)
        dp = _convolve_distributions(dp, comp_counts, n_current, n_comp, n)
        n_current += n_comp

    orbit_expectations = np.zeros((n + 1, n + 1, n + 1), dtype=np.float64)
    for p in range(n + 1):
        for q in range(n + 1 - p):
            for r in range(n + 1 - p - q):
                orbit_expectations[p, q, r] = dp[p, q, r] / _orbit_size(n, p, q, r)

    e_in = np.zeros(len(g_basis), dtype=np.float64)
    for i, basis_dict in enumerate(g_basis):
        val = 0.0
        for (p, q, r), coeff in basis_dict.items():
            val += float(coeff) * orbit_expectations[p, q, r]
        e_in[i] = val

    return e_in