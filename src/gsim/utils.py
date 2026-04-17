"""
Shared utility helpers for basis construction and sparse-data loading.

This module contains small helper routines used across the example scripts.
Its functionality falls into two categories:

1. Basis construction
   - Pauli-orbit basis labels for permutation-invariant simulations.
   - MGGM basis-index lookup arrays for fixed-Hamming-weight simulations.

2. Sparse-data loading
   - Reconstruction of targeted sparse generator matrices from compressed
     coordinate arrays stored in `.npz` files.
   - Reconstruction of Pauli-string adjoint block data from compressed
     block-format arrays.

These helpers are intentionally lightweight and experiment-oriented. They do
not implement Lie closure or preprocessing themselves; instead, they provide
the basis metadata and file-loading utilities needed to use the optimized
backends in `gstrings.py`, `gorbits.py`, and `gmggm.py`.

Notes
-----
- Orbit basis elements are represented as sparse single-key dictionaries
  `{(p, q, r): 1}`.
- The MGGM basis is represented by a dense lookup array
  `basis_map_arr[type, a, b] -> basis index`.
- Sparse matrices loaded by this module are returned in CSR format, ready for
  use with `scipy.sparse.linalg.expm_multiply`.
"""

import numpy as np
import scipy.sparse as sp
import math

def get_orbit_basis(n: int) -> list[dict]:
    """
    Construct the full single-orbit basis on `n` qubits.

    Parameters
    ----------
    n : int
        Number of qubits.

    Returns
    -------
    list[dict]
        List of sparse single-key dictionaries of the form `{(p, q, r): 1}`,
        one for each nontrivial Pauli orbit with `0 < p + q + r <= n`.

    Notes
    -----
    Each triple `(p, q, r)` labels the permutation-invariant orbit containing
    all Pauli strings with exactly `p` X's, `q` Y's, and `r` Z's. The basis is
    ordered by total support size `s = p + q + r`, then by increasing `p`,
    then by increasing `q`.
    """
    return [
        {(p, q, s - p - q): 1}
        for s in range(1, n + 1)
        for p in range(s + 1)
        for q in range(s - p + 1)
    ]

def get_mggm_basis(n: int, k: int):
    """
    Construct the MGGM basis-index lookup array for the fixed-HW sector `(n, k)`.

    Parameters
    ----------
    n : int
        Number of qubits.
    k : int
        Fixed Hamming weight.

    Returns
    -------
    np.ndarray
        Integer lookup array `basis_map_arr` of shape `(3, d_k, d_k)`, where
        `d_k = binom(n, k)`. The first axis corresponds to the MGGM sectors
        `TYPE_A`, `TYPE_S`, and `TYPE_P`, and valid entries contain the dense
        basis index of the corresponding generator.

    Notes
    -----
    The sectors are indexed as
        TYPE_A = 0   antisymmetric off-diagonal generators,
        TYPE_S = 1   symmetric off-diagonal generators,
        TYPE_P = 2   diagonal/projector-like generators.

    Only canonical pairs `a < b` are stored for the off-diagonal sectors.
    Diagonal generators are stored as `basis_map_arr[TYPE_P, a, a]`.
    Invalid entries are filled with `-1`.
    """
    TYPE_A = 0
    TYPE_S = 1
    TYPE_P = 2

    dk = math.comb(n, k)
    basis_map_arr = np.full((3, dk, dk), -1, dtype=np.int32)
    a_idx, b_idx = np.triu_indices(dk, k=1)
    a_idx = np.array(a_idx, dtype=np.int32)
    b_idx = np.array(b_idx, dtype=np.int32)
    N_off = len(a_idx)

    for i, (a, b) in enumerate(zip(a_idx, b_idx)):
        basis_map_arr[TYPE_A, a, b] = i  # Map A
        basis_map_arr[TYPE_S, a, b] = N_off + i  # Map S

    dk = math.comb(n, k)
    for a in range(dk):
        basis_map_arr[TYPE_P, a, a] = 2 * N_off + a  # Map P

    return basis_map_arr

def load_adjoint_generators(filepath: str, dim_g: int) -> list:
    """
    Load targeted sparse generator matrices from a compressed `.npz` file.

    Parameters
    ----------
    filepath : str
        Path to the `.npz` file containing sparse coordinate data.
    dim_g : int
        Dimension of the Lie basis, used to define the matrix shape.

    Returns
    -------
    list[scipy.sparse.csr_matrix]
        List of sparse CSR matrices representing the loaded generator actions.

    Notes
    -----
    The `.npz` file is expected to store arrays named
        `rows_0`, `cols_0`, `vals_0`,
        `rows_1`, `cols_1`, `vals_1`, ...
    for each target generator. Each triple is interpreted as COO-format sparse
    data and converted into a CSR matrix of shape `(dim_g, dim_g)`.

    This loader is used for targeted preprocessing outputs, where the stored
    objects are already the actual sparse evolution / adjoint matrices rather
    than grouped raw structure-constant data.
    """
    data = np.load(filepath)
    num_targets = sum(1 for key in data.keys() if key.startswith('rows_'))

    mats = []
    for i in range(num_targets):
        r = data[f"rows_{i}"]
        c = data[f"cols_{i}"]
        v = data[f"vals_{i}"]

        M = sp.csr_matrix((v, (r, c)), shape=(dim_g, dim_g))
        mats.append(M)

    return mats

def load_adj_blocks(filepath):
    """
    Load Pauli-string adjoint block data from a compressed `.npz` file.

    Parameters
    ----------
    filepath : str
        Path to the `.npz` file containing compressed block arrays.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray, np.ndarray]]
        List of triples `(u, v, f)` encoding the 2x2 block structure of the
        adjoint generators.

    Notes
    -----
    The `.npz` file is expected to store arrays
        `u_0`, `v_0`, `f_0`,
        `u_1`, `v_1`, `f_1`, ...
    where each stored triple represents only one oriented half of the block
    data. This function reconstructs the full antisymmetric block structure as

        u_full = [u, v]
        v_full = [v, u]
        f_full = [f, -f]

    so that each returned triple can be used directly by the fast Pauli-string
    evolution helpers based on planar 2x2 rotations.
    """
    data = np.load(filepath)
    dim_dla = sum(1 for key in data.keys() if key.startswith('u_'))

    adj_blocks = []
    for i in range(dim_dla):
        u = data[f"u_{i}"]
        v = data[f"v_{i}"]
        f = data[f"f_{i}"]
        adj_blocks.append((np.concatenate([u, v]), np.concatenate([v, u]), np.concatenate([f, -f])))

    return adj_blocks