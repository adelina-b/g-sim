"""
Adjoint-space simulator for the fixed-Hamming-weight amplitude encoder.

This module reproduces the HW-preserving state-preparation experiment by
constructing a target q-Gaussian probability distribution, translating the
corresponding encoder circuit into MGGM generators, and simulating the
resulting adjoint-space evolution with g-sim.

For fixed `(n, k)`, the main entry point:

1. determines the dimension `d_k = binom(n, k)`,
2. defines the target probability distribution on `d_k` grid points,
3. computes the deterministic hyperspherical rotation angles,
4. constructs the Chase-ordered sequence of controlled RBS gates,
5. maps each gate to a sparse MGGM generator,
6. evolves the initial projector in adjoint space, and
7. reads out the encoded probabilities from diagonal MGGM coordinates.

The returned arrays can be used directly for plotting the encoded and
target distributions.

Notes
-----
- This is an experiment-specific driver rather than a general-purpose
  fixed-HW simulation API.
- The target distribution is currently hard-coded to the q-Gaussian-like
  profile used in the manuscript example.
- The simulation uses the targeted MGGM backend, so only the required
  generator matrices are constructed on the fly.
"""

import math
import numpy as np
import scipy.sparse as sp

import gsim
from hwencoder_helpers import build_state_maps, get_circuit_gates, get_RBSgate_MGGM


def simulator(n: int, k: int):
    """
    Simulate the fixed-HW amplitude encoder for a q-Gaussian target distribution.

    Parameters
    ----------
    n : int
        Number of qubits.
    k : int
        Fixed Hamming weight.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        `(xs, ps_sim, ps_targ)`, where
        - `xs` are the grid points on `[-2, 2]`,
        - `ps_sim` are the probabilities produced by the simulated encoder,
        - `ps_targ` are the normalized target probabilities.

    Notes
    -----
    The target amplitudes are converted into hyperspherical angles, and the
    encoder is simulated by evolving an initial diagonal MGGM basis vector
    through the sequence of targeted adjoint generators corresponding to the
    controlled RBS gates.
    """
    dk = math.comb(n, k)
    dim_g = dk ** 2
    g_basis_map = gsim.utils.get_mggm_basis(n, k)

    # Target Probability Distribution (q-Gaussian)
    xs = np.array([-2 + 4 * j / (dk - 1) for j in range(dk)], dtype=np.float64)
    ps_targ = np.array([(1 + x**2)**(-2) for x in xs], dtype=np.float64)
    ps_targ = ps_targ / ps_targ.sum()
    as_targ = np.sqrt(ps_targ)

    # Compute rotation angles
    thetas = []
    for j in range(dk-2):
        rem_norm = np.sqrt(np.sum(as_targ[j + 1:] ** 2))
        thetas.append(math.atan2(rem_norm, as_targ[j]))
    thetas.append(math.atan2(as_targ[-1], as_targ[-2]))

    # Circuit Architecture
    chase_states, logical_states, state_to_logical, chase_to_logical = build_state_maps(n, k)
    gates = get_circuit_gates(chase_states)
    target_basis = gsim.gmggm.FastMGGMBasisTargeted(n, k)

    # Initialize e_in = projector onto the FIRST Chase state
    vec_state = np.zeros(dim_g, dtype=np.float64)
    logical_init = chase_to_logical[0]
    idx_P0 = g_basis_map[2, logical_init, logical_init]  # TYPE_P
    vec_state[idx_P0] = 1.0

    # Adjoint state evolution
    for j in range(dk - 1):
        in_idx, out_idx, ctrl = gates[j]
        vec_G = get_RBSgate_MGGM(in_idx, out_idx, ctrl, logical_states,
                                 state_to_logical, g_basis_map)
        ad_G = target_basis.build_targeted_adjoint_generator(vec_G)

        vec_state = sp.linalg.expm_multiply(thetas[j] * ad_G, vec_state)

    ps_sim = []
    for chase_pos in range(dk):
        logical_idx = chase_to_logical[chase_pos]
        idx_Pa = g_basis_map[2, logical_idx, logical_idx]
        ps_sim.append(vec_state[idx_Pa])

    return xs, np.array(ps_sim), ps_targ