"""
g-sim simulator functions for the permutation-equivariant QNN experiment.

This module assembles callable expectation-value and gradient functions for
the graph-state classification experiment based on permutation-invariant
Pauli-orbit generators.

For a fixed system size, circuit depth, and input graph instance, the main
entry point `simulator_functions(...)`:

1. builds the Pauli-orbit basis,
2. loads or computes the targeted orbit-generator matrices,
3. constructs the graph-state input expectation vector e_in,
4. defines the XX-type observable used in the experiment,
5. returns closures for evaluating the loss and, optionally, its gradient.

The resulting functions operate entirely in adjoint / expectation space and
are intended for large-scale numerical experiments where direct state-vector
simulation becomes impractical.
"""

from pathlib import Path
import scipy.sparse as sp
import numpy as np
import gsim

from peqnn_helpers import generate_ein_disconnected_graph

def simulator_functions(size: int, layers: int, graph_list: list,
                        return_grad: bool = True):
    """
    Build expectation-value and gradient callables for one graph-state instance.

    Parameters
    ----------
    size : int
        Number of qubits.
    layers : int
        Number of circuit layers.
    graph_list : list[np.ndarray]
        Connected-component adjacency matrices describing the disconnected
        input graph.
    return_grad : bool, optional
        If `True`, also return a gradient function.

    Returns
    -------
    callable or tuple[callable, callable]
        If `return_grad=False`, returns a function
            f(thetas) -> expectation value.
        If `return_grad=True`, returns
            (f, grad_f)
        where `grad_f(thetas)` evaluates the gradient with respect to all
        circuit parameters.

    Notes
    -----
    Each layer applies three generators in sequence, corresponding to the
    permutation-invariant orbit elements `(1,0,0)`, `(0,1,0)`, and `(0,0,2)`.
    The observable is the normalized orbit `(2,0,0)`, i.e. the collective
    two-body XX measurement used in the experiment.
    """
    # construct basis
    g_basis = gsim.utils.get_orbit_basis(size)
    dim_g = len(g_basis)

    # get adjoint representation
    try:
        DATA_DIR = Path(__file__).resolve().parent / "data"
        adj_path = DATA_DIR / f"adj_gorbits_targeted_n{size}.npz"
        adj_gens = gsim.utils.load_adjoint_generators(adj_path, dim_g)
    except FileNotFoundError:
        # orbit generators in layer order: X, Y, ZZ
        target_indices = [
            g_basis.index({(1, 0, 0): 1}),
            g_basis.index({(0, 1, 0): 1}),
            g_basis.index({(0, 0, 2): 1})
        ]
        adj_gens = gsim.gorbits.adjoint_orbit_generators_targeted(g_basis,
                                       target_indices, size)

    # initial state
    e_in = generate_ein_disconnected_graph(size, graph_list, g_basis)

    # observable: normalized XX orbit
    idxs = [g_basis.index({(2, 0, 0): 1})]
    w = np.zeros(dim_g, dtype=np.float64)
    w[idxs] = 1.0

    def get_expectation_value(thetas):
        e_out = e_in.copy()
        for l in range(layers):
            alpha, beta, gamma = thetas[3 * l], thetas[3 * l + 1], thetas[3 * l + 2]

            e_out = sp.linalg.expm_multiply(alpha * adj_gens[0], e_out, traceA=0.0)
            e_out = sp.linalg.expm_multiply(beta * adj_gens[1], e_out, traceA=0.0)
            e_out = sp.linalg.expm_multiply(gamma * adj_gens[2], e_out, traceA=0.0)

        return np.dot(w, e_out)

    if return_grad:
        adj_blocks_T = [M.transpose().tocsr() for M in adj_gens]
        def get_expectation_gradient(thetas):
            grad = np.zeros(len(thetas), dtype=np.float64)

            # Forward-state caching
            forward_states = [e_in.copy()]
            current_state = e_in.copy()
            # forward_states[j] stores the state after j generator applications


            for l in range(layers):
                alpha, beta, gamma = thetas[3 * l], thetas[3 * l + 1], thetas[3 * l + 2]

                current_state = sp.linalg.expm_multiply(
                    alpha * adj_gens[0], current_state, traceA=0.0)
                forward_states.append(current_state.copy())

                current_state = sp.linalg.expm_multiply(
                    beta * adj_gens[1], current_state, traceA=0.0)
                forward_states.append(current_state.copy())

                current_state = sp.linalg.expm_multiply(
                    gamma * adj_gens[2], current_state, traceA=0.0)
                forward_states.append(current_state.copy())

            # Reverse-mode differentiation
            vec_l = w.copy()

            for l in reversed(range(layers)):
                alpha, beta, gamma = thetas[3 * l], thetas[3 * l + 1], thetas[3 * l + 2]

                # state after gamma_l
                vec_r_gamma = forward_states[3 * l + 3]
                grad[3 * l + 2] = np.dot(vec_l, adj_gens[2].dot(vec_r_gamma))
                vec_l = sp.linalg.expm_multiply(gamma * adj_blocks_T[2], vec_l, traceA=0.0)

                # state after beta_l
                vec_r_beta = forward_states[3 * l + 2]
                grad[3 * l + 1] = np.dot(vec_l, adj_gens[1].dot(vec_r_beta))
                vec_l = sp.linalg.expm_multiply(beta * adj_blocks_T[1], vec_l, traceA=0.0)

                # state after alpha_l
                vec_r_alpha = forward_states[3 * l + 1]
                grad[3 * l] = np.dot(vec_l, adj_gens[0].dot(vec_r_alpha))
                vec_l = sp.linalg.expm_multiply(alpha * adj_blocks_T[0], vec_l, traceA=0.0)

            return grad

        return get_expectation_value, get_expectation_gradient

    else:
        return get_expectation_value