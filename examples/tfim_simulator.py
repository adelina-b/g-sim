"""
g-sim simulator closures for the freely-parameterized TFIM-HVA experiment.

This module assembles callable expectation-value and gradient functions for
the overparameterized TFIM benchmark described in the manuscript. It uses
the Pauli-string preprocessing backend `gsim.gstrings` together with the
special block structure of Pauli-string adjoint generators to obtain fast
adjoint-space evolution and reverse-mode gradients.

For a fixed system size, circuit depth, field strength, initial product
state, and boundary convention, the main entry point `simulator_functions`
performs the following steps:

1. builds the freely-parameterized TFIM generator set,
2. loads or computes the corresponding Lie basis and adjoint block data,
3. constructs the initial expectation vector e_in,
4. assembles the TFIM observable coefficient vector,
5. returns closures for repeated loss and gradient evaluation.

The resulting functions are intended for repeated optimization runs in the
TFIM numerical experiment rather than as a fully general simulator API.
"""

import numpy as np
import PauliEngine as pe
from pathlib import Path

import gsim

from tfim_helpers import initial_expvector ,get_TFIM_generators, get_TFIM_coefficients, \
    pauli_expm_multiply, pauli_undo_expm_multiply, adj_multiply

def simulator_functions(size: int, layers: int, g: float,
                        rho: str = 'plus', bounds: str = 'open',
                        return_grad: bool = True):
    """
    Build TFIM-HVA expectation-value and gradient callables.

    Parameters
    ----------
    size : int
        Number of qubits.
    layers : int
        Number of ansatz layers.
    g : float
        Transverse-field strength in the TFIM Hamiltonian.
    rho : {'plus', 'zero'}, optional
        Initial product state used to construct the input expectation vector.
    bounds : {'open', 'periodic'}, optional
        Boundary convention for the TFIM couplings.
    return_grad : bool, optional
        If `True`, also return a gradient function.

    Returns
    -------
    callable or tuple[callable, callable]
        If `return_grad=False`, returns
            f(thetas) -> expectation value.
        If `return_grad=True`, returns
            (f, grad_f),
        where `grad_f(thetas)` is the reverse-mode gradient with respect to all
        circuit parameters.

    Notes
    -----
    The circuit uses the freely-parameterized TFIM generator set: each layer
    contains one independently parameterized gate per generator. The parameter
    ordering is
        all X rotations first, then all ZZ rotations,
    repeated layer by layer, matching the internal evolution loops used in
    this function.
    """

    generators = [pe.PauliString(op, 1) for op in get_TFIM_generators(size, bounds)]

    # get Lie algebra basis and adjoint data
    try:
        DATA_DIR = Path(__file__).resolve().parent / "data"
        adj_path = DATA_DIR / f"adj_gstrings_TFIM_n{size}.npz"
        adj_blocks = gsim.utils.load_adj_blocks(adj_path)
        g_basis = gsim.gstrings.lie_closure(generators)
    except FileNotFoundError:
        g_basis, adj_blocks = gsim.gstrings.pauli_preprocessing(generators)
    dim_g = len(g_basis)

    # initial state (based on 'rho')
    e_in = initial_expvector(g_basis, rho)

    # TFIM observable with J=1
    J = 1
    w = np.zeros(dim_g, dtype=float)
    w[:len(generators)] = get_TFIM_coefficients(size, g, J, bounds)

    num_zz_terms = size if bounds=='periodic' else size - 1
    num_x_terms = size

    def get_expectation_value(thetas):
        num_params = len(thetas)
        num_gates = int((num_x_terms + num_zz_terms) * layers)
        if num_params != num_gates:
            raise ValueError(f'Only {num_params} parameters were given, but there are {num_gates} gates.')

        e_out = e_in.copy()
        idx = 0
        for layer in range(layers):
            for i in range(num_zz_terms, num_zz_terms + num_x_terms):
                pauli_expm_multiply(adj_blocks[i], e_out, thetas[idx])
                idx += 1
            for i in range(num_zz_terms):
                pauli_expm_multiply(adj_blocks[i], e_out, thetas[idx])
                idx += 1
        return np.dot(w, e_out)

    if return_grad:
        def get_expectation_gradient(thetas):
            grad = np.zeros(len(thetas))

            e_out = e_in.copy()
            idx = 0
            for layer in range(layers):
                for i in range(num_zz_terms, num_zz_terms + num_x_terms):
                    pauli_expm_multiply(adj_blocks[i], e_out, thetas[idx])
                    idx += 1
                for i in range(num_zz_terms):
                    pauli_expm_multiply(adj_blocks[i], e_out, thetas[idx])
                    idx += 1

            # reverse mode calculation
            vec_r = e_out.copy()
            vec_l = w.copy()

            for layer in reversed(range(layers)):

                for i in reversed(range(num_zz_terms)):
                    idx -= 1
                    grad[idx] = np.dot(vec_l, adj_multiply(adj_blocks[i], vec_r))
                    pauli_undo_expm_multiply(adj_blocks[i], vec_r, thetas[idx])
                    pauli_undo_expm_multiply(adj_blocks[i], vec_l, thetas[idx])

                for i in reversed(range(num_zz_terms, num_zz_terms + num_x_terms)):
                    idx -= 1
                    grad[idx] = np.dot(vec_l, adj_multiply(adj_blocks[i], vec_r))
                    pauli_undo_expm_multiply(adj_blocks[i], vec_r, thetas[idx])
                    pauli_undo_expm_multiply(adj_blocks[i], vec_l, thetas[idx])

            return grad

    if return_grad:
        return get_expectation_value, get_expectation_gradient
    else:
        return get_expectation_value
