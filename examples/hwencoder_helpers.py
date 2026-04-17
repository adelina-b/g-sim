"""
Helper routines for the fixed-Hamming-weight amplitude-encoding experiment.

This module contains the combinatorial and basis-mapping utilities needed
to reproduce the HW-preserving encoder discussed in the numerical
experiment. Its role is to bridge three different descriptions of the same
fixed-Hamming-weight sector:

1. Chase / revolving-door ordering of HW-k bitstrings
   This is the ordering used by the encoder circuit, where consecutive
   basis states differ by Hamming distance 2.

2. Logical MGGM ordering
   This is the fixed lexicographic ordering used to label the logical
   basis states and the associated MGGM generators.

3. Physical RBS-gate description
   Consecutive Chase states determine reconfigurable beam-splitter (RBS)
   operations, which are then translated into sparse linear combinations
   of antisymmetric MGGM generators.

Main functionality
------------------
- build the Chase ordering and logical ordering of HW-k bitstrings,
- construct maps between both orderings,
- infer the sequence of controlled RBS gates from the Chase ordering,
- express each physical RBS gate as a sparse combination of MGGM
  antisymmetric generators.

Notes
-----
- The helpers in this file are specific to the fixed-HW encoder experiment.
- The returned MGGM vectors use the same logical ordering convention as
  the `gsim.gmggm` backend.
- The gate-to-generator mapping currently assumes the antisymmetric
  MGGM sector (`TYPE_A`) for RBS-type evolutions.
"""

from itertools import combinations
import numpy as np

def get_ordered_bitstrings(n: int, k: int):
    """
    Return the Chase / revolving-door ordering of HW-k bitstrings.

    Parameters
    ----------
    n : int
        Total number of qubits.
    k : int
        Fixed Hamming weight.

    Returns
    -------
    list[tuple[int, ...]]
        Ordered list of length-`n` bitstrings with exactly `k` ones.

    Notes
    -----
    Adjacent bitstrings in this ordering differ by Hamming distance 2,
    which makes the sequence directly compatible with the RBS-based
    fixed-HW encoder circuit.
    """

    def chase(n, k):
        if k == 0: return [[0] * n]
        if n == k: return [[1] * n]
        seq = []
        for sub in chase(n - 1, k - 1):
            seq.append([1] + sub)
        for sub in reversed(chase(n - 1, k)):
            seq.append([0] + sub)
        return seq

    return [tuple(s) for s in chase(n, k)]

def get_logical_bitstrings(n: int, k: int):
    """
    Return the logical MGGM ordering of HW-k bitstrings.

    Parameters
    ----------
    n : int
        Total number of qubits.
    k : int
        Fixed Hamming weight.

    Returns
    -------
    list[tuple[int, ...]]
        Lexicographically ordered list of all HW-k bitstrings.

    Notes
    -----
    This ordering defines the logical labels `0, ..., d_k - 1` used in
    the MGGM basis and is intentionally independent of the Chase order
    used by the encoder circuit.
    """
    states = []
    for ones in combinations(range(n), k):
        bits = [0] * n
        for i in ones:
            bits[i] = 1
        states.append(tuple(bits))
    return states

def build_state_maps(n: int, k: int):
    """
    Build the correspondence between Chase ordering and logical MGGM ordering.

    Parameters
    ----------
    n : int
        Total number of qubits.
    k : int
        Fixed Hamming weight.

    Returns
    -------
    tuple
        `(chase_states, logical_states, state_to_logical, chase_to_logical)`,
        where
        - `chase_states` is the encoder ordering,
        - `logical_states` is the MGGM logical ordering,
        - `state_to_logical` maps a bitstring to its logical index,
        - `chase_to_logical` maps a Chase position to a logical index.

    Notes
    -----
    This function is the central bookkeeping utility for translating
    between circuit-level state orderings and MGGM-level logical labels.
    """
    chase_states = get_ordered_bitstrings(n, k)
    logical_states = get_logical_bitstrings(n, k)

    state_to_logical = {st: i for i, st in enumerate(logical_states)}
    chase_to_logical = np.array([state_to_logical[st] for st in chase_states], dtype=np.int32)

    return chase_states, logical_states, state_to_logical, chase_to_logical

def get_circuit_gates(states):
    """
    Infer the sequence of physical RBS gates from an ordered list of states.

    Parameters
    ----------
    states : list[tuple[int, ...]]
        Ordered bitstrings, typically in Chase order.

    Returns
    -------
    list[tuple[int, int, set[int]]]
        List of gate specifications `(in_idx, out_idx, ctrl)`, where
        `in_idx` is the occupied site to be emptied,
        `out_idx` is the empty site to be filled, and
        `ctrl` is the set of sites that remain occupied in both states.

    Notes
    -----
    This routine assumes that consecutive states differ by Hamming
    distance 2, as guaranteed by the Chase ordering.
    """
    gate_params = []
    for i in range(len(states) - 1):
        st1 = states[i]
        st2 = states[i + 1]

        I1 = set(idx for idx, val in enumerate(st1) if val == 1)
        I2 = set(idx for idx, val in enumerate(st2) if val == 1)

        ctrl = I1.intersection(I2)
        in_idx = (I1 - ctrl).pop()
        out_idx = (I2 - ctrl).pop()

        gate_params.append((in_idx, out_idx, ctrl))
    return gate_params

def get_RBSgate_MGGM(in_idx, out_idx, ctrl, logical_states, state_to_logical, basis_map_arr):
    """
    Express a physical controlled-RBS gate as a sparse MGGM generator vector.

    Parameters
    ----------
    in_idx : int
        Input mode that loses occupation.
    out_idx : int
        Output mode that gains occupation.
    ctrl : collection[int]
        Sites that must remain occupied for the gate to act.
    logical_states : list[tuple[int, ...]]
        Logical HW-k basis states in MGGM ordering.
    state_to_logical : dict
        Map from bitstring to logical MGGM index.
    basis_map_arr : np.ndarray
        MGGM basis lookup array.

    Returns
    -------
    dict[int, float]
        Sparse coefficient dictionary encoding the corresponding linear
        combination of antisymmetric MGGM generators.

    Notes
    -----
    The returned vector has support only on `TYPE_A` generators and follows
    the MGGM sign convention determined by the canonical ordering of the
    logical labels.
    """
    vec = {}
    TYPE_A = 0

    for a, st_a in enumerate(logical_states):
        if st_a[in_idx] == 1 and st_a[out_idx] == 0:
            if all(st_a[c] == 1 for c in ctrl):
                st_b = list(st_a)
                st_b[in_idx] = 0
                st_b[out_idx] = 1
                st_b = tuple(st_b)

                b = state_to_logical[st_b]

                # Generator is |b><a| - |a><b|
                if a < b:
                    lam_idx = basis_map_arr[TYPE_A, a, b]
                    vec[lam_idx] = vec.get(lam_idx, 0.0) - 1.0
                else:
                    lam_idx = basis_map_arr[TYPE_A, b, a]
                    vec[lam_idx] = vec.get(lam_idx, 0.0) + 1.0

    return {mu: c for mu, c in vec.items() if abs(c) > 1e-12}