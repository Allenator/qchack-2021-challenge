from typing import List, Tuple

import numpy as np
import cirq

from solution.transpiler import *
from solution.utils import *


def matrix_to_sycamore_operations(
    target_qubits: List[cirq.GridQubit], matrix: np.ndarray
) -> Tuple[cirq.OP_TREE, List[cirq.GridQubit]]:
    """A method to convert a unitary matrix to a list of Sycamore operations.

    This method will return a list of `cirq.Operation`s using the qubits and (optionally) ancilla
    qubits to implement the unitary matrix `matrix` on the target qubits `qubits`.
    The operations are also supported by `cirq.google.gate_sets.SYC_GATESET`.

    Args:
        target_qubits: list of qubits the returned operations will act on. The qubit order defined by the list
            is assumed to be used by the operations to implement `matrix`.
        matrix: a matrix that is guaranteed to be unitary and of size (2**len(qs), 2**len(qs)).
    Returns:
        A tuple of operations and ancilla qubits allocated.
            Operations: In case the matrix is supported, a list of operations `ops` is returned.
                `ops` acts on `qs` qubits and for which `cirq.unitary(ops)` is equal to `matrix` up
                 to certain tolerance. In case the matrix is not supported, it might return NotImplemented to
                 reduce the noise in the judge output.
            Ancilla qubits: In case ancilla qubits are allocated a list of ancilla qubits. Otherwise
                an empty list.
        .
    """
    
    RANDOM_UNITARY_SMALL_CUTOFF = 2
    RANDOM_UNITARY_ADAPTIVE_CUTOFF = 6

    n_qubits = len(target_qubits)
    gqm = GridQubitMap(n_qubits, grid_qubits=target_qubits)

    if is_identity(matrix):
        return [], []
    
    elif is_incrementer(matrix):
        response_circuit = incrementer_decompose(n_qubits, matrix)
        optm_circuit = cirq_optimize(n_qubits, response_circuit, gqm, target='adaptive')
        return list(optm_circuit.all_operations()), []
    
    elif is_diagonal(matrix):
        response_circuit = diagonal_decompose(n_qubits, matrix)
        optm_circuit = cirq_optimize(n_qubits, response_circuit, gqm, target='adaptive')
        return list(optm_circuit.all_operations()), []

    else:
        response_circuit = random_decompose(n_qubits, matrix)
        if n_qubits <= RANDOM_UNITARY_SMALL_CUTOFF:
            optm_circuit = cirq_optimize(n_qubits, response_circuit, gqm, target='small+adaptive')
        elif n_qubits <= RANDOM_UNITARY_ADAPTIVE_CUTOFF:
            optm_circuit = cirq_optimize(n_qubits, response_circuit, gqm, target='adaptive')
        else:
            optm_circuit = cirq_optimize(n_qubits, response_circuit, gqm, target='small_unitary')
        return list(optm_circuit.all_operations()), []
