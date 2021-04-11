import copy
import numpy as np
import cirq


def trace_distance(response_unitary, expected_unitary):
    # extra points for exact equality
    u = response_unitary @ expected_unitary.conj().T
    trace_distance = cirq.trace_distance_from_angle_list(
        np.angle(np.linalg.eigvals(u))
    )
    return trace_distance


def lb_two_qubit_gates(n_qubits):
    return int(1 / 4 * (4 ** n_qubits - 3 * n_qubits - 1))


def n_two_qubit_gates(circuit):
    return len([op for op in circuit.all_operations() if cirq.num_qubits(op) == 2])


def is_identity(m):
    return np.allclose(m, np.eye(len(m)))


# https://stackoverflow.com/questions/43884189/check-if-a-large-matrix-is-diagonal-matrix-in-python
def is_diagonal(m):
    b = np.zeros_like(m)
    np.fill_diagonal(b, m.diagonal())
    return np.allclose(m, b)


def is_incrementer(m):
    b = np.empty_like(m)
    b[1:] = np.eye(len(m))[:-1]
    b[:1] = np.eye(len(m))[-1:]
    return np.allclose(m, b)


class GridQubitMap:
    def __init__(self, n_qubits, grid_qubits=None):
        self.__n_qubits = n_qubits
        self.__named_qubits = [cirq.NamedQubit('q_{}'.format(i)) for i in range(n_qubits)]
        if grid_qubits is not None:
            self.__grid_qubits = copy.deepcopy(grid_qubits)
        else:
            self.__grid_qubits = self._get_grid_qubits(n_qubits)
        self.__mapping_dict = dict(zip(self.__named_qubits, self.__grid_qubits))
        self.__inv_mapping_dict = {v: k for k, v in self.__mapping_dict.items()}
        self.__coupling_map = self._generate_coupling_map(n_qubits)
    
    @property
    def named_qubits(self):
        return copy.deepcopy(self.__named_qubits)
    
    @property
    def grid_qubits(self):
        return copy.deepcopy(self.__grid_qubits)
    
    @property
    def mapping_dict(self):
        return copy.deepcopy(self.__mapping_dict)
    
    @property
    def inv_mapping_dict(self):
        return copy.deepcopy(self.__inv_mapping_dict)
    
    @property
    def coupling_map(self):
        return copy.deepcopy(self.__coupling_map)
    
    def mapping(self, named_qubit):
        return self.__mapping_dict[named_qubit]

    @classmethod
    def _get_grid_qubits(self, n_qubits):
        if n_qubits < 4:
            qs = cirq.GridQubit.rect(1, n_qubits, 3, 3)
        elif int(np.sqrt(n_qubits)) ** 2 == n_qubits:
            qs = cirq.GridQubit.square(int(np.sqrt(n_qubits)), 3, 3)
        elif n_qubits % 2 == 0:
            qs = cirq.GridQubit.rect(2, int(n_qubits / 2), 3, 3)
        else:
            qs = cirq.GridQubit.rect(2, int((n_qubits + 1) / 2), 3, 3)[:-1]
        return qs

    def _generate_coupling_map(self, n_qubits):
        coupling_map = {cirq.NamedQubit('q_{}'.format(i)): [] for i in range(n_qubits)}
        for i in range(n_qubits):
            gqi = self.__grid_qubits[i]
            for j in range(i + 1, n_qubits):
                gqj = self.__grid_qubits[j]
                if abs(gqi.row - gqj.row) + abs(gqi.col - gqj.col) == 1:
                    coupling_map[cirq.NamedQubit('q_{}'.format(i))].append(cirq.NamedQubit('q_{}'.format(j)))
                    coupling_map[cirq.NamedQubit('q_{}'.format(j))].append(cirq.NamedQubit('q_{}'.format(i)))
        return coupling_map
