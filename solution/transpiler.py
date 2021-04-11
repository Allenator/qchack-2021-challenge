from pathlib import Path
import shutil
import os
from joblib import Parallel, delayed

import numpy as np

from openql import openql as ql

import cirq
from cirq.contrib.qasm_import import circuit_from_qasm

import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from solution.utils import *


TMP_DIR = '/tmp'
DIR_FN = 'sycamore_transpiler'
CONFIG_FN = 'backend_config.json'
RAW_QASM_FN = 'raw.qasm'
IR0_QASM_FN = 'ir0.qasm'
IR1_QASM_FN = 'ir1.qasm'

PLATFORM_STR = 'generic'
PROGRAM_STR = 'program'
KERNEL_STR = 'kernel'
UNITARY_STR = 'unitary'
COMPILER_STR = 'compiler'
WRITER_STR = 'openql'


def _make_tmp_dir():
    dir_path = Path(TMP_DIR, DIR_FN)
    if dir_path.exists():
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)
    return dir_path


DIR_PATH = _make_tmp_dir()
CONFIG_PATH = Path(DIR_PATH, CONFIG_FN)


def _write_backend_config(config_path):
    backend_json = '''{
    "eqasm_compiler" : "qx",
    "hardware_settings": {
        "qubit_number": 10,
        "cycle_time" : 20
    },
    "resources": {},
    "topology" : {},
    "instructions": {}
}'''
    with open(config_path, 'w') as writer:
        writer.write(backend_json)


def _openql_compile(dir_path, config_path, n_qubits, unitary):
    ql.set_option('output_dir', str(dir_path))
    platform = ql.Platform(PLATFORM_STR, str(config_path))
    p = ql.Program(PROGRAM_STR, platform, n_qubits)
    k = ql.Kernel(KERNEL_STR, platform, n_qubits)
    u_mat = ql.Unitary(UNITARY_STR, unitary.flatten())
    u_mat.decompose()
    k.gate(u_mat, range(n_qubits))
    p.add_kernel(k)
    c = ql.Compiler(COMPILER_STR)
    c.add_pass_alias('Writer', WRITER_STR)
    c.set_pass_option(WRITER_STR, 'write_qasm_files', 'yes')
    c.compile(p)


def _make_qasm_preamble(n_qubits):
    return 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{}];\n'.format(n_qubits)


def _rewrite_openql_qasm(dir_path, n_qubits):
    OPENQL_OUTPUT_PATH = Path(dir_path, PROGRAM_STR + '_' + WRITER_STR + '_out.qasm')
    RAW_OUTPUT_PATH = Path(dir_path, RAW_QASM_FN)
    preamble = _make_qasm_preamble(n_qubits)

    output_flag = False
    body = ''
    with open(OPENQL_OUTPUT_PATH, 'r') as reader:
        for line in reader:
            if output_flag:
                line = line.strip(' \n')

                if 'ry' in line or 'rz' in line:
                    arr = line.split()
                    arr[0] += '({})'.format(arr[2])
                    line = arr[0] + ' ' + arr[1].strip(',')
                line = line.replace('cnot', 'cx')
                body += line + ';\n'

            if '.' + KERNEL_STR in line:
                output_flag = True

    with open(RAW_OUTPUT_PATH, 'w') as writer:
        writer.write(preamble + body)
    
    return preamble + body


def _invert_qubit_order(n_qubits, dir_path, input_fn, output_fn):
    qubit_str_array = ['q[{}]'.format(i) for i in range(n_qubits)]
    qubit_tmp_array = ['QUBIT_TMP[{}]'.format(i) for i in range(n_qubits)]
    qubit_str_dict = dict(zip(qubit_str_array, qubit_tmp_array[::-1]))
    qubit_tmp_dict = dict(zip(qubit_tmp_array, qubit_str_array))

    body = ''
    with open(Path(dir_path, input_fn), 'r') as reader:
        for line in reader:
            for p, q in qubit_str_dict.items():
                line = line.replace(p, q)
            for p, q in qubit_tmp_dict.items():
                line = line.replace(p, q)
            body += line
    
    with open(Path(dir_path, output_fn), 'w') as writer:
        writer.write(body)
    
    return body


# def _global_phase_gate(circuit, theta, qubit):
#     circuit.p(theta, qubit)
#     circuit.x(qubit)
#     circuit.p(theta, qubit)
#     circuit.x(qubit)


def _make_response_circuit(n_qubits, response_qasm_string):
    # expected_circ = QuantumCircuit(n_qubits)
    # expected_circ.unitary(expected_unitary, range(n_qubits))
    response_circuit = QuantumCircuit.from_qasm_str(response_qasm_string)

    # expected_sv = np.array(Statevector.from_instruction(expected_circ))
    # response_sv = np.array(Statevector.from_instruction(response_circ))

    # mean_phase_diff = np.angle(np.average(expected_sv / response_sv))

    # for i in range(n_qubits):
    #     _global_phase_gate(response_circ, mean_phase_diff / 4, i)
    
    return response_circuit


def _initial_transpile(
        circuit,
        basis_gates=['rx', 'ry', 'rz', 'cx'],
        coupling_list=None,
        optimization_level=3
    ):
    layout = qiskit.transpiler.Layout(dict(zip(circuit.qubits, range(len(circuit.qubits)))))

    return qiskit.compiler.transpile(
        circuit,
        basis_gates=basis_gates,
        coupling_map=coupling_list,
        optimization_level=optimization_level,
        initial_layout=layout
    )


def _make_cnot_bridge(qubits_arr):
    ops_arr = []
    n_qubits = len(qubits_arr)
    for round in range(2):
        for i in range(n_qubits - 1):
            ops_arr.append(cirq.CNOT(qubits_arr[i], qubits_arr[i + 1]))
        for i in range(n_qubits - 3, 0, -1):
            ops_arr.append(cirq.CNOT(qubits_arr[i], qubits_arr[i + 1]))
    return ops_arr


def _make_grid_route(gq0, gq1):
    d_r = gq1.row - gq0.row
    d_c = gq1.col - gq0.col

    im_qubits = [gq0]

    # if gq0 is in the upper (smaller) row, move down first
    if d_r <= -1:
        for r in range(0, d_r, -1):
            im_qubits.append(cirq.GridQubit(gq0.row + r - 1, gq0.col))
        if d_c != 0:
            for c in range(0, d_c, np.sign(d_c)):
                im_qubits.append(cirq.GridQubit(gq1.row, gq0.col + c + np.sign(d_c)))
    # if gq0 is in the lower (greater) row, move sideways first
    elif d_r >= 1:
        if d_c != 0:
            for c in range(0, d_c, np.sign(d_c)):
                im_qubits.append(cirq.GridQubit(gq0.row, gq0.col + c + np.sign(d_c)))
        for r in range(d_r):
            im_qubits.append(cirq.GridQubit(gq0.row + r + 1, gq1.col))
    # same row
    else:
        for c in range(0, d_c, np.sign(d_c)):
            im_qubits.append(cirq.GridQubit(gq0.row, gq0.col + c + np.sign(d_c)))

    return im_qubits


def incrementer_decompose(n_qubits, unitary):
    response_circuit = QuantumCircuit(n_qubits)
    for i in reversed(range(1, n_qubits)):
        response_circuit.mcx(list(range(i)), i)
    response_circuit.x(0)
    qasm_string = response_circuit.qasm(filename=Path(DIR_PATH, RAW_QASM_FN))
    
    response_circuit = _initial_transpile(response_circuit)
    qasm_string = response_circuit.qasm(filename=Path(DIR_PATH, IR0_QASM_FN))

    # print(response_circuit.draw())

    qasm_string = _invert_qubit_order(n_qubits, DIR_PATH, IR0_QASM_FN, IR1_QASM_FN)

    return circuit_from_qasm(qasm_string)


def diagonal_decompose(n_qubits, unitary):
    response_circuit = qiskit.circuit.library.Diagonal(unitary.diagonal())
    qasm_string = response_circuit.qasm(filename=Path(DIR_PATH, RAW_QASM_FN))

    response_circuit = _initial_transpile(response_circuit)
    qasm_string = response_circuit.qasm(filename=Path(DIR_PATH, IR0_QASM_FN))

    qasm_string = _invert_qubit_order(n_qubits, DIR_PATH, IR0_QASM_FN, IR1_QASM_FN)

    return circuit_from_qasm(qasm_string)


def random_decompose(n_qubits, unitary):
    _write_backend_config(CONFIG_PATH)
    _openql_compile(DIR_PATH, CONFIG_PATH, n_qubits, unitary)
    qasm_string = _rewrite_openql_qasm(DIR_PATH, n_qubits)
    qasm_string = _invert_qubit_order(n_qubits, DIR_PATH, RAW_QASM_FN, IR0_QASM_FN)
    
    response_circuit = _make_response_circuit(n_qubits, qasm_string)
    response_circuit = _initial_transpile(response_circuit)

    qasm_string = response_circuit.qasm(filename=Path(DIR_PATH, IR1_QASM_FN))

    return circuit_from_qasm(qasm_string)


def _topology_rewrite(circuit, gqm):
    new_op_list = []
    for op in circuit.all_operations():
        if len(op.qubits) == 1:
            new_op_list.append(op)
        elif op.qubits[1] in gqm.coupling_map.get(op.qubits[0]):
            new_op_list.append(op)
        else:
            gq0 = gqm.mapping_dict[op.qubits[0]]
            gq1 = gqm.mapping_dict[op.qubits[1]]

            im_qubits = _make_grid_route(gq0, gq1)
            im_named_qubits = [gqm.inv_mapping_dict.get(q) for q in im_qubits]

            new_op_list.extend(_make_cnot_bridge(im_named_qubits))
    
    return cirq.Circuit(new_op_list)


class SmallUnitary(cirq.Gate):
    def __init__(self, n_qubits, unitary):
        super(SmallUnitary, self)
        self.__n_qubits = n_qubits
        self.__unitary = unitary

    def _num_qubits_(self):
        return self.__n_qubits

    def _unitary_(self):
        return self.__unitary

    def _circuit_diagram_info_(self, args):
        return ['SmallUnitary'] * self._num_qubits_()


# bin structure: qubit: [(q0, q1), [ops]]
class Binman():
    def __init__(self, qubits):
        self.__num_qubits = len(qubits)
        self.__qubits = qubits
        self.__bins_dict = {q: None for q in qubits}
        self.__ops_list = []

    @property
    def bins_dict(self):
        return copy.deepcopy(self.__bins_dict)
    
    @property
    def ops_list(self):
        return copy.deepcopy(self.__ops_list)

    def add_bin(self, q0, q1):
        q_bin = [tuple(sorted([q0, q1])), []]
        self.__bins_dict[q0] = q_bin
        self.__bins_dict[q1] = q_bin
    
    def which_bin(self, q):
        q_bin = self.__bins_dict.get(q)
        if q_bin is None: 
            return None
        else:
            return q_bin[1]
    
    def destroy_bin(self, q):
        q_bin = self.__bins_dict.get(q)
        if q_bin is not None:
            q0, q1 = q_bin[0]
            self.__bins_dict[q0] = None
            self.__bins_dict[q1] = None
    
    def compress_bin(self, q):
        q_bin = self.__bins_dict.get(q)
        if q_bin is not None:
            q0, q1 = q_bin[0]
            sub_circ = cirq.Circuit(q_bin[1])
            self.__ops_list.append(SmallUnitary(2, sub_circ.unitary()).on(q0, q1))
    
    def process_op(self, op):
        if len(op.qubits) == 1:
            q_bin = self.which_bin(op.qubits[0])
            if q_bin is not None:
                q_bin.append(op)
            else:
                self.__ops_list.append(op)
        elif len(op.qubits) == 2:
            q0, q1 = op.qubits
            q0_bin = self.which_bin(q0)
            q1_bin = self.which_bin(q1)
            if q0_bin is not None and q0_bin == q1_bin:
                q0_bin.append(op)
            # in two different bins, or q1 does not have a bin
            else:
                self.compress_bin(q0)
                self.compress_bin(q1)
                self.destroy_bin(q0)
                self.destroy_bin(q1)
                self.add_bin(q0, q1)
                self.which_bin(q0).append(op)
        else:
            raise NotImplementedError
    
    def close_all_bins(self):
        for q in self.__qubits:
            self.compress_bin(q)
            self.destroy_bin(q)
    
    def output_circuit(self):
        return cirq.Circuit(self.__ops_list)


def cirq_optimize(n_qubits, circuit, gqm, target='small_unitary'):
    if target == 'xmon':
        optm_circuit = cirq.google.optimized_for_sycamore(
            circuit,
            qubit_map=gqm.mapping,
            optimizer_type='xmon'
        )
    elif target == 'small_unitary':
        ir_circuit = cirq.google.optimized_for_sycamore(
            circuit,
            qubit_map=gqm.mapping,
            optimizer_type='xmon'
        )
        bm = Binman(ir_circuit.all_qubits())
        for op in ir_circuit.all_operations():
            bm.process_op(op)
        bm.close_all_bins()
        optm_circuit = bm.output_circuit()

    elif target == 'sycamore':
        ir_circuit = _topology_rewrite(circuit, gqm)
        optm_circuit = cirq.google.optimized_for_sycamore(
            ir_circuit,
            qubit_map=gqm.mapping,
            new_device=cirq.google.Sycamore,
            optimizer_type='sycamore'
        )
    elif target == 'fake_sycamore':
        ir_circuit = _topology_rewrite(circuit, gqm)
        optm_circuit = cirq.google.optimized_for_sycamore(
            ir_circuit,
            qubit_map=gqm.mapping,
            optimizer_type='sycamore'
        )
    elif target == 'small':
        optm_circuit = cirq.Circuit(SmallUnitary(n_qubits, circuit.unitary()).on(*gqm.grid_qubits))
    elif target == 'adaptive':
        process = lambda target: cirq_optimize(n_qubits, circuit, gqm, target=target)
        x_circ, f_circ, s_circ = Parallel(n_jobs=3)(delayed(process)(target) for target in ['small_unitary', 'fake_sycamore', 'sycamore'])
        # x_circ = process('small_unitary')
        # f_circ = process('fake_sycamore')
        # s_circ = process('sycamore')
        if n_two_qubit_gates(s_circ) < lb_two_qubit_gates(n_qubits):
            optm_circuit = s_circ
        elif n_two_qubit_gates(f_circ) < lb_two_qubit_gates(n_qubits) and n_two_qubit_gates(x_circ) < lb_two_qubit_gates(n_qubits):
            optm_circuit = f_circ
        else:
            optm_circuit = x_circ
    elif target == 'small+adaptive':
        process = lambda target: cirq_optimize(n_qubits, circuit, gqm, target=target)
        a_circ = process('adaptive')
        sm_sirc = process('small')
        if n_qubits == 1 or n_two_qubit_gates(a_circ) < lb_two_qubit_gates(n_qubits):
            optm_circuit = a_circ
        else:
            optm_circuit = sm_sirc
    else:
        raise NotImplementedError
    
    return optm_circuit
