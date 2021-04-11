"""
---------------------------- QCHACK 2021 Stanford x Yale ----------------------------

This is a submission from Team 33 for the Google Challenge at QC Hack 2021 by Stanford and Yale University

Team members:
- Ryan hill @ryanhill1
- Mathias Goncalves
- Vishal Sharathchandra Bajpe @mrvee-qC

Papers referenced for the solution:
[1] Decomposition of unitary matrices and quantum gates C.K. Li, R. Roberts, X. Yin, 2013.
[3] Decomposition of unitary matrix into quantum gates D. Fedoriaka, 2019.
[2] Efficient decomposition of unitary matrices in quantum circuit compilers A. M. Krol, A. Sarkar, I. Ashraf,
Z. Al-Ars, K. Bertels, 2021.

"""
from typing import List, Tuple

# !pip install cirq
import cirq
import math
import numpy as np


def is_power_of_two(x):
    """Checks if the value x is of the form 2^N

    Args:
        x: value to be check

    Returns:
        bool: True if value is a power of two

    """

    return (x & (x - 1)) == 0 and x != 0


def permute_matrix(m, perm):
    """Returns list of permutations perm applied to matrix A

    Args:
        m: Matrix to be permuted
        perm: Permutations to be applied

    Returns:
        A: Matrix list with permutations applied

    """
    m = np.array(m)
    m[:, :] = m[:, perm]
    m[:, :] = m[perm, :]
    return m


def two_level_decompose(m):
    """Decomposes a unitary matrix into two level operations, if possible and returns list of decomposed unitary
    matrices with indexing.

    Args:
        m: Matrix to be decomposed

    Returns:
        result: Returns list of decomposed unitary matrices
        idx: Returns the matrix indexes

     Raises:
        AssertionError:
            Matrix m is not a unitary matrix

    """

    assert cirq.is_unitary(m)
    n = m.shape[0]
    A = np.array(m, dtype=np.complex128)

    result = []
    idxs = []

    for i in range(n - 2):
        for j in range(n - 1, i, -1):
            a = A[i, j - 1]
            b = A[i, j]
            if abs(A[i, j]) < 1e-9:
                u_2x2 = np.eye(2, dtype=np.complex128)
                if j == i + 1:
                    u_2x2 = np.array([[1 / a, 0], [0, a]], dtype=np.complex128)
            elif abs(A[i, j - 1]) < 1e-9:
                u_2x2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
                if j == i + 1:
                    u_2x2 = np.array([[0, b], [1 / b, 0]], dtype=np.complex128)
            else:
                theta = np.arctan(np.abs(b / a))
                lmbda = -np.angle(a)
                mu = np.pi + np.angle(b) - np.angle(a) - lmbda
                u_2x2 = np.array([[np.cos(theta) * np.exp(1j * lmbda),
                                   np.sin(theta) * np.exp(1j * mu)],
                                  [-np.sin(theta) * np.exp(-1j * mu),
                                   np.cos(theta) * np.exp(-1j * lmbda)]], dtype=np.complex128)

            A[:, (j - 1, j)] = A[:, (j - 1, j)] @ u_2x2
            if not np.allclose(u_2x2, np.eye(2, dtype=np.complex128)):
                result.append(u_2x2.conj().T)
                idxs.append((j - 1, j))

    last_matrix = A[n - 2:n, n - 2:n]
    if not np.allclose(last_matrix, np.eye(2, dtype=np.complex128)):
        result.append(last_matrix)
        idxs.append((n - 2, n - 1))

    return result, idxs


def two_level_decompose_gray(m):
    """Decomposes a unitary matrix into two level operations, if possible and returns list matrices which multiply to
    A with indexing.

    Args:
        m: Matrix to be decomposed

    Returns:
        result: Returns list of decomposed matrices (single bit acting)
        idx: Returns the matrix indexes

    Raises:
        AssertionError:
            Matrix m must be a power of 2
            Matrix m must be a square matrix
            Matrix m is not a unitary matrix

    """

    n = m.shape[0]
    assert is_power_of_two(n)
    assert m.shape == (n, n), "Matrix must be square."
    assert cirq.is_unitary(m)

    perm = [x ^ (x // 2) for x in range(n)]  # Gray code.
    result, idxs = two_level_decompose(permute_matrix(m, perm))
    for i in range(len(idxs)):
        index1, index2 = idxs[i]
        idxs[i] = perm[index1], perm[index2]

    return result, idxs


def su_to_gates(m):
    """Decomposes two level special unitaries to Ry and Rz gates

    Args:
        m: Matrix to be converted into gates

    Returns:
        result: Returns list of gates to be applied

    Raises:
        AssertionError:
            Matrix m is not a special unitary matrix

    """
    assert cirq.is_special_unitary(m)

    u00 = m[0, 0]
    u01 = m[0, 1]
    theta = np.arccos(np.abs(u00))
    lmbda = np.angle(u00)
    mu = np.angle(u01)

    result = []
    if np.abs(lmbda - mu) > 1e-9:
        result.append(('Rz', lmbda - mu))
    if np.abs(theta) > 1e-9:
        result.append(('Ry', 2 * theta))
    if np.abs(lmbda + mu) > 1e-9:
        result.append(('Rz', lmbda + mu))

    return result


def unitary2x2_to_gates(m):
    """Decomposes a two level unitary to Ry, Rz and R1 gates

    Args:
        m: Matrix to be converted into gates

    Returns:
        result: Returns result (list of gates to be applied from function su_to_gates) with additional R1 gates

    Raises:
        AssertionError:
            Matrix m is not a unitary matrix

    """

    assert cirq.is_unitary(m)
    phi = np.angle(np.linalg.det(m))
    if np.abs(phi) < 1e-9:
        return su_to_gates(m)
    elif np.allclose(m, np.array([[0, 1], [1, 0]], dtype=np.complex128)):
        return [('X', 'n/a')]
    else:
        m = np.diag([1.0, np.exp(-1j * phi)]) @ m
        return su_to_gates(m) + [('R1', phi)]


def add_flips(flip_mask, gates):
    """Adds X gates for all qubits specified by qubit_mask.

    Args:
        gates: list of quantum gates to be applied
        flip_mask: int mask indicating whether qubit flip (X gate) is needed

    Returns:
        --

    """

    qubit_id = 0
    while flip_mask > 0:
        if (flip_mask % 2) == 1:
            gates.append(('Single', 'X', qubit_id))
        flip_mask //= 2
        qubit_id += 1


def matrix_to_gates(m):
    """ Returns list of gate sequences equivalent to the action of input matrix on respective qubit registers

    Args:
        m: Matrix to be converted into gate sequences (must be of the form 2^N * 2^N)

    Returns:
        gates: Returns sequence of gates as a list to be applied on a particular qubit

    Raises:
        AssertionError:
            Matrix m is not a unitary matrix
            Matrix m does not have dimensions of the order of 2^N

    """

    matrices, idxs = two_level_decompose_gray(m)

    gates = []
    prev_flip_mask = 0
    for i in range(len(matrices)):
        index1, index2 = idxs[i]
        qubit_id_mask = index1 ^ index2
        assert is_power_of_two(qubit_id_mask)
        qubit_id = int(math.log2(qubit_id_mask))

        flip_mask = (m.shape[0] - 1) - index2

        add_flips(flip_mask ^ prev_flip_mask, gates)
        for gate2 in unitary2x2_to_gates(matrices[i]):
            gates.append(('FC', gate2, qubit_id))
        prev_flip_mask = flip_mask
    add_flips(prev_flip_mask, gates)

    return gates


def gate_to_cirq(gate1):
    """Converts list of gate sequences to its cirq analogue

    Args:
        gate1: Sequence of gate to be transcribed

    Returns:
        --

    Raises:
        RuntimeError: Gate value passed cannot be implemented by cirq library

    """

    if gate1[0] == 'X':
        return cirq.X
    elif gate1[0] == 'Ry':
        return cirq.ry(-gate1[1])
    elif gate1[0] == 'Rz':
        return cirq.rz(-gate1[1])
    elif gate1[0] == 'R1':
        return cirq.ZPowGate(exponent=gate1[1] / np.pi)
    else:
        raise RuntimeError("Can't implement: %s" % gate1)


def matrix_to_cirq_circuit(m, qubits):
    """Converts unitary matrix to a cirq.circuit.Circuit list

    Args:
        m: Unitary matrix to be converted
        qubits: Target qubits specified to be acted on

    Returns:
        circuit: A mutable list of groups of operations to apply to some qubits.

    Raises:
        AssertionError:
            Matrix m is not a unitary matrix

        RuntimeError: Gate value passed cannot be implemented by cirq library

    """

    gates = matrix_to_gates(m)
    qubits_count = int(np.log2(m.shape[0]))
    circuit = cirq.Circuit()
    operations = []

    for gate in gates:
        if gate[0] == 'FC':
            controls = [qubits[i] for i in range(qubits_count) if i != gate[2]]
            target = qubits[gate[2]]
            arg_gates = controls + [target]
            # cgate = cirq.ControlledGate(
            #     gate_to_cirq(gate[1]),
            #     num_controls=qubits_count - 1)

            ops = cirq.decompose_multi_controlled_rotation(cirq.unitary(gate_to_cirq(gate[1])), controls, target)
            # print(ops)
            # circuit.append(cgate.on(*arg_gates))
            circuit.append(ops)
            operations.append(ops)

            # print(cirq.unitary(cgate).shape)
            # print(cirq.decompose_multi_controlled_rotation(cirq.unitary(cgate), controls=controls, target=target))
        elif gate[0] == 'Single':
            ops = gate_to_cirq(gate[1]).on(qubits[gate[2]])
            circuit.append(ops)
            operations.append(ops)
        else:
            raise RuntimeError('Unknown gate type.')

    return circuit, operations


def matrix_to_sycamore_operations(target_qubits: List[cirq.GridQubit], matrix: np.ndarray) -> \
        Tuple[cirq.OP_TREE, List[cirq.GridQubit]]:
    """ A method to convert a unitary matrix to a list of Sycamore operations.

    This method will return a list of `cirq.Operation`s using the qubits and (optionally) ancilla
    qubits to implement the unitary matrix `matrix` on the target qubits `qubits`.
    The operations are also supported by `cirq.google.gate_sets.SYC_GATESET`.

    Args:
        target_qubits: list of qubits the returned operations will act on. The qubit order defined by the list is
        assumed to be used by the operations to implement `matrix`.
        matrix: a matrix that is guaranteed to be unitary and of size (2**len(qs), 2**len(qs)).

    Returns:
        A tuple of operations and ancilla qubits allocated.
          Operations: In case the matrix is supported, a list of operations `ops` is returned. `ops` acts on `qs`
          qubits and for which `cirq.unitary(ops)` is equal to `matrix` up to certain tolerance. In case the matrix
          is not supported, it might return NotImplemented to reduce the noise in the judge output.

          Ancilla qubits: In case ancilla qubits are allocated a list of ancilla qubits. Otherwise an empty list.
        .
    """

    return matrix_to_cirq_circuit(matrix, target_qubits), []
