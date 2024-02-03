
import cirq
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from cirq.circuits import InsertStrategy
import scipy
from scipy import linalg
from cirq import protocols
from cirq.testing import gate_features
import random
N = 5000
PMS1 = 0.999
PMS2 = 0.99


T0 = 25

zZ = np.array([[1,0,0]]).T
eE = np.array([[0,1,0]]).T
fF = np.array([[0,0,1]]).T
A = [zZ, eE, fF]

B = []
for i1 in range(3):
    for i2 in range(3):
        for i3 in range(3):
            for i4 in range(3):
                B.append(np.kron(np.kron(np.kron(A[i1], A[i2]), A[i3]), A[i4]))






X = np.array([[0,1,0], [1,0,0], [0,0,1]])
Y = np.array([[0,complex(0,-1), 0], [complex(0,1), 0, 0], [0,0,1]])
Z = np.array([[1,0,0],[0,-1,0], [0,0,1]])
id = np.eye(3)

z = np.array([[1,0,0]]).T
e = np.array([[0,1,0]]).T
f = np.array([[0,0,1]]).T
basis = [z,e,f]
paulies1 = [id, X, Y, Z]

def dag(matrix):
    return np.conj(matrix.T)

def EE(bas, i, j, p0, paulies, pinv):
    v1 = bas[i]
    v2 = bas[j]
    id = paulies[0]
    x = paulies[1]
    y = paulies[2]
    z = paulies[3]
    K0 = (1-p0)**0.5  * id
    #K0 = id
    K1 = p0**0.5 / 3**0.5 * x
    #K1 = x
    K2 = p0**0.5 / 3**0.5 * y
    #K2 = y
    K3 = p0 ** 0.5 / 3**0.5 * z
    #K3 = z
    #mat_sum = K0 @ dag(K0) + K1 @ dag(K1) + K2 @ dag(K2) + K3 @ dag(K3)
    #print(mat_sum)
    #print(np.trace(mat_sum))
    #print()
    #print(dag(K3))
    _rho = v1 @ (v2.T)
    #print(_rho)
    if i == 0 and j == 0:
        ksgfsg = 0
        #print('eij', K0 @ _rho @ dag(K0) + K1 @ _rho @ dag(K1) + K2 @ _rho @ dag(K2) + K3 @ _rho @ dag(K3))
    #print('ee', np.trace(K0 @ _rho @ dag(K0) + K1 @ _rho @ dag(K1) + K2 @ _rho @ dag(K2) + K3 @ _rho @ dag(K3)))
    #print()
    return (K0 @ _rho @ dag(K0) + K1 @ _rho @ dag(K1) + K2 @ _rho @ dag(K2) + K3 @ _rho @ dag(K3))


def E(bas, i, j, p0, paulies):
    v1 = bas[i]
    v2 = bas[j]
    id = paulies[0]
    x = paulies[1]
    y = paulies[2]
    z = paulies[3]
    #K0 = (1-p0)**0.5 * id
    K0 = id / 2
    #K1 = p0**0.5 / 3**0.5 * x
    K1 = x / 2
    #K2 = p0**0.5 / 3**0.5 * y
    K2 = y / 2
    #K3 = p0 ** 0.5 / 3**0.5 * z
    K3 = z / 2
    #mat_sum = K0 @ dag(K0) + K1 @ dag(K1) + K2 @ dag(K2) + K3 @ dag(K3)
    #print(mat_sum)
    #print()
    #print(dag(K3))
    _rho = v1 @ (v2.T)

    #print(_rho)
    if i == 0 and j == 0:
        cgjjjfjk = 0
    #print('e', np.trace(K0 @ _rho @ dag(K0) + K1 @ _rho @ dag(K1) + K2 @ _rho @ dag(K2) + K3 @ _rho @ dag(K3)))
    #print()
    return K0 @ _rho @ dag(K0) + K1 @ _rho @ dag(K1) + K2 @ _rho @ dag(K2) + K3 @ _rho @ dag(K3)

def nice_repr(parameter):
    """Nice parameter representation
        SymPy symbol - as is
        float number - 3 digits after comma
    """
    if isinstance(parameter, float):
        return f'{parameter:.3f}'
    else:
        return f'{parameter}'


def levels_connectivity_check(l1, l2):
    """Check ion layers connectivity for gates"""
    connected_layers_list = [{0, i} for i in range(max(l1, l2) + 1)]
    assert {l1, l2} in connected_layers_list, "Layers are not connected"


def generalized_sigma(index, i, j, dimension=4):
    """Generalized sigma matrix for qudit gates implementation"""

    sigma = np.zeros((dimension, dimension), dtype='complex')

    if index == 0:
        # identity matrix elements
        sigma[i][i] = 1
        sigma[j][j] = 1
    elif index == 1:
        # sigma_x matrix elements
        sigma[i][j] = 1
        sigma[j][i] = 1
    elif index == 2:
        # sigma_y matrix elements
        sigma[i][j] = -1j
        sigma[j][i] = 1j
    elif index == 3:
        # sigma_z matrix elements
        sigma[i][i] = 1
        sigma[j][j] = -1

    return sigma


class QuditGate(cirq.Gate):
    """Base class for qudits gates"""

    def __init__(self, dimension=4, num_qubits=1):
        self.d = dimension
        self.n = num_qubits
        self.symbol = None

    def _num_qubits_(self):
        return self.n

    def _qid_shape_(self):
        return (self.d,) * self.n

    def _circuit_diagram_info_(self, args):
        return (self.symbol,) * self.n


class QuditRGate(QuditGate):
    """Rotation between two specified qudit levels: l1 and l2"""

    def __init__(self, l1, l2, theta, phi, dimension=4):
        super().__init__(dimension=dimension)
        levels_connectivity_check(l1, l2)
        self.l1 = l1
        self.l2 = l2
        self.theta = theta
        self.phi = phi

    def _unitary_(self):
        sigma_x = generalized_sigma(1, self.l1, self.l2, dimension=self.d)
        sigma_y = generalized_sigma(2, self.l1, self.l2, dimension=self.d)

        s = np.sin(self.phi)
        c = np.cos(self.phi)

        u = scipy.linalg.expm(-1j * self.theta / 2 * (c * sigma_x + s * sigma_y))

        return u

    def _is_parameterized_(self) -> bool:
        return cirq.protocols.is_parameterized(any((self.theta, self.phi)))

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool):
        return self.__class__(self.l1, self.l2, resolver.value_of(self.theta, recursive), resolver.value_of(self.phi, recursive), dimension=self.d)

    def _circuit_diagram_info_(self, args):
        self.symbol = 'R'
        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        return f'{self.symbol}{str(self.l1).translate(SUB)}{str(self.l2).translate(SUP)}' + f'({nice_repr(self.theta)}, {nice_repr(self.phi)})'


class QuditXXGate(QuditGate):
    """Two qudit rotation for two specified qudit levels: l1 and l2"""

    def __init__(self, l1, l2, theta, dimension=4):
        levels_connectivity_check(l1, l2)
        super().__init__(dimension=dimension, num_qubits=2)
        self.l1 = l1
        self.l2 = l2
        self.theta = theta

    def _unitary_(self):
        sigma_x = generalized_sigma(1, self.l1, self.l2, dimension=self.d)
        u = scipy.linalg.expm(-1j * self.theta / 2 * np.kron(sigma_x, sigma_x))

        return u

    def _is_parameterized_(self) -> bool:
        return cirq.protocols.is_parameterized(self.theta)

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool):
        return self.__class__(self.l1, self.l2, resolver.value_of(self.theta, recursive), dimension=self.d)

    def _circuit_diagram_info_(self, args):
        self.symbol = 'XX'
        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        info = f'{self.symbol}{str(self.l1).translate(SUB)}{str(self.l2).translate(SUP)}'.translate(
            SUB) + f'({nice_repr(self.theta)})'
        return info, info


class QuditZZGate(QuditGate):
    """Two qudit rotation for two specified qudit levels: l1 and l2"""

    def __init__(self, l1, l2, theta, dimension=4):
        levels_connectivity_check(l1, l2)
        super().__init__(dimension=dimension, num_qubits=2)
        self.l1 = l1
        self.l2 = l2
        self.theta = theta

    def _unitary_(self):
        sigma_z = generalized_sigma(3, self.l1, self.l2, dimension=self.d)
        u = scipy.linalg.expm(-1j * self.theta / 2 * np.kron(sigma_z, sigma_z))

        return u

    def _is_parameterized_(self) -> bool:
        return cirq.protocols.is_parameterized(self.theta)

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool):
        return self.__class__(self.l1, self.l2, resolver.value_of(self.theta, recursive), dimension=self.d)

    def _circuit_diagram_info_(self, args):
        self.symbol = 'ZZ'
        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        info = f'{self.symbol}{str(self.l1).translate(SUB)}{str(self.l2).translate(SUP)}'.translate(
            SUB) + f'({nice_repr(self.theta)})'
        return info, info


class QuditBarrier(QuditGate):
    """Just barrier for visual separation in circuit diagrams. Does nothing"""

    def __init__(self, dimension=4, num_qudits=2):
        super().__init__(dimension=dimension, num_qubits=num_qudits)
        self.symbol = '|'

    def _unitary_(self):
        return np.eye(self.d * self.d)


class QuditArbitraryUnitary(QuditGate):
    """Random unitary acts on qubits"""

    def __init__(self, dimension=4, num_qudits=2):
        super().__init__(dimension=dimension, num_qubits=num_qudits)
        self.unitary = np.array(scipy.stats.unitary_group.rvs(self.d ** self.n))
        self.symbol = 'U'

    def _unitary_(self):
        return self.unitary

'''
if __name__ == '__main__':
    n = 3  # number of qudits
    d = 4  # dimension of qudits

    qudits = cirq.LineQid.range(n, dimension=d)

    alpha = sympy.Symbol('alpha')
    beta = sympy.Symbol('beta')

    print('Qudit R Gate')
    circuit = cirq.Circuit(QuditRGate(0, 1, alpha, beta, dimension=d).on(qudits[0]))
    param_resolver = cirq.ParamResolver({'alpha': 0.2, 'beta': 0.3})
    resolved_circuit = cirq.resolve_parameters(circuit, param_resolver)
    print(resolved_circuit)
    print()

    print('Qudit XX Gate')
    circuit = cirq.Circuit(QuditXXGate(0, 2, beta, dimension=d).on(*qudits[:2]))
    param_resolver = cirq.ParamResolver({'alpha': 0.2, 'beta': 0.3})
    resolved_circuit = cirq.resolve_parameters(circuit, param_resolver)
    print(resolved_circuit)
    print()

    print('Qudit Barrier')
    circuit = cirq.Circuit(QuditBarrier(num_qudits=n, dimension=d).on(*qudits))
    print(circuit)
    print()

    print('Qudit Arbitrary Unitary Gate')
    circuit = cirq.Circuit(QuditArbitraryUnitary(num_qudits=n, dimension=d).on(*qudits))
    print(circuit)
'''




class QutritDepolarizingChannel(QuditGate):

    def __init__(self,PP, p_matrix=None):
        super().__init__(dimension=3, num_qubits=1)

        # Calculation of the parameter p based on average experimental error of single qudit gate
        f1 = 0.9
        self.p1 = (1 - f1) / (1 - 1 / self.d ** 2)
        self.p1 = PP
        #print(self.d)
        #print((1 / self.d ** 2))

        # Choi matrix initialization
        '''
        if p_matrix is None:
            self.p_matrix = (1 - self.p1) / (self.d ** 2) * np.ones((self.d, self.d))
            self.p_matrix = np.zeros_like(self.p_matrix)
            #self.p_matrix = np.ones((self.d, self.d))
        else:
            self.p_matrix = p_matrix
        #self.p_matrix[0, 0] += (1 - self.p1)  # identity probability
        for o in range(3):
            for oo in range(3):
                #self.p_matrix[o, oo] = 1 / np.trace(E(basis, o, oo, self.p1, paulies1))
                self.p_matrix[o, oo] = 1 / 9
        #self.p_matrix[0, 0] += 1
        '''

        if p_matrix is None:
            self.p_matrix = self.p1 / (self.d ** 2) * np.ones((self.d, self.d))
        else:
            self.p_matrix = p_matrix
        self.p_matrix[0, 0] += (1 - self.p1)  # identity probability
        self.p_matrix = np.array([[(1 - self.p1), self.p1 / 3], [self.p1 / 3, self.p1 / 3]])
        #print('prob[0,0]', self.p_matrix[0, 0])
        #print('prob_sum', self.p_matrix.sum())

        #print('prob_sum', self.p_matrix.sum())

    def _mixture_(self):
        ps = []
        for i in range(self.d):
            for j in range(self.d):
                pinv = np.linalg.inv(self.p_matrix)
                op = E(basis, i, j, self.p1, paulies1)
                #print(np.trace(op))
                ps.append(op)
        #print('total_sum', (np.trace(np.array(ps)) * self.p_matrix).sum())
        #chm = np.kron(np.ones(3), ps)
        X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        Y = np.array([[0, complex(0, -1), 0], [complex(0, 1), 0, 0], [0, 0, 1]])
        Z = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        id = np.eye(3)
        shiz_massiv = [id, X, Y, Z]
        return tuple(zip(self.p_matrix.flatten(), shiz_massiv))

    def _circuit_diagram_info_(self, args):
        return f"Φ(p1={self.p1:.3f})"



class QutritAmplitudeChannel(QuditGate):

    def __init__(self,PP, p_matrix=None):
        super().__init__(dimension=3, num_qubits=1)

        # Calculation of the parameter p based on average experimental error of single qudit gate
        f1 = 0.9
        self.p1 = (1 - f1) / (1 - 1 / self.d ** 2)
        self.p1 = PP
        #print(self.d)
        #print((1 / self.d ** 2))

        # Choi matrix initialization
        '''
        if p_matrix is None:
            self.p_matrix = (1 - self.p1) / (self.d ** 2) * np.ones((self.d, self.d))
            self.p_matrix = np.zeros_like(self.p_matrix)
            #self.p_matrix = np.ones((self.d, self.d))
        else:
            self.p_matrix = p_matrix
        #self.p_matrix[0, 0] += (1 - self.p1)  # identity probability
        for o in range(3):
            for oo in range(3):
                #self.p_matrix[o, oo] = 1 / np.trace(E(basis, o, oo, self.p1, paulies1))
                self.p_matrix[o, oo] = 1 / 9
        #self.p_matrix[0, 0] += 1
        '''

        if p_matrix is None:
            self.p_matrix = self.p1 / (self.d ** 2) * np.ones((self.d, self.d))
        else:
            self.p_matrix = p_matrix
        self.p_matrix[0, 0] += (1 - self.p1)  # identity probability
        self.p_matrix = np.array([[(1 - self.p1), self.p1 / 3], [self.p1 / 3, self.p1 / 3]])
        #print('prob[0,0]', self.p_matrix[0, 0])
        #print('prob_sum', self.p_matrix.sum())

        #print('prob_sum', self.p_matrix.sum())

    def _mixture_(self):
        ps = []
        for i in range(self.d):
            for j in range(self.d):
                pinv = np.linalg.inv(self.p_matrix)
                op = E(basis, i, j, self.p1, paulies1)
                #print(np.trace(op))
                ps.append(op)
        #print('total_sum', (np.trace(np.array(ps)) * self.p_matrix).sum())
        #chm = np.kron(np.ones(3), ps)
        X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        Y = np.array([[0, complex(0, -1), 0], [complex(0, 1), 0, 0], [0, 0, 1]])
        Z = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        Ea1 = np.array([[1, 0, 0], [0, (1-self.p1)**0.5, 0], [0, 0, (1-self.p1)**0.5]])
        Ea2 = np.array([[0, self.p1**0.5, 0], [0, 0, 0], [0, 0, 0]])
        Ea3 = np.array([[0, 0, self.p1**0.5], [0, 0, 0], [0, 0, 0]])
        id = np.eye(3)
        shiz_massiv = [Ea1, Ea2, Ea3]
        return tuple(zip(self.p_matrix.flatten(), shiz_massiv))

    def _circuit_diagram_info_(self, args):
        return f"Φ(p1={self.p1:.3f})"









def adde(circuit, gate, qud, ind):
    if ind == 1:
        primen = [gate[i](qud[i]) for i in range(len(qud))]
    else:
        primen = [gate[0](qud[0], qud[1])]
    circuit.append(primen, strategy=InsertStrategy.INLINE)

    error(circuit, qud, ind)


def R(fi, hi, i=0, j=1):
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    x01_for_ms = np.array([[0, 1, 0],
                           [1, 0, 0],
                           [0, 0, 0]])
    y01_for_ms = np.array([[0, complex(0, -1), 0],
                           [complex(0, 1), 0, 0],
                           [0, 0, 0]])
    x12_for_ms = np.array([[0, 0, 0],
                           [0, 0, 1],
                           [0, 1, 0]])
    y12_for_ms = np.array([[0, 0, 0],
                           [0, 0, complex(0, -1)],
                           [0, complex(0, 1), 0]])
    x02_for_ms = np.array([[0, 0, 1],
                           [0, 0, 0],
                           [1, 0, 0]])
    y02_for_ms = np.array([[0, 0, complex(0, -1)],
                           [0, 0, 0],
                           [complex(0, 1), 0, 0]])
    if (i, j) == (0, 1):
        x_for_ms = x01_for_ms
        y_for_ms = y01_for_ms
    elif (i, j) == (1, 2):
        x_for_ms = x12_for_ms
        y_for_ms = y12_for_ms
    else:
        x_for_ms = x02_for_ms
        y_for_ms = y02_for_ms
    m = np.cos(fi) * x_for_ms + np.sin(fi) * y_for_ms

    return linalg.expm(complex(0, -1) * m * hi / 2)


def make_ms_matrix(fi, hi, i=0, j=1, k=0, l=1):
    x_for_ms = np.array([[0, 1, 0],
                         [1, 0, 0],
                         [0, 0, 0]])
    y_for_ms = np.array([[0, complex(0, -1), 0],
                         [complex(0, 1), 0, 0],
                         [0, 0, 0]])
    m = np.kron((np.cos(fi) * x_for_ms + np.sin(fi) * y_for_ms), (np.cos(fi) * x_for_ms + np.sin(fi) * y_for_ms))
    m = complex(0, -1) * m * hi
    return linalg.expm(m)


class TwoQuditMSGate3_c(gate_features.TwoQubitGate
                        ):

    def _json_dict_(self):
        return {
            'cirq_type': self.__class__.__name__
        }

    @classmethod
    def _from_json_dict_(cls, **kwargs):
        return cls()

    def _qid_shape_(self):
        return (3, 3,)

    def _unitary_(self):
        matrix = make_ms_matrix(0, -np.pi / 2)
        return matrix

    def num_controls(self):
        return 2

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                               ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('XX0101_c',
                          'XX0101_c'))


class TwoQuditMSGate3(gate_features.TwoQubitGate
                      ):

    def _json_dict_(self):
        return {
            'cirq_type': self.__class__.__name__
        }

    @classmethod
    def _from_json_dict_(cls, **kwargs):
        return cls()

    def _qid_shape_(self):
        return (3, 3,)

    def _unitary_(self):
        matrix = make_ms_matrix(0, np.pi / 2)
        return matrix

    def num_controls(self):
        return 2

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                               ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=('XX0101',
                          'XX0101'))


class U(cirq.Gate):
    def __init__(self, mat, diag_i='R'):
        self.mat = mat
        self.diag_info = diag_i


    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return self.mat

    def _circuit_diagram_info_(self, args):
        return self.diag_info


def U1(cirquit, q1, q2):
    u1 = U(R(0, -np.pi, 1, 2), 'Rx(-π)12')
    u2 = U(R(np.pi / 2, np.pi / 2, 0, 1), 'Ry(π/2)01')
    u6 = U(R(np.pi / 2, -np.pi, 0, 2), 'Ry(-π)02')
    #cirquit.append([u1(q1), u6(q2)], strategy=InsertStrategy.INLINE)
    adde(cirquit, [u1, u6], [q1, q2], 1)
    #cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    adde(cirquit, [u2], [q1], 1)
    xx = TwoQuditMSGate3()
    #cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)
    #error(cirquit, [q1, q2], PMS)
    adde(cirquit, [xx], [q1, q2], 2)

def U1_clear(cirquit, q1, q2):
    u1 = U(R(0, -np.pi, 1, 2), 'Rx(-π)12')
    u2 = U(R(np.pi / 2, np.pi / 2, 0, 1), 'Ry(π/2)01')
    u6 = U(R(np.pi / 2, -np.pi, 0, 2), 'Ry(-π)02')
    cirquit.append([u1(q1), u6(q2)], strategy=InsertStrategy.INLINE)
    #adde(cirquit, [u1, u6], [q1, q2], 1)
    cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    #adde(cirquit, [u2], [q1], 1)
    xx = TwoQuditMSGate3()
    cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)
    #error(cirquit, [q1, q2], PMS)
    #adde(cirquit, [xx], [q1, q2], 2)



def U1_c(cirquit, q1, q2):
    u1 = U(R(0, np.pi, 1, 2), 'Rx(π)12')
    u2 = U(R(np.pi / 2, -np.pi / 2, 0, 1), 'Ry(-π/2)01')
    u6 = U(R(np.pi / 2, np.pi, 0, 2), 'Ry(π)02')
    xx_c = TwoQuditMSGate3_c()
    #cirquit.append([xx_c(q1, q2)], strategy=InsertStrategy.INLINE)
    adde(cirquit, [xx_c], [q1, q2], 2)
    #error(cirquit, [q1, q2], PMS)
    adde(cirquit, [u2], [q1], 1)
    #cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    adde(cirquit, [u1, u6], [q1, q2], 1)
    #cirquit.append([u1(q1), u6(q2)], strategy=InsertStrategy.INLINE)

def U1_c_clear(cirquit, q1, q2):
    u1 = U(R(0, np.pi, 1, 2), 'Rx(π)12')
    u2 = U(R(np.pi / 2, -np.pi / 2, 0, 1), 'Ry(-π/2)01')
    u6 = U(R(np.pi / 2, np.pi, 0, 2), 'Ry(π)02')
    xx_c = TwoQuditMSGate3_c()
    cirquit.append([xx_c(q1, q2)], strategy=InsertStrategy.INLINE)
    #adde(cirquit, [xx_c], [q1, q2], 2)
    #error(cirquit, [q1, q2], PMS)
    #adde(cirquit, [u2], [q1], 1)
    cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    #adde(cirquit, [u1, u6], [q1, q2], 1)
    cirquit.append([u1(q1), u6(q2)], strategy=InsertStrategy.INLINE)

def CX_clear(cirquit, q1, q2):
    u1 = U(R(0, -np.pi, 1, 2), 'Rx(-π)12')
    u2 = U(R(np.pi / 2, np.pi / 2, 0, 1), 'Ry(π/2)01')
    u3 = U(R(0, -np.pi, 0, 1), 'Rx(-π)01')
    u4 = U(R(np.pi / 2, -np.pi / 2, 0, 1), 'Ry(-π/2)01')
    u5 = U(R(0, np.pi, 1, 2), 'Rx(π)12')
    #adde(cirquit, [u1], [q1], 1)
    #adde(cirquit, [u2], [q1], 1)
    cirquit.append([u1(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    xx = TwoQuditMSGate3()
    #adde(cirquit, [xx], [q1, q2], 2)
    cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)
    #error(cirquit, [q1, q2], 2)
    #adde(cirquit, [u3, u3], [q1, q2], 1)
    #adde(cirquit, [u4], [q1], 1)
    #adde(cirquit, [u5], [q1], 1)
    cirquit.append([u3(q1), u3(q2)], strategy=InsertStrategy.INLINE)
    cirquit.append([u4(q1)], strategy=InsertStrategy.INLINE)
    cirquit.append([u5(q1)], strategy=InsertStrategy.INLINE)


def CX(cirquit, q1, q2):
    u1 = U(R(0, -np.pi, 1, 2), 'Rx(-π)12')
    u2 = U(R(np.pi / 2, np.pi / 2, 0, 1), 'Ry(π/2)01')
    u3 = U(R(0, -np.pi, 0, 1), 'Rx(-π)01')
    u4 = U(R(np.pi / 2, -np.pi / 2, 0, 1), 'Ry(-π/2)01')
    u5 = U(R(0, np.pi, 1, 2), 'Rx(π)12')
    adde(cirquit, [u1], [q1], 1)
    adde(cirquit, [u2], [q1], 1)
    #cirquit.append([u1(q1)], strategy=InsertStrategy.INLINE)
    #cirquit.append([u2(q1)], strategy=InsertStrategy.INLINE)
    xx = TwoQuditMSGate3()
    adde(cirquit, [xx], [q1, q2], 2)
    #cirquit.append([xx(q1, q2)], strategy=InsertStrategy.INLINE)
    #error(cirquit, [q1, q2], 2)
    adde(cirquit, [u3, u3], [q1, q2], 1)
    adde(cirquit, [u4], [q1], 1)
    adde(cirquit, [u5], [q1], 1)
    #cirquit.append([u3(q1), u3(q2)], strategy=InsertStrategy.INLINE)
    #cirquit.append([u4(q1)], strategy=InsertStrategy.INLINE)
    #cirquit.append([u5(q1)], strategy=InsertStrategy.INLINE)



def CCX(cirquit, q1, q2, q3):
    U1(cirquit, q1, q2)
    CX(cirquit, q2, q3)
    U1_c(cirquit, q1, q2)


def CZ(cirquit, q1, q2):
    h = H()
    #cirquit.append(h(q2), strategy=InsertStrategy.INLINE)
    adde(cirquit, [h], [q2], 1)
    CX(cirquit, q1, q2)
    #cirquit.append(h(q2), strategy=InsertStrategy.INLINE)
    adde(cirquit, [h], [q2], 1)

def CCZ(cirquit, q1, q2, q3):
    h = H()
    #cirquit.append(h(q3), strategy=InsertStrategy.INLINE)
    adde(cirquit, [h], [q3], 1)
    CCX(cirquit, q1, q2, q3)
    #cirquit.append(h(q3), strategy=InsertStrategy.INLINE)
    adde(cirquit, [h], [q3], 1)


class H(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return R(0, np.pi, 0, 1) @ R(np.pi / 2, np.pi / 2, 0, 1)

    def _circuit_diagram_info_(self, args):
        return 'H'


class X1_conj(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([[0, complex(0, -1), 0], [complex(0, -1), 0, 0], [0, 0, 1]])

    def _circuit_diagram_info_(self, args):
        return 'X1_c'


class X2_conj(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.conj(np.array([[0, 0, complex(0, -1)],
                                 [0, 1, 0],
                                 [complex(0, -1), 0, 0]]))

    def _circuit_diagram_info_(self, args):
        return 'X2_c'



class Z1(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return R(0, np.pi, 0, 1) @ R(np.pi / 2, np.pi, 0, 1)

    def _circuit_diagram_info_(self, args):
        return 'Z1'

class Y1(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return R(np.pi /2, np.pi, 0, 1)

    def _circuit_diagram_info_(self, args):
        return 'Y1'


class X2(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return R(0, np.pi, 0, 2)

    def _circuit_diagram_info_(self, args):
        return 'X2'


class X1(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return R(0, np.pi, 0, 1)

    def _circuit_diagram_info_(self, args):

        return 'X1'

def encoding_qubit(circuit, log_qubit):
    x = X1()
    h = H()
    q1, q2, q3, q4, q5 = log_qubit[0], log_qubit[1], log_qubit[2], log_qubit[3], log_qubit[4]
    #gates = [h(q2), h(q3), h(q4)]
    adde(circuit, [h, h, h], [q2, q3, q4], 1)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)
    CCZ(circuit, q1, q3, q4)
    #gates = [x(q3), x(q4)]
    adde(circuit, [x, x], [q3, q4], 1)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)
    CCZ(circuit, q4, q3, q1)
    #gates = [x(q3), x(q4)]
    adde(circuit, [x, x], [q3, q4], 1)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)
    CX(circuit, q1, q5)
    CX(circuit, q2, q5)
    CX(circuit, q2, q1)
    CX(circuit, q4, q1)
    CX(circuit, q3, q5)
    CZ(circuit, q4, q5)

def decoding_qubit(circuit, log_qubit):
    x = X1()
    h = H()
    q1, q2, q3, q4, q5 = log_qubit[0], log_qubit[1], log_qubit[2], log_qubit[3], log_qubit[4]
    CZ(circuit, q4, q5)
    CX(circuit, q3, q5)
    CX(circuit, q4, q1)
    CX(circuit, q2, q1)
    CX(circuit, q2, q5)
    CX(circuit, q1, q5)
    #gates = [x(q3), x(q4)]
    adde(circuit, [x, x], [q3, q4], 1)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)
    CCZ(circuit, q1, q3, q4)
    #gates = [x(q3), x(q4)]
    adde(circuit, [x, x], [q3, q4], 1)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)
    CCZ(circuit, q1, q3, q4)
    #gates = [h(q2), h(q3), h(q4)]
    adde(circuit, [h,h,h], [q2, q3, q4], 1)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)

def XZZXI(cirquit, qudits, a1):
    h = H()
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    CX(cirquit, a1, qudits[0])
    CZ(cirquit,  a1, qudits[1])
    CZ(cirquit, a1, qudits[2])
    CX(cirquit, a1, qudits[3])
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    cirquit.append([cirq.measure(a1)])

def ZZXIX(cirquit, qudits, a1):
    h = H()
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    CZ(cirquit,  a1, qudits[0])
    CZ(cirquit, a1, qudits[1])
    CX(cirquit, a1, qudits[2])
    CX(cirquit, a1, qudits[4])
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    cirquit.append([cirq.measure(a1)])

def XXIZX(cirquit, qudits, a1):
    h = H()
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    CX(cirquit, a1, qudits[0])
    CX(cirquit, a1, qudits[1])
    CZ(cirquit, a1, qudits[3])
    CX(cirquit, a1, qudits[4])
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    cirquit.append([cirq.measure(a1)])

def IXXXZ(cirquit, qudits, a1):
    h = H()
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    CX(cirquit, a1, qudits[1])
    CX(cirquit, a1, qudits[2])
    CX(cirquit, a1, qudits[3])
    CZ(cirquit, a1, qudits[4])
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    cirquit.append([cirq.measure(a1)])

def XZZXI_r(cirquit, qudits, a1):
    h = H()
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    CX(cirquit, a1, qudits[3])
    CZ(cirquit, a1, qudits[2])
    CZ(cirquit,  a1, qudits[1])
    CX(cirquit, a1, qudits[0])
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    cirquit.append([cirq.measure(a1)])

def ZZXIX_r(cirquit, qudits, a1):
    h = H()
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    CX(cirquit, a1, qudits[4])
    CX(cirquit, a1, qudits[2])
    CZ(cirquit, a1, qudits[1])
    CZ(cirquit, a1, qudits[0])
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    cirquit.append([cirq.measure(a1)])

def XXIZX_r(cirquit, qudits, a1):
    h = H()
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    CX(cirquit, a1, qudits[4])
    CZ(cirquit, a1, qudits[3])
    CX(cirquit, a1, qudits[1])
    CX(cirquit, a1, qudits[0])
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    # cirquit.append([cirq.measure(a1)])

def IXXXZ_r(cirquit, qudits, a1):
    h = H()
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    CZ(cirquit, a1, qudits[4])
    CX(cirquit, a1, qudits[3])
    CX(cirquit, a1, qudits[2])
    CX(cirquit, a1, qudits[1])
    cirquit.append([h(a1)], strategy=InsertStrategy.INLINE)
    cirquit.append([cirq.measure(a1)])

'''
def get_syndrome(circuit, qutrits):
    q0 = qutrits1[0]
    q1 = qutrits1[1]
    q2 = qutrits1[2]
    q3 = qutrits1[3]
    q4 = qutrits1[4]
    a1 = qutrits1[5]
    a2 = qutrits1[6]
    a3 = qutrits1[7]
    a4 = qutrits1[8]

    XZZXI(circuit1, [q0, q1, q2, q3, q4], qutrits1[5])
    res1 = sim.simulate(circuit1)
    measured_bit = res1.measurements[str(qutrits1[5])][0]
    print(f'Measured bit: {measured_bit}')

    ZZXIX(circuit1, [q0, q1, q2, q3, q4], qutrits1[6])
    res1 = sim.simulate(circuit1)
    measured_bit = res1.measurements[str(qutrits1[6])][0]
    print(f'Measured bit: {measured_bit}')

    XXIZX(circuit1, [q0, q1, q2, q3, q4], qutrits1[7])
    res1 = sim.simulate(circuit1)
    measured_bit = res1.measurements[str(qutrits1[7])][0]
    print(f'Measured bit: {measured_bit}')

    IXXXZ(circuit1, [q0, q1, q2, q3, q4], qutrits1[8])
    res1 = sim.simulate(circuit1)
    measured_bit = res1.measurements[str(qutrits1[8])][0]
    print(f'Measured bit: {measured_bit}')

def get_syndrome_r(circuit, qutrits):
    q0 = qutrits1[0]
    q1 = qutrits1[1]
    q2 = qutrits1[2]
    q3 = qutrits1[3]
    q4 = qutrits1[4]
    a1 = qutrits1[5]
    a2 = qutrits1[6]
    a3 = qutrits1[7]
    a4 = qutrits1[8]

    IXXXZ(circuit1, [q0, q1, q2, q3, q4], qutrits1[8])
    res1 = sim.simulate(circuit1)
    measured_bit = res1.measurements[str(qutrits1[8])][0]
    print(f'Measured bit: {measured_bit}')

    XXIZX(circuit1, [q0, q1, q2, q3, q4], qutrits1[7])
    res1 = sim.simulate(circuit1)
    measured_bit = res1.measurements[str(qutrits1[7])][0]
    print(f'Measured bit: {measured_bit}')
    ZZXIX(circuit1, [q0, q1, q2, q3, q4], qutrits1[6])
    res1 = sim.simulate(circuit1)
    measured_bit = res1.measurements[str(qutrits1[6])][0]
    print(f'Measured bit: {measured_bit}')
    XZZXI(circuit1, [q0, q1, q2, q3, q4], qutrits1[5])
    res1 = sim.simulate(circuit1)
    measured_bit = res1.measurements[str(qutrits1[5])][0]
    print(f'Measured bit: {measured_bit}')

'''

def CCCCX(cirquit, q1, q2, q3, q4, q5):
    U1_clear(cirquit, q1, q2)
    U1_clear(cirquit, q2, q3)
    U1_clear(cirquit, q3, q4)
    CX_clear(cirquit, q4, q5)
    U1_c_clear(cirquit, q3, q4)
    U1_c_clear(cirquit, q2, q3)
    U1_c_clear(cirquit, q1, q2)

def CCCCZ(cirquit, q1, q2, q3, q4, q5):
    h = H()
    cirquit.append(h(q5), strategy=InsertStrategy.INLINE)
    CCCCX(cirquit, q1, q2, q3, q4, q5)
    cirquit.append(h(q5), strategy=InsertStrategy.INLINE)

def CCCCY(cirquit, q1, q2, q3, q4, q5):
    h = H()
    CCCCZ(cirquit, q1, q2, q3, q4, q5)
    CCCCX(cirquit, q1, q2, q3, q4, q5)


'''
def ec(circuit, qutrits):
    circuit.append([cirq.measure(qutrits[1])])
    circuit.append([cirq.measure(qutrits[2])])
    circuit.append([cirq.measure(qutrits[3])])
    circuit.append([cirq.measure(qutrits[4])])
    res1 = sim.simulate(circuit1)
    r1,r2,r3,r4 = res1.measurements[str(qutrits1[1])][0], res1.measurements[str(qutrits1[2])][0], res1.measurements[str(qutrits1[3])][0], res1.measurements[str(qutrits1[4])][0]
    if r1 == 0 and r2 == 1 and r3 == 1 and r4 == 1:
        circuit.append([x(qutrits[0])], strategy=InsertStrategy.INLINE)
    elif r1 == 0 and r2 == 1 and r3 == 1 and r4 == 0:
        circuit.append([x(qutrits[0])], strategy=InsertStrategy.INLINE)
    elif r1 == 1 and r2 == 0 and r3 == 1 and r4 == 1:
        circuit.append([x(qutrits[0])], strategy=InsertStrategy.INLINE)
    elif r1 == 1 and r2 == 1 and r3 == 1 and r4 == 0:
        circuit.append([x(qutrits[0])], strategy=InsertStrategy.INLINE)
    elif r1 == 1 and r2 == 0 and r3 == 0 and r4 == 1:
        circuit.append([x(qutrits[0])], strategy=InsertStrategy.INLINE)

    elif r1 == 0 and r2 == 0 and r3 == 0 and r4 == 1:
        circuit.append([z(qutrits[0])], strategy=InsertStrategy.INLINE)
    elif r1 == 0 and r2 == 1 and r3 == 0 and r4 == 1:
        circuit.append([z(qutrits[0])], strategy=InsertStrategy.INLINE)
    elif r1 == 1 and r2 == 1 and r3 == 1 and r4 == 1:
        circuit.append([z(qutrits[0])], strategy=InsertStrategy.INLINE)
    elif r1 == 1 and r2 == 0 and r3 == 1 and r4 == 0:
        circuit.append([z(qutrits[0])], strategy=InsertStrategy.INLINE)
    elif r1 == 1 and r2 == 1 and r3 == 0 and r4 == 0:
        circuit.append([z(qutrits[0])], strategy=InsertStrategy.INLINE)


    elif r1 == 1 and r2 == 1 and r3 == 0 and r4 == 1:
        circuit.append([y(qutrits[0])], strategy=InsertStrategy.INLINE)

'''


def error_correction(circuit, qutrits1):
    '''
    #get_syndrome(circuit, qutrits)
    # get_syndrome_r(circuit1, qutrits1)
    circuit.append([cirq.measure(qutrits[1])])
    circuit.append([cirq.measure(qutrits[2])])
    circuit.append([cirq.measure(qutrits[3])])
    circuit.append([cirq.measure(qutrits[4])])
    '''
    q0 = qutrits1[0]
    q1 = qutrits1[1]
    q2 = qutrits1[2]
    q3 = qutrits1[3]
    q4 = qutrits1[4]


    # Операции для исправления ошибок X
    circuit.append([x(q1)], strategy=InsertStrategy.INLINE)
    CCCCX(circuit, q1, q2, q3, q4, q0)
    circuit.append([x(q1)], strategy=InsertStrategy.INLINE)

    circuit.append([x(q1), x(q4)], strategy=InsertStrategy.INLINE)
    CCCCX(circuit, q1, q2, q3, q4, q0)
    circuit.append([x(q1), x(q4)], strategy=InsertStrategy.INLINE)

    circuit.append([x(q1), x(q2), x(q3)], strategy=InsertStrategy.INLINE)
    CCCCZ(circuit, q1, q2, q3, q4, q0)
    circuit.append([x(q1), x(q2), x(q3)], strategy=InsertStrategy.INLINE)

    circuit.append([x(q2)], strategy=InsertStrategy.INLINE)
    CCCCX(circuit, q1, q2, q3, q4, q0)
    circuit.append([x(q2)], strategy=InsertStrategy.INLINE)


    # Операции для исправления ошибок Y
    circuit.append([x(q3)], strategy=InsertStrategy.INLINE)
    CCCCY(circuit, q1, q2, q3, q4, q0)
    circuit.append([x(q3)], strategy=InsertStrategy.INLINE)

    circuit.append([x(q4)], strategy=InsertStrategy.INLINE)
    CCCCX(circuit, q1, q2, q3, q4, q0)
    circuit.append([x(q4)], strategy=InsertStrategy.INLINE)

    circuit.append([x(q1), x(q3)], strategy=InsertStrategy.INLINE)
    CCCCZ(circuit, q1, q2, q3, q4, q0)
    circuit.append([x(q1), x(q3)], strategy=InsertStrategy.INLINE)

    circuit.append([x(q2), x(q3)], strategy=InsertStrategy.INLINE)
    CCCCX(circuit, q1, q2, q3, q4, q0)
    circuit.append([x(q2), x(q3)], strategy=InsertStrategy.INLINE)

    CCCCZ(circuit, q1, q2, q3, q4, q0)

    # Операции для исправления ошибок Z
    circuit.append([x(q2), x(q4)], strategy=InsertStrategy.INLINE)
    CCCCZ(circuit, q1, q2, q3, q4, q0)
    circuit.append([x(q2), x(q4)], strategy=InsertStrategy.INLINE)


    circuit.append([x(q3), x(q4)], strategy=InsertStrategy.INLINE)
    CCCCZ(circuit, q1, q2, q3, q4, q0)
    circuit.append([x(q3), x(q4)], strategy=InsertStrategy.INLINE)


def time_error(circuit, qutrits, t):

    p = np.exp(-t / T0)
    dpg_t = QutritDepolarizingChannel(1.000001 - p)
    for q in qutrits:
        circuit.append([dpg_t.on(q)], strategy=InsertStrategy.INLINE)
        '''
        if random.randint(0,1000) > p * 1000:
            a1 = random.randint(0, 1000)
            a2 = random.randint(0, 1000)
            a3 = random.randint(0, 1000)
            a4 = random.randint(0, 1000)
            if 1 - p == 0:
                p = 0.99999999
            sss = (a1 + a2 + a3 + a4) ** 0.5

            mx = R(0, np.pi, 0, 1)
            my = R(np.pi / 2, np.pi, 0, 1)
            mz = R(0, np.pi, 0, 1) @ R(np.pi / 2, np.pi, 0, 1)
            mi = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            mat = (a1 ** 0.5 * mi + a2 ** 0.5 * mx + a3 ** 0.5 * my + a4 ** 0.5 * mz) / sss
            er_gate = U(mat)
            circuit.append([er_gate(q)], strategy=InsertStrategy.INLINE)
        '''

def error(circuit, qutrits, ind):
    if ind == 1:
        p = PMS1
    else:
        p = PMS2
    dpg = QutritDepolarizingChannel(1.0001 - p)
    for q in qutrits:
        circuit.append([dpg.on(q)], strategy=InsertStrategy.INLINE)
    '''
    if ind == 1:
        p = PMS1
    else:
        p = PMS2

    for q in qutrits:
        rv_e = random.randint(0,1000)
        if rv_e > 1000 * p:
            a1 = random.randint(0, 1000)
            a2 = random.randint(0, 1000)
            a3 = random.randint(0, 1000)
            a4 = random.randint(0, 1000)
            sss = (a1 + a2 + a3 + a4) ** 0.5
            mx = R(0, np.pi, 0, 1)
            my = R(np.pi / 2, np.pi, 0, 1)
            mz = R(0, np.pi, 0, 1) @ R(np.pi / 2, np.pi, 0, 1)
            mi = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            mat = (a1 ** 0.5 * mi + a2 ** 0.5 * mx + a3 ** 0.5 * my + a4 ** 0.5 * mz) / sss
            er_gate = U(mat)
            circuit.append([er_gate(q)], strategy=InsertStrategy.INLINE)
    '''

def X1_l(circuit, lqubits):
    x = X1()
    z = Z1()
    q1, q2, q3, q4, q5 = lqubits[0], lqubits[1], lqubits[2], lqubits[3], lqubits[4]
    #gates = [z(q1), z(q4)]
    adde(circuit, [z,z], [q1, q4], 1)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)
    #gates = [x(q1), x(q2) ,x(q3) ,x(q4) ,x(q5)]
    adde(circuit, [x, x, x, x, x], [q1, q2, q3, q4, q5], 1)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)

def Z1_l(circuit, lqubits):
    x = X1()
    z = Z1()
    q1, q2, q3, q4, q5 = lqubits[0], lqubits[1], lqubits[2], lqubits[3], lqubits[4]
    #gates = [x(q1), x(q4)]
    adde(circuit, [x, x], [q1, q4], 1)
    #circuit.append(gates, strategy=InsertStrategy.INLINE)
    adde(circuit, [z], [q5], 1)
    #gates = [z(q5)]
    #circuit.append(gates, strategy=InsertStrategy.INLINE)


# def make_error(p):


# Основная операция
x = X1()
x2 = X2()
z = Z1()
y = Y1()
x_conj = X1_conj()
x2_conj = X2_conj()
h = H()
'''
sim = cirq.Simulator()
circuit1 = cirq.Circuit()
qutrits1 = []



for i in range(10):
    qutrits1.append(cirq.LineQid(i, dimension=3))
'''
# кодируемое состояние
#gates1 = [h(qutrits1[0])]
#circuit1.append(gates1)
#encoding_qubit(circuit1, qutrits1)
# ошибка
#gates1 = [z(qutrits1[4])]
# circuit1.append(gates1)
#xxx = TwoQuditMSGate3()
#adde(circuit1, [xxx], [qutrits1[1], qutrits1[2]], 2)
#adde(circuit1, [h, h, h], [qutrits1[3], qutrits1[2], qutrits1[4]], 1)
# error_correction(circuit1, qutrits1)

# error(circuit1, qutrits1, 0.5)
#decoding_qubit(circuit1, qutrits1)
# circuit1.append([cirq.measure(qutrits1[1])])
# circuit1.append([cirq.measure(qutrits1[2])])
# circuit1.append([cirq.measure(qutrits1[3])])
# circuit1.append([cirq.measure(qutrits1[4])])
'''
res1 = sim.simulate(circuit1)
measured_bit = res1.measurements[str(qutrits1[1])][0]
print(f'Measured bit: {measured_bit}')
res1 = sim.simulate(circuit1)
measured_bit = res1.measurements[str(qutrits1[2])][0]
print(f'Measured bit: {measured_bit}')
res1 = sim.simulate(circuit1)
measured_bit = res1.measurements[str(qutrits1[3])][0]
print(f'Measured bit: {measured_bit}')
res1 = sim.simulate(circuit1)
measured_bit = res1.measurements[str(qutirts1[4])][0]
print(f'Measured bit: {measured_bit}')
'''
def m(a ,b, c, d, e):
    return np.kron(np.kron(np.kron(np.kron(a, b), c), d), e)

def partial_trace(rho_ab):
    tr = np.eye(3) - np.eye(3)
    for i in range(3):
        for j in range(3):
            for k in range(81):
                tr = tr + np.kron(A[i].T, B[k].T) @ rho_ab @ np.kron(A[j], B[k]) * A[i] @ A[j].T
    return tr

sps1 = []
sps2 = []

def run_circit(t, N):

    fidelity = 0
    sch = 0
    for alf1 in np.linspace(0, 2 * np.pi, N):
        for alf2 in np.linspace(0, np.pi, N//2):
            alf1 = random.randint(0, 1000) / 1000 * 2 * np.pi
            alf2 = random.randint(0, 1000) / 1000 * 2 * np.pi
            sch += 1
            x = X1()
            y = Y1()
            sim = cirq.Simulator()
            circuit1 = cirq.Circuit()
            qutrits1 = []
            for j in range(5):
                qutrits1.append(cirq.LineQid(j, dimension=3))


            povorot = R(alf1, alf2, 0, 1)
#!

            pg = U(povorot)
            circuit1.append([pg(qutrits1[0])], strategy=InsertStrategy.INLINE)

            encoding_qubit(circuit1, qutrits1)
            time_error(circuit1, qutrits1, t)
            #circuit1.append([y(qutrits1[2])])
            decoding_qubit(circuit1, qutrits1)
            error_correction(circuit1, qutrits1)

#!

            povorot_r = R(alf1, -alf2, 0, 1)
            pg_r = U(povorot_r)
            circuit1.append([pg_r(qutrits1[0])], strategy=InsertStrategy.INLINE)



            ro_ab = cirq.final_density_matrix(circuit1, qubit_order = qutrits1)

            mat_0 = partial_trace(np.array(ro_ab))
            #print(mat_0)
            fidelity += mat_0[0][0]
    return fidelity / sch

def run_single_qudit(t, N):
    fidelity = 0
    sch = 0
    for alf1 in np.linspace(0, np.pi, N // 2):
        for alf2 in np.linspace(0, 2 * np.pi, N):
            alf2 += 2 * np.pi / N / 2
            alf1 = random.randint(0, 1000) / 1000 * 2 * np.pi
            alf2 = random.randint(0, 1000) / 1000 * 2 * np.pi
            sch += 1
            circuit1 = cirq.Circuit()
            qutrits1 = []
            qutrits1.append(cirq.LineQid(0, dimension=3))

            povorot = R(alf1, alf2, 0, 1)
            # !

            pg = U(povorot)
            #circuit1.append([h(qutrits1[0])], strategy=InsertStrategy.INLINE)
            circuit1.append([pg(qutrits1[0])], strategy=InsertStrategy.INLINE)
            #print(cirq.final_density_matrix(circuit1, qubit_order=qutrits1))
            #print()

            time_error(circuit1, qutrits1, t)

            povorot_r = R(alf1, -alf2, 0, 1)
            pg_r = U(povorot_r)
            circuit1.append([pg_r(qutrits1[0])], strategy=InsertStrategy.INLINE)
            #circuit1.append([h(qutrits1[0])], strategy=InsertStrategy.INLINE)

            ro_ab = cirq.final_density_matrix(circuit1)


            # print(mat_0)
            fidelity += ro_ab[0][0]
    return fidelity / sch

def main(T, k, N):
    code_line = []
    single_qudit_line = []
    for t_ in np.linspace(0, T, k):
        print(t_)
        code_line.append(run_circit(t_, N))
        #code_line.append(0.5)
        single_qudit_line.append(run_single_qudit(t_, N))
    '''
    print(code_line)
    print(single_qudit_line)
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot()
    ax.scatter(np.linspace(0, T, k), single_qudit_line, color='b', s=5, label='без коррекции')
    ax.scatter(np.linspace(0, T, k), code_line, color='r', s=5, label='c коррекции')

    ax.set_xlabel('t, mcs (T0 = 25 mcs)')
    ax.set_ylabel('fid')
    plt.title('P1 = 0.999, P2 = 0.99')
    plt.legend()
    plt.grid()

    plt.show()
    '''
    return code_line, single_qudit_line, np.linspace(0, T, k)

def graph(c,s,t):
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot()
    ax.scatter(t, s, color='b', s=5, label='без коррекции')
    ax.scatter(t, c, color='r', s=5, label='c коррекции')

    ax.set_xlabel('t, mcs (T0 = 25 mcs)')
    ax.set_ylabel('fid')
    plt.title('P1 = 0.999, P2 = 0.99')
    plt.legend()
    plt.grid()

    plt.show()

PMS1 = 0.99
#PMS2 = 0.99
#N = 3
#T = 100
#c = [(0.8565517202725534+3.395137123470405e-08j), (0.7195716415300618+2.1850220699353334e-10j), (0.599032604033861-2.345406221093399e-09j), (0.5222457941217726-1.7974584144362756e-09j), (0.4899097485445054+1.6283321192778794e-09j), (0.4898734727162264-5.299983852578137e-10j), (0.48510653972061846+2.123842613722725e-09j), (0.48112311625148624+7.388549248671417e-10j), (0.4902382209378023+2.910851000013422e-09j), (0.48332990607195825-3.3510880924009664e-09j), (0.4806859651293962-9.44438148354458e-10j), (0.48183611255929765-4.409083452501381e-09j), (0.4760857342883658-1.1525566063163228e-09j), (0.4722917696447742+2.55770145740292e-10j), (0.47302660028390164+1.5277126690625012e-09j), (0.4654884947155248-5.280128293801107e-09j), (0.47168131055317036-1.2135048989432287e-09j), (0.4688896764914716+2.324239438993144e-09j), (0.4635384561408197-5.114246990133735e-09j), (0.45816098383769105+7.744124710456607e-10j)]
#s = [(0.9999993046124775-1.546850529130041e-08j), (0.8734378417332966+3.945847797205134e-09j), (0.7709030310312907+1.0565259047239592e-18j), (0.6878337264060974+4.867137063769291e-09j), (0.6205344001452128+2.5756173489889494e-09j), (0.5660114089647929-3.2946397091985582e-09j), (0.5218391418457031+1.094105907723361e-09j), (0.48605262239774066-7.872636140215938e-09j), (0.45705993970235187-3.92671936633171e-11j), (0.4335712492465973-2.9606396975227303e-09j), (0.41454172134399414+8.409256670634708e-10j), (0.39912482102711994+6.954834856169137e-11j), (0.3866346478462219-1.0067211185053417e-09j), (0.3765157163143158+1.4064110612945127e-09j), (0.3683176736036936+3.3446373076713562e-09j), (0.36167605717976886-3.457652571666377e-09j), (0.35629530747731525+4.0823362121300555e-09j), (0.3519360323746999+3.570069869359334e-09j), (0.3484043478965759-1.5906890619514038e-09j), (0.3455430765946706-2.5427838538464676e-10j)]
#t = np.linspace(0, 100, 20)
#main(T, 20, N)
#c, s, t = main(T, 20, N)

def graph_3d(ms1, ms2, kms, T, N, k):
    code_surf = []
    single_qudit_surf = []
    for ms in np.linspace(ms1, ms2, kms):
        PMS2 = ms
        cl, sl, tl = main(T,k,N)
        code_surf.append(cl)
        single_qudit_surf.append(sl)
    print(code_surf)
    print()
    print(single_qudit_surf)
    code_surf = np.array(code_surf)
    single_qudit_surf = np.array(single_qudit_surf)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    T, MS = np.meshgrid(np.linspace(0,T,k), np.linspace(ms1, ms2, kms))
    ax.plot_wireframe(T, MS, code_surf, color = 'r')
    ax.plot_wireframe(T, MS, single_qudit_surf, color='b')
    plt.show()

'''

ms1, ms2, kms, T, N, k = 0.9,1,20, 100, 2, 20
code_surf = []
single_qudit_surf = []
for ms in np.linspace(ms1, ms2, kms):
    PMS2 = ms
    cl, sl, tl = main(T,k,N)
    code_surf.append(cl)
    single_qudit_surf.append(sl)
print(code_surf)
print()
print(single_qudit_surf)
code_surf = np.array(code_surf)
single_qudit_surf = np.array(single_qudit_surf)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
T, MS = np.meshgrid(np.linspace(0,T,k), np.linspace(ms1, ms2, kms))
ax.plot_wireframe(T, MS, code_surf, color = 'r')
ax.plot_wireframe(T, MS, single_qudit_surf, color='b')
plt.show()
'''

cs = np.array([[(0.40646447538165376+1.949782557142484e-09j), (0.4346093237400055-2.356950194662039e-10j), (0.4039295100956224+1.0974464715419805e-10j), (0.40801449870923534-6.719451659277042e-10j), (0.39478514075744897-5.355242920419393e-10j), (0.4231309476890601-6.257925290414942e-10j), (0.3936230191611685+2.864057320769087e-10j), (0.3689566606772132-5.286170869158083e-10j), (0.37357087596319616+1.1826238341486775e-09j), (0.3775536030298099+3.897093205167515e-10j), (0.39938682434149086-2.169119994759434e-09j), (0.43758267303928733+1.3848786843943951e-09j), (0.4048577412031591+2.5533473112598e-09j), (0.4065297738998197+1.027563256628263e-09j), (0.4164427744690329-6.039975325355159e-10j), (0.4168452605372295+1.2290511107409702e-09j), (0.38759166188538074-2.9371913792449e-09j), (0.3793499371386133+1.0591361101815569e-10j), (0.4166476971004158-2.712720714695266e-09j), (0.4168987662997097-5.968232573172766e-10j)], [(0.4603690844960511-6.412650757757475e-09j), (0.4587688036262989-1.3568625463382135e-09j), (0.37829225044697523+1.9996317936340257e-09j), (0.39715484163025394-4.40278125064391e-09j), (0.404340774693992-1.3970578027547867e-09j), (0.39170807891059667-1.6160659480779612e-09j), (0.41311119589954615-7.963184487953203e-10j), (0.44026599737117067+3.7989763568633307e-10j), (0.4116287280921824+1.1705914299300533e-09j), (0.3770142531138845-4.768936150037083e-10j), (0.40021976217394695+1.9862550909466776e-09j), (0.4311069755931385+1.2326997756583473e-09j), (0.41183589375577867-5.849843838083334e-10j), (0.38695709977764636+5.310101301851118e-11j), (0.39370169589528814-7.560796434997716e-10j), (0.3862380536738783+1.3728836114334484e-10j), (0.4098314273869619+8.088194837285262e-10j), (0.40383011143421754+2.1683841389105826e-09j), (0.4043603159952909+1.0453573562062877e-09j), (0.41678759292699397+9.526872627231873e-10j)], [(0.47680025291629136+9.066815078673731e-10j), (0.3923743258055765-1.7291426067645524e-09j), (0.41147872112924233+3.0930278337401683e-09j), (0.42878526845015585-7.228213027781303e-10j), (0.4454627587692812-8.173150295949446e-10j), (0.40995706274406984+9.66945135096557e-10j), (0.4332875575637445+1.1095019834339407e-09j), (0.4109244800056331-9.570478933333226e-11j), (0.39321497242781334+1.9041031538344e-09j), (0.41553122585173696+1.7016808935831527e-09j), (0.44320495112333447+2.1465505205658536e-09j), (0.37258100119652227+6.0047421180833e-10j), (0.4328438179800287+1.2138719687043994e-09j), (0.39883945920155384+1.196172327922735e-09j), (0.39961632364429533-2.498773743495397e-09j), (0.3797127684520092+7.019183568573809e-10j), (0.3810075960645918+6.827333837443878e-10j), (0.410268884152174+1.0125586481520674e-09j), (0.40582912566605955+1.9531272157399205e-09j), (0.429477215919178+7.704416150046787e-10j)], [(0.4724881036381703+4.316792256803938e-09j), (0.397931304294616-8.84570694705208e-10j), (0.42976331437239423-1.0986553336739412e-09j), (0.39355577845708467+3.9382497261948864e-09j), (0.4067768156528473-1.3689888112084407e-09j), (0.3777256909525022-6.30633403812012e-10j), (0.39040415547788143-3.01137397303162e-10j), (0.41139193225535564-1.6683680748522303e-09j), (0.40742820099694654-1.9879141666399078e-09j), (0.4299364443286322+4.56970897235272e-09j), (0.42584637395339087+8.260589432933654e-10j), (0.3921753661124967-1.9701729690300602e-09j), (0.384740260313265-3.7129214594227044e-09j), (0.40177052485523745+1.538863258866819e-09j), (0.40577361575560644-2.3713058182486e-09j), (0.39977470348821953-3.100803527219212e-09j), (0.4110770290135406+1.5846560175253842e-09j), (0.39640633371891454-1.2308280534619027e-12j), (0.4251743286731653-8.722444979356566e-11j), (0.4129206249199342-2.0324616940599565e-10j)], [(0.4287443759967573+1.4929279978039148e-11j), (0.45017824036767706-1.9629247396756535e-09j), (0.4530842445092276+3.296194872958438e-09j), (0.39740637625800446-8.797251036308403e-10j), (0.41065506386803463+1.5788660379152589e-09j), (0.4248182426090352+1.915569147932043e-09j), (0.4149425973300822-1.3852235121409478e-09j), (0.4033580193936359+2.7616316318918387e-09j), (0.42225327383494005+1.7431668198084454e-09j), (0.4449216205975972-1.703181422469911e-09j), (0.3938670679635834-3.013993497317423e-10j), (0.388310567883309-1.2010796801622355e-09j), (0.43228369212010875-8.987516557766676e-10j), (0.3807969229237642+1.9637902561878895e-10j), (0.40207501366967335+3.37073785536783e-09j), (0.4167054308927618+3.814004667821634e-09j), (0.4002585801354144+2.446038090896387e-09j), (0.4303837029146962-1.2765808408746308e-09j), (0.3967903334123548+2.3017394657481344e-09j), (0.41980897646863014+2.9067303984626775e-11j)], [(0.4407654279493727+6.6460223991543204e-09j), (0.45407585936482064-4.899014530580857e-10j), (0.41914294127491303-6.785594346171997e-11j), (0.4321016071771737-1.1060595778735134e-09j), (0.3981755619170144-2.668082387192883e-09j), (0.40738566397340037+2.8781240462592357e-10j), (0.4132399578811601-1.784526900240213e-10j), (0.43478326668264344-1.2524294021541797e-09j), (0.4169050650962163-4.1028555413989395e-09j), (0.4045347274513915+2.0101175171791423e-09j), (0.44847143872175366-1.7901511372402874e-10j), (0.4386463266564533+1.1574950442052088e-09j), (0.39538262999849394-8.424263230965193e-10j), (0.4099400107515976+4.848295098451972e-10j), (0.3942832263710443+6.302047774228745e-10j), (0.41829804205917753-2.8539057783000143e-09j), (0.38723617140203714-3.8065994787149336e-10j), (0.38329861240345053-6.445012299865917e-10j), (0.42091708281077445-2.6131878887307857e-09j), (0.398779904760886+1.5474971305758746e-09j)], [(0.5165518499561585-1.5353115173266744e-09j), (0.4971804771339521+8.248752848441826e-09j), (0.4524991920334287-1.306510674330387e-12j), (0.4174719273869414-7.633342845704737e-10j), (0.43732022793847136-2.2320744145665814e-09j), (0.42722231478546746-2.101480230274414e-09j), (0.4309578419197351-5.212171821762284e-10j), (0.39360892586410046-1.4905313320245381e-09j), (0.43470675725257024+1.778650525487857e-09j), (0.4315440631180536-5.552697729619131e-10j), (0.41615458225714974+1.396601334685241e-09j), (0.4015933560440317-7.408733429003183e-10j), (0.3972705166670494+5.962095265007878e-10j), (0.416523119318299-2.1186510325014394e-09j), (0.4222395774559118+1.6552146322779184e-09j), (0.39810725999996066+4.046476488359537e-10j), (0.4060313611989841+2.411614352671772e-09j), (0.3960011369199492+7.773693357356177e-10j), (0.3949565274233464-1.8493931370948847e-10j), (0.419269266189076-1.0255546369856777e-10j)], [(0.51301144019817-1.9710455493879173e-09j), (0.48921879625413567+3.699578865299722e-09j), (0.40707585198106244-5.631609071394385e-10j), (0.44720059545943514-1.9642079078116496e-09j), (0.4559990401030518+7.142289155714976e-10j), (0.41255159501452+2.7944005623164366e-10j), (0.4254875839978922-1.1979110642887909e-09j), (0.3955773736233823-4.0503052439587573e-10j), (0.4427835848473478+5.357811282442581e-10j), (0.4014656686631497+1.7793655438608862e-10j), (0.4213382338930387-3.788767077754192e-10j), (0.4245370542339515+8.799263534986273e-10j), (0.42496221329201944-5.677403133350488e-10j), (0.41720766000798903+3.738680393153697e-09j), (0.4180496388289612+4.3518886272701884e-09j), (0.40705121573409997-9.180423646247582e-10j), (0.44023069157265127+3.800785041061616e-09j), (0.4153005451662466-2.4462580846207367e-09j), (0.4241874498256948+3.140272485960398e-10j), (0.396776496025268-6.549916188790048e-11j)], [(0.5276658600923838-5.41751121745356e-09j), (0.4761913654219825-1.9626213988359154e-09j), (0.45770257245749235+3.0452272926429822e-09j), (0.43841497210087255-3.487022448638751e-09j), (0.42769637765013613+2.4504354099751615e-09j), (0.41577361058443785-2.284992639629091e-09j), (0.43107759454869665+3.293842560022606e-10j), (0.40444627893157303+1.7601795107630244e-10j), (0.44883916145772673-4.443545661505152e-09j), (0.3965674020291772+3.485470661779314e-10j), (0.4292611246346496-2.122552717243106e-09j), (0.4086554395908024-1.1689365448982052e-09j), (0.4035837879637256-1.472420657804305e-10j), (0.41998858549050055+3.5707137240221836e-10j), (0.4202306145161856-2.5197754992971657e-10j), (0.4418152311409358+1.742612276673799e-09j), (0.43304459570208564+2.557776213055016e-11j), (0.4395025482517667+1.2767525920080876e-09j), (0.4299365149636287+4.757026328640943e-09j), (0.4052120980049949-4.237567630448794e-09j)], [(0.474119284626795+3.633091605081768e-10j), (0.46259508025832474-2.5664229629388832e-09j), (0.4278991803585086+3.3954830059030517e-09j), (0.4363883239857387-1.0462268271508814e-09j), (0.41078835912048817+1.2848093142620802e-09j), (0.4273523987212684-1.6797400133336218e-10j), (0.41242206892638933-5.4223152981521e-09j), (0.4540787332516629+9.052096452535592e-10j), (0.45739691928611137+7.997542678099086e-10j), (0.42988821129256394-1.9444231224924983e-09j), (0.43732162375818007+5.418564485622695e-10j), (0.42335999039642047+9.097230049452024e-10j), (0.4141825025435537-7.65621028474451e-10j), (0.43640413126558997-6.416584485819649e-10j), (0.4265215945924865+1.2188661624235565e-09j), (0.3974140864011133-1.0335010645085552e-10j), (0.41268362346454524+2.7285160792560597e-09j), (0.4348682672716677-4.731219860413701e-10j), (0.4092097994289361-2.755136323875686e-09j), (0.45293908211169764-7.875699122455759e-10j)], [(0.5615002640988678+4.308690252756623e-09j), (0.5160105275572278-8.242984078906372e-10j), (0.45516398041218054-8.282512452889641e-10j), (0.4422264608583646+4.841800737764308e-09j), (0.41319150051276665-2.808569377088981e-09j), (0.4344443841109751-1.346468491508161e-10j), (0.4399049067287706-3.18129216553347e-10j), (0.4517763591138646+1.1503807023584614e-09j), (0.41961569151317235-2.051764592336189e-09j), (0.41317802856792696-2.080397469081616e-09j), (0.43181514988827985+2.5375264388729295e-10j), (0.459166563290637-3.5684870665834357e-10j), (0.42164402223716024+3.316718844235696e-09j), (0.4184394290932687+1.040363802104688e-09j), (0.43460054966271855+3.627040889853106e-09j), (0.42771571705816314+1.0418710341444622e-09j), (0.4333148727455409-2.7571719888261967e-09j), (0.4558992679230869+1.6952406140224217e-10j), (0.43802041086019017-1.7350911542209236e-09j), (0.41947462399548385+2.262562248761669e-09j)], [(0.5855257219227497+5.886214847604555e-10j), (0.4976321025606012-5.173421690820646e-09j), (0.4446670222823741-4.8347743588244116e-09j), (0.44745266290556174+8.546040600051621e-11j), (0.4453364537184825-6.943464096178355e-09j), (0.4531908554636175+1.0672395016005442e-09j), (0.4388508855190594+3.60928938115041e-09j), (0.42998747459205333-2.773271513963329e-10j), (0.42703427046944853+4.693592186150324e-10j), (0.45233770814957097-1.46411471869477e-09j), (0.42256952344905585+2.7175294931081806e-09j), (0.4297422494564671+1.6297709848468978e-09j), (0.4074492208601441+1.8124848825024744e-10j), (0.40837082905636635-6.910666761789282e-10j), (0.4390740820381325+1.2151871141443338e-10j), (0.4376170872274088-5.357347454240897e-09j), (0.40621410838502925+4.624730738172937e-10j), (0.43187504399975296-1.0722018831532799e-09j), (0.4382828577363398+2.7041754947316963e-10j), (0.41253971672267653-3.502631074085882e-09j)], [(0.5043936653746641-3.5359271067013054e-09j), (0.5165361391409533+3.215575403035337e-09j), (0.45766754337819293+2.1809061223439442e-09j), (0.43391079967841506+1.3478516079250406e-09j), (0.44849461258854717+2.03738048622388e-09j), (0.41564118016685825-2.6041672133227703e-10j), (0.43076561190537177+1.3720878668204155e-09j), (0.4188713108887896-1.8242409789244738e-09j), (0.41603891203703824+5.193762279951404e-10j), (0.43874468650028575+2.0435293458873933e-09j), (0.46420702055911534+2.571744859833212e-09j), (0.4483548019488808+9.195663811940705e-10j), (0.45856239715067204+1.1524292601495641e-10j), (0.4374612769315718-1.752460371895521e-09j), (0.4135210563108558+1.5126975224633125e-09j), (0.4415514548163628+2.784389496459676e-09j), (0.43243708403315395-4.822866553072278e-10j), (0.431685901407036+1.2394689719148394e-09j), (0.4382028801192064-8.999190329334823e-11j), (0.43391159326711204-1.2016687642144065e-10j)], [(0.5622097491868772+3.762892442093425e-09j), (0.5126303429278778-2.0075585639314524e-09j), (0.48377681577403564-1.6352794675873243e-09j), (0.45654370296688285-9.189132789666836e-10j), (0.46641983807785437+7.806911776501838e-10j), (0.4544486146332929-3.786803475081718e-09j), (0.46045962344214786-4.4806220215859655e-10j), (0.46461127830843907+7.229374400495338e-10j), (0.4583839876577258+4.092370895406329e-11j), (0.43454412945720833+1.4417499494833572e-10j), (0.4607519990677247+7.854441541761616e-10j), (0.44639888349047396+2.0811210534509424e-09j), (0.4411199916576152-1.8574889680654803e-09j), (0.4239607848066953-5.022028342759242e-10j), (0.4299549088755157-3.530293835372769e-10j), (0.4166995335035608-1.2649498881369808e-09j), (0.4347499737632461-2.130334464502082e-09j), (0.43189808596798684+1.5825452615079737e-09j), (0.4496995371300727-1.0897833748663022e-09j), (0.4224613529149792+2.1140422764649374e-10j)], [(0.6172275147546316-4.737611525353389e-09j), (0.5089501140028005-3.8616915777958455e-09j), (0.495679871433822-2.9941800986362895e-09j), (0.4595940739818616-5.470547360998938e-10j), (0.4570629001100315-2.7617579337345677e-09j), (0.4549773530452512-2.8391099474585774e-09j), (0.4680725306097884+2.894615557504379e-09j), (0.4349413125382853+1.5864167174129842e-09j), (0.4297534096331219+3.4773606146539147e-09j), (0.4492203154586605+6.575696095657531e-10j), (0.4657975167647237+2.880607969043446e-10j), (0.4478397926868638-1.043006270531159e-09j), (0.4459203172737034+1.908889860356895e-09j), (0.4475440356109175+1.38495703778679e-09j), (0.428070556728926-2.6479756165116537e-09j), (0.4553958466567565-3.917062675165743e-09j), (0.464034599965089+8.408352902826894e-10j), (0.44950257646269165-5.058743138857125e-10j), (0.44176781980058877-2.0249182212771068e-10j), (0.42863175513775786-1.2739791441834518e-09j)], [(0.6358998304640409+3.7492560167606825e-09j), (0.5571662225120235-4.6128212332768145e-10j), (0.48464073019567877+2.4608945990636903e-09j), (0.480114676123776-1.1058127750882666e-09j), (0.4693898505429388+9.989683865644283e-10j), (0.45422696956666186+3.1104389377405363e-10j), (0.4668198607978411-1.534578019413728e-09j), (0.4444493242481258+1.2091720240776581e-09j), (0.463434462697478-1.816984099553632e-09j), (0.469961455106386+2.4447395149556958e-09j), (0.45244107833423186+1.054244691967065e-09j), (0.4706113449574332+1.5398862252689265e-09j), (0.44441009996808134+2.928963738608043e-09j), (0.46265016122197267-4.4624601239212795e-10j), (0.4479126433725469+1.677722405288345e-09j), (0.4536541429333738-1.5312053844317527e-09j), (0.4600255975383334-4.341189546593812e-09j), (0.44475000933744013-3.5164348102550518e-09j), (0.429288733190333+1.0266112403604458e-09j), (0.4270880066687823+1.735488417769773e-10j)], [(0.6057604900070146-1.6844448059046784e-09j), (0.5643450293973729-5.247728030141524e-09j), (0.49102803949426743-5.911344610386934e-10j), (0.4668007837390178-5.365332179094154e-09j), (0.4822346462897258-1.1190370713278886e-09j), (0.4569797685107915-6.353119549381378e-10j), (0.4571033983156667-2.419656280913788e-09j), (0.453668985159311-2.5462543461228557e-09j), (0.460671896697022-7.151019778112616e-10j), (0.4623882184459944-4.827673262096319e-10j), (0.4515916265445412-2.3733153780455e-09j), (0.4648642545362236-5.249218838617866e-10j), (0.46055440110649215+8.199399603637108e-09j), (0.4632505460467655-2.1127840431286122e-10j), (0.46156317827262683-2.049699965800428e-09j), (0.45133681206789333+2.3721220534199763e-09j), (0.4483453737520904-4.659427850128511e-10j), (0.4549571247844142-4.963172622916873e-09j), (0.4489842483490065+1.7168551023141613e-09j), (0.46145135643018875-4.762410244235282e-09j)], [(0.6635651819924533+7.407374269513205e-09j), (0.5639755891170353+3.839041973250991e-09j), (0.5012753016853821-1.9351691362081883e-09j), (0.46925682733854046+1.4764906241405719e-09j), (0.4622722407548281-1.8340145801109964e-09j), (0.45434208874212345+2.7189812067292558e-09j), (0.45156644671806134-5.494504713258319e-09j), (0.4665069675029372+7.677617666293007e-09j), (0.47437134257052094+5.985543781874133e-09j), (0.4589937343989732+2.1008679448214446e-09j), (0.47250469808932394-3.340892007904944e-10j), (0.44405354031550814+1.48817319083741e-10j), (0.46874595408371533+2.8273530037920477e-09j), (0.44699116945048445-8.774546855844726e-11j), (0.4565151143760886+7.232792138630187e-10j), (0.45753186385627487+1.0935332637063383e-09j), (0.45275432085691136-5.826805703293793e-10j), (0.4545976501394762+2.113499631871684e-11j), (0.44456840639395523+6.440603609239501e-10j), (0.46192072312987875-2.6868932622592114e-10j)], [(0.6929881106134417-1.3607731830234685e-08j), (0.5769850083197525+3.348764820737609e-09j), (0.5154564952845249-2.278003702526568e-09j), (0.4920591226145916+3.3888447521704656e-10j), (0.47222934207457-8.674843510419196e-09j), (0.4713129683477746-1.0528611176002426e-09j), (0.48245091269564+1.5365259974572475e-09j), (0.4564032856033009-7.022160285011503e-10j), (0.4652447252992715-1.9726845348249015e-09j), (0.47736135041486705+4.13477809495766e-09j), (0.4663282448218524+5.087419933697764e-10j), (0.4647809025536844-5.25294467575959e-10j), (0.46239300105298753+1.1499436400445275e-09j), (0.4609509461588459-2.0654207236114896e-09j), (0.45061349908246484+5.292942001163555e-09j), (0.4699732081826369+1.136949422658548e-09j), (0.4608713529960369+1.5699608278204225e-09j), (0.45874239041950204+9.312888236980064e-10j), (0.46954494258534396+5.794669186484698e-10j), (0.4651178763851931-1.6921264399242303e-09j)], [(0.7645633181282392-1.502429996382607e-08j), (0.6385556895584159-3.624533562434096e-09j), (0.5443711416064616+9.12388386845886e-10j), (0.49996834325793316-2.5340751723900037e-09j), (0.4811883581405709-2.404842290049612e-10j), (0.47296372301752854-9.602724483357806e-10j), (0.48574788447876927+7.781082736088488e-10j), (0.4623655543764471-7.189933648651633e-10j), (0.46988132445949304-5.56794346734256e-10j), (0.4745217056952242-2.4770642199000484e-09j), (0.4727052213202114+1.2447802782001753e-09j), (0.4695304553642927+1.0291819029699326e-09j), (0.4658152639094624-1.2083887226986816e-09j), (0.45590815649120486+3.515595589939108e-10j), (0.46632662956108106+3.9368019830138403e-10j), (0.4536143725272268+1.5683681296682482e-09j), (0.4689862510531384-2.4393047585440378e-09j), (0.4600703586147574-1.4013049059001439e-09j), (0.4600628273747134-1.5580869952296494e-10j), (0.4539118320117268+9.135570909432571e-10j)]])
ss = np.array([[(0.9999993741512299+1.3155790345997787e-09j), (0.8734378814697266+2.9909186232183744e-09j), (0.7709031105041504-4.0277188251280904e-09j), (0.6878336668014526-1.3701515720240026e-10j), (0.6205344200134277-1.8732366387219646e-09j), (0.5660113096237183+1.455697881680429e-09j), (0.5218391120433807+4.626597549517442e-09j), (0.48605261743068695-6.96159063728885e-09j), (0.45705994963645935-8.25699501827426e-10j), (0.43357129395008087-1.8014517308428957e-10j), (0.4145417660474777-3.5541909831904306e-09j), (0.3991248160600662-7.27236808421747e-12j), (0.3866347074508667-3.770203694380969e-09j), (0.3765156716108322+1.4708219531200939e-08j), (0.3683176785707474+1.3960695756983638e-09j), (0.3616761267185211+2.249545882904158e-09j), (0.35629527270793915-5.125043729776923e-10j), (0.35193607211112976+9.07041308728651e-09j), (0.34840431809425354+5.97783333944335e-09j), (0.3455430418252945-5.043973873991581e-09j)], [(0.9999994337558746+1.4831007531508902e-08j), (0.8734377920627594-1.9205799173249716e-09j), (0.7709029912948608+4.802293562811144e-10j), (0.687833696603775+1.9320611499562546e-09j), (0.6205343902111053-8.867654610611453e-09j), (0.566011369228363-1.0410847739450446e-09j), (0.5218391716480255-1.6504112659854187e-09j), (0.48605261743068695-7.456404244043924e-10j), (0.4570598751306534+2.7153070902841137e-09j), (0.4335712641477585-1.4562882913460307e-09j), (0.41454170644283295+1.2650222384370802e-10j), (0.3991248607635498+4.43823273076592e-09j), (0.3866347074508667-2.9103830456733704e-10j), (0.376515731215477-6.249240057723537e-11j), (0.36831775307655334+2.5856758401054947e-09j), (0.36167609691619873-1.2475753208285312e-09j), (0.35629524290561676+4.268437450716256e-09j), (0.3519360423088074-7.978058769175789e-10j), (0.34840428829193115-6.657504769690244e-09j), (0.34554311633110046+7.45058059692341e-09j)], [(0.9999993145465851-1.564318267976983e-08j), (0.8734378218650818-4.688865412667699e-10j), (0.7709030210971832-5.587935447692871e-09j), (0.6878336668014526-1.4767040923402419e-08j), (0.6205343306064606-5.2731586076826265e-09j), (0.5660114288330078-6.976647592971119e-10j), (0.5218391418457031+2.995180561904398e-10j), (0.48605260252952576+3.200631620847716e-09j), (0.45705990493297577-1.4347034671397184e-09j), (0.4335712492465973+9.005854603727492e-10j), (0.41454170644283295+4.325906813318348e-09j), (0.39912480115890503+2.2802773003149923e-09j), (0.3866346925497055+3.9065977119889794e-09j), (0.3765157461166382+1.1827800975163875e-09j), (0.36831770837306976+2.6069204572885718e-09j), (0.3616761118173599+1.3333464887743673e-09j), (0.35629531741142273+1.2701474403492563e-09j), (0.351936012506485-7.3278538790343125e-09j), (0.34840433299541473-1.8605167684260238e-09j), (0.3455430418252945-2.502749683285263e-10j)], [(0.9999993741512299+1.4663883440846348e-09j), (0.8734379410743713+1.9249881688665482e-09j), (0.7709029912948608-2.163486723105734e-09j), (0.6878337264060974+1.0476657857747962e-18j), (0.6205344498157501+7.653596756362901e-09j), (0.5660113394260406-5.983583947766213e-09j), (0.5218391120433807+3.3478310124124278e-09j), (0.48605260252952576-2.1443948006183433e-10j), (0.45705990493297577+8.835325110423398e-10j), (0.4335712790489197+1.3505279117254076e-09j), (0.41454172134399414+4.945364118214002e-10j), (0.39912478625774384+2.1432704500942257e-09j), (0.3866346925497055-2.738463899730397e-10j), (0.37651580572128296-5.772547321768631e-11j), (0.36831772327423096-5.957124488142895e-09j), (0.3616761416196823+4.072259966736436e-09j), (0.35629531741142273-2.276965034705633e-10j), (0.35193605720996857+1.4901161193828825e-08j), (0.34840428829193115-4.759115768958926e-09j), (0.3455430567264557-1.3875757997875595e-11j)], [(0.9999994039535522-1.511676506193993e-08j), (0.8734378218650818+2.6627643919675334e-18j), (0.7709030508995056-1.5267414000203066e-08j), (0.6878336668014526+2.8729330026067146e-08j), (0.6205343902111053+7.463009765729112e-09j), (0.5660114586353302+7.085911579718385e-10j), (0.5218391418457031+2.5422192129198606e-10j), (0.48605261743068695+3.0062452527346295e-11j), (0.45705994963645935-8.268642742725874e-09j), (0.4335712492465973-5.902533739554627e-10j), (0.41454170644283295+1.1414794387487603e-09j), (0.3991248309612274-6.706023969164584e-11j), (0.3866346925497055+2.2210334960082179e-10j), (0.3765157014131546-2.614377382151163e-09j), (0.36831772327423096+4.967423361534884e-09j), (0.36167609691619873+5.497003967806802e-10j), (0.35629530251026154-7.334165275096893e-09j), (0.3519360274076462+2.266310972220964e-09j), (0.34840430319309235+3.9613573532548685e-09j), (0.3455430418252945+1.1646715362967353e-10j)], [(0.9999994337558746+6.026263932401334e-10j), (0.8734378218650818-1.5204096509569e-08j), (0.7709031105041504+2.820118027990759e-09j), (0.687833696603775+5.497819974032013e-10j), (0.6205344200134277-1.4338542408953714e-08j), (0.566011369228363-2.748742622404876e-09j), (0.5218390822410583+2.5316105548706114e-11j), (0.48605264723300934-6.95374671039195e-10j), (0.45705991983413696-7.45131172649044e-09j), (0.4335712343454361+8.756060546760702e-11j), (0.41454169154167175-1.9399803985464814e-10j), (0.39912480115890503-5.999456713822206e-19j), (0.3866346925497055-2.1548475226396135e-09j), (0.3765157461166382+3.606305615244665e-09j), (0.36831773817539215-7.568804474410627e-10j), (0.3616761267185211-8.242100918431916e-09j), (0.35629527270793915+1.1088988616236861e-09j), (0.3519359827041626+2.5501445399811473e-10j), (0.34840431809425354+1.5721860140524376e-18j), (0.3455430716276169+7.450580596968696e-09j)], [(0.9999993443489075+1.4650343382882625e-08j), (0.8734378814697266+2.0059820471381176e-09j), (0.7709030508995056+7.831967990812316e-09j), (0.6878337264060974+7.1961391290287224e-09j), (0.6205344200134277+2.5691582195008777e-09j), (0.566011369228363-2.7048430162324166e-09j), (0.5218391418457031-6.401890351170891e-09j), (0.48605264723300934-5.066386520383159e-09j), (0.45705990493297577+4.5894635314347454e-11j), (0.43357130885124207-2.961931366840531e-09j), (0.4145417660474777+5.210944870048806e-10j), (0.3991248607635498+4.666151662106877e-10j), (0.3866346776485443+1.3040345692161281e-08j), (0.3765157610177994+4.656612868277607e-10j), (0.36831772327423096-9.313225746603287e-10j), (0.36167606711387634+2.506428398163507e-09j), (0.35629527270793915+5.36360145186876e-09j), (0.35193607211112976-2.4087008776429997e-09j), (0.34840431809425354-2.328306470829626e-10j), (0.3455430418252945-7.450580597194715e-09j)], [(0.9999993145465851+1.034864323105289e-17j), (0.8734378516674042+9.339704565292095e-10j), (0.7709029912948608+8.181780457492602e-11j), (0.6878337860107422+6.7915928436690365e-09j), (0.6205343902111053-9.201079342879837e-10j), (0.5660114884376526-6.101786263756903e-11j), (0.5218391418457031+2.2154034162724656e-09j), (0.48605260252952576+3.7306026323680186e-11j), (0.45705993473529816-1.2834240337156189e-09j), (0.43357130885124207-7.450580596811898e-09j), (0.41454169154167175-2.0321227686692644e-10j), (0.3991248309612274+1.534949253912225e-11j), (0.3866347074508667+1.3354774673277059e-09j), (0.37651577591896057-4.6695827204956686e-09j), (0.36831769347190857+8.110811577921595e-09j), (0.3616761118173599+8.187833522306917e-09j), (0.35629525780677795+6.371594085674559e-09j), (0.35193605720996857-4.038133091799345e-09j), (0.34840427339076996+1.5364222782920933e-08j), (0.3455430269241333-2.559287329246781e-12j)], [(0.9999992549419403-1.1175870895385742e-08j), (0.8734378516674042-3.0400983952461047e-09j), (0.7709030508995056+4.625462236257262e-09j), (0.6878337562084198-1.902058391583528e-18j), (0.6205343902111053+6.608696272517034e-11j), (0.5660114586353302-6.067696345368745e-11j), (0.5218392312526703+3.6374899620161827e-10j), (0.48605263233184814+1.6168107495190265e-09j), (0.45705990493297577+9.694627045586657e-11j), (0.4335712194442749-7.904367826938596e-09j), (0.41454175114631653-3.5399794207080504e-09j), (0.3991248160600662-6.450796452561747e-09j), (0.3866346925497055+2.824607214790831e-10j), (0.3765156865119934-1.4594169622794695e-09j), (0.36831772327423096-4.381762674920964e-09j), (0.3616761416196823+8.596541260824211e-10j), (0.35629524290561676-1.8482800429042712e-09j), (0.3519359976053238+5.6802131886968255e-09j), (0.34840428829193115-1.862645149230957e-09j), (0.3455430418252945+5.742471177522688e-14j)], [(0.9999992847442627+1.629726076313176e-08j), (0.873437762260437-1.0791160631740127e-08j), (0.7709029912948608-1.5456529389723528e-09j), (0.6878336668014526+4.734224421341915e-09j), (0.6205344498157501-7.42471772952058e-09j), (0.5660113990306854+5.435907626805125e-10j), (0.5218392014503479+3.2306950448202088e-09j), (0.48605261743068695+4.796956831754073e-09j), (0.45705991983413696+7.657193767940385e-10j), (0.4335712641477585-2.0621082619243225e-10j), (0.4145417660474777+1.7893196633271565e-10j), (0.3991248309612274+1.2145258132534305e-09j), (0.3866347223520279+8.74085250230572e-11j), (0.376515731215477-3.512256602103072e-10j), (0.36831776797771454-1.4564606187761342e-09j), (0.3616761565208435-1.33212958546014e-09j), (0.35629530251026154+1.1762882334024597e-10j), (0.3519359976053238-4.704988482151506e-10j), (0.34840431809425354-4.8403474561808935e-09j), (0.3455430865287781+4.382259842496644e-11j)], [(0.9999993145465851-3.887384747436329e-09j), (0.8734377324581146-6.134334928908913e-09j), (0.7709029614925385-4.0435801373917e-09j), (0.6878337562084198-1.715213542350343e-19j), (0.6205344796180725-5.890539667152694e-09j), (0.5660113990306854+3.1680509882114904e-09j), (0.5218391120433807-2.763413109452273e-11j), (0.48605257272720337-3.455503660987347e-10j), (0.45705996453762054+4.2908676611830277e-10j), (0.43357129395008087+9.402658918394174e-10j), (0.41454172134399414+3.779190807517985e-09j), (0.3991248905658722+1.296339311937511e-09j), (0.3866346478462219-4.36479869608819e-10j), (0.3765157163143158+7.511845027952102e-09j), (0.3683176785707474+8.912793170168243e-09j), (0.3616761714220047-1.3197449799662309e-09j), (0.35629530251026154-1.0915558040780482e-10j), (0.35193607211112976-9.312179378301311e-10j), (0.34840430319309235+2.6345169101826826e-09j), (0.3455430567264557+6.636598381959402e-09j)], [(0.9999994337558746-6.769908189596663e-09j), (0.873437762260437+3.577313362201906e-09j), (0.7709031105041504-7.095423719979123e-10j), (0.6878337562084198-5.49173773123357e-09j), (0.620534360408783+4.798612725925855e-09j), (0.5660113990306854-1.6640833155889823e-10j), (0.5218391716480255+8.298138988449555e-09j), (0.48605263233184814+3.0845539455981452e-09j), (0.45705993473529816+1.548417505325972e-10j), (0.4335712790489197-5.465160712869732e-09j), (0.41454175114631653-8.103168136486261e-11j), (0.3991248607635498+1.7071485752806481e-09j), (0.3866347372531891+6.541406860627319e-10j), (0.37651579082012177+1.2413868552599658e-10j), (0.36831773817539215+1.16415321816365e-10j), (0.36167605221271515+2.0559778324269473e-09j), (0.35629528760910034+9.313225733106358e-10j), (0.351936012506485+3.0551190466354683e-09j), (0.34840433299541473-3.045090598077782e-09j), (0.3455430269241333-5.587935447692871e-09j)], [(0.9999991953372955+9.337975670486998e-10j), (0.8734378218650818-1.1744290303747427e-18j), (0.7709030210971832+7.45058059690243e-09j), (0.6878336369991302+1.9372161097486185e-09j), (0.6205344796180725+2.1848577391736512e-09j), (0.566011369228363+1.4548862129911914e-10j), (0.5218391716480255-4.536212960815078e-09j), (0.48605260252952576+4.150887480958154e-10j), (0.45705991983413696-2.328306436994022e-10j), (0.4335712641477585+8.614879087875948e-10j), (0.41454173624515533-4.949081977567715e-10j), (0.39912478625774384-3.8218828102287716e-10j), (0.3866347074508667-1.5052001589577202e-09j), (0.3765157163143158-4.30530661121864e-09j), (0.36831770837306976+3.3081417466496177e-10j), (0.3616761416196823+1.4068523235266639e-08j), (0.35629528760910034+4.65661291277901e-10j), (0.35193605720996857+1.0814962259075855e-09j), (0.34840431809425354-2.3283064221977792e-10j), (0.3455430418252945-5.128735214299395e-10j)], [(0.9999993741512299-6.009572839360544e-10j), (0.8734378218650818-1.0283313760828234e-08j), (0.7709030508995056+1.980301256310213e-09j), (0.6878337562084198-4.461971888840932e-10j), (0.6205344200134277-1.4264561754018246e-11j), (0.566011369228363-3.7262959384776195e-09j), (0.5218391716480255+6.570260092930713e-09j), (0.48605263233184814-7.1382477151438195e-09j), (0.45705994963645935-3.04797270755941e-10j), (0.4335712492465973+2.2143510679569984e-10j), (0.41454169154167175+1.735316335071957e-09j), (0.3991248309612274+2.3934328963193252e-09j), (0.3866347074508667-1.4935947478811329e-09j), (0.3765157014131546+7.868914964070939e-10j), (0.36831775307655334-4.4682941791052144e-09j), (0.3616761416196823+9.313225785004563e-10j), (0.35629524290561676+3.858667607659072e-09j), (0.3519360423088074+3.150453786737728e-09j), (0.34840428829193115-3.2952043871325998e-09j), (0.3455430567264557+8.047951198003262e-10j)], [(0.9999993145465851-2.6111628415037558e-08j), (0.8734377324581146+1.050785678398874e-09j), (0.7709029912948608-8.125779216161533e-10j), (0.6878337860107422-7.450580600376654e-09j), (0.6205344200134277-2.758603234731538e-09j), (0.5660113990306854+3.436359252706467e-10j), (0.5218391418457031+8.504460691227678e-09j), (0.48605260252952576+1.406377542201298e-09j), (0.45705996453762054-1.4881545093436976e-08j), (0.4335712343454361+1.7865309231979154e-10j), (0.41454175114631653+5.386349699598725e-09j), (0.39912478625774384-9.524584121400892e-09j), (0.3866347223520279-7.5373451334515e-09j), (0.376515731215477-3.691506267333722e-10j), (0.36831770837306976+1.446403591983848e-11j), (0.36167609691619873-2.98152702526977e-09j), (0.35629528760910034-7.176562372146478e-09j), (0.3519360423088074+1.436318163364203e-09j), (0.34840431809425354+2.688848033116642e-10j), (0.3455430865287781-9.436397080397896e-10j)], [(0.9999994039535522+2.910382792038205e-11j), (0.8734378218650818+4.9046583600054205e-09j), (0.7709029912948608+1.2159330209371433e-09j), (0.6878336668014526+1.5062434355339605e-09j), (0.6205343902111053-5.665811707864479e-10j), (0.5660114288330078+7.886374220308972e-09j), (0.5218391418457031+3.3335894669694888e-09j), (0.48605261743068695-1.023080073783711e-09j), (0.45705990493297577+1.3935069763300081e-10j), (0.4335712492465973-6.966462962217485e-09j), (0.41454175114631653+1.0683350037954398e-11j), (0.3991248607635498+3.0133207040705656e-09j), (0.3866346627473831-2.6020152699146593e-09j), (0.3765157461166382-5.602292518780416e-10j), (0.36831775307655334+5.820766092441566e-11j), (0.3616761118173599-5.296772048746234e-10j), (0.35629527270793915+3.0500074688077916e-10j), (0.351936012506485+3.970415113403643e-10j), (0.34840428829193115+4.6566128730773926e-09j), (0.3455430567264557+1.821736247964445e-09j)], [(0.9999992549419403+2.6522981588783523e-09j), (0.8734377026557922-6.003376907690362e-09j), (0.7709030508995056+2.1759558155309833e-08j), (0.687833696603775+2.0682940915506265e-09j), (0.6205343902111053-6.059561741267316e-09j), (0.5660114586353302+4.322342816998059e-09j), (0.5218391716480255-1.559311083032533e-10j), (0.48605263233184814+1.7329149226696927e-09j), (0.45705996453762054+1.5030638844670818e-08j), (0.4335712194442749+3.5272368341745705e-10j), (0.4145417809486389+5.041681724882174e-10j), (0.3991248160600662-2.234852193388169e-09j), (0.3866346925497055+3.59728912429864e-10j), (0.3765157163143158+3.7252902984605153e-09j), (0.36831770837306976-3.6088749766349792e-09j), (0.36167608201503754+9.313225746154785e-09j), (0.3562953472137451+5.70352681750208e-10j), (0.35193605720996857-3.942526305422689e-09j), (0.34840433299541473+5.364960697979404e-10j), (0.3455430269241333+6.885861603134202e-09j)], [(0.9999992847442627-4.733461733685829e-09j), (0.8734377920627594+1.1816356282334795e-08j), (0.7709030210971832-5.2852944243397815e-11j), (0.687833696603775+9.312657314846089e-10j), (0.6205344200134277-4.2762393001605226e-09j), (0.5660114884376526-1.566462474924396e-09j), (0.5218391120433807+2.0510623754965707e-10j), (0.48605263233184814-3.5087807104261515e-09j), (0.4570598900318146+8.783114346933019e-09j), (0.4335712790489197-4.704360734297808e-09j), (0.41454173624515533+6.631884993425763e-11j), (0.39912480115890503+1.7959028708602887e-10j), (0.3866346627473831-7.40034211688112e-09j), (0.376515731215477+1.2835744690469893e-09j), (0.36831772327423096-3.372655266176139e-09j), (0.36167609691619873+4.2052196169706235e-09j), (0.35629528760910034+7.870601503867647e-09j), (0.3519360274076462-9.725871885635229e-09j), (0.34840427339076996+7.335009120923441e-09j), (0.34554310142993927+1.8626451492309568e-09j)], [(0.9999994933605194-1.3665912068638875e-08j), (0.8734377920627594-1.639505862831481e-09j), (0.7709029316902161+1.2137957750013584e-08j), (0.6878336668014526-7.899349396822686e-09j), (0.6205344200134277+1.5834410727055825e-09j), (0.5660113394260406+2.2569662805337274e-10j), (0.5218391716480255-1.997690031898505e-11j), (0.48605258762836456+7.103224675120146e-10j), (0.45705994963645935+1.3447087584064765e-10j), (0.43357130885124207+4.157540312021979e-09j), (0.41454173624515533+8.247014739593989e-10j), (0.3991248309612274-5.873584951743283e-10j), (0.3866347074508667+4.733818614877094e-10j), (0.3765157014131546+3.988128708367242e-12j), (0.36831772327423096-1.0380289405809151e-09j), (0.36167609691619873-5.917569989566681e-09j), (0.35629531741142273+3.732558179203593e-09j), (0.35193605720996857+1.963977314112242e-09j), (0.34840427339076996+5.329985341973043e-10j), (0.3455430269241333+1.0872331802219692e-09j)], [(0.9999993145465851-5.7230373218253305e-11j), (0.873437911272049+1.9364698932919366e-09j), (0.7709030508995056-3.4428265793806645e-09j), (0.6878337860107422-1.4911398330821233e-08j), (0.6205344200134277-2.0125201505219564e-09j), (0.5660113990306854+7.313565309719449e-09j), (0.5218391120433807-7.287159695112955e-10j), (0.48605260252952576-3.538667969760212e-11j), (0.4570598900318146-7.412362388188907e-10j), (0.4335712641477585-3.643383303386649e-09j), (0.41454169154167175-2.097358869912469e-09j), (0.3991248160600662-3.725290298618076e-09j), (0.3866346776485443+2.5983068475464724e-10j), (0.37651579082012177-6.812761466079564e-11j), (0.36831770837306976+6.4973848522746595e-09j), (0.3616761118173599-5.470073560776001e-10j), (0.35629528760910034+6.419474955490888e-10j), (0.3519360423088074+7.083559780407533e-11j), (0.3484043478965759-4.357935958289091e-10j), (0.3455430865287781-7.645167165985412e-09j)]])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
T, MS = np.meshgrid(np.linspace(0,100,20), np.linspace(0.9, 1, 20))
ax.plot_surface(T, MS, cs, color = 'r')
ax.plot_wireframe(T, MS, ss, color='b')
plt.show()
#graph_3d(0.95,1,4, 100, 2, 5)


