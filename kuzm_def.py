import numpy as np
import cirq


import numpy as np
import sympy
import scipy.stats

import cirq


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

    def __init__(self, dimension=3, num_qubits=1):
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

class QuquartDepolarizingChannel(QuditGate):

    def __init__(self, p_matrix=None):
        super().__init__(dimension=4, num_qubits=1)

        # Calculation of the parameter p based on average experimental error of single qudit gate
        f1 = 0.5
        self.p1 = (1 - f1) / (1 - 1 / self.d ** 2)
        self.p1 = 0.9999
        # Choi matrix initialization
        if p_matrix is None:
            self.p_matrix = self.p1 / (self.d ** 2) * np.ones((self.d, self.d))
        else:
            self.p_matrix = p_matrix
        self.p_matrix[0, 0] += (1 - self.p1)  # identity probability
        print(p_matrix)
        print()
        #print('prob[0,0]', self.p_matrix[0, 0])
        #print('prob_sum', self.p_matrix.sum())

    def _mixture_(self):
        ps = []
        for i in range(self.d):
            for j in range(self.d):
                op = np.kron(generalized_sigma(i, 0, 1, dimension=2), generalized_sigma(j, 0, 1, dimension=2))
                #print(np.trace(op) * self.p_matrix[i][j])
                #print(i, j)
                #print(op)
                ps.append(op)

        print('total_sum', (np.trace(np.array(ps)) * self.p_matrix).sum())
        return tuple(zip(self.p_matrix.flatten(), ps))

    def _circuit_diagram_info_(self, args):
        return f"Φ(p1={self.p1:.3f})"

def printm(m):
    for i in m:
        print(*[round(j,2) for j in i])

class DoubleQuquartDepolarizingChannel(QuditGate):
    def __init__(self, p_matrix=None):
        super().__init__(dimension=4, num_qubits=2)

        # Calculation of the parameter p2 based on average experimental error of two qudit gate
        f2 = 0.96
        self.p2 = (1 - f2) / (1 - 1 / (self.d ** 2) ** 2)
        self.p2 = 0.9999

        # Choi matrix initialization
        self.p_matrix = self.p2 / 256 * np.ones((16, 16)) if p_matrix is None else p_matrix
        self.p_matrix[0, 0] += (1 - self.p2)  # identity probability

    def _mixture_(self):
        ps = []
        for i0 in range(self.d):
            for i1 in range(self.d):
                for i2 in range(self.d):
                    for i3 in range(self.d):
                        op = np.kron(np.kron(generalized_sigma(i0, 0, 1, dimension=2),
                                                         generalized_sigma(i1, 0, 1, dimension=2)),
                                           np.kron(generalized_sigma(i2, 0, 1, dimension=2),
                                                         generalized_sigma(i3, 0, 1, dimension=2)))
                        ps.append(op)
        return tuple(zip(self.p_matrix.flatten(), ps))

    def _circuit_diagram_info_(self, args):
        return f"ΦΦ(p2={self.p2:.3f})", f"ΦΦ(p2={self.p2:.3f})"


if __name__ == '__min__':
    X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    Y = np.array([[0, complex(0, -1), 0], [complex(0, 1), 0, 0], [0, 0, 1]])
    Z = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    id = np.eye(3)
    P = 0.9
    n = 1  # number of qudits
    d = 3  # dimension of qudits

    q0 = cirq.LineQid(0, dimension=d)

    print('Ququart single depolarization channel. f1 = 0.99')
    dpg = QuquartDepolarizingChannel()
    ops = [id * (1 - P), X * P / 3, Y * P / 3, Z * P / 3]
    x_meas = cirq.KrausChannel(ops, dimension = 3)
    #dpg._mixture_()
    circuit = cirq.Circuit(x_meas.on(q0))
    #print(circuit)
    #print()
    #print('final_trace', np.trace(cirq.final_density_matrix(circuit)))
    print(cirq.final_density_matrix(circuit))

    #print('choi')
    f1 = 0.8
    f1 = 0.59
    p0 = (1 - f1) / (1 - 1 / 3 ** 2)
    x = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    y = np.array([[0, complex(0, -1), 0], [complex(0, 1), 0, 0], [0, 0, 1]])
    z = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    id = np.eye(3)
    x = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0,0,0,1]])
    y = np.array([[0, complex(0, -1), 0, 0], [complex(0, 1), 0, 0, 0], [0, 0, 1, 0], [0,0,0,1]])
    z = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0] , [0,0,0,1]])
    id = np.eye(4)
    K0 = (1 - p0) ** 0.5 * id
    #K0 = id
    p1 = 0.5
    Ea1 = 1 / 3**0.5 *np.array([[1, 0, 0], [0, (1 - p1) ** 0.5, 0], [0, 0, (1 - p1) ** 0.5]])
    Ea2 = 1 / 3**0.5 * np.array([[0, p1 ** 0.5, 0], [0, 0, 0], [0, 0, 0]])
    Ea3 = 1 / 3**0.5 * np.array([[0, 0, p1 ** 0.5], [0, 0, 0], [0, 0, 0]])
print('gfsgir')
from scipy import linalg
def R(fi, hi, i=0, j=1):
    N = 3
    if i == j:
        return np.eye(N)
    if i > j:
        i, j = j, i
    x_for_ms = np.zeros((N, N))
    x_for_ms[i][j] = 1
    x_for_ms[j][i] = 1
    y_for_ms = np.zeros((N, N))
    y_for_ms[i][j] = -1
    y_for_ms[j][i] = 1
    y_for_ms = y_for_ms * 1j

    m = np.cos(fi) * x_for_ms + np.sin(fi) * y_for_ms

    return linalg.expm(-1j * m * hi / 2)
p = 0.5
print(cirq.kraus_to_choi([p * np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), (1-p) * np.eye(3)]))
    #print(np.trace(cirq.kraus_to_choi([Ea1, Ea2, Ea3])))

