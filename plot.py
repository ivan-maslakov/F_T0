import cirq

# Определение подцепи с анциллой
def my_subcircuit(q0: cirq.Qid, ancilla: cirq.Qid) -> cirq.Circuit:
    return cirq.Circuit(cirq.X(ancilla), cirq.CNOT(ancilla, q0)).freeze()

# Создание исходной цепи
q0 = cirq.LineQubit(0)
q1 = cirq.LineQubit(1)
circuit = cirq.Circuit(cirq.X(q0), cirq.X(q1))

# Вставка подцепи в качестве гейта
ancilla = cirq.LineQubit(2)
circuit.append(cirq.CircuitOperation(my_subcircuit(q0, ancilla)))

sim = cirq.Simulator()
res1 = sim.simulate(circuit)
print(res1)
print(circuit)