import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.primitives import Sampler
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import matplotlib.pyplot as plt

class ComputationalWormhole:
    def __init__(self, n_qubits):
        try:
            print("Initializing ComputationalWormhole")
            self.n_qubits = n_qubits
            self.qubits = QuantumRegister(n_qubits)
            self.cbits = ClassicalRegister(n_qubits)
            self.circuit = QuantumCircuit(self.qubits, self.cbits)
            self.circuit.h(self.qubits)
            self.state_vector = Statevector.from_instruction(self.circuit)
            self.state_vector_list = [self.state_vector]
            self.density_matrix = DensityMatrix.from_instruction(self.circuit)
            self.density_matrix_list = [self.density_matrix]
            print("Initialized ComputationalWormhole")
        except Exception as e:
            print(f"Error in __init__: {e}")

    def create_wormhole(self, qubit1, qubit2):
        try:
            print(f"Creating wormhole between qubits {qubit1} and {qubit2}")
            self.circuit.h(qubit1)
            self.circuit.cx(qubit1, qubit2)
            self.circuit.h(qubit1)
            self.state_vector = Statevector.from_instruction(self.circuit)
            self.state_vector_list.append(self.state_vector)
            self.density_matrix = DensityMatrix.from_instruction(self.circuit)
            self.density_matrix_list.append(self.density_matrix)
            self.circuit.barrier()
            print("Wormhole created")
        except Exception as e:
            print(f"Error in create_wormhole: {e}")

    def apply_6t_cccz(self, control_qubits, target_qubit):
        try:
            print(f"Applying 6T-CCCZ with control qubits {control_qubits} and target qubit {target_qubit}")
            self.create_wormhole(control_qubits[0], control_qubits[1])
            self.circuit.ccx(control_qubits[0], control_qubits[1], control_qubits[2])
            self.circuit.cz(control_qubits[2], target_qubit)
            self.circuit.ccx(control_qubits[0], control_qubits[1], control_qubits[2])
            self.create_wormhole(control_qubits[0], control_qubits[1])
            self.circuit.barrier()
            self.state_vector = Statevector.from_instruction(self.circuit)
            self.state_vector_list.append(self.state_vector)
            self.density_matrix = DensityMatrix.from_instruction(self.circuit)
            self.density_matrix_list.append(self.density_matrix)
            self.circuit.barrier()
            print("6T-CCCZ applied")
        except Exception as e:
            print(f"Error in apply_6t_cccz: {e}")

    def measure_quantum_kolmogorov_complexity(self):
        print("Measuring quantum Kolmogorov complexity")
        try:
            print(f"circuit.depth(): {self.circuit.depth()}")
            print(f"self.n_qubits: {self.n_qubits}")
            complexity = self.circuit.depth() + self.n_qubits
            print(f"Quantum Kolmogorov complexity: {complexity}")
            return complexity
        except Exception as e:
            print(f"Error in measure_quantum_kolmogorov_complexity: {e}")

    def measure_classical_kolmogorov_complexity(self):
        print("Measuring classical Kolmogorov complexity")
        try:
            complexity = self.n_qubits
            print(f"Classical Kolmogorov complexity: {complexity}")
            return complexity
        except Exception as e:
            print(f"Error in measure_classical_kolmogorov_complexity: {e}")

    def measure_quantum_shannon_entropy(self):
        print("Measuring quantum Shannon entropy")
        try:
            state_vector = Statevector.from_instruction(self.circuit)
            entropy = state_vector.entropy()
            print(f"Quantum Shannon entropy: {entropy}")
            return entropy
        except Exception as e:
            print(f"Error in measure_quantum_shannon_entropy: {e}")

    def measure_classical_shannon_entropy(self):
        print("Measuring classical Shannon entropy")
        try:
            entropy = self.n_qubits
            print(f"Classical Shannon entropy: {entropy}")
            return entropy
        except Exception as e:
            print(f"Error in measure_classical_shannon_entropy: {e}")

    def measure_quantum_linear_entropy(self):
        print("Measuring quantum linear entropy")
        try:
            state_vector = Statevector.from_instruction(self.circuit)
            reduced_density_matrix = state_vector.reduce([0])
            purity = reduced_density_matrix.purity()
            entropy = 2 * (1 - purity)
            print(f"Quantum linear entropy: {entropy}")
            return entropy
        except Exception as e:
            print(f"Error in measure_quantum_linear_entropy: {e}")

    def measure_classical_linear_entropy(self):
        print("Measuring classical linear entropy")
        try:
            entropy = self.n_qubits
            print(f"Classical linear entropy: {entropy}")
            return entropy
        except Exception as e:
            print(f"Error in measure_classical_linear_entropy: {e}")

    def measure_linear_entropy(self):
        print("Measuring linear entropy")
        try:
            state_vector = Statevector.from_instruction(self.circuit)
            reduced_density_matrix = state_vector.reduce([0])
            purity = reduced_density_matrix.purity()
            entropy = 2 * (1 - purity)
            print(f"Linear entropy: {entropy}")
            return entropy
        except Exception as e:
            print(f"Error in measure_linear_entropy: {e}")

class EnhancedNonAbelianQFT(ComputationalWormhole):
    def apply_enqft(self, depth):
        try:
            print(f"Applying Enhanced Non-Abelian QFT with depth {depth}")
            for _ in range(depth):
                for i in range(self.n_qubits):
                    print(f"Applying H gate to qubit {i}")
                    self.circuit.h(i)
                    for j in range(i+1, self.n_qubits):
                        print(f"Applying CP gate between qubits {i} and {j}")
                        self.circuit.cp(np.pi/2**(j-i), i, j)
                for i in range(self.n_qubits - 1):
                    self.create_wormhole(i, i+1)
                    self.circuit.cry(np.pi/4, i, i+1)
                self.state_vector = Statevector.from_instruction(self.circuit)
                self.state_vector_list.append(self.state_vector)
                self.density_matrix = DensityMatrix.from_instruction(self.circuit)
                self.density_matrix_list.append(self.density_matrix)
            print("Enhanced Non-Abelian QFT applied")
        except Exception as e:
            print(f"Error in apply_enqft: {e}")

    def measure_entropy(self):
        print("Measuring entropy")
        try:
            measurement_circuit = QuantumCircuit(self.qubits, self.cbits)
            measurement_circuit.compose(self.circuit, inplace=True)
            measurement_circuit.measure(self.qubits, self.cbits)
            sampler = Sampler()
            job = sampler.run(measurement_circuit, shots=8192)
            result = job.result()
            counts = result.quasi_dists[0]
            probabilities = [p for p in counts.values() if p > 0]
            entropy = -sum(p * np.log2(p) for p in probabilities)
            print(f"Measured entropy: {entropy}")
            return entropy
        except Exception as e:
            print(f"Error in measure_entropy: {e}")

    def measure_wormhole_effect(self):
        print("Measuring wormhole effect")
        try:
            initial_state = Statevector.from_instruction(self.circuit)
            wormhole_circuit = self.circuit.copy()
            wormhole_circuit.measure_all()
            self.create_wormhole(0, self.n_qubits-1)
            final_state = Statevector.from_instruction(wormhole_circuit)
            effect = 1 - initial_state.inner(final_state)
            print(f"Wormhole effect: {effect}")
            return effect
        except Exception as e:
            print(f"Error in measure_wormhole_effect: {e}")

    def measure_entanglement(self):
        print("Measuring entanglement")
        try:
            state_vector = Statevector.from_instruction(self.circuit)
            reduced_density_matrix = state_vector.reduce([0])
            purity = reduced_density_matrix.purity()
            entanglement = 2 * (1 - purity)
            print(f"Entanglement: {entanglement}")
            return entanglement
        except Exception as e:
            print(f"Error in measure_entanglement: {e}")

def visualize_quantum_manifold(circuit):
    print("Visualizing quantum manifold")
    try:
        state_vector = Statevector.from_instruction(circuit)
        plot_bloch_multivector(state_vector)
        plt.show()
        print(state_vector)
        print(state_vector.probabilities_dict())
        plt.figure()
        plt.bar(state_vector.probabilities_dict().keys(), state_vector.probabilities_dict().values())
        plt.show()
        return state_vector
    except Exception as e:
        print(f"Error in visualize_quantum_manifold: {e}")

def run_experiment(n_qubits, max_depth):
    entropies = []
    entanglements = []
    complexities = []
    wormhole_effects = []
    linear_entropies = []
    quantum_shannon_entropies = []
    classical_shannon_entropies = []
    quantum_linear_entropies = []
    classical_linear_entropies = []

    for depth in range(1, max_depth + 1):
        print(f"Depth: {depth}")
        try:
            enqft = EnhancedNonAbelianQFT(n_qubits)
            print("Creating initial wormhole")
            enqft.create_wormhole(0, n_qubits-1)
            print("Applying ENQFT")
            enqft.apply_enqft(depth)

            if depth % 2 == 0 and n_qubits >= 4:
                print("Applying 6T-CCCZ")
                enqft.apply_6t_cccz([0, 1, 2], 3)

            print("Measuring entropy")
            entropy = enqft.measure_entropy()
            print("Measuring entanglement")
            entanglement = enqft.measure_entanglement()
            print("Measuring quantum Kolmogorov complexity")
            complexity = enqft.measure_quantum_kolmogorov_complexity()
            print("Measuring wormhole effect")
            wormhole_effect = enqft.measure_wormhole_effect()
            print("Measuring linear entropy")
            linear_entropy = enqft.measure_linear_entropy()
            print("Measuring quantum Shannon entropy")
            quantum_shannon_entropy = enqft.measure_quantum_shannon_entropy()
            print("Measuring classical Shannon entropy")
            classical_shannon_entropy = enqft.measure_classical_shannon_entropy()
            print("Measuring quantum linear entropy")
            quantum_linear_entropy = enqft.measure_quantum_linear_entropy()
            print("Measuring classical linear entropy")
            classical_linear_entropy = enqft.measure_classical_linear_entropy()

            entropies.append(entropy)
            entanglements.append(entanglement)
            complexities.append(complexity)
            wormhole_effects.append(wormhole_effect)
            linear_entropies.append(linear_entropy)
            quantum_shannon_entropies.append(quantum_shannon_entropy)
            classical_shannon_entropies.append(classical_shannon_entropy)
            quantum_linear_entropies.append(quantum_linear_entropy)
            classical_linear_entropies.append(classical_linear_entropy)

            print(f"Qubits: {n_qubits}")
            print(f"Entropy: {entropy}")
            print(f"Entanglement: {entanglement}")
            print(f"Complexity: {complexity}")
            print(f"Wormhole Effect: {wormhole_effect}")
            print(f"Linear Entropy: {linear_entropy}")
            print(f"Quantum Shannon Entropy: {quantum_shannon_entropy}")
            print(f"Classical Shannon Entropy: {classical_shannon_entropy}")
            print(f"Quantum Linear Entropy: {quantum_linear_entropy}")
            print(f"Classical Linear Entropy: {classical_linear_entropy}")

            if depth == max_depth:
                visualize_quantum_manifold(enqft.circuit)
        except Exception as e:
            print(f"Error occurred at depth {depth}: {str(e)}")

    return (entropies, entanglements, complexities, wormhole_effects, linear_entropies,
            quantum_shannon_entropies, classical_shannon_entropies,
            quantum_linear_entropies, classical_linear_entropies)

# 実験の実行と結果のプロット
n_qubits = 4
max_depth = 5
results = run_experiment(n_qubits, max_depth)

plt.figure(figsize=(12, 8))
labels = ['Entropy', 'Entanglement', 'Complexity', 'Wormhole Effect', 'Linear Entropy', 'Quantum Shannon Entropy', 'Classical Shannon Entropy', 'Quantum Linear Entropy', 'Classical Linear Entropy']
for i, data in enumerate(results):
    if data:  # データが空でない場合のみプロット
        plt.plot(range(1, len(data) + 1), data, label=labels[i])

plt.xlabel("Depth")
plt.ylabel("Value")
plt.title("Quantum Circuit Metrics vs Depth")
plt.legend()
plt.grid(True)
plt.show()
