from qiskit.providers.fake_provider import GenericBackendV2
import networkx as nx
import numpy as np

class RealPhysicsEmulator:
    """
    [Research Grade] Simulates Quantum Fidelity.
    Uses a physically motivated error model: Fidelity = (1 - error)^depth.
    """
    def __init__(self, n_qubits=127):
        try:
            self.backend = GenericBackendV2(num_qubits=n_qubits, seed=42)
            self.coupling_map = self.backend.coupling_map
            edges = list(self.coupling_map.get_edges())
            if not edges: raise ValueError("Empty map")
            self.graph = nx.Graph()
            self.graph.add_edges_from(edges)
        except:
            # Fallback to Grid
            grid_w = int(np.sqrt(n_qubits)) + 1
            G = nx.grid_2d_graph(grid_w, grid_w)
            self.graph = nx.convert_node_labels_to_integers(G)
            
        self.distances = dict(nx.all_pairs_shortest_path_length(self.graph))
        
        # Physics Parameters (Typical Superconducting Qubit Errors)
        self.gate_error_1q = 0.0003  # 0.03% error
        self.gate_error_2q = 0.0080  # 0.8% error (CNOT)
        self.decoherence_rate = 0.001 # Idle error

    def get_fidelity_cost(self, q1, q2):
        """
        Calculates fidelity loss for connecting q1-q2.
        Returns: Expected Fidelity (0.0 to 1.0)
        """
        if q1 == q2: return 1.0
        try:
            dist = self.distances[q1][q2]
            if dist == 1:
                return 1.0 - self.gate_error_2q
            else:
                # SWAP chain: Each SWAP is 3 CNOTs
                num_swaps = dist - 1
                gates_involved = 1 + (num_swaps * 3)
                # Fidelity decays exponentially with gate depth
                fidelity = (1 - self.gate_error_2q) ** gates_involved
                return fidelity
        except KeyError:
            return 1e-9 # Disconnected

    def calculate_reward(self, trace_step):
        """
        Reward = Log-Likelihood of Circuit Success.
        """
        op = trace_step['op']
        qubits = trace_step['qubits']
        
        if len(qubits) != 2: return 0.0
            
        q1, q2 = qubits[0], qubits[1]
        fidelity = self.get_fidelity_cost(q1, q2)
        
        # Reward is Log-Fidelity (Standard in RL for probability maximization)
        # Perfect gate = log(1) = 0
        # Bad gate = log(small) = negative
        reward = np.log(fidelity + 1e-9)
        
        # Add Thermal Noise (Langevin Force)
        # Simulates measurement shot noise
        noise = np.random.normal(0, 0.02)
        
        return max(reward + noise, -10.0)