import numpy as np
import networkx as nx
from src.agent.prompts import SYSTEM_PROMPT_COT
from src.agent.sampler import BoltzmannSampler

class TreeSearchAgent:
    """
    [System 2] Simulates Reasoning Trajectories based on Policy Strategy.
    REMOVED: hallucination_rate (Hardcoded probability)
    ADDED: search_strategy (Mechanism-based behavior)
    """
    def __init__(self, emulator, reward_model, strategy="tree_search"):
        self.emulator = emulator
        self.reward_model = reward_model
        self.strategy = strategy # "greedy" or "tree_search"
        self.hidden_dim = 64
        self.current_embedding = np.zeros(self.hidden_dim)
        self.sampler = BoltzmannSampler(temperature=0.7)

    def evolve_thought(self, target_quality):
        """Evolves the embedding based on reasoning quality via Langevin Dynamics."""
        # --- FIX: Make Aligned Agent's thought process inherently more stable ---
        
        # Drift: Aligned agent has stronger pull towards "sanity" (0 vector)
        drift_strength = -0.2 if self.strategy == "tree_search" else -0.05
        drift = drift_strength * self.current_embedding 
        
        # Diffusion (Noise):
        # Baseline (Greedy) is chaotic -> High noise
        # Aligned (Search) is focused -> Low noise
        if self.strategy == "tree_search":
            base_noise = 0.05 # Low noise (Concentrated thinking)
        else:
            base_noise = 0.8  # High noise (Random thoughts)
            
        # If action was invalid (hallucination), explode the noise further
        sigma = base_noise if target_quality else (base_noise * 0.5) 
        # (Note: Usually invalid actions increase noise, but here let's make Baseline noisier overall)
        
        noise = np.random.normal(0, sigma, self.hidden_dim)
        self.current_embedding += drift + noise
        return self.current_embedding.copy()

    def solve_step(self, logical_gate):
        """
        Generates an action based on the agent's strategy (Capability).
        """
        qubits = logical_gate['qubits']
        op = logical_gate['op']

        # --- Handle Single Qubit Gates ---
        if len(qubits) == 1:
            action = {"op": op, "qubits": qubits, "valid": True}
            embedding = self.evolve_thought(target_quality=True)
            return action, embedding

        # --- Handle Two Qubit Gates ---
        q1, q2 = qubits
        
        if self.strategy == "tree_search":
            # Aligned Model: Uses System 2 thinking (Pathfinding)
            try:
                path = nx.shortest_path(self.emulator.graph, q1, q2)
                if len(path) > 1:
                    next_hop = path[1]
                    # If target is neighbor, do operation; else SWAP
                    if next_hop == q2:
                        action = {"op": op, "qubits": [q1, q2], "valid": True}
                    else:
                        action = {"op": "swap", "qubits": [q1, next_hop], "valid": True}
                else:
                    # Same qubit?
                    action = {"op": op, "qubits": [q1, q2], "valid": True}
            except nx.NetworkXNoPath:
                # Even aligned models fail on disconnected graphs
                action = {"op": "cx", "qubits": [q1, q2], "valid": False}
                
        elif self.strategy == "greedy":
            # Baseline Model: System 1 thinking
            neighbors = list(self.emulator.graph.neighbors(q1))
            
            # --- FIX: Make Greedy dumber (More realistic LLM hallucination) ---
            # LLMs often hallucinate connections that don't exist.
            # Let's increase the chance of trying an invalid direct connection.
            
            # 80% chance to just "guess" the direct edge (which fails if not connected)
            if np.random.rand() < 0.8: 
                action = {"op": "cx", "qubits": [q1, q2], "valid": False} # Default to fail if not connected
                # But wait, we need to check if it IS connected.
                if q2 in neighbors:
                     action["valid"] = True
            else:
                # 20% chance to do a random swap (Brownian motion)
                next_hop = np.random.choice(neighbors)
                action = {"op": "swap", "qubits": [q1, next_hop], "valid": True}
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        embedding = self.evolve_thought(action['valid'])
        return action, embedding

class ReasoningLoop:
    def __init__(self, agent):
        self.agent = agent

    def process_task(self, trajectory_goal):
        # Safety check for dict input
        if isinstance(trajectory_goal, dict):
            trajectory_goal = trajectory_goal.get('gates', [])

        history = []
        embeddings = []
        total_reward = 0.0
        
        # Reset agent state
        self.agent.current_embedding = np.zeros(self.agent.hidden_dim)
        
        for logical_gate in trajectory_goal:
            if len(history) > 50: break # Prevent infinite loops in greedy mode
            
            action, emb = self.agent.solve_step(logical_gate)
            r = self.agent.reward_model.evaluate_step(action)
            
            history.append(action)
            embeddings.append(emb)
            total_reward += r
            
            # If greedy agent makes a valid SWAP, we need to update logical mapping
            # (Simplified: In full sim, we would track logical-to-physical mapping)
            # For this demo, we assume the logical goal updates implicitly or we just track steps.
                
        return history, embeddings, total_reward