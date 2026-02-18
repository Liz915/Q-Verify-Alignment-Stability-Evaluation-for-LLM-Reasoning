import numpy as np

class ProcessRewardModel:
    """
    [Research Grade] Process Reward Model (PRM) for Quantum Circuits.
    
    Unlike Outcome Reward Models (ORM) that only evaluate the final result,
    PRM evaluates *every step* of the reasoning chain (step-by-step verification).
    
    References:
    - Lightman et al. (2023) "Let's Verify Step by Step"
    - Uesato et al. (2022) "Solving Math Word Problems with Process- and Outcome-Based Feedback"
    """
    def __init__(self, emulator):
        self.emulator = emulator
        # Hyperparameters for reward shaping
        self.fidelity_weight = 1.0
        self.sparsity_penalty = 0.1

    def evaluate_step(self, action):
        """
        Evaluates a single reasoning step.
        Returns a scalar reward.
        """
        if not action.get('valid', False):
            # Penalize hallucinations immediately (Dense Reward Signal)
            return -10.0
        
        op = action['op']
        qubits = action['qubits']
        
        # 1. Physical Fidelity Component
        # Delegate to physics emulator for hardware realism
        if len(qubits) == 2:
            fidelity = self.emulator.get_fidelity_cost(qubits[0], qubits[1])
            r_phys = np.log(fidelity + 1e-9)
        else:
            r_phys = 0.1 # Small bonus for valid single-qubit gates
            
        # 2. Efficiency Component (Penalty for SWAPs)
        r_eff = 0.0
        if op == 'swap':
            r_eff = -0.5 # Soft penalty to encourage shorter paths
            
        # Total Reward
        total_reward = (self.fidelity_weight * r_phys) + r_eff
        
        # Add slight stochasticity for robust training simulation
        return total_reward + np.random.normal(0, 0.01)
