import sys
import os
import numpy as np

sys.path.append(os.getcwd())
from src.environment.simulator import RealPhysicsEmulator
from src.environment.reward_model import ProcessRewardModel
from src.agent.reasoning_loop import TreeSearchAgent, ReasoningLoop

def run_demo():
    print("=== Q-Verify Demo: Single Task Execution ===")
    
    # 1. Setup
    emu = RealPhysicsEmulator(n_qubits=127)
    prm = ProcessRewardModel(emulator=emu)
    agent = TreeSearchAgent(emulator=emu, reward_model=prm, hallucination_rate=0.1) # Aligned
    loop = ReasoningLoop(agent)
    
    # 2. Define Task: Create a Bell Pair between distant qubits (0 and 5)
    # This forces the agent to use SWAPs
    task = [
        {"op": "h", "qubits": [0]},
        {"op": "cx", "qubits": [0, 5]} 
    ]
    
    print(f"Goal: Apply CNOT between Q0 and Q5 (Distance: {emu.distances[0][5]})")
    
    # 3. Execute
    history, _, total_reward = loop.process_task(task)
    
    # 4. Report
    print(f"\nResult: Total Reward = {total_reward:.4f}")
    print("Trace:")
    for i, step in enumerate(history):
        valid_mark = "✅" if step['valid'] else "❌"
        print(f"Step {i+1}: {step['op'].upper()} {step['qubits']} | {valid_mark}")

if __name__ == "__main__":
    run_demo()