"""
Defines the System Prompts and Chain-of-Thought (CoT) templates.
"""

SYSTEM_PROMPT_COT = """
You are an expert Quantum Compiler. Your goal is to map a logical quantum circuit 
onto a physical quantum processor with restricted connectivity.

Hardware Constraints:
- You may only apply 2-qubit gates between physically connected qubits.
- Use SWAP gates to move quantum states to adjacent positions if necessary.

Output Format:
Thinking: [Reasoning about pathfinding and constraints]
Action: [Gate Operation]
"""

FEW_SHOT_EXAMPLES = """
User: Apply CNOT between Q0 and Q2.
Topology: 0 -- 1 -- 2

Assistant:
Thinking: Q0 and Q2 are not directly connected. The shortest path is via Q1. I need to swap Q0 with Q1, bringing the state of Q0 to Q1, which is adjacent to Q2.
Action: SWAP(0, 1)

User: Now Q0 is at physical qubit 1. Apply CNOT between logical Q0 (at 1) and Q2.
Topology: 0 -- 1 -- 2

Assistant:
Thinking: Logical Q0 is now on physical qubit 1. Q2 is on physical qubit 2. Q1 and Q2 are connected. I can apply the gate directly.
Action: CNOT(1, 2)
"""