import sys
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_qverify")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import torch
import gc

sys.path.append(os.getcwd())
from src.agent.llm_agent import RealLLMAgent
from src.environment.simulator import RealPhysicsEmulator
from src.environment.reward_model import ProcessRewardModel

def switch_topology(emulator, topo_name):
    """Dynamically switches the emulator's backend graph."""
    if topo_name == "Google Sycamore":
        # Create a 53-qubit grid graph approximation
        G = nx.grid_2d_graph(6, 9) 
        nodes_to_remove = [n for n in G.nodes() if n[0] > 5 or n[1] > 8]
        G.remove_nodes_from(nodes_to_remove)
        emulator.graph = nx.convert_node_labels_to_integers(G)
        emulator.distances = dict(nx.all_pairs_shortest_path_length(emulator.graph))
    else:
        # Re-init IBM Eagle
        emulator.__init__(n_qubits=127)

def evaluate_agent_on_topology(agent, topo_name, n_trials=30):
    # Setup Env
    emu = RealPhysicsEmulator(n_qubits=127)
    switch_topology(emu, topo_name)
    
    success_count = 0
    pbar_desc = f"Testing on {topo_name}"
    
    for _ in tqdm(range(n_trials), desc=pbar_desc, leave=False):
        nodes = list(emu.graph.nodes())
        if len(nodes) < 2: continue
        
        q1, q2 = np.random.choice(nodes, 2, replace=False)
        gate = {"op": "cx", "qubits": [int(q1), int(q2)]}
        
        # Run inference
        action, _ = agent.step(gate, emu)
        
        if action['valid']:
            success_count += 1
            
    return success_count / n_trials

def main():
    print("[EVAL] Starting Topological Robustness Test (Sequential Mode)...")
    data = []
    topologies = ["IBM Eagle", "Google Sycamore"]

    # --- PHASE 1: Baseline ---
    print("\n[1/2] Testing Baseline Model...")
    try:
        agent = RealLLMAgent(adapter_path=None)
        for topo in topologies:
            acc = evaluate_agent_on_topology(agent, topo)
            data.append({"Topology": topo, "Model": "Baseline", "Success Rate": acc})
        
        # Cleanup
        del agent
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception as e:
        print(f"[ERROR] Baseline failed: {e}")

    # --- PHASE 2: Aligned ---
    print("\n[2/2] Testing Aligned Model...")
    adapter_path = "./results/dpo_checkpoints/final_adapter"
    if os.path.exists(adapter_path):
        try:
            agent = RealLLMAgent(adapter_path=adapter_path)
            for topo in topologies:
                acc = evaluate_agent_on_topology(agent, topo)
                data.append({"Topology": topo, "Model": "Q-Verify", "Success Rate": acc})
            
            # Cleanup
            del agent
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception as e:
            print(f"[ERROR] Aligned failed: {e}")
    else:
        print("[WARN] No adapter found, skipping Phase 2.")

    # Visualization
    if not data:
        return

    df = pd.DataFrame(data)
    
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    
    ax = sns.barplot(data=df, x="Topology", y="Success Rate", hue="Model", palette={"Baseline": "#e74c3c", "Q-Verify": "#2ecc71"})
    
    plt.title("Zero-Shot Topological Robustness", fontsize=14)
    plt.ylabel("Validity Rate (0.0 - 1.0)", fontsize=12)
    plt.ylim(0, 1.1)
    
    # Add labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)
    
    os.makedirs("results/topology", exist_ok=True)
    save_path = "results/topology/robustness.png"
    plt.savefig(save_path)
    print(f"[SUCCESS] Topology plot saved to {save_path}")

if __name__ == "__main__":
    main()
