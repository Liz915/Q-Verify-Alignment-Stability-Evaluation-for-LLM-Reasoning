import sys
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_qverify")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

try:
    from src.agent.llm_agent import RealLLMAgent
    from src.environment.simulator import RealPhysicsEmulator
    from src.environment.reward_model import ProcessRewardModel
except ImportError:
    print("[ERROR] Modules not found. Make sure you are in the project root.")
    sys.exit(1)

def evaluate_model(model_name, model_path):
    print(f"\n[INFO] Loading {model_name}...")
    # Evaluate Base Model Scaling (No Adapter)
    try:
        # Note: This will download the model if not present.
        # 1.5B fits easily in Mac memory (approx 3-4GB)
        agent = RealLLMAgent(model_path=model_path, adapter_path=None, device="cpu")
    except Exception as e:
        print(f"[ERR] Skipping {model_name}: {e}")
        return -10.0

    emu = RealPhysicsEmulator(n_qubits=127)
    prm = ProcessRewardModel(emulator=emu)
    
    rewards = []
    print(f"[INFO] Evaluating {model_name} (20 samples)...")
    
    # Run 20 samples to get a stable mean
    for _ in tqdm(range(20)): 
        nodes = list(emu.graph.nodes())
        task = {"op": "cx", "qubits": [int(x) for x in np.random.choice(nodes, 2, replace=False)]}
        
        # System 1 only (No Reflexion) to test raw model capacity
        act, _ = agent.step(task, emu, enable_reflexion=False)
        rewards.append(prm.evaluate_step(act))
        
    del agent # Free memory
    return np.mean(rewards)

def main():
    print("[EXP] Experiment C: Scaling Law Analysis...")
    os.makedirs("results/analysis", exist_ok=True)

    # Compare 0.5B and 1.5B
    # Qwen-1.5B-Instruct is small enough for overnight Mac run
    models = {
        "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
        "1.5B": "Qwen/Qwen2.5-1.5B-Instruct" 
    }
    
    results = {}
    for name, path in models.items():
        results[name] = evaluate_model(name, path)
        print(f"    > {name} Score: {results[name]:.4f}")

    # Plotting results
    print("[VIS] Generating Scaling Plot...")
    plt.figure(figsize=(6, 5))
    sns.set_style("whitegrid")
    
    # Bar plot with distinct colors
    ax = sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis")
    
    plt.title("Scaling Law: Zero-Shot Routing Performance")
    plt.ylabel("Average Physical Reward")
    plt.ylim(-11.0, 0.5) # Allow space for negative rewards
    
    # Add labels
    for i in ax.containers:
        ax.bar_label(i, fmt='%.2f', padding=3)

    plt.savefig("results/analysis/experiment_c_scaling.png", dpi=300)
    print("[DONE] Saved results/analysis/experiment_c_scaling.png")

if __name__ == "__main__":
    main()
