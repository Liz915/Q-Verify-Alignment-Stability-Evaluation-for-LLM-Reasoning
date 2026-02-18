import sys
import os
import json

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_qverify")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --- PATH FIX START ---
# 1. Add Project Root to Path (for src.* imports)
sys.path.append(os.getcwd())
# 2. Add experiments folder to Path (to import generate_data.py as a sibling)
sys.path.append(os.path.join(os.getcwd(), "experiments"))

try:
    from generate_data import run_experiment
except ImportError:
    # Fallback: try importing as package if structure differs
    from experiments.generate_data import run_experiment
# --- PATH FIX END ---

def run_ablation_suite():
    print("[ABLATION] Starting comprehensive component analysis...")
    np.random.seed(42)
    os.makedirs("data/ablation", exist_ok=True)
    
    # Config 1: Full VPA (Tree Search + Dense Reward)
    print("\n[1/3] Running Full VPA (Tree Search)...")
    run_experiment(n_samples=200, strategy="tree_search", output_file="data/ablation/full_vpa.jsonl", mode="Full VPA")
    
    # Config 2: Baseline (Greedy)
    print("\n[2/3] Running Baseline (Greedy)...")
    run_experiment(n_samples=200, strategy="greedy", output_file="data/ablation/no_search.jsonl", mode="w/o Tree Search")
    
    # Config 3: Outcome-only proxy
    print("\n[3/3] Running Outcome-Only Proxy...")
    # This remains a proxy setting built from the same policy family.
    # No post-hoc reward shifts are applied.
    run_experiment(n_samples=200, strategy="greedy", output_file="data/ablation/outcome_only.jsonl", mode="Outcome Reward Only")
    
    plot_ablation_results()

def plot_ablation_results():
    print("[ABLATION] Plotting results...")
    data = []
    
    # Load Data
    # 1. Full VPA
    with open("data/ablation/full_vpa.jsonl", 'r') as f:
        for line in f: data.append(json.loads(line))
        
    # 2. No Search
    with open("data/ablation/no_search.jsonl", 'r') as f:
        for line in f: data.append(json.loads(line))
    
    # 3. Outcome-only proxy (no manual perturbation)
    with open("data/ablation/outcome_only.jsonl", 'r') as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Scientific Bar Plot
    ax = sns.barplot(
        data=df, 
        x='mode', 
        y='reward', 
        hue='mode', 
        palette="viridis", 
        errorbar="sd", 
        capsize=.1
    )
    
    plt.title("Ablation Study: Contribution of VPA Components", fontsize=14)
    plt.ylabel("Average Physical Reward (Log Fidelity)", fontsize=12)
    plt.xlabel("Configuration", fontsize=12)
    plt.ylim(bottom=-35, top=5) # Adjust based on data scale
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3)

    os.makedirs("results/preliminary_plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig("results/preliminary_plots/ablation_study.png", dpi=300)
    print("[SUCCESS] Ablation plot saved to results/preliminary_plots/ablation_study.png")

if __name__ == "__main__":
    run_ablation_suite()
