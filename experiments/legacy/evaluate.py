import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_qverify")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def load_data(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

print("[INFO] Loading data for visualization...")

try:
    df_un = load_data("data/raw_traces/unaligned.jsonl")
    df_al = load_data("data/raw_traces/aligned.jsonl")
except FileNotFoundError:
    print("[ERROR] Data files not found. Run generate_data.py first.")
    exit(1)

df_un['Status'] = 'Baseline (Unaligned)'
df_al['Status'] = 'Q-Verify (Aligned)'
df = pd.concat([df_un, df_al])

os.makedirs("results/preliminary_plots", exist_ok=True)

# --- Plot 1: Lyapunov Stability vs Reward ---
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Scientific color palette (Blue vs Orange/Red)
palette = {'Baseline (Unaligned)': '#e74c3c', 'Q-Verify (Aligned)': '#2ecc71'}

try:
    sns.scatterplot(
        data=df, 
        x='ftle_stability', 
        y='reward', 
        hue='Status', 
        style='Status', 
        palette=palette,
        alpha=0.7, 
        s=100
    )
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    plt.title("Lyapunov Stability vs. Physical Reward (Correlation Analysis)", fontsize=14)
    plt.xlabel("Finite-Time Lyapunov Exponent (FTLE)\n< 0 implies converging thoughts", fontsize=12)
    plt.ylabel("Physical Reward (Higher is Better)", fontsize=12)
    plt.tight_layout()
    plt.savefig("results/preliminary_plots/stability_correlation.png", dpi=300)
    print("[INFO] Saved: stability_correlation.png")
except Exception as e:
    print(f"[WARN] Plot 1 failed: {e}")

# --- Plot 2: Reward Distribution (KDE) ---
plt.figure(figsize=(10, 6))

try:
    # Check for zero variance (Singular Matrix Error prevention)
    var_un = np.var(df_un['reward'])
    var_al = np.var(df_al['reward'])
    
    # If variance is too low (model is too perfect), add jitter for visualization
    if var_un < 1e-6 or var_al < 1e-6:
        print("[WARN] Low variance detected in data. Adding visualization jitter.")
        df_un['reward'] += np.random.normal(0, 0.01, len(df_un))
        df_al['reward'] += np.random.normal(0, 0.01, len(df_al))

    sns.kdeplot(
        data=df_un, x='reward', fill=True, 
        label='Baseline (High Hallucination)', color='#e74c3c', alpha=0.3, 
        warn_singular=False
    )
    sns.kdeplot(
        data=df_al, x='reward', fill=True, 
        label='Q-Verify (Low Hallucination)', color='#2ecc71', alpha=0.3, 
        warn_singular=False
    )
    
    plt.title("Impact of Verifiable Process Alignment on Hardware Compliance", fontsize=14)
    plt.xlabel("Reward (Higher is Better)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/preliminary_plots/reward_distribution.png", dpi=300)
    print("[INFO] Saved: reward_distribution.png")
except Exception as e:
    print(f"[WARN] Plot 2 failed: {e}")
