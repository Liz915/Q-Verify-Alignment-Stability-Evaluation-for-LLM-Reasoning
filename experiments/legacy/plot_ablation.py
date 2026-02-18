import matplotlib.pyplot as plt
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_qverify")
os.environ.setdefault("MPLBACKEND", "Agg")

import seaborn as sns
import pandas as pd

def main():
    score_path = "results/generalization/ood_scores.csv"
    if not os.path.exists(score_path):
        print("[ERROR] Missing results/generalization/ood_scores.csv. Run evaluate_generalization.py first.")
        return

    raw = pd.read_csv(score_path)
    if raw.empty:
        print("[ERROR] Score file is empty.")
        return

    df = raw.rename(columns={"Model": "Method"})[["Method", "Reward"]]
    
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    
    palette = ["#e74c3c", "#58d68d", "#1d8348"][: len(df)]
    
    ax = sns.barplot(data=df, x="Method", y="Reward", palette=palette)
    
    plt.title("Ablation: Impact of System 2 Inference Strategies", fontsize=14)
    plt.ylabel("Average Physical Reward (Higher is Better)", fontsize=12)
    plt.ylim(bottom=df["Reward"].min() - 0.1, top=max(0.0, df["Reward"].max() + 0.05))
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontweight='bold')

    plt.savefig("results/generalization/final_ablation.png", dpi=300)
    print("Saved final ablation plot.")

if __name__ == "__main__":
    main()
