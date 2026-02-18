import os
import sys
import argparse

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_qverify")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.agent.llm_agent import RealLLMAgent
from src.agent.noise_utils import LayerNoiseController, get_target_layer_robust
from src.data.task_benchmark import build_eval_task_pool, read_jsonl, sample_eval_tasks
from src.environment.simulator import RealPhysicsEmulator


def calculate_error_severity(action, emulator):
    """
    0.0: correct.
    dist-1: logical error distance.
    10.0: parse/syntax error.
    """
    if action["op"] == "error":
        return 10.0

    q1, q2 = action["qubits"]
    try:
        if emulator.graph.has_edge(q1, q2):
            return 0.0
        dist = nx.shortest_path_length(emulator.graph, source=q1, target=q2)
        return max(0, dist - 1)
    except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
        return 10.0


def evaluate_model(model_name, adapter_path, noise_levels, tasks, device="cpu"):
    print(f"\n--- Evaluating failure modes: {model_name} ---")
    try:
        agent = RealLLMAgent(adapter_path=adapter_path, device=device)
        layer = get_target_layer_robust(agent.model, layer_idx=12)
        noise_ctrl = LayerNoiseController(layer)
        emu = RealPhysicsEmulator(n_qubits=127)
    except Exception as e:
        print(f"[ERR] Setup failed for {model_name}: {e}")
        return []

    rows = []
    try:
        for noise in noise_levels:
            severities = []
            syntax_errors = 0

            noise_ctrl.set_noise(noise)
            for task in tqdm(tasks, desc=f"{model_name} | noise={noise}", leave=False):
                action, _ = agent.step(task, emu, enable_reflexion=False)
                severity = calculate_error_severity(action, emu)
                severities.append(severity)
                if action["op"] == "error":
                    syntax_errors += 1

            noise_ctrl.clear()
            rows.append(
                {
                    "Model": model_name,
                    "Noise": noise,
                    "Error Severity (Distance)": float(np.mean(severities)),
                    "Syntax Error Rate": syntax_errors / len(tasks),
                }
            )
    finally:
        noise_ctrl.clear()

    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate failure modes under hidden-state noise.")
    parser.add_argument("--n_samples", type=int, default=50, help="Samples per noise level.")
    parser.add_argument("--device", type=str, default=os.environ.get("QVERIFY_DEVICE", "cpu"), help="cpu|mps|cuda|auto")
    parser.add_argument("--manifest_path", type=str, default="data/task_manifests/test.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs("results/analysis", exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)

    noise_levels = [0.0, 0.1, 0.2, 0.5]
    n_samples = args.n_samples
    adapter = "./results/dpo_checkpoints/final_adapter"

    emu_for_tasks = RealPhysicsEmulator(n_qubits=127)
    if args.manifest_path and os.path.exists(args.manifest_path):
        rows = read_jsonl(args.manifest_path)
        pool = build_eval_task_pool(rows, max_pairs_per_task=64)
        tasks = sample_eval_tasks(pool, n_tasks=n_samples, seed=args.seed)
    else:
        tasks = []

    if not tasks:
        rng = np.random.default_rng(args.seed)
        nodes = list(emu_for_tasks.graph.nodes())
        tasks = [
            {"op": "cx", "qubits": [int(q1), int(q2)], "task_id": "random", "family": "fallback"}
            for q1, q2 in (rng.choice(nodes, 2, replace=False) for _ in range(n_samples))
        ]

    all_rows = []
    all_rows.extend(evaluate_model("Baseline", None, noise_levels, tasks=tasks, device=args.device))
    if os.path.exists(adapter):
        all_rows.extend(evaluate_model("Q-Verify", adapter, noise_levels, tasks=tasks, device=args.device))
    else:
        print(f"[WARN] Adapter missing: {adapter}")

    if not all_rows:
        print("[ERR] No failure-mode data collected.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv("results/analysis/failure_modes_data.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.set_style("whitegrid")
    sns.lineplot(
        data=df,
        x="Noise",
        y="Error Severity (Distance)",
        hue="Model",
        marker="o",
        ax=axes[0],
        palette={"Baseline": "#e74c3c", "Q-Verify": "#2ecc71"},
    )
    axes[0].set_title("Logical Degradation")
    axes[0].set_ylabel("Average Error Distance")

    sns.lineplot(
        data=df,
        x="Noise",
        y="Syntax Error Rate",
        hue="Model",
        marker="o",
        ax=axes[1],
        palette={"Baseline": "#e74c3c", "Q-Verify": "#2ecc71"},
    )
    axes[1].set_title("Structural Collapse")
    axes[1].set_ylabel("Syntax Error Rate")
    axes[1].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig("results/analysis/failure_modes.png", dpi=300)
    plt.savefig("results/analysis/failure_modes.pdf")
    print("[DONE] Saved results/analysis/failure_modes.(png|pdf)")


if __name__ == "__main__":
    main()
