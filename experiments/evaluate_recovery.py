import os
import sys
import argparse

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_qverify")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.agent.llm_agent import RealLLMAgent
from src.agent.noise_utils import LayerNoiseController, get_target_layer_robust
from src.data.task_benchmark import build_eval_task_pool, read_jsonl, sample_eval_tasks
from src.environment.reward_model import ProcessRewardModel
from src.environment.simulator import RealPhysicsEmulator


def run_task(agent, emu, noise_ctrl, noise_std, use_reflexion, task):
    noise_ctrl.set_noise(noise_std)
    prompt = agent.format_prompt(task, emu)
    resp1 = agent._generate(prompt, do_sample=False)
    act1 = agent.parse_action(resp1, emu)
    final_act = act1

    if use_reflexion and not act1["valid"]:
        noise_ctrl.clear()
        err = f"Error: {act1['qubits']} are NOT connected."
        ref_prompt = (
            f"{prompt} {act1['raw']}\n"
            f"User: Invalid! {err} Try again.\nAssistant: Action:"
        )
        resp2 = agent._generate(ref_prompt, do_sample=True, temp=0.3)
        act2 = agent.parse_action(resp2, emu)
        if act2["op"] != "error":
            final_act = act2

    return final_act


def evaluate_condition(model_name, adapter_path, use_reflexion, tasks, noise_std=0.2, device="cpu"):
    label = f"{model_name} + {'Reflexion' if use_reflexion else 'Greedy'}"
    print(f"\n--- {label} ---")

    try:
        agent = RealLLMAgent(adapter_path=adapter_path, device=device)
        layer = get_target_layer_robust(agent.model, layer_idx=12)
        noise_ctrl = LayerNoiseController(layer)
    except Exception as e:
        print(f"[ERR] Setup failed for {label}: {e}")
        return []

    emu = RealPhysicsEmulator(n_qubits=127)
    prm = ProcessRewardModel(emulator=emu)
    rows = []

    try:
        for episode, task in enumerate(tqdm(tasks, desc=label, leave=False)):
            action = run_task(agent, emu, noise_ctrl, noise_std=noise_std, use_reflexion=use_reflexion, task=task)
            reward = prm.evaluate_step(action)
            rows.append(
                {
                    "Episode": episode,
                    "Model": model_name,
                    "Mode": "Reflexion" if use_reflexion else "Greedy",
                    "Condition": label,
                    "Noise": noise_std,
                    "Reward": reward,
                    "Valid": int(action["valid"]),
                }
            )
    finally:
        noise_ctrl.clear()

    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate recovery at critical noise.")
    parser.add_argument("--episodes", type=int, default=30, help="Episodes per condition.")
    parser.add_argument("--noise", type=float, default=0.2, help="Noise std for intervention.")
    parser.add_argument("--device", type=str, default=os.environ.get("QVERIFY_DEVICE", "cpu"), help="cpu|mps|cuda|auto")
    parser.add_argument("--manifest_path", type=str, default="data/task_manifests/test.jsonl")
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    os.makedirs("results/analysis", exist_ok=True)
    np.random.seed(100)
    torch.manual_seed(100)

    adapter = "./results/dpo_checkpoints/final_adapter"
    if not os.path.exists(adapter):
        print(f"[ERR] Adapter missing: {adapter}")
        return

    emu_for_tasks = RealPhysicsEmulator(n_qubits=127)
    if args.manifest_path and os.path.exists(args.manifest_path):
        rows = read_jsonl(args.manifest_path)
        pool = build_eval_task_pool(rows, max_pairs_per_task=64)
        tasks = sample_eval_tasks(pool, n_tasks=args.episodes, seed=args.seed)
    else:
        tasks = []

    if not tasks:
        rng = np.random.default_rng(args.seed)
        nodes = list(emu_for_tasks.graph.nodes())
        tasks = [
            {"op": "cx", "qubits": [int(q1), int(q2)], "task_id": "random", "family": "fallback"}
            for q1, q2 in (rng.choice(nodes, 2, replace=False) for _ in range(args.episodes))
        ]

    all_rows = []
    all_rows.extend(evaluate_condition("Baseline", None, use_reflexion=False, tasks=tasks, noise_std=args.noise, device=args.device))
    all_rows.extend(evaluate_condition("Baseline", None, use_reflexion=True, tasks=tasks, noise_std=args.noise, device=args.device))
    all_rows.extend(evaluate_condition("Q-Verify", adapter, use_reflexion=False, tasks=tasks, noise_std=args.noise, device=args.device))
    all_rows.extend(evaluate_condition("Q-Verify", adapter, use_reflexion=True, tasks=tasks, noise_std=args.noise, device=args.device))

    if not all_rows:
        print("[ERR] No recovery data collected.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv("results/analysis/recovery_data.csv", index=False)

    summary = df.groupby("Condition", as_index=False).agg({"Reward": "mean", "Valid": "mean"})
    print(summary)

    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=summary, x="Condition", y="Reward", hue="Condition", palette="Set2", legend=False)
    plt.title(f"Inference-Time Recovery at Critical Noise ($\\sigma={args.noise}$)")
    plt.ylabel("Average Physical Reward")
    plt.xlabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    plt.savefig("results/analysis/recovery_comparison.png", dpi=300)
    plt.savefig("results/analysis/recovery_comparison.pdf")
    print("[DONE] Saved results/analysis/recovery_comparison.(png|pdf)")


if __name__ == "__main__":
    main()
