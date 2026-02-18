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
from src.environment.simulator import RealPhysicsEmulator


def build_tasks(manifest_path, n_samples, seed, emulator):
    if manifest_path and os.path.exists(manifest_path):
        rows = read_jsonl(manifest_path)
        pool = build_eval_task_pool(rows, max_pairs_per_task=64)
        tasks = sample_eval_tasks(pool, n_tasks=n_samples, seed=seed)
        if tasks:
            return tasks

    rng = np.random.default_rng(seed)
    nodes = list(emulator.graph.nodes())
    return [
        {"op": "cx", "qubits": [int(q1), int(q2)], "task_id": "random", "family": "fallback"}
        for q1, q2 in (rng.choice(nodes, 2, replace=False) for _ in range(n_samples))
    ]


def run_evaluation(model_name, adapter_path, noise_levels, tasks, device="cpu"):
    print(f"\n--- Evaluating {model_name} ---")
    try:
        agent = RealLLMAgent(adapter_path=adapter_path, device=device)
        layer = get_target_layer_robust(agent.model, layer_idx=12)
        noise_ctrl = LayerNoiseController(layer)
        emu = RealPhysicsEmulator(n_qubits=127)
    except Exception as e:
        print(f"[ERR] Setup failed for {model_name}: {e}")
        return []

    results = []
    modes = [("Sys 1 (Greedy)", False)]
    if "Q-Verify" in model_name:
        modes.append(("Sys 2 (Reflexion)", True))

    try:
        for noise in noise_levels:
            for mode_label, use_reflexion in modes:
                success_count = 0
                for task in tqdm(tasks, desc=f"{model_name} | noise={noise} | {mode_label}", leave=False):

                    noise_ctrl.set_noise(noise)
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

                    if final_act["valid"]:
                        success_count += 1

                results.append(
                    {
                        "Model": model_name,
                        "Mode": mode_label,
                        "Noise": noise,
                        "Validity Rate": success_count / len(tasks),
                    }
                )
    finally:
        noise_ctrl.clear()

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate stability radius under hidden-state noise.")
    parser.add_argument("--n_samples", type=int, default=50, help="Samples per noise level per mode.")
    parser.add_argument("--device", type=str, default=os.environ.get("QVERIFY_DEVICE", "cpu"), help="cpu|mps|cuda|auto")
    parser.add_argument("--manifest_path", type=str, default="data/task_manifests/test.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs("results/analysis", exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)

    noise_levels = [0.0, 0.1, 0.2, 0.5]
    n_samples = args.n_samples
    all_data = []
    emu_for_tasks = RealPhysicsEmulator(n_qubits=127)
    tasks = build_tasks(args.manifest_path, n_samples=n_samples, seed=args.seed, emulator=emu_for_tasks)

    all_data.extend(run_evaluation("Baseline", None, noise_levels, tasks=tasks, device=args.device))

    adapter = "./results/dpo_checkpoints/final_adapter"
    if os.path.exists(adapter):
        all_data.extend(run_evaluation("Q-Verify", adapter, noise_levels, tasks=tasks, device=args.device))
    else:
        print(f"[WARN] Adapter missing: {adapter}")

    if not all_data:
        print("[ERR] No stability data collected.")
        return

    df = pd.DataFrame(all_data)
    df.to_csv("results/analysis/stability_data.csv", index=False)

    df["Legend"] = df["Model"] + " " + df["Mode"]

    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    sns.lineplot(
        data=df,
        x="Noise",
        y="Validity Rate",
        hue="Legend",
        style="Legend",
        markers=True,
        linewidth=2.5,
    )
    plt.title("Stability Radius: Impact of Alignment & Reflexion")
    plt.ylabel("Validity Rate")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig("results/analysis/stability_radius.png", dpi=300)
    plt.savefig("results/analysis/stability_radius.pdf")
    print("[DONE] Saved results/analysis/stability_radius.(png|pdf)")


if __name__ == "__main__":
    main()
