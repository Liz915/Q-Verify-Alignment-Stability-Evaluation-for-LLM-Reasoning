import argparse
import os
import random
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_qverify")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

import sys
sys.path.append(os.getcwd())
from src.agent.llm_agent import RealLLMAgent
from src.agent.noise_utils import LayerNoiseController, get_target_layer_robust
from src.data.task_benchmark import build_eval_task_pool, read_jsonl, sample_eval_tasks
from src.environment.simulator import RealPhysicsEmulator


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def bootstrap_ci(values: np.ndarray, alpha: float = 0.05, n_boot: int = 2000, seed: int = 0):
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(float(np.mean(sample)))
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1 - alpha / 2))
    return low, high


def permutation_pvalue(a: np.ndarray, b: np.ndarray, n_perm: int = 5000, seed: int = 0):
    if len(a) == 0 or len(b) == 0:
        return np.nan
    observed = abs(float(np.mean(a) - np.mean(b)))
    combined = np.concatenate([a, b])
    n_a = len(a)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        diff = abs(float(np.mean(combined[:n_a]) - np.mean(combined[n_a:])))
        if diff >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def load_eval_tasks(manifest_path: str, n_tasks: int, seed: int, emulator):
    if manifest_path and os.path.exists(manifest_path):
        rows = read_jsonl(manifest_path)
        pool = build_eval_task_pool(rows, max_pairs_per_task=64)
        tasks = sample_eval_tasks(pool, n_tasks=n_tasks, seed=seed)
        if tasks:
            return tasks

    # Fallback to random qubit pairs if manifest is missing/empty.
    rng = np.random.default_rng(seed)
    nodes = list(emulator.graph.nodes())
    fallback = []
    for _ in range(n_tasks):
        q1, q2 = rng.choice(nodes, 2, replace=False)
        fallback.append({"op": "cx", "qubits": [int(q1), int(q2)], "task_id": "random", "family": "fallback"})
    return fallback


def run_single_strategy(agent, emu, noise_ctrl, task, noise, strategy, best_of_n=2, noise_type="gaussian"):
    prompt = agent.format_prompt(task, emu)

    if strategy == "sys1":
        noise_ctrl.set_noise(noise, noise_type=noise_type)
        resp = agent._generate(prompt, do_sample=False)
        return agent.parse_action(resp, emu), 1

    if strategy == "reflexion":
        noise_ctrl.set_noise(noise, noise_type=noise_type)
        resp1 = agent._generate(prompt, do_sample=False)
        act1 = agent.parse_action(resp1, emu)
        calls = 1
        final = act1
        if not act1["valid"] and act1["op"] != "error":
            # Stabilization assumption: remove perturbation for correction pass.
            noise_ctrl.clear()
            err = f"Error: {act1['qubits']} are NOT connected."
            ref_prompt = f"{prompt} {act1['raw']}\nUser: Invalid! {err} Try again.\nAssistant: Action:"
            resp2 = agent._generate(ref_prompt, do_sample=True, temp=0.3)
            act2 = agent.parse_action(resp2, emu)
            calls += 1
            if act2["op"] != "error":
                final = act2
        return final, calls

    if strategy == "bestofn":
        best = None
        calls = 0
        for i in range(best_of_n):
            noise_ctrl.set_noise(noise, noise_type=noise_type)
            if i == 0:
                resp = agent._generate(prompt, do_sample=False)
            else:
                resp = agent._generate(prompt, do_sample=True, temp=0.7)
            act = agent.parse_action(resp, emu)
            calls += 1
            if best is None:
                best = act
            elif (not best["valid"]) and act["valid"]:
                best = act
        return best, calls

    raise ValueError(f"Unknown strategy: {strategy}")


def evaluate_model(
    model_name: str,
    adapter_path: str | None,
    conditions: List[Dict],
    seeds: List[int],
    noise_levels: List[float],
    tasks_by_seed: Dict[int, List[Dict]],
    layer_idx: int,
    device: str,
    noise_type: str,
):
    rows = []
    print(f"\n[MODEL] {model_name}")
    agent = RealLLMAgent(adapter_path=adapter_path, device=device)
    layer = get_target_layer_robust(agent.model, layer_idx=layer_idx)
    noise_ctrl = LayerNoiseController(layer)
    emu = RealPhysicsEmulator(n_qubits=127)

    try:
        for seed in seeds:
            set_seed(seed)
            tasks = tasks_by_seed[seed]
            for noise in noise_levels:
                for cond in conditions:
                    if cond["model"] != model_name:
                        continue
                    cond_name = cond["name"]
                    strategy = cond["strategy"]
                    best_of_n = cond.get("best_of_n", 2)
                    iterator = tqdm(tasks, desc=f"{cond_name} | seed={seed} | noise={noise}", leave=False)
                    for task in iterator:
                        action, calls = run_single_strategy(
                            agent,
                            emu,
                            noise_ctrl,
                            task,
                            noise,
                            strategy=strategy,
                            best_of_n=best_of_n,
                            noise_type=noise_type,
                        )
                        rows.append(
                            {
                                "seed": seed,
                                "noise": noise,
                                "model": model_name,
                                "condition": cond_name,
                                "strategy": strategy,
                                "task_id": task.get("task_id", "unknown"),
                                "family": task.get("family", "other"),
                                "valid": int(action["valid"]),
                                "calls": calls,
                            }
                        )
    finally:
        noise_ctrl.clear()
        del agent

    return rows


def main():
    parser = argparse.ArgumentParser(description="Stability evaluation with seeds/CI/significance and fair compute controls.")
    parser.add_argument("--manifest_path", type=str, default="data/task_manifests/test.jsonl")
    parser.add_argument("--n_tasks", type=int, default=300)
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--noise_levels", type=str, default="0.0,0.1,0.2,0.5")
    parser.add_argument("--best_of_n", type=int, default=2)
    parser.add_argument("--device", type=str, default=os.environ.get("QVERIFY_DEVICE", "cpu"))
    parser.add_argument("--layer_idx", type=int, default=12)
    parser.add_argument("--noise_type", type=str, default="gaussian")
    parser.add_argument("--output_prefix", type=str, default="results/analysis/stability_stats")
    parser.add_argument("--include_base_reflexion", action="store_true")
    parser.add_argument("--include_bestofn", action="store_true")
    args = parser.parse_args()

    os.makedirs("results/analysis", exist_ok=True)
    seeds = parse_int_list(args.seeds)
    noise_levels = parse_float_list(args.noise_levels)
    if not seeds:
        raise ValueError("At least one seed is required.")

    conditions = [
        {"name": "Base-Sys1", "model": "Base", "strategy": "sys1"},
        {"name": "DPO-Sys1", "model": "Q-Verify", "strategy": "sys1"},
        {"name": "DPO-Reflexion", "model": "Q-Verify", "strategy": "reflexion"},
    ]
    if args.include_base_reflexion:
        conditions.append({"name": "Base-Reflexion", "model": "Base", "strategy": "reflexion"})
    if args.include_bestofn:
        conditions.append(
            {
                "name": f"DPO-BestOf{args.best_of_n}",
                "model": "Q-Verify",
                "strategy": "bestofn",
                "best_of_n": args.best_of_n,
            }
        )

    emulator_for_sampling = RealPhysicsEmulator(n_qubits=127)
    tasks_by_seed = {
        s: load_eval_tasks(
            manifest_path=args.manifest_path,
            n_tasks=args.n_tasks,
            seed=s,
            emulator=emulator_for_sampling,
        )
        for s in seeds
    }

    raw_rows = []
    raw_rows.extend(
        evaluate_model(
            model_name="Base",
            adapter_path=None,
            conditions=conditions,
            seeds=seeds,
            noise_levels=noise_levels,
            tasks_by_seed=tasks_by_seed,
            layer_idx=args.layer_idx,
            device=args.device,
            noise_type=args.noise_type,
        )
    )

    adapter_path = "./results/dpo_checkpoints/final_adapter"
    if os.path.exists(adapter_path):
        raw_rows.extend(
            evaluate_model(
                model_name="Q-Verify",
                adapter_path=adapter_path,
                conditions=conditions,
                seeds=seeds,
                noise_levels=noise_levels,
                tasks_by_seed=tasks_by_seed,
                layer_idx=args.layer_idx,
                device=args.device,
                noise_type=args.noise_type,
            )
        )
    else:
        print(f"[WARN] Adapter missing: {adapter_path}. DPO conditions are skipped.")

    if not raw_rows:
        print("[ERR] No data collected.")
        return

    raw_df = pd.DataFrame(raw_rows)
    raw_path = f"{args.output_prefix}_raw.csv"
    raw_df.to_csv(raw_path, index=False)

    per_seed = (
        raw_df.groupby(["condition", "noise", "seed"], as_index=False)
        .agg(validity_rate=("valid", "mean"), avg_calls=("calls", "mean"))
    )
    per_seed["valid_per_call"] = per_seed["validity_rate"] / per_seed["avg_calls"].clip(lower=1e-9)
    per_seed_path = f"{args.output_prefix}_per_seed.csv"
    per_seed.to_csv(per_seed_path, index=False)

    summary_rows = []
    for (condition, noise), grp in per_seed.groupby(["condition", "noise"]):
        vals = grp["validity_rate"].to_numpy()
        calls = grp["avg_calls"].to_numpy()
        eff = grp["valid_per_call"].to_numpy()
        ci_low, ci_high = bootstrap_ci(vals, seed=123)
        summary_rows.append(
            {
                "condition": condition,
                "noise": noise,
                "mean_validity": float(np.mean(vals)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "mean_calls": float(np.mean(calls)),
                "mean_valid_per_call": float(np.mean(eff)),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["condition", "noise"])
    summary_path = f"{args.output_prefix}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    pair_defs = [
        ("DPO-Sys1", "DPO-Reflexion"),
    ]
    if args.include_bestofn:
        pair_defs.append(("DPO-Reflexion", f"DPO-BestOf{args.best_of_n}"))
    if args.include_base_reflexion:
        pair_defs.append(("Base-Sys1", "Base-Reflexion"))

    sig_rows = []
    for noise in sorted(per_seed["noise"].unique()):
        noise_df = per_seed[per_seed["noise"] == noise]
        for a_name, b_name in pair_defs:
            a = noise_df[noise_df["condition"] == a_name]["validity_rate"].to_numpy()
            b = noise_df[noise_df["condition"] == b_name]["validity_rate"].to_numpy()
            if len(a) == 0 or len(b) == 0:
                continue
            p = permutation_pvalue(a, b, seed=42)
            sig_rows.append(
                {
                    "noise": noise,
                    "condition_a": a_name,
                    "condition_b": b_name,
                    "mean_a": float(np.mean(a)),
                    "mean_b": float(np.mean(b)),
                    "delta": float(np.mean(b) - np.mean(a)),
                    "p_value_perm": p,
                }
            )
    sig_df = pd.DataFrame(sig_rows)
    sig_path = f"{args.output_prefix}_significance.csv"
    sig_df.to_csv(sig_path, index=False)

    sns.set_style("whitegrid")
    plt.figure(figsize=(9, 6))
    for condition, grp in summary_df.groupby("condition"):
        grp = grp.sort_values("noise")
        y = grp["mean_validity"].to_numpy()
        yerr = np.vstack([y - grp["ci_low"].to_numpy(), grp["ci_high"].to_numpy() - y])
        plt.errorbar(grp["noise"], y, yerr=yerr, marker="o", linewidth=2, capsize=3, label=condition)
    plt.title("Stability Radius with Multi-Seed Confidence Intervals")
    plt.xlabel("Noise")
    plt.ylabel("Validity Rate")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}_validity.png", dpi=300)
    plt.savefig(f"{args.output_prefix}_validity.pdf")

    plt.figure(figsize=(9, 6))
    for condition, grp in summary_df.groupby("condition"):
        grp = grp.sort_values("noise")
        plt.plot(grp["noise"], grp["mean_valid_per_call"], marker="o", linewidth=2, label=condition)
    plt.title("Compute-Normalized Validity (Validity per Generation Call)")
    plt.xlabel("Noise")
    plt.ylabel("Validity per Call")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.output_prefix}_efficiency.png", dpi=300)
    plt.savefig(f"{args.output_prefix}_efficiency.pdf")

    print("[DONE] Wrote:")
    print(f"  - {raw_path}")
    print(f"  - {per_seed_path}")
    print(f"  - {summary_path}")
    print(f"  - {sig_path}")
    print(f"  - {args.output_prefix}_validity.(png|pdf)")
    print(f"  - {args.output_prefix}_efficiency.(png|pdf)")


if __name__ == "__main__":
    main()
