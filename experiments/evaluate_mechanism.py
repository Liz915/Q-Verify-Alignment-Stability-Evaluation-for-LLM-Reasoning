import argparse
import os
import random
import time
from typing import List

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


def parse_str_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def log(msg: str, log_file: str | None = None) -> None:
    print(msg)
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def config_key(model: str, seed: int, layer_idx: int, noise_type: str, noise: float):
    return (model, int(seed), int(layer_idx), str(noise_type), round(float(noise), 6))


def load_tasks(manifest_path: str, n_tasks: int, seed: int, emulator):
    if manifest_path and os.path.exists(manifest_path):
        rows = read_jsonl(manifest_path)
        pool = build_eval_task_pool(rows, max_pairs_per_task=64)
        tasks = sample_eval_tasks(pool, n_tasks=n_tasks, seed=seed)
        if tasks:
            return tasks

    rng = np.random.default_rng(seed)
    nodes = list(emulator.graph.nodes())
    tasks = []
    for _ in range(n_tasks):
        q1, q2 = rng.choice(nodes, 2, replace=False)
        tasks.append({"op": "cx", "qubits": [int(q1), int(q2)], "task_id": "random", "family": "fallback"})
    return tasks


def evaluate_configuration(agent, layer_idx, tasks, noise, noise_type, seed, emulator):
    try:
        layer = get_target_layer_robust(agent.model, layer_idx=layer_idx)
    except Exception:
        return None

    noise_ctrl = LayerNoiseController(layer)
    valid = 0
    try:
        set_seed(seed)
        for task in tqdm(tasks, desc=f"layer={layer_idx} noise={noise} type={noise_type}", leave=False):
            noise_ctrl.set_noise(noise, noise_type=noise_type)
            action, _ = agent.step(task, emulator, enable_reflexion=False)
            valid += int(action["valid"])
    finally:
        noise_ctrl.clear()
    return valid / max(1, len(tasks))


def main():
    parser = argparse.ArgumentParser(description="Mechanism checks: layer and noise-type sensitivity.")
    parser.add_argument("--manifest_path", type=str, default="data/task_manifests/test.jsonl")
    parser.add_argument("--n_tasks", type=int, default=200)
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--layers", type=str, default="4,8,12,16,20")
    parser.add_argument("--noise_levels", type=str, default="0.1,0.2,0.5")
    parser.add_argument("--noise_types", type=str, default="gaussian,uniform,signflip,dropout")
    parser.add_argument("--device", type=str, default=os.environ.get("QVERIFY_DEVICE", "cpu"))
    parser.add_argument("--output_prefix", type=str, default="results/analysis/mechanism")
    parser.add_argument("--resume", action="store_true", help="Resume from existing raw CSV if present.")
    parser.add_argument("--log_file", type=str, default=None, help="Optional progress log file.")
    args = parser.parse_args()

    os.makedirs("results/analysis", exist_ok=True)
    seeds = parse_int_list(args.seeds)
    layers = parse_int_list(args.layers)
    noise_levels = parse_float_list(args.noise_levels)
    noise_types = parse_str_list(args.noise_types)

    emu_for_sampling = RealPhysicsEmulator(n_qubits=127)
    tasks_by_seed = {s: load_tasks(args.manifest_path, args.n_tasks, s, emu_for_sampling) for s in seeds}

    model_specs = [
        ("Base", None),
        ("Q-Verify", "./results/dpo_checkpoints/final_adapter"),
    ]

    raw_path = f"{args.output_prefix}_raw.csv"
    summary_path = f"{args.output_prefix}_summary.csv"

    raw_rows = []
    completed = set()
    if (not args.resume) and os.path.exists(raw_path):
        os.remove(raw_path)
    if args.resume and os.path.exists(raw_path):
        prev = pd.read_csv(raw_path)
        if not prev.empty:
            raw_rows = prev.to_dict("records")
            for _, r in prev.iterrows():
                completed.add(
                    config_key(
                        r["model"],
                        int(r["seed"]),
                        int(r["layer_idx"]),
                        str(r["noise_type"]),
                        float(r["noise"]),
                    )
                )
            log(f"[RESUME] Loaded {len(prev)} rows from {raw_path}", args.log_file)

    enabled_model_specs = []
    for model_name, adapter_path in model_specs:
        if adapter_path and not os.path.exists(adapter_path):
            log(f"[WARN] Missing adapter for {model_name}: {adapter_path}", args.log_file)
            continue
        enabled_model_specs.append((model_name, adapter_path))

    total_cfg = (
        len(enabled_model_specs)
        * len(seeds)
        * len(layers)
        * len(noise_types)
        * len(noise_levels)
    )
    done_cfg = len(completed)
    start_time = time.time()
    log(f"[INFO] Mechanism configs total={total_cfg}, completed={done_cfg}", args.log_file)

    wrote_header = os.path.exists(raw_path) and os.path.getsize(raw_path) > 0
    for model_name, adapter_path in model_specs:
        if adapter_path and not os.path.exists(adapter_path):
            continue
        log(f"\n[MODEL] {model_name}", args.log_file)
        agent = RealLLMAgent(adapter_path=adapter_path, device=args.device)
        emu = RealPhysicsEmulator(n_qubits=127)
        for seed in seeds:
            tasks = tasks_by_seed[seed]
            for layer_idx in layers:
                for noise_type in noise_types:
                    for noise in noise_levels:
                        key = config_key(model_name, seed, layer_idx, noise_type, noise)
                        if key in completed:
                            continue
                        validity = evaluate_configuration(
                            agent=agent,
                            layer_idx=layer_idx,
                            tasks=tasks,
                            noise=noise,
                            noise_type=noise_type,
                            seed=seed,
                            emulator=emu,
                        )
                        if validity is None:
                            done_cfg += 1
                            continue
                        row = {
                            "model": model_name,
                            "seed": seed,
                            "layer_idx": layer_idx,
                            "noise_type": noise_type,
                            "noise": noise,
                            "validity_rate": validity,
                        }
                        raw_rows.append(row)
                        pd.DataFrame([row]).to_csv(
                            raw_path,
                            mode="a",
                            header=not wrote_header,
                            index=False,
                        )
                        wrote_header = True
                        completed.add(key)
                        done_cfg += 1
                        elapsed = time.time() - start_time
                        speed = done_cfg / max(elapsed, 1e-6)
                        remain = max(0, total_cfg - done_cfg)
                        eta_min = remain / max(speed, 1e-9) / 60.0
                        log(
                            (
                                f"[PROGRESS] cfg {done_cfg}/{total_cfg} "
                                f"| model={model_name} seed={seed} layer={layer_idx} "
                                f"type={noise_type} noise={noise} val={validity:.4f} "
                                f"| ETA~{eta_min:.1f} min"
                            ),
                            args.log_file,
                        )
        del agent

    if not raw_rows:
        log("[ERR] No mechanism data collected.", args.log_file)
        return

    raw_df = pd.DataFrame(raw_rows)
    # De-duplicate in case of partial reruns.
    raw_df = raw_df.drop_duplicates(subset=["model", "seed", "layer_idx", "noise_type", "noise"], keep="last")
    raw_df.to_csv(raw_path, index=False)

    summary = (
        raw_df.groupby(["model", "layer_idx", "noise_type", "noise"], as_index=False)
        .agg(mean_validity=("validity_rate", "mean"), std_validity=("validity_rate", "std"))
    )
    summary.to_csv(summary_path, index=False)

    sns.set_style("whitegrid")
    for model_name in summary["model"].unique():
        model_df = summary[summary["model"] == model_name]
        for noise_type in noise_types:
            sub = model_df[model_df["noise_type"] == noise_type]
            if sub.empty:
                continue
            pivot = sub.pivot(index="noise", columns="layer_idx", values="mean_validity").sort_index()
            plt.figure(figsize=(7, 4))
            sns.heatmap(pivot, annot=True, vmin=0.0, vmax=1.0, cmap="viridis")
            plt.title(f"{model_name} | {noise_type} noise | Layer Sensitivity")
            plt.xlabel("Layer Index")
            plt.ylabel("Noise Std")
            plt.tight_layout()
            out = f"{args.output_prefix}_{model_name.lower().replace(' ', '_')}_{noise_type}"
            plt.savefig(f"{out}.png", dpi=300)
            plt.savefig(f"{out}.pdf")
            plt.close()

    log("[DONE] Wrote:", args.log_file)
    log(f"  - {raw_path}", args.log_file)
    log(f"  - {summary_path}", args.log_file)
    log(f"  - {args.output_prefix}_<model>_<noise_type>.(png|pdf)", args.log_file)


if __name__ == "__main__":
    main()
