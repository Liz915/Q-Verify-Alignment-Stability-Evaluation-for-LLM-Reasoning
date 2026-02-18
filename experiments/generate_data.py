import sys
import os
import json
import numpy as np
import argparse
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.analysis.math_metrics import calculate_ftle
from src.data.task_benchmark import collect_task_entries, parse_qasm_file, parse_yaml_task, read_jsonl


def _entry_to_task(entry):
    source_path = entry["source_path"]
    if source_path.endswith(".qasm"):
        gates = parse_qasm_file(source_path, keep_single_qubit=False)
    else:
        _, gates = parse_yaml_task(source_path, keep_single_qubit=False)
    return {"task_name": entry.get("task_name", os.path.basename(source_path)), "gates": gates}


def load_benchmark_tasks(task_root="benchmarks/tasks", manifest_path=None, max_tasks=None):
    if manifest_path and os.path.exists(manifest_path):
        entries = read_jsonl(manifest_path)
    else:
        entries = collect_task_entries(
            task_root=task_root,
            recursive=True,
            keep_single_qubit=False,
            max_eval_pairs_per_task=64,
            min_two_qubit_gates=1,
        )

    if max_tasks is not None:
        entries = entries[:max_tasks]

    tasks = []
    for entry in entries:
        task = _entry_to_task(entry)
        if task["gates"]:
            tasks.append(task)

    if not tasks:
        tasks = [
            {
                "task_name": "Synthetic_GHZ",
                "gates": [
                    {"op": "cx", "qubits": [0, 1]},
                    {"op": "cx", "qubits": [1, 2]},
                    {"op": "cx", "qubits": [2, 3]},
                ],
            }
        ]
    return tasks


def run_experiment(
    n_samples=50,
    strategy="greedy",
    output_file="data.jsonl",
    mode="baseline",
    task_root="benchmarks/tasks",
    manifest_path=None,
    max_tasks=None,
):
    print(f"[EXP] Running Research Simulation | Strategy={strategy} | Mode={mode}")

    from src.agent.reasoning_loop import ReasoningLoop, TreeSearchAgent
    from src.environment.reward_model import ProcessRewardModel
    from src.environment.simulator import RealPhysicsEmulator

    emu = RealPhysicsEmulator(n_qubits=127) 
    prm = ProcessRewardModel(emulator=emu) 
    
    # REPLACED: hallucination_rate -> strategy
    agent = TreeSearchAgent(emulator=emu, reward_model=prm, strategy=strategy)
    loop = ReasoningLoop(agent)
    
    tasks = load_benchmark_tasks(task_root=task_root, manifest_path=manifest_path, max_tasks=max_tasks)
    print(f"[INFO] Loaded {len(tasks)} benchmark tasks for data generation.")
    results = []
    
    with open(output_file, 'w') as f:
        for i in tqdm(range(n_samples), desc=f"Simulating ({mode})"):
            task_config = tasks[i % len(tasks)]
            circuit_goal = task_config.get('gates', [])
            if not circuit_goal: continue 

            history, embeddings, total_reward = loop.process_task(circuit_goal)
            
            if not history: continue
            
            ftle = calculate_ftle(embeddings)
            
            record = {
                "id": i,
                "task": task_config.get('task_name', 'Unknown'),
                "strategy": strategy,  # Log strategy instead of hallucination_rate
                "reward": total_reward,
                "ftle_stability": ftle,
                "trajectory_length": len(history),
                "mode": mode,
                "embeddings": [e.tolist() for e in embeddings]
            }
            f.write(json.dumps(record) + "\n")
            results.append(record)
            
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--task_root", type=str, default="benchmarks/tasks")
    parser.add_argument("--manifest_path", type=str, default=None, help="Optional train/val/test manifest JSONL path.")
    parser.add_argument("--max_tasks", type=int, default=None, help="Optional cap on number of benchmark tasks.")
    args = parser.parse_args()

    os.makedirs("data/raw_traces", exist_ok=True)
    
    # 1. Baseline = Greedy (Simulates unaligned LLM)
    run_experiment(
        n_samples=args.n_samples,
        strategy="greedy",
        output_file="data/raw_traces/unaligned.jsonl",
        mode="Baseline (Greedy)",
        task_root=args.task_root,
        manifest_path=args.manifest_path,
        max_tasks=args.max_tasks,
    )
    
    # 2. Aligned = Tree Search (Simulates VPA-aligned LLM)
    run_experiment(
        n_samples=args.n_samples,
        strategy="tree_search",
        output_file="data/raw_traces/aligned.jsonl",
        mode="Q-Verify (Aligned)",
        task_root=args.task_root,
        manifest_path=args.manifest_path,
        max_tasks=args.max_tasks,
    )
    
    print(f"[SUCCESS] Research-grade data generation complete ({args.n_samples} samples).")
