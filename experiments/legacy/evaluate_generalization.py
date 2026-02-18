import sys
import os
import json

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_qverify")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import torch
import gc

sys.path.append(os.getcwd())
from src.agent.llm_agent import RealLLMAgent 
from src.environment.simulator import RealPhysicsEmulator
from src.environment.reward_model import ProcessRewardModel

def run_evaluation(agent, emulator, reward_model, model_name, n_problems=30, n_samples=1, use_reflexion=False):
    """
    Evaluates agent performance. 
    - Captures interesting Case Studies (Reflexion corrections).
    - If n_samples > 1, runs Best-of-N.
    - If n_samples == 1, runs standard/reflexion mode.
    """
    final_rewards = []
    case_studies = []
    
    pbar_desc = f"Eval {model_name}"
    
    for i in tqdm(range(n_problems), desc=pbar_desc):
        # 1. Random OOD Task
        nodes = list(emulator.graph.nodes())
        q1, q2 = np.random.choice(nodes, 2, replace=False)
        logical_gate = {"op": "cx", "qubits": [int(q1), int(q2)]}
        
        # 2. Inference Strategy
        best_r = -100.0
        best_trace = None
        
        # If running Best-of-N (e.g., for Baseline)
        current_n = n_samples
        
        # IMPORTANT: For Reflexion/System 2, we don't need 16 samples. 
        # The agent fixes itself. 1 sample is enough.
        if "Reflexion" in model_name:
            current_n = 1 
            
        for _ in range(current_n):
            # Capture output
            # Ensure your agent.step returns 'action' dict which contains 'raw' text
            action, _ = agent.step(logical_gate, emulator, enable_reflexion=use_reflexion)
            r = reward_model.evaluate_step(action)
            
            if r > best_r:
                best_r = r
                best_trace = action
        
        final_rewards.append(best_r)
        
        # --- 3. Capture Case Study ---
        # We want cases where Reflexion triggered (System 2).
        # We identify this by checking if the trace contains "Correction needed" or similar logs
        # OR if we simply want to see valid reasoning for the paper.
        if best_trace and best_r > -0.1: # Successful routing
            case_data = {
                "id": i,
                "task": f"CNOT({q1}, {q2})",
                "model": model_name,
                "reward": best_r,
                "trace": best_trace.get("raw", "No raw trace"),
                "valid": best_trace.get("valid", False)
            }
            case_studies.append(case_data)

    return np.mean(final_rewards), case_studies

def main():
    print("[EVAL] Starting Evaluation with Case Study Capture...")
    os.makedirs("results/generalization", exist_ok=True)
    
    emu = RealPhysicsEmulator(n_qubits=127)
    prm = ProcessRewardModel(emulator=emu)
    results = []
    all_cases = []
    
    # --- PHASE 1: Baseline (System 1 / Best-of-5) ---
    print("\n[1/2] Evaluating Baseline...")
    try:
        agent = RealLLMAgent(adapter_path=None)
        # Baseline is weak, give it 5 tries (Best-of-N)
        score, cases = run_evaluation(agent, emu, prm, "Baseline", n_samples=5, use_reflexion=False)
        results.append({"Model": "Baseline", "Task": "Random OOD", "Reward": score})
        # Baseline cases usually aren't interesting unless they fail, skipping save
        del agent
        gc.collect()
    except Exception as e:
        print(f"Baseline error: {e}")

    # --- PHASE 2: Q-Verify + Reflexion (System 2) ---
    print("\n[2/2] Evaluating Q-Verify (Reflexion)...")
    adapter_path = "./results/dpo_checkpoints/final_adapter"
    
    if os.path.exists(adapter_path):
        try:
            agent = RealLLMAgent(adapter_path=adapter_path)
            # Reflexion is smart. Only run ONCE (n_samples=1). 
            # The agent.step() handles the retry internally.
            score, cases = run_evaluation(
                agent,
                emu,
                prm,
                "Q-Verify+Reflexion",
                n_samples=1,
                use_reflexion=True,
            )
            
            results.append({"Model": "Q-Verify+Reflexion", "Task": "Random OOD", "Reward": score})
            all_cases.extend(cases)
            
            del agent
            gc.collect()
        except Exception as e:
            print(f"Aligned error: {e}")

    # --- Save Results ---
    df = pd.DataFrame(results)
    print("\nFinal Results:", df)
    df.to_csv("results/generalization/ood_scores.csv", index=False)
    print("[INFO] Saved aggregate metrics to results/generalization/ood_scores.csv")
    
    # Save Case Studies to JSON for your paper
    with open("results/generalization/case_studies.json", "w") as f:
        json.dump(all_cases, f, indent=2)
    print("[INFO] Saved successful reasoning traces to results/generalization/case_studies.json")

    # Plot
    if not results: return
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df, x="Task", y="Reward", hue="Model", 
                     palette={"Baseline": "#e74c3c", "Q-Verify+Reflexion": "#2ecc71"})
    plt.title("Impact of System 2 Reflexion", fontsize=14)
    plt.ylabel("Physical Reward")
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
    
    plt.savefig("results/generalization/ood_performance.png")

if __name__ == "__main__":
    main()
