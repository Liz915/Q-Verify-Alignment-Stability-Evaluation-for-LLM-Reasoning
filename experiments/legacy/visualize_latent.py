import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import pickle # Used for saving/loading vectors

sys.path.append(os.getcwd())
from src.agent.llm_agent import RealLLMAgent 
from src.environment.simulator import RealPhysicsEmulator

def get_vectors(agent, emu, n=50):
    vecs = []
    print(f"Collecting {n} samples...")
    nodes = list(emu.graph.nodes())
    for _ in tqdm(range(n)):
        q1, q2 = np.random.choice(nodes, 2, replace=False)
        gate = {"op": "cx", "qubits": [int(q1), int(q2)]}
        try:
            _, emb = agent.step(gate, emu, return_embedding=True)
            if emb is not None:
                vec = emb.mean(axis=0) if len(emb.shape) > 1 else emb
                vecs.append(vec)
        except: continue
    return np.array(vecs)

def main():
    os.makedirs("results/cache", exist_ok=True)
    emu = RealPhysicsEmulator(n_qubits=127)
    
    # --- MODE SELECTION ---
    # Check if we already computed vectors to avoid re-running expensive inference
    base_file = "results/cache/vecs_base.pkl"
    align_file = "results/cache/vecs_align.pkl"
    
    # 1. Run Baseline (Only if not cached)
    if not os.path.exists(base_file):
        print("\n[1/2] Computing Baseline Vectors...")
        agent = RealLLMAgent(adapter_path=None, device="cpu")
        vecs_base = get_vectors(agent, emu, n=100) # Increased to 100 for better plot
        with open(base_file, 'wb') as f: pickle.dump(vecs_base, f)
        del agent # Free memory
    else:
        print("[INFO] Loading cached Baseline vectors...")
        with open(base_file, 'rb') as f: vecs_base = pickle.load(f)

    # 2. Run Aligned (Only if not cached)
    if not os.path.exists(align_file):
        print("\n[2/2] Computing Aligned Vectors...")
        # Check adapter
        adapter = "./results/dpo_checkpoints/final_adapter"
        if os.path.exists(adapter):
            agent = RealLLMAgent(adapter_path=adapter, device="cpu")
            vecs_align = get_vectors(agent, emu, n=100)
            with open(align_file, 'wb') as f: pickle.dump(vecs_align, f)
            del agent
        else:
            vecs_align = np.array([])
    else:
        print("[INFO] Loading cached Aligned vectors...")
        with open(align_file, 'rb') as f: vecs_align = pickle.load(f)

    # 3. Plotting
    print("Plotting...")
    if len(vecs_align) > 0:
        # Balance dataset size
        min_len = min(len(vecs_base), len(vecs_align))
        X = np.vstack([vecs_base[:min_len], vecs_align[:min_len]])
        y = ["Baseline"] * min_len + ["Q-Verify"] * min_len
        
        score = silhouette_score(X, [0]*min_len + [1]*min_len)
        print(f"Silhouette Score: {score:.4f}")
        
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_emb = tsne.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=X_emb[:,0], y=X_emb[:,1], hue=y, 
                        palette={"Baseline":"#e74c3c", "Q-Verify":"#2ecc71"}, alpha=0.7)
        sns.kdeplot(x=X_emb[:,0], y=X_emb[:,1], hue=y, 
                    palette={"Baseline":"#e74c3c", "Q-Verify":"#2ecc71"}, levels=3, alpha=0.3)
        
        plt.title(f"Latent Space (Silhouette: {score:.2f})")
        plt.savefig("results/preliminary_plots/latent_space_tsne.png", dpi=300)
        print("Done!")

if __name__ == "__main__":
    main()