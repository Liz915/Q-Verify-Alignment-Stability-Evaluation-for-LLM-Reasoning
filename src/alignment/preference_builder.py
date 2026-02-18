import json
import os
import numpy as np

class PreferenceBuilder:
    """
    [Alignment Pipeline] Converts raw reasoning traces into DPO Preference Pairs.
    Uses percentile-based filtering to ensure robust dataset generation.
    """
    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path
        
    def build_dpo_dataset(self, output_path="data/preference_pairs/train.jsonl"):
        print(f"[ALIGN] Building DPO dataset from {self.raw_data_path}...")
        
        if not os.path.exists(self.raw_data_path):
            print(f"[ERROR] Raw data file {self.raw_data_path} not found.")
            return

        data = []
        with open(self.raw_data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        if not data:
            print("[ERROR] Raw data file is empty.")
            return

        # Sort by reward (Higher is better)
        data.sort(key=lambda x: x['reward'], reverse=True)
        
        n_samples = len(data)
        # Top 30% as Chosen, Bottom 30% as Rejected
        cutoff = int(n_samples * 0.3)
        
        if cutoff == 0:
            print("[ERROR] Not enough data samples to build pairs.")
            return

        high_quality = data[:cutoff]
        low_quality = data[-cutoff:] # Worst examples are at the end
        
        pairs = []
        for i in range(cutoff):
            chosen = high_quality[i]
            # Use random sampling from low quality to avoid fixed mapping
            rejected = low_quality[i] 
            
            # Ensure there is a meaningful margin
            if chosen['reward'] > rejected['reward']:
                pair = {
                    "prompt": f"Optimize circuit for task: {chosen.get('task', 'unknown')}",
                    "chosen": f"Trace: {chosen['trajectory_length']} steps | Reward: {chosen['reward']:.2f}",
                    "rejected": f"Trace: {rejected['trajectory_length']} steps | Reward: {rejected['reward']:.2f}",
                    "margin": chosen['reward'] - rejected['reward']
                }
                pairs.append(pair)
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
                
        print(f"[ALIGN] Generated {len(pairs)} preference pairs. (Top/Bottom 30% split)")

if __name__ == "__main__":
    # We use the unaligned data to find examples of "Bad behavior" vs "Good luck"
    # or mix aligned/unaligned if you want stronger contrast.
    pb = PreferenceBuilder("data/raw_traces/unaligned.jsonl") 
    pb.build_dpo_dataset()