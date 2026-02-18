import sys
import os
import torch
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Path setup
sys.path.append(os.getcwd())
from src.alignment.real_dpo import RealDPORunner

def main():
    parser = argparse.ArgumentParser(description="Train Q-Verify DPO adapter.")
    parser.add_argument("--device", type=str, default=os.environ.get("QVERIFY_DEVICE", "auto"), help="auto|cpu|mps|cuda")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    print("[EXP] Initializing Real DPO Training Pipeline...")
    
    # 1. Load Dataset
    data_path = "data/preference_pairs/train.jsonl"
    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset not found at {data_path}")
        return
        
    # Load JSONL into HuggingFace Dataset format
    dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"[INFO] Loaded {len(dataset)} preference pairs.")

    # 2. Load Base Model (Small model for M1/Demo purposes)
    # Using Qwen 0.5B or 1.5B for fast iteration on local machine
    model_name = "Qwen/Qwen2.5-0.5B-Instruct" 
    print(f"[INFO] Loading Model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    req_device = args.device.lower()
    if req_device == "auto":
        if torch.cuda.is_available():
            req_device = "cuda"
        elif torch.backends.mps.is_available():
            req_device = "mps"
        else:
            req_device = "cpu"

    if req_device == "cuda" and torch.cuda.is_available():
        dtype = torch.float16
        device_map = "auto"
    elif req_device == "mps" and torch.backends.mps.is_available():
        dtype = torch.float16
        device_map = {"": "mps"}
    else:
        req_device = "cpu"
        dtype = torch.float32
        device_map = {"": "cpu"}

    print(f"[INFO] Training device: {req_device} | dtype: {dtype}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=dtype,
        )
    except TypeError:
        # Backward compatibility for older transformers versions.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype,
        )

    # 3. Run Training
    runner = RealDPORunner(model, tokenizer)
    runner.run(dataset, epochs=args.epochs)

if __name__ == "__main__":
    main()
