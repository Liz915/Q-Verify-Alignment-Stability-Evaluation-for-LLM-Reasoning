import torch
import os
from transformers import TrainingArguments, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class RealDPORunner:
    """
    [Research Grade] Encapsulates the Alignment logic.
    Current Stack: DoRA (Structural Adapter) + DPO (Standard Loss)
    """
    def __init__(self, model, tokenizer, output_dir="./results/dpo_checkpoints"):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir

    def run(self, dataset, epochs=3):
        # --- FEATURE: DoRA (Weight-Decomposed Low-Rank Adaptation) ---
        # DoRA is robust and effective for small models. We KEEP this.
        peft_config = LoraConfig(
            r=32, 
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
            use_dora=True  # <--- DoRA 依然开启，这是核心亮点
        )

        # --- CONFIG: Standard DPO ---
        training_args = DPOConfig(
            output_dir=self.output_dir,
            beta=0.1,   # <--- 回退到 0.1 (标准 DPO 参数)
            learning_rate=5e-5,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=epochs,
            logging_steps=10,
            save_steps=50,
            gradient_checkpointing=False,
            fp16=False,
            remove_unused_columns=False,
            max_length=512,
            max_prompt_length=256,
            loss_type="sigmoid", # <--- 回退到标准 DPO，保证能在 Mac 上跑通
        )

        # 3. Initialize Trainer
        trainer = DPOTrainer(
            model=self.model,
            ref_model=None, 
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
            peft_config=peft_config,
        )

        print(f"[ALIGN] Starting DoRA + Standard DPO Training (Epochs={epochs})...")
        trainer.train()
        
        trainer.save_model(os.path.join(self.output_dir, "final_adapter"))
        print(f"[ALIGN] Model saved to {self.output_dir}")