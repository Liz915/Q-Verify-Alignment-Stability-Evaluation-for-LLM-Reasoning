import os
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ACTION_PATTERN = re.compile(r"(SWAP|CX|CNOT)[^0-9]*(\d+)[^0-9]*(\d+)", re.IGNORECASE)


class RealLLMAgent:
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        adapter_path: Optional[str] = None,
        device: str = "auto",
    ) -> None:
        self.device = self._resolve_device(device)
        self.dtype = torch.float16 if self.device != "cpu" else torch.float32

        print(f"[LLM] Loading base model: {model_path} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"dtype": self.dtype}
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs,
            )
        except TypeError:
            # Backward compatibility for older transformers.
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
            )
        self.model.to(self.device)

        if adapter_path and os.path.exists(adapter_path):
            print(f"[LLM] Loading Adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model.to(self.device)

        self._sanitize_generation_config()
        self.model.eval()

    def _sanitize_generation_config(self) -> None:
        """
        Remove sampling-only defaults from model generation config so greedy decoding
        does not emit repeated transformers warnings.
        """
        gen_cfg = getattr(self.model, "generation_config", None)
        if gen_cfg is None:
            return
        for key in ("temperature", "top_p", "top_k", "typical_p"):
            if hasattr(gen_cfg, key):
                setattr(gen_cfg, key, None)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device is None:
            device = "auto"
        device = str(device).strip()
        if device == "":
            device = "auto"

        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return device

    def format_prompt(self, logical_gate: Dict[str, Any], emulator_state: Any) -> str:
        q1, q2 = logical_gate["qubits"]
        # Lobotomy Prompt: return action only, no chain-of-thought.
        prompt = f"""You are a Quantum Compiler.
Hardware: IBM Eagle (Heavy-Hex).
Rule: CNOT(q1, q2) is valid ONLY if q1 and q2 are connected. If not, use SWAP(q1, neighbor).

[Example 1]
Task: CNOT(0, 2)
Action: SWAP 0 1

[Example 2]
Task: CNOT(1, 2)
Action: CX 1 2

[Current Task]
Task: CNOT({q1}, {q2})
Current Topology Edges: {str(list(emulator_state.graph.edges)[:15])}...
Action:"""
        return prompt

    def _generate(
        self,
        prompt: str,
        do_sample: bool = False,
        temp: Optional[float] = None,
        max_new_tokens: int = 16,
        return_embedding: bool = False,
    ) -> str | Tuple[str, Optional[np.ndarray]]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": 1.2,
            "do_sample": do_sample,
            "return_dict_in_generate": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "output_hidden_states": return_embedding,
        }
        if do_sample and temp is not None:
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = 0.9

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = outputs.sequences[0][prompt_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if not return_embedding:
            return response

        embedding = None
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states:
            try:
                embedding = hidden_states[-1][-1][0, -1, :].detach().float().cpu().numpy()
            except (IndexError, AttributeError, TypeError):
                embedding = None
        return response, embedding

    def parse_action(self, response_text: str, emulator: Any) -> Dict[str, Any]:
        if "Action:" in response_text:
            response_text = response_text.split("Action:")[-1].strip()

        # Universal regex for SWAP/CX/CNOT with flexible separators.
        match = ACTION_PATTERN.search(response_text)
        if not match:
            return {"op": "error", "qubits": [], "valid": False, "raw": response_text}

        op = match.group(1).lower()
        q_a, q_b = int(match.group(2)), int(match.group(3))
        if op == "cnot":
            op = "cx"
        is_valid = emulator.graph.has_edge(q_a, q_b)
        return {"op": op, "qubits": [q_a, q_b], "valid": is_valid, "raw": response_text}

    def step(
        self,
        logical_gate: Dict[str, Any],
        emulator: Any,
        return_embedding: bool = False,
        enable_reflexion: bool = False,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
        prompt = self.format_prompt(logical_gate, emulator)
        do_sample = False
        temp = None
        if generation_kwargs:
            do_sample = generation_kwargs.get("do_sample", False)
            temp = generation_kwargs.get("temperature")

        if return_embedding:
            resp1, embedding = self._generate(
                prompt,
                do_sample=do_sample,
                temp=temp,
                return_embedding=True,
            )
        else:
            resp1 = self._generate(prompt, do_sample=do_sample, temp=temp)
            embedding = None
        action = self.parse_action(resp1, emulator)

        if enable_reflexion and not action["valid"] and action["op"] != "error":
            error_msg = f"Error: {action['qubits']} are NOT connected. Use SWAP."
            reflexion_prompt = (
                f"{prompt} {action['raw']}\n"
                f"User: Invalid! {error_msg} Try again.\nAssistant: Action:"
            )

            if return_embedding:
                resp2, embedding_v2 = self._generate(
                    reflexion_prompt,
                    do_sample=True,
                    temp=0.3,
                    return_embedding=True,
                )
            else:
                resp2 = self._generate(reflexion_prompt, do_sample=True, temp=0.3)
                embedding_v2 = None
            action_v2 = self.parse_action(resp2, emulator)
            if action_v2["op"] != "error":
                action = action_v2
                if return_embedding:
                    embedding = embedding_v2

        return action, embedding
