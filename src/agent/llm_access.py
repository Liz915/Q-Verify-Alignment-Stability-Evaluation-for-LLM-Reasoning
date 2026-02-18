import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class LLMEngine:
    """
    [Research Grade] Wrapper for Real LLM Inference.
    Replaces the 'Mock Agent' with actual transformer computations.
    """
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct", load_in_4bit=True):
        print(f"[LLM] Loading real model: {model_name}...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        ) if load_in_4bit else None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.model.eval()

    def generate_step(self, prompt, temperature=0.7):
        """
        Generates the next reasoning step using the LLM.
        Returns: (text_output, hidden_states)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            # Enable output_hidden_states to calculate FTLE later
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=temperature,
                do_sample=True,
                return_dict_in_generate=True,
                output_hidden_states=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_ids = outputs.sequences[0][len(inputs.input_ids[0]):]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Extract last hidden state of the last token as the "Thought Vector"
        # Tuple of (layer, batch, seq, dim) -> take last layer, last token
        last_hidden_state = outputs.hidden_states[-1][-1][0, -1, :].cpu().numpy()
        
        return response, last_hidden_state